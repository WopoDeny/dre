"""Colored ink separation: stamps, signatures and handwriting are not text.

Official documents use colored ink for three different things: printed
letterheads (text), stamps and signatures (pictures), and handwritten inserts
(field values). Black text is recognized without the colored layer, the colored
layer is recognized on its own, and only confident colored words stay text.
The remaining colored ink becomes pictures (large marks) or empty field lines
(small handwriting).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from ..model import Box, Graphic
from .lexicon import TOKEN, Lexicon
from .pixels import channel_max, channel_min, png_bytes, pure_color
from .types import Word

# Chroma (max - min channel) above which a pixel is colored ink, not paper or black toner.
INK_CHROMA = 40
# Pixels darker than this in every channel are black text even inside a stamp.
BLACK_UNDER_COLOR = 80
# If most ink is colored the page has a color cast (photo, sepia): no separation.
MONOCHROME_COLOR_SHARE = 0.6
MIN_COLOR_PIXELS = 400
MAX_MARK_SHARE = 0.3        # of the page area: no stamp or signature is that large
PRINTED_COLOR_CONFIDENCE = 0.80
TOP_ZONE = 0.25            # colored print below this share of the page height is a letterhead no more
# Colored ink this pale on average (darkest channel) is a watermark.
WATERMARK_LEVEL = 165


@dataclass
class InkLayers:
    black: Image.Image
    color: Image.Image | None
    color_mask: np.ndarray | None
    diagnostics: dict[str, object]


def split_ink(image: Image.Image) -> InkLayers:
    rgb = np.asarray(image.convert("RGB"))
    high = channel_max(rgb).astype(np.int16)
    low = channel_min(rgb).astype(np.int16)
    chroma = high - low
    ink = low < 190
    color = pure_color(rgb, (chroma > INK_CHROMA) & (high > BLACK_UNDER_COLOR) & (low < 215))
    ink_pixels = int(ink.sum())
    color_pixels = int((color & ink).sum())
    share = color_pixels / max(1, ink_pixels)
    diagnostics = {"color_ink_pixels": color_pixels, "color_ink_share": round(share, 4)}
    if color_pixels < MIN_COLOR_PIXELS or share >= MONOCHROME_COLOR_SHARE:
        diagnostics["ink_separation"] = False
        return InkLayers(image, None, None, diagnostics)
    # Colored ink is light in at least one channel, black toner is dark in all:
    # the per-pixel brightest channel removes stamps and signatures but keeps
    # black text, including text printed under a stamp.
    # The colored layer takes every ink pixel near a clearly colored one (dark
    # cores and pale edges of strokes), except true black toner.
    near = ndimage.binary_dilation(color, iterations=2)
    true_black = (high < 90) & (chroma < 25)
    colored_ink = near & (low < 225) & ~true_black
    black = np.repeat(high.astype(np.uint8)[:, :, None], 3, axis=2)
    colored = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    colored[colored_ink] = low[colored_ink].astype(np.uint8)
    color = colored_ink
    diagnostics["ink_separation"] = True
    return InkLayers(Image.fromarray(black, mode="RGB"), Image.fromarray(colored, mode="L"), color, diagnostics)


def printed_color_words(words: list[Word], lexicon: Lexicon | None = None, *, page_height: float | None = None,
                        pen: list[Word] | None = None) -> list[Word]:
    """Lines of the colored layer that are printed text (letterheads), not handwriting.

    Tesseract confidence is noisy per word, so the decision is made per line: a
    word supports the line if it is confident or a dictionary word read with
    moderate confidence. Handwriting yields weak non-words.
    """
    lines: dict[tuple[int, ...], list[Word]] = {}
    for word in words:
        if any(character.isalnum() for character in word.text):
            lines.setdefault(word.line_id, []).append(word)
    # Digits in colored ink are handwritten numbers and dates, not print:
    # a printed colored line (a letterhead) has words.
    lines = {key: line for key, line in lines.items() if sum(character.isalpha() for word in line for character in word.text) >= 3}

    def supports(word: Word) -> bool:
        if word.confidence >= PRINTED_COLOR_CONFIDENCE:
            return True
        tokens = TOKEN.findall(word.text)
        return bool(lexicon) and word.confidence >= 0.3 and bool(tokens) and all(len(t) >= 3 and lexicon.known(t) for t in tokens)

    def known(word: Word) -> bool:
        tokens = TOKEN.findall(word.text)
        return bool(tokens) and all(lexicon.known(token) for token in tokens)

    printed: list[Word] = []
    for line in lines.values():
        if sum(supports(word) for word in line) / len(line) < 0.6:
            continue
        # Stamp lettering around a ring reads as short confident junk; a printed
        # line (letterhead) is mostly dictionary words.
        if lexicon and sum(known(word) for word in line) / len(line) < 0.5:
            continue
        if page_height and _pen_line(line, page_height):
            if pen is not None:
                pen.extend(line)  # handwritten text: not a document text, and not a picture either
            continue
        printed.extend(line)
    return printed


def _pen_line(line: list[Word], page_height: float) -> bool:
    """Colored print in documents is a letterhead or a stamp-like mark ("КОПИЯ ВЕРНА"): capitals, or at
    the top of the page. A lowercase colored line in the body is written with a pen, however well it reads."""
    letters = [character for word in line for character in word.text if character.isalpha()]
    if not letters or sum(character.isupper() for character in letters) >= 0.6 * len(letters):
        return False
    return min(word.box.y0 for word in line) > TOP_ZONE * page_height


def resolve_layers(image: Image.Image, color_mask: np.ndarray, black_words: list[Word], color_words: list[Word], *, width: float, height: float) -> list[Word]:
    """Black-layer words that belong to the colored layer are dropped.

    The brightest-channel black layer still shows colored ink faintly: a printed
    colored word is read twice, and colored handwriting yields weak junk there.
    """
    sx, sy = image.width / width, image.height / height
    rgb = np.asarray(image.convert("RGB"))
    low, high = channel_min(rgb), channel_max(rgb)
    chroma = high.astype(np.int16) - low.astype(np.int16)
    page_ink = low < 150
    # Chroma of ordinary print on this page (paper tint and JPEG add a little).
    print_chroma = float(np.median(chroma[page_ink])) if page_ink.any() else 0.0
    kept = []
    for word in black_words:
        if any(color.box.contains_center(word.box) or word.box.contains_center(color.box) for color in color_words):
            continue
        # Black toner stays dark in every channel; a word seen only through colored
        # ink (handwritten numbers, stamp lettering) has almost no dark pixels in
        # its brightest channel, however confidently it was read.
        x0, y0 = max(0, int(word.box.x0 * sx)), max(0, int(word.box.y0 * sy))
        x1, y1 = int(np.ceil(word.box.x1 * sx)), int(np.ceil(word.box.y1 * sy))
        ink_mask = low[y0:y1, x0:x1] < 190
        ink = int(ink_mask.sum())
        if not ink:
            kept.append(word)
            continue
        colored = (color_mask[y0:y1, x0:x1] & ink_mask).sum() / ink
        black = int((high[y0:y1, x0:x1] < 140).sum()) / ink
        # Only colored ink makes a ghost: light gray print is text (a pale scan).
        if colored >= 0.5 and black < 0.3:
            continue
        short = len(word.text) <= 3 or not any(character.isalpha() for character in word.text)
        if short and colored >= 0.6:
            continue  # a handwritten number in colored ink, darkened by shadow or paper tint
        dark = low[y0:y1, x0:x1] < 150
        if short and dark.sum() >= 10 and float(np.median(chroma[y0:y1, x0:x1][dark])) > max(25.0, 3 * print_chroma):
            continue  # its ink is clearly more colored than the print (a photo dims the blue)
        kept.append(word)
    return kept


def _color_asset(image: Image.Image, mask: np.ndarray, left: int, top: int, right: int, bottom: int) -> bytes:
    crop = np.asarray(image.convert("RGB").crop((left, top, right, bottom))).astype(np.int16)
    chroma = channel_max(crop) - channel_min(crop)
    alpha = np.clip((chroma - 20) * 255 / 45, 0, 255)
    alpha[~ndimage.binary_dilation(mask[top:bottom, left:right], iterations=2)] = 0
    rgba = np.dstack([crop.clip(0, 255), alpha]).astype(np.uint8)
    return png_bytes(Image.fromarray(rgba, mode="RGBA"))


def color_marks(
    image: Image.Image, color_mask: np.ndarray, text_words: list[Word], *, index: int, width: float, height: float,
    pen: list[Word] = (),
) -> tuple[list[Graphic], list[Word], dict[str, object]]:
    """Turn colored ink that is not printed text into pictures or empty field lines. Handwritten text lines
    (pen) are skipped."""
    sx, sy = image.width / width, image.height / height
    layer = channel_min(np.asarray(image.convert("RGB")))
    mask = color_mask.copy()
    for word in list(text_words) + list(pen):
        x0, y0 = max(0, int(word.box.x0 * sx) - 2), max(0, int(word.box.y0 * sy) - 2)
        x1, y1 = min(image.width, int(np.ceil(word.box.x1 * sx)) + 2), min(image.height, int(np.ceil(word.box.y1 * sy)) + 2)
        mask[y0:y1, x0:x1] = False
    heights = [word.box.height * sy for word in text_words if word.confidence >= 0.75]
    text_px = float(np.median(heights)) if heights else image.height / 60
    # Strokes of one signature or stamp are joined; separate handwritten inserts stay apart.
    joined = ndimage.binary_dilation(mask, iterations=max(2, int(text_px * 0.35)))
    labels, _ = ndimage.label(joined)
    graphics: list[Graphic] = []
    blanks: list[Word] = []
    roles: dict[str, int] = {}
    ignored = 0
    for label_index, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        ys, xs = slices
        raw = mask[ys, xs] & (labels[ys, xs] == label_index)
        pixels = int(raw.sum())
        if pixels < max(30, text_px * text_px * 0.15):
            continue
        if float(np.percentile(layer[ys, xs][raw], 25)) > WATERMARK_LEVEL:
            # A pale colored mark behind the text is a watermark: background, not content. Judged by its darker
            # strokes: the pale edges of a stamp's thin lines would make a real stamp look like a watermark.
            ignored += 1
            continue
        rows, cols = np.nonzero(raw)
        top, bottom = ys.start + int(rows.min()), ys.start + int(rows.max()) + 1
        left, right = xs.start + int(cols.min()), xs.start + int(cols.max()) + 1
        h, w = bottom - top, right - left
        box = Box(left / sx, top / sy, right / sx, bottom / sy)
        if h * w > MAX_MARK_SHARE * image.width * image.height:
            ignored += 1
            continue  # a color cast over the page (light, a tinted background), not a stamp or a signature
        near_edge = left < image.width * 0.08 or right > image.width * 0.92
        if near_edge and w < h * 0.25 and w < text_px * 3:
            # Small vertical text on the margin (registration notes) is not part of the document.
            ignored += 1
            continue
        if h < text_px * 2.2 and w < text_px * 12:
            # A short handwritten insert (number, date, initials): keep an empty line.
            length = max(3, int(round(w / max(1.0, text_px * 0.55))))
            blanks.append(Word("_" * length, box, 1.0, size=max(6.0, text_px / sy), line_id=(9000 + label_index,),
                               source_page=index, source_region="blank_field"))
            continue
        aspect = w / max(1, h)
        density = pixels / max(1, w * h)
        if 0.7 <= aspect <= 1.45 and h >= text_px * 3.5:
            role = "seal"
        elif density >= 0.25 and aspect < 1.8:
            role = "logo"
        else:
            role = "signature"
        near = next((g for g in graphics if g.role in ("seal", "signature") and role in ("seal", "signature")
                     and g.box.x0 - text_px / sx <= box.x1 and box.x0 <= g.box.x1 + text_px / sx
                     and g.box.y0 - text_px / sy <= box.y1 and box.y0 <= g.box.y1 + text_px / sy), None)
        if near is not None:
            # Pieces of one signature or stamp (a detached stroke) form one picture.
            graphics.remove(near)
            box = Box(min(box.x0, near.box.x0), min(box.y0, near.box.y0), max(box.x1, near.box.x1), max(box.y1, near.box.y1))
            left, top, right, bottom = int(box.x0 * sx), int(box.y0 * sy), int(np.ceil(box.x1 * sx)), int(np.ceil(box.y1 * sy))
            role = "seal" if "seal" in (role, near.role) else "signature"
        roles[role] = roles.get(role, 0) + 1
        graphics.append(Graphic(
            box, index, _color_asset(image, color_mask, left, top, right, bottom),
            description="Colored document mark", role=role, confidence=0.8,
            source_region="raster_graphic:color_ink",
        ))
    return graphics, blanks, {"color_graphics": roles, "color_blank_fields": len(blanks), "color_margin_ignored": ignored}
