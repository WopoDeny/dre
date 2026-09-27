"""Letter size estimate and OCR upscaling for low-resolution scans and photos."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

# Tesseract reads best when a typical letter is about this tall (measured on
# handoff/samples: 2x upscaling of 7-8 px letters gave the best dictionary rate).
TARGET_LETTER_PX = 14.0
MAX_UPSCALE = 3.0


def letter_height(image: Image.Image) -> float:
    """Letter height of the main text, in pixels of `image`.

    Letter-like components are grouped into text rows; each row gives its
    median height and the page value is the median over rows weighted by their
    letter count. Small print (letterhead details) cannot decide for the body.
    """
    gray = np.asarray(image.convert("L"))
    threshold = min(160.0, float(np.median(gray)) - 40.0)
    labels, _ = ndimage.label(gray < threshold)
    letters = []
    for slices in ndimage.find_objects(labels):
        if slices is None:
            continue
        height = slices[0].stop - slices[0].start
        width = slices[1].stop - slices[1].start
        if 3 <= height <= 80 and 2 <= width <= 2.5 * height:
            letters.append(((slices[0].start + slices[0].stop) / 2, height))
    if len(letters) < 20:
        return 0.0
    letters.sort()
    rows: list[list[float]] = []
    last_center = None
    for center, height in letters:
        if last_center is None or center - last_center > max(3.0, 0.6 * height):
            rows.append([])
        rows[-1].append(height)
        last_center = center
    weighted = sorted((float(np.median(row)), len(row)) for row in rows if len(row) >= 8)
    if not weighted:
        return float(np.median([height for _, height in letters]))
    total = sum(count for _, count in weighted)
    running = 0
    for height, count in weighted:
        running += count
        if running >= total / 2:
            return height
    return weighted[-1][0]


# Letters taller than this are shrunk for speed: OCR time grows with pixels,
# accuracy does not (clean 300-dpi scan: same result at 200 and 300 dpi).
MAX_LETTER_PX = 24.0
DOWNSCALE_TARGET_PX = 20.0


def ocr_scale(height_px: float) -> float:
    if height_px <= 0:
        return 1.0
    if height_px < TARGET_LETTER_PX:
        return min(MAX_UPSCALE, TARGET_LETTER_PX / height_px)
    if height_px > MAX_LETTER_PX:
        return DOWNSCALE_TARGET_PX / height_px
    return 1.0


def upscale(image: Image.Image, factor: float) -> Image.Image:
    if abs(factor - 1.0) < 0.02:
        return image
    return image.resize((round(image.width * factor), round(image.height * factor)), Image.Resampling.LANCZOS)


PHOTO_HEADING_WIDTH = 0.6   # on a photo only a line shorter than this share of the text width may be bold


def mark_bold(image: Image.Image, words: list, *, width: float, height: float, photo: bool = False) -> int:
    """Bold OCR lines have thicker strokes than the page's regular text.

    Stroke width of a line is estimated as 2 * ink area / ink edge length inside
    its word boxes; a line clearly thicker than the page median is bold. On a photo the
    strokes thicken wherever the picture is less sharp, so only heading-like short lines
    may be bold there.
    """
    gray = np.asarray(image.convert("L"))
    ink = gray < min(150.0, float(np.quantile(gray, 0.9)) * 0.6)
    edges = ink & ~ndimage.binary_erosion(ink)
    sx, sy = image.width / width, image.height / height
    lines: dict[tuple[int, ...], list] = {}
    for word in words:
        if word.source_region != "blank_field" and any(character.isalpha() for character in word.text):
            lines.setdefault(word.line_id, []).append(word)
    strokes: dict[tuple[int, ...], float] = {}
    for key, members in lines.items():
        area = edge = 0
        for word in members:
            x0, y0 = max(0, int(word.box.x0 * sx)), max(0, int(word.box.y0 * sy))
            x1, y1 = int(np.ceil(word.box.x1 * sx)), int(np.ceil(word.box.y1 * sy))
            area += int(ink[y0:y1, x0:x1].sum())
            edge += int(edges[y0:y1, x0:x1].sum())
        if edge >= 40:
            strokes[key] = 2 * area / edge
    if len(strokes) < 3:
        return 0
    typical = float(np.median(list(strokes.values())))
    spans = {key: max(word.box.x1 for word in members) - min(word.box.x0 for word in members) for key, members in lines.items()}
    text_width = max(spans.values()) if spans else 0.0
    marked = 0
    for key, stroke in strokes.items():
        if photo and spans[key] >= PHOTO_HEADING_WIDTH * text_width:
            continue
        if stroke >= typical * 1.3:
            for word in lines[key]:
                word.bold = True
            marked += 1
    return marked
