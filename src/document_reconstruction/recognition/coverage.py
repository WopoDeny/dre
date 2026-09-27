"""Ink that looks like text but produced no words means silently lost content.

Everything recognized (words, empty fields, pictures, tables, ruling lines) is
masked out of the black ink layer. Letter-sized components that remain and line
up into a text-like cluster are text the recognizer did not read: another
script, a line hidden by a stamp, faint print. Such a page is not issued.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from ..model import Box, Graphic, Table
from .pixels import dilate_rect
from .types import Word

# A cluster needs this many letter-sized components to count as lost text.
MIN_LETTERS = 4
# Letters of small print (a note of the executor, "орынд: Э.Канатова") are about a quarter of a word box.
SMALL_LETTER = 0.25
STAMP_TEXT_DARKNESS = 110


@dataclass
class Coverage:
    clusters: list[Box]
    letters: int
    read_share: float = 1.0

    @property
    def lost_text(self) -> bool:
        return bool(self.clusters)


def _component_height(ink: np.ndarray) -> float:
    labels, _ = ndimage.label(ink)
    heights = [sl[0].stop - sl[0].start for sl in ndimage.find_objects(labels)
               if sl is not None and 3 <= sl[0].stop - sl[0].start <= 80 and sl[1].stop - sl[1].start <= 2.5 * (sl[0].stop - sl[0].start)]
    # Median component is about the x-height; a word box is about 1.6 of it.
    return float(np.median(heights)) * 1.6 if len(heights) >= 20 else 0.0


ROW_LETTERS = 6        # letters in one line height: printed text, not the loops of a signature
MARK_STROKE = 1.6      # × text height: ink this tall inside a left-out signature is its strokes
PEN_TINT = 12          # blue channel above red in the ink of a pen, on average


def _letter_row(mask: np.ndarray, text_h: float) -> bool:
    """A row of printed letters: at least ROW_LETTERS letter marks whose centers fit in one line height."""
    labels, count = ndimage.label(mask)
    if count < ROW_LETTERS:
        return False
    centers = sorted((slices[0].start + slices[0].stop) / 2 for slices in ndimage.find_objects(labels) if slices is not None)
    return any(sum(1 for other in centers[index:] if other - center <= text_h) >= ROW_LETTERS
               for index, center in enumerate(centers))


def _pen_ink(image: Image.Image, box: Box, *, width: float, height: float) -> bool:
    sx, sy = image.width / width, image.height / height
    crop = np.asarray(image.convert("RGB").crop((int(box.x0 * sx), int(box.y0 * sy), int(np.ceil(box.x1 * sx)), int(np.ceil(box.y1 * sy))))).astype(np.int16)
    if not crop.size:
        return False
    ink = crop.min(axis=2) < np.quantile(crop.min(axis=2), 0.9) - 40
    if ink.sum() < 20:
        return False
    tint = crop[..., 2][ink] - crop[..., 0][ink]
    return float(np.mean(tint)) >= PEN_TINT


def uncovered_text(
    ocr_image: Image.Image, words: list[Word], graphics: list[Graphic], tables: list[Table], *, width: float, height: float,
    stamps: list[Box] = (), skipped: list[Box] = (), colored: Image.Image | None = None, marks: list[Box] = (),
) -> Coverage:
    gray = np.asarray(ocr_image.convert("L"))
    scale = min(1.0, 1600 / max(gray.shape))
    if scale < 1.0:
        small = ocr_image.convert("L").resize((round(ocr_image.width * scale), round(ocr_image.height * scale)), Image.Resampling.BILINEAR)
        gray = np.asarray(small)
    h_px, w_px = gray.shape
    sx, sy = w_px / width, h_px / height
    paper = float(np.quantile(gray, 0.9))
    ink = gray < min(170.0, paper - 60.0)  # pale print is ink too
    for box in stamps:
        # Under a colored stamp the black layer keeps the stamp as a pale ring: only dark ink there is text.
        x0, y0 = max(0, int(box.x0 * w_px / width)), max(0, int(box.y0 * h_px / height))
        x1, y1 = min(w_px, int(np.ceil(box.x1 * w_px / width))), min(h_px, int(np.ceil(box.y1 * h_px / height)))
        ink[y0:y1, x0:x1] &= gray[y0:y1, x0:x1] < STAMP_TEXT_DARKNESS

    heights = [word.box.height * sy for word in words if word.confidence >= 0.75 and word.source_region != "blank_field"]
    # A few confident words give the text size better than ink components (specks and stamp remnants pull those down).
    text_h = float(np.median(heights)) if len(heights) >= 3 else _component_height(ink) or h_px / 70

    covered = np.zeros_like(ink)
    pad = max(2, int(text_h * 0.25))
    for box in [word.box for word in words] + [graphic.box for graphic in graphics] + [table.box for table in tables] + list(skipped):
        x0, y0 = max(0, int(box.x0 * sx) - pad), max(0, int(box.y0 * sy) - pad)
        x1, y1 = min(w_px, int(np.ceil(box.x1 * sx)) + pad), min(h_px, int(np.ceil(box.y1 * sy)) + pad)
        covered[y0:y1, x0:x1] = True
    # Ruling lines and underlines are not text.
    run = max(8, int(text_h * 3))
    rules = ndimage.maximum_filter1d(ndimage.minimum_filter1d(ink, run, axis=1), run, axis=1)
    rules |= ndimage.maximum_filter1d(ndimage.minimum_filter1d(ink, run, axis=0), run, axis=0)
    residual = ink & ~covered & ~ndimage.binary_dilation(rules, iterations=1)
    letters = np.zeros_like(residual)
    count = 0
    # Letters as they are, and after closing small gaps: faint typewriter or stamped digits break into pieces
    # ("001705"), while closing glues tight bold letters into one word-long blob ("Председатель").
    closed = ndimage.binary_closing(residual, structure=np.ones((3, 3), dtype=bool)) & ~covered
    for mask in (residual, closed):
        labels, _ = ndimage.label(mask)
        for index, slices in enumerate(ndimage.find_objects(labels), start=1):
            if slices is None:
                continue
            hh, ww = slices[0].stop - slices[0].start, slices[1].stop - slices[1].start
            area = int((labels[slices] == index).sum())
            if SMALL_LETTER * text_h <= hh <= 2.5 * text_h and ww <= 4 * text_h and area >= 0.02 * text_h * text_h:
                letters[slices] |= labels[slices] == index
                count += mask is residual

    joined = dilate_rect(letters, max(1, int(text_h * 0.3)), max(1, int(text_h * 1.2)))
    groups, _ = ndimage.label(joined)
    clusters: list[Box] = []
    for index, slices in enumerate(ndimage.find_objects(groups), start=1):
        if slices is None:
            continue
        members = ndimage.label(letters[slices] & (groups[slices] == index))[1]
        ys, xs = slices
        hh, ww = ys.stop - ys.start, xs.stop - xs.start
        near_edge = xs.start < w_px * 0.06 or xs.stop > w_px * 0.94
        if near_edge and hh > ww * 2:
            continue  # small vertical text on the margin is ignored
        if members >= MIN_LETTERS and ww >= 2 * text_h:
            box = Box(xs.start / sx, ys.start / sy, xs.stop / sx, ys.stop / sy)
            if colored is not None and _pen_ink(colored, box, width=width, height=height):
                continue  # a pale pen (a signature's loops) the color layer did not take: not print
            if (hh > MARK_STROKE * text_h and any(mark.contains_center(box) for mark in marks)
                    and not _letter_row(letters[slices] & (groups[slices] == index), text_h)):
                continue  # loops of a signature left out over the text: taller than a line, no row of letters
            clusters.append(box)
    # How much of the letter-like ink is owned by recognized words at all.
    word_mask = np.zeros_like(ink)
    for box in [word.box for word in words] + list(skipped):
        x0, y0 = max(0, int(box.x0 * sx) - pad), max(0, int(box.y0 * sy) - pad)
        x1, y1 = min(w_px, int(np.ceil(box.x1 * sx)) + pad), min(h_px, int(np.ceil(box.y1 * sy)) + pad)
        word_mask[y0:y1, x0:x1] = True
    pictures = np.zeros_like(ink)
    for graphic in graphics:
        box = graphic.box
        pictures[max(0, int(box.y0 * sy)):int(np.ceil(box.y1 * sy)), max(0, int(box.x0 * sx)):int(np.ceil(box.x1 * sx))] = True
    # Ink of pictures (an emblem, a logo) is no text that could be left unread.
    all_labels, _ = ndimage.label(ink & ~pictures & ~ndimage.binary_dilation(rules, iterations=1))
    letters_total = letters_read = 0
    for index, slices in enumerate(ndimage.find_objects(all_labels), start=1):
        if slices is None:
            continue
        hh, ww = slices[0].stop - slices[0].start, slices[1].stop - slices[1].start
        if SMALL_LETTER * text_h <= hh <= 2.5 * text_h and ww <= 4 * text_h:
            letters_total += 1
            cy, cx = (slices[0].start + slices[0].stop) // 2, (slices[1].start + slices[1].stop) // 2
            letters_read += bool(word_mask[cy, cx])
    return Coverage(clusters, count, letters_read / letters_total if letters_total >= 50 else 1.0)
