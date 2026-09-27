"""Underlined words: a rule drawn right under the words of a line, not a separator across the page.

The OCR reads the rule under a word as underscores glued to it ("__денсаулық") or as tokens of
underscores between words; both are the underline, not text. Such words keep their letters and
carry `underline`, so the DOCX shows them underlined like the source (form values: "Атауы: ____").
Empty fields (handwritten inserts) are not touched.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

from .types import Word


def mark_underlines(image: Image.Image, words: list[Word], *, width: float, height: float,
                    tables=()) -> tuple[list[Word], int]:
    gray = np.asarray(image.convert("L"))
    paper = float(np.quantile(gray, 0.9))
    ink = gray < min(170.0, paper - 60.0)
    sx, sy = image.width / width, image.height / height
    heights = [word.box.height * sy for word in words if word.confidence >= 0.75 and word.source_region != "blank_field"]
    if len(heights) < 10:
        return words, 0
    text_h = float(np.median(heights))
    run = max(8, int(text_h * 3))
    rules = ndimage.maximum_filter1d(ndimage.minimum_filter1d(ink, run, axis=1), run, axis=1)
    labels, _ = ndimage.label(rules)
    spans = ndimage.find_objects(labels)

    lines: dict[tuple[int, ...], list[Word]] = {}
    for word in words:
        if word.source_region != "blank_field":
            lines.setdefault(word.line_id, []).append(word)

    underlined: set[int] = set()
    for members in lines.values():
        lettered = [word for word in members if any(character.isalnum() for character in word.text)]
        if not lettered:
            continue
        line_x0 = min(word.box.x0 for word in lettered) * sx
        line_x1 = max(word.box.x1 for word in lettered) * sx
        for word in lettered:
            if any(table.contains_center(word.box) for table in tables):
                continue  # under a word in a cell runs the cell border, not an underline
            x0, x1 = int(word.box.x0 * sx), int(np.ceil(word.box.x1 * sx))
            h = max(1.0, word.box.height * sy)
            top = max(0, int(word.box.y1 * sy - 0.25 * h))
            bottom = min(gray.shape[0], int(np.ceil(word.box.y1 * sy + 0.45 * h)))
            if x1 <= x0 or bottom <= top:
                continue
            band = labels[top:bottom, x0:x1]
            if (band > 0).any(axis=0).mean() < 0.7:
                continue
            # The rule belongs to this line only (an underline), not a separator across the page.
            ids = [int(value) for value in np.unique(band) if value]
            if all(spans[i - 1][1].start >= line_x0 - 3 * text_h and spans[i - 1][1].stop <= line_x1 + 3 * text_h
                   for i in ids if spans[i - 1] is not None):
                underlined.add(id(word))

    if not underlined:
        return words, 0
    result: list[Word] = []
    for word in words:
        core = word.text.strip("_")
        if id(word) in underlined:
            word.underline = True
            if core and core != word.text:
                word.text = core
                if word.alternative:
                    word.alternative = word.alternative.strip("_") or word.alternative
            result.append(word)
            continue
        if word.source_region != "blank_field" and not core and word.text:
            # Underscores lying on the same rule as underlined words of their line: the rule itself.
            neighbours = [other for other in lines.get(word.line_id, []) if id(other) in underlined]
            if neighbours and any(abs(other.box.x0 - word.box.x1) <= 3 * text_h / sy or abs(word.box.x0 - other.box.x1) <= 3 * text_h / sy
                                  for other in neighbours):
                continue
        result.append(word)
    return result, len(underlined)
