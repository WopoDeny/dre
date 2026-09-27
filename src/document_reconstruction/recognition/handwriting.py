"""Black-ink handwriting that the reader took for words.

Colored ink (a blue pen) is separated before recognition. Black ink is not, and the reader
turns it into "words" that both readings see differently ("Woh|Guh", "2025|BORZ"). Such
words are never text of the document:

- a scribble much taller than the print is a signature: its ink becomes a picture;
- weak words written on a printed form line (a date, a number) become an empty field, as
  TASK.md asks for handwritten inserts; a printed "№" in front of them stays.

Both need all of: low recognition confidence, readings that have little in common, and the
geometry of handwriting. Printed text read badly keeps similar readings or confident letters
and stays an error (the page is refused), so print is never blanked here. Nothing inside a
seal or stamp is touched: printed text there must be read.
"""

from __future__ import annotations

import difflib

import numpy as np
from PIL import Image
from scipy import ndimage

from ..model import Box, Graphic
from .restore import in_lookalikes, line_script
from .types import Word
from .vote import normalized

WEAK = 0.7                 # recognition confidence of a handwritten "word"
SIGNATURE_WEAK = 0.6
DISSIMILAR = 0.5           # similarity of the two readings below this: they saw different things
SIGNATURE_HEIGHT = 2.0     # × the typical printed word height
FIELD_RUN = 6


def _dissimilar(word: Word) -> bool:
    if word.alternative is None:
        return False
    if not word.alternative:
        return True
    return difflib.SequenceMatcher(None, normalized(word.text), normalized(word.alternative)).ratio() < DISSIMILAR


def _typical_height(words: list[Word]) -> float:
    printed = sorted(word.box.height for word in words
                     if word.confidence >= 0.9 and sum(character.isalpha() for character in word.text) >= 3)
    return printed[len(printed) // 2] if len(printed) >= 10 else 0.0


def signatures_from_scribbles(image: Image.Image, words: list[Word], protected: list[Box], *, index: int,
                              width: float, height: float, asset) -> tuple[list[Word], list[Graphic]]:
    typical = _typical_height(words)
    if not typical:
        return words, []
    kept: list[Word] = []
    graphics: list[Graphic] = []
    for word in words:
        if (word.source_region != "blank_field" and word.confidence < SIGNATURE_WEAK and _dissimilar(word)
                and word.box.height >= SIGNATURE_HEIGHT * typical
                and not any(box.contains_center(word.box) for box in protected)):
            pad = 0.15 * word.box.height
            box = Box(max(0.0, word.box.x0 - pad), max(0.0, word.box.y0 - pad), min(width, word.box.x1 + pad), min(height, word.box.y1 + pad))
            graphics.append(Graphic(box, index, asset(image, box, width=width, height=height, mode="dark"),
                                    description="Black-ink signature", role="signature", confidence=0.7,
                                    source_region="raster_graphic:black_handwriting"))
            continue
        kept.append(word)
    return kept, graphics


def _lines(words: list[Word]) -> dict[tuple[int, ...], list[Word]]:
    lines: dict[tuple[int, ...], list[Word]] = {}
    for word in words:
        lines.setdefault(word.line_id, []).append(word)
    return lines


def handwritten_fields(image: Image.Image, words: list[Word], *, width: float, height: float,
                       tables: list[Box] = ()) -> tuple[list[Word], int]:
    typical = _typical_height(words)
    if not typical:
        return words, 0
    gray = np.asarray(image.convert("L"))
    ink = gray < min(170.0, float(np.quantile(gray, 0.9)) - 60.0)
    sx, sy = image.width / width, image.height / height
    run_px = max(8, int(typical * sy * 3))
    rules = ndimage.maximum_filter1d(ndimage.minimum_filter1d(ink, run_px, axis=1), run_px, axis=1)

    def on_rule(word: Word) -> bool:
        x0, x1 = int(word.box.x0 * sx), int(np.ceil(word.box.x1 * sx))
        top = max(0, int((word.box.y1 - 0.4 * typical) * sy))
        bottom = min(gray.shape[0], int(np.ceil((word.box.y1 + 0.8 * typical) * sy)))
        return x1 > x0 and bottom > top and rules[top:bottom, x0:x1].any(axis=0).mean() >= 0.6

    scripts = {key: line_script([word for word in members]) for key, members in _lines(words).items()}

    def handwritten(word: Word) -> bool:
        if in_lookalikes(word.text, scripts.get(word.line_id)) is not None:
            return False  # print read with the other alphabet's letters ("коʻсһаѕі"), not a pen
        # A table border under a word is no form line: words in cells are printed and are read.
        return (word.source_region != "blank_field" and word.confidence < WEAK and _dissimilar(word)
                and any(character.isalnum() for character in word.text) and on_rule(word)
                and not any(table.contains_center(word.box) for table in tables))

    lines: dict[tuple[int, ...], list[Word]] = {}
    for word in words:
        lines.setdefault(word.line_id, []).append(word)
    replaced: dict[int, list[Word]] = {}
    fields = 0
    for members in lines.values():
        members.sort(key=lambda word: word.box.x0)
        position = 0
        while position < len(members):
            if not handwritten(members[position]):
                position += 1
                continue
            end = position
            while end + 1 < len(members) and handwritten(members[end + 1]):
                end += 1
            run = members[position:end + 1]
            if len(run) <= FIELD_RUN:
                first = run[0]
                parts: list[Word] = []
                if first.text.startswith("№"):
                    # "№91" for "№ 34": the printed number sign stays, the handwriting goes.
                    parts.append(Word("№", Box(first.box.x0, first.box.y0, first.box.x0 + first.box.height, first.box.y1), 1.0,
                                      size=first.size, line_id=first.line_id, source_page=first.source_page, source_region="ocr",
                                      alternative="№"))
                box = Box(min(word.box.x0 for word in run) + (first.box.height if parts else 0.0), min(word.box.y0 for word in run),
                          max(word.box.x1 for word in run), max(word.box.y1 for word in run))
                length = max(3, int(round(box.width / max(1.0, box.height * 0.5))))
                parts.append(Word("_" * length, box, 1.0, size=first.size, line_id=first.line_id,
                                  source_page=first.source_page, source_region="blank_field"))
                replaced[id(first)] = parts
                for word in run[1:]:
                    replaced[id(word)] = []
                fields += 1
            position = end + 1
    if not fields:
        return words, 0
    result: list[Word] = []
    for word in words:
        result.extend(replaced.get(id(word), [word]))
    return result, fields
