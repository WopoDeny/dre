"""Two different readings of every line; only text read identically twice is trusted.

The second reading uses another image: a different scale and a local (adaptive)
binarization, so its segmentation and character decisions are independent
enough to disagree where the first reading guessed.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
from PIL import Image
from scipy import ndimage

from ..core import DeadlineBudget
from .types import Word

SECOND_SCALE = 1.3
_APOSTROPHES = str.maketrans({"’": "ʻ", "‘": "ʻ", "ʼ": "ʻ", "`": "ʻ", "'": "ʻ", "—": "-", "–": "-", "‐": "-", "‑": "-"})
# "З" alone is not a word in any supported language: it is the digit 3 ("З." in a numbered list).
LONE_ZE = re.compile(r"^[Зз](?=[.)]?$)")


# The same glyph read as a Latin or a Cyrillic letter is the same reading; which alphabet
# the word belongs to is decided by the line-level script rules, not by the vote.
_LOOKALIKES = str.maketrans(dict(zip("ABCEHKMOPTXYaceopxy", "АВСЕНКМОРТХУасеорху")))


def normalized(text: str) -> str:
    return unicodedata.normalize("NFC", text).translate(_APOSTROPHES).translate(_LOOKALIKES).replace(" ", "").strip("|[]{}_")


def second_image(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    gray = gray.resize((round(gray.width * SECOND_SCALE), round(gray.height * SECOND_SCALE)), Image.Resampling.LANCZOS)
    values = np.asarray(gray).astype(np.float32)
    window = max(15, int(min(values.shape) / 40)) | 1
    local = ndimage.uniform_filter(values, size=window)
    binary = np.where(values < local - 12, 0, 255).astype(np.uint8)
    return Image.fromarray(binary, mode="L")


def second_reading(image: Image.Image, backend, languages: str, budget: DeadlineBudget, *, width: float, height: float) -> list[Word]:
    return backend.recognize(second_image(image), languages, budget, page_width=width, page_height=height, psm=3)


def attach_alternatives(words: list[Word], alternatives: list[Word]) -> int:
    """Give every word the text the second reading found in the same place; returns agreements."""
    agreed = 0
    for word in words:
        slack = 0.15 * word.box.height
        inside = [
            other for other in alternatives
            if word.box.x0 - slack <= (other.box.x0 + other.box.x1) / 2 <= word.box.x1 + slack
            and min(word.box.y1, other.box.y1) - max(word.box.y0, other.box.y0) > 0.5 * min(word.box.height, other.box.height)
        ]
        inside.sort(key=lambda other: other.box.x0)
        word.alternative = "".join(other.text for other in inside)
        agreed += agrees(word)
    return agreed


def agrees(word: Word) -> bool:
    """Both readings saw the same characters. Even a missed comma counts: it can be a lost
    fragment ("koʻchasi, 5-uy" read as "koʻchasi -uy")."""
    return bool(word.alternative) and normalized(word.alternative) == normalized(word.text)


def disagrees(word: Word) -> bool:
    """The second reading saw something else, or nothing at all, where this word is."""
    return word.alternative is not None and not agrees(word)


def digits_disagree(word: Word) -> bool:
    """The two readings see different digits: only a third reading may decide."""
    return bool(word.alternative) and not agrees(word) and digits(word.text) != digits(word.alternative)


def digits(text: str) -> str:
    return "".join(character for character in text if character.isdigit())


def _third_reading(image: Image.Image, word: Word, backend, languages: str, budget: DeadlineBudget, *, width: float, height: float,
                   scale: float = 1.5) -> str | None:
    sx, sy = image.width / width, image.height / height
    h = word.box.height
    left, right = max(0.0, word.box.x0 - 0.25 * h), min(width, word.box.x1 + 0.25 * h)
    top, bottom = max(0.0, word.box.y0 - 0.3 * h), min(height, word.box.y1 + 0.3 * h)
    crop = image.crop((int(left * sx), int(top * sy), int(right * sx), int(bottom * sy)))
    if crop.width < 8 or crop.height < 8:
        return None
    crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.Resampling.LANCZOS)
    found = backend.recognize(crop, languages, budget, page_width=right - left, page_height=bottom - top, psm=7)
    return "".join(candidate.text for candidate in sorted(found, key=lambda candidate: candidate.box.x0))


def confirm_numbers(image: Image.Image, words: list[Word], backend, languages: str, budget: DeadlineBudget, *, width: float, height: float) -> int:
    """Numbers need a third, line-mode reading of their own crop.

    Two page readings of the same image can share a mistake ("2024 il" -> "20241");
    a crop read as a single line segments differently. When the two page readings
    agree, the third must see the same digits. When they disagree, the digits two
    of the three readings share win; otherwise the number stays read two ways,
    which is an error that nothing may repair.
    """
    checked = 0
    for word in words:
        word.text = LONE_ZE.sub("3", word.text)
        if word.alternative:
            word.alternative = LONE_ZE.sub("3", word.alternative)
        if not any(character.isdigit() for character in word.text + (word.alternative or "")):
            continue
        budget.require_fit(0, reserve_ms=3_000, stage="ocr")  # an unchecked number would change the result
        if not agrees(word) and not digits_disagree(word):
            continue  # same digits, different letters: the dictionary decides
        third = _third_reading(image, word, backend, languages, budget, width=width, height=height)
        if third is None:
            continue
        checked += 1
        third = LONE_ZE.sub("3", third)
        if agrees(word):
            # A crop read with no digits at all ("2." -> "д.") is the line reader failing on one
            # glyph, not other digits; other digits ("2024il" -> "20241") overrule the agreement.
            if digits(third) and digits(third) != digits(word.text) or not third.strip():
                word.alternative = third or "?"
        elif digits(third) == digits(word.text):
            word.alternative = word.text
        elif normalized(third) == normalized(word.alternative):
            word.text = word.alternative
    checked += _confirm_initials(image, words, backend, languages, budget, width=width, height=height)
    _confirm_list_numbers(words)
    return checked


# "М.Ж.", "И.Фролова", "Т.З.Шакирову": initials cannot be checked against a dictionary, and two page
# readings share the same slips (Ж read as Х). The crop, enlarged, is read a third time.
INITIALS = re.compile(r"^((?:[A-ZА-ЯЁӘҒҚҢӨҰҮҺІЎҲӢӮҶ]\.){1,3})")


def initials(text: str) -> str:
    match = INITIALS.match(normalized(text))
    return match.group(1) if match else ""


def _confirm_initials(image: Image.Image, words: list[Word], backend, languages: str, budget: DeadlineBudget, *, width: float, height: float) -> int:
    checked = 0
    for word in words:
        own = initials(word.text)
        if not own and not initials(word.alternative or ""):
            continue
        budget.require_fit(0, reserve_ms=3_000, stage="ocr")  # unchecked initials would change the result
        third = _third_reading(image, word, backend, languages, budget, width=width, height=height, scale=1.5)
        if third is None:
            continue
        checked += 1
        if agrees(word):
            # A crop read with no initials at all ("Т." -> "1") is the line reader failing on a lone glyph;
            # other initials ("М.Ж." -> "М.Х.") overrule the agreement.
            if initials(third) and initials(third) != own:
                word.alternative = third
        elif initials(third) == own and own:
            word.alternative = word.text
        elif word.alternative and initials(third) == initials(word.alternative):
            word.text = word.alternative
        else:
            word.alternative = word.alternative if word.alternative is not None else (third or "?")
    return checked


LIST_NUMBER = re.compile(r"^(\d{1,2})[.)]$")


def _confirm_list_numbers(words: list[Word]) -> None:
    """A disputed item number at the start of a line is trusted when it continues the
    sequence of item numbers read identically twice ("1." "2." "3." down the page)."""
    first: dict[tuple, Word] = {}
    for word in words:
        if word.line_id not in first or word.box.x0 < first[word.line_id].box.x0:
            first[word.line_id] = word
    items = sorted((word for word in first.values() if LIST_NUMBER.match(word.text)), key=lambda word: word.box.y0)
    for index, word in enumerate(items):
        if agrees(word):
            continue
        number = int(LIST_NUMBER.match(word.text).group(1))
        before = items[index - 1] if index else None
        after = items[index + 1] if index + 1 < len(items) else None
        follows = before is not None and agrees(before) and int(LIST_NUMBER.match(before.text).group(1)) == number - 1
        opens = number == 1 and before is None and after is not None and agrees(after) and int(LIST_NUMBER.match(after.text).group(1)) == 2
        if follows or opens:
            word.alternative = word.text
