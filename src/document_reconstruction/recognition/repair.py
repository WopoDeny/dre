"""Second reading of the words that failed the dictionary check.

Only the line around each unreliable word is recognized again (single-line
mode, the more accurate model when installed). A new reading replaces the word
only if it passes the same dictionary check; nothing is guessed.
"""

from __future__ import annotations

from PIL import Image

from ..core import DeadlineBudget
from ..model import Box
from .languages import letter_scripts
from .lexicon import Lexicon
from .types import Word
from .vote import agrees, digits_disagree, normalized
from .wordcheck import word_verdict

MAX_REPAIRS = 30
MIN_CONFIDENCE = 0.6
STEP_RESERVE_MS = 3_000


def repair_words(
    ocr_image: Image.Image, words: list[Word], errors: list[Word], backend, languages: str, budget: DeadlineBudget,
    lexicon: Lexicon, *, width: float, height: float,
) -> tuple[list[Word], int]:
    sx, sy = ocr_image.width / width, ocr_image.height / height
    replacements: dict[int, Word] = {}
    page_letters = letter_scripts(" ".join(word.text for word in words))
    # Several neighbouring errors on one line are often one token split apart
    # ("02 | пайуе | ги | тето" for "02_native_ru_memo"): read the span at once.
    words, handled = _repair_runs(ocr_image, words, errors, backend, languages, budget, lexicon, width=width, height=height)
    errors = [word for word in errors if id(word) not in handled and not digits_disagree(word)]
    for word in errors[:MAX_REPAIRS]:
        budget.require_fit(0, reserve_ms=STEP_RESERVE_MS, stage="ocr")  # a skipped check would change the result
        h = word.box.height
        left, right = max(0.0, word.box.x0 - 3 * h), min(width, word.box.x1 + 3 * h)
        top, bottom = max(0.0, word.box.y0 - 0.45 * h), min(height, word.box.y1 + 0.45 * h)
        crop = ocr_image.crop((int(left * sx), int(top * sy), int(right * sx), int(bottom * sy)))
        if crop.width < 8 or crop.height < 8:
            continue
        crop = crop.resize((int(crop.width * 1.5), int(crop.height * 1.5)), Image.Resampling.LANCZOS)
        # The page profile first, then only the alphabet that dominates the line:
        # a mixed profile can read a Cyrillic letter as its Latin lookalike.
        line_letters = letter_scripts(" ".join(other.text for other in words if other.line_id == word.line_id))
        if sum(line_letters.values()) < 3:
            line_letters = page_letters  # a date line has too few letters to show its alphabet
        readings = []
        for profile in _profiles(languages, line_letters):
            replacement, reading = _read_word(crop, word, profile, backend, budget, left, top, right, bottom, lexicon)
            readings.append(reading)
            if replacement is not None:
                replacements[id(word)] = replacement
                break
        else:
            if any(reading is not None and reading.text == word.text and reading.confidence >= 0.5 for reading in readings):
                # Two independent readings agree: read stably (a name, a rare form). This
                # is not correctness: consistent junk also reads the same twice.
                replacements[id(word)] = _copy(word, word.confidence, "ocr_confirmed")
            else:
                # The readings disagree: handwriting, a smudge or a blurred field.
                replacements[id(word)] = _copy(word, word.confidence, "ocr_unstable")
    changed = len(handled) + sum(1 for replacement in replacements.values() if replacement.source_region == "ocr_repair")
    result = [replacements.get(id(word), word) for word in words]
    return drop_duplicates(result, {id(word) for word in errors} | {id(replacement) for replacement in replacements.values()}), changed


def drop_duplicates(words: list[Word], doubtful: set[int]) -> list[Word]:
    """The page reader sometimes gives one printed line twice (two line numbers). A word of the
    other copy lies inside a trusted word (reread, or read the same twice) of the same ink: if it
    failed the check or is a piece of that word's text, it is dropped."""
    def is_trusted(word: Word) -> bool:
        return word.source_region == "ocr_repair" or agrees(word)
    # Trusted and larger words claim their ink first; of two equal copies one stays.
    order = sorted(words, key=lambda word: (not is_trusted(word), -word.box.width * word.box.height))
    kept: list[Word] = []
    dropped: set[int] = set()
    for word in order:
        cx, cy = (word.box.x0 + word.box.x1) / 2, (word.box.y0 + word.box.y1) / 2
        own = normalized(word.text).lower()
        if any(is_trusted(other) and other.line_id != word.line_id
               and other.box.x0 <= cx <= other.box.x1 and other.box.y0 <= cy <= other.box.y1
               and (id(word) in doubtful or (own and own in normalized(other.text).lower()))
               for other in kept):
            dropped.add(id(word))
        else:
            kept.append(word)
    return [word for word in words if id(word) not in dropped]


def _profiles(languages: str, line_letters: dict[str, int]) -> list[str]:
    """The page profile, the same models in the other order, the line's own alphabet."""
    profiles = [languages]
    models = languages.split("+")
    if len(models) > 1:
        profiles.append("+".join(reversed(models)))
        if line_letters:
            dominant = max(line_letters, key=line_letters.get)
            if dominant in models:
                profiles.append(dominant)
    return profiles


def _repair_runs(ocr_image, words: list[Word], errors: list[Word], backend, languages: str, budget: DeadlineBudget,
                 lexicon: Lexicon, *, width: float, height: float) -> tuple[list[Word], set[int]]:
    sx, sy = ocr_image.width / width, ocr_image.height / height
    error_ids = {id(word) for word in errors}
    lines: dict[tuple[int, ...], list[Word]] = {}
    for word in words:
        lines.setdefault(word.line_id, []).append(word)
    runs: list[list[Word]] = []
    for members in lines.values():
        members.sort(key=lambda word: word.box.x0)
        current: list[Word] = []
        for index, word in enumerate(members + [None]):
            following = members[index + 1] if index + 1 < len(members) else None
            # A short word between two errors ("ги") belongs to the same broken token.
            bridge = (word is not None and current and len(word.text) <= 3
                      and following is not None and id(following) in error_ids)
            if word is not None and (id(word) in error_ids or bridge):
                current.append(word)
            else:
                if len(current) >= 2:
                    runs.append(current)
                current = []
    handled: set[int] = set()
    result = list(words)
    for run in runs[:MAX_REPAIRS]:
        budget.require_fit(0, reserve_ms=STEP_RESERVE_MS, stage="ocr")  # a skipped check would change the result
        span = Box.enclosing([word.box for word in run])
        h = max(word.box.height for word in run)
        left, right = max(0.0, span.x0 - 2 * h), min(width, span.x1 + 2 * h)
        top, bottom = max(0.0, span.y0 - 0.45 * h), min(height, span.y1 + 0.45 * h)
        crop = ocr_image.crop((int(left * sx), int(top * sy), int(right * sx), int(bottom * sy)))
        if crop.width < 8 or crop.height < 8:
            continue
        line_letters = letter_scripts(" ".join(word.text for word in lines[run[0].line_id]))
        for profile in _profiles(languages, line_letters):
            found = backend.recognize(crop, profile, budget, page_width=right - left, page_height=bottom - top, psm=7)
            inside = []
            for candidate in found:
                box = Box(candidate.box.x0 + left, candidate.box.y0 + top, candidate.box.x1 + left, candidate.box.y1 + top)
                if span.x0 - 0.3 * h <= (box.x0 + box.x1) / 2 <= span.x1 + 0.3 * h:
                    inside.append(Word(candidate.text, box, candidate.confidence, size=run[0].size, line_id=run[0].line_id,
                                       source_page=run[0].source_page, source_region="ocr_repair", language_profile=profile))
            covered = sum(word.box.width for word in inside)
            if (inside and covered >= 0.8 * span.width and all(word.confidence >= MIN_CONFIDENCE for word in inside)
                    and all(word_verdict(word, lexicon)[0] == "ok" for word in inside)):
                position = result.index(run[0])
                result = [word for word in result if word not in run]
                result[position:position] = inside
                handled.update(id(word) for word in run)
                break
    return result, handled


def _copy(word: Word, confidence: float, region: str) -> Word:
    return Word(word.text, word.box, confidence, size=word.size, line_id=word.line_id, source_page=word.source_page,
                source_region=region, script_evidence=word.script_evidence, language_profile=word.language_profile,
                alternative=word.alternative)


def _read_word(crop, word: Word, profile: str, backend, budget: DeadlineBudget, left: float, top: float, right: float, bottom: float, lexicon: Lexicon) -> tuple[Word | None, Word | None]:
    found = backend.recognize(crop, profile, budget, page_width=right - left, page_height=bottom - top, psm=7)
    inside = []
    for candidate in found:
        box = Box(candidate.box.x0 + left, candidate.box.y0 + top, candidate.box.x1 + left, candidate.box.y1 + top)
        overlap = max(0.0, min(box.x1, word.box.x1) - max(box.x0, word.box.x0))
        if overlap >= 0.5 * min(box.width, word.box.width):
            inside.append((box, candidate))
    if len(inside) != 1:
        return None, None  # the second reading splits or merges the word: no safe replacement
    box, candidate = inside[0]
    punctuation = {character for character in candidate.text if not character.isalnum()}
    if (candidate.confidence < MIN_CONFIDENCE or not 0.6 <= box.width / max(1.0, word.box.width) <= 1.6
            or not punctuation <= {character for character in word.text if not character.isalnum()}):
        return None, candidate  # weak, differently sized, or adds punctuation: not the same word read better
    readings = {normalized(word.text)} | ({normalized(word.alternative)} if word.alternative else set())
    majority = normalized(candidate.text) in readings  # the third reading matches one of the first two
    replacement = Word(candidate.text, word.box, candidate.confidence, size=word.size, line_id=word.line_id,
                       source_page=word.source_page, source_region="ocr_repair",
                       script_evidence=word.script_evidence, language_profile=profile,
                       alternative=candidate.text if majority else None)
    if majority:
        return replacement, candidate
    return (replacement if replacement.text != word.text and word_verdict(replacement, lexicon)[0] == "ok" else None), candidate


def read_regions(
    ocr_image: Image.Image, regions: list[Box], backend, languages: str, budget: DeadlineBudget, *, width: float, height: float,
) -> list[Word]:
    """Recognize regions that the page layout analysis skipped, each as one text block."""
    sx, sy = ocr_image.width / width, ocr_image.height / height
    found: list[Word] = []
    for number, region in enumerate(regions):
        budget.require_fit(0, reserve_ms=STEP_RESERVE_MS, stage="ocr")  # a skipped check would change the result
        pad = max(4.0, region.height * 0.15)
        left, top = max(0.0, region.x0 - pad), max(0.0, region.y0 - pad)
        right, bottom = min(width, region.x1 + pad), min(height, region.y1 + pad)
        crop = ocr_image.crop((int(left * sx), int(top * sy), int(right * sx), int(bottom * sy)))
        if crop.width < 8 or crop.height < 8:
            continue
        for word in backend.recognize(crop, languages, budget, page_width=right - left, page_height=bottom - top, psm=6):
            word.box = Box(word.box.x0 + left, word.box.y0 + top, word.box.x1 + left, word.box.y1 + top)
            if region.contains_center(word.box):
                word.line_id = (8000 + number,) + word.line_id
                word.source_region = "ocr_region"
                found.append(word)
    return found
