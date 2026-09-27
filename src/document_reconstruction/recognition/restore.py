"""Doubtful words are restored the way a person would read them, not refused.

For every word the checks left doubtful the candidates are: both readings, the reading moved to the
alphabet of its line when it is written in lookalikes ("kb" -> "кв"), and dictionary words one typical
OCR slip away (a lookalike or diacritic letter, a letter lost or added, two letters swapped). A candidate
must be close to what one of the readings saw and be a real word: known to Hunspell, used elsewhere on
the page, or a longer word of the page languages. The most probable one is taken. Every change is
reported (was -> became, how).

Small print that cannot be read (letterhead requisites, footers, serial lines) is removed. What remains
unreadable in the body text decides whether the page is readable at all.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from statistics import median

from .languages import letter_scripts
from .lexicon import Lexicon, normalize
from .types import Word
from .vote import normalized

TOKEN_CORE = re.compile(r"^([^\w]*)(.*?)([^\w]*)$", re.S)
SIMILAR = 0.6                  # a candidate must be this close to one of the readings
SMALL_PRINT = 0.8              # lines lower than this share of the body text height are small print
SPECK_CONFIDENCE = 0.6         # a lone weak letter or two below this is a speck or a stroke, not text
EDGE_ZONE = (0.12, 0.85)       # top / bottom share of the page: letterhead and footer lines
SERVICE_LINE_WORDS = 6         # a footer or requisites line at the edge is long; a signature line is short
JUNK_LINE_SHARE = 0.5          # small-print lines with at least this share of doubtful words are removed
_LOOKALIKES = list(zip("ABCEHKMOPTXYaceopxyk", "АВСЕНКМОРТХУасеорхук"))
_TO_CYRILLIC = dict(_LOOKALIKES) | {"b": "в"}            # "kb." for "кв." (square metres)
_TO_LATIN = {cyrillic: latin for latin, cyrillic in _LOOKALIKES} | dict(zip("һѕіјԁӏ", "hsijdl"))


@dataclass
class Restoration:
    words: list[Word]
    changes: list[dict[str, str]] = field(default_factory=list)
    dropped_lines: list[str] = field(default_factory=list)
    body_tokens: int = 0
    lost_tokens: int = 0
    garbled: list[str] = field(default_factory=list)

    @property
    def lost_share(self) -> float:
        return self.lost_tokens / max(1, self.body_tokens)


def _case_like(model: str, word: str) -> str:
    if model.isupper() and len(model) > 1:
        return word.upper()
    if model[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


def _split(text: str) -> tuple[str, str, str]:
    match = TOKEN_CORE.match(text)
    return (match.group(1), match.group(2), match.group(3)) if match else ("", text, "")


def _similar(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, normalized(left).lower(), normalized(right).lower()).ratio()


def _script(text: str) -> str | None:
    scripts = letter_scripts(text)
    return max(scripts, key=scripts.get) if scripts else None


def _dominant(words: list[Word]) -> str | None:
    scripts: dict[str, int] = {}
    for word in words:
        for script, count in letter_scripts(word.text).items():
            scripts[script] = scripts.get(script, 0) + count
    return max(scripts, key=scripts.get) if sum(scripts.values()) >= 3 else None


class Restorer:
    def __init__(self, lexicon: Lexicon, languages: set[str]) -> None:
        self.lexicon = lexicon
        self.languages = languages

    def acceptable(self, word: str, vocabulary: dict[str, int]) -> tuple[bool, bool]:
        """(a real word, known to Hunspell or used elsewhere on the page)."""
        if self.lexicon.spelled_any(word) or vocabulary.get(word.lower(), 0) > 0:
            return True, True
        return len(word) >= 5 and self.lexicon._in_languages(normalize(word), self.languages or None), False

    def restore(self, word: Word, vocabulary: dict[str, int], line_script: str | None,
                proper: bool = False) -> tuple[str, str] | None:
        """(new text, how); ("", "") when the reading itself is the best word; None when nothing fits."""
        lead, core, tail = _split(word.text)
        if any(character.isdigit() for character in core):
            return _year_with_suffix(word)
        if not core:
            return None
        other = _split(word.alternative or "")[1]
        readings: list[tuple[str, float, str]] = [(core, 2.0, "reading")]
        if other and other != core and not any(character.isdigit() for character in other):
            readings.append((other, 1.0, "second reading"))
        for reading, weight, _ in list(readings):
            if "ё" in reading.lower():
                readings.append((reading.replace("ё", "е").replace("Ё", "Е"), weight, "reading"))  # "отдёла": a speck over «е»
        table = _TO_CYRILLIC if line_script == "Cyrillic" else _TO_LATIN if line_script == "Latin" else None
        for reading, weight, _ in list(readings):
            # A word written in lookalikes of the other alphabet ("BOJJOCTOKOB" in a Cyrillic line).
            if table and _script(reading) not in (None, line_script) and all(ch in table or _script(ch) != _script(reading) for ch in reading):
                readings.append(("".join(table.get(ch, ch) for ch in reading), weight - 0.2, "alphabet"))
        if "-" in core.strip("-"):
            parts = core.split("-")
            fixed = []
            for part in parts:
                piece = Word(part, word.box, word.confidence, alternative=None) if part else None
                restored = (self.restore(piece, vocabulary, line_script, proper)
                            if piece is not None and len(part) >= 3 else None)
                known = piece is None or len(part) < 3 or self.acceptable(part, vocabulary)[0]
                if restored and restored[0]:
                    fixed.append(restored[0])
                elif known or (restored is not None and restored[0] == ""):
                    fixed.append(part)
                else:
                    fixed = []
                    break
            if fixed:
                text = lead + "-".join(fixed) + tail
                return (text, "document" if proper else "dictionary") if text != word.text else ("", "")
        best: tuple[float, str, str] | None = None
        for reading, weight, source in readings:
            if len(letter_scripts(reading)) > 1 or (line_script and _script(reading) not in (None, line_script)):
                continue
            candidates: list[tuple[str, float, str]] = []
            known, strong = self.acceptable(reading, vocabulary)
            if known:
                candidates.append((reading, 10 + 3 * strong + weight, source))
            elif source == "alphabet" and len(reading) >= 3:
                candidates.append((reading, 7 + weight, source))  # lookalikes of the line's own alphabet
            for rank, variant in enumerate(self.lexicon.corrections(reading, self.languages or None, limit=8)):
                if len(reading) <= 2 and len(variant) != len(reading):
                    continue  # a letter or two is too little to grow a word from ("Ea" is no "Ева")
                if reading.lower().startswith(variant.lower()):
                    continue  # cutting the end off ("собра" → "собр") makes a word, not the word that was printed
                if proper and vocabulary.get(variant.lower(), 0) == 0:
                    continue  # a capitalized word may be a name: only the readings and the document's own words decide
                known, strong = self.acceptable(variant, vocabulary)
                if known:
                    candidates.append((variant, 6 + 3 * strong - 0.5 * rank + 0.5 * weight, "document" if proper else "dictionary"))
            for candidate, score, how in candidates:
                if proper and how != "alphabet" and candidate != core and vocabulary.get(candidate.lower(), 0) == 0:
                    continue  # a name keeps its own reading unless the document itself spells it the other way
                if len(candidate) < len(core) and core.lower().startswith(candidate.lower()):
                    continue  # a cut-off reading ("собр" for "собра…") is not the printed word
                closeness = max(_similar(candidate, core), _similar(candidate, other))
                if closeness < SIMILAR:
                    continue
                score += 2 * closeness + 3 * (vocabulary.get(candidate.lower(), 0) > 0)
                if best is None or score > best[0]:
                    best = (score, candidate, how)
        if best is None:
            return None
        text = lead + _case_like(core, best[1]) + tail
        return (text, best[2]) if text != word.text else ("", "")


YEAR_SUFFIX = re.compile(r"((?:19|20)\d\d)\s?([^\W\d_]{1,3}\.[,;:]?)")


def _year_with_suffix(word: Word) -> tuple[str, str] | None:
    """"2024." and "2024й." read for one word: the year with its abbreviation ("2024 й.", "2024 ý.", "2024 ж.")."""
    for reading in (word.text, word.alternative or ""):
        match = YEAR_SUFFIX.search(reading)
        if match and (match.group(1) in word.text) and (match.group(1) in (word.alternative or "")):
            text = f"{match.group(1)} {match.group(2)}"
            return (text, "second reading") if text != word.text else ("", "")
    return None


def _abbreviation_after_name(word: Word, previous: Word | None, languages: set[str]) -> str | None:
    """"Астана K.," for "Астана қ.,": after a place name a Kazakh page abbreviates қала as "қ."."""
    lead, core, tail = _split(word.text)
    if (core in ("K", "k", "К", "к") and tail.startswith(".") and "kaz" in languages and previous is not None
            and _split(previous.text)[1][:1].isupper() and _script(previous.text) == "Cyrillic"):
        return lead + "қ" + tail
    return None


GARBAGE_SIGNS = set("<>#~^={}[]\\|¦")
OPENING, CLOSING = "[{", "]}"


def _stray_brackets(members: list[Word], result: Restoration) -> None:
    """A bracket with no pair on its line is the edge of a rule, a frame or a stroke read as a sign
    ("{Директор", "[Серікбаев"); after a «...» opened on the line it is the closing quote ("«14}")."""
    line = " ".join(word.text for word in members)
    for word in members:
        text = word.text
        if text[:1] in OPENING and CLOSING[OPENING.index(text[0])] not in line and len(text) > 1:
            text = text[1:]
        if text[-1:] in CLOSING and OPENING[CLOSING.index(text[-1])] not in line and len(text) > 1:
            text = text[:-1] + ("»" if line.count("«") > line.count("»") else "")
        if text != word.text:
            result.changes.append({"from": word.text, "to": text, "how": "border"})
            word.text = text
            if word.alternative is not None:
                word.alternative = text
RULE_TOKENS = set("|=¦~_")


def _garbled(text: str) -> bool:
    core = _split(text)[1]
    return any(character in GARBAGE_SIGNS for character in core) and any(character.isalnum() for character in core)


def _initial(text: str) -> bool:
    """"Б." — an initial of a name."""
    lead, core, tail = _split(text)
    return len(core) == 1 and core.isupper() and tail.startswith(".")


def in_lookalikes(text: str, line_script: str | None) -> str | None:
    """The word moved to the alphabet of its line when it is written in lookalikes of the other one
    ("kb." in a Cyrillic line, "коʻсһаѕі" in a Latin one); None otherwise."""
    table = _TO_CYRILLIC if line_script == "Cyrillic" else _TO_LATIN if line_script == "Latin" else None
    core = _split(text)[1]
    if table is None or _script(core) in (None, line_script) or len(letter_scripts(core)) > 1:
        return None
    other = _script(core)
    if not all(character in table or _script(character) != other for character in core):
        return None
    return "".join(table.get(character, character) for character in text)


def line_script(words: list[Word]) -> str | None:
    return _dominant(words)


def _own_alphabet(text: str, line_script: str | None, lexicon: Lexicon) -> str | None:
    """It looks the same; in Word it must be the same letters too."""
    if len(_split(text)[1]) >= 4 and lexicon.spelled_any(_split(text)[1]):
        return None  # a real word of the other language ("Excel" in a Russian letter)
    return in_lookalikes(text, line_script)


def _merge_halves(word: Word, candidates: list[Word], result: Restoration) -> None:
    """"Му" + "сина", split by a signature stroke (the halves may even get different lines), where the other
    reading saw "Мусина" whole."""
    other = _split(word.alternative or "")[1]
    core = _split(word.text)[1]
    if len(other) < 4 or not core:
        return
    h = word.box.height
    for neighbour in candidates:
        if neighbour is word or not neighbour.text:
            continue
        overlap = min(word.box.y1, neighbour.box.y1) - max(word.box.y0, neighbour.box.y0)
        if overlap < 0.5 * min(h, neighbour.box.height):
            continue
        before = 0 <= word.box.x0 - neighbour.box.x1 <= 0.6 * h or -0.2 * h <= word.box.x0 - neighbour.box.x1 < 0
        after = 0 <= neighbour.box.x0 - word.box.x1 <= 0.6 * h or -0.2 * h <= neighbour.box.x0 - word.box.x1 < 0
        piece = _split(neighbour.text)[1]
        for adjacent, joined in ((before, piece + core), (after, core + piece)):
            if adjacent and joined.lower() == other.lower():
                first, last = (neighbour, word) if joined.startswith(piece) and before else (word, neighbour)
                text = _split(first.text)[0] + other + _split(last.text)[2]
                result.changes.append({"from": f"{first.text} {last.text}", "to": text, "how": "joined halves"})
                word.text = word.alternative = text
                word.box = type(word.box)(min(word.box.x0, neighbour.box.x0), min(word.box.y0, neighbour.box.y0),
                                          max(word.box.x1, neighbour.box.x1), max(word.box.y1, neighbour.box.y1))
                neighbour.text = neighbour.alternative = ""
                return


def restore_page(words: list[Word], doubtful: list[Word], lexicon: Lexicon | None, languages: set[str], *,
                 page_height: float) -> Restoration:
    result = Restoration(words)
    if lexicon is None:
        return result
    restorer = Restorer(lexicon, languages)
    doubtful_ids = {id(word) for word in doubtful}
    vocabulary: dict[str, int] = {}
    for word in words:
        if id(word) not in doubtful_ids and word.source_region != "blank_field":
            core = _split(word.text)[1].lower()
            if len(core) >= 3 and not any(character.isdigit() for character in core):
                vocabulary[core] = vocabulary.get(core, 0) + 1

    lines: dict[tuple[int, ...], list[Word]] = {}
    for word in words:
        if word.source_region != "blank_field" and any(character.isalnum() for character in word.text):
            lines.setdefault(word.line_id, []).append(word)
    heights = [word.box.height for word in words
               if word.confidence >= 0.9 and sum(character.isalpha() for character in word.text) >= 3]
    body_height = median(heights) if len(heights) >= 5 else 0.0
    page_script = _dominant(words)

    for word in words:
        if word.text and word.source_region != "blank_field" and set(word.text.strip()) <= {"|", "¦"}:
            # A vertical rule read as a sign.
            result.changes.append({"from": word.text, "to": "", "how": "removed"})
            word.text = word.alternative = ""
    for members in lines.values():
        _stray_brackets(sorted(members, key=lambda word: word.box.x0), result)
    for word in doubtful:
        if word.text and word.source_region != "blank_field" and set(word.text.strip()) <= RULE_TOKENS:
            # "|", "=": a ruling line or a speck read as a sign.
            result.changes.append({"from": word.text, "to": "", "how": "removed"})
            word.text = word.alternative = ""
    for word in doubtful:
        if word.text:
            _merge_halves(word, doubtful, result)
    unrestored: set[int] = set()
    for members in lines.values():
        members.sort(key=lambda word: word.box.x0)
        line_script = _dominant([word for word in members if id(word) not in doubtful_ids]) or page_script
        for index, word in enumerate(members):
            if not word.text:
                continue
            if id(word) not in doubtful_ids:
                converted = _own_alphabet(word.text, line_script, lexicon)
                if converted:
                    result.changes.append({"from": word.text, "to": converted, "how": "alphabet"})
                    word.text = word.alternative = converted
                continue
            abbreviation = _abbreviation_after_name(word, members[index - 1] if index else None, languages)
            if abbreviation:
                result.changes.append({"from": word.text, "to": abbreviation, "how": "context"})
                word.text = word.alternative = abbreviation
                continue
            core = _split(word.text)[1]
            # Names, initials, proper nouns, codes: never replaced from a dictionary. A reading in lookalikes of the
            # other alphabet ("ВОДОСТОКоОВ" | "BOJOCTOKOB" for "водостоков") says nothing about capitals.
            other = _split(word.alternative or "")[1]
            proper = core[:1].isupper() and _script(core) in (None, line_script) and _script(other) in (None, line_script)
            restored = restorer.restore(word, vocabulary, line_script, proper)
            if restored is not None:
                text, how = restored
                if text:
                    result.changes.append({"from": word.text, "to": text, "how": how})
                    word.text = word.alternative = text
                continue
            core = _split(word.text)[1]
            name_like = len(core) >= 3 and core[:1].isupper() and len(letter_scripts(core)) == 1 and word.confidence >= 0.6
            if any(character.isalpha() for character in core) and not any(character.isdigit() for character in core) and not name_like:
                if (len(core) <= 2 and word.confidence < SPECK_CONFIDENCE and not _initial(word.text)
                        and _script(core) != line_script):
                    # One or two weak letters that are no word: a stroke of a signature or a speck read as text.
                    result.changes.append({"from": word.text, "to": "", "how": "removed"})
                    word.text = word.alternative = ""
                    continue
                unrestored.add(id(word))  # neither a word nor a plausible name
            elif _garbled(word.text):
                unrestored.add(id(word))  # "46>Ао", "&#E": signs no document text is made of

    dropped: set[int] = {id(word) for word in words if word.text == ""}
    for members in lines.values():
        members = [word for word in members if word.text]
        if not members:
            continue
        sized = [word.box.height for word in members if sum(character.isalpha() for character in word.text) >= 3] \
            or [word.box.height for word in members]
        small = bool(body_height) and median(sized) < SMALL_PRINT * body_height
        doubtful_count = sum(1 for word in members if id(word) in doubtful_ids)
        top, bottom = min(word.box.y0 for word in members), max(word.box.y1 for word in members)
        edge = bottom < EDGE_ZONE[0] * page_height or top > EDGE_ZONE[1] * page_height
        lost = sum(1 for word in members if id(word) in unrestored)
        service = edge and (lost >= JUNK_LINE_SHARE * len(members)
                            or (len(members) >= SERVICE_LINE_WORDS and doubtful_count >= JUNK_LINE_SHARE * len(members)))
        if (small and doubtful_count >= JUNK_LINE_SHARE * len(members)) or service:
            # Small print or a service line at the edge (letterhead requisites, a footer) that cannot be read:
            # removed, not guessed.
            dropped.update(id(word) for word in members)
            result.dropped_lines.append(" ".join(word.text for word in members)[:160])
            continue
        result.body_tokens += len(members)
        result.lost_tokens += sum(1 for word in members if id(word) in unrestored)
        result.garbled.extend(word.text for word in members if _garbled(word.text))
    result.words = [word for word in words if id(word) not in dropped]
    return result
