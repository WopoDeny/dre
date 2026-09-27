"""Is the recognized text real text? Dictionary and character checks before a DOCX is issued.

Every OCR word gets a verdict:
  ok          dictionary word, number, e-mail/URL, punctuation;
  unverified  not in a dictionary but plausible: a name or abbreviation, or a
              word with Kazakh/Kyrgyz/Uzbek/Tajik/Turkmen letters (their
              dictionaries miss many inflected forms), read with high confidence;
  error       anything else (weak non-word, mixed alphabets, junk symbols, an
              unknown plain Russian/English word: Hunspell knows their forms).

An isolated run of at most two errors apart from running text is a field that
could not be read (handwriting, a smudged number): it becomes an empty line.
Errors inside running text make the page unreadable.
"""

from __future__ import annotations

import re
from contextvars import ContextVar
from dataclasses import dataclass

from .languages import letter_scripts
from .lexicon import TOKEN, Lexicon
from .types import Word
from .vote import agrees, digits_disagree, disagrees, normalized

# Letters that exist in Kazakh, Kyrgyz, Uzbek, Tajik (Cyrillic) or Azerbaijani,
# Uzbek, Turkmen (Latin) but not in Russian or English.
NATIONAL_LETTERS = set("әғқңөұүһіўҳӣӯҷјəğıöüşçñäňýžʻ")
URL_OR_EMAIL = re.compile(r"@|https?:|www\.|\.(?:ru|kz|uz|kg|tj|tm|az|com|org|net|gov)\b", re.IGNORECASE)
# Symbols that do not occur inside words; an underscore only joins identifiers (02_native_ru).
JUNK = re.compile(r"[%^~|{}<>\\#=+*]|(?<![0-9A-Za-zА-Яа-яЁё])_|_(?![0-9A-Za-zА-Яа-яЁё])")
# Table rules and brackets read at the edge of a word.
EDGE_MARKS = "|[]{}"
# Letters that are never a word or an abbreviation on their own.
# Letters of the stage 1 languages (ru, kk, ky, uz, tg, tk, az, en). The Latin and Cyrillic models
# can output any letter of their script: "ýylyňă" (Turkmen has no ă) is a misread even if read twice.
_LATIN = "abcdefghijklmnopqrstuvwxyz" + "çəğıöşüäňýž"
_CYRILLIC = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя" + "әғқңөұүһі" + "ўҳ" + "ӣӯҷ"
ALPHABET = set(_LATIN + _LATIN.upper() + "İ" + _CYRILLIC + _CYRILLIC.upper() + "ʻʼ")
# Surnames and patronymics are one letter from dictionary forms ("Раскулов" ~ "Расулов"): their endings.
SURNAME = re.compile(r"(ов|ев|ёв|ин|ова|ева|ёва|ина|ову|еву|ину|овой|евой|иной|овым|евым|иным|ович|евич|овича|евича|овичу|евичу"
                     r"|овна|евна|ична|овны|евны|кызы|қызы|улы|ұлы|uly|qizi|ov|ev|ova|eva)$")
# The dictionary check of words read identically twice needs a dictionary that covers the page.
NEAR_KNOWN_MIN_COVERAGE = 0.6
PAGE_COVERAGE: ContextVar[float] = ContextVar("PAGE_COVERAGE", default=1.0)
NEVER_ALONE = set("ыьъЫЬЪзЗ")  # з/З alone is a misread digit 3
FIELD_RUN = 2
# Languages of the page being checked (the word lists holding most of its words).
PAGE_LANGUAGES: ContextVar[set[str]] = ContextVar("PAGE_LANGUAGES", default=set())
KNOWN_MIN_CONFIDENCE = 0.5
# Words next to an empty field weaker than this belong to the handwritten insert.
FIELD_INSERT_CONFIDENCE = 0.7


@dataclass
class TextCheck:
    words: list[Word]
    tokens: int
    errors: int
    unverified: int
    blank_fields: int
    error_words: list[str]
    error_objects: list[Word]

    @property
    def error_share(self) -> float:
        return self.errors / max(1, self.tokens)

    @property
    def unverified_share(self) -> float:
        return self.unverified / max(1, self.tokens)


def _token_verdict(token: str, confidence: float, lexicon: Lexicon, national_page: bool, confirmed: bool = False, abbreviation: bool = False,
                   *, agreed: bool = False, disagreed: bool = False) -> str:
    if len(letter_scripts(token)) > 1:
        return "error"
    if any(character.isalpha() and character not in ALPHABET for character in token):
        return "error"  # a letter no supported language has
    if disagreed:
        return "error"  # two readings differ: only a third, matching reading can decide
    if len(token) == 1:
        # One letter: a word ("о", "в", "a"), a unit ("м") or, with a period, an
        # abbreviation ("г.", "ж.", "й."). Letters that never stand alone mean a split word.
        if token in NEVER_ALONE:
            return "error"
        if token.isupper() and abbreviation:
            return "ok" if agreed else "error"  # an initial ("Ж."): two readings must see the same letter
        return "ok" if agreed or confidence >= (0.8 if abbreviation else 0.9) else "error"
    page_languages = PAGE_LANGUAGES.get()
    for index in range(1, len(token)):
        # A doubled letter that is not in the word ("таълимии" for "таълими"), even if read twice:
        # the form without it is a word of this page's language, the form with it is not.
        if token[index] != token[index - 1]:
            continue
        single = token[:index] + token[index + 1:]
        if not lexicon.known(single):
            continue
        if not lexicon.known(token) or (page_languages and not (lexicon.languages_of(token) & page_languages)
                                         and lexicon.languages_of(single) & page_languages):
            return "error"
    slip = getattr(lexicon, "diacritic_slip", None)
    if slip is not None and slip(token, PAGE_LANGUAGES.get()):
        return "error"  # "КАЛАСЫ" for "ҚАЛАСЫ": a Kazakh diacritic lost, even if read so twice
    if agreed:
        # Read identically twice: trusted without a dictionary, unless one edit turns it into a word of
        # this page's language: then both readings likely made the same slip ("созданин", "Статъя").
        near = getattr(lexicon, "near_known", None)
        surname = token[:1].isupper() and SURNAME.search(token.lower())
        if (near is not None and len(token) >= 4 and not surname and PAGE_COVERAGE.get() >= NEAR_KNOWN_MIN_COVERAGE
                and not lexicon.known(token) and near(token, PAGE_LANGUAGES.get())):
            return "error"
        return "ok"
    name_like = token[0].isupper() and not token.isupper()
    if confirmed and name_like and len(token) >= 3 and confidence >= 0.5 and not lexicon.known(token):
        return "unverified"  # a name read the same way twice (Т.З.Шакирову)
    if len(token) < 3:
        # Almost any pair of letters is a word somewhere; short tokens need confident OCR.
        return "ok" if confidence >= 0.9 or (confidence >= 0.8 and lexicon.known(token)) else "error"
    if lexicon.known(token):
        # Dictionaries hold rare forms and names; a word the OCR itself barely
        # recognized ("сеу" read from a handwritten date) is not trusted.
        return "ok" if confidence >= KNOWN_MIN_CONFIDENCE else "error"
    if confidence < 0.6:
        return "error"
    for index in range(1, len(token)):
        # A doubled letter that is not in the word ("таълимии" for "таълими").
        if token[index] == token[index - 1] and lexicon.known(token[:index] + token[index + 1:]):
            return "error"
    name_like = token[0].isupper() and (token[1:].islower() or token.isupper())
    if name_like and confidence >= 0.8:
        return "unverified"
    national = any(character in NATIONAL_LETTERS for character in token.lower())
    if national and confidence >= 0.85:
        return "unverified"
    # Kazakh, Kyrgyz, Turkmen word forms are often missing from dictionaries even
    # without special letters; a misread Russian word is one edit from a real one.
    if national_page and confidence >= 0.85 and not lexicon.near_russian(token):
        return "unverified"
    return "error"


def word_verdict(word: Word, lexicon: Lexicon, joined: str | None = None, national_page: bool = False, first_in_line: bool = False) -> tuple[str, int]:
    """(verdict, number of letter tokens) for one OCR word."""
    text = word.text.strip(EDGE_MARKS) or word.text
    if URL_OR_EMAIL.search(text):
        return ("error" if disagrees(word) else "ok"), 1
    letters = [character for character in text if character.isalpha()]
    if digits_disagree(word):
        return "error", max(1, len(TOKEN.findall(text)))  # a number read two ways: never guessed
    agreed = agrees(word)
    if not letters:
        # Numbers cannot be checked against a dictionary: two identical readings or confident digits.
        digits = any(character.isdigit() for character in text)
        if agreed:
            return "ok", 0
        return ("error" if (digits and word.confidence < 0.7) or (word.confidence < 0.3 and len(text) > 2) else "ok"), 0
    if JUNK.search(text):
        return "error", 1
    best = ("error", 0)
    rank = {"ok": 0, "unverified": 1, "error": 2}
    # A word broken at a line end is either one word or a hyphenated compound.
    for variant in ([joined.replace("\u00ad", ""), joined.replace("\u00ad", "-")] if joined else [text]):
        tokens = TOKEN.findall(variant)
        initials = set(re.findall(r"(?<![^\W\d_])([^\W\d_])\.", variant)) if not first_in_line else set()  # "Ә." "Т.З.Шакирову"
        confirmed = word.source_region == "ocr_confirmed"
        abbreviated = set(re.findall(r"([^\W\d_])\.", variant)) if not first_in_line else set()
        verdicts = ["ok" if len(token) == 1 and token in initials and token.isupper() and word.confidence >= 0.6 and agrees(word)
                    else _token_verdict(token, word.confidence, lexicon, national_page, confirmed, token in abbreviated,
                                        agreed=agreed, disagreed=disagrees(word))
                    for token in tokens]
        verdict = max(verdicts, key=rank.__getitem__, default="ok")
        if best[1] == 0 or rank[verdict] < rank[best[0]]:
            best = (verdict, len(tokens))
    return best


def _hyphen_joins(words: list[Word]) -> dict[int, str]:
    """A word broken at the end of a line is checked together with its continuation."""
    joins: dict[int, str] = {}
    for index in range(len(words) - 1):
        head, tail = words[index], words[index + 1]
        if head.text.endswith(("-", "‑", "¬")) and head.line_id != tail.line_id and tail.text[:1].isalpha():
            # Soft hyphen marks the "no hyphen" reading; the plain hyphen keeps a compound.
            joined = head.text.rstrip("-‑¬") + "\u00ad" + tail.text
            joins[index] = joins[index + 1] = joined
    return joins


BULLET_LOOKALIKES = set("eoc•©°*·")


def _between_other_alphabet(word: Word, words: list[Word]) -> bool:
    """A short word of one alphabet between two words of the other ("Астана k., Сарыарқа",
    "ten.: 8 (7172)" for "Тел.:") is a misread lookalike, not a foreign word."""
    own = letter_scripts(word.text)
    if len(own) != 1 or sum(own.values()) > 4 or word.text.isupper() and sum(own.values()) > 1:
        return False
    script = next(iter(own))
    line = sorted((other for other in words if other.line_id == word.line_id), key=lambda other: other.box.x0)
    position = line.index(word)
    def neighbour(step: int) -> str | None:
        # The nearest word with letters, at most two words away (numbers are skipped).
        index = position + step
        while 0 <= index < len(line) and abs(index - position) <= 2:
            scripts = letter_scripts(line[index].text)
            # A stray letter (a quote mark of an empty date field read as "L") shows no alphabet.
            if sum(scripts.values()) >= 2 and not URL_OR_EMAIL.search(line[index].text):
                return max(scripts, key=scripts.get)
            index += step
        return None
    left, right = neighbour(-1), neighbour(1)
    others = [side for side in (left, right) if side is not None]
    return bool(others) and all(side != script for side in others)


def _same_alphabet_follows(word: Word, words: list[Word]) -> bool:
    """The next word on the line is of the letter's alphabet and read the same twice ("Г. ТАШКЕНТА")."""
    own = letter_scripts(word.text)
    following = min((other for other in words if other.line_id == word.line_id and other.box.x0 >= word.box.x1 - 1 and other is not word),
                    key=lambda other: other.box.x0, default=None)
    if following is None or not agrees(following) or following.box.x0 - word.box.x1 > 2 * word.box.height:
        return False
    scripts = letter_scripts(following.text)
    return sum(scripts.values()) >= 3 and set(scripts) == set(own)


def _minority_in_mixed_line(word: Word, scripts: dict[str, int], dominant: str | None) -> bool:
    """A line that mixes alphabets on a page of one alphabet is suspect: garbage
    from a stamp or a smudge reads as short words of both alphabets. Only
    confident words of the page's minority alphabet survive in such a line."""
    own = letter_scripts(word.text)
    if len(own) != 1 or dominant is None or next(iter(own)) == dominant or word.confidence >= 0.9:
        return False
    total = sum(scripts.values())
    minority = total - scripts.get(dominant, 0)
    return total > 0 and 0.25 <= minority / total <= 0.75



def _foreign_short(word: Word, scripts: dict[str, int]) -> bool:
    """One or two letters of the other alphabet inside a line (Latin K for Kazakh қ)."""
    own = letter_scripts(word.text)
    letters = sum(own.values())
    capitals = all(character.isupper() for character in word.text if character.isalpha())
    if not 0 < letters <= (4 if capitals else 2) or len(own) != 1:
        return False
    script = next(iter(own))
    others = {name: count - own.get(name, 0) for name, count in scripts.items()}
    dominant = max(others, key=others.get, default=None)
    return dominant is not None and dominant != script and others[dominant] >= 10 and others.get(script, 0) < 3


def _bullets(words: list[Word]) -> list[Word]:
    """A list bullet read as a stray letter at the start of a line becomes a bullet."""
    first: dict[tuple[int, ...], Word] = {}
    for word in words:
        if word.source_region != "blank_field" and (word.line_id not in first or word.box.x0 < first[word.line_id].box.x0):
            first[word.line_id] = word
    result = []
    for word in words:
        if first.get(word.line_id) is word and len(word.text) == 1 and word.text in BULLET_LOOKALIKES and word.text != "•":
            following = [other for other in words if other.line_id == word.line_id and other is not word]
            if following and min(other.box.x0 for other in following) - word.box.x1 >= word.box.height * 0.5:
                result.append(Word("•", word.box, 1.0, size=word.size, line_id=word.line_id, source_page=word.source_page,
                                   source_region=word.source_region, script_evidence=word.script_evidence, language_profile=word.language_profile))
                continue
        result.append(word)
    return result


def check_page_text(words: list[Word], lexicon: Lexicon, *, make_fields: bool = True, protected=()) -> TextCheck:
    """Classify words; with `make_fields`, isolated inserts that were read unstably
    (a second reading disagreed) or are hardly legible become empty fields."""
    words = _bullets(words)
    texts = [word.text for word in words if word.source_region != "blank_field" and any(character.isalpha() for character in word.text)]
    national_page = sum(any(character in NATIONAL_LETTERS for character in text.lower()) for text in texts) >= max(3, 0.05 * len(texts))
    languages: dict[str, int] = {}
    languages_of = getattr(lexicon, "languages_of", lambda token: set())
    for text in texts[:200]:
        for token in TOKEN.findall(text):
            if len(token) >= 4:
                for language in languages_of(token):
                    languages[language] = languages.get(language, 0) + 1
    long_tokens = [token for text in texts[:200] for token in TOKEN.findall(text) if len(token) >= 4]
    PAGE_COVERAGE.set(sum(1 for token in long_tokens if lexicon.known(token)) / len(long_tokens) if long_tokens else 1.0)
    top = max(languages.values(), default=0)
    PAGE_LANGUAGES.set({language for language, count in languages.items() if count >= 0.8 * top} if top else set())
    for word in words:
        if not word.alternative or agrees(word):
            continue
        if NUMBER_SIGN.match(word.text) and word.alternative.startswith("№"):
            word.text = word.alternative = "№"  # the second reading saw the number sign itself
            continue
        # One reading lost the apostrophe of "maʼqullandi": the dictionary word of the two wins.
        plain, other = normalized(word.text), normalized(word.alternative)
        if plain.replace("ʻ", "") == other.replace("ʻ", "") and plain != other:
            spelled = getattr(lexicon, "spelled_any", lexicon.known)
            known = [text for text in (word.text, word.alternative) if spelled(text.strip(EDGE_MARKS + ".,;:!?«»\"()"))]
            if len(known) == 1:
                word.text = word.alternative = known[0]
    agreed_before = {id(word) for word in words if agrees(word)}
    normalize_lookalikes(words, lexicon, national_page)
    for word in words:
        if id(word) in agreed_before:
            word.alternative = word.text  # both readings had the same lookalike
    candidates = [word for word in words if word.source_region != "blank_field"]
    joins = _hyphen_joins(candidates)
    lettered = [word.text for word in candidates if any(character.isalpha() for character in word.text)]
    national_page = sum(any(character in NATIONAL_LETTERS for character in text.lower()) for text in lettered) >= max(3, 0.05 * len(lettered))
    line_scripts: dict[tuple[int, ...], dict[str, int]] = {}
    for word in candidates:
        for script, count in letter_scripts(word.text).items():
            line_scripts.setdefault(word.line_id, {}).setdefault(script, 0)
            line_scripts[word.line_id][script] += count
    verdicts: dict[int, str] = {}
    tokens = unverified = 0
    page_scripts: dict[str, int] = {}
    for counts in line_scripts.values():
        for script, count in counts.items():
            page_scripts[script] = page_scripts.get(script, 0) + count
    dominant = max(page_scripts, key=page_scripts.get, default=None)
    line_starts = {}
    for word in candidates:
        if word.line_id not in line_starts or word.box.x0 < line_starts[word.line_id].box.x0:
            line_starts[word.line_id] = word
    for index, word in enumerate(candidates):
        # A letter with a period at the start of a line may be a list number ("З." for "3.").
        verdict, count = word_verdict(word, lexicon, joins.get(index), national_page, line_starts.get(word.line_id) is word)
        if verdict != "error" and _foreign_short(word, line_scripts.get(word.line_id, {})):
            verdict = "error"
        if URL_OR_EMAIL.search(word.text):
            pass
        elif verdict != "error" and _minority_in_mixed_line(word, line_scripts.get(word.line_id, {}), dominant):
            verdict = "error"
        if verdict != "error" and not URL_OR_EMAIL.search(word.text) and _between_other_alphabet(word, candidates):
            verdict = "error"
        own = letter_scripts(word.text)
        total = sum(page_scripts.values())
        if (verdict != "error" and sum(own.values()) == 1 and dominant is not None and dominant not in own
                and total and page_scripts.get(dominant, 0) / total >= 0.8 and not _same_alphabet_follows(word, candidates)):
            verdict = "error"  # one letter of the other alphabet ("x." for "ж.")
        verdicts[id(word)] = verdict
        tokens += count
        unverified += verdict == "unverified"

    # A weak word right next to an empty field is part of the handwritten insert.
    fields = [word for word in words if word.source_region == "blank_field"]
    field_inserts: set[int] = set()
    printed = sorted(word.box.height for word in candidates if word.confidence >= 0.9 and sum(character.isalpha() for character in word.text) >= 3)
    typical_height = printed[len(printed) // 2] if printed else 0.0

    def printed_word(word: Word) -> bool:
        """A confident dictionary word is print, never an empty field ("работах." alone on a wrapped line)."""
        if word.confidence < 0.9:
            return False
        for text in (word.text, word.alternative or ""):
            core = text.strip(EDGE_MARKS + ".,;:!?«»\"()")
            if sum(character.isalpha() for character in core) >= 3 and lexicon.known(core):
                return True
        return False

    def written_by_hand(word: Word) -> bool:
        """Handwriting is taller than the print; a smudged field is flatter."""
        return bool(typical_height) and (word.box.height >= 1.25 * typical_height or word.box.height <= 0.6 * typical_height)

    for word in candidates:
        # A short, weak, clearly taller word is a handwritten number or date.
        if typical_height and word.confidence < 0.9 and len(word.text) <= 4 and word.box.height >= 1.35 * typical_height:
            field_inserts.add(id(word))
            verdicts[id(word)] = "error"
    for word in candidates:
        if word.confidence >= FIELD_INSERT_CONFIDENCE:
            continue
        for field in fields:
            overlap = min(word.box.y1, field.box.y1) - max(word.box.y0, field.box.y0)
            gap = max(field.box.x0 - word.box.x1, word.box.x0 - field.box.x1, 0.0)
            if overlap > 0.3 * min(word.box.height, field.box.height) and gap <= 2 * word.box.height:
                field_inserts.add(id(word))
                verdicts[id(word)] = "error"
                break
    lines: dict[tuple[int, ...], list[Word]] = {}
    for word in candidates:
        lines.setdefault(word.line_id, []).append(word)

    def visual_line(run: list[Word]) -> list[Word]:
        """Words on the same printed line, whatever line number the OCR gave them."""
        top, bottom = min(word.box.y0 for word in run), max(word.box.y1 for word in run)
        return [word for word in candidates
                if min(bottom, word.box.y1) - max(top, word.box.y0) >= 0.5 * min(bottom - top, word.box.height)]
    blanked: set[int] = set()
    body_errors: list[Word] = []
    for line in lines.values():
        line.sort(key=lambda word: word.box.x0)
        position = 0
        while position < len(line):
            if verdicts[id(line[position])] != "error":
                position += 1
                continue
            end = position
            while end + 1 < len(line) and verdicts[id(line[end + 1])] == "error":
                end += 1
            run = line[position:end + 1]
            reach = max(word.box.height for word in run) * 2.5
            # A short handwritten insert next to printed digits or quotes ("« 27 » 05 2019 года")
            # is still a field: only neighbours with letters count as running text.
            short = all(len(word.text) <= 4 for word in run)
            others = [word for word in visual_line(run) if word not in run and not (short and not any(character.isalpha() for character in word.text))]
            left = max((word for word in others if word.box.x1 <= run[0].box.x0 + 1), key=lambda word: word.box.x1, default=None)
            right = min((word for word in others if word.box.x0 >= run[-1].box.x1 - 1), key=lambda word: word.box.x0, default=None)
            isolated = (left is None or run[0].box.x0 - left.box.x1 > reach) and (right is None or right.box.x0 - run[-1].box.x1 > reach)
            unreadable = all(
                id(word) in field_inserts or word.confidence < 0.3
                or (word.source_region == "ocr_unstable" and (written_by_hand(word) or not typical_height))
                for word in run
            ) and not any(printed_word(word) for word in run) and not any(
                box.contains_center(word.box) for word in run for box in protected  # text under a stamp: read or refuse
            ) and not any(word.source_region == "ocr_region" or (word.line_id[:1] and 8000 <= word.line_id[0] < 9000)
                          for word in run)  # print the first reading skipped ("001705"), even after a reread
            if make_fields and isolated and len(run) <= FIELD_RUN and unreadable:
                blanked.update(id(word) for word in run)
            else:
                body_errors.extend(run)
            position = end + 1

    result = []
    for word in words:
        if id(word) in blanked:
            length = max(3, int(round(word.box.width / max(1.0, word.box.height * 0.5))))
            result.append(Word("_" * length, word.box, 1.0, size=word.size, line_id=word.line_id,
                               source_page=word.source_page, source_region="blank_field"))
        else:
            result.append(word)
    return TextCheck(result, tokens, len(body_errors), unverified, len(blanked), [f"{word.text}|{word.alternative}" if word.alternative and word.alternative != word.text else word.text for word in body_errors], body_errors)


# Identical glyphs in Latin and Cyrillic. A word written in one alphabet with a
# few lookalikes of the other is normalized to its own alphabet. Single letters
# are normalized only when the glyph has no Kazakh sibling with a diacritic
# (K could be Қ, O could be Ө, so they stay as they are and fail the check).
_LATIN_TO_CYRILLIC = dict(zip("ABCEHKMOPTXYaceopxyƏəIi", "АВСЕНКМОРТХУасеорхуӘәІі"))
_CYRILLIC_TO_LATIN = {cyrillic: latin for latin, cyrillic in _LATIN_TO_CYRILLIC.items()}
# Cyrillic letters that only look Latin ("Ко“сһаѕі" for "Koʻchasi"): converted inside Latin lines only.
_CYRILLIC_TO_LATIN.update(zip("ѕһјԛԝӏЅҺЈԚԜ", "shjqwlSHJQW"))
_SAFE_SINGLE = set("Əə")


NUMBER_SIGN = re.compile(r"^(?:N[eoº°]|No\.?|N)$")
SAFE_SINGLE_ANYWHERE = dict(zip("ABCEMPTacep", "АВСЕМРТасер"))  # not x: OCR reads ж as x
SAFE_SINGLE_RUSSIAN = dict(zip("KOHoy", "КОНоу"))
SAFE_SINGLE_LATIN = dict(zip("АВСЕНКМОРТХасеорху", "ABCEHKMOPTXaceopxy"))


# Option labels of Cyrillic lists and tests ("а) б) в) г)") and the shapes OCR gives them.
OPTION_LABEL = re.compile(r"^([^\W_]|\d)\)$")
_OPTION_LOOKALIKES = {"a": "а", "а": "а", "6": "б", "б": "б", "B": "в", "в": "в", "r": "г", "г": "г", "д": "д", "e": "е", "е": "е"}


def _cyrillic_option_labels(ordered: list[Word]) -> int:
    """In a line with at least two option labels of which one is a letter, the labels are the Cyrillic
    letters they look like ("a) … 6) … B) … r)" → "а) … б) … в) … г)"): a lone "6)" is not a digit there."""
    lines: dict[tuple[int, ...], list[Word]] = {}
    for word in ordered:
        if OPTION_LABEL.match(word.text) or OPTION_LABEL.match(word.alternative or ""):
            lines.setdefault(word.line_id, []).append(word)
    changed = 0
    for labels in lines.values():
        letters = [word for word in labels if word.text[0].isalpha() and word.text[0] in _OPTION_LOOKALIKES]
        if len(labels) < 2 or not letters:
            continue
        for word in labels:
            for attribute in ("text", "alternative"):
                value = getattr(word, attribute) or ""
                if OPTION_LABEL.match(value) and value[0] in _OPTION_LOOKALIKES and _OPTION_LOOKALIKES[value[0]] + ")" != value:
                    setattr(word, attribute, _OPTION_LOOKALIKES[value[0]] + ")")
                    changed += attribute == "text"
    return changed


def normalize_lookalikes(words: list[Word], lexicon: Lexicon | None = None, national_page: bool = False) -> int:
    """Fix typical OCR lookalikes in place; returns the number of changed words."""
    changed = 0
    page_scripts: dict[str, int] = {}
    for word in words:
        for script, count in letter_scripts(word.text).items():
            page_scripts[script] = page_scripts.get(script, 0) + count
    ordered = sorted(words, key=lambda word: (word.line_id, word.box.x0))
    if page_scripts.get("Cyrillic", 0) > 3 * page_scripts.get("Latin", 0):
        changed += _cyrillic_option_labels(ordered)
    for word, following in zip(ordered, ordered[1:] + [None]):
        # "1800 000" for "1 800 000": the space between digit groups was lost in the first group.
        if (following is not None and following.line_id == word.line_id and re.fullmatch(r"\d{4,6}", word.text)
                and re.fullmatch(r"\d{3}[.,;]?", following.text) and following.box.x0 - word.box.x1 < 0.8 * word.box.height):
            head, tail = word.text[:-3], word.text[-3:]
            word.text = f"{head} {tail}"
            if word.alternative and re.fullmatch(r"\d{4,6}", word.alternative):
                word.alternative = f"{word.alternative[:-3]} {word.alternative[-3:]}"
            changed += 1
        # "Ne 12", "No ____" for "№": the number sign read as letters.
        at_line_end = following is None or following.line_id != word.line_id
        if NUMBER_SIGN.match(word.text) and (at_line_end or following.text[:1].isdigit()
                                             or following.source_region == "blank_field" or following.text.startswith("_")):
            word.text = "№"
            changed += 1
        # "Ten.: 8 (7172)" for "Тел.:": л read as n before a phone number.
        phone = following is not None and following.line_id == word.line_id and re.match(r"[+(\d]", following.text)
        if phone and re.fullmatch(r"[Tt]en\.?:?", word.text):
            word.text = word.text.replace("T", "Т").replace("t", "т").replace("e", "е").replace("n", "л")
            changed += 1
        apostrophes = re.sub(r"[ʻ‘’'`]{2,}", "ʻ", word.text)
        apostrophes = re.sub(r"(?<=[^\W\d_])[“”](?=[^\W\d_])", "ʻ", apostrophes)  # "Ko“chasi": a quote inside a word
        if apostrophes != word.text:
            word.text = apostrophes
            changed += 1
    line_scripts: dict[tuple[int, ...], dict[str, int]] = {}
    for word in words:
        for script, count in letter_scripts(word.text).items():
            line_scripts.setdefault(word.line_id, {}).setdefault(script, 0)
            line_scripts[word.line_id][script] += count
    for word in words:
        scripts = letter_scripts(word.text)
        line = line_scripts.get(word.line_id, {})
        others = {name: count - scripts.get(name, 0) for name, count in line.items()}
        if sum(others.values()) < 3:
            others = page_scripts  # a short line ("«__» ____ 2024 c.") takes the page's alphabet
        cyrillic_line = others.get("Cyrillic", 0) >= 3 * max(1, others.get("Latin", 0))
        # "2024r." for "2024г." in a Cyrillic line.
        if cyrillic_line and re.fullmatch(r"\d+r\.?[,;]?", word.text):
            word.text = word.text.replace("r", " г")
            changed += 1
            continue
        # "r." alone for "г." (год, город) in a Cyrillic line.
        if cyrillic_line and re.fullmatch(r"r\.[,;]?", word.text):
            word.text = word.text.replace("r", "г")
            changed += 1
            continue
        glued = re.fullmatch(r"((?:19|20)\d\d)([^\W\d_]+\.?[,;:]?)", word.text)
        if glued:
            # A year glued to the next word ("2024й." for "2024 й.", "2024il" for "2024 il", "2019года").
            word.text = f"{glued.group(1)} {glued.group(2)}"
            changed += 1
            continue
        letters = [character for character in word.text if character.isalpha()]
        latin_line = others.get("Latin", 0) >= 3 * max(1, others.get("Cyrillic", 0))
        if latin_line and scripts.get("Cyrillic") == 1 and len(letters) == 1 and letters[0] in SAFE_SINGLE_LATIN:
            # "2024 у." for "2024 y." in a Latin line.
            word.text = word.text.replace(letters[0], SAFE_SINGLE_LATIN[letters[0]])
            changed += 1
            continue
        if cyrillic_line and scripts.get("Latin") == 1 and len(letters) == 1:
            table = dict(SAFE_SINGLE_ANYWHERE, **({} if national_page else SAFE_SINGLE_RUSSIAN))
            if letters[0] in table:
                word.text = word.text.replace(letters[0], table[letters[0]])
                changed += 1
                continue
        if lexicon is not None and len(scripts) == 1 and sum(scripts.values()) >= 2:
            # A whole word of lookalikes in a line of the other alphabet ("ҚАРАҒАНДЫ CY").
            own = next(iter(scripts))
            line = line_scripts.get(word.line_id, {})
            other = "Cyrillic" if own == "Latin" else "Latin"
            table = _LATIN_TO_CYRILLIC if own == "Latin" else _CYRILLIC_TO_LATIN
            letters = [character for character in word.text if character.isalpha()]
            if line.get(other, 0) >= 3 * max(1, line.get(own, 0) - scripts[own]) and all(character in table for character in letters):
                fixed = "".join(table.get(character, character) for character in word.text)
                alone = line.get(own, 0) == scripts[own] and line.get(other, 0) >= 10
                original_known = lexicon.known(word.text.strip(".,;:!?«»\"'()"))
                if fixed != word.text and (alone or not original_known) and lexicon.known(fixed.strip(".,;:!?«»\"'()")):
                    word.text = fixed
                    changed += 1
                    continue
        scripts = letter_scripts(word.text)
        if set(scripts) != {"Latin", "Cyrillic"} and not (set(word.text) & _SAFE_SINGLE):
            continue
        if set(word.text) & _SAFE_SINGLE and "Latin" in scripts and scripts.get("Latin", 0) == sum(1 for c in word.text if c in _SAFE_SINGLE):
            target, table = "Cyrillic", _LATIN_TO_CYRILLIC
        else:
            target = max(scripts, key=scripts.get)
            if scripts[target] == scripts.get("Cyrillic" if target == "Latin" else "Latin", 0):
                continue
            table = _LATIN_TO_CYRILLIC if target == "Cyrillic" else _CYRILLIC_TO_LATIN
        fixed = "".join(table.get(character, character) for character in word.text)
        if fixed != word.text and len(letter_scripts(fixed)) == 1:
            word.text = fixed
            changed += 1
    if "uzb" in PAGE_LANGUAGES.get():
        # Uzbek Latin: oʻ and gʻ take the turned comma (U+02BB), the tutuq belgisi the apostrophe (U+02BC).
        for word in words:
            if not letter_scripts(word.text).get("Latin"):
                continue
            fixed = re.sub(r"(?<=[A-Za-z])[ʻʼ‘’'`](?=[A-Za-z])",
                           lambda match: "ʻ" if match.string[match.start() - 1] in "oOgG" else "ʼ", word.text)
            if fixed != word.text:
                word.text = fixed
                changed += 1
    return changed
