"""Word lists of the supported languages.

1. Tesseract models: each traineddata carries a word list (lstm-word-dawg). It is
   unpacked once with `combine_tessdata` + `dawg2wordlist` into sorted 64-bit
   hashes (all languages ~15 MB).
2. Hunspell dictionaries (ru, kk, uz, en; system packages hunspell-*), checked
   with spylls: they know inflected forms, which matters for Russian.
Nothing is downloaded at run time.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
import unicodedata
from pathlib import Path

import numpy as np

LANGUAGES = ("rus", "kaz", "kir", "uzb", "uzb_cyrl", "tgk", "aze", "eng")
HUNSPELL = ("ru_RU", "kk_KZ", "uz_UZ", "en_US")
TOKEN = re.compile(r"[^\W\d_]+(?:[’'ʻʼ`-][^\W\d_]+)*")
_APOSTROPHES = str.maketrans({"’": "'", "ʻ": "'", "ʼ": "'", "`": "'", "‘": "'"})


def normalize(word: str) -> str:
    return re.sub(r"'{2,}", "'", unicodedata.normalize("NFC", word).casefold().translate(_APOSTROPHES))


def _hash(word: str) -> int:
    return int.from_bytes(hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest(), "little")


def _contains(table: np.ndarray, word: str) -> bool:
    if not table.size:
        return False
    key = np.uint64(_hash(normalize(word)))
    index = int(np.searchsorted(table, key))
    return index < table.size and table[index] == key


def default_directory() -> Path:
    return Path(os.environ.get("DRE_LEXICON_DIR") or Path.home() / ".cache" / "document_reconstruction" / "lexicon")


def build(tessdata: Path, directory: Path, languages: tuple[str, ...] = LANGUAGES) -> list[str]:
    """Unpack word lists of installed models into `directory` (run at deployment)."""
    directory.mkdir(parents=True, exist_ok=True)
    built = []
    for language in languages:
        model = tessdata / f"{language}.traineddata"
        if not model.exists():
            continue
        with tempfile.TemporaryDirectory() as work:
            prefix = str(Path(work) / language) + "."
            subprocess.run(["combine_tessdata", "-u", str(model), prefix], check=True, capture_output=True)
            dawg, unicharset = Path(prefix + "lstm-word-dawg"), Path(prefix + "lstm-unicharset")
            if not dawg.exists():
                continue
            listing = Path(work) / "words.txt"
            subprocess.run(["dawg2wordlist", str(unicharset), str(dawg), str(listing)], check=True, capture_output=True)
            words = {normalize(line.strip()) for line in listing.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()}
        hashes = np.unique(np.fromiter((_hash(word) for word in words), dtype=np.uint64, count=len(words)))
        np.save(directory / f"{language}.npy", hashes)
        built.append(language)
    return built


# Letters Tesseract confuses (either way): shapes that differ by a stroke, a tail or a diacritic.
_LOOKALIKE_PAIRS = ("ин", "ип", "нп", "ий", "ьъ", "ьы", "ес", "шщ", "цщ", "зэ", "бв", "кқ", "гғ", "нң", "оө",
                    "уү", "уұ", "үұ", "аә", "хһ", "іи", "ўу", "ҳх", "ӣи", "ӯу", "ҷч", "лп", "тг",
                    "sş", "cç", "oö", "uü", "iı", "gğ", "eə", "aä", "nň", "yý", "zž", "il", "ce")
LOOKALIKE_LETTERS: dict[str, str] = {}
for _pair in _LOOKALIKE_PAIRS:
    LOOKALIKE_LETTERS[_pair[0]] = LOOKALIKE_LETTERS.get(_pair[0], "") + _pair[1]
    LOOKALIKE_LETTERS[_pair[1]] = LOOKALIKE_LETTERS.get(_pair[1], "") + _pair[0]

_KAZAKH_DIACRITICS = {"к": "қ", "г": "ғ", "н": "ң", "о": "ө", "у": "үұ", "а": "ә", "и": "і", "х": "һ"}
_UZBEK_DIACRITICS = {"к": "қ", "г": "ғ", "у": "ў", "х": "ҳ"}

HUNSPELL_DIRECTORY = Path("/usr/share/hunspell")
# Uzbek Latin -> Cyrillic, to check Uzbek Latin words with the (Cyrillic) uz_UZ Hunspell dictionary.
# Digraphs first; a few spellings are ambiguous and give several candidates.
_UZ_DIGRAPHS = (("oʻ", ("ў",)), ("gʻ", ("ғ",)), ("sh", ("ш",)), ("ch", ("ч",)), ("yo", ("ё", "йо")),
                ("yu", ("ю", "йу")), ("ya", ("я", "йа")), ("ye", ("е", "йе")), ("ts", ("ц", "тс")))
_UZ_LETTERS = dict(zip("abdefghijklmnopqrstuvxyzʼ", "абдефгҳийклмнопқрстувхйзъ"))
_UZ_APOSTROPHES = str.maketrans({"'": "ʻ", "‘": "ʻ", "’": "ʻ", "`": "ʻ", "ʼ": "ʻ"})


def uzbek_cyrillic(word: str) -> list[str]:
    """Cyrillic spellings of an Uzbek Latin word (empty if the word is not Uzbek Latin)."""
    lower = word.lower().translate(_UZ_APOSTROPHES)
    if not lower or any(not ("a" <= character <= "z" or character == "ʻ") for character in lower):
        return []
    variants = [""]
    index = 0
    while index < len(lower):
        pair = lower[index:index + 2]
        options = next((cyrillic for latin, cyrillic in _UZ_DIGRAPHS if latin == pair), None)
        if options:
            index += 2
        elif lower[index] == "ʻ":
            options, index = ("ъ",), index + 1  # the tutuq belgisi between letters
        elif lower[index] in _UZ_LETTERS:
            letter = _UZ_LETTERS[lower[index]]
            # "e" opens a word as "э"; "c" and "w" do not occur in Uzbek.
            options, index = (("э",) if letter == "е" and index == 0 else (letter,)), index + 1
        else:
            return []
        variants = [variant + option for variant in variants for option in options][:16]
    return variants


class Lexicon:
    def __init__(self, directory: Path | None = None, hunspell: Path | None = None) -> None:
        self.directory = directory or default_directory()
        tables = [np.load(path) for path in sorted(self.directory.glob("*.npy"))]
        self.languages = [path.stem for path in sorted(self.directory.glob("*.npy"))]
        self._hashes = np.unique(np.concatenate(tables)) if tables else np.zeros(0, dtype=np.uint64)
        self._by_language = {path.stem: np.load(path) for path in sorted(self.directory.glob("*.npy"))}
        self._russian = self._by_language.get("rus", np.zeros(0, dtype=np.uint64))
        self.hunspell: dict[str, object] = {}
        folder = hunspell or HUNSPELL_DIRECTORY
        try:
            from spylls.hunspell import Dictionary
        except ImportError:
            Dictionary = None
        if Dictionary is not None:
            for name in HUNSPELL:
                if (folder / f"{name}.dic").exists() and (folder / f"{name}.aff").exists():
                    self.hunspell[name] = Dictionary.from_files(str(folder / name))
        self._cache: dict[str, bool] = {}
        self._languages_cache: dict[str, frozenset[str]] = {}
        self._slip_cache: dict[tuple, bool] = {}
        self._spelled_cache: dict[tuple[str, str], bool] = {}

    def spelled(self, token: str, language: str) -> bool:
        dictionary = self.hunspell.get(language)
        if dictionary is None:
            return False
        key = (token, language)
        cached = self._spelled_cache.get(key)
        if cached is None:
            forms = dict.fromkeys((token, token.lower(), token[:1].upper() + token[1:].lower()))
            cached = any(dictionary.lookup(form) for form in forms)
            if len(self._spelled_cache) < 500_000:
                self._spelled_cache[key] = cached  # Hunspell lookups are slow; restoration asks for the same forms often
        return cached

    @classmethod
    def ensure(cls, tessdata: str | None) -> "Lexicon":
        directory = default_directory()
        if not any(directory.glob("*.npy")) and isinstance(tessdata, str):
            build(Path(tessdata), directory)
        return cls(directory)

    def __bool__(self) -> bool:
        return bool(self._hashes.size)

    def __contains__(self, word: str) -> bool:
        key = np.uint64(_hash(normalize(word)))
        index = int(np.searchsorted(self._hashes, key))
        return index < self._hashes.size and self._hashes[index] == key

    def languages_of(self, token: str) -> set[str]:
        """Languages whose word list holds the token (Tesseract lists, Hunspell; Uzbek Latin via Cyrillic)."""
        cached = self._languages_cache.get(token)
        if cached is None:
            cached = frozenset(self._languages_of(token))
            if len(self._languages_cache) < 200_000:
                self._languages_cache[token] = cached
        return set(cached)

    def _languages_of(self, token: str) -> set[str]:
        key = np.uint64(_hash(normalize(token)))
        found = set()
        for language, table in self._by_language.items():
            index = int(np.searchsorted(table, key))
            if index < table.size and table[index] == key:
                found.add(language)
        for name, code in (("ru_RU", "rus"), ("kk_KZ", "kaz"), ("uz_UZ", "uzb"), ("en_US", "eng")):
            if code not in found and self.spelled(token, name):
                found.add(code)
        if "uzb" not in found and any(self.spelled(variant, "uz_UZ") for variant in uzbek_cyrillic(token)):
            found.add("uzb")
        return found

    def spelled_any(self, token: str) -> bool:
        """In a Hunspell dictionary (stricter than the Tesseract lists, which hold common typos)."""
        return any(self.spelled(token, name) for name in self.hunspell) or \
            ("uz_UZ" in self.hunspell and any(self.spelled(variant, "uz_UZ") for variant in uzbek_cyrillic(token)))

    def near_known(self, token: str, languages: set[str] | None = None) -> bool:
        """True if one typical OCR slip, undone, gives a word of these languages (all, when none are given):
        a word both readings got wrong the same way. Slips: a lookalike letter ("созданин" → "создании",
        "Статъя" → "Статья", "бұйрыкпен" → "бұйрықпен"), a stray letter in front ("горганизации"), a letter
        lost inside the word ("Республикаының" → "Республикасының"). Endings are not touched: a word one
        ending away from a dictionary form is usually just another form ("Ерлану", "müdirine")."""
        word = normalize(token)
        if len(word) < 4:
            return False
        tables = [table for name, table in self._by_language.items() if not languages or name in languages] or [self._hashes]
        cyrillic = sum("\u0400" <= character <= "\u04ff" for character in word)
        alphabet = ("абвгдежзийклмнопрстуфхцчшщъыьэюяәғқңөұүһіўҳӣӯҷ" if cyrillic * 2 >= len(word)
                    else "abcdefghijklmnopqrstuvwxyzçəğıöşüäňýž")
        variants = {word[:i] + other + word[i + 1:] for i, letter in enumerate(word) for other in LOOKALIKE_LETTERS.get(letter, "")}
        variants.add(word[1:])  # a stray letter in front
        variants |= {word[:i] + letter + word[i:] for i in range(2, len(word)) for letter in alphabet}  # lost inside
        variants.discard(word)
        keys = np.fromiter((_hash(variant) for variant in variants if len(variant) >= 3), dtype=np.uint64)
        for table in tables:
            if table.size:
                positions = np.searchsorted(table, keys).clip(0, table.size - 1)
                if (table[positions] == keys).any():
                    return True
        return False

    def diacritic_slip(self, token: str, languages: set[str]) -> bool:
        """A Kazakh or Uzbek Cyrillic word read without its diacritic ("КАЛАСЫ" for "ҚАЛАСЫ"): the web word
        lists hold such misspellings, Hunspell does not. True if the token is not a Hunspell word of the page
        language (nor Russian) but putting back one diacritic gives one."""
        checks = [(name, pairs) for code, name, pairs in (("kaz", "kk_KZ", _KAZAKH_DIACRITICS), ("uzb_cyrl", "uz_UZ", _UZBEK_DIACRITICS))
                  if code in languages and name in self.hunspell]
        word = token.lower()
        if not checks or len(word) < 3 or not all("\u0400" <= character <= "\u04ff" for character in word):
            return False
        cached = self._slip_cache.get((word, tuple(name for name, _ in checks)))
        if cached is not None:
            return cached
        result = False
        if not (self.spelled(word, "ru_RU") or _contains(self._russian, word)):
            for name, pairs in checks:
                if self.spelled(word, name):
                    break
                if any(self.spelled(word[:i] + other + word[i + 1:], name)
                       for i, letter in enumerate(word) for other in pairs.get(letter, "")):
                    result = True
                    break
        if len(self._slip_cache) < 200_000:
            self._slip_cache[(word, tuple(name for name, _ in checks))] = result
        return result

    def corrections(self, token: str, languages: set[str] | None = None, *, limit: int = 12) -> list[str]:
        """Dictionary words one typical OCR slip away from an unknown token, best first: a lookalike
        or a diacritic letter, a letter lost or added anywhere, two letters swapped. Words Hunspell
        knows come first (the Tesseract lists hold common misspellings too)."""
        word = normalize(token)
        if len(word) < 2:
            return []
        cyrillic = sum("\u0400" <= character <= "\u04ff" for character in word)
        alphabet = ("абвгдежзийклмнопрстуфхцчшщъыьэюяәғқңөұүһіўҳӣӯҷё" if cyrillic * 2 >= len(word)
                    else "abcdefghijklmnopqrstuvwxyzçəğıöşüäňýžʻ")
        ranked: list[tuple[int, str]] = []
        seen = {word}

        def consider(variant: str, cost: int) -> None:
            if variant in seen or len(variant) < 2:
                return
            seen.add(variant)
            if self._in_languages(variant, languages):
                ranked.append((cost - 2 * self.spelled_any(variant), variant))

        for i, letter in enumerate(word):
            for other in LOOKALIKE_LETTERS.get(letter, ""):
                consider(word[:i] + other + word[i + 1:], 0)
        for i in range(len(word)):
            consider(word[:i] + word[i + 1:], 1)
        for i in range(len(word) - 1):
            consider(word[:i] + word[i + 1] + word[i] + word[i + 2:], 1)
        for i in range(len(word) + 1):
            for letter in alphabet:
                consider(word[:i] + letter + word[i:], 2)
        for i in range(len(word)):
            for letter in alphabet:
                consider(word[:i] + letter + word[i + 1:], 2)
        ranked.sort()
        return [variant for _, variant in ranked[:limit]]

    def _in_languages(self, word: str, languages: set[str] | None) -> bool:
        key = np.uint64(_hash(word))
        tables = [table for name, table in self._by_language.items() if not languages or name in languages] or [self._hashes]
        for table in tables:
            if table.size:
                index = int(np.searchsorted(table, key))
                if index < table.size and table[index] == key:
                    return True
        return False

    def near_russian(self, token: str) -> bool:
        """True if one edit (delete, insert, replace, swap) turns `token` into a Russian word form:
        the typical OCR misreading of a Russian word (созданин → создании)."""
        word = normalize(token)
        if not self._russian.size or not word:
            return False
        alphabet = "абвгдежзийклмнопрстуфхцчшщъыьэюя"
        variants = {word[:i] + word[i + 1:] for i in range(len(word))}
        variants |= {word[:i] + word[i + 1] + word[i] + word[i + 2:] for i in range(len(word) - 1)}
        variants |= {word[:i] + letter + word[i + 1:] for i in range(len(word)) for letter in alphabet}
        variants |= {word[:i] + letter + word[i:] for i in range(len(word) + 1) for letter in alphabet}
        variants.discard(word)
        keys = np.fromiter((_hash(variant) for variant in variants if len(variant) >= 3), dtype=np.uint64)
        positions = np.searchsorted(self._russian, keys).clip(0, self._russian.size - 1)
        return bool((self._russian[positions] == keys).any())

    def _known_single(self, token: str) -> bool:
        if token in self:
            return True
        cached = self._cache.get(token)
        if cached is None:
            cached = any(self.spelled(token, name) for name in self.hunspell)
            if not cached and "uz_UZ" in self.hunspell:
                cached = any(self.spelled(variant, "uz_UZ") for variant in uzbek_cyrillic(token))
            if len(self._cache) < 200_000:
                self._cache[token] = cached
        return cached

    def known(self, token: str) -> bool:
        """A token is known if it, or every part of a hyphenated compound, is in a word list."""
        if self._known_single(token):
            return True
        parts = [part for part in token.split("-") if part]  # an apostrophe is a letter in Uzbek
        return len(parts) > 1 and all(self._known_single(part) for part in parts)
