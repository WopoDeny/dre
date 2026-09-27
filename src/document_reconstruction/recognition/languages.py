"""Bounded automatic script profiles, extensible through the local registry."""

from __future__ import annotations

import unicodedata
from collections import Counter

from ..core import EngineError, ErrorCode
from .registry import MIXED_PROBE, MIXED_SCRIPT_MODELS, SCRIPT_MODEL_PROFILES, SCRIPT_PROFILES

PROFILES = SCRIPT_PROFILES
SCRIPT_ALIASES = {"Japanese": "Han", "Korean": "Han", "HanS": "Han", "HanT": "Han", "CJK": "Han"}


def script_counts(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for character in text:
        name = unicodedata.name(character, "")
        if "CYRILLIC" in name:
            counts["Cyrillic"] += 1
        elif "ARABIC" in name and character.isalpha():
            counts["Arabic"] += 1
        elif "CJK" in name or "IDEOGRAPH" in name:
            counts["Han"] += 1
        elif "LATIN" in name:
            counts["Latin"] += 1
    return counts


def direction(text: str) -> str:
    rtl = sum(unicodedata.bidirectional(c) in ("R", "AL") for c in text)
    ltr = sum(unicodedata.bidirectional(c) == "L" for c in text)
    return "rtl" if rtl > ltr else "ltr"


def canonical_script(script: str) -> str:
    """Map OSD aliases to the script-family names used by the profile registry."""
    return SCRIPT_ALIASES.get(script, script)


def tajik_profile_evidence(text: str) -> bool:
    """Detect multiple Tajik-specific Cyrillic letter forms in OCR observations.

    One shape is not sufficient to select an exact-language OCR retry. This is
    script evidence, not a claim that every token on a mixed-language page is
    Tajik, and it does not modify recognized characters.
    """
    return len({character.casefold() for character in text if character.casefold() in {"ҷ", "ӯ", "ӣ"}}) >= 2


def _complete_generic_profile(script: str, available: set[str]) -> list[str]:
    required = SCRIPT_MODEL_PROFILES.get(script, ())
    return list(required) if required and all(model in available for model in required) else []


def profile(script: str, available: set[str], *, broader: bool = False) -> str:
    script = canonical_script(script)
    if script == "Tajik":
        if "tgk" not in available:
            raise EngineError(ErrorCode.LANGUAGE_UNAVAILABLE, "The targeted local Tajik model is not installed.", stage="script", details={"script": script})
        return "tgk"
    if script == "Mixed":
        if all(model in available for model in MIXED_SCRIPT_MODELS):
            return "+".join(MIXED_SCRIPT_MODELS)
        selected = [language for language in MIXED_PROBE if language in available]
        represented = sum(bool(set(models) & set(selected)) for models in PROFILES.values())
        if represented < 2:
            raise EngineError(ErrorCode.LANGUAGE_UNAVAILABLE, "A mixed-script page cannot be covered by the installed local models.", stage="script", details={"script": script})
        return "+".join(selected)
    if script not in PROFILES:
        raise EngineError(ErrorCode.LANGUAGE_UNAVAILABLE, "The detected script is not supported by the configured local model registry.", stage="script", details={"script": script})

    generic = _complete_generic_profile(script, available)
    if generic:
        return "+".join(generic)

    candidates = list(PROFILES[script])
    if broader and script in ("Latin", "Cyrillic"):
        candidates = list(dict.fromkeys(candidates + list(PROFILES["Cyrillic"]) + list(PROFILES["Latin"])))
    selected = [language for language in candidates if language in available]
    if not selected or (script != "Latin" and not any(language != "eng" for language in selected)):
        raise EngineError(ErrorCode.LANGUAGE_UNAVAILABLE, "The detected script has no installed local recognition model.", stage="script", details={"script": script})
    return "+".join(selected)


# Unicode script names that identify a writing system of a letter. Letters of
# other scripts (Greek symbols, modifier letters, ordinals) are not counted.
_SCRIPT_NAME_PREFIXES = {
    "LATIN": "Latin", "CYRILLIC": "Cyrillic", "ARABIC": "Arabic", "CJK": "Han",
    "HIRAGANA": "Han", "KATAKANA": "Han", "HANGUL": "Han", "BOPOMOFO": "Han",
    "HEBREW": "Hebrew", "THAI": "Thai", "DEVANAGARI": "Devanagari", "GEORGIAN": "Georgian",
    "ARMENIAN": "Armenian", "SYRIAC": "Syriac", "THAANA": "Thaana", "ETHIOPIC": "Ethiopic",
    "BENGALI": "Indic", "TAMIL": "Indic", "TELUGU": "Indic", "KANNADA": "Indic", "MALAYALAM": "Indic",
    "GUJARATI": "Indic", "GURMUKHI": "Indic", "SINHALA": "Indic", "KHMER": "Khmer", "LAO": "Lao",
    "TIBETAN": "Tibetan", "MONGOLIAN": "Mongolian", "MYANMAR": "Myanmar", "YI": "Yi",
}
# Tesseract OSD names mapped to script families; None means "not a script
# decision" (for example Greek, which OSD reports for noisy Cyrillic/Latin).
OSD_SCRIPT_FAMILY = {
    "Latin": "Latin", "Fraktur": "Latin", "Vietnamese": "Latin", "Cyrillic": "Cyrillic",
    "Arabic": "Arabic", "Han": "Han", "HanS": "Han", "HanT": "Han", "Japanese": "Han",
    "Korean": "Han", "Katakana": "Han", "Hiragana": "Han", "Hangul": "Han", "Greek": None,
}


def letter_script(character: str) -> str | None:
    if not character.isalpha():
        return None
    name = unicodedata.name(character, "")
    script = _SCRIPT_NAME_PREFIXES.get(name.split(" ", 1)[0])
    if script is None and "IDEOGRAPH" in name:
        return "Han"
    return script


def letter_scripts(text: str) -> Counter[str]:
    return Counter(script for script in map(letter_script, text) if script is not None)


def disallowed_letters(text: str, allowed: tuple[str, ...]) -> Counter[str]:
    return Counter({name: count for name, count in letter_scripts(text).items() if name not in allowed})


def ocr_profile(allowed: tuple[str, ...], available: set[str], *, first: str | None = None) -> str:
    """One Tesseract profile covering every allowed script; `first` is the primary model."""
    ordered = sorted(allowed, key=lambda script: script != first)
    models: list[str] = []
    for script in ordered:
        generic = SCRIPT_MODEL_PROFILES.get(script, ())
        if generic and all(model in available for model in generic):
            models.extend(generic)
        else:
            models.extend(model for model in PROFILES.get(script, ()) if model in available)
    if not models:
        raise EngineError(ErrorCode.LANGUAGE_UNAVAILABLE, "No local OCR model is installed for the allowed scripts.", stage="script")
    return "+".join(dict.fromkeys(models))
