"""Local OCR model registry for automatic script/language routing.

The registry describes required or acceptable local Tesseract model IDs only.
It never downloads models and is independent of DOCX reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageSpec:
    name: str
    script: str
    tesseract_models: tuple[str, ...]


PRIORITY_LANGUAGES: tuple[LanguageSpec, ...] = (
    LanguageSpec("English", "Latin", ("eng",)),
    LanguageSpec("Uzbek Latin", "Latin", ("uzb",)),
    LanguageSpec("Turkmen", "Latin", ("tkm",)),
    LanguageSpec("Azerbaijani", "Latin", ("aze",)),
    LanguageSpec("Russian", "Cyrillic", ("rus",)),
    LanguageSpec("Kazakh", "Cyrillic", ("kaz",)),
    LanguageSpec("Kyrgyz", "Cyrillic", ("kir",)),
    LanguageSpec("Uzbek Cyrillic", "Cyrillic", ("uzb_cyrl",)),
    LanguageSpec("Tajik", "Cyrillic", ("tgk",)),
    LanguageSpec("Arabic", "Arabic", ("ara",)),
    LanguageSpec("Persian/Farsi", "Arabic", ("fas",)),
    LanguageSpec("Simplified Chinese", "Han", ("chi_sim",)),
    LanguageSpec("Traditional Chinese", "Han", ("chi_tra",)),
)

# Exact-language fallback profiles are retained for deployments that do not
# ship Tesseract's generic script traineddata.
SCRIPT_PROFILES: dict[str, tuple[str, ...]] = {
    "Latin": ("eng", "uzb", "tkm", "aze"),
    "Cyrillic": ("rus", "kaz", "kir", "uzb_cyrl", "tgk", "eng"),
    "Arabic": ("ara", "fas", "eng"),
    "Han": ("chi_sim", "chi_tra", "eng"),
}

# Generic script models are preferred when the complete local profile is
# available. They avoid quality regressions observed from combining several
# language-specific models while keeping routing bounded to one profile.
SCRIPT_MODEL_PROFILES: dict[str, tuple[str, ...]] = {
    "Latin": ("Latin",),
    "Cyrillic": ("Cyrillic",),
    "Arabic": ("Arabic",),
    "Han": ("HanS", "HanT"),
}

MIXED_PROBE = ("eng", "rus", "ara", "fas", "chi_sim", "chi_tra")
MIXED_SCRIPT_MODELS = ("Latin", "Cyrillic", "Arabic", "HanS", "HanT")


def production_models_for_script(script: str, available: set[str]) -> tuple[str, ...]:
    """Return the bounded production profile for a supported script family."""
    generic = SCRIPT_MODEL_PROFILES.get(script, ())
    if generic and all(model in available for model in generic):
        return generic
    return tuple(model for model in SCRIPT_PROFILES.get(script, ()) if model in available)


def capability_report(available: set[str]) -> dict[str, object]:
    priority = []
    for spec in PRIORITY_LANGUAGES:
        installed = [model for model in spec.tesseract_models if model in available]
        route_models = list(production_models_for_script(spec.script, available))
        generic_models = SCRIPT_MODEL_PROFILES.get(spec.script, ())
        generic_ready = bool(generic_models) and all(model in available for model in generic_models)
        production_route_ready = generic_ready or bool(installed)
        priority.append({
            "language": spec.name,
            "script": spec.script,
            "models": list(spec.tesseract_models),
            "installed_models": installed,
            # Preserve the historical meaning of ready: an exact language model
            # is present. Production-route readiness is reported separately.
            "ready": bool(installed),
            "exact_model_ready": bool(installed),
            "production_route_ready": production_route_ready,
            "production_route_models": route_models,
            "production_route_basis": "generic_script" if generic_ready else "exact_language" if installed else "insufficient_fallback",
            "production_route_uses_exact_model": bool(installed) and any(model in installed for model in route_models),
        })
    return {
        "backend": "tesseract",
        "priority_languages": priority,
        "osd_ready": "osd" in available,
        "generic_script_profiles": {
            script: {
                "models": list(models),
                "ready": all(model in available for model in models),
            }
            for script, models in SCRIPT_MODEL_PROFILES.items()
        },
    }
