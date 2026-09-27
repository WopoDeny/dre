"""Deployment settings read from the environment.

DRE_OCR_ALLOWED_SCRIPTS     scripts recognized on scans/photos (default Latin,Cyrillic)
DRE_NATIVE_ALLOWED_SCRIPTS  scripts accepted from PDF text layers (default Latin,Cyrillic)
DRE_MAX_PAGES               larger documents are rejected before recognition (default 10)
DRE_MIN_LETTER_PX           scans with smaller letters (x-height, pixels) are rejected (default 8)
DRE_PAGE_THREADS            pages recognized at once (default auto: by cores and memory, at most 4)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .resources import page_threads

DEFAULT_SCRIPTS = ("Latin", "Cyrillic")
KNOWN_SCRIPTS = ("Latin", "Cyrillic", "Arabic", "Han")


def _scripts(name: str) -> tuple[str, ...]:
    raw = os.environ.get(name, "")
    values = tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
    unknown = [value for value in values if value not in KNOWN_SCRIPTS]
    if unknown:
        raise ValueError(f"{name} contains unknown scripts: {', '.join(unknown)}")
    return values or DEFAULT_SCRIPTS


@dataclass(frozen=True)
class Settings:
    ocr_scripts: tuple[str, ...]
    native_scripts: tuple[str, ...]
    max_pages: int
    min_letter_px: float = 8.0
    page_threads: int = 1

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            ocr_scripts=_scripts("DRE_OCR_ALLOWED_SCRIPTS"),
            native_scripts=_scripts("DRE_NATIVE_ALLOWED_SCRIPTS"),
            max_pages=int(os.environ.get("DRE_MAX_PAGES", "10")),
            min_letter_px=float(os.environ.get("DRE_MIN_LETTER_PX", "8")),
            page_threads=page_threads(),
        )
