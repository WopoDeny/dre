"""How fast this machine is compared with the reference laptop core.

At start-up the engine recognizes a small built-in page and compares the time with
the time on the reference machine (one core of the development laptop). Cost
estimates are multiplied by the ratio, so a slow server refuses a document it
cannot finish in time before spending the time on it. DRE_SPEED_FACTOR fixes the
ratio instead (for tests and for machines with a known ratio).
"""

from __future__ import annotations

import os
import time
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

from .core import DeadlineBudget

# recognize() of reference_page() on one core of the reference laptop, best of three.
REFERENCE_MS = 330.0
MIN_FACTOR, MAX_FACTOR = 0.5, 20.0
LINES = (
    "Главным управлением рассмотрено Ваше обращение по вопросу оформления документов.",
    "По итогам рассмотрения материалы будут направлены в установленном порядке.",
    "Toshkent shahar, Yashnobod tumani, Mahtumquli koʻchasi, 21-uy. Tel.: 71-233-45-67",
    "Сіздің 2024 жылғы 12 қарашадағы хатыңызға сәйкес мәлімет ұсынамыз.",
    "The documents were reviewed and approved on 5 March 2025, No. 45/8-37.",
    "О результатах рассмотрения будете проинформированы органами внутренних дел.",
)


@lru_cache(maxsize=1)
def reference_page() -> Image.Image:
    image = Image.new("L", (2000, 70 * len(LINES) + 60), 255)
    draw = ImageDraw.Draw(image)
    font = None
    for path in ("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
                 "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf"):
        if os.path.exists(path):
            font = ImageFont.truetype(path, 40)
            break
    font = font or ImageFont.load_default()
    for index, line in enumerate(LINES):
        draw.text((40, 30 + 70 * index), line, font=font, fill=0)
    return image


def measure(backend, languages: str, repeats: int = 3) -> float:
    """Best time (ms) of recognizing the reference page."""
    image = reference_page()
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        backend.recognize(image, languages, DeadlineBudget(30_000), page_width=image.width, page_height=image.height, psm=6)
        best = min(best, (time.perf_counter() - started) * 1000)
    return best


def speed_factor(backend, languages: str) -> tuple[float, float | None]:
    """(factor, measured ms): estimates are multiplied by the factor (> 1 on a slower machine)."""
    fixed = os.environ.get("DRE_SPEED_FACTOR", "").strip()
    if fixed:
        return max(MIN_FACTOR, min(MAX_FACTOR, float(fixed))), None
    measured = measure(backend, languages)
    return max(MIN_FACTOR, min(MAX_FACTOR, measured / REFERENCE_MS)), measured
