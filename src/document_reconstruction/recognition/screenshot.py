"""Phone screenshots: keep the page content, drop the phone and app interface.

A screenshot shows a black status bar on top (icons, clock), often an app bar under it
("< На главную   Конституция   ≡") and a black navigation bar at the bottom. Such interface
must not reach the DOCX. The content area is found by its background: interface bars have
a background of their own. When bars are present but the content cannot be told apart from
the interface, the page is refused.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..core import EngineError, ErrorCode

DARK = 60             # status and navigation bars are near black
BAR_SHARE = 0.9       # of the screen width
MIN_BAR = 0.006       # of the page height: thicker than a ruling line
MAX_BAR = 0.05        # of the page height: a status bar, not a dark background
BAR_BLACK_SHARE = 0.8  # of a bar's pixels are true black (the rest: icons); a desk is dark gray
BACKGROUND_STEP = 4   # the interface background differs from the content by at least this much
UI_ZONE = 0.15        # the app bar lies within this share of the screen height from a bar


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    rows = np.flatnonzero(flags)
    if not rows.size:
        return []
    breaks = np.flatnonzero(np.diff(rows) > 1)
    starts = np.concatenate(([rows[0]], rows[breaks + 1]))
    ends = np.concatenate((rows[breaks], [rows[-1]]))
    return list(zip(starts.tolist(), (ends + 1).tolist()))


def strip_phone_ui(image: Image.Image, *, page_index: int = 0) -> tuple[Image.Image, dict[str, int] | None]:
    gray = np.asarray(image.convert("L")).astype(np.int16)
    height, width = gray.shape
    dark = gray < DARK
    # A status bar: a thick band of near-black rows across one contiguous span (the screen).
    bars = [(top, bottom) for top, bottom in _runs(dark.mean(axis=1) > 0.3) if bottom - top >= MIN_BAR * height]
    if not bars:
        return image, None
    top_bar = bars[0]
    columns = np.flatnonzero(dark[top_bar[0]:top_bar[1]].mean(axis=0) > 0.8)
    if columns.size < 0.3 * width:
        return image, None
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    if (dark[top_bar[0]:top_bar[1], x0:x1].mean(axis=1) < BAR_SHARE).mean() > 0.5:
        return image, None
    # A phone bar is thin, flat black with light screen under it; a desk around a photographed sheet is
    # thick, textured and dark further down.
    bar = gray[top_bar[0]:top_bar[1], x0:x1]
    below = gray[top_bar[1]:min(height, top_bar[1] + (top_bar[1] - top_bar[0]) * 3), x0:x1]
    if (top_bar[1] - top_bar[0] > MAX_BAR * height or float(np.median(bar)) > 20 or float((bar < 25).mean()) < BAR_BLACK_SHARE
            or not below.size or float(np.median(below)) < 180):
        return image, None
    screen = gray[:, x0 + 4:x1 - 4]
    screen_bars = [(top, bottom) for top, bottom in bars if (dark[top:bottom, x0:x1].mean(axis=1) >= BAR_SHARE).mean() > 0.5]
    screen_top = screen_bars[0][1]
    screen_bottom = screen_bars[-1][0] if len(screen_bars) > 1 else height
    if screen_bottom - screen_top < 0.2 * height:
        return image, None
    background = np.median(screen[screen_top:screen_bottom], axis=1)
    middle = background[(screen_bottom - screen_top) // 3: 2 * (screen_bottom - screen_top) // 3]
    content_bg = float(np.median(middle))
    zone = int(UI_ZONE * (screen_bottom - screen_top))
    differs = np.abs(background - content_bg) >= BACKGROUND_STEP
    # Interface right under the status bar: rows of another background (with the line under the app bar).
    def interface_rows(flags: np.ndarray) -> int:
        """Rows before the first run of 10 rows with the content background (the edge of a bar is skipped)."""
        for start in range(zone):
            if not flags[start:start + 10].any():
                return start
        return zone
    top_ui = interface_rows(differs)
    bottom_ui = interface_rows(differs[::-1])
    if top_ui >= zone or bottom_ui >= zone:
        raise EngineError(ErrorCode.UNREADABLE_DOCUMENT, "The screenshot interface could not be separated from the page content.",
                          stage="geometry", details={"page_index": page_index, "reason": "screenshot_ui_not_separated"})
    if top_ui == 0 and len(screen_bars) == 1:
        # Only a status bar and nothing that tells an app bar from the content: not safe to guess.
        raise EngineError(ErrorCode.UNREADABLE_DOCUMENT, "The screenshot interface could not be separated from the page content.",
                          stage="geometry", details={"page_index": page_index, "reason": "screenshot_ui_not_separated"})
    content_top = screen_top + top_ui          # the interface rows include the line under the app bar
    content_bottom = screen_bottom - bottom_ui
    crop = image.crop((x0, content_top, x1, content_bottom))
    # The content keeps its scale: it is placed at the top of a sheet with the proportions of the page.
    sheet = Image.new(image.mode, (crop.width, max(crop.height, round(crop.width * height / width))), "white")
    sheet.paste(crop, (0, 0))
    return sheet, {"x0": x0, "y0": content_top, "x1": x1, "y1": content_bottom, "interface_top_px": top_ui, "interface_bottom_px": bottom_ui}
