"""Will a source page fit on one Word page? Text is measured with Liberation
Serif, which has the same metrics as Times New Roman, and wrapped greedily the
way Word does. The writer shrinks type and spacing until the page fits with a
margin, so a one-page letter stays one page in Word.
"""

from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageFont

from ..model import Graphic, Page, Paragraph, Table

FONT_DIRS = ("/usr/share/fonts/truetype/liberation", "/usr/share/fonts/truetype/liberation2")
LINE_HEIGHT_EM = 1.15      # Word single spacing for Times New Roman
SCALES = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75)
SAFETY = 0.94              # leave room for Word/LibreOffice differences


@lru_cache(maxsize=2)
def _font(bold: bool) -> ImageFont.FreeTypeFont | None:
    name = "LiberationSerif-Bold.ttf" if bold else "LiberationSerif-Regular.ttf"
    for folder in [os.environ.get("DRE_FONT_DIR", ""), *FONT_DIRS]:
        path = Path(folder) / name
        if folder and path.exists():
            return ImageFont.truetype(str(path), 100)
    return None


def text_width(text: str, size: float, bold: bool = False) -> float:
    font = _font(bold)
    if font is None:
        return len(text) * size * 0.5
    return font.getlength(text) * size / 100


def wrapped_lines(text: str, width: float, size: float, *, bold: bool = False, first_indent: float = 0.0) -> int:
    lines = 0
    for part in text.split("\n"):
        words = part.split()
        if not words:
            lines += 1
            continue
        current = 0.0
        limit = width - first_indent
        space = text_width(" ", size, bold)
        lines += 1
        for word in words:
            length = text_width(word, size, bold)
            if current and current + space + length > limit:
                lines += 1
                current, limit = length, width
            else:
                current += (space if current else 0.0) + length
    return lines


def picture_size(data: bytes, max_width: float, max_height: float) -> tuple[float, float]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
    except Exception:
        return max_width, max_height
    if width <= 0 or height <= 0:
        return max_width, max_height
    scale = min(max_width / width, max_height / height, 1.0 if width > max_width else max_width / width)
    return max(1.0, width * scale), max(1.0, height * scale)


def page_height(page: Page, width: float, scale: float, spacing: float) -> float:
    body = page.body_size
    total = 0.0
    for block in page.blocks:
        if isinstance(block, Paragraph):
            size = (block.size or body) * scale
            indent = max(0.0, block.left_indent)
            lines = wrapped_lines(block.text.replace("\t", "    "), width - indent, size, bold=block.bold,
                                  first_indent=max(0.0, block.first_line_indent))
            total += lines * size * LINE_HEIGHT_EM * spacing + (block.space_before + block.space_after) * scale
        elif isinstance(block, Table):
            widths = block.column_widths or [width / max(1, block.columns)] * block.columns
            share = [width * value / max(1.0, sum(widths)) for value in widths]
            size = (body - (0 if block.borderless else 1)) * scale
            rows: dict[int, float] = {}
            for cell in block.cells:
                cell_width = sum(share[cell.column:cell.column + cell.column_span]) - 8
                if cell.graphic is not None:
                    height = min(cell.graphic.box.height, 120.0) * scale
                else:
                    height = wrapped_lines(cell.text, cell_width, size, bold=cell.bold) * size * LINE_HEIGHT_EM + 4
                rows[cell.row] = max(rows.get(cell.row, 0.0), height)
            total += sum(rows.values()) + 6 * scale
        elif isinstance(block, Graphic) and block.placement == "inline":
            total += picture_size(block.data, min(block.box.width, width), block.box.height)[1] * scale + 4
    return total


def choose_fit(page: Page, width: float, height: float) -> tuple[float, float]:
    """(type scale, line spacing) so that the page fits: the largest that does."""
    for spacing in (1.1, 1.0):
        for scale in SCALES:
            if page_height(page, width, scale, spacing) <= height * SAFETY:
                return scale, spacing
    return SCALES[-1], 1.0
