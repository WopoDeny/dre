"""Native PDF extraction using MuPDF; this module never invokes OCR."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import re

import pymupdf

from ..model import Box, Cell, Graphic, Table
from .types import RecognizedPage, Word


@dataclass(frozen=True)
class _SpanStyle:
    box: pymupdf.Rect
    size: float
    bold: bool
    italic: bool


def _native_text_dict(page: pymupdf.Page) -> dict:
    flags = getattr(pymupdf, "TEXTFLAGS_TEXT", None)
    try:
        return page.get_text("dict", flags=flags) if flags is not None else page.get_text("dict")
    except TypeError:  # pragma: no cover - compatibility with older PyMuPDF
        return page.get_text("dict")


def _span_styles(text_dict: dict) -> list[_SpanStyle]:
    bold_flag = int(getattr(pymupdf, "TEXT_FONT_BOLD", 16))
    italic_flag = int(getattr(pymupdf, "TEXT_FONT_ITALIC", 2))
    styles: list[_SpanStyle] = []
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                box = pymupdf.Rect(span.get("bbox", (0, 0, 0, 0)))
                if box.is_empty:
                    continue
                font = str(span.get("font", "")).casefold()
                span_flags = int(span.get("flags", 0))
                size = float(span.get("size", 11.0))
                styles.append(
                    _SpanStyle(
                        box,
                        min(96.0, max(4.0, size)),
                        bool(span_flags & bold_flag) or "bold" in font,
                        bool(span_flags & italic_flag) or "italic" in font or "oblique" in font,
                    )
                )
    return styles



def native_rtl_digit_order_anomalies(source: pymupdf.Page | dict) -> int:
    """Count independent Arabic-Indic numeric runs with reversed glyph geometry.

    In normal numeric display, successive *logical* decimal characters increase
    in horizontal position even when surrounding Arabic words flow right-to-left.
    A repeated reverse position is evidence that the selectable native Unicode
    layer is in visual order; it is not safe to repair it by guessing digits.
    Only count runs in lines containing Arabic letters, with distinct digit
    values and strictly separated character boxes. This is a narrow signal,
    not a general native-RTL correctness or completeness test.
    """
    if isinstance(source, dict):
        raw = source
    else:
        flags = getattr(pymupdf, "TEXTFLAGS_TEXT", None)
        raw = source.get_text("rawdict", flags=flags) if flags is not None else source.get_text("rawdict")
    anomalous_lines = 0
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            runs: list[list[dict]] = []
            run: list[dict] = []
            characters = [character for span in line.get("spans", []) for character in span.get("chars", [])]
            if not any(("\u0621" <= item.get("c", "") <= "\u064a") for item in characters):
                continue
            for item in characters + [{"c": " "}]:
                char = item.get("c", "")
                if len(char) == 1 and ("\u0660" <= char <= "\u0669" or "\u06f0" <= char <= "\u06f9"):
                    run.append(item)
                else:
                    if len(run) >= 2:
                        runs.append(run)
                    run = []
            for digits in runs:
                if len({item["c"] for item in digits}) < 2:
                    continue
                try:
                    centers = [(float(item["bbox"][0]) + float(item["bbox"][2])) / 2 for item in digits]
                    widths = [float(item["bbox"][2]) - float(item["bbox"][0]) for item in digits]
                except (IndexError, ValueError, TypeError, KeyError):
                    continue
                if (all(width >= 1.0 for width in widths)
                        and all(left - right >= min(widths[index], widths[index + 1]) * 0.35
                                for index, (left, right) in enumerate(zip(centers, centers[1:])))):
                    anomalous_lines += 1
                    break
    return anomalous_lines


def _explicit_native_spaces(text_dict: dict, native_words: list[tuple]) -> set[int]:
    """Keep only spaces proven by a matching source span, never guessed OCR gaps.

    MuPDF's word API removes whitespace. A native line whose source text tokens
    match its word records can prove that a source space precedes a token. More
    than one span is usable only when their boxes and word boxes are in the same
    horizontal reading order. Ambiguous/RTL-shaped lines fall back to existing
    script-aware joining rather than guessing source text.
    """
    grouped: dict[tuple[int, int], list[tuple[int, str]]] = defaultdict(list)
    for index, item in enumerate(native_words):
        grouped[(int(item[5]), int(item[6]))].append((index, str(item[4])))
    explicit: set[int] = set()
    for block_index, block in enumerate(text_dict.get("blocks", [])):
        if block.get("type") != 0:
            continue
        for line_index, line in enumerate(block.get("lines", [])):
            spans = line.get("spans", [])
            if not spans:
                continue
            if len(spans) > 1:
                # Text-span order may differ from reading order on RTL and
                # positioned text. Never infer whitespace from its concatenated
                # string unless both MuPDF views show a left-to-right sequence.
                line_dir = line.get("dir", (1.0, 0.0))
                if abs(float(line_dir[0]) - 1.0) > 0.05 or abs(float(line_dir[1])) > 0.05:
                    continue
                boxes = [pymupdf.Rect(span.get("bbox", (0, 0, 0, 0))) for span in spans]
                if any(box.is_empty for box in boxes):
                    continue
                if any(right.x0 + 0.5 < left.x1 or
                       (left & right).height < min(left.height, right.height) * 0.5
                       for left, right in zip(boxes, boxes[1:])):
                    continue
            source = "".join(str(span.get("text", "")) for span in spans)
            tokens = list(re.finditer(r"\S+", source))
            words = grouped.get((block_index, line_index), [])
            if len(tokens) != len(words) or any(token.group() != value for token, (_, value) in zip(tokens, words)):
                continue
            if len(spans) > 1:
                ordered = [native_words[index] for index, _ in words]
                if any(float(right[0]) + 0.5 < float(left[2]) for left, right in zip(ordered, ordered[1:])):
                    continue
            for token, (index, _) in zip(tokens, words):
                if token.start() and source[token.start() - 1].isspace():
                    explicit.add(index)
    return explicit


def _styles_for_box(styles: list[_SpanStyle], rect: pymupdf.Rect) -> list[_SpanStyle]:
    return [style for style in styles if style.box.intersects(rect)]


def _splits_native_text_span(detected, selected_rows: list[int], columns: set[int], styles: list[_SpanStyle]) -> bool:
    """Reject fabricated text-grid columns that cut across a native text span.

    PyMuPDF's text-table strategy can divide a *single prose line* into several
    artificial cells when successive paragraphs share the same left margin.
    Independent table cells have separate native spans; a candidate boundary
    cutting through one span is therefore evidence against safe cell ownership.
    Spanning titles outside the selected dense table rows are not penalized.
    """
    for row_index in selected_rows:
        cells = [pymupdf.Rect(detected.rows[row_index].cells[column])
                 for column in sorted(columns)
                 if detected.rows[row_index].cells[column] is not None]
        if len(cells) < 2:
            continue
        for style in styles:
            overlaps = 0
            for cell in cells:
                intersection = style.box & cell
                if (intersection.height >= min(2.0, cell.height * 0.25)
                        and intersection.width >= min(5.0, cell.width * 0.12)):
                    overlaps += 1
                    if overlaps >= 2:
                        return True
    return False


def _borderless_text_table(page: pymupdf.Page, index: int, styles: list[_SpanStyle]) -> list[Table]:
    """Detect a narrow, high-confidence class of native borderless tables.

    PyMuPDF's text strategy is intentionally noisy on ordinary prose and columns.
    We therefore accept only candidates with a bold multi-column header followed
    by at least two similarly populated rows. This is a practical extension, not
    a general borderless-table recognizer.
    """

    try:
        found = page.find_tables(strategy="text")
    except (TypeError, ValueError, RuntimeError):
        return []
    result: list[Table] = []
    for detected in found.tables:
        if not 2 <= detected.col_count <= 12:
            continue
        extracted = detected.extract()
        dense: list[tuple[int, list[str]]] = []
        for row_index, values in enumerate(extracted):
            normalized = [(value or "").strip() for value in values]
            if sum(bool(value) for value in normalized) >= 2:
                dense.append((row_index, normalized))
        if len(dense) < 3:
            continue
        # MuPDF's text strategy may wrap an actual table together with the
        # preceding document title/prose and the following note. Check each
        # candidate header independently, but accept only an uninterrupted
        # run of real, separately positioned cells. Never use a split prose
        # span as table evidence or pull surrounding text into the table.
        selected: list[tuple[int, list[str]]] = []
        populated_columns: set[int] = set()
        for start, (header_index, header_values) in enumerate(dense):
            columns = {column for column, value in enumerate(header_values) if value}
            if len(columns) < 2 or len(columns) != detected.col_count:
                continue
            header_row = detected.rows[header_index]
            header_styles: list[_SpanStyle] = []
            per_column_bold = True
            for column in sorted(columns):
                rect = header_row.cells[column]
                column_styles = _styles_for_box(styles, pymupdf.Rect(rect)) if rect is not None else []
                header_styles.extend(column_styles)
                if not any(style.bold for style in column_styles):
                    per_column_bold = False
            if (not per_column_bold or not header_styles
                    or sum(style.bold for style in header_styles) / len(header_styles) < 0.65
                    or _splits_native_text_span(detected, [header_index], columns, styles)):
                continue
            candidate = [(header_index, header_values)]
            previous_row = header_index
            previous_box = pymupdf.Rect(header_row.cells[min(columns)])
            for row_index, values in dense[start + 1:]:
                # PyMuPDF commonly inserts one empty spacer row between text
                # rows. More than one indicates a distinct table or section.
                gap_rows = row_index - previous_row - 1
                if gap_rows > 2:
                    break
                if gap_rows == 2:
                    # One compact, one-cell annotation plus an empty spacer
                    # can occur between true table rows. Keep the independent
                    # annotation outside the cell ownership below; do not
                    # bridge two populated rows or unrelated document sections.
                    intermediates = [
                        [(value or "").strip() for value in extracted[between]]
                        for between in range(previous_row + 1, row_index)
                    ]
                    if sorted(sum(bool(value) for value in row) for row in intermediates) != [0, 1]:
                        break
                if sum(bool(values[column]) for column in columns) < 2:
                    break
                row_rect = detected.rows[row_index].cells[min(columns)]
                if row_rect is None:
                    break
                box = pymupdf.Rect(row_rect)
                if box.y0 - previous_box.y0 > max(36.0, 3.5 * previous_box.height):
                    break
                if _splits_native_text_span(detected, [row_index], columns, styles):
                    break
                candidate.append((row_index, values))
                previous_row, previous_box = row_index, box
            if len(candidate) >= 3:
                selected, populated_columns = candidate, columns
                break
        if not selected:
            continue
        cells: list[Cell] = []
        boxes: list[Box] = []
        for logical_row, (source_row, values) in enumerate(selected):
            row = detected.rows[source_row]
            for column in sorted(populated_columns):
                rect = row.cells[column]
                if rect is None:
                    continue
                box = Box(*map(float, rect))
                boxes.append(box)
                cells.append(
                    Cell(
                        values[column], logical_row, column,
                        box=box, source_page=index,
                        source_region="native_borderless_table", confidence=0.86,
                    )
                )
        if not boxes:
            continue
        # Compact sparse column numbers after dropping PyMuPDF's incidental
        # empty columns, while retaining deterministic left-to-right order.
        remap = {source: target for target, source in enumerate(sorted(populated_columns))}
        for cell in cells:
            cell.column = remap[cell.column]
        ordered_columns = sorted(populated_columns)
        widths: list[float] = []
        for column in ordered_columns:
            rects = [detected.rows[row_index].cells[column] for row_index, _ in selected if detected.rows[row_index].cells[column] is not None]
            widths.append(float(sum(pymupdf.Rect(rect).width for rect in rects) / len(rects)))
        result.append(
            Table(
                Box.enclosing(boxes), index, len(selected), len(ordered_columns), cells,
                widths, confidence=0.86, source_region="native_borderless_table",
            )
        )
    return result



def _filled_header_row_rule_tables(
    page: pymupdf.Page, index: int, drawings: list[dict], existing: list[Table],
) -> list[Table]:
    """Recover a narrow class of partially ruled native tables from page vectors.

    Some PDF producers paint individual header-cell rectangles and draw a
    horizontal rule beneath *each* data cell, but no vertical ruling. MuPDF's
    line finder can therefore return only a header row or skip the table.
    Recover only when the painted header establishes a contiguous column grid,
    at least three consecutive complete rows of aligned rules exist, and every
    resulting cell contains observable native text. This does not infer tables
    from headings, source words, or visual alignment alone.
    """
    fills = [pymupdf.Rect(d['rect']) for d in drawings
             if d.get('type') == 'f' and d.get('fill') is not None
             and 12 <= pymupdf.Rect(d['rect']).height <= 45
             and 30 <= pymupdf.Rect(d['rect']).width <= 400]
    rules = [pymupdf.Rect(d['rect']) for d in drawings
             if d.get('type') == 's' and d.get('color') is not None
             and pymupdf.Rect(d['rect']).height <= 1.2
             and pymupdf.Rect(d['rect']).width >= 30]
    found: list[Table] = []
    for first in sorted(fills, key=lambda rect: (rect.y0, rect.x0)):
        header = sorted([r for r in fills
                         if abs(r.y0 - first.y0) <= 0.8
                         and abs(r.y1 - first.y1) <= 0.8], key=lambda r: r.x0)
        if not 3 <= len(header) <= 8 or header[0] != first:
            continue
        if any(abs(a.x1 - b.x0) > 0.8 for a, b in zip(header, header[1:])):
            continue
        if max(r.width for r in header) > min(r.width for r in header) * 3.0:
            continue
        width = header[0].height
        if any(r.height < width * 0.9 for r in header):
            continue
        # A complete separator has one independently drawn segment per column.
        # Requiring repeated, consecutive separators excludes painted parallel
        # headings and independent horizontal dividers between prose blocks.
        possible = sorted({r.y0 for r in rules
                           if first.y1 - 1 <= r.y0 <= first.y1 + 25 * 45
                           and abs(r.x0 - header[0].x0) <= 1.0
                           and abs(r.x1 - header[0].x1) <= 1.0})
        separators: list[float] = []
        for y in possible:
            if not separators and abs(y - first.y1) > 1.5:
                break
            if separators and not 0.6 * width <= y - separators[-1] <= 2.3 * width:
                break
            if not all(any(abs(rule.y0 - y) <= 0.6
                           and abs(rule.x0 - cell.x0) <= 1.0
                           and abs(rule.x1 - cell.x1) <= 1.0
                           for rule in rules) for cell in header):
                break
            separators.append(y)
        if len(separators) < 3:
            continue
        bounds = [first.y0, *separators]
        if any(bounds[i+1] - bounds[i] < 8 for i in range(len(bounds)-1)):
            continue
        bbox = pymupdf.Rect(header[0].x0, first.y0, header[-1].x1, bounds[-1])
        if any((bbox & pymupdf.Rect(table.box.x0, table.box.y0, table.box.x1, table.box.y1)).get_area()
               > 0.05 * bbox.get_area() for table in [*existing, *found]):
            continue
        cells: list[Cell] = []
        for row in range(len(bounds)-1):
            for column, head in enumerate(header):
                rect = pymupdf.Rect(head.x0, bounds[row], head.x1, bounds[row+1])
                text = page.get_textbox(rect).strip()
                # A blank cell or ambiguous multiline extraction has no
                # unambiguous cell ownership in this deliberately narrow path.
                if not text or '\n' in text:
                    cells = []
                    break
                cells.append(Cell(text, row, column, box=Box(*map(float, rect)),
                                  source_page=index, source_region='native_partial_rule_table',
                                  confidence=0.92))
            if not cells:
                break
        if len(cells) != (len(bounds)-1) * len(header):
            continue
        found.append(Table(Box(*map(float, bbox)), index, len(bounds)-1,
                           len(header), cells, [float(r.width) for r in header],
                           confidence=0.92, source_region='native_partial_rule_table'))
    return found


def extract_native(page: pymupdf.Page, index: int, *, confidence: float = 1.0) -> RecognizedPage:
    """Extract editable native observations in a normalized unrotated coordinate system."""

    original_rotation = int(page.rotation) % 360
    if original_rotation:
        page.set_rotation(0)
    try:
        result = _extract_unrotated(page, index, confidence=confidence)
        result.source_rotation = original_rotation
        return result
    finally:
        if original_rotation:
            page.set_rotation(original_rotation)


def _extract_unrotated(page: pymupdf.Page, index: int, *, confidence: float) -> RecognizedPage:
    words: list[Word] = []
    text_dict = _native_text_dict(page)
    styles = _span_styles(text_dict)
    native_words = page.get_text("words", sort=False)
    native_space_indices = _explicit_native_spaces(text_dict, native_words)

    def transformed(box: object) -> Box:
        rect = pymupdf.Rect(box)
        return Box(rect.x0, rect.y0, rect.x1, rect.y1)

    # Word boxes carry stable native block/line identifiers. Font metadata is
    # matched spatially to text-only spans, avoiding expensive font discovery.
    for source_index, (x0, y0, x1, y1, text, block_id, line_id, _) in enumerate(native_words):
        if not text.strip():
            continue
        native_box = pymupdf.Rect(x0, y0, x1, y1)
        center = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
        style = next((candidate for candidate in styles if center in candidate.box), None)
        box = transformed(native_box)
        words.append(
            Word(
                text,
                box,
                confidence=max(0.0, min(1.0, confidence)),
                size=style.size if style else min(96.0, max(4.0, box.height / 1.2)),
                bold=style.bold if style else False,
                line_id=(block_id, line_id),
                italic=style.italic if style else False,
                source_page=index,
                source_region="native_pdf",
                source_space_before=source_index in native_space_indices,
            )
        )

    tables: list[Table] = []
    # Detection only runs when enough vector rules exist to suggest a ruled table.
    drawings = page.get_drawings()
    if len(drawings) >= 4:
        finder = page.find_tables(paths=drawings, strategy="lines_strict")
        for detected in finder.tables:
            extracted = detected.extract()
            cells = []
            occupied: set[tuple[int, int]] = set()
            for row_index, row in enumerate(detected.rows):
                for column_index, rect in enumerate(row.cells):
                    if rect is None or (row_index, column_index) in occupied:
                        continue
                    span_columns = 1
                    while column_index + span_columns < detected.col_count and row.cells[column_index + span_columns] is None:
                        span_columns += 1
                    span_rows = 1
                    while row_index + span_rows < detected.row_count and detected.rows[row_index + span_rows].cells[column_index] is None:
                        span_rows += 1
                    occupied.update((r, c) for r in range(row_index, row_index + span_rows) for c in range(column_index, column_index + span_columns))
                    cells.append(Cell(extracted[row_index][column_index] or "", row_index, column_index, span_rows, span_columns, box=transformed(rect), source_page=index, source_region="native_table"))
            tables.append(Table(transformed(detected.bbox), index, detected.row_count, detected.col_count, cells, source_region="native_table"))

    tables.extend(_filled_header_row_rule_tables(page, index, drawings, tables))
    if not tables:
        tables.extend(_borderless_text_table(page, index, styles))

    graphics: list[Graphic] = []
    for image in page.get_image_info(xrefs=True):
        box = transformed(image["bbox"])
        if image.get("xref", 0) <= 0 or box.area / max(1.0, page.rect.get_area()) > 0.35:
            continue
        if image["width"] * image["height"] > 16_000_000:
            continue
        pixmap = pymupdf.Pixmap(page.parent, image["xref"])
        if pixmap.colorspace and pixmap.colorspace.n > 3:
            pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
        smask = page.parent.extract_image(image["xref"]).get("smask", 0)
        if smask and not pixmap.alpha:
            # Keep the transparency of the source picture (a signature without its background).
            mask = pymupdf.Pixmap(page.parent, smask)
            if (mask.width, mask.height) == (pixmap.width, pixmap.height):
                pixmap = pymupdf.Pixmap(pixmap, mask)
        graphics.append(Graphic(box, index, pixmap.tobytes("png"), source_region="native_graphic"))
    return RecognizedPage(index, page.rect.width, page.rect.height, words, tables, graphics)
