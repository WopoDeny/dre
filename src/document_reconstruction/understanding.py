"""Recognition-independent document understanding and normalization.

Native extraction and raster recognition both produce ``RecognizedPage``
observations. This module is the only place where those observations become the
Unified Document Model consumed by reconstruction.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, replace
from statistics import median

from .model import (
    Box,
    Cell,
    Document,
    Graphic,
    InlineRun,
    Page,
    Paragraph,
    StructuralIssue,
    Table,
)
from .recognition.languages import direction, script_counts
from .recognition.types import RecognizedPage, Word

LIST_PREFIX = re.compile(r"^\s*((?:[\u2022\u25cf\u25aa*\-\u2013])|(?:\(?\d{1,3}[.)])|(?:[A-Za-z][.)]))\s+")
TERMINAL_PUNCTUATION = (".", "!", "?", ":", ";", "。", "！", "？", "：", "；", "؟", "؛")
NO_SPACE_BEFORE = set(",.;:!?%)]}»›，。！？、；：）》」』】؟؛،")
NO_SPACE_AFTER = set("([{«‹（《「『【")
CJK_PUNCTUATION = set("，。！？、；：）》」』】（《「『【")


def _dominant_script(text: str) -> str | None:
    counts = script_counts(text)
    return counts.most_common(1)[0][0] if counts else None


def _inline_separator(left: str, right: str) -> str:
    if not left or not right:
        return ""
    if left[-1] in NO_SPACE_AFTER or right[0] in NO_SPACE_BEFORE:
        return ""
    left_han = bool(script_counts(left[-1:]).get("Han"))
    right_han = bool(script_counts(right[:1]).get("Han"))
    if left_han and (right_han or right[0] in NO_SPACE_BEFORE or right[0].isalnum()):
        return ""
    if right_han and (left[-1] in CJK_PUNCTUATION or left[-1].isalnum()):
        return ""
    return " "


def _line_join_mode(left: str, right: str) -> str:
    """Return ``drop_hyphen``, ``tight`` or ``space`` for a visual line break."""

    if not left or not right:
        return "tight"
    if left.endswith("-") and len(left) > 1:
        token = re.split(r"\s+", left[:-1])[-1]
        # Conservative dehyphenation: only a reasonably long plain word followed
        # by lowercase text. Short compounds and identifiers retain the hyphen.
        if token.isalpha() and len(token) >= 4 and "-" not in token and right[:1].islower():
            return "drop_hyphen"
        if right[:1].isalnum():
            return "tight"
    if _inline_separator(left, right) == "":
        return "tight"
    return "space"


def join_text(left: str, right: str) -> str:
    if not left:
        return right.strip()
    right = right.strip()
    mode = _line_join_mode(left, right)
    if mode == "drop_hyphen":
        return left[:-1] + right
    if mode == "tight":
        return left.rstrip() + right
    return left.rstrip() + " " + right.lstrip()


@dataclass
class _Line:
    text: str
    box: Box
    size: float
    confidence: float
    runs: list[InlineRun]
    bold_ratio: float = 0.0
    italic_ratio: float = 0.0
    column: int = 0


def _word_run(word: Word, fallback_page: int) -> InlineRun:
    page = fallback_page if word.source_page is None else word.source_page
    return InlineRun(
        word.text,
        word.box,
        page,
        confidence=max(0.0, min(1.0, word.confidence)),
        direction=direction(word.text),
        bold=word.bold,
        italic=word.italic,
        size=word.size,
        script=_dominant_script(word.text),
        source_region=word.source_region,
        underline=getattr(word, "underline", False),
    )


def _same_run_style(left: InlineRun, right: InlineRun) -> bool:
    return (
        left.bold == right.bold
        and left.italic == right.italic
        and left.underline == right.underline
        and left.direction == right.direction
        and left.script == right.script
        and left.source_page == right.source_page
        and left.source_region == right.source_region
        and abs(left.size - right.size) <= max(0.75, min(left.size, right.size) * 0.08)
    )


def _line_from_words(words: list[Word], page_index: int) -> _Line:
    runs: list[InlineRun] = []
    text = ""
    for word in words:
        run = _word_run(word, page_index)
        separator = (" " if text and word.source_region == "native_pdf" and word.source_space_before
                     else _inline_separator(text, run.text))
        if runs and _same_run_style(runs[-1], run):
            runs[-1].text += separator + run.text
            runs[-1].box = Box.enclosing([runs[-1].box, run.box])
            runs[-1].confidence = min(runs[-1].confidence, run.confidence)
        else:
            if separator:
                run.text = separator + run.text
            runs.append(run)
        text += separator + word.text
    weights = [max(1, len(word.text)) for word in words]
    total = sum(weights) or 1
    return _Line(
        text,
        Box.enclosing([word.box for word in words]),
        median(word.size for word in words),
        sum(word.confidence * weight for word, weight in zip(words, weights)) / total,
        runs,
        sum(weight for word, weight in zip(words, weights) if word.bold) / total,
        sum(weight for word, weight in zip(words, weights) if word.italic) / total,
    )


def _merge_line_into(paragraph: Paragraph, line: _Line) -> None:
    mode = _line_join_mode(paragraph.text, line.text)
    paragraph.text = join_text(paragraph.text, line.text)
    incoming = [replace(run) for run in line.runs]
    if incoming:
        if mode == "drop_hyphen" and paragraph.runs:
            paragraph.runs[-1].text = paragraph.runs[-1].text[:-1] if paragraph.runs[-1].text.endswith("-") else paragraph.runs[-1].text
        elif mode == "space":
            incoming[0].text = " " + incoming[0].text.lstrip()
        paragraph.runs.extend(incoming)
    paragraph.box = Box.enclosing([paragraph.box, line.box])
    paragraph.confidence = min(paragraph.confidence, line.confidence)
    paragraph.source_lines += 1
    paragraph.direction = direction(paragraph.text)


def _normalized_text(value: str) -> str:
    return " ".join(value.split())


def _cell_runs_from_words(cell: Cell, words: list[Word], page_index: int) -> tuple[list[InlineRun], str]:
    """Build table-cell runs only when word evidence reproduces the owned cell text.

    Table text remains owned by the table detector. Word observations are used only
    to enrich that text with style/provenance; a mismatch is surfaced rather than
    replacing or duplicating cell content.
    """

    if cell.box is None or not cell.text.strip():
        return [], ""
    contained = [word for word in words if cell.box.contains_center(word.box)]
    if not contained:
        return [], ""
    groups: dict[tuple[int, ...], list[Word]] = defaultdict(list)
    for word in contained:
        groups[word.line_id or (round(word.box.y0 / 4),)].append(word)
    lines = sorted((_line_from_words(group, page_index) for group in groups.values() if group), key=lambda line: (line.box.y0, line.box.x0))
    if not lines:
        return [], ""
    candidate = ""
    runs: list[InlineRun] = []
    for line in lines:
        mode = _line_join_mode(candidate, line.text) if candidate else "tight"
        incoming = [replace(run) for run in line.runs]
        if candidate and incoming:
            if mode == "drop_hyphen" and runs and runs[-1].text.endswith("-"):
                runs[-1].text = runs[-1].text[:-1]
            elif mode == "space":
                incoming[0].text = " " + incoming[0].text.lstrip()
        candidate = join_text(candidate, line.text) if candidate else line.text
        runs.extend(incoming)
    return runs, candidate


def _normalized_table(table: Table, page_index: int, words: list[Word]) -> Table:
    cells: list[Cell] = []
    table_issues = list(table.issues)
    for source in table.cells:
        cell = replace(source)
        cell.source_page = page_index if cell.source_page is None else cell.source_page
        cell.direction = direction(cell.text)
        candidate_runs, candidate_text = _cell_runs_from_words(cell, words, page_index)
        # Inline observations are optional style evidence, never an alternative
        # transcription. Similar whitespace is not enough: a detector-owned
        # multiline cell and space-joined word observations are different
        # editable Word contents. Attach runs only when they reproduce the cell
        # text byte-for-byte, including Unicode punctuation and line breaks.
        if candidate_runs and candidate_text == cell.text and "".join(run.text for run in candidate_runs) == cell.text:
            cell.runs = candidate_runs
        elif candidate_text and (candidate_text != cell.text or "".join(run.text for run in candidate_runs) != cell.text):
            cell.issues = [*cell.issues, StructuralIssue(
                "table_cell_run_mismatch",
                "Word observations inside the cell did not reproduce the detector-owned cell text; inline style evidence was not attached.",
                "info",
                min(cell.confidence, min((run.confidence for run in candidate_runs), default=cell.confidence)),
                cell.box,
            )]
        if not cell.text.strip():
            cell.issues = [*cell.issues, StructuralIssue("empty_table_cell", "No text was assigned to this detected table cell.", "info", 0.5, cell.box)]
        cells.append(cell)
    if not cells and table.rows * table.columns:
        table_issues.append(StructuralIssue("table_structure_without_cells", "A table grid was detected but no cell observations were available.", "warning", 0.25, table.box))
    if table.source_region == "raster_table" and any(cell.row_span > 1 or cell.column_span > 1 for cell in cells):
        table_issues.append(StructuralIssue("inferred_table_span", "One or more raster-table cell spans were inferred from grid geometry.", "info", table.confidence, table.box))
    return replace(table, cells=cells, issues=table_issues)


def _normalized_graphic(graphic: Graphic, page_index: int) -> Graphic:
    role = graphic.role
    description = graphic.description.casefold()
    if role == "graphic" and any(value in description for value in ("stamp", "seal", "document mark")):
        role = "stamp"
    return replace(graphic, source_page=page_index, role=role)


# ---------------------------------------------------------------------------
# Page layout: the page should look typed in Word, not copied from the scan.
#
# Words form segments (a line split at wide gaps), segments form rows. A run of
# rows with several side-by-side segments is a multi-column band (bilingual
# letterhead, two blocks next to each other) and becomes a borderless table; a
# single such row is a tab line ("Директор школы <tab> Э. Володина"). All other
# rows flow into paragraphs with one alignment, a first-line indent and a size.
# ---------------------------------------------------------------------------

# Height of an OCR word box (capitals, ascenders, descenders) relative to the font size.
OCR_BOX_PER_EM = 0.9
GAP_EM = 2.2          # a horizontal gap wider than this splits a line into segments
FULL_LINE_EM = 2.5    # a line that ends closer than this to the right margin is full
FLOATING_ROLES = {"signature", "seal", "stamp"}
# A list number or bullet standing apart from its text still starts that line.
LIST_MARKER_ONLY = re.compile(r"^(?:[\u2022\u25cf\u25aa*\-\u2013]|\(?\d{1,3}[.)]|[A-Za-z][.)])$")


SIZE_TOLERANCE = 0.15        # line sizes this close to the body size are the body size
PHOTO_SIZE_TOLERANCE = 0.3   # on a photographed sheet
PHOTO_FULL_LINE = 0.6        # a photo line at least this share of the text width wide is running text
PHOTO_BODY_RANGE = (0.6, 1.7)


def _ocr_sizes(words: list[Word], hint: float | None = None, photo: bool = False) -> None:
    """Font size of OCR words from the tallest word of their line."""
    lines: dict[tuple[int, ...], list[Word]] = defaultdict(list)
    for word in words:
        if word.source_region.startswith("ocr") or word.source_region == "blank_field":
            lines[word.line_id or (round(word.box.y0 / 4),)].append(word)
    for members in lines.values():
        heights = sorted(word.box.height for word in members if any(character.isalpha() for character in word.text))
        if not heights:
            continue
        tallest = heights[int(0.8 * (len(heights) - 1))]
        size = max(6.0, min(40.0, tallest / OCR_BOX_PER_EM))
        for word in members:
            word.size = size
    sizes = sorted(word.size for members in lines.values() for word in members)
    if sizes:
        typical = sizes[len(sizes) // 2]
        body = hint or typical
        spans = {key: max(word.box.x1 for word in members) - min(word.box.x0 for word in members) for key, members in lines.items()}
        text_width = max(spans.values(), default=0.0)
        for key, members in lines.items():
            if photo and spans[key] >= PHOTO_FULL_LINE * text_width and PHOTO_BODY_RANGE[0] <= members[0].size / typical <= PHOTO_BODY_RANGE[1]:
                # A full-width line of a photographed sheet is running text: its measured size only
                # reflects blur and residual perspective. Headings are short lines.
                for word in members:
                    word.size = body
                continue
            for word in members:
                # Height estimates of one printed size scatter by a few points;
                # only clearly larger or smaller lines keep their own size.
                ratio = word.size / typical
                word.size = body if abs(ratio - 1) <= (PHOTO_SIZE_TOLERANCE if photo else SIZE_TOLERANCE) else round(body * ratio * 2) / 2


def _segments(words: list[Word], page_index: int) -> list[_Line]:
    groups: dict[tuple[int, ...], list[Word]] = defaultdict(list)
    for word in words:
        groups[word.line_id or (round(word.box.y0 / 4),)].append(word)
    segments: list[_Line] = []
    for group in groups.values():
        group.sort(key=lambda word: word.box.x0)
        em = max(4.0, median(word.size for word in group))
        current = [group[0]]
        for word in group[1:]:
            marker_only = len(current) == 1 and LIST_MARKER_ONLY.match(current[0].text)
            if word.box.x0 - current[-1].box.x1 > GAP_EM * em and not marker_only:
                segments.append(_line_from_words(current, page_index))
                current = [word]
            else:
                current.append(word)
        segments.append(_line_from_words(current, page_index))
    for number, segment in enumerate(segments):
        if segment.runs and direction(segment.text) == "rtl":
            # Right-to-left text is stored in logical order: right-most word first.
            ordered = sorted((word for word in words if segment.box.contains_center(word.box) and word.line_id == _segment_key(segment, words)),
                             key=lambda word: -word.box.x1)
            if ordered:
                segments[number] = _line_from_words(ordered, page_index)
    return segments


def _segment_key(segment: _Line, words: list[Word]) -> tuple[int, ...]:
    for word in words:
        if segment.box.contains_center(word.box):
            return word.line_id
    return ()


def _center(box: Box) -> float:
    return (box.y0 + box.y1) / 2


def _rows(segments: list[_Line]) -> list[list[_Line]]:
    rows: list[list[_Line]] = []
    for segment in sorted(segments, key=lambda line: _center(line.box)):
        for row in rows:
            row_center = sum(_center(line.box) for line in row) / len(row)
            height = min(segment.box.height, *(line.box.height for line in row))
            if abs(_center(segment.box) - row_center) <= 0.45 * height and all(
                segment.box.x1 <= line.box.x0 or segment.box.x0 >= line.box.x1 for line in row
            ):
                row.append(segment)
                break
        else:
            rows.append([segment])
    for row in rows:
        row.sort(key=lambda line: line.box.x0)
    return sorted(rows, key=lambda row: min(line.box.y0 for line in row))


def _join_marker(row: list[_Line]) -> list[_Line]:
    """"1." or "•" set apart from the item text is one line with it."""
    if len(row) < 2 or not LIST_MARKER_ONLY.match(row[0].text):
        return row
    marker, text = row[0], row[1]
    runs = [replace(run) for run in marker.runs] + [replace(run) for run in text.runs]
    if text.runs:
        runs[len(marker.runs)].text = " " + runs[len(marker.runs)].text.lstrip()
    joined = _Line(marker.text + " " + text.text, Box.enclosing([marker.box, text.box]), text.size,
                   min(marker.confidence, text.confidence), runs, text.bold_ratio, text.italic_ratio)
    return [joined] + row[2:]


def _slots(rows: list[list[_Line]]) -> list[list[float]]:
    intervals = sorted((line.box.x0, line.box.x1) for row in rows for line in row)
    merged: list[list[float]] = []
    for left, right in intervals:
        if merged and left <= merged[-1][1] + 4:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return merged


def _row_box(row: list[_Line]) -> Box:
    return Box.enclosing([line.box for line in row])


def _joins_band(band: list[list[_Line]], row: list[_Line], content_width: float) -> bool:
    previous = _row_box(band[-1])
    box = _row_box(row)
    height = max(previous.height, box.height)
    if box.y0 - previous.y1 > 1.8 * height:
        return False
    slots = _slots(band)
    if len(slots) < 2:
        return len(row) >= 2
    def slot_of(line: _Line) -> int | None:
        hits = [index for index, (left, right) in enumerate(slots) if min(right, line.box.x1) - max(left, line.box.x0) > 0]
        return hits[0] if len(hits) == 1 else None
    if len(row) == 1:
        return row[0].box.width < 0.6 * content_width and slot_of(row[0]) is not None
    return all(slot_of(line) is not None for line in row) and len({slot_of(line) for line in row}) == len(row)


def _units(rows: list[list[_Line]], content_width: float) -> list[tuple[str, list[list[_Line]]]]:
    units: list[tuple[str, list[list[_Line]]]] = []
    index = 0
    while index < len(rows):
        if len(rows[index]) >= 2:
            band = [rows[index]]
            while index + 1 < len(rows) and _joins_band(band, rows[index + 1], content_width):
                index += 1
                band.append(rows[index])
            # A block that starts one line higher on one side (letterhead) begins with single rows.
            while units and units[-1][0] == "row" and len(_slots(band)) >= 2 and _text_columns(band, _slots(band)) and _joins_band([units[-1][1][0]] + band, band[0], content_width) \
                    and _fits_slot(units[-1][1][0][0], _slots(band), content_width) and _same_style(units[-1][1][0][0], band) and _row_box(band[0]).y0 - _row_box(units[-1][1][0]).y1 <= 1.8 * _row_box(band[0]).height:
                band.insert(0, units.pop()[1][0])
            units.append(("band", band))
        else:
            units.append(("row", [rows[index]]))
        index += 1
    return units


def _same_style(line: _Line, band: list[list[_Line]]) -> bool:
    """A title above side-by-side blocks (larger or bold) is not part of them."""
    sizes = [member.size for row in band for member in row]
    typical = median(sizes)
    bold = sum(member.bold_ratio >= 0.6 for row in band for member in row) > len(sizes) / 2
    return abs(line.size - typical) <= 0.15 * typical and (line.bold_ratio >= 0.6) == bold


def _fits_slot(line: _Line, slots: list[list[float]], content_width: float) -> bool:
    hits = [slot for slot in slots if min(slot[1], line.box.x1) - max(slot[0], line.box.x0) > 0]
    return len(hits) == 1 and line.box.width < 0.6 * content_width


def _is_full(line: _Line, right: float) -> bool:
    return line.box.x1 >= right - FULL_LINE_EM * max(4.0, line.size)


def _is_centered(line: _Line, left: float, right: float) -> bool:
    middle = (left + right) / 2
    return (abs((line.box.x0 + line.box.x1) / 2 - middle) <= 0.04 * (right - left)
            and line.box.x0 > left + 1.5 * max(4.0, line.size) and not _is_full(line, right))


def _wrapped(previous: _Line, current: _Line, right: float) -> bool:
    """Ragged-right text: the line broke because the next word did not fit."""
    words = current.text.split()
    if not words:
        return False
    first = current.box.width * len(words[0]) / max(1, len(current.text))
    space = 0.3 * max(4.0, previous.size)
    return previous.box.x1 + space + first > right - 0.5 * space


def _new_paragraph(previous: _Line, current: _Line, paragraph: list[_Line], left: float, right: float, pitch: float) -> bool:
    em = max(4.0, previous.size)
    if current.box.y0 - previous.box.y1 > max(0.75 * pitch, 0.6 * em):
        return True
    if abs(current.size - previous.size) > 0.2 * previous.size:
        return True
    if (previous.bold_ratio >= 0.6) != (current.bold_ratio >= 0.6):
        return True
    if LIST_PREFIX.match(current.text):
        return True
    if LIST_PREFIX.match(paragraph[0].text):
        # A wrapped list item continues under its text, right of the marker.
        return not (current.box.x0 >= paragraph[0].box.x0 + 0.3 * em and current.box.y0 - previous.box.y1 <= 0.6 * em)
    if _is_centered(previous, left, right) or _is_centered(current, left, right):
        return True  # centered lines (titles, letterhead lines) stay separate lines
    if not _is_full(previous, right) and not _wrapped(previous, current, right):
        return True
    body_left = min((line.box.x0 for line in paragraph[1:]), default=left)
    return current.box.x0 > min(body_left, previous.box.x0) + 0.8 * em


def _paragraph_from_lines(lines: list[_Line], page_index: int, left: float, right: float, next_list_id: int) -> Paragraph:
    first = lines[0]
    paragraph = Paragraph(first.text, first.box, page_index, confidence=first.confidence, direction=direction(first.text),
                          runs=[replace(run) for run in first.runs])
    for line in lines[1:]:
        _merge_line_into(paragraph, line)
    em = max(4.0, median(line.size for line in lines))
    paragraph.size = em
    paragraph.bold = sum(line.bold_ratio * len(line.text) for line in lines) / max(1, sum(len(line.text) for line in lines)) >= 0.6
    paragraph.source_lines = len(lines)
    if len(lines) > 1:
        rest_left = min(line.box.x0 for line in lines[1:])
        if all(_is_centered(line, left, right) for line in lines):
            paragraph.alignment = "center"
        elif all(_is_full(line, right) for line in lines[:-1]):
            paragraph.alignment = "justify"
        else:
            paragraph.alignment = "left"
        if paragraph.alignment != "center":
            paragraph.left_indent = rest_left - left if rest_left - left > 0.8 * em else 0.0
            paragraph.first_line_indent = first.box.x0 - rest_left
    else:
        em_one = max(4.0, first.size)
        symmetric = abs((first.box.x0 - left) - (right - first.box.x1)) <= 1.5 * em_one and first.box.x0 - left > 0.5 * em_one
        if _is_centered(first, left, right) or symmetric:
            paragraph.alignment = "center"
        elif first.box.x1 >= right - em and first.box.x0 > left + 0.3 * (right - left):
            paragraph.alignment = "right"
        else:
            paragraph.alignment = "justify" if _is_full(first, right) else "left"
            if first.box.x0 - left > 0.8 * em:
                paragraph.first_line_indent = first.box.x0 - left
    match = LIST_PREFIX.match(paragraph.text)
    if match:
        marker = match.group(1)
        paragraph.kind = "list_item"
        paragraph.list_marker = marker
        paragraph.list_ordered = bool(re.match(r"^\(?\d|^[A-Za-z][.)]", marker))
        paragraph.list_id = next_list_id
        paragraph.first_line_indent = 0.0
        paragraph.left_indent = max(0.0, first.box.x0 - left)
    return paragraph


def _tab_line(row: list[_Line], page_index: int, left: float) -> Paragraph:
    parts = row[:3]
    text = "\t".join(line.text for line in parts)
    runs: list[InlineRun] = []
    for number, line in enumerate(parts):
        incoming = [replace(run) for run in line.runs]
        if number and incoming:
            incoming[0].text = "\t" + incoming[0].text.lstrip()
        runs.extend(incoming)
    paragraph = Paragraph(text, _row_box(parts), page_index, confidence=min(line.confidence for line in parts),
                          direction=direction(text), runs=runs if "".join(run.text for run in runs) == text else [])
    paragraph.tab_line = True
    paragraph.alignment = "left"
    paragraph.size = max(4.0, median(line.size for line in parts))
    paragraph.bold = all(line.bold_ratio >= 0.6 for line in parts)
    if parts[0].box.x0 - left > 0.8 * paragraph.size:
        paragraph.first_line_indent = parts[0].box.x0 - left
    return paragraph


def _text_columns(band: list[list[_Line]], slots: list[list[float]]) -> bool:
    """A letterhead-like band: every column is a block of at least two lines of words."""
    for slot_left, slot_right in slots:
        lines = [line for row in band for line in row if slot_left - 1 <= (line.box.x0 + line.box.x1) / 2 <= slot_right + 1]
        worded = [line for line in lines if sum(character.isalpha() for character in line.text) >= 3]
        if len(worded) < 2:
            return False
    return True


def _band_table(band: list[list[_Line]], graphics: list[Graphic], page_index: int) -> Table:
    slots = _slots(band)
    box = Box.enclosing([line.box for row in band for line in row])
    inside = [graphic for graphic in graphics if box.y0 - 4 <= _center(graphic.box) <= box.y1 + 4
              and any(slots[i][1] <= graphic.box.x0 and graphic.box.x1 <= slots[i + 1][0] for i in range(len(slots) - 1))]
    columns: list[tuple[float, float, Graphic | None]] = [(left, right, None) for left, right in slots]
    for graphic in inside:
        columns.append((graphic.box.x0, graphic.box.x1, graphic))
    columns.sort(key=lambda column: column[0])
    cells: list[Cell] = []
    for number, (column_left, column_right, graphic) in enumerate(columns):
        members = [line for row in band for line in row if column_left - 1 <= (line.box.x0 + line.box.x1) / 2 <= column_right + 1] if graphic is None else []
        members.sort(key=lambda line: line.box.y0)
        # Lines of one column are reflowed into paragraphs; "\n" separates paragraphs.
        paragraphs: list[list[_Line]] = []
        pitch = median([b.box.y0 - a.box.y1 for a, b in zip(members, members[1:])] or [0.0])
        for line in members:
            previous = paragraphs[-1][-1] if paragraphs else None
            sentence_end = previous is not None and previous.text.rstrip().endswith((".", "!", "?", ":", ";")) and line.text[:1].isupper()
            if paragraphs and not sentence_end and not _new_paragraph(previous, line, paragraphs[-1], column_left, column_right, max(0.0, pitch)):
                paragraphs[-1].append(line)
            else:
                paragraphs.append([line])
        texts = []
        for group in paragraphs:
            value = group[0].text
            for line in group[1:]:
                value = join_text(value, line.text)
            texts.append(value)
        text = "\n".join(texts)
        cell_box = Box(column_left, box.y0, column_right, box.y1) if graphic is None else graphic.box
        centered = members and all(abs((line.box.x0 + line.box.x1) / 2 - (column_left + column_right) / 2) <= 0.08 * (column_right - column_left) for line in members)
        flush_right = members and not centered and all(column_right - line.box.x1 <= 0.05 * (column_right - column_left) for line in members)
        cells.append(Cell(text, 0, number, direction=direction(text), box=cell_box,
                          source_page=page_index, source_region="layout_column", alignment="center" if centered or graphic else "right" if flush_right else "left",
                          bold=bool(members) and all(line.bold_ratio >= 0.6 for line in members), graphic=graphic))
    widths = [right - left for left, right, _ in columns]
    table_box = Box.enclosing([box] + [graphic.box for graphic in inside])
    table = Table(table_box, page_index, 1, len(columns), cells, widths, kind="layout", source_region="layout_columns", borderless=True)
    return table


def _body_size(paragraphs: list[Paragraph]) -> float:
    weighted = sorted((paragraph.size, len(paragraph.text)) for paragraph in paragraphs if paragraph.kind == "paragraph" and paragraph.size)
    if not weighted:
        return 12.0
    total = sum(weight for _, weight in weighted)
    running = 0
    for size, weight in weighted:
        running += weight
        if running >= total / 2:
            return round(size * 2) / 2
    return 12.0


def understand_page(recognized: RecognizedPage) -> Page:
    """Convert recognition observations into one normalized logical page."""

    tables = [_normalized_table(table, recognized.index, recognized.words) for table in recognized.tables]
    graphics = [_normalized_graphic(graphic, recognized.index) for graphic in recognized.graphics]
    words = [
        word for word in recognized.words
        if not any(
            (any(cell.box is not None and cell.box.contains_center(word.box) for cell in table.cells)
             if all(cell.box is not None for cell in table.cells) else table.box.contains_center(word.box))
            for table in tables
        )
    ]
    _ocr_sizes(words, recognized.body_size_hint, recognized.photo)
    segments = _segments(words, recognized.index) if words else []
    rows = [_join_marker(row) for row in _rows(segments)]
    wide = [line for row in rows if len(row) == 1 for line in row if line.box.width >= 0.45 * recognized.width]
    every = [line for row in rows for line in row]
    def quantile(values: list[float], q: float, default: float) -> float:
        values = sorted(values)
        return values[int(q * (len(values) - 1))] if values else default
    # Continuation lines of justified paragraphs share the margins; a few wider
    # lines (a letterhead address, a stray mark) must not move them.
    left = quantile([line.box.x0 for line in wide or every], 0.2, 0.0)
    right = quantile([line.box.x1 for line in wide], 0.8, recognized.width) if wide else recognized.width - left
    content_width = max(1.0, right - left)

    blocks: list[object] = []
    used_graphics: set[int] = set()
    flow: list[_Line] = []
    heights = [line.box.height for line in every] or [12.0]
    gaps = [b[0].box.y0 - a[0].box.y1 for a, b in zip(rows, rows[1:]) if len(a) == len(b) == 1]
    pitch = median([gap for gap in gaps if gap >= 0] or [median(heights) * 0.3])
    next_list_id = 1

    def flush() -> None:
        if not flow:
            return
        group = [flow[0]]
        for line in flow[1:]:
            if _new_paragraph(group[-1], line, group, left, right, pitch):
                blocks.append(_paragraph_from_lines(group, recognized.index, left, right, next_list_id))
                group = [line]
            else:
                group.append(line)
        blocks.append(_paragraph_from_lines(group, recognized.index, left, right, next_list_id))
        flow.clear()

    for kind, unit in _units(rows, content_width):
        if kind == "row":
            flow.append(unit[0][0])
            continue
        flush()
        slots = _slots(unit)
        if len(slots) >= 2 and not _text_columns(unit, slots):
            # Fields side by side ("____ ... «27» 05 2019 года"): one tab line per row.
            for row in unit:
                if len(row) == 1:
                    flow_row = row[0]
                    blocks.append(_paragraph_from_lines([flow_row], recognized.index, left, right, next_list_id))
                elif len(row) <= 3:
                    blocks.append(_tab_line(row, recognized.index, left))
                else:
                    blocks.append(_band_table([row], [], recognized.index))
        elif len(unit) == 1 and len(unit[0]) <= 3:
            blocks.append(_tab_line(unit[0], recognized.index, left))
        elif len(slots) >= 2:
            table = _band_table(unit, [graphic for graphic in graphics if graphic.placement == "inline" and graphic.role not in FLOATING_ROLES], recognized.index)
            used_graphics.update(id(cell.graphic) for cell in table.cells if cell.graphic is not None)
            blocks.append(table)
        else:
            flow.extend(line for row in unit for line in row)
    flush()

    # Lists: consecutive items share one Word list; deeper indentation is a nested level.
    list_id = 0
    previous_item = False
    base_x = 0.0
    for block in blocks:
        if isinstance(block, Paragraph) and block.kind == "list_item":
            if not previous_item:
                list_id += 1
                base_x = block.box.x0
            block.list_id = list_id
            step = max(18.0, 1.6 * (block.size or 12.0))
            block.level = min(8, max(0, int((block.box.x0 - base_x + 4.0) // step)))
            block.left_indent = max(0.0, base_x - left)
            previous_item = True
        else:
            previous_item = False

    body = _body_size([block for block in blocks if isinstance(block, Paragraph)])
    for block in blocks:
        if isinstance(block, Paragraph) and block.kind == "paragraph" and len(block.text) < 150 and (
            block.size >= body * 1.3 or (block.bold and block.source_lines <= 2)
        ):
            block.kind = "heading"
            block.keep_with_next = True
            first_text = block is next((item for item in blocks if isinstance(item, Paragraph)), None)
            block.level = 0 if block.size >= body * 1.35 or (first_text and block.size >= body * 1.15) else 1

    for graphic in graphics:
        if id(graphic) in used_graphics:
            continue
        beside_text = any(
            min(graphic.box.y1, line.box.y1) - max(graphic.box.y0, line.box.y0) >= 0.5 * line.box.height
            and not graphic.box.contains_center(line.box)
            for line in every
        )
        graphic.placement = "float" if graphic.role in FLOATING_ROLES or beside_text else "inline"
        blocks.append(graphic)
    blocks.extend(tables)
    blocks.sort(key=lambda block: (block.box.y0, block.box.x0))

    # Vertical space: the source gap minus the normal leading of a line (~0.2 em).
    previous_bottom: float | None = None
    previous_paragraph: Paragraph | None = None
    for block in blocks:
        if isinstance(block, Paragraph):
            if previous_bottom is not None:
                block.space_before = max(0.0, min(30.0, block.box.y0 - previous_bottom - 0.2 * (block.size or body)))
            previous_bottom = block.box.y1
            previous_paragraph = block
        elif isinstance(block, Table) or (isinstance(block, Graphic) and block.placement == "inline"):
            if previous_paragraph is not None and previous_bottom is not None and isinstance(block, Table):
                previous_paragraph.space_after = max(0.0, min(24.0, block.box.y0 - previous_bottom - 0.2 * body))
            previous_bottom = block.box.y1

    content = [block.box for block in blocks if hasattr(block, "box")]
    margin_left = left if content else None
    margin_right = recognized.width - right if content else None
    margin_top = min(box.y0 for box in content) if content else None
    margin_bottom = recognized.height - max(box.y1 for box in content) if content else None
    issues: list[StructuralIssue] = list(recognized.issues)
    if any(isinstance(block, Paragraph) and block.confidence < 0.50 for block in blocks):
        issues.append(StructuralIssue("low_text_confidence", "One or more logical text blocks have low source confidence.", "warning", 0.5))
    return Page(
        recognized.index, recognized.width, recognized.height, blocks, recognized.intentionally_blank,
        source_rotation=recognized.source_rotation, margin_left=margin_left, margin_right=margin_right,
        margin_top=margin_top, margin_bottom=margin_bottom, issues=issues, body_size=body,
    )


def understand(pages: list[RecognizedPage]) -> Document:
    return Document(
        [understand_page(page) for page in pages],
        metadata={
            "coordinate_system": "normalized-unrotated-points-top-left",
            "pagination_policy": "preserve-source-pages",
            "style_policy": "times-new-roman-word-like-layout",
        },
    )
