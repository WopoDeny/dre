"""Clean Word reconstruction from the unified model, with no OCR/PDF imports."""

from __future__ import annotations

import io
import re
import unicodedata

from docx import Document as WordDocument
from docx.enum.section import WD_ORIENT, WD_SECTION_START
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from PIL import Image

from ..model import Document, Graphic, InlineRun, Page, Paragraph, Table
from ..recognition.languages import direction as text_direction
from .fit import choose_fit, text_width

BODY_FONT = "Times New Roman"

# Strong glyph-variant evidence, not a dictionary or a text converter. These
# characters have different simplified/traditional written forms; characters
# shared by both systems cannot establish a variant on their own. Additional
# writing systems may be supported by extending font hints at the model layer.
TRADITIONAL_CJK_EVIDENCE = frozenset("編號額審繼續體國學書門臺灣與為後發務東長車網電機報資產專業會議寫頁顯讀實際數據處")
SIMPLIFIED_CJK_EVIDENCE = frozenset("编号额审继续体国学书门台湾与为后发务东长车网电机报资产专业会议写页显读实际数据处")


def _cjk_variant_hint(text: str) -> str | None:
    """Select a regional font only with unambiguous recovered glyph evidence.

    This influences DOCX typography, never the recognized Unicode text.
    Mixed simplified/traditional content intentionally has no forced variant.
    """
    traditional = any(char in TRADITIONAL_CJK_EVIDENCE for char in text)
    simplified = any(char in SIMPLIFIED_CJK_EVIDENCE for char in text)
    if traditional and not simplified:
        return "TC"
    if simplified and not traditional:
        return "SC"
    return None


def _property(parent: object, tag: str, **attributes: str) -> object:
    element = OxmlElement(tag)
    for name, value in attributes.items():
        element.set(qn(name), value)
    parent.append(element)
    return element


def _font_names(text: str, script: str | None = None, cjk_variant: str | None = None) -> tuple[str, str]:
    """Choose independent East Asian and complex-script hints for mixed runs.

    A single editable Word run may contain Han text and Arabic-script text or
    digits. Its eastAsia and cs attributes are independent: selecting the
    Arabic branch first must not suppress the required Han font hint.
    """
    has_han = script == "Han" or any("\u3400" <= character <= "\u9fff" for character in text)
    if has_han:
        variant = cjk_variant or _cjk_variant_hint(text)
        east_asia = "Noto Sans CJK TC" if variant == "TC" else "Noto Sans CJK SC"
    else:
        east_asia = BODY_FONT
    return east_asia, "Noto Naskh Arabic"


def _paragraph_direction(paragraph: object, direction: str) -> None:
    if direction == "rtl":
        _property(paragraph._p.get_or_add_pPr(), "w:bidi")
        paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT


def _append_text_run(paragraph: object, value: str, *, direction: str, bold: bool = False, italic: bool = False, size: float | None = None, script: str | None = None, cjk_variant: str | None = None, preserve_breaks: bool = False, underline: bool = False) -> None:
    if not value:
        return
    # A detector-owned multiline table cell is editable content, not a visual
    # line-wrap hint. python-docx serializes newlines as w:br and tabs as w:tab.
    # Keep the existing paragraph normalization policy outside table cells.
    run = paragraph.add_run(value if preserve_breaks else value.replace("\n", " ").replace("\t", " "))
    # Only meaningful inline emphasis overrides the coherent paragraph style.
    # OCR/native pixel-size estimates are retained in the model as evidence,
    # never imposed on individual Word runs (which fragments typography).
    # In particular, an unbolded source run must inherit a heading's bold style.
    if bold:
        run.bold = True
    if italic:
        run.italic = True
    if underline:
        run.underline = True
    properties = run._r.get_or_add_rPr()
    if direction == "rtl":
        _property(properties, "w:rtl")
    east_asia, complex_script = _font_names(value, script, cjk_variant)
    _property(properties, "w:rFonts", **{"w:ascii": BODY_FONT, "w:hAnsi": BODY_FONT, "w:eastAsia": east_asia, "w:cs": complex_script})
    if size:
        run.font.size = Pt(size)


def _write_text(paragraph: object, text: str, direction: str, *, preserve_breaks: bool = False, size: float | None = None, bold: bool = False) -> None:
    _paragraph_direction(paragraph, direction)
    # Preserve logical Unicode order. Word performs bidi resolution and shaping.
    parts: list[tuple[str, bool]] = []
    for character in text:
        if ord(character) < 32 and character not in "\t\n":
            continue
        bidi = unicodedata.bidirectional(character)
        rtl = bidi in ("R", "AL")
        if bidi not in ("R", "AL", "L") and parts:
            rtl = parts[-1][1]
        if parts and parts[-1][1] == rtl:
            parts[-1] = (parts[-1][0] + character, rtl)
        else:
            parts.append((character, rtl))
    variant = _cjk_variant_hint(text)
    for value, rtl in parts:
        _append_text_run(paragraph, value, direction="rtl" if rtl else "ltr", cjk_variant=variant, preserve_breaks=preserve_breaks, size=size, bold=bold)


def _write_runs(paragraph: object, block: Paragraph, *, text: str | None = None, size: float | None = None) -> None:
    target = block.text if text is None else text
    if not block.runs or text is not None and target != block.text:
        _write_text(paragraph, target, block.direction, preserve_breaks=block.tab_line, size=size)
        return
    _paragraph_direction(paragraph, block.direction)
    variant = _cjk_variant_hint(target)
    for source in block.runs:
        _append_text_run(
            paragraph,
            source.text,
            direction=source.direction,
            bold=source.bold,
            italic=source.italic,
            size=size,
            script=source.script,
            cjk_variant=variant,
            preserve_breaks=block.tab_line,
            underline=source.underline,
        )


def _trim_runs(runs: list[InlineRun], prefix_characters: int) -> list[InlineRun]:
    """Remove a textual prefix while preserving the surviving run styling."""
    remaining = max(0, prefix_characters)
    values: list[InlineRun] = []
    for source in runs:
        text = source.text
        if remaining >= len(text):
            remaining -= len(text)
            continue
        if remaining:
            text = text[remaining:]
            remaining = 0
        if text:
            values.append(InlineRun(
                text, source.box, source.source_page,
                source_region=source.source_region, confidence=source.confidence,
                direction=source.direction, bold=source.bold, italic=source.italic,
                size=source.size, script=source.script,
            ))
    return values


def _write_list_runs(paragraph: object, block: Paragraph, text: str, removed_prefix: int, size: float | None = None) -> None:
    if removed_prefix and block.runs:
        runs = _trim_runs(block.runs, removed_prefix)
        if runs and "".join(run.text for run in runs) == text:
            _paragraph_direction(paragraph, block.direction)
            variant = _cjk_variant_hint(text)
            for source in runs:
                _append_text_run(
                    paragraph, source.text, direction=source.direction, bold=source.bold,
                    italic=source.italic, size=size, script=source.script,
                    cjk_variant=variant, underline=source.underline,
                )
            return
    _write_runs(paragraph, block, text=text if text != block.text else None, size=size)


def _graphic_dimensions(data: bytes, max_width: float, max_height: float) -> tuple[float, float]:
    """Fit an isolated raster inside its model box without changing aspect ratio."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
    except Exception:
        return max_width, max_height
    if width <= 0 or height <= 0:
        return max_width, max_height
    scale = min(max_width / width, max_height / height)
    return max(1.0, width * scale), max(1.0, height * scale)


def _numbering(document: object, start: int, *, number_format: str = "decimal", level_text: str = "%1.", level_index: int = 0) -> int:
    numbering = document.part.numbering_part.element
    abstract_ids = [int(element.get(qn("w:abstractNumId"))) for element in numbering.findall(qn("w:abstractNum"))]
    number_ids = [int(element.get(qn("w:numId"))) for element in numbering.findall(qn("w:num"))]
    abstract_id, number_id = max(abstract_ids, default=-1) + 1, max(number_ids, default=0) + 1
    abstract = _property(numbering, "w:abstractNum", **{"w:abstractNumId": str(abstract_id)})
    _property(abstract, "w:multiLevelType", **{"w:val": "multilevel"})
    level_index = max(0, min(8, level_index))
    # A nested paragraph must point to a *complete* multilevel definition.
    # An abstractNum containing only ilvl=1 is repaired inconsistently by
    # document renderers; some display an alphabetic child as decimal "1.".
    for current_level in range(level_index + 1):
        level = _property(abstract, "w:lvl", **{"w:ilvl": str(current_level)})
        active = current_level == level_index
        _property(level, "w:start", **{"w:val": str(start if active else 1)})
        _property(level, "w:numFmt", **{"w:val": number_format if active else "decimal"})
        # OOXML placeholders are 1-based: ilvl=1 requires %2, not %1.
        actual_text = level_text.replace("%1", f"%{current_level + 1}") if active else f"%{current_level + 1}."
        _property(level, "w:lvlText", **{"w:val": actual_text})
        paragraph_properties = _property(level, "w:pPr")
        _property(paragraph_properties, "w:ind", **{"w:left": str(360 * (current_level + 1)), "w:hanging": "360"})
    number = _property(numbering, "w:num", **{"w:numId": str(number_id)})
    _property(number, "w:abstractNumId", **{"w:val": str(abstract_id)})
    return number_id


EMU_PER_PT = 12700
LIST_MARKER = re.compile(r"^\s*((?:[•●▪*\-–])|(?:\(?\d{1,3}[.)])|(?:[A-Za-z][.)]))\s+")
ALIGNMENTS = {"left": WD_ALIGN_PARAGRAPH.LEFT, "center": WD_ALIGN_PARAGRAPH.CENTER,
              "right": WD_ALIGN_PARAGRAPH.RIGHT, "justify": WD_ALIGN_PARAGRAPH.JUSTIFY}


def _size(value: float) -> float:
    return max(7.0, min(28.0, round(value * 2) / 2))


def _margins(page: Page) -> tuple[float, float, float, float]:
    clamp = lambda value, low, high, fallback: fallback if value is None else max(low, min(high, value))
    return (clamp(page.margin_left, 42.0, 100.0, 72.0), clamp(page.margin_right, 28.0, 72.0, 42.0),
            clamp(page.margin_top, 28.0, 72.0, 42.0), clamp(page.margin_bottom, 28.0, 72.0, 42.0))


def _tab_stops(paragraph: object, block: Paragraph, width: float) -> None:
    stops = paragraph.paragraph_format.tab_stops
    if block.text.count("\t") >= 2:
        stops.add_tab_stop(Pt(width / 2), WD_TAB_ALIGNMENT.CENTER)
    stops.add_tab_stop(Pt(width - 1), WD_TAB_ALIGNMENT.RIGHT)


def _floating(run: object, x: float, y: float) -> None:
    """Turn the inline picture of `run` into one drawn behind the text at
    (x from the page edge, y from the anchor paragraph), like a real signature."""
    drawing = run._r.find(qn("w:drawing"))
    inline = drawing.find(qn("wp:inline"))
    anchor = OxmlElement("wp:anchor")
    for name, value in (("distT", "0"), ("distB", "0"), ("distL", "0"), ("distR", "0"), ("simplePos", "0"),
                        ("relativeHeight", "251659264"), ("behindDoc", "1"), ("locked", "0"),
                        ("layoutInCell", "1"), ("allowOverlap", "1")):
        anchor.set(name, value)
    simple = OxmlElement("wp:simplePos")
    simple.set("x", "0")
    simple.set("y", "0")
    anchor.append(simple)
    for tag, relative, offset in (("wp:positionH", "page", x), ("wp:positionV", "paragraph", y)):
        position = OxmlElement(tag)
        position.set("relativeFrom", relative)
        value = OxmlElement("wp:posOffset")
        value.text = str(int(offset * EMU_PER_PT))
        position.append(value)
        anchor.append(position)
    anchor.append(inline.find(qn("wp:extent")))
    effect = OxmlElement("wp:effectExtent")
    for side in ("l", "t", "r", "b"):
        effect.set(side, "0")
    anchor.append(effect)
    anchor.append(OxmlElement("wp:wrapNone"))
    for tag in ("wp:docPr", "wp:cNvGraphicFramePr"):
        element = inline.find(qn(tag))
        if element is not None:
            anchor.append(element)
    anchor.append(inline.find(qn("a:graphic")))
    drawing.remove(inline)
    drawing.append(anchor)


def _borders(table: object, value: str) -> None:
    borders = _property(table._tbl.tblPr, "w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        if value == "none":
            _property(borders, f"w:{edge}", **{"w:val": "nil"})
        else:
            _property(borders, f"w:{edge}", **{"w:val": "single", "w:sz": "4", "w:color": "000000"})


def _write_table(document: object, block: Table, width: float, body: float, scale: float, spacing: float) -> None:
    table = document.add_table(rows=block.rows, cols=block.columns)
    table.autofit = False
    widths = block.column_widths or [1.0] * block.columns
    widths = [width * value / max(1.0, sum(widths)) for value in widths]
    for column, column_width in zip(table.columns, widths):
        column.width = Pt(column_width)
    size = _size((body - (0 if block.borderless else 1)) * scale)
    for row in table.rows:
        for position, cell in enumerate(row.cells):
            cell.width = Pt(widths[position])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP if block.borderless else WD_CELL_VERTICAL_ALIGNMENT.CENTER
    for source in block.cells:
        cell = table.cell(source.row, source.column)
        if source.row_span > 1 or source.column_span > 1:
            cell = cell.merge(table.cell(source.row + source.row_span - 1, source.column + source.column_span - 1))
        cell.text = ""
        paragraph = cell.paragraphs[0]
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.line_spacing = spacing
        paragraph.alignment = ALIGNMENTS.get(source.alignment, WD_ALIGN_PARAGRAPH.LEFT)
        if source.graphic is not None:
            graphic = source.graphic
            picture_width = min(graphic.box.width * scale, widths[source.column] - 6)
            picture_height = graphic.box.height * picture_width / max(1.0, graphic.box.width)
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            picture = paragraph.add_run().add_picture(io.BytesIO(graphic.data), width=Pt(picture_width), height=Pt(picture_height))
            picture._inline.docPr.set("descr", graphic.description)
            continue
        if source.runs and "".join(run.text for run in source.runs) == source.text:
            _paragraph_direction(paragraph, source.direction)
            variant = _cjk_variant_hint(source.text)
            for run in source.runs:
                _append_text_run(paragraph, run.text, direction=run.direction, bold=run.bold or source.bold,
                                 italic=run.italic, size=size, script=run.script, cjk_variant=variant, underline=run.underline)
        elif block.borderless:
            # Layout cells hold reflowed paragraphs, one Word paragraph each.
            for number, part in enumerate(source.text.split("\n")):
                if number:
                    paragraph = cell.add_paragraph()
                    paragraph.paragraph_format.space_after = Pt(0)
                    paragraph.paragraph_format.line_spacing = spacing
                    paragraph.alignment = ALIGNMENTS.get(source.alignment, WD_ALIGN_PARAGRAPH.LEFT)
                _write_text(paragraph, part, text_direction(part), size=size, bold=source.bold)
        else:
            _write_text(paragraph, source.text, source.direction, preserve_breaks=True, size=size, bold=source.bold)
    _borders(table, "none" if block.borderless else "single")


def reconstruct(model: Document) -> bytes:
    document = WordDocument()
    for field_name in ("author", "last_modified_by", "title", "comments"):
        setattr(document.core_properties, field_name, "")
    normal = document.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = Pt(_size(model.pages[0].body_size if model.pages else 12))
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.widow_control = True
    _property(normal.element.get_or_add_rPr(), "w:rFonts", **{"w:ascii": BODY_FONT, "w:hAnsi": BODY_FONT, "w:cs": BODY_FONT, "w:eastAsia": BODY_FONT})
    for list_name in ("List Number", "List Bullet"):
        document.styles[list_name].font.name = BODY_FONT
    # A small set of styles: body (Normal), two heading levels.
    for name in ("Heading", "Subheading"):
        style = document.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        style.base_style = normal
        style.font.name = BODY_FONT
        style.font.bold = True
        style.paragraph_format.keep_with_next = True
        _property(style.element.get_or_add_pPr(), "w:outlineLvl", **{"w:val": "0" if name == "Heading" else "1"})

    for index, page in enumerate(model.pages):
        section = document.sections[0] if index == 0 else document.add_section(WD_SECTION_START.NEW_PAGE)
        section.page_width, section.page_height = Pt(page.width), Pt(page.height)
        section.orientation = WD_ORIENT.LANDSCAPE if page.width > page.height else WD_ORIENT.PORTRAIT
        left, right, top, bottom = _margins(page)
        section.left_margin, section.right_margin = Pt(left), Pt(right)
        section.top_margin, section.bottom_margin = Pt(top), Pt(bottom)
        section.header_distance = section.footer_distance = Pt(12)
        width = page.width - left - right
        scale, spacing = choose_fit(page, width, page.height - top - bottom)
        body = page.body_size
        if page.intentionally_blank and not page.blocks:
            document.add_paragraph("")
        numbering_ids: dict[tuple[int | None, int, str, str], int] = {}
        numbering_last: dict[tuple[int | None, int, str, str], int] = {}
        anchor: tuple[object, Paragraph] | None = None
        pending: list[Graphic] = []

        def place(graphic: Graphic, paragraph: object, source: Paragraph) -> None:
            picture_width = min(graphic.box.width, page.width * 0.5)
            picture_height = graphic.box.height * picture_width / max(1.0, graphic.box.width)
            run = paragraph.add_run()
            picture = run.add_picture(io.BytesIO(graphic.data), width=Pt(picture_width), height=Pt(picture_height))
            picture._inline.docPr.set("descr", graphic.description)
            _floating(run, graphic.box.x0, (graphic.box.y0 - source.box.y0) * scale)

        for block in page.blocks:
            if isinstance(block, Paragraph):
                paragraph = document.add_paragraph(style=("Heading" if block.level == 0 else "Subheading") if block.kind == "heading" else None)
                layout = paragraph.paragraph_format
                layout.line_spacing = spacing
                layout.space_before = Pt(block.space_before * scale)
                layout.space_after = Pt(block.space_after * scale)
                layout.keep_with_next = block.keep_with_next or None
                heading = block.kind == "heading"
                size = _size((block.size or (body * 1.2 if heading else body)) * scale)
                if block.source_lines == 1 and block.kind != "list_item":
                    # A line that was one line in the source stays one line in Word.
                    room = width - max(0.0, block.left_indent) - max(0.0, block.first_line_indent) - 2
                    measured = text_width(block.text.replace("\t", "    "), size, block.bold)
                    if measured > room:
                        size = max(7.0, _size(size * room / measured) - 0.5)
                text = block.text
                removed_prefix = 0
                if block.kind == "list_item":
                    match = LIST_MARKER.match(text)
                    marker = block.list_marker or (match.group(1) if match else "")
                    numeric = re.match(r"^\(?(\d+)[.)]$", marker)
                    alpha = re.match(r"^([A-Za-z])[.)]$", marker)
                    if numeric:
                        value, number_format = int(numeric.group(1)), "decimal"
                        level_text = "(%1)" if marker.startswith("(") else "%1)" if marker.endswith(")") else "%1."
                    elif alpha:
                        value = ord(alpha.group(1).lower()) - ord("a") + 1
                        number_format = "upperLetter" if alpha.group(1).isupper() else "lowerLetter"
                        level_text = "%1)" if marker.endswith(")") else "%1."
                    else:
                        value, number_format, level_text = 1, "bullet", "•"
                    level = max(0, min(8, block.level))
                    key = (block.list_id, level, number_format, level_text)
                    number_id = numbering_ids.get(key)
                    if number_id is None or (number_format != "bullet" and numbering_last.get(key) is not None and value != numbering_last[key] + 1):
                        number_id = _numbering(document, value, number_format=number_format, level_text=level_text, level_index=level)
                        numbering_ids[key] = number_id
                    numbering_last[key] = value
                    properties = _property(paragraph._p.get_or_add_pPr(), "w:numPr")
                    _property(properties, "w:ilvl", **{"w:val": str(level)})
                    _property(properties, "w:numId", **{"w:val": str(number_id)})
                    layout.left_indent = Pt(block.left_indent + 18 * (level + 1))
                    layout.first_line_indent = Pt(-18)
                    if match:
                        removed_prefix = match.end()
                        text = text[removed_prefix:]
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY if block.source_lines > 1 else WD_ALIGN_PARAGRAPH.LEFT
                    _write_list_runs(paragraph, block, text, removed_prefix, size=size)
                else:
                    if block.direction != "rtl":
                        paragraph.alignment = ALIGNMENTS.get(block.alignment, WD_ALIGN_PARAGRAPH.LEFT)
                    if block.left_indent > 0:
                        layout.left_indent = Pt(min(block.left_indent, width * 0.6))
                    if block.first_line_indent and paragraph.alignment != WD_ALIGN_PARAGRAPH.CENTER:
                        layout.first_line_indent = Pt(max(-block.left_indent, min(block.first_line_indent, width * 0.6)))
                    if block.tab_line:
                        _tab_stops(paragraph, block, width - max(0.0, block.left_indent))
                    _write_runs(paragraph, block, size=size)
                if block.bold or heading:
                    for run in paragraph.runs:
                        run.bold = True
                anchor = (paragraph, block)
                for graphic in pending:
                    place(graphic, paragraph, block)
                pending.clear()
            elif isinstance(block, Table):
                _write_table(document, block, width, body, scale, spacing)
            elif isinstance(block, Graphic):
                if block.box.area > page.width * page.height * 0.35:
                    raise ValueError("Full-page or oversized source rasters are not permitted in DOCX.")
                if block.placement == "float":
                    if anchor is not None:
                        place(block, anchor[0], anchor[1])
                    else:
                        pending.append(block)
                    continue
                paragraph = document.add_paragraph()
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                picture_width, picture_height = _graphic_dimensions(block.data, min(block.box.width * scale, width), block.box.height * scale)
                picture = paragraph.add_run().add_picture(io.BytesIO(block.data), width=Pt(picture_width), height=Pt(picture_height))
                picture._inline.docPr.set("descr", block.description)
        if pending:
            paragraph = document.add_paragraph()
            for graphic in pending:
                place(graphic, paragraph, Paragraph("", graphic.box, page.index))
    buffer = io.BytesIO()
    document.save(buffer)
    return _stable_zip(buffer.getvalue())


def _stable_zip(data: bytes) -> bytes:
    """The same document gives the same bytes: zip entries carry a fixed timestamp."""
    import zipfile
    source = zipfile.ZipFile(io.BytesIO(data))
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as target:
        for item in source.infolist():
            entry = zipfile.ZipInfo(item.filename, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = item.external_attr
            target.writestr(entry, source.read(item.filename))
    return output.getvalue()
