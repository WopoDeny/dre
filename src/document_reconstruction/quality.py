"""Fast DOCX package and semantic checks; rendered pagination remains an offline gate."""

from __future__ import annotations

import io
import posixpath
import re
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from xml.etree import ElementTree as ET

from PIL import Image

from .recognition.languages import direction as text_direction
from .model import Document, Graphic, Paragraph, Table

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
PKG_CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
DRAW_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
MAIN_DOCUMENT_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
IMAGE_CONTENT_TYPES = {
    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
    "gif": "image/gif", "bmp": "image/bmp", "tif": "image/tiff", "tiff": "image/tiff",
}
NS = {"w": W_NS, "r": R_NS}


@dataclass
class QualityReport:
    passed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    paragraphs: int = 0
    tables: int = 0
    images: int = 0
    hard_line_breaks: int = 0
    used_styles: int = 0
    logical_pages: int = 0
    rendered_pages: int | None = None
    text_coverage: float = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _expected_text(model: Document) -> str:
    values: list[str] = []
    for page in model.pages:
        for block in page.blocks:
            if isinstance(block, Paragraph):
                value = block.text
                if block.kind == "list_item":
                    value = re.sub(r"^\s*(?:[\u2022\u25cf\u25aa*\-\u2013]|\(?\d{1,3}[.)]|[A-Za-z][.)])\s+", "", value)
                values.append(value)
            elif isinstance(block, Table):
                for cell in sorted(block.cells, key=lambda item: (item.row, item.column)):
                    values.append(cell.text)
    return _normalized_text(" ".join(values))


def _rels_owner(name: str) -> str:
    path = PurePosixPath(name)
    if path.name == ".rels" and str(path.parent) == "_rels":
        return ""
    # word/_rels/document.xml.rels -> word/document.xml
    parent = path.parent.parent
    owner_name = path.name[:-5]  # remove .rels
    return str(parent / owner_name)


def _validate_relationships(package: zipfile.ZipFile, names: set[str], failures: list[str]) -> None:
    root_office_document = False
    for name in sorted(item for item in names if item.endswith(".rels")):
        try:
            root = ET.fromstring(package.read(name))
        except (KeyError, ET.ParseError):
            failures.append(f"invalid_relationship_part:{name}")
            continue
        owner = _rels_owner(name)
        base = posixpath.dirname(owner)
        for rel in root.findall(f"{{{PKG_REL_NS}}}Relationship"):
            target = rel.get("Target", "")
            mode = rel.get("TargetMode", "Internal")
            rel_type = rel.get("Type", "")
            if name == "_rels/.rels" and rel_type.endswith("/officeDocument"):
                root_office_document = True
            if mode == "External":
                # Reconstructed outputs are self-contained. Remote assets are not
                # permitted, even when Word could fetch them after conversion.
                failures.append(f"external_relationship:{name}")
                continue
            if not target:
                continue
            resolved = posixpath.normpath(posixpath.join(base, target)).lstrip("/")
            if resolved.startswith("../") or resolved not in names:
                failures.append(f"broken_relationship:{name}:{target}")
    if not root_office_document:
        failures.append("missing_office_document_relationship")


def _validate_content_types(content_types: ET.Element, media: list[str], failures: list[str]) -> None:
    """Check that every emitted raster has a declared, matching package media type."""
    if content_types.tag != f"{{{PKG_CT_NS}}}Types":
        failures.append("invalid_content_types_root")
        return
    defaults: dict[str, str] = {}
    overrides: dict[str, str] = {}
    for element in content_types:
        if element.tag == f"{{{PKG_CT_NS}}}Default":
            extension, mime = element.get("Extension", "").lower(), element.get("ContentType", "")
            if extension in defaults and defaults[extension] != mime:
                failures.append(f"conflicting_default_content_type:{extension}")
            defaults[extension] = mime
        elif element.tag == f"{{{PKG_CT_NS}}}Override":
            part, mime = element.get("PartName", ""), element.get("ContentType", "")
            if part in overrides and overrides[part] != mime:
                failures.append(f"conflicting_override_content_type:{part}")
            overrides[part] = mime
    if overrides.get("/word/document.xml") != MAIN_DOCUMENT_CONTENT_TYPE:
        failures.append("invalid_main_document_content_type")
    if defaults.get("rels") != "application/vnd.openxmlformats-package.relationships+xml":
        failures.append("invalid_relationship_content_type")
    for name in media:
        extension = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        declared = overrides.get("/" + name, defaults.get(extension))
        if not declared:
            failures.append(f"missing_image_content_type:{name}")
        elif declared != IMAGE_CONTENT_TYPES.get(extension):
            failures.append(f"invalid_image_content_type:{name}")


def _word_property_enabled(properties: ET.Element | None, property_name: str) -> bool:
    """Interpret OOXML on/off properties, including explicit w:val="0"."""
    node = properties.find(f"w:{property_name}", NS) if properties is not None else None
    return node is not None and node.get(f"{{{W_NS}}}val", "true").lower() not in (
        "0", "false", "off",
    )


def _validate_drawing_ownership(root: ET.Element, rels: ET.Element | None,
                                media: list[str], graphics: list[Graphic],
                                package: zipfile.ZipFile, failures: list[str]) -> int:
    """Count placed graphics, and require each to reference a local media part."""
    drawings = root.findall(".//w:drawing", NS)
    if len(drawings) != len(graphics):
        failures.append("image_drawing_count_mismatch")
    image_rels: dict[str, str] = {}
    if rels is not None:
        for relationship in rels.findall(f"{{{PKG_REL_NS}}}Relationship"):
            if relationship.get("Type", "").endswith("/image") and relationship.get("TargetMode", "Internal") != "External":
                image_rels[relationship.get("Id", "")] = posixpath.normpath(posixpath.join(
                    "word", relationship.get("Target", "")
                ))
    referenced: set[str] = set()
    placed: list[bytes] = []
    for drawing in drawings:
        images = drawing.findall(f".//{{{DRAW_NS}}}blip")
        if len(images) != 1:
            failures.append("invalid_drawing_image_reference")
            continue
        embed = images[0].get(f"{{{R_NS}}}embed", "")
        target = image_rels.get(embed)
        if target not in media:
            failures.append("invalid_drawing_image_reference")
        else:
            referenced.add(target)
            placed.append(package.read(target))
    if referenced != set(media):
        failures.append("orphaned_embedded_image")
    # python-docx embeds each isolated graphic unchanged (floating pictures are
    # attached to a paragraph, so the order may differ from the model order).
    if sorted(placed) != sorted(graphic.data for graphic in graphics):
        failures.append("embedded_graphic_content_mismatch")
    return len(drawings)


def verify(docx: bytes, model: Document) -> QualityReport:
    report = QualityReport(False)
    try:
        with zipfile.ZipFile(io.BytesIO(docx)) as package:
            bad = package.testzip()
            if bad is not None:
                raise ValueError(f"Corrupt package member: {bad}")
            member_names = package.namelist()
            names = set(member_names)
            if len(member_names) != len(names):
                # Ambiguous duplicate names can hide a different XML part from readers.
                report.failures.extend(
                    f"duplicate_zip_member:{name}"
                    for name, count in sorted(Counter(member_names).items()) if count > 1
                )
                return report
            required = {"[Content_Types].xml", "_rels/.rels", "word/document.xml", "word/styles.xml"}
            missing = required - names
            if missing:
                report.failures.extend(f"missing_required_part:{name}" for name in sorted(missing))
                return report
            parsed: dict[str, ET.Element] = {}
            for name in sorted(item for item in names if item.endswith(".xml") or item.endswith(".rels")):
                try:
                    parsed[name] = ET.fromstring(package.read(name))
                except ET.ParseError:
                    report.failures.append(f"invalid_xml:{name}")
            if report.failures:
                return report
            root = parsed["word/document.xml"]
            _validate_relationships(package, names, report.failures)

            media = sorted(name for name in names if name.startswith("word/media/") and not name.endswith("/"))
            _validate_content_types(parsed["[Content_Types].xml"], media, report.failures)
            expected_graphics = [block for page in model.pages for block in page.blocks if isinstance(block, Graphic)] + [
                cell.graphic for page in model.pages for block in page.blocks if isinstance(block, Table)
                for cell in block.cells if cell.graphic is not None
            ]
            report.images = _validate_drawing_ownership(
                root, parsed.get("word/_rels/document.xml.rels"), media,
                expected_graphics, package, report.failures
            )
            for name in media:
                try:
                    with Image.open(io.BytesIO(package.read(name))) as image:
                        width, height = image.size
                        if width <= 0 or height <= 0:
                            report.failures.append(f"invalid_image_dimensions:{name}")
                        image.verify()
                except Exception:
                    report.failures.append(f"invalid_embedded_image:{name}")
    except (zipfile.BadZipFile, KeyError, ValueError):
        report.failures.append("invalid_docx_package")
        return report

    report.paragraphs = len(root.findall(".//w:p", NS))
    report.tables = len(root.findall(".//w:tbl", NS))
    report.hard_line_breaks = len(root.findall(".//w:br", NS))
    report.used_styles = len({element.get(f"{{{W_NS}}}val") for element in root.findall(".//w:pStyle", NS)})
    report.logical_pages = len(model.pages)

    # Hidden Word text remains in w:t and passes text/accounting comparisons,
    # yet can disappear in ordinary Word/LibreOffice reading and print views.
    # The reconstruction writer intentionally emits no hidden body content.
    if any(_word_property_enabled(node, property_name)
           for node in root.findall(".//w:rPr", NS) + root.findall(".//w:pPr", NS)
           for property_name in ("vanish", "webHidden")):
        report.failures.append("hidden_word_content")

    # Include editable Word line breaks in the global text ledger. In
    # particular, a line break inside an editable table cell is source text,
    # not an extraneous layout artifact.
    paragraph_texts = [
        "".join(
            (node.text or "") if node.tag == f"{{{W_NS}}}t" else
            "\n" if node.tag in (f"{{{W_NS}}}br", f"{{{W_NS}}}cr") and node.get(f"{{{W_NS}}}type") != "column" else
            "\t" if node.tag == f"{{{W_NS}}}tab" else ""
            for node in paragraph.iter()
        )
        for paragraph in root.findall(".//w:p", NS)
    ]
    actual_text = _normalized_text(" ".join(paragraph_texts))
    expected_text = _expected_text(model)
    if expected_text:
        # Coverage is an ordered-prefix-independent diagnostic; equality below is the gate.
        matches = sum(1 for a, b in zip(actual_text, expected_text) if a == b)
        report.text_coverage = matches / max(1, len(expected_text))
    else:
        report.text_coverage = 1.0
    if actual_text != expected_text:
        report.failures.append("text_missing_duplicated_or_reordered")

    expected_tables = sum(isinstance(block, Table) for page in model.pages for block in page.blocks)
    if report.tables != expected_tables:
        report.failures.append("table_count_mismatch")
    if report.images != len(expected_graphics):
        report.failures.append("image_count_mismatch")
    # One Word section per source page: a source page never becomes two.
    if len(root.findall(".//w:sectPr", NS)) != len(model.pages):
        report.failures.append("source_section_count_mismatch")
    # Line breaks exist only inside layout/table cells; paragraphs reflow in Word.
    expected_cell_breaks = sum(
        cell.text.count("\n")
        for page in model.pages for block in page.blocks if isinstance(block, Table) and not block.borderless
        for cell in block.cells
    )
    if report.hard_line_breaks != expected_cell_breaks:
        report.failures.append("unexpected_hard_line_breaks")
    if report.used_styles > 8:
        report.failures.append("style_fragmentation")

    # The model has no table-header semantic today; the writer must not invent one.
    if root.findall(".//w:tblHeader", NS):
        report.failures.append("invented_table_header")
    if root.findall(".//w:trHeight", NS):
        report.failures.append("fixed_table_row_height")

    expected_rtl = 0
    needs_arabic_font = False
    needs_cjk_font = False
    for page in model.pages:
        if not page.blocks and not page.intentionally_blank:
            report.failures.append(f"empty_page:{page.index}")
        for block in page.blocks:
            if isinstance(block, Paragraph):
                expected_rtl += block.direction == "rtl"
                needs_arabic_font |= block.direction == "rtl" or any("\u0600" <= char <= "\u06ff" for char in block.text)
                needs_cjk_font |= any("\u3400" <= char <= "\u9fff" for char in block.text)
                if "\ufffd" in block.text:
                    report.failures.append(f"replacement_characters:{page.index}")
            elif isinstance(block, Table):
                for cell in block.cells:
                    if block.borderless:
                        expected_rtl += sum(text_direction(part) == "rtl" for part in cell.text.split("\n")) if cell.graphic is None else 0
                    else:
                        expected_rtl += cell.direction == "rtl"
                    needs_arabic_font |= cell.direction == "rtl" or any("\u0600" <= char <= "\u06ff" for char in cell.text)
                    needs_cjk_font |= any("\u3400" <= char <= "\u9fff" for char in cell.text)
            elif isinstance(block, Graphic) and block.box.area > page.width * page.height * 0.35:
                report.failures.append(f"oversized_raster:{page.index}")

    if len(root.findall(".//w:bidi", NS)) != expected_rtl:
        report.failures.append("rtl_paragraph_mismatch")
    xml_text = ET.tostring(root, encoding="unicode")
    if needs_arabic_font and "Noto Naskh Arabic" not in xml_text:
        report.failures.append("missing_complex_script_font_hint")
    if needs_cjk_font and "Noto Sans CJK" not in xml_text:
        report.failures.append("missing_cjk_font_hint")

    for paragraph in [block for page in model.pages for block in page.blocks if isinstance(block, Paragraph)]:
        if paragraph.source_lines == 1 and len(paragraph.text) > 350:
            report.warnings.append("long_single_observation_paragraph")
    report.warnings.append("rendered_pagination_requires_offline_review")
    report.passed = not report.failures
    return report
