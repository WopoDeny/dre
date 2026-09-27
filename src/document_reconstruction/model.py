"""Recognition-independent and DOCX-independent unified document model.

All geometry is expressed in points in the normalized, unrotated page coordinate
system with a top-left origin. Text is stored in logical Unicode order. The model
carries enough source provenance and structural uncertainty to make reconstruction
policy explicit without exposing OCR- or PDF-specific APIs to the DOCX layer.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Literal

TextDirection = Literal["ltr", "rtl"]
Alignment = Literal["left", "center", "right", "justify", "unknown"]
IssueSeverity = Literal["info", "warning", "error"]
GraphicRole = Literal["graphic", "signature", "stamp", "seal", "logo", "emblem", "diagram", "photo"]


@dataclass(frozen=True)
class Box:
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.x0, self.y0, self.x1, self.y1)) or self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("A box must have finite, ordered coordinates.")

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    def contains_center(self, other: Box) -> bool:
        return self.x0 <= (other.x0 + other.x1) / 2 <= self.x1 and self.y0 <= (other.y0 + other.y1) / 2 <= self.y1

    @classmethod
    def enclosing(cls, boxes: list[Box]) -> Box:
        if not boxes:
            raise ValueError("At least one box is required.")
        return cls(min(b.x0 for b in boxes), min(b.y0 for b in boxes), max(b.x1 for b in boxes), max(b.y1 for b in boxes))


@dataclass(frozen=True)
class SourceRef:
    """A source relationship retained without exposing recognition internals."""

    page: int
    region: str = "page"
    box: Box | None = None


@dataclass(frozen=True)
class StructuralIssue:
    code: str
    message: str
    severity: IssueSeverity = "warning"
    confidence: float = 0.0
    box: Box | None = None


@dataclass
class InlineRun:
    text: str
    box: Box
    source_page: int
    confidence: float = 1.0
    direction: TextDirection = "ltr"
    bold: bool = False
    italic: bool = False
    size: float = 11.0
    script: str | None = None
    source_region: str = "page"
    underline: bool = False

    @property
    def provenance(self) -> SourceRef:
        return SourceRef(self.source_page, self.source_region, self.box)


@dataclass(frozen=True)
class ColumnRelationship:
    index: int
    box: Box
    reading_order: int
    direction: TextDirection = "ltr"
    relationship: Literal["parallel", "spanning"] = "parallel"


@dataclass
class Paragraph:
    text: str
    box: Box
    source_page: int
    kind: Literal["paragraph", "heading", "list_item", "footer", "page_number"] = "paragraph"
    confidence: float = 1.0
    direction: TextDirection = "ltr"
    level: int = 0
    column: int = 0
    source_lines: int = 1
    runs: list[InlineRun] = field(default_factory=list)
    alignment: Alignment = "unknown"
    list_marker: str | None = None
    list_ordered: bool | None = None
    list_id: int | None = None
    first_line_indent: float = 0.0
    issues: list[StructuralIssue] = field(default_factory=list)
    # Typography chosen by the layout step (points; 0 = page body size).
    size: float = 0.0
    bold: bool = False
    left_indent: float = 0.0
    space_before: float = 0.0
    space_after: float = 0.0
    # Text holds one or two tab characters: "left\tright" or "left\tcenter\tright".
    tab_line: bool = False
    keep_with_next: bool = False

    @property
    def provenance(self) -> list[SourceRef]:
        if self.runs:
            refs: list[SourceRef] = []
            seen: set[tuple[int, str, Box]] = set()
            for run in self.runs:
                key = (run.source_page, run.source_region, run.box)
                if key not in seen:
                    seen.add(key)
                    refs.append(run.provenance)
            return refs
        return [SourceRef(self.source_page, "page", self.box)]


@dataclass
class Cell:
    text: str
    row: int
    column: int
    row_span: int = 1
    column_span: int = 1
    direction: TextDirection = "ltr"
    box: Box | None = None
    source_page: int | None = None
    source_region: str = "page"
    confidence: float = 1.0
    runs: list[InlineRun] = field(default_factory=list)
    issues: list[StructuralIssue] = field(default_factory=list)
    alignment: Alignment = "left"
    bold: bool = False
    # A picture inside a layout cell (the emblem between two letterhead columns).
    graphic: "Graphic | None" = None

    @property
    def provenance(self) -> SourceRef | None:
        return SourceRef(self.source_page, self.source_region, self.box) if self.source_page is not None else None


@dataclass(frozen=True)
class TableRow:
    index: int
    cells: tuple[Cell, ...]


@dataclass
class Table:
    box: Box
    source_page: int
    rows: int
    columns: int
    cells: list[Cell]
    column_widths: list[float] = field(default_factory=list)
    kind: str = "table"
    confidence: float = 1.0
    source_region: str = "page"
    issues: list[StructuralIssue] = field(default_factory=list)
    # A layout table (bilingual letterhead, side-by-side blocks) has no borders.
    borderless: bool = False

    @property
    def row_items(self) -> list[TableRow]:
        return [TableRow(index, tuple(cell for cell in self.cells if cell.row == index)) for index in range(self.rows)]

    @property
    def provenance(self) -> SourceRef:
        return SourceRef(self.source_page, self.source_region, self.box)


@dataclass
class Graphic:
    box: Box
    source_page: int
    data: bytes
    media_type: str = "image/png"
    kind: str = "graphic"
    description: str = "Document graphic"
    role: GraphicRole = "graphic"
    confidence: float = 1.0
    source_region: str = "page"
    issues: list[StructuralIssue] = field(default_factory=list)
    # "float": drawn over the text at its source position (signatures, stamps);
    # "inline": an own centered paragraph (emblems, logos, photos).
    placement: str = "inline"

    @property
    def provenance(self) -> SourceRef:
        return SourceRef(self.source_page, self.source_region, self.box)


Block = Paragraph | Table | Graphic


@dataclass
class Page:
    index: int
    width: float
    height: float
    blocks: list[Block]
    intentionally_blank: bool = False
    source_rotation: int = 0
    columns: list[ColumnRelationship] = field(default_factory=list)
    margin_left: float | None = None
    margin_right: float | None = None
    margin_top: float | None = None
    margin_bottom: float | None = None
    issues: list[StructuralIssue] = field(default_factory=list)
    body_size: float = 12.0


@dataclass
class Document:
    pages: list[Page]
    schema_version: int = 2
    metadata: dict[str, str] = field(default_factory=dict)
    issues: list[StructuralIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        pages = []
        for page in self.pages:
            blocks = []
            for block in page.blocks:
                value = asdict(block)
                if isinstance(block, Graphic):
                    value.pop("data")
                    value["data_bytes"] = len(block.data)
                blocks.append(value)
            page_value = asdict(page)
            page_value["blocks"] = blocks
            pages.append(page_value)
        return {
            "schema_version": self.schema_version,
            "metadata": dict(self.metadata),
            "issues": [asdict(issue) for issue in self.issues],
            "pages": pages,
        }

    def text(self) -> str:
        values = []
        for page in self.pages:
            for block in page.blocks:
                if isinstance(block, Paragraph):
                    values.append(block.text)
                elif isinstance(block, Table):
                    values.extend(cell.text for cell in block.cells)
        return "\n".join(values)

    def integrity_issues(self, *, tolerance: float = 2.0) -> list[StructuralIssue]:
        """Return recognition-independent model invariant violations.

        This is deliberately stricter than semantic confidence. Ambiguous structure
        belongs in ``issues`` with a warning/info severity; contradictions such as
        lost run text, invalid spans, or provenance pointing to another page are
        model errors and must not be hidden by the DOCX writer.
        """

        issues: list[StructuralIssue] = []
        page_indices = [page.index for page in self.pages]
        if len(set(page_indices)) != len(page_indices):
            issues.append(StructuralIssue("duplicate_page_index", "Document pages must have unique source indices.", "error", 1.0))

        def inside(box: Box, page: Page) -> bool:
            return (
                box.x0 >= -tolerance
                and box.y0 >= -tolerance
                and box.x1 <= page.width + tolerance
                and box.y1 <= page.height + tolerance
            )

        for page in self.pages:
            if not math.isfinite(page.width) or not math.isfinite(page.height) or page.width <= 0 or page.height <= 0:
                issues.append(StructuralIssue("invalid_page_geometry", "Page dimensions must be finite and positive.", "error", 1.0))
                continue
            column_indices = [column.index for column in page.columns]
            reading_orders = [column.reading_order for column in page.columns]
            if len(set(column_indices)) != len(column_indices) or len(set(reading_orders)) != len(reading_orders):
                issues.append(StructuralIssue("invalid_column_relationships", "Column indices and reading-order positions must be unique within a page.", "error", 1.0))
            for column in page.columns:
                if not inside(column.box, page):
                    issues.append(StructuralIssue("column_outside_page", "A normalized column relationship extends outside the page.", "error", 1.0, column.box))

            for block in page.blocks:
                if block.source_page != page.index:
                    issues.append(StructuralIssue("block_provenance_mismatch", "A block points to a different source page than its containing page.", "error", 1.0, block.box))
                if not inside(block.box, page):
                    issues.append(StructuralIssue("block_outside_page", "A normalized block extends outside the page.", "error", 1.0, block.box))
                if isinstance(block, Paragraph):
                    if block.runs and "".join(run.text for run in block.runs) != block.text:
                        issues.append(StructuralIssue("paragraph_run_text_mismatch", "Inline runs do not reproduce the paragraph text exactly.", "error", 1.0, block.box))
                    for run in block.runs:
                        if run.source_page != page.index:
                            issues.append(StructuralIssue("run_provenance_mismatch", "An inline run points to a different source page than its paragraph.", "error", 1.0, run.box))
                        if not inside(run.box, page) or not block.box.contains_center(run.box):
                            issues.append(StructuralIssue("run_geometry_mismatch", "An inline run is not geometrically owned by its paragraph.", "error", 1.0, run.box))
                elif isinstance(block, Table):
                    occupied: set[tuple[int, int]] = set()
                    for cell in block.cells:
                        if cell.row < 0 or cell.column < 0 or cell.row_span < 1 or cell.column_span < 1 or cell.row + cell.row_span > block.rows or cell.column + cell.column_span > block.columns:
                            issues.append(StructuralIssue("invalid_table_span", "A table cell span extends outside the table grid.", "error", 1.0, cell.box or block.box))
                            continue
                        coordinates = {(row, column) for row in range(cell.row, cell.row + cell.row_span) for column in range(cell.column, cell.column + cell.column_span)}
                        if occupied.intersection(coordinates):
                            issues.append(StructuralIssue("overlapping_table_cells", "Two table cells claim the same grid position.", "error", 1.0, cell.box or block.box))
                        occupied.update(coordinates)
                        if cell.source_page is not None and cell.source_page != page.index:
                            issues.append(StructuralIssue("cell_provenance_mismatch", "A table cell points to a different source page than its table.", "error", 1.0, cell.box or block.box))
                        if cell.box is not None and (not inside(cell.box, page) or not block.box.contains_center(cell.box)):
                            issues.append(StructuralIssue("cell_geometry_mismatch", "A table cell is not geometrically owned by its table.", "error", 1.0, cell.box))
                        if cell.runs and "".join(run.text for run in cell.runs) != cell.text:
                            issues.append(StructuralIssue("cell_run_text_mismatch", "Inline runs do not reproduce the table-cell text exactly.", "error", 1.0, cell.box or block.box))
                        for run in cell.runs:
                            if run.source_page != page.index:
                                issues.append(StructuralIssue("cell_run_provenance_mismatch", "A table-cell run points to a different source page.", "error", 1.0, run.box))
                elif isinstance(block, Graphic) and not block.data:
                    issues.append(StructuralIssue("empty_graphic_asset", "A graphic block has no isolated image data.", "error", 1.0, block.box))
        return issues
