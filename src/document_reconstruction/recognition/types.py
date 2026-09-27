"""Recognition observations that precede document understanding."""

from dataclasses import dataclass, field

from ..model import Box, Graphic, StructuralIssue, Table


@dataclass
class Word:
    text: str
    box: Box
    confidence: float = 1.0
    size: float = 11.0
    bold: bool = False
    line_id: tuple[int, ...] = ()
    italic: bool = False
    source_page: int | None = None
    source_region: str = "page"
    script_evidence: tuple[str, ...] = ()
    language_profile: str | None = None
    source_space_before: bool = False
    # The second, independent reading of the same place (None: not read twice; "": nothing there).
    alternative: str | None = None
    underline: bool = False


@dataclass
class RecognizedPage:
    index: int
    width: float
    height: float
    words: list[Word]
    tables: list[Table] = field(default_factory=list)
    graphics: list[Graphic] = field(default_factory=list)
    intentionally_blank: bool = False
    source_rotation: int = 0
    issues: list[StructuralIssue] = field(default_factory=list)
    # Body font size (pt) measured from the letter height of a scanned page.
    body_size_hint: float | None = None
    # A photographed sheet: blur and residual perspective scatter the measured letter sizes more than a scan does.
    photo: bool = False
