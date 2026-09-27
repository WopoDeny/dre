"""Cheap per-page preflight, routing, text-layer trust, and cost admission."""

from __future__ import annotations

import math
import unicodedata
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Iterator

import numpy as np
import pymupdf

from .core import ConversionContext, EngineError, ErrorCode, PageRoute


@dataclass(frozen=True)
class TextLayerAssessment:
    """Cheap evidence for a raster page carrying an existing text layer."""

    status: str
    confidence: float
    invisible_fraction: float
    bad_character_ratio: float
    duplicate_ratio: float
    valid_geometry_ratio: float
    ink_coverage: float
    evidence: str


@dataclass(frozen=True)
class PagePlan:
    index: int
    width: float
    height: float
    rotation: int
    route: PageRoute
    native_characters: int
    image_count: int
    image_dimensions: tuple[tuple[int, int], ...]
    image_coverage: float
    mixed_content: bool
    geometry_required: bool
    render_megapixels: float
    estimated_ms: float
    # Compatibility fields are appended with defaults because a few callers
    # construct plans positionally.
    searchable_scan: bool = False
    native_words: int = 0
    text_layer_status: str = "NONE"
    text_layer_confidence: float = 0.0
    text_layer_ink_coverage: float = 0.0
    text_layer_duplicate_ratio: float = 0.0
    text_layer_bad_character_ratio: float = 0.0
    ocr_layer_evidence: bool = False
    recovery_reason: str | None = None
    render_dpi: int = 0
    render_pixel_limit: int = 0


@dataclass(frozen=True)
class CostPredictor:
    """Calibratable CPU priors, not a machine-independent SLA predictor."""

    native_page_ms: float = 65
    searchable_layer_page_ms: float = 90
    # A scanned page on the reference core (1 core of the development laptop, 2026-09): median 2.8 s,
    # 95th percentile 4.8 s, hardly dependent on the image size (small images are enlarged for OCR).
    # 3.3 s covers most pages; the slow rest is caught by the per-page check during conversion.
    ocr_page_ms: float = 3300
    large_page_megapixel_ms: float = 60   # beyond 11 MP (A4 at 300 dpi)
    understanding_page_ms: float = 35
    docx_page_ms: float = 25
    qa_page_ms: float = 20
    cleanup_ms: float = 60
    fixed_request_ms: float = 150
    deadline_tail_reserve_ms: float = 300
    safety_factor: float = 1.1
    # Measured at start-up (speed.py): > 1 on a machine slower than the reference core.
    speed_factor: float = 1.0
    # Pages recognized at once (DRE_PAGE_THREADS).
    parallel_pages: int = 1

    def page_ms(
        self,
        route: PageRoute,
        megapixels: float,
        geometry: bool,
        *,
        searchable_scan: bool = False,
    ) -> float:
        if route == PageRoute.BLANK:
            return 5
        if route == PageRoute.NATIVE:
            return self.native_page_ms * self.speed_factor
        if searchable_scan:
            return self.searchable_layer_page_ms * self.speed_factor
        return self.speed_factor * (self.ocr_page_ms + self.large_page_megapixel_ms * max(0.0, megapixels - 11.0))

    def remaining_ms(self, plans: list[PagePlan]) -> float:
        pages = len(plans)
        finalization = (
            self.fixed_request_ms
            + self.understanding_page_ms * pages
            + self.docx_page_ms * pages
            + self.qa_page_ms * pages
            + self.cleanup_ms
        )
        scans = [plan.estimated_ms for plan in plans if plan.route == PageRoute.OCR and not plan.searchable_scan]
        others = sum(plan.estimated_ms for plan in plans) - sum(scans)
        # Scanned pages run in parallel threads: the slowest thread decides (pages dealt in turn).
        lanes = max(1, min(self.parallel_pages, len(scans)))
        scan_ms = max((sum(sorted(scans, reverse=True)[lane::lanes]) for lane in range(lanes)), default=0.0)
        return (others + scan_ms + finalization * self.speed_factor) * self.safety_factor

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@contextmanager
def _normalized_page(page: pymupdf.Page) -> Iterator[pymupdf.Page]:
    """Expose one unrotated coordinate system while retaining rotation metadata."""

    original_rotation = int(page.rotation) % 360
    if original_rotation:
        page.set_rotation(0)
    try:
        yield page
    finally:
        if original_rotation:
            page.set_rotation(original_rotation)


def _bad_character_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad = 0
    considered = 0
    for character in text:
        if character.isspace():
            continue
        considered += 1
        if character in ("\ufffd", "\x00") or (unicodedata.category(character) == "Cc"):
            bad += 1
    return bad / max(1, considered)


def _word_geometry(words: list[tuple[object, ...]], rect: pymupdf.Rect) -> tuple[float, float]:
    if not words:
        return 0.0, 0.0
    valid = 0
    signatures: set[tuple[object, ...]] = set()
    duplicates = 0
    tolerance = 2.0
    for word in words:
        try:
            x0, y0, x1, y1 = (float(value) for value in word[:4])
            text = str(word[4]).strip()
        except (TypeError, ValueError, IndexError):
            continue
        finite = all(math.isfinite(value) for value in (x0, y0, x1, y1))
        inside = (
            x1 > x0
            and y1 > y0
            and x0 >= rect.x0 - tolerance
            and y0 >= rect.y0 - tolerance
            and x1 <= rect.x1 + tolerance
            and y1 <= rect.y1 + tolerance
            and (y1 - y0) <= max(144.0, rect.height * 0.25)
            and (x1 - x0) <= max(288.0, rect.width * 0.80)
        )
        if finite and inside:
            valid += 1
        signature = (text.casefold(), round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1))
        if signature in signatures:
            duplicates += 1
        else:
            signatures.add(signature)
    return valid / max(1, len(words)), duplicates / max(1, len(words))


def _text_layer_evidence(page: pymupdf.Page) -> tuple[float, bool]:
    """Return invisible-text fraction and a weak OCR-font hint without decoding images."""

    invisible_characters = 0
    traced_characters = 0
    try:
        for trace in page.get_texttrace():
            count = len(trace.get("chars", ()))
            traced_characters += count
            if trace.get("type") == 3:  # PDF text rendering mode 3: invisible text.
                invisible_characters += count
    except (RuntimeError, ValueError, TypeError):
        pass
    invisible_fraction = invisible_characters / max(1, traced_characters)

    font_hint = False
    flags = getattr(pymupdf, "TEXTFLAGS_TEXT", None)
    try:
        text_dict = page.get_text("dict", flags=flags) if flags is not None else page.get_text("dict")
    except TypeError:  # pragma: no cover - compatibility with older PyMuPDF
        text_dict = page.get_text("dict")
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font = str(span.get("font", "")).casefold()
                if any(marker in font for marker in ("glyphless", "tesseract", "ocrb")):
                    font_hint = True
                    break
            if font_hint:
                break
        if font_hint:
            break
    return invisible_fraction, font_hint


def _ink_coverage(page: pymupdf.Page, words: list[tuple[object, ...]], *, dpi: int = 24) -> float:
    """Compare low-resolution raster ink with text boxes; never materialize source images."""

    if not words:
        return 0.0
    pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False, annots=False)
    try:
        samples = np.frombuffer(pixmap.samples, dtype=np.uint8)
        if samples.size != pixmap.width * pixmap.height:
            return 0.0
        gray = samples.reshape((pixmap.height, pixmap.width))
        ink = gray < 220
        ink_count = int(np.count_nonzero(ink))
        if ink_count == 0:
            return 0.0
        mask = np.zeros_like(ink, dtype=bool)
        scale_x = pixmap.width / max(1.0, page.rect.width)
        scale_y = pixmap.height / max(1.0, page.rect.height)
        padding = 3.0
        for word in words:
            try:
                x0, y0, x1, y1 = (float(value) for value in word[:4])
            except (TypeError, ValueError, IndexError):
                continue
            left = max(0, int(math.floor((x0 - padding) * scale_x)))
            top = max(0, int(math.floor((y0 - padding) * scale_y)))
            right = min(pixmap.width, int(math.ceil((x1 + padding) * scale_x)))
            bottom = min(pixmap.height, int(math.ceil((y1 + padding) * scale_y)))
            if right > left and bottom > top:
                mask[top:bottom, left:right] = True
        return float(np.count_nonzero(ink & mask) / ink_count)
    finally:
        del pixmap


def _assess_text_layer(page: pymupdf.Page, text: str, words: list[tuple[object, ...]]) -> TextLayerAssessment:
    invisible_fraction, font_hint = _text_layer_evidence(page)
    evidence = "invisible_text" if invisible_fraction >= 0.50 else "ocr_font_hint" if font_hint else "none"
    if evidence == "none":
        return TextLayerAssessment("AMBIGUOUS", 0.0, invisible_fraction, _bad_character_ratio(text), 0.0, 0.0, 0.0, evidence)

    bad_ratio = _bad_character_ratio(text)
    valid_geometry_ratio, duplicate_ratio = _word_geometry(words, page.rect)
    ink_coverage = _ink_coverage(page, words)
    useful = sum(character.isalnum() for character in text)

    encoding_score = max(0.0, 1.0 - bad_ratio * 20)
    duplication_score = max(0.0, 1.0 - duplicate_ratio * 10)
    coverage_score = min(1.0, ink_coverage / 0.75)
    evidence_score = 1.0 if invisible_fraction >= 0.50 else 0.80
    confidence = min(
        0.95,
        0.20 * encoding_score
        + 0.20 * duplication_score
        + 0.20 * valid_geometry_ratio
        + 0.30 * coverage_score
        + 0.10 * evidence_score,
    )
    trusted = (
        useful >= 12
        and len(words) >= 3
        and bad_ratio <= 0.02
        and duplicate_ratio <= 0.05
        and valid_geometry_ratio >= 0.98
        and ink_coverage >= 0.65
        and confidence >= 0.70
    )
    return TextLayerAssessment(
        "TRUSTED" if trusted else "UNTRUSTED",
        confidence,
        invisible_fraction,
        bad_ratio,
        duplicate_ratio,
        valid_geometry_ratio,
        ink_coverage,
        evidence,
    )


def analyze(
    pdf: pymupdf.Document,
    context: ConversionContext,
    predictor: CostPredictor,
    *,
    dpi: int = 200,
    max_render_pixels: int = 36_000_000,
    max_pages: int | None = None,
) -> list[PagePlan]:
    """Classify each page with metadata/text evidence before full rendering or OCR."""

    if pdf.needs_pass or pdf.page_count == 0:
        raise EngineError(ErrorCode.INVALID_PDF, "The PDF is encrypted or contains no pages.", stage="preflight")
    if max_pages is not None and pdf.page_count > max_pages:
        raise EngineError(
            ErrorCode.DOCUMENT_TOO_EXPENSIVE,
            "The document has more pages than this service accepts.",
            stage="preflight",
            details={"pages": pdf.page_count, "max_pages": max_pages, "reason": "too_many_pages"},
        )
    plans: list[PagePlan] = []
    for index in range(pdf.page_count):
        context.budget.check(stage="preflight")
        page = pdf[index]
        original_rotation = int(page.rotation) % 360
        with _normalized_page(page):
            rect = page.rect
            if rect.width <= 0 or rect.height <= 0 or rect.width > 14_400 or rect.height > 14_400:
                raise EngineError(ErrorCode.INVALID_PDF, "The PDF contains an unsupported page size.", stage="preflight")

            text = page.get_text("text")
            useful = sum(character.isalnum() for character in text)
            bad_ratio = _bad_character_ratio(text)
            words = page.get_text("words", sort=False) if useful else []
            images = page.get_image_info()
            dimensions = tuple((int(image["width"]), int(image["height"])) for image in images)
            image_areas = [
                max(0.0, (pymupdf.Rect(image["bbox"]) & rect).get_area())
                for image in images
            ]
            coverage = min(1.0, sum(image_areas) / max(1.0, rect.get_area()))
            largest_image_coverage = max(image_areas, default=0.0) / max(1.0, rect.get_area())
            large_raster = largest_image_coverage >= 0.35
            scan_raster = largest_image_coverage >= 0.72 or coverage >= 0.85
            native_text = useful > 0 and bad_ratio < 0.05

            assessment = TextLayerAssessment("NONE", 0.0, 0.0, bad_ratio, 0.0, 0.0, 0.0, "none")
            searchable_scan = False
            recovery_reason: str | None = None
            if native_text and scan_raster:
                assessment = _assess_text_layer(page, text, words)
                if assessment.status == "TRUSTED":
                    route = PageRoute.OCR
                    searchable_scan = True
                    recovery_reason = "trusted_existing_text_layer"
                elif assessment.status == "UNTRUSTED":
                    # Strong OCR-layer evidence exists, but the layer is incomplete,
                    # duplicated, malformed, or geometrically implausible. Recover the
                    # raster instead of silently accepting the bad layer.
                    route = PageRoute.OCR
                    recovery_reason = "untrusted_existing_text_layer_recover_raster"
                else:
                    raise EngineError(
                        ErrorCode.CONVERSION_FAILED,
                        "The page combines native text with a large raster region and cannot be reconstructed safely.",
                        stage="preflight",
                        details={"page_index": index, "reason": "ambiguous_hybrid_page"},
                    )
            elif native_text:
                if large_raster:
                    raise EngineError(
                        ErrorCode.CONVERSION_FAILED,
                        "The page combines native text with a large raster region and cannot be reconstructed safely.",
                        stage="preflight",
                        details={"page_index": index, "reason": "ambiguous_hybrid_page"},
                    )
                route = PageRoute.NATIVE
                recovery_reason = "useful_native_text"
            elif images:
                route = PageRoute.OCR
                recovery_reason = "raster_page_without_useful_native_text"
            else:
                # Avoid a drawing walk when text or image metadata already establishes
                # the route. Vector-only marks are nonblank and require recovery.
                route = PageRoute.OCR if page.get_drawings() else PageRoute.BLANK
                recovery_reason = "vector_only_page" if route == PageRoute.OCR else "blank_page"

            megapixels = rect.width * rect.height * (dpi / 72) ** 2 / 1_000_000
            if route == PageRoute.OCR and dimensions and scan_raster:
                # A scan is rendered no finer than its own image.
                megapixels = min(megapixels, max(width * height for width, height in dimensions) / 1_000_000)
            if route == PageRoute.OCR and not searchable_scan and megapixels * 1_000_000 > max_render_pixels:
                raise EngineError(
                    ErrorCode.DOCUMENT_TOO_EXPENSIVE,
                    "A raster page exceeds the configured pixel limit.",
                    stage="preflight",
                    details={"page_index": index, "render_megapixels": megapixels, "pixel_limit": max_render_pixels},
                )
            geometry = route == PageRoute.OCR and not searchable_scan
            render_megapixels = megapixels if geometry else 0.0
            estimated_ms = predictor.page_ms(route, render_megapixels, geometry, searchable_scan=searchable_scan)
            plans.append(
                PagePlan(
                    index=index,
                    width=rect.width,
                    height=rect.height,
                    rotation=original_rotation,
                    route=route,
                    native_characters=useful,
                    image_count=len(images),
                    image_dimensions=dimensions,
                    image_coverage=coverage,
                    mixed_content=native_text and bool(images),
                    geometry_required=geometry,
                    render_megapixels=render_megapixels,
                    estimated_ms=estimated_ms,
                    searchable_scan=searchable_scan,
                    native_words=len(words),
                    text_layer_status=assessment.status,
                    text_layer_confidence=assessment.confidence,
                    text_layer_ink_coverage=assessment.ink_coverage,
                    text_layer_duplicate_ratio=assessment.duplicate_ratio,
                    text_layer_bad_character_ratio=assessment.bad_character_ratio,
                    ocr_layer_evidence=assessment.evidence != "none",
                    recovery_reason=recovery_reason,
                    render_dpi=dpi if route == PageRoute.OCR and not searchable_scan else 0,
                    render_pixel_limit=max_render_pixels,
                )
            )

        # Incremental admission stops an obviously unaffordable request before
        # preflight walks every later page. The predictor is a configurable prior;
        # calibration data are recorded for accepted requests below.
        context.budget.require_fit(
            predictor.remaining_ms(plans),
            reserve_ms=predictor.deadline_tail_reserve_ms,
            stage="preflight",
        )

    context.set_page_routes([plan.route for plan in plans])
    context.metrics.record_admission_estimate(
        predictor.remaining_ms(plans),
        ocr_pages=sum(plan.route == PageRoute.OCR and not plan.searchable_scan for plan in plans),
        searchable_layer_pages=sum(plan.searchable_scan for plan in plans),
        render_megapixels=sum(plan.render_megapixels for plan in plans),
        budget_remaining_ms=context.budget.remaining_ms,
        render_dpi=dpi,
        render_pixel_limit=max_render_pixels,
    )
    return plans
