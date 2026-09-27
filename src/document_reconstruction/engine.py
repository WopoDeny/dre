"""One conversion engine: analysis, recognition, understanding, reconstruction, QA."""

from __future__ import annotations

import dataclasses
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median

import numpy as np
import pymupdf
from PIL import Image

from .config import Settings
from .core import ConversionContext, EngineError, ErrorCode, PageRoute
from .core.deadline import INTERNAL_STOP_MS
from .core.errors import public_error
from .intake import to_pdf
from .model import Document, StructuralIssue
from .preflight import CostPredictor, PagePlan, analyze
from .quality import verify
from .speed import REFERENCE_MS, speed_factor
from .recognition.geometry import TransformStep, recover
from .recognition.languages import OSD_SCRIPT_FAMILY, direction as text_direction, disallowed_letters, letter_scripts, ocr_profile
from .recognition.native import extract_native, native_rtl_digit_order_anomalies
from .recognition.coverage import uncovered_text
from .recognition.repair import read_regions, repair_words
from .recognition.lexicon import Lexicon
from .recognition.marks import color_marks, printed_color_words, resolve_layers, split_ink
from .recognition.scale import letter_height, mark_bold, ocr_scale, upscale
from .recognition.regions import _cell_text, _ink_asset, blank_obscured_words, detect_regions, find_pictures, prepare_text_ocr_image, words_outside_pictures
from .recognition.tesseract import TesseractBackend
from .recognition.handwriting import handwritten_fields, signatures_from_scribbles
from .recognition.screenshot import strip_phone_ui
from .recognition.types import RecognizedPage
from .recognition.underline import mark_underlines
from .recognition.vote import agrees, attach_alternatives, confirm_numbers, second_reading
from .recognition.restore import restore_page
from .recognition.wordcheck import PAGE_LANGUAGES, check_page_text
from .reconstruction.docx import reconstruct
from .understanding import understand

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
# A whole-page OSD script decision this strong is trusted without OCR.
UNSUPPORTED_SCRIPT_OSD_CONFIDENCE = 5.0
# Disallowed-script letters tolerated in a text layer (stray symbols, names).
NATIVE_DISALLOWED_LETTERS = 10
NATIVE_DISALLOWED_SHARE = 0.01
# Pages with more unreliable words are not worth a second reading.
MAX_REPAIRABLE_ERRORS = 30
# Pages with more skipped text regions are not read a second time.
MAX_LOST_REGIONS = 16
# Share of the body words that could be neither read nor restored above which the page is unreadable.
MAX_LOST_TEXT_SHARE = 0.1
SURE_ORIENTATION = 6.0       # OSD this sure of the orientation: bad text is not an upside-down page
MAX_GARBLED_WORDS = 2        # words with signs inside that no text is made of
# Printed words next to a handwritten insert must be read identically twice or very confidently:
# handwriting spills over print and a misread word there would pass the dictionary.
NEAR_HANDWRITING_CONFIDENCE = 0.9
# Below this share of Cyrillic letters a page with errors is read again with the Latin model only.
LATIN_ONLY_CYRILLIC_SHARE = 0.10
# Before each further scanned page: the slowest page so far × pages left × this margin must fit,
# with this reserve left for layout, DOCX and checks.
PAGE_TIME_MARGIN = 1.1
FINAL_STAGES_RESERVE_MS = 1500
# Pages where more words are only plausible (names, national word forms) than this are not issued.
MAX_UNVERIFIED_SHARE = 0.15
# Letter-like ink components that must lie inside recognized words (the rest: pictures, marks, noise).
MIN_LETTERS_READ = 0.8


DENSE_COLOR = 0.3   # share of a colored mark's box in colored ink: a filled picture, not a stamp ring


def _color_density(mask, box, *, width: float, height: float) -> float:
    if mask is None:
        return 0.0
    sy, sx = mask.shape[0] / height, mask.shape[1] / width
    part = mask[int(box.y0 * sy):int(np.ceil(box.y1 * sy)), int(box.x0 * sx):int(np.ceil(box.x1 * sx))]
    return float(part.mean()) if part.size else 0.0


def _crosses_text(box, words) -> bool:
    """A confident word inside the box whose line goes on outside it."""
    def inside(word) -> bool:
        return box.contains_center(word.box)
    for word in words:
        if word.confidence < 0.75 or sum(character.isalpha() for character in word.text) < 2 or not inside(word):
            continue
        if any(other.line_id == word.line_id and other.confidence >= 0.75 and not inside(other) for other in words):
            return True
    return False


LETTERHEAD_ZONE = 0.25   # top share of the page where the letterhead is


def _structure_problems(words, graphics, *, height: float) -> list[str]:
    """Signs that the page would come out broken even though its words were read.

    The letterhead is half text, half picture: colored printed words in the letterhead zone touch a colored
    mark taken for a signature or a logo (the rest of the same letterhead)."""
    problems = []
    zone = LETTERHEAD_ZONE * height
    printed = [word for word in words if word.source_region == "ocr:color_ink" and word.box.y1 < zone]
    marks = [graphic.box for graphic in graphics if graphic.source_region == "raster_graphic:color_ink"
             and graphic.role in ("signature", "logo") and graphic.box.y1 < zone]
    for word in printed:
        gap = word.box.height
        if any(box.x0 < word.box.x1 and word.box.x0 < box.x1 and box.y0 - gap < word.box.y1 and word.box.y0 < box.y1 + gap
               for box in marks):
            problems.append("letterhead_split")
            break
    return problems


RULE_GLYPHS = "|¦│"
AMOUNT_DASH = re.compile(r"^(\d[\d\s]*[.,]\d{2})[-–—]$")


def _without_rules(word):
    """A double border of a cell is read as "|" next to the text ("1 |", "| Кол-во"); an amount
    touching the border gets a dash ("350,00-"). Neither is in the cell."""
    text = word.text.strip(RULE_GLYPHS)
    text = AMOUNT_DASH.sub(r"\1", text)
    return word if text == word.text else dataclasses.replace(word, text=text, alternative=text)


def _refilled(table, words):
    if table.source_region != "raster_table":
        return table
    cells = []
    for cell in table.cells:
        contained = [_without_rules(word) for word in words if cell.box is not None and cell.box.contains_center(word.box)]
        contained = [word for word in contained if word.text]
        text = _cell_text(contained)
        cells.append(dataclasses.replace(cell, text=text, direction=text_direction(text),
                                         confidence=float(np.mean([word.confidence for word in contained])) if contained else 0.5))
    return dataclasses.replace(table, cells=cells)


def _doubtful_near_handwriting(words) -> list[str]:
    """Words beside or just above/below an empty field (a handwritten date or number) that
    were neither read identically twice nor recognized confidently."""
    fields = [word for word in words if word.source_region == "blank_field"]
    doubtful = []
    for word in words:
        if word.source_region == "blank_field" or agrees(word) or word.confidence >= NEAR_HANDWRITING_CONFIDENCE:
            continue
        if not any(character.isalnum() for character in word.text):
            continue
        for field in fields:
            h = max(field.box.height, word.box.height)
            vertical = max(field.box.y0 - word.box.y1, word.box.y0 - field.box.y1, 0.0)
            horizontal = max(field.box.x0 - word.box.x1, word.box.x0 - field.box.x1, 0.0)
            if vertical <= 1.0 * h and horizontal <= 3.0 * h:
                doubtful.append(word.text)
                break
    return doubtful


def _render_dpi(page: pymupdf.Page, dpi: int) -> int:
    """Render a scanned page at most at the resolution of its image.

    Rendering a 72-dpi photo at 200 dpi only interpolates pixels; letter size
    checks and OCR upscaling must see the real resolution.
    """
    area = max(1.0, page.rect.get_area())
    best = None
    for image in page.get_image_info():
        box = pymupdf.Rect(image["bbox"]) & page.rect
        if box.get_area() / area >= 0.5 and box.width > 0:
            native = image["width"] / (box.width / 72)
            best = native if best is None else max(best, native)
    return max(50, min(dpi, int(best))) if best else dpi


def _reject_disallowed_text(text: str, allowed: tuple[str, ...], page_index: int, source: str) -> None:
    letters = sum(letter_scripts(text).values())
    foreign = disallowed_letters(text, allowed)
    count = sum(foreign.values())
    if count >= NATIVE_DISALLOWED_LETTERS or (letters and count / letters >= NATIVE_DISALLOWED_SHARE and count >= 3):
        script = foreign.most_common(1)[0][0]
        raise EngineError(
            ErrorCode.LANGUAGE_UNAVAILABLE,
            "The document contains a script that is not enabled for this deployment.",
            stage="script",
            details={"page_index": page_index, "script": script, "letters": count, "reason": f"script_not_allowed_{source}"},
        )


@dataclass
class ConversionResult:
    request_id: str
    docx: bytes | None
    metrics: dict[str, object]
    error: dict[str, object] | None = None
    model: Document | None = None
    quality: dict[str, object] | None = None
    diagnostics: list[dict[str, object]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.docx is not None and self.error is None


class ConversionEngine:
    def __init__(self, *, ocr: TesseractBackend | None = None, predictor: CostPredictor | None = None, dpi: int = 300, settings: Settings | None = None) -> None:
        self.ocr = ocr
        self.settings = settings or Settings.from_env()
        self.lexicon: Lexicon | None = None
        self.predictor = predictor or CostPredictor()
        self.dpi = dpi
        self._pool: ThreadPoolExecutor | None = None
        self._threads_warm = False
        self.speed: dict[str, object] | None = None

    def warm(self) -> None:
        if self.ocr is None:
            self.ocr = TesseractBackend()
        if self.lexicon is None:
            self.lexicon = Lexicon.ensure(getattr(self.ocr, "tessdata_dir", None))
        if self.speed is None and isinstance(self.ocr, TesseractBackend):
            # Calibrate cost estimates to this machine: a slow server refuses early instead of timing out.
            allowed = self.settings.ocr_scripts
            factor, measured = speed_factor(self.ocr, ocr_profile(allowed, self.ocr.available, first=allowed[-1]))
            self.speed = {"factor": round(factor, 3), "reference_ms": REFERENCE_MS, "measured_ms": measured}
            self.predictor = dataclasses.replace(self.predictor, speed_factor=factor, parallel_pages=self.settings.page_threads)

        prewarm = getattr(self.ocr, "prewarm_profiles", None)
        if callable(prewarm):
            try:
                prewarm(self.settings.ocr_scripts)  # only the enabled scripts: no memory for the others
            except TypeError:
                prewarm()
            if isinstance(self.ocr, TesseractBackend):
                # The exact profiles a page is read with ("Cyrillic+Latin", "Latin+Cyrillic"), not loaded mid-request.
                allowed = self.settings.ocr_scripts
                for script in allowed:
                    self.ocr._handle(ocr_profile(allowed, self.ocr.available, first=script))
            in_pool = threading.current_thread().name.startswith("dre-page")
            pool = self._page_pool() if not self._threads_warm and not in_pool else None
            if pool is not None:
                self._threads_warm = True  # warm() is also called from page threads: never wait on the pool there
                # Every page thread loads its own OCR models once, before the first request.
                barrier = threading.Barrier(self.settings.page_threads)

                allowed = self.settings.ocr_scripts
                profiles = {ocr_profile(allowed, self.ocr.available, first=script) for script in allowed}

                def load() -> None:
                    barrier.wait(timeout=60)
                    for languages in ["osd", *sorted(profiles)]:
                        self.ocr._handle(languages)
                for future in [pool.submit(load) for _ in range(self.settings.page_threads)]:
                    future.result()

    def _page_pool(self) -> ThreadPoolExecutor | None:
        """Threads for recognizing pages at once (Tesseract releases the interpreter lock)."""
        if self.settings.page_threads <= 1:
            return None
        if self._pool is None:
            self._pool = ThreadPoolExecutor(self.settings.page_threads, thread_name_prefix="dre-page")
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
            self._threads_warm = False
        if self.ocr is not None:
            self.ocr.close()
            self.ocr = None

    def _render(self, pdf: pymupdf.Document, plan: PagePlan) -> Image.Image:
        source_page = pdf[plan.index]
        original_rotation = int(source_page.rotation) % 360
        if original_rotation:
            source_page.set_rotation(0)
        try:
            pixmap = source_page.get_pixmap(dpi=_render_dpi(source_page, self.dpi), colorspace=pymupdf.csRGB, alpha=False)
        finally:
            if original_rotation:
                source_page.set_rotation(original_rotation)
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

    def _scan(self, pdf: pymupdf.Document | None, plan: PagePlan, context: ConversionContext,
              source: Image.Image | None = None, hint: str | None = None,
              skip_latin: bool = False) -> tuple[RecognizedPage, dict[str, object]]:
        """`hint`: an earlier page of this document was read best with the Latin model alone, so this one
        starts there; `skip_latin`: on an earlier page the Latin model alone did not read better."""
        passes = 1
        page, details = self._recognize_page(pdf, plan, context, profile=hint, source=source)
        tokens = details.get("text_tokens") or 0
        if hint is not None and tokens >= 20 and (details.get("text_errors") or 0) > 0.3 * tokens:
            passes += 1
            page, details = self._recognize_page(pdf, plan, context, source=source)  # not a Latin page after all
            tokens = details.get("text_tokens") or 0
        if (tokens >= 20 and (details.get("text_errors") or 0) > 0.3 * tokens
                and (details.get("orientation_confidence") or 0.0) < SURE_ORIENTATION):
            context.budget.require_fit(0, reserve_ms=8_000, stage="ocr")
            # Mostly unreadable text: an upside-down page that OSD did not detect (it was unsure).
            passes += 1
            turned, turned_details = self._recognize_page(pdf, plan, context, turn=180, source=source)
            if (turned_details.get("text_errors") or 0) < details["text_errors"]:
                page, details = turned, turned_details
        scripts = details.get("recognized_script_counts") or {}
        letters = sum(scripts.values())
        latin_retry = None
        latin_only = ocr_profile(("Latin",), self.ocr.available)
        if (not skip_latin and latin_only != details.get("profile") and details.get("text_errors") and letters
                and scripts.get("Cyrillic", 0) < LATIN_ONLY_CYRILLIC_SHARE * letters and "Latin" in self.settings.ocr_scripts):
            context.budget.require_fit(0, reserve_ms=15_000, stage="ocr")
            # A Latin page (Turkmen, Azerbaijani) read with the Cyrillic model alongside picks up
            # Cyrillic lookalikes ("ѕаһегіпаакі"); the Latin model alone reads it as it is.
            passes += 1
            again, again_details = self._recognize_page(pdf, plan, context, turn=details.get("turned") or 0, profile=latin_only, source=source)
            latin_retry = "lost"
            if (again_details.get("text_errors") or 0) < details["text_errors"]:
                page, details = again, again_details
                latin_retry = "won"
        details["reading_passes"] = passes
        details["latin_retry"] = latin_retry
        return self._checked_scan(plan, page, details)

    def _checked_scan(self, plan: PagePlan, page: RecognizedPage, details: dict[str, object]) -> tuple[RecognizedPage, dict[str, object]]:
        region_diagnostics = details.get("region_detection", {})
        if details.get("structure_problems"):
            raise EngineError(ErrorCode.UNREADABLE_DOCUMENT, "The page structure cannot be rebuilt reliably.", stage="qa",
                              details={"page_index": plan.index, "reason": "unreliable_structure",
                                       "problems": " ".join(details["structure_problems"])})
        if len(details.get("garbled_words") or []) >= MAX_GARBLED_WORDS:
            # Signs no text is made of ("46>Уо", "&#E"): a script of another stage or an unreadable insert.
            raise EngineError(ErrorCode.UNREADABLE_DOCUMENT, "Part of the page text cannot be read.", stage="qa",
                              details={"page_index": plan.index, "reason": "text_unreadable",
                                       "words": " ".join(details["garbled_words"])[:400]})
        if (details.get("lost_text_share") or 0.0) > MAX_LOST_TEXT_SHARE:
            raise EngineError(
                ErrorCode.UNREADABLE_DOCUMENT,
                "The main text of the page cannot be read.",
                stage="qa",
                details={"page_index": plan.index, "reason": "text_unreadable", "lost_share": float(details["lost_text_share"]),
                         "words": " ".join(details.get("text_error_words") or [])[:400]},
            )
        if (details.get("letters_read_share") or 1.0) < MIN_LETTERS_READ:
            raise EngineError(
                ErrorCode.UNREADABLE_DOCUMENT,
                "Most of the text on the page was not recognized.",
                stage="qa",
                details={"page_index": plan.index, "reason": "text_not_read", "letters_read_share": float(details["letters_read_share"])},
            )
        if details.get("unrecognized_text_regions"):
            raise EngineError(
                ErrorCode.UNREADABLE_DOCUMENT,
                "Part of the page looks like text but was not recognized.",
                stage="qa",
                details={"page_index": plan.index, "reason": "unrecognized_text", "regions": int(details["unrecognized_text_regions"])},
            )
        weak_table_cells = list(region_diagnostics.get("table_weak_cells", []))
        if weak_table_cells:
            raise EngineError(
                ErrorCode.UNREADABLE_DOCUMENT,
                "Table recovery left one or more ink-bearing cells unreliable.",
                stage="ocr",
                details={"page_index": plan.index, "weak_table_cell_count": len(weak_table_cells), "reason": "unresolved_table_cells"},
            )
        return page, details

    def _checked_text(self, ocr_image, words, languages, context, *, width, height, protected=()):
        """Dictionary check with one second reading of the unreliable words."""
        if not self.lexicon:
            return None, 0
        text_check = check_page_text(words, self.lexicon, make_fields=False)
        repaired = 0
        if 0 < text_check.errors <= MAX_REPAIRABLE_ERRORS:
            words, repaired = repair_words(
                ocr_image, text_check.words, text_check.error_objects, self.ocr, languages,
                context.budget, self.lexicon, width=width, height=height,
            )
        return check_page_text(words, self.lexicon, protected=protected), repaired

    def _recognize_page(self, pdf: pymupdf.Document | None, plan: PagePlan, context: ConversionContext, *, turn: int = 0,
                        profile: str | None = None, source: Image.Image | None = None) -> tuple[RecognizedPage, dict[str, object]]:
        """Recognize one raster page without quality gates (also used by diagnostics tools)."""
        context.assert_ocr_page(plan.index)
        if source is not None:
            image = source.copy()
        else:
            with context.stage("render"):
                image = self._render(pdf, plan)
        with context.stage("geometry"):
            source_size = image.size
            image, screen = strip_phone_ui(image, page_index=plan.index)
            recovered = recover(image, context.budget)
            if screen is not None:
                recovered.operations.insert(0, "phone_interface_removed")
                recovered.transforms.insert(0, TransformStep("screenshot_content", source_size, image.size, screen))
            image = recovered.image
            source_letter_px = letter_height(image)
            points_per_pixel = plan.width / max(1, image.width)
            scale = ocr_scale(source_letter_px)
            if min_letter_px := self.settings.min_letter_px:
                if 0 < source_letter_px < min_letter_px:
                    raise EngineError(
                        ErrorCode.UNREADABLE_DOCUMENT,
                        "The scan resolution is too low for reliable recognition.",
                        stage="geometry",
                        details={"page_index": plan.index, "reason": "low_resolution", "letter_height_px": float(source_letter_px), "minimum_px": float(min_letter_px)},
                    )
            if abs(scale - 1.0) >= 0.02:
                source_size = image.size
                image = upscale(image, scale)
                recovered.transforms.append(TransformStep("rescale", source_size, image.size, {"factor": round(scale, 3)}))
        preview = image.convert("L")
        preview.thumbnail((400, 400))
        if float((np.asarray(preview) < 200).mean()) < 0.0005:
            return RecognizedPage(plan.index, plan.width, plan.height, [], intentionally_blank=True, source_rotation=plan.rotation), {
                "page": plan.index, "route": "OCR", "blank_scan": True, "ocr_performed": False,
                "recognition_source": "raster_blank_check", "estimated_ms": plan.estimated_ms,
                "transformations": [step.to_dict() for step in recovered.transforms],
            }
        with context.stage("script"):
            self.warm()
            observation = self.ocr.detect_script(image, context.budget)
        projection = np.asarray(preview) < 180
        row_strength = float(np.var(projection.mean(axis=1)))
        column_strength = float(np.var(projection.mean(axis=0)))
        quarter_turn_evidence = observation.clockwise_orientation in (90, 270) and column_strength > row_strength * 1.5
        orientation_applied = 0
        if observation.clockwise_orientation and (observation.orientation_confidence >= 2.0 or (quarter_turn_evidence and observation.orientation_confidence >= 0.5)):
            with context.stage("geometry"):
                source_size = image.size
                orientation_applied = observation.clockwise_orientation
                image = image.rotate(orientation_applied, expand=True, fillcolor="white")
                recovered.operations.append(f"orientation:{orientation_applied}")
                recovered.transforms.append(TransformStep("orientation", source_size, image.size, {"clockwise_degrees": orientation_applied}))
            with context.stage("script"):
                observation = self.ocr.detect_script(image, context.budget)

        if turn:
            image = image.rotate(turn, expand=True, fillcolor="white")
            recovered.transforms.append(TransformStep("orientation", image.size, image.size, {"clockwise_degrees": turn}))
        with context.stage("script"):
            # OSD is only a hint: on small or noisy pages it confuses Latin and
            # Cyrillic. A very strong whole-page decision for a script that is
            # not enabled is trusted, so such scans are rejected before OCR.
            osd_family = OSD_SCRIPT_FAMILY.get(observation.script, observation.script)
            allowed = self.settings.ocr_scripts
            if osd_family is not None and osd_family not in allowed and observation.script_confidence >= UNSUPPORTED_SCRIPT_OSD_CONFIDENCE:
                raise EngineError(
                    ErrorCode.LANGUAGE_UNAVAILABLE,
                    "The page is written in a script that is not enabled for recognition.",
                    stage="script",
                    details={"page_index": plan.index, "script": osd_family, "script_confidence": float(observation.script_confidence), "reason": "script_not_allowed_ocr"},
                )
            first = osd_family if osd_family in allowed and observation.script_confidence >= 2.0 else None
            languages = profile or ocr_profile(allowed, self.ocr.available, first=first)

        width, height = plan.width, plan.height
        if (image.width > image.height) != (width > height) and orientation_applied in (90, 270):
            width, height = height, width
        layers = split_ink(image)
        color_ocr_diagnostics = dict(layers.diagnostics)
        if layers.color is None:
            ocr_image, legacy = prepare_text_ocr_image(image)
            color_ocr_diagnostics.update(legacy)
        else:
            ocr_image = layers.black
        color_graphics = None
        blank_fields = 0
        pen: list = []  # handwritten lines in colored ink: skipped, never text
        with context.stage("ocr"):
            context.metrics.pages_ocr_performed += 1
            words = self.ocr.recognize(ocr_image, languages, context.budget, page_width=width, page_height=height, psm=3)
            second = second_reading(ocr_image, self.ocr, languages, context.budget, width=width, height=height)
            agreed_words = attach_alternatives(words, second)
            numbers_checked = confirm_numbers(ocr_image, words, self.ocr, languages, context.budget, width=width, height=height)
            if layers.color is not None:
                colored = printed_color_words(self.ocr.recognize(layers.color, languages, context.budget, page_width=width, page_height=height, psm=3), self.lexicon, page_height=height, pen=pen)
                for word in colored:
                    word.source_region = "ocr:color_ink"
                words = resolve_layers(image, layers.color_mask, words, colored + pen, width=width, height=height) + colored
                color_graphics, blanks, color_diagnostics = color_marks(image, layers.color_mask, words, index=plan.index, width=width, height=height, pen=pen)
                words.extend(blanks)
                blank_fields = len(blanks)
                color_ocr_diagnostics.update(color_diagnostics, printed_color_words=len(colored))
        recognized_scripts = letter_scripts(" ".join(word.text for word in words))
        confidence = mean(word.confidence for word in words) if words else 0
        for word in words:
            word.source_page = plan.index
            if word.source_region == "page":
                word.source_region = "ocr"
            word.script_evidence = tuple(recognized_scripts.keys())
            word.language_profile = languages
        with context.stage("understanding"):
            tables, graphics, region_diagnostics, region_issues = detect_regions(image, words, index=plan.index, width=width, height=height, color_graphics=color_graphics)
            pictures = find_pictures(ocr_image, words, index=plan.index, width=width, height=height, color_mask=layers.color_mask)
            graphics = [graphic for graphic in graphics if not any(picture.box.contains_center(graphic.box) for picture in pictures)] + pictures
            # A black picture that a line of read text runs through is a signature over the text: it cannot be cut
            # out without printed letters, so it is left out and the text stays text.
            over_text = [graphic for graphic in pictures if _crosses_text(graphic.box, words)]
            graphics = [graphic for graphic in graphics if graphic not in over_text]
            unclean_marks = [graphic.box for graphic in over_text]
            words, picture_words = words_outside_pictures(words, graphics)
            # Overlap issues of words that turned out to be picture or stamp remnants are void.
            kept_boxes = {id(word.box) for word in words}
            region_issues = [issue for issue in region_issues if issue.code != "unresolved_graphic_text_overlap" or id(issue.box) in kept_boxes]
            # Printed text under a seal or stamp must be read; a weak word there is never an empty field.
            protected = [graphic.box for graphic in graphics if graphic.role in ("seal", "stamp")]
            words, scribbles = signatures_from_scribbles(ocr_image, words, protected, index=plan.index, width=width, height=height, asset=_ink_asset)
            graphics = graphics + scribbles
            words, region_issues, obscured_blanks = blank_obscured_words(words, region_issues, protected)
            blank_fields += obscured_blanks
        with context.stage("qa"):
            words, underlined = mark_underlines(ocr_image, words, width=width, height=height,
                                                 tables=[table.box for table in tables])
            words, handwritten = handwritten_fields(ocr_image, words, width=width, height=height,
                                                   tables=[table.box for table in tables])
            text_check, repaired = self._checked_text(ocr_image, words, languages, context, width=width, height=height, protected=protected)
            if text_check is not None:
                words = text_check.words
            # Colored stamps and signatures are gone from the black layer the OCR read: dark ink left inside
            # their frames is text under the stamp and must have been read like any other ("Директор").
            # A dense colored picture (a state emblem, a logo) owns the dark details inside it; only a ring or a
            # stroke of colored ink lies over text.
            dense = [graphic for graphic in graphics if graphic.source_region == "raster_graphic:color_ink"
                     and _color_density(layers.color_mask, graphic.box, width=width, height=height) >= DENSE_COLOR]
            covering = [graphic for graphic in graphics if graphic.source_region != "raster_graphic:color_ink"] + dense
            stamps = [graphic.box for graphic in graphics if graphic.source_region == "raster_graphic:color_ink" and graphic not in dense]
            coverage = uncovered_text(ocr_image, words, covering, tables, width=width, height=height, stamps=stamps,
                                      skipped=[word.box for word in pen], colored=image, marks=unclean_marks)
            filled = 0
            if coverage.clusters and len(coverage.clusters) <= MAX_LOST_REGIONS:
                context.budget.require_fit(0, reserve_ms=6_000, stage="ocr")
                # Automatic page layout sometimes skips whole lines (slight skew, a
                # line next to a table). Read the skipped regions as plain blocks;
                # the new words pass the same checks.
                extra = read_regions(ocr_image, coverage.clusters, self.ocr, languages, context.budget, width=width, height=height)
                attach_alternatives(extra, second)  # words of skipped lines are voted on too
                confirm_numbers(ocr_image, extra, self.ocr, languages, context.budget, width=width, height=height)
                if extra:
                    filled = len(extra)
                    text_check, more = self._checked_text(ocr_image, words + extra, languages, context, width=width, height=height, protected=protected)
                    repaired += more
                    if text_check is not None:
                        words = text_check.words
                    coverage = uncovered_text(ocr_image, words, covering, tables, width=width, height=height, stamps=stamps,
                                      skipped=[word.box for word in pen], colored=image, marks=unclean_marks)
            blank_fields = sum(word.source_region == "blank_field" for word in words)
            if text_check is not None:
                # A word touched by a stamp or signature that both readings saw identically and
                # that passed the check is read: the overlap is resolved.
                # A reread word that passed is read too (its box is a new object: match by place).
                failed = {id(word) for word in text_check.error_objects}
                read = [word.box for word in words
                        if (agrees(word) or word.source_region == "ocr_repair") and id(word) not in failed]
                region_issues = [issue for issue in region_issues
                                 if not (issue.code == "unresolved_graphic_text_overlap" and issue.box is not None
                                         and any(box.contains_center(issue.box) and issue.box.contains_center(box) for box in read))]
            region_issues = region_issues + [
                StructuralIssue("unrecognized_text", "Text-like ink produced no recognized words.", "error", 0.0, box)
                for box in coverage.clusters
            ]

        restoration = restore_page(words, text_check.error_objects if text_check is not None else [], self.lexicon,
                                   PAGE_LANGUAGES.get(), page_height=height)
        words = restoration.words
        # Table text was taken from the first reading when the table was found; the checked words
        # (reread, normalized, blanked) are what may be issued, so the cells are filled again from them.
        tables = [_refilled(table, words) for table in tables]
        # A cell is unreliable only if its ink produced no text at all: weak words in it were checked and
        # restored like any other word of the page.
        region_diagnostics["table_weak_cells"] = [
            cell for cell in region_diagnostics.get("table_weak_cells", [])
            if cell["table"] >= len(tables) or not any(
                other.row == cell["row"] and other.column == cell["column"] and other.text.strip() for other in tables[cell["table"]].cells)
        ]
        photo = "paper_rectification" in recovered.operations
        bold_lines = mark_bold(ocr_image, words, width=width, height=height, photo=photo)
        unresolved_rotation = len(words) > 8 and median(word.box.height for word in words) > median(word.box.width for word in words) * 1.8
        transforms = [step.to_dict() for step in recovered.transforms]
        transforms.append({
            "kind": "ocr_pixels_to_normalized_page_points",
            "source_size": [image.width, image.height],
            "target_size": [width, height],
            "parameters": {"scale_x": width / image.width, "scale_y": height / image.height},
        })
        # Times New Roman x-height is 0.46 em; the median letter component is about the x-height.
        body_hint = round(source_letter_px * points_per_pixel / 0.46 * 2) / 2 if source_letter_px else None
        return RecognizedPage(plan.index, width, height, words, tables, graphics, source_rotation=plan.rotation, issues=region_issues,
                              body_size_hint=body_hint if body_hint and 8 <= body_hint <= 16 else None, photo=photo), {
            "page": plan.index, "route": "OCR", "profile": languages,
            "osd_script": observation.script, "script_confidence": observation.script_confidence,
            "recognized_script_counts": dict(recognized_scripts),
            "orientation": observation.clockwise_orientation, "orientation_confidence": observation.orientation_confidence,
            "operations": recovered.operations, "transformations": transforms,
            "confidence": confidence, "words": len(words), "unresolved_rotation": unresolved_rotation,
            "ocr_performed": True, "recognition_source": "local_ocr",
            "estimated_ms": plan.estimated_ms, "render_megapixels": plan.render_megapixels,
            "geometry_required": plan.geometry_required, "recovery_reason": plan.recovery_reason,
            "render_dpi": plan.render_dpi, "render_pixel_limit": plan.render_pixel_limit,
            "region_detection": region_diagnostics,
            "blank_fields": blank_fields, "picture_words": picture_words,
            "letter_height_px": source_letter_px, "ocr_scale": round(scale, 3),
            "text_tokens": text_check.tokens if text_check else None,
            "text_errors": text_check.errors if text_check else None,
            "restored_words": restoration.changes, "dropped_lines": restoration.dropped_lines,
            "lost_text_share": round(restoration.lost_share, 4), "garbled_words": restoration.garbled,
            "structure_problems": _structure_problems(words, graphics, height=height),
            "text_error_words": text_check.error_words[:20] if text_check else [],
            "text_unverified_share": round(text_check.unverified_share, 4) if text_check else None,
            "unrecognized_text_regions": len(coverage.clusters),
            "letters_read_share": round(coverage.read_share, 3),
            "bold_lines": bold_lines, "underlined_words": underlined, "handwritten_fields": handwritten, "repaired_words": repaired, "agreed_words": agreed_words, "numbers_checked": numbers_checked, "turned": turn, "lost_region_words": filled,
            **color_ocr_diagnostics,
        }

    def convert(self, payload: bytes, *, timeout_ms: float = INTERNAL_STOP_MS, request_id: str | None = None, workspace_root: Path | None = None) -> ConversionResult:
        context = ConversionContext(timeout_ms=timeout_ms, request_id=request_id, workspace_root=workspace_root)
        output = None
        model = None
        quality = None
        diagnostics: list[dict[str, object]] = []
        error = None
        try:
            with context:
                with context.stage("preflight"):
                    if len(payload) > MAX_UPLOAD_BYTES:
                        raise EngineError(ErrorCode.UPLOAD_TOO_LARGE, "The PDF exceeds the upload size limit.", stage="preflight")
                    payload = to_pdf(payload)
                    if not payload or b"%PDF-" not in payload[:1024]:
                        raise EngineError(ErrorCode.INVALID_PDF, "The request body is not a PDF.", stage="preflight")
                    try:
                        pdf = pymupdf.open(stream=payload, filetype="pdf")
                    except (pymupdf.FileDataError, RuntimeError) as cause:
                        raise EngineError(ErrorCode.INVALID_PDF, "The PDF could not be opened.", stage="preflight") from cause
                with pdf:
                    with context.stage("preflight"):
                        plans = analyze(pdf, context, self.predictor, dpi=self.dpi, max_pages=self.settings.max_pages)
                    recognized = []
                    scans = [plan for plan in plans if plan.route == PageRoute.OCR and not plan.searchable_scan]
                    hint: str | None = None
                    skip_latin = False
                    pool = self._page_pool() if len(scans) > 1 else None
                    futures = {}
                    scan_seconds: list[float] = []
                    if pool is not None:
                        self.warm()
                        # PyMuPDF is not thread-safe: pages are rendered here, recognized in the pool.
                        for plan in scans:
                            with context.stage("render"):
                                source = self._render(pdf, plan)
                            futures[plan.index] = pool.submit(self._scan, None, plan, context, source)
                    for plan in plans:
                        context.budget.check()
                        if plan.route == PageRoute.NATIVE:
                            with context.stage("preflight"):
                                page = extract_native(pdf[plan.index], plan.index)
                                _reject_disallowed_text(" ".join(word.text for word in page.words), self.settings.native_scripts, plan.index, "native")
                                # Multiple independently reversed native Arabic-Indic
                                # number runs prove that the text-layer order is not
                                # a safe editable transcription. Do not guess the
                                # intended digits or route this native page to OCR.
                                native_rtl_digits = any(
                                    any("\u0660" <= character <= "\u0669" or "\u06f0" <= character <= "\u06f9"
                                        for character in word.text)
                                    for word in page.words
                                )
                                if native_rtl_digits and native_rtl_digit_order_anomalies(pdf[plan.index]) >= 2:
                                    raise EngineError(
                                        ErrorCode.UNREADABLE_DOCUMENT,
                                        "Native numeric text order is unreliable; safe editable transcription is unavailable.",
                                        stage="preflight",
                                        details={"page_index": plan.index, "reason": "unreliable_native_rtl_digit_order"},
                                    )
                            diagnostics.append({
                                "page": plan.index, "route": "NATIVE", "words": len(page.words),
                                "ocr_performed": False, "recognition_source": "native_pdf",
                                "rotation": plan.rotation, "estimated_ms": plan.estimated_ms,
                                "recovery_reason": plan.recovery_reason,
                            })
                        elif plan.route == PageRoute.BLANK:
                            page = RecognizedPage(plan.index, plan.width, plan.height, [], intentionally_blank=True, source_rotation=plan.rotation)
                            diagnostics.append({
                                "page": plan.index, "route": "BLANK", "ocr_performed": False,
                                "recognition_source": "blank", "rotation": plan.rotation,
                                "estimated_ms": plan.estimated_ms, "recovery_reason": plan.recovery_reason,
                            })
                        else:
                            if plan.searchable_scan:
                                # A searchable scan already carries a local
                                # OCR text layer. Use its editable observations
                                # without copying the raster and without
                                # running OCR a second time.
                                with context.stage("preflight"):
                                    page = extract_native(pdf[plan.index], plan.index, confidence=plan.text_layer_confidence)
                                    for word in page.words:
                                        word.source_region = "searchable_text_layer"
                                    _reject_disallowed_text(" ".join(word.text for word in page.words), self.settings.ocr_scripts, plan.index, "text_layer")
                                context.metrics.pages_text_layer_reused += 1
                                details = {
                                    "page": plan.index, "route": "OCR_TEXT_LAYER", "words": len(page.words),
                                    "confidence": plan.text_layer_confidence, "unresolved_rotation": False,
                                    "searchable_scan": True, "ocr_performed": False,
                                    "recognition_source": "existing_text_layer", "rotation": plan.rotation,
                                    "text_layer_status": plan.text_layer_status,
                                    "text_layer_ink_coverage": plan.text_layer_ink_coverage,
                                    "text_layer_duplicate_ratio": plan.text_layer_duplicate_ratio,
                                    "text_layer_bad_character_ratio": plan.text_layer_bad_character_ratio,
                                    "estimated_ms": plan.estimated_ms, "recovery_reason": plan.recovery_reason,
                                }
                            elif plan.index in futures:
                                try:
                                    page, details = futures.pop(plan.index).result()  # the first failing page decides, as in order
                                except BaseException:
                                    for pending in futures.values():
                                        pending.cancel()
                                    for pending in futures.values():
                                        if not pending.cancelled():
                                            pending.exception()  # wait: the pool is free for the next request
                                    raise
                            else:
                                if scan_seconds:
                                    # Pages this document actually takes on this machine: if the slowest one so far,
                                    # times the pages left, does not fit, refuse now instead of at the deadline.
                                    left = sum(1 for other in scans if other.index >= plan.index)
                                    context.budget.require_fit(max(scan_seconds) * 1000 * left * PAGE_TIME_MARGIN,
                                                               reserve_ms=FINAL_STAGES_RESERVE_MS, stage="ocr")
                                started = time.perf_counter()
                                page, details = self._scan(pdf, plan, context, hint=hint, skip_latin=skip_latin)
                                # The next pages start with what this one ended with and are read once.
                                if details.get("latin_retry") == "won":
                                    hint = details["profile"]
                                elif details.get("latin_retry") == "lost":
                                    skip_latin = True
                                scan_seconds.append((time.perf_counter() - started) / max(1, details.get("reading_passes") or 1))
                            diagnostics.append(details)
                        recognized.append(page)
                    for page in recognized:
                        unresolved_borderless = next((issue for issue in page.issues if issue.code == "unresolved_borderless_raster_table" and issue.severity == "error"), None)
                        if unresolved_borderless is not None:
                            raise EngineError(
                                ErrorCode.UNREADABLE_DOCUMENT,
                                "A likely borderless raster table could not be reconstructed safely.",
                                stage="understanding",
                                details={"page_index": page.index, "reason": "unresolved_borderless_raster_table"},
                            )
                        unresolved_ruled = next((issue for issue in page.issues if issue.code == "unresolved_ruled_raster_table" and issue.severity == "error"), None)
                        if unresolved_ruled is not None:
                            raise EngineError(
                                ErrorCode.UNREADABLE_DOCUMENT,
                                "A visible ruled raster table could not be reconstructed safely.",
                                stage="understanding",
                                details={"page_index": page.index, "reason": "unresolved_ruled_raster_table"},
                            )
                        if plans[page.index].route == PageRoute.OCR and len(page.words) > 8 and median(word.box.height for word in page.words) > median(word.box.width for word in page.words) * 1.8:
                            raise EngineError(ErrorCode.UNREADABLE_DOCUMENT, "Page orientation could not be resolved confidently.", stage="qa", details={"page_index": page.index})
                        if not page.intentionally_blank and not page.words and not page.graphics:
                            raise EngineError(ErrorCode.UNREADABLE_DOCUMENT, "A nonblank page has no recoverable text or isolated graphics.", stage="qa", details={"page_index": page.index})
                        if page.words and mean(word.confidence for word in page.words) < 0.55:
                            raise EngineError(ErrorCode.UNREADABLE_DOCUMENT, "Recognition confidence remains below the quality threshold.", stage="qa", details={"page_index": page.index})
                    with context.stage("understanding"):
                        model = understand(recognized)
                        integrity_issues = model.integrity_issues()
                        if integrity_issues:
                            model.issues.extend(integrity_issues)
                            errors = [issue for issue in integrity_issues if issue.severity == "error"]
                            if errors:
                                raise EngineError(
                                    ErrorCode.INVARIANT_VIOLATION,
                                    "The unified document model failed integrity validation.",
                                    stage="understanding",
                                    details={"issue_count": len(errors)},
                                )
                    with context.stage("docx"):
                        output = reconstruct(model)
                    with context.stage("qa"):
                        checked = verify(output, model)
                        quality = checked.to_dict()
                        if not checked.passed:
                            raise EngineError(ErrorCode.CONVERSION_FAILED, "The reconstructed document did not pass the quality gate.", stage="qa", details={"failure_count": len(checked.failures)})
        except Exception as cause:
            structured = public_error(cause)
            details = dict(structured.details)
            if structured.code == ErrorCode.SLA_REJECTED and ("estimated_ms" in details or structured.stage in ("preflight", "qa")):
                # Refused by the time forecast before the work, not stopped by the deadline.
                code = ErrorCode.DOCUMENT_TOO_EXPENSIVE
                details.setdefault("reason", "estimated_time_over_budget")
            elif structured.code == ErrorCode.SLA_REJECTED:
                code = ErrorCode.FAST_SLA_EXCEEDED
            else:
                code = {ErrorCode.DEADLINE_EXCEEDED: ErrorCode.FAST_SLA_EXCEEDED, ErrorCode.INTERNAL_ERROR: ErrorCode.CONVERSION_FAILED}.get(structured.code, structured.code)
            error = EngineError(code, structured.message, stage=structured.stage, details=details).to_dict(request_id=context.request_id)
            output = None
        return ConversionResult(context.request_id, output, context.metrics.snapshot(), error, model, quality, diagnostics)
