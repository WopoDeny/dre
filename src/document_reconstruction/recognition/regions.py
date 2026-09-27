"""Conservative table and isolated-graphic recovery for raster pages.

The detector deliberately separates text ownership from graphic recovery. Raster
marks are stored as masked transparent assets, so editable body text remains text
even when a stamp or signature overlaps it. Black-ink signature recovery is a
bounded heuristic and is not presented as general handwriting recognition.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from scipy import ndimage

from ..model import Box, Cell, Graphic, StructuralIssue, Table
from .languages import direction, script_counts
from .pixels import channel_max, channel_min, close_rect, dilate_rect, png_bytes, pure_color
from .types import Word


def _centers(values: np.ndarray) -> list[int]:
    indices = np.flatnonzero(values)
    if not len(indices):
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 2) + 1)
    return [int(round(float(group.mean()))) for group in groups]


def _has_vertical_cell_separator(
    vertical: np.ndarray, top: int, bottom: int, x: int,
) -> bool:
    """Detect a continuous narrow raster rule without assuming five-pixel ink.

    After vertical line opening, a genuine thin rule can occupy just one pixel
    of the five-pixel tolerance band. Averaging the band then reports 0.20 and
    falsely converts every cell into a horizontal span. Examine persistence
    along each *individual* pixel column instead, requiring a stroke through
    most of the cell's interior. A missing or interrupted separator must still
    remain a real cell span rather than an invented split.
    """
    interior = vertical[top + 3:bottom - 2, max(0, x - 2):x + 3]
    if interior.shape[0] < 3 or not interior.size:
        return False
    return bool(np.max(np.mean(interior, axis=0)) >= 0.78)


def detect_text_columns(image: Image.Image, *, max_columns: int = 4) -> list[tuple[int, int]]:
    """Return conservative full-height raster column bounds for mixed OCR."""

    if max_columns < 2 or image.width < 700 or image.height < 500:
        return []
    preview = image.convert("L")
    preview.thumbnail((1200, 1200))
    gray = np.asarray(preview)
    ink = gray < 205
    row_density = ink.mean(axis=1)
    active_rows = np.flatnonzero(row_density > 0.0015)
    if len(active_rows) < 20:
        return []
    top, bottom = int(active_rows[0]), int(active_rows[-1]) + 1
    column_density = ink[top:bottom].mean(axis=0)
    window = max(3, preview.width // 180)
    smooth = ndimage.uniform_filter1d(column_density.astype(float), size=window, mode="nearest")
    positive = smooth[smooth > 0]
    if not len(positive):
        return []
    threshold = max(0.0008, float(np.quantile(positive, 0.12)) * 0.40)
    blank = smooth <= threshold
    minimum_gutter = max(10, preview.width // 45)
    margin = max(minimum_gutter, preview.width // 14)
    runs: list[tuple[int, int]] = []
    indices = np.flatnonzero(blank)
    if len(indices):
        groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        for group in groups:
            left, right = int(group[0]), int(group[-1]) + 1
            if right - left < minimum_gutter or left < margin or right > preview.width - margin:
                continue
            runs.append((left, right))
    if not runs:
        return []

    selected = sorted(runs, key=lambda run: run[1] - run[0], reverse=True)[: max_columns - 1]
    cuts = sorted((left + right) / 2 for left, right in selected)
    bounds = [0.0, *cuts, float(preview.width)]
    segments = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    useful: list[tuple[float, float]] = []
    for left, right in segments:
        if right - left < preview.width * 0.14:
            continue
        segment_ink = float(ink[top:bottom, int(left): max(int(left) + 1, int(right))].mean())
        if segment_ink >= 0.002:
            useful.append((left, right))
    if not 2 <= len(useful) <= max_columns:
        return []
    ink_totals = [int(np.count_nonzero(ink[top:bottom, int(left): max(int(left) + 1, int(right))])) for left, right in useful]
    if min(ink_totals) < max(40, max(ink_totals) * 0.04):
        return []

    scale = image.width / preview.width
    result: list[tuple[int, int]] = []
    for index, (left, right) in enumerate(useful):
        source_left = 0 if index == 0 else max(0, int(round(left * scale)))
        source_right = image.width if index == len(useful) - 1 else min(image.width, int(round(right * scale)))
        if source_right > source_left:
            result.append((source_left, source_right))
    return result if 2 <= len(result) <= max_columns else []



def prepare_text_ocr_image(image: Image.Image) -> tuple[Image.Image, dict[str, int]]:
    """Return a conservative OCR working copy with sparse color marks suppressed.

    The original image remains untouched for isolated-graphic recovery. Only large,
    sparse saturated components (stamp/seal/signature-like geometry) are whitened.
    Dense colored logos and ordinary small colored text are intentionally retained so
    this helper cannot silently erase a meaningful colored text line.
    """

    preview = image.convert("RGB")
    preview.thumbnail((1200, 1200))
    rgb = np.asarray(preview)
    color = pure_color(rgb, (channel_max(rgb).astype(int) - channel_min(rgb).astype(int) > 55) & (channel_min(rgb) < 190), radius=2)
    connected, _ = ndimage.label(ndimage.binary_dilation(color, iterations=5))
    candidate_mask = np.zeros(color.shape, dtype=bool)
    candidates = 0
    for label_index, slices in enumerate(ndimage.find_objects(connected), start=1):
        if slices is None:
            continue
        ys, xs = slices
        raw = color[ys, xs] & (connected[ys, xs] == label_index)
        pixels = int(raw.sum())
        width = xs.stop - xs.start
        height = ys.stop - ys.start
        density = pixels / max(1, width * height)
        area_share = width * height / max(1, preview.width * preview.height)
        # These bounds are deliberately stricter than graphic extraction. A large
        # sparse mark is safe to suppress for OCR; a dense logo or text-shaped
        # colored component stays visible to recognition.
        if pixels < 120 or height < 50 or width < 60:
            continue
        if not 0.002 <= area_share <= 0.20 or density > 0.18:
            continue
        local = candidate_mask[ys, xs]
        local |= ndimage.binary_dilation(raw, iterations=1)
        candidates += 1
    if not candidates:
        return image, {"ocr_color_mask_candidates": 0, "ocr_color_masked_pixels": 0}

    scaled = Image.fromarray((candidate_mask.astype(np.uint8) * 255), mode="L").resize(image.size, Image.Resampling.NEAREST)
    full_rgb = np.asarray(image.convert("RGB")).copy()
    full_mask = np.asarray(scaled) > 0
    # Restrict whitening to genuinely saturated/dark pixels inside the selected
    # component geometry. Black body text intersecting the mark is not erased.
    saturation = channel_max(full_rgb).astype(int) - channel_min(full_rgb).astype(int)
    # Faint stamp edges are colored too; very dark pixels are black text under the mark.
    full_mask = ndimage.binary_dilation(full_mask, iterations=2)
    selected = full_mask & (saturation > 22) & (channel_max(full_rgb) > 80)
    masked_pixels = int(selected.sum())
    if not masked_pixels:
        return image, {"ocr_color_mask_candidates": candidates, "ocr_color_masked_pixels": 0}
    full_rgb[selected] = 255
    return Image.fromarray(full_rgb, mode="RGB"), {
        "ocr_color_mask_candidates": candidates,
        "ocr_color_masked_pixels": masked_pixels,
    }

PICTURE_ROLES = ("graphic", "logo", "emblem")


def _is_picture(graphic: Graphic) -> bool:
    """Emblems and logos own the text drawn inside them; sparse colored marks
    (stamps, signatures, pen marks) lie over document text instead."""
    if graphic.role in ("logo", "emblem") or graphic.source_region == PICTURE_SOURCE:
        return True
    return graphic.role == "graphic" and graphic.source_region == "raster_graphic:black_unresolved"


def words_outside_pictures(words: list[Word], graphics: list[Graphic]) -> tuple[list[Word], int]:
    """Weak text recognized inside an emblem or logo is part of that picture.

    Inside a stamp, weak words that have no confident neighbour on their line
    outside the stamp are remnants of the stamp lettering, not document text.
    """
    pictures = [graphic.box for graphic in graphics if _is_picture(graphic)]
    stamps = [graphic.box for graphic in graphics if not _is_picture(graphic)]

    def stamp_remnant(word: Word) -> bool:
        stamp = next((box for box in stamps if box.contains_center(word.box)), None)
        if stamp is None:
            return False
        if word.source_region == "ocr:color_ink":
            return True  # lettering of the stamp itself
        if not any(character.isalpha() for character in word.text):
            return True
        if word.confidence >= 0.6:
            return False
        return not any(other.line_id == word.line_id and other.confidence >= 0.75 and not stamp.contains_center(other.box) for other in words)

    drawn = [graphic.box for graphic in graphics if graphic.source_region == PICTURE_SOURCE]

    def running_text(word: Word, box: Box) -> bool:
        """The word's line goes on outside the picture: a signature drawn over text, not text of a picture."""
        return sum(1 for other in words if other.line_id == word.line_id and other.confidence >= 0.75
                   and not box.contains_center(other.box)) >= 2

    kept = [
        word for word in words
        if not any(box.contains_center(word.box) and not running_text(word, box) for box in drawn)
        and not (word.confidence < 0.75 and any(box.contains_center(word.box) and not running_text(word, box) for box in pictures))
        and not stamp_remnant(word)
    ]
    return kept, len(words) - len(kept)


def blank_obscured_words(words: list[Word], issues: list[StructuralIssue], protected=()) -> tuple[list[Word], list[StructuralIssue], int]:
    """A weak word under a signature that stands apart from running text (strokes of the signature
    read as letters) becomes an empty field. Obscured words inside running text stay errors: a gap in a
    sentence is not acceptable; so do words under a seal or stamp (`protected`): printed text there
    ("Директор") must be read, or the document is refused."""
    obscured = {id(issue.box) for issue in issues if issue.code == "unresolved_graphic_text_overlap"
                and issue.box is not None and not any(box.contains_center(issue.box) for box in protected)}
    result: list[Word] = []
    blanked: set[int] = set()
    for word in words:
        reach = word.box.height * 2.5
        if id(word.box) in obscured and not any(
            other is not word and other.line_id == word.line_id and other.confidence >= 0.75 and id(other.box) not in obscured
            and other.box.x0 - reach <= word.box.x1 and word.box.x0 <= other.box.x1 + reach
            for other in words
        ):
            length = max(3, int(round(word.box.width / max(1.0, word.box.height * 0.5))))
            result.append(Word("_" * length, word.box, 1.0, size=word.size, line_id=word.line_id, source_page=word.source_page, source_region="blank_field"))
            blanked.add(id(word.box))
        else:
            result.append(word)
    remaining = [issue for issue in issues if not (issue.code == "unresolved_graphic_text_overlap" and id(issue.box) in blanked)]
    return result, remaining, len(blanked)


def _ink_asset(image: Image.Image, box: Box, *, width: float, height: float, mode: str) -> bytes:
    """Crop at full resolution with a transparent background.

    mode "color": only colored ink (stamps, signatures) is opaque.
    mode "dark":  paper becomes transparent, ink keeps its tone (emblems, logos).
    """
    sx, sy = image.width / width, image.height / height
    left, top = max(0, int(box.x0 * sx)), max(0, int(box.y0 * sy))
    right, bottom = min(image.width, int(np.ceil(box.x1 * sx))), min(image.height, int(np.ceil(box.y1 * sy)))
    crop = np.asarray(image.convert("RGB").crop((left, top, right, bottom))).astype(np.int16)
    if mode == "color":
        saturation = channel_max(crop) - channel_min(crop)
        alpha = np.clip((saturation - 20) * 255 / 45, 0, 255)
    else:
        luminance = crop.mean(axis=2)
        paper = float(np.quantile(luminance, 0.90)) if luminance.size else 255.0
        alpha = np.clip((paper - luminance - 12) * 255 / 90, 0, 255)
    rgba = np.dstack([crop.clip(0, 255), alpha]).astype(np.uint8)
    return png_bytes(Image.fromarray(rgba, mode="RGBA"))


def _separator(left: str, right: str) -> str:
    if not left or not right:
        return ""
    no_before = ",.;:!?%)]}»›，。！？、；：）》」』】؟؛،"
    no_after = "([{«‹（《「『【"
    if left[-1] in no_after or right[0] in no_before:
        return ""
    left_han = bool(script_counts(left[-1:]).get("Han"))
    right_han = bool(script_counts(right[:1]).get("Han"))
    if left_han and (right_han or right[0].isalnum()):
        return ""
    if right_han and left[-1].isalnum():
        return ""
    return " "


def _cell_text(words: list[Word]) -> str:
    groups: dict[tuple[int, ...], list[Word]] = defaultdict(list)
    for word in words:
        groups[word.line_id or (round(word.box.y0 / 4),)].append(word)
    lines = sorted(groups.values(), key=lambda group: (min(word.box.y0 for word in group), min(word.box.x0 for word in group)))
    text = ""
    for group in lines:
        rtl = direction(" ".join(word.text for word in group)) == "rtl"
        ordered = sorted(group, key=lambda word: word.box.x0, reverse=rtl)
        line = ""
        for word in ordered:
            line += _separator(line, word.text) + word.text
        text += _separator(text, line) + line
    return text.strip()


def _preview_box(box: Box, *, width: float, height: float, preview_width: int, preview_height: int, pad: int = 0) -> tuple[int, int, int, int]:
    x0 = int(np.floor(box.x0 / max(1.0, width) * preview_width)) - pad
    y0 = int(np.floor(box.y0 / max(1.0, height) * preview_height)) - pad
    x1 = int(np.ceil(box.x1 / max(1.0, width) * preview_width)) + pad
    y1 = int(np.ceil(box.y1 / max(1.0, height) * preview_height)) + pad
    return max(0, x0), max(0, y0), min(preview_width, x1), min(preview_height, y1)


def _masked_asset(image: Image.Image, preview_mask: np.ndarray, ys: slice, xs: slice, preview_size: tuple[int, int]) -> bytes:
    preview_width, preview_height = preview_size
    xfactor, yfactor = image.width / preview_width, image.height / preview_height
    left = max(0, int(np.floor(xs.start * xfactor)))
    top = max(0, int(np.floor(ys.start * yfactor)))
    right = min(image.width, int(np.ceil(xs.stop * xfactor)))
    bottom = min(image.height, int(np.ceil(ys.stop * yfactor)))
    crop = image.convert("RGBA").crop((left, top, right, bottom))
    local_mask = preview_mask if preview_mask.shape == (ys.stop - ys.start, xs.stop - xs.start) else preview_mask[ys, xs]
    alpha_small = Image.fromarray((local_mask.astype(np.uint8) * 255), mode="L")
    alpha = alpha_small.resize(crop.size, Image.Resampling.NEAREST)
    crop.putalpha(alpha)
    return png_bytes(crop)


def _role_for_color(box: Box, pixels: int, preview_area: int) -> tuple[str, float]:
    aspect = box.width / max(1.0, box.height)
    density = pixels / max(1, preview_area)
    if aspect >= 2.4 and box.height <= 90:
        return "signature", 0.78
    if 0.65 <= aspect <= 1.45 and box.width >= 22 and box.height >= 22 and density < 0.40:
        return "seal", 0.76
    if density >= 0.22:
        return "logo", 0.72
    return "graphic", 0.62




def _unresolved_borderless_table_issue(words: list[Word], tables: list[Table], *, width: float, height: float) -> StructuralIssue | None:
    """Return a high-confidence unresolved borderless raster-table signal.

    This intentionally detects only compact three-or-more-column grids with
    repeated horizontal alignment. Two-column material is too easily confused
    with ordinary document columns and is left to the understanding layer.
    """

    if tables or len(words) < 9:
        return None
    # Recognition confidence measures transcription reliability, not whether ink
    # physically occupies a position on the page. Dropping low-confidence words
    # here creates fake blank gutters inside normal paragraphs (especially RTL
    # lines with mixed-script numeric tokens). Use every nonempty word box for
    # *geometry*, while requiring enough reliable observations before asserting
    # an unresolved grid. Never turn a poorly recognized token into free space.
    candidates = [
        word for word in words
        if word.text.strip() and word.box.x1 > word.box.x0 and word.box.y1 > word.box.y0
    ]
    if len(candidates) < 9 or sum(word.confidence >= 0.60 for word in candidates) < 9:
        return None
    heights = [word.box.height for word in candidates if word.box.height > 0]
    if not heights:
        return None
    median_height = float(np.median(heights))
    y_tolerance = max(3.0, median_height * 0.72)
    rows: list[list[Word]] = []
    for word in sorted(candidates, key=lambda item: ((item.box.y0 + item.box.y1) / 2, item.box.x0)):
        center = (word.box.y0 + word.box.y1) / 2
        if not rows:
            rows.append([word])
            continue
        previous_center = float(np.mean([(item.box.y0 + item.box.y1) / 2 for item in rows[-1]]))
        if abs(center - previous_center) <= y_tolerance:
            rows[-1].append(word)
        else:
            rows.append([word])

    row_cells: list[tuple[list[float], list[str], list[Word]]] = []
    split_gap = max(22.0, median_height * 2.0)
    for row in rows:
        ordered = sorted(row, key=lambda item: item.box.x0)
        groups: list[list[Word]] = []
        for word in ordered:
            if not groups or word.box.x0 - max(item.box.x1 for item in groups[-1]) > split_gap:
                groups.append([word])
            else:
                groups[-1].append(word)
        # The fail-closed detector intentionally handles only a compact
        # three-column raster grid. Four-or-more repeated vertical groups are
        # too easily confused with ordinary multi-column document layout and
        # must remain available to the column understanding path.
        if len(groups) != 3:
            continue
        starts = [min(item.box.x0 for item in group) for group in groups]
        texts = [" ".join(item.text for item in group) for group in groups]
        if max((len(text) for text in texts), default=0) > 48:
            continue
        row_cells.append((starts, texts, [item for group in groups for item in group]))

    if len(row_cells) < 3:
        return None
    # Find a contiguous run of at least three rows whose first three anchors are
    # stable. This avoids treating free-form text columns as a borderless grid.
    best: list[tuple[list[float], list[str], list[Word]]] = []
    anchor_tolerance = max(12.0, width * 0.035)
    for start in range(len(row_cells)):
        current = [row_cells[start]]
        reference = row_cells[start][0][:3]
        for candidate in row_cells[start + 1:]:
            anchors = candidate[0][:3]
            if len(anchors) < 3 or any(abs(left - right) > anchor_tolerance for left, right in zip(reference, anchors)):
                break
            current.append(candidate)
            reference = [float(np.mean([row[0][index] for row in current])) for index in range(3)]
        if len(current) > len(best):
            best = current
    if len(best) < 3:
        return None
    all_words = [word for row in best for word in row[2]]
    box = Box.enclosing([word.box for word in all_words])
    if box.width < width * 0.35 or box.height < median_height * 2.2:
        return None
    confidence = min(0.92, 0.72 + 0.04 * len(best))
    return StructuralIssue(
        "unresolved_borderless_raster_table",
        "Repeated multi-column alignment suggests a borderless raster table, but safe cell reconstruction is unavailable.",
        "error", confidence, box,
    )



def _unresolved_ruled_raster_table_issue(
    gray: np.ndarray, words: list[Word], tables: list[Table], *,
    width: float, height: float,
) -> StructuralIssue | None:
    """Flag visible multirow raster grids missed by the conservative table detector.

    A true ruled table can be flattened when its vertical strokes are faint,
    skewed or interrupted. This is a *source-pixel* completeness guard, not a
    reconstructed grid: require four closely spaced, long, parallel boundaries
    with aligned ends and independently ink-bearing bands between them. The
    arbitrary source text and its transcription are never consulted. Restrict
    this narrow guard to pages where no editable table was recovered at all.
    """
    if tables or len(words) < 9 or gray.size == 0:
        return None
    image_height, image_width = gray.shape
    if image_width < 240 or image_height < 240:
        return None
    ink = gray < 205  # Include lightly printed anti-aliased grid strokes.
    length = max(35, image_width // 18)
    horizontal = ndimage.maximum_filter1d(
        ndimage.minimum_filter1d(ink, length, axis=1), length, axis=1,
    )
    indices = np.flatnonzero(horizontal.mean(axis=1) >= 0.55)
    if len(indices) < 4:
        return None
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 2) + 1)
    rule_rows: list[tuple[int, int, int]] = []
    for group in groups:
        if len(group) == 0:
            continue
        coverage = horizontal[group].mean(axis=0) >= 0.5
        x_locations = np.flatnonzero(coverage)
        if len(x_locations) < image_width * 0.55:
            continue
        rule_rows.append((int(round(float(group.mean()))), int(x_locations[0]), int(x_locations[-1])))
    if len(rule_rows) < 4:
        return None
    for start in range(len(rule_rows) - 3):
        candidate = rule_rows[start:start + 4]
        gaps = [candidate[j + 1][0] - candidate[j][0] for j in range(3)]
        if not all(10 <= gap <= image_height * 0.065 for gap in gaps):
            continue
        if max(gaps) > min(gaps) * 1.45:
            continue
        if max(item[1] for item in candidate) - min(item[1] for item in candidate) > image_width * 0.15:
            continue
        if max(item[2] for item in candidate) - min(item[2] for item in candidate) > image_width * 0.08:
            continue
        x0 = max(item[1] for item in candidate)
        x1 = min(item[2] for item in candidate)
        if x1 - x0 < image_width * 0.50:
            continue
        ink_bands = 0
        for top, bottom in zip(candidate, candidate[1:]):
            y0, y1 = top[0] + 3, bottom[0] - 3
            if y1 <= y0:
                continue
            # Ink must occupy more than one cell-width region inside multiple
            # rows. Empty horizontal dividers alone are not evidence of a table.
            thirds = np.linspace(x0, x1, 4, dtype=int)
            occupied = sum(
                float((gray[y0:y1, thirds[col]:thirds[col+1]] < 170).mean()) >= 0.007
                for col in range(3) if thirds[col+1] > thirds[col]
            )
            if occupied >= 2:
                ink_bands += 1
        if ink_bands < 2:
            continue
        sx, sy = width / image_width, height / image_height
        return StructuralIssue(
            'unresolved_ruled_raster_table',
            'A source-pixel ruled table is visible but no editable table was recovered.',
            'error', 0.86,
            Box(x0 * sx, candidate[0][0] * sy, x1 * sx, candidate[-1][0] * sy),
        )
    return None


def detect_regions(
    image: Image.Image,
    words: list[Word],
    *,
    index: int,
    width: float,
    height: float,
    color_graphics: list[Graphic] | None = None,
) -> tuple[list[Table], list[Graphic], dict[str, object], list[StructuralIssue]]:
    """Detect ruled tables and isolated raster marks.

    Returns model observations plus explicit retained/rejected/unresolved graphic
    diagnostics. Black-ink recovery is intentionally limited to isolated,
    signature-like residual ink after recognized text and ruled lines are masked.
    """

    preview = image.convert("RGB")
    preview.thumbnail((1600, 1600))
    rgb = np.asarray(preview)
    gray = np.asarray(preview.convert("L"))
    # Faint printed rules remain visible to a person but are often lighter than
    # the original 150-level cutoff. Bound the line threshold below the page's
    # prevailing background so dim photographs do not become solid grids.
    background = float(np.quantile(gray, 0.65))
    # Only bright, paper-like scans may use the more permissive threshold.
    # Photographed parchment/book spreads have broad, uneven gray regions and
    # the permissive threshold can fabricate ruled grids out of text strokes.
    rule_threshold = 205 if background >= 220 else 150
    binary = gray < rule_threshold
    horizontal = ndimage.maximum_filter1d(ndimage.minimum_filter1d(binary, max(25, preview.width // 25), axis=1), max(25, preview.width // 25), axis=1)
    vertical = ndimage.maximum_filter1d(ndimage.minimum_filter1d(binary, max(25, preview.height // 30), axis=0), max(25, preview.height // 30), axis=0)
    labels, _ = ndimage.label(horizontal | vertical)
    tables: list[Table] = []
    sx, sy = width / preview.width, height / preview.height
    for slices in ndimage.find_objects(labels):
        if slices is None:
            continue
        ys, xs = slices
        if xs.stop - xs.start < preview.width * 0.18 or ys.stop - ys.start < 30:
            continue
        region_h = horizontal[ys, xs]
        region_v = vertical[ys, xs]
        rows = [ys.start + value for value in _centers(region_h.mean(axis=1) > 0.45)]
        columns = [xs.start + value for value in _centers(region_v.mean(axis=0) > 0.45)]
        if not 3 <= len(rows) <= 101 or not 3 <= len(columns) <= 31:
            continue
        # A pair of nearby ink rules is not a meaningful editable Word row.
        # The old 3-pixel interior probe produced empty NumPy slices here,
        # then treated incidental marks (e.g. photographed book text) as
        # unreliable table cells. Do not fabricate a table for such geometry.
        if any(bottom - top < 8 for top, bottom in zip(rows, rows[1:])):
            continue
        cells: list[Cell] = []
        for row in range(len(rows) - 1):
            column = 0
            while column < len(columns) - 1:
                span = 1
                while column + span < len(columns) - 1:
                    x = columns[column + span]
                    if _has_vertical_cell_separator(vertical, rows[row], rows[row + 1], x):
                        break
                    span += 1
                box = Box(columns[column] * sx, rows[row] * sy, columns[column + span] * sx, rows[row + 1] * sy)
                contained = [word for word in words if box.contains_center(word.box)]
                text = _cell_text(contained)
                cells.append(
                    Cell(
                        text, row, column, column_span=span, direction=direction(text), box=box,
                        source_page=index, source_region="raster_table",
                        confidence=float(np.mean([word.confidence for word in contained])) if contained else 0.5,
                    )
                )
                column += span
        tables.append(
            Table(
                Box(columns[0] * sx, rows[0] * sy, columns[-1] * sx, rows[-1] * sy),
                index, len(rows) - 1, len(columns) - 1, cells,
                [float(value * sx) for value in np.diff(columns)],
                confidence=float(np.mean([cell.confidence for cell in cells])) if cells else 0.5,
                source_region="raster_table",
            )
        )

    weak_table_cells: list[dict[str, object]] = []
    table_cells_total = 0
    for table_index, table in enumerate(tables):
        for cell in table.cells:
            table_cells_total += 1
            if cell.box is None:
                continue
            x0, y0, x1, y1 = _preview_box(cell.box, width=width, height=height, preview_width=preview.width, preview_height=preview.height, pad=-3)
            if x1 <= x0 or y1 <= y0:
                continue
            cell_ink = gray[y0:y1, x0:x1] < 190
            cell_ink_ratio = float(cell_ink.mean())
            weak = cell_ink_ratio >= 0.004 and _has_marks(cell_ink) and (not cell.text.strip() or cell.confidence < 0.60)
            if weak:
                weak_table_cells.append({
                    "table": table_index, "row": cell.row, "column": cell.column,
                    "ink_ratio": cell_ink_ratio, "confidence": float(cell.confidence),
                    "has_text": bool(cell.text.strip()),
                })

    diagnostics: dict[str, object] = {
        "tables_detected": len(tables),
        "table_cells_total": table_cells_total,
        "table_weak_cells": weak_table_cells,
        "table_weak_cell_count": len(weak_table_cells),
        "graphics_color_candidates": 0,
        "graphics_black_candidates": 0,
        "graphics_retained": 0,
        "graphics_rejected": 0,
        "graphics_unresolved": 0,
        "graphics_roles": {},
        "graphics_rejection_reasons": {},
    }
    reasons: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    graphics: list[Graphic] = []
    issues: list[StructuralIssue] = []
    unresolved_borderless = _unresolved_borderless_table_issue(words, tables, width=width, height=height)
    if unresolved_borderless is not None:
        issues.append(unresolved_borderless)
        diagnostics["unresolved_borderless_table"] = True
    else:
        diagnostics["unresolved_borderless_table"] = False
    unresolved_ruled = _unresolved_ruled_raster_table_issue(
        gray, words, tables, width=width, height=height,
    )
    if unresolved_ruled is not None:
        issues.append(unresolved_ruled)
    diagnostics["unresolved_ruled_table"] = unresolved_ruled is not None

    # Saturated color remains strong evidence of a separate document mark, but
    # only the color pixels are embedded. This prevents underlying black body
    # text from being copied into the graphic asset when a stamp overlaps it.
    color = pure_color(rgb, (channel_max(rgb).astype(int) - channel_min(rgb).astype(int) > 55) & (channel_min(rgb) < 190), radius=2)
    color_components, _ = ndimage.label(ndimage.binary_dilation(color, iterations=14))
    retained_color_mask = np.zeros(color.shape, dtype=bool)
    if color_graphics is not None:
        # Colored ink was separated before OCR; its marks are already classified.
        color_components = np.zeros_like(color_components)
        for graphic in color_graphics:
            x0, y0, x1, y1 = _preview_box(graphic.box, width=width, height=height, preview_width=preview.width, preview_height=preview.height)
            retained_color_mask[y0:y1, x0:x1] |= color[y0:y1, x0:x1]
            graphics.append(graphic)
            roles[graphic.role] += 1
    for label_index, slices in enumerate(ndimage.find_objects(color_components), start=1):
        if slices is None:
            continue
        ys, xs = slices
        diagnostics["graphics_color_candidates"] = int(diagnostics["graphics_color_candidates"]) + 1
        component = color_components[ys, xs] == label_index
        raw = color[ys, xs] & component
        pixels = int(raw.sum())
        box = Box(xs.start * sx, ys.start * sy, xs.stop * sx, ys.stop * sy)
        if pixels < 100:
            reasons["color_too_small"] += 1
            continue
        if box.area / max(1.0, width * height) > 0.30:
            reasons["color_too_large"] += 1
            continue
        if min(box.width, box.height) < 8:
            reasons["color_too_thin"] += 1
            continue
        contained_words = [word for word in words if box.contains_center(word.box) and word.confidence >= 0.75]
        if contained_words:
            text_height = float(np.median([word.box.height for word in contained_words]))
            if box.height < text_height * 1.8:
                reasons["color_text_like"] += 1
                continue
        role, role_confidence = _role_for_color(box, pixels, max(1, raw.size))
        graphics.append(
            Graphic(
                box, index,
                _ink_asset(image, box, width=width, height=height, mode="dark" if role in PICTURE_ROLES else "color"),
                description="Isolated color document mark",
                role=role,
                confidence=role_confidence,
                source_region="raster_graphic:color_mask",
            )
        )
        retained_color_mask[ys, xs] |= raw
        roles[role] += 1

    # Recover only a narrow class of isolated black signatures. Recognized body
    # text and ruled lines are masked first. Other residual black regions remain
    # unresolved rather than being guessed as decorative images.
    text_mask = np.zeros(gray.shape, dtype=bool)
    for word in words:
        x0, y0, x1, y1 = _preview_box(word.box, width=width, height=height, preview_width=preview.width, preview_height=preview.height, pad=4)
        if x1 > x0 and y1 > y0:
            text_mask[y0:y1, x0:x1] = True
    table_mask = np.zeros(gray.shape, dtype=bool)
    for table in tables:
        x0, y0, x1, y1 = _preview_box(table.box, width=width, height=height, preview_width=preview.width, preview_height=preview.height, pad=3)
        if x1 > x0 and y1 > y0:
            table_mask[y0:y1, x0:x1] = True
    protected = ndimage.binary_dilation(text_mask | table_mask | horizontal | vertical | color, iterations=2)
    residual = (gray < 105) & ~protected
    connected = dilate_rect(residual, 5, 17)  # a 3×9 rectangle applied twice
    black_components, _ = ndimage.label(connected)
    for label_index, slices in enumerate(ndimage.find_objects(black_components), start=1):
        if slices is None:
            continue
        ys, xs = slices
        raw = residual[ys, xs] & (black_components[ys, xs] == label_index)
        pixels = int(raw.sum())
        if pixels < 55:
            continue
        diagnostics["graphics_black_candidates"] = int(diagnostics["graphics_black_candidates"]) + 1
        box = Box(xs.start * sx, ys.start * sy, xs.stop * sx, ys.stop * sy)
        area_share = box.area / max(1.0, width * height)
        aspect = box.width / max(1.0, box.height)
        density = pixels / max(1, raw.size)
        contained_words = [word for word in words if box.contains_center(word.box)]
        middle = (box.y0 + box.y1) / 2
        line_neighbours = sum(
            1 for word in words
            if word.confidence >= 0.6 and word.box.y0 <= middle <= word.box.y1 and not box.contains_center(word.box)
            and word.box.height >= box.height * 0.6
        )
        strong_signature = (
            0.00035 <= area_share <= 0.10
            and 1.8 <= aspect <= 12.0
            and 10 <= box.height <= 110
            and box.width >= 45
            and 0.008 <= density <= 0.24
            and ((box.y0 + box.y1) / 2) >= height * 0.38
            and len(contained_words) <= 1
            and line_neighbours < 2
        )
        if strong_signature:
            graphics.append(
                Graphic(
                    box, index,
                    _ink_asset(image, box, width=width, height=height, mode="dark"),
                    description="Isolated black-ink signature-like mark",
                    role="signature",
                    confidence=0.72,
                    source_region="raster_graphic:black_signature_heuristic",
                    issues=[
                        StructuralIssue(
                            "heuristic_black_signature",
                            "Black-ink signature recovery is based on isolated stroke geometry, not handwriting recognition.",
                            "info", 0.72, box,
                        )
                    ],
                )
            )
            roles["signature"] += 1
        elif (
            0.00025 <= area_share <= 0.16
            and 0.50 <= aspect <= 2.0
            and box.width >= 30 and box.height >= 30
            and 0.006 <= density <= 0.28
            and len(contained_words) == 0
        ):
            # Preserve compact isolated black graphics without inventing a
            # signature/logo/diagram role. The uncertainty remains explicit in
            # the UDM and acceptance diagnostics.
            issue = StructuralIssue(
                "unresolved_graphic_role",
                "An isolated black graphic was retained, but its semantic role could not be classified confidently.",
                "warning", 0.40, box,
            )
            graphics.append(
                Graphic(
                    box, index,
                    _ink_asset(image, box, width=width, height=height, mode="dark"),
                    description="Isolated black graphic with unresolved role",
                    role="graphic", confidence=0.40,
                    source_region="raster_graphic:black_unresolved",
                    issues=[issue],
                )
            )
            roles["graphic"] += 1
            diagnostics["graphics_unresolved"] = int(diagnostics["graphics_unresolved"]) + 1
        elif 0.00025 <= area_share <= 0.16 and box.width >= 30 and box.height >= 8 and len(contained_words) <= 2:
            diagnostics["graphics_unresolved"] = int(diagnostics["graphics_unresolved"]) + 1
            issues.append(
                StructuralIssue(
                    "unresolved_raster_graphic_region",
                    "An isolated black-ink region was detected but could not be preserved safely as a standalone graphic.",
                    "warning", 0.35, box,
                )
            )
        else:
            reasons["black_not_signature_like"] += 1

    # A retained stamp/seal/signature can physically obscure otherwise editable
    # text. If the intersected word is already low-confidence, do not turn that
    # uncertainty into a successful DOCX with a plausible but wrong character.
    # High-confidence text near a graphic remains allowed.
    unresolved_overlap = 0
    pictures = [graphic.box for graphic in graphics if _is_picture(graphic)]
    for word in words:
        if word.confidence >= 0.75 or not word.text.strip():
            continue
        if any(picture.contains_center(word.box) for picture in pictures):
            continue
        x0, y0, x1, y1 = _preview_box(
            word.box, width=width, height=height,
            preview_width=preview.width, preview_height=preview.height,
        )
        if x1 <= x0 or y1 <= y0:
            continue
        overlap = retained_color_mask[y0:y1, x0:x1]
        if overlap.size and int(overlap.sum()) >= 5 and float(overlap.mean()) >= 0.01:
            unresolved_overlap += 1
            issues.append(
                StructuralIssue(
                    "unresolved_graphic_text_overlap",
                    "A retained color mark intersects low-confidence editable text; the obscured text cannot be recovered safely.",
                    "error", float(word.confidence), word.box,
                )
            )
    diagnostics["graphics_text_overlap_unresolved"] = unresolved_overlap

    diagnostics["graphics_retained"] = len(graphics)
    diagnostics["graphics_rejected"] = sum(reasons.values())
    diagnostics["graphics_roles"] = dict(roles)
    diagnostics["graphics_rejection_reasons"] = dict(reasons)
    return tables, graphics, diagnostics, issues


PICTURE_SOURCE = "raster_graphic:picture"


def _text_lines(ink: np.ndarray, text_h: float) -> int:
    """Number of text-line-like bands: ink rows of about a letter height between empty gaps."""
    profile = ink.mean(axis=1) > 0.01
    bands = 0
    run = 0
    for filled in list(profile) + [False]:
        if filled:
            run += 1
        else:
            if 0.5 * text_h <= run <= 2.5 * text_h:
                bands += 1
            run = 0
    return bands


def _has_marks(ink: np.ndarray) -> bool:
    """Something written in a cell: at least two marks of letter size, not scanner dust."""
    labels, count = ndimage.label(ink)
    if count < 2:
        return False
    sizes = [slices[0].stop - slices[0].start for slices in ndimage.find_objects(labels) if slices is not None]
    return sum(size >= 4 for size in sizes) >= 2


def _letters_only(ink: np.ndarray, text_h: float) -> bool:
    """Most of the ink is in many letter-sized marks: a text line read badly (under a seal), not an emblem."""
    labels, count = ndimage.label(ink)
    if count < 8:
        return False
    in_letters = 0
    letters = 0
    for index, slices in enumerate(ndimage.find_objects(labels), start=1):
        hh, ww = slices[0].stop - slices[0].start, slices[1].stop - slices[1].start
        if 0.3 * text_h <= hh <= 1.5 * text_h and ww <= 1.5 * text_h:
            letters += 1
            in_letters += int((labels[slices] == index).sum())
    return letters >= 8 and in_letters >= 0.7 * ink.sum()


def find_pictures(image: Image.Image, words: list[Word], *, index: int, width: float, height: float,
                  color_mask: np.ndarray | None = None) -> list[Graphic]:
    """Emblems, logos and other black-ink pictures.

    Ink is joined at the scale of a text line; a large blob that confident words
    barely cover is a picture. Paragraphs are large blobs too, but words cover them.
    """
    preview = image.convert("L")
    preview.thumbnail((1200, 1200))
    gray = np.asarray(preview)
    sx, sy = preview.width / width, preview.height / height
    paper = float(np.quantile(gray, 0.9))
    ink = gray < min(170.0, paper * 0.7)
    if color_mask is not None:
        # Black ink only: the pale trace of a blue seal next to printed words would glue them into one "picture".
        ink &= ~np.asarray(Image.fromarray(color_mask).resize(preview.size, Image.Resampling.NEAREST))
    strong = [word for word in words if word.confidence >= 0.75 and word.source_region != "blank_field"]
    text_h = float(np.median([word.box.height * sy for word in strong])) if strong else preview.height / 70
    covered = np.zeros_like(ink)
    for word in strong:
        covered[max(0, int(word.box.y0 * sy)):int(np.ceil(word.box.y1 * sy)), max(0, int(word.box.x0 * sx)):int(np.ceil(word.box.x1 * sx))] = True
    # Ruling lines (underlines, header rules, table borders) are not pictures and
    # would glue nearby text lines into one blob.
    run = max(8, int(text_h * 4))
    rules = ndimage.maximum_filter1d(ndimage.minimum_filter1d(ink, run, axis=1), run, axis=1)
    rules |= ndimage.maximum_filter1d(ndimage.minimum_filter1d(ink, run, axis=0), run, axis=0)
    ink = ink & ~ndimage.binary_dilation(rules, iterations=1)
    step = max(1, int(text_h * 0.6))
    joined = close_rect(ink, step, step)
    labels, _ = ndimage.label(joined)
    pictures: list[Graphic] = []
    for label_index, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        ys, xs = slices
        h, w = ys.stop - ys.start, xs.stop - xs.start
        if h < 3 * text_h or w < 3 * text_h or h * w > 0.35 * gray.size:
            continue
        component = labels[ys, xs] == label_index
        blob_ink = ink[ys, xs] & component
        if blob_ink.sum() < 0.05 * h * w:
            continue  # a frame or a few scattered strokes, not a picture
        if (covered[ys, xs] & blob_ink).sum() > 0.3 * blob_ink.sum():
            continue  # text
        if _text_lines(blob_ink, text_h) >= 3 or _letters_only(blob_ink, text_h):
            continue  # rows of letters, or a line made of letter-sized marks: text, even if it was not read
        inside = sum(1 for word in strong if xs.start <= (word.box.x0 + word.box.x1) / 2 * sx < xs.stop and ys.start <= (word.box.y0 + word.box.y1) / 2 * sy < ys.stop)
        if inside > 3:
            continue  # several confident words: a text block, perhaps with a thick rule or frame
        box = Box(xs.start / sx, ys.start / sy, xs.stop / sx, ys.stop / sy)
        role = "emblem" if box.y1 < height * 0.35 else "graphic"
        pictures.append(Graphic(box, index, _ink_asset(image, box, width=width, height=height, mode="dark"),
                                description="Black-ink picture", role=role, confidence=0.8, source_region=PICTURE_SOURCE))
    return pictures
