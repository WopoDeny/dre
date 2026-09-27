"""Selective page recovery using bounded previews and conservative evidence."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage
from scipy.spatial import ConvexHull

from ..core import DeadlineBudget


@dataclass(frozen=True)
class TransformStep:
    """A recorded image-space transform; OCR boxes are later scaled to page points."""

    kind: str
    source_size: tuple[int, int]
    target_size: tuple[int, int]
    parameters: dict[str, float | int | str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "source_size": list(self.source_size),
            "target_size": list(self.target_size),
            "parameters": dict(self.parameters),
        }


@dataclass
class RecoveryResult:
    image: Image.Image
    operations: list[str] = field(default_factory=list)
    transforms: list[TransformStep] = field(default_factory=list)


def _outside_edge_fraction(gray: np.ndarray, corners: np.ndarray) -> float:
    """Estimate content-like structure outside a candidate sheet.

    Long page-boundary edges and resampling stripes are not treated as readable
    external content. Smaller repeated components (annotations, neighboring text,
    signatures) remain part of the safety signal.
    """
    mask_image = Image.new("1", (gray.shape[1], gray.shape[0]), 0)
    ImageDraw.Draw(mask_image).polygon([tuple(point) for point in corners], fill=1)
    inside = ndimage.binary_dilation(np.asarray(mask_image, dtype=bool), iterations=4)
    values = gray.astype(float)
    gradient = np.hypot(ndimage.sobel(values, axis=0), ndimage.sobel(values, axis=1))
    strong = gradient > 160
    strong_count = int(strong.sum())
    if not strong_count:
        return 0.0
    external = strong & ~inside
    labels, _ = ndimage.label(external)
    relevant = 0
    height, width = gray.shape
    for label_index, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        ys, xs = slices
        count = int(np.count_nonzero(labels[slices] == label_index))
        if count < 8:
            continue
        touches_frame = xs.start <= 2 or ys.start <= 2 or xs.stop >= width - 2 or ys.stop >= height - 2
        component_area = (xs.stop - xs.start) * (ys.stop - ys.start)
        boundary_like = component_area > gray.size * 0.10 or count > strong_count * 0.12
        if touches_frame or boundary_like:
            continue
        relevant += count
    return float(relevant / strong_count)


def _otsu(gray: np.ndarray) -> float:
    histogram = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = histogram.sum()
    levels = np.arange(256)
    weight = np.cumsum(histogram)
    mean = np.cumsum(histogram * levels)
    between = (mean[-1] * weight / total - mean) ** 2 / np.maximum(weight * (total - weight) / total, 1e-9)
    return float(np.argmax(between[:-1]))


def _sheet_corners(points: np.ndarray, shape: tuple[int, ...]) -> np.ndarray | None:
    """Corners of the sheet as crossings of its four sides.

    The sides are straight runs of the outline (its convex hull). A shadow that bites into a corner
    makes the paper mask lose that corner; the sides still run straight up to the bite, so their
    crossing restores it. None when the outline has no four clear sides."""
    try:
        hull = ConvexHull(points)
    except Exception:  # noqa: BLE001 - degenerate outline
        return None
    vertices = points[hull.vertices]
    center = vertices.mean(axis=0)
    sides: dict[str, list[tuple[np.ndarray, np.ndarray, float]]] = {"top": [], "bottom": [], "left": [], "right": []}
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
        dx, dy = end - start
        length = float(np.hypot(dx, dy))
        middle = (start + end) / 2
        if abs(dy) < 0.4 * abs(dx):
            sides["top" if middle[1] < center[1] else "bottom"].append((start, end, length))
        elif abs(dx) < 0.4 * abs(dy):
            sides["left" if middle[0] < center[0] else "right"].append((start, end, length))
    span = {"top": np.ptp(points[:, 0]), "bottom": np.ptp(points[:, 0]), "left": np.ptp(points[:, 1]), "right": np.ptp(points[:, 1])}
    lines = {}
    for name, edges in sides.items():
        if sum(length for _, _, length in edges) < 0.3 * span[name]:
            return None
        # The side is the line most of the outline lies on; the cut across a shadowed corner is one edge only.
        def support(edge: tuple[np.ndarray, np.ndarray, float]) -> float:
            start, end, _ = edge
            direction = (end - start) / max(1e-6, float(np.hypot(*(end - start))))
            normal = np.array([-direction[1], direction[0]])
            return sum(length for a, b, length in edges
                       if abs(float(np.dot(a - start, normal))) <= 3 and abs(float(np.dot(b - start, normal))) <= 3)
        start, end, _ = max(edges, key=support)
        lines[name] = (start, end)

    def cross(first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]) -> np.ndarray | None:
        (p1, p2), (p3, p4) = first, second
        d1, d2 = p2 - p1, p4 - p3
        denominator = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(denominator) < 1e-6:
            return None
        t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denominator
        return p1 + t * d1

    corners = [cross(lines["top"], lines["left"]), cross(lines["top"], lines["right"]),
               cross(lines["bottom"], lines["right"]), cross(lines["bottom"], lines["left"])]
    if any(corner is None for corner in corners):
        return None
    corners = np.array(corners)
    height, width = shape[:2]
    if (corners[:, 0] < -0.05 * width).any() or (corners[:, 0] > 1.05 * width).any() \
            or (corners[:, 1] < -0.05 * height).any() or (corners[:, 1] > 1.05 * height).any():
        return None
    return np.clip(corners, 0, [width - 1, height - 1])


def rectify_paper(image: Image.Image) -> tuple[Image.Image, TransformStep | None]:
    """Rectify a clear paper quadrilateral only when doing so will not crop content."""
    preview = image.convert("L")
    preview.thumbnail((600, 600))
    array = np.asarray(preview)
    border = np.concatenate((array[0], array[-1], array[:, 0], array[:, -1]))
    if np.median(border) > 185:
        return image, None
    # Paper vs desk: Otsu threshold (a shadow on the sheet must not cut it off).
    threshold = _otsu(array)
    if threshold <= np.median(border) + 10:
        return image, None
    mask = ndimage.binary_opening(ndimage.binary_fill_holes(array > threshold), iterations=2)
    labels, count = ndimage.label(mask)
    if not count:
        return image, None
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    label = int(areas.argmax())
    fraction = areas[label] / array.size
    if not 0.40 < fraction < 0.95:
        return image, None
    yy, xx = np.where(labels == label)
    points = np.column_stack((xx, yy)).astype(float)
    preview_corners = _sheet_corners(points, array.shape)
    if preview_corners is None:
        sums, differences = points.sum(axis=1), points[:, 0] - points[:, 1]
        preview_corners = np.array([points[sums.argmin()], points[differences.argmax()], points[sums.argmax()], points[differences.argmin()]])
    if len(np.unique(preview_corners, axis=0)) != 4:
        return image, None
    # If appreciable ink exists outside the candidate sheet, rectification could
    # silently crop annotations, signatures, footnotes, or a second object.
    if _outside_edge_fraction(array, preview_corners) > 0.15:
        return image, None
    corners = preview_corners * [image.width / preview.width, image.height / preview.height]
    width = int(max(np.linalg.norm(corners[1] - corners[0]), np.linalg.norm(corners[2] - corners[3])))
    height = int(max(np.linalg.norm(corners[3] - corners[0]), np.linalg.norm(corners[2] - corners[1])))
    if min(width, height) < 100 or width * height > image.width * image.height * 1.3:
        return image, None
    ratio = min(width, height) / max(width, height)
    if not 0.55 <= ratio <= 0.85:
        return image, None  # not a sheet of paper (A4 is 0.71): do not guess a crop
    target = [(0, 0), (width, 0), (width, height), (0, height)]
    matrix, values = [], []
    for (x, y), (u, v) in zip(target, corners):
        matrix.extend(([x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]))
        values.extend((u, v))
    try:
        coefficients = np.linalg.solve(np.asarray(matrix), np.asarray(values))
    except np.linalg.LinAlgError:
        return image, None
    source_size = image.size
    output = image.transform((width, height), Image.Transform.PERSPECTIVE, coefficients, resample=Image.Resampling.BICUBIC, fillcolor="white")
    step = TransformStep(
        "perspective_rectification",
        source_size,
        output.size,
        {f"output_to_input_c{i}": float(value) for i, value in enumerate(coefficients)},
    )
    return output, step


PAPER_TINT = 20         # chroma of the paper background: yellow light, a tinted sheet
PAPER_FLOOR = 150       # the darkest paper level that is still evened out
PAPER_SHADE = 40        # brightness spread of the paper background: a shadow across a photographed sheet


def flatten_paper(image: Image.Image) -> tuple[Image.Image, TransformStep | None]:
    """A photographed sheet under warm light or in a shadow: the paper is made white and even again.

    Each channel is divided by the paper background (the brightest level around every point), so
    black print stays black, a blue pen stays blue, and a yellow tint or a shadow is no longer taken
    for colored ink or for a picture."""
    small = image.convert("RGB")
    small.thumbnail((300, 300))
    array = np.asarray(small).astype(np.float32)
    size = max(5, int(max(array.shape[:2]) / 25))
    background = np.stack([ndimage.median_filter(ndimage.maximum_filter(array[:, :, channel], size=size), size=size)
                           for channel in range(3)], axis=2)
    tint = float(np.median(background.max(axis=2) - background.min(axis=2)))
    low, high = np.percentile(background.mean(axis=2), (5, 95))
    if tint < PAPER_TINT and high - low < PAPER_SHADE:
        return image, None
    # Where the "paper" is dark (the desk at an edge, a deep shadow) it is not lifted: noise there would turn to color.
    full = Image.fromarray(np.clip(background, PAPER_FLOOR, 255).astype(np.uint8)).resize(image.size, Image.Resampling.BILINEAR)
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    flat = np.clip(rgb * 255.0 / np.maximum(np.asarray(full).astype(np.float32), 1.0), 0, 255).astype(np.uint8)
    step = TransformStep("paper_flattening", image.size, image.size, {"paper_tint": round(tint, 1), "paper_shade": round(float(high - low), 1)})
    return Image.fromarray(flat, mode="RGB"), step


def recover(image: Image.Image, budget: DeadlineBudget) -> RecoveryResult:
    budget.check(stage="geometry")
    transforms: list[TransformStep] = []
    image, rectification = rectify_paper(image)
    operations: list[str] = []
    if rectification is not None:
        transforms.append(rectification)
        operations.append("paper_rectification")
        # Only a photographed sheet (the desk around it was cut away) has light to even out; a scan's
        # tint is the paper itself and stays as it is.
        image, flattening = flatten_paper(image)
        if flattening is not None:
            transforms.append(flattening)
            operations.append("paper_flattening")
    preview = image.convert("L")
    preview.thumbnail((850, 850))
    gray = np.asarray(preview)
    p5, p95 = np.percentile(gray, (5, 95))
    if p95 - p5 < 80 and p95 < 230:
        source_size = image.size
        image = ImageOps.autocontrast(image, cutoff=0.3)
        preview = ImageOps.autocontrast(preview, cutoff=0.3)
        operations.append("contrast")
        transforms.append(TransformStep("contrast_normalization", source_size, image.size, {"cutoff_percent": 0.3}))
    binary = preview.point(lambda value: 255 if value < 170 else 0)

    def score(angle: float) -> float:
        rotated = np.asarray(binary.rotate(angle, resample=Image.Resampling.NEAREST, expand=False, fillcolor=0))
        rows = (rotated > 0).sum(axis=1)
        return float(np.var(rows))

    baseline = score(0)
    if baseline > 0:
        # Coarse +/-8 degree search, followed by a bounded local refinement.
        coarse: list[tuple[float, float]] = []
        for angle in np.arange(-8, 8.01, 1.0):
            budget.check(stage="geometry")
            coarse.append((score(float(angle)), float(angle)))
        coarse_score, coarse_angle = max(coarse)
        refined: list[tuple[float, float]] = []
        for angle in np.arange(coarse_angle - 0.75, coarse_angle + 0.751, 0.25):
            budget.check(stage="geometry")
            refined.append((score(float(angle)), float(angle)))
        best_score, best_angle = max(refined)
        if 0.4 <= abs(best_angle) <= 8.75 and best_score > baseline * 1.16:
            source_size = image.size
            image = image.rotate(best_angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor="white")
            operations.append(f"deskew:{best_angle:g}")
            transforms.append(TransformStep("deskew", source_size, image.size, {"counterclockwise_degrees": best_angle}))
    budget.check(stage="geometry")
    return RecoveryResult(image, operations, transforms)
