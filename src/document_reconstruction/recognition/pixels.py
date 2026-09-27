"""Fast per-pixel helpers (same results as the plain numpy forms, several times faster)."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def channel_max(rgb: np.ndarray) -> np.ndarray:
    """rgb.max(axis=2): a reduction over the last axis of H×W×3 is slow in numpy."""
    return np.maximum(np.maximum(rgb[..., 0], rgb[..., 1]), rgb[..., 2])


def channel_min(rgb: np.ndarray) -> np.ndarray:
    """rgb.min(axis=2)."""
    return np.minimum(np.minimum(rgb[..., 0], rgb[..., 1]), rgb[..., 2])


def png_bytes(image: Image.Image) -> bytes:
    """A picture for the DOCX: fast zlib level (the file is a little larger, the pixels the same)."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=3)
    return buffer.getvalue()


def dilate_rect(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """ndimage.binary_dilation(mask, structure=np.ones((height, width))) as two 1-D maximum filters.

    Same pixels (binary dilation reflects the element, hence the origin shift for even sizes),
    but the cost does not grow with the rectangle.
    """
    from scipy import ndimage
    origin = (-1 if height % 2 == 0 else 0, -1 if width % 2 == 0 else 0)
    return ndimage.maximum_filter(mask, size=(height, width), mode="constant", cval=0, origin=origin).astype(bool)


def erode_rect(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """ndimage.binary_erosion(mask, structure=np.ones((height, width))) (outside the image counts as empty)."""
    from scipy import ndimage
    return ndimage.minimum_filter(mask, size=(height, width), mode="constant", cval=0).astype(bool)


def close_rect(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """ndimage.binary_closing(mask, structure=np.ones((height, width)))."""
    return erode_rect(dilate_rect(mask, height, width), height, width)


PURE_HUE_SHARE = 0.65


def pure_color(rgb: np.ndarray, color: np.ndarray, radius: int = 3) -> np.ndarray:
    """Colored pixels that belong to one-hue ink (a stamp, a signature, a colored line).

    A scan with color fringing (JPEG chroma halos, misregistered channels) puts red and blue
    around every black letter: such components have no dominant hue (≈ half red, half blue),
    while stamp and pen ink is one hue throughout (measured: 0.45–0.51 against 1.00).
    """
    from scipy import ndimage
    if not color.any():
        return color
    labels, count = ndimage.label(ndimage.binary_dilation(color, iterations=radius))
    dominant = np.argmax(rgb, axis=2)
    keys = labels[color].astype(np.int64) * 3 + dominant[color]
    counts = np.bincount(keys, minlength=3 * (count + 1)).reshape(-1, 3)
    totals = counts.sum(axis=1)
    keep = (counts.max(axis=1) >= PURE_HUE_SHARE * np.maximum(totals, 1)) & (totals >= 20)
    keep[0] = False
    return color & keep[labels]
