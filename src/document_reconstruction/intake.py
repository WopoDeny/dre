"""Accepted inputs: PDF, or one image (JPEG, PNG, WEBP, HEIC) that becomes a one-page PDF.

Images get their EXIF rotation applied, transparency filled with white, very
large photos reduced, and are placed on an A4 page (landscape when wider).
"""

from __future__ import annotations

import io

import pymupdf
from PIL import Image, ImageOps

from .core import EngineError, ErrorCode

# Long side of an image after reduction: A4 at ~340 dpi, enough for OCR.
MAX_IMAGE_SIDE = 4000
A4 = (595.0, 842.0)


def image_kind(payload: bytes) -> str | None:
    head = payload[:32]
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[4:8] == b"ftyp" and head[8:12] in (b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1", b"heim", b"heis"):
        return "heic"
    return None


def image_to_pdf(payload: bytes) -> bytes:
    if image_kind(payload) == "heic":
        try:
            from pillow_heif import register_heif_opener
        except ImportError as cause:
            raise EngineError(ErrorCode.INVALID_INPUT, "HEIC images are not supported by this installation.", stage="preflight") from cause
        register_heif_opener()
    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
    except Exception as cause:
        raise EngineError(ErrorCode.INVALID_INPUT, "The image could not be decoded.", stage="preflight") from cause
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        image = Image.new("RGB", rgba.size, "white")
        image.paste(rgba, mask=rgba.split()[-1])
    image = image.convert("RGB")
    if max(image.size) > MAX_IMAGE_SIDE:
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "PNG" if image_kind(payload) == "png" else "JPEG", quality=92)
    width, height = (A4[1], A4[0]) if image.width > image.height * 1.05 else A4
    scale = min(width / image.width, height / image.height)
    placed_width, placed_height = image.width * scale, image.height * scale
    document = pymupdf.open()
    page = document.new_page(width=width, height=height)
    left, top = (width - placed_width) / 2, (height - placed_height) / 2
    page.insert_image(pymupdf.Rect(left, top, left + placed_width, top + placed_height), stream=buffer.getvalue())
    return document.tobytes()


def to_pdf(payload: bytes) -> bytes:
    """The payload as PDF bytes; images are wrapped, anything else is left to the PDF check."""
    return image_to_pdf(payload) if image_kind(payload) else payload
