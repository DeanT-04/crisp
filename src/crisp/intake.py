"""Open image and PDF files as normalised Pages."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import Image, ImageOps, ImageSequence

from crisp.types import Page, Raster, UnsupportedInput

SUPPORTED_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".pdf"}
)

# Big scans are legitimate input; the resource guard (output size vs RAM) is the real limit.
Image.MAX_IMAGE_PIXELS = None

_FORMATS = {
    "JPEG": "jpeg",
    "MPO": "jpeg",
    "PNG": "png",
    "TIFF": "tiff",
    "BMP": "bmp",
    "WEBP": "webp",
}
_MIN_DPI = 20.0


def open_file(path: Path) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedInput(f"unsupported file type: {path.suffix or path.name}")
    if suffix == ".pdf":
        return _open_pdf(path)
    return _open_raster(path)


def _open_pdf(path: Path) -> list[Page]:
    try:
        pdf = pdfium.PdfDocument(path)
    except pdfium.PdfiumError as e:
        if "password" in str(e).lower():
            raise UnsupportedInput("password-protected PDF") from e
        raise UnsupportedInput(f"could not read PDF: {e}") from e
    try:
        count = len(pdf)
        if count == 0:
            raise UnsupportedInput("PDF has no pages")
        return [_pdf_page(path, pdf, i, count) for i in range(count)]
    finally:
        pdf.close()


def _invisible_text(obj) -> bool:
    """OCR layers are invisible text (render mode 3) and do not make a page vector."""
    mode = pdfium_c.FPDFTextObj_GetTextRenderMode(obj.raw)
    return mode == pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE


def _pdf_page(path: Path, pdf: pdfium.PdfDocument, index: int, count: int) -> Page:
    page = pdf[index]
    width, height = page.get_size()
    area = width * height
    image_cov = other_cov = 0.0
    largest = None  # (area, object)
    for obj in page.get_objects(max_depth=15):
        left, bottom, right, top = obj.get_bounds()
        w = max(0.0, min(right, width) - max(left, 0.0))
        h = max(0.0, min(top, height) - max(bottom, 0.0))
        if obj.type == pdfium_c.FPDF_PAGEOBJ_IMAGE:
            image_cov += w * h
            if largest is None or w * h > largest[0]:
                largest = (w * h, obj, right - left)
        elif obj.type == pdfium_c.FPDF_PAGEOBJ_PATH:
            other_cov += w * h
        elif obj.type == pdfium_c.FPDF_PAGEOBJ_TEXT:
            other_cov += 0.0 if _invisible_text(obj) else w * h
    image_only = area > 0 and image_cov / area >= 0.9 and other_cov / area < 0.01
    if image_only and largest is not None:
        _, obj, bounds_width = largest
        try:
            pil = obj.get_bitmap(render=False).to_pil()
        except Exception:  # exotic filters: fall back to rendering the page directly
            pil = None
        if pil is not None:
            raster = _to_raster(pil)
            dpi = raster.pixels.shape[1] / (bounds_width / 72.0) if bounds_width > 0 else None
            return Page(path, index, count, raster, dpi, False, None, "pdf")
    return Page(path, index, count, None, 72.0, True, (width, height), "pdf")


def _open_raster(path: Path) -> list[Page]:
    try:
        with Image.open(path) as img:
            fmt = _FORMATS.get(img.format or "", (img.format or "").lower())
            # Only TIFF is treated as multi-page; MPO/animated WebP contribute their first frame.
            frames = ImageSequence.Iterator(img) if img.format == "TIFF" else [img]
            decoded = []
            for frame in frames:
                frame.load()
                dpi = _dpi(frame)
                decoded.append((_to_raster(ImageOps.exif_transpose(frame)), dpi))
    except UnsupportedInput:
        raise
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as e:
        raise UnsupportedInput(f"could not read image: {e}") from e
    return [
        Page(path, i, len(decoded), raster, dpi, False, None, fmt)
        for i, (raster, dpi) in enumerate(decoded)
    ]


def _dpi(img: Image.Image) -> float | None:
    dpi = img.info.get("dpi")
    if not dpi:
        return None
    try:
        value = float(dpi[0])
    except (TypeError, ValueError):
        return None
    return value if value >= _MIN_DPI else None


def _to_raster(img: Image.Image) -> Raster:
    mode = img.mode
    if mode in ("I;16", "I;16L", "I;16B", "I"):
        arr = np.asarray(img, dtype=np.float32) / 65535.0
        return Raster(np.clip(arr, 0, 1)[..., None], None)
    if mode == "P" and "transparency" in img.info or mode in ("PA", "RGBa", "La"):
        img = img.convert("RGBA" if mode != "La" else "LA")
        mode = img.mode
    if mode in ("1", "L"):
        arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        return Raster(arr[..., None], None)
    if mode == "LA":
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return Raster(arr[..., :1], arr[..., 1])
    if mode == "RGBA":
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return Raster(np.ascontiguousarray(arr[..., :3]), np.ascontiguousarray(arr[..., 3]))
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return Raster(arr, None)
