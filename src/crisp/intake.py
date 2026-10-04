"""Open image and PDF files as normalised Pages."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
    raise NotImplementedError


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
