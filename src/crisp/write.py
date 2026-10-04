"""Output naming and atomic PNG writing."""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from crisp.types import Raster

# Matches the stem of a file crisp wrote: a@4x, a@1.5x, plan-p3@600dpi, a@A3-300dpi.
OUTPUT_PATTERN = re.compile(
    r"@(\d+(\.\d+)?x|\d+dpi|(A[0-5]|Letter|Tabloid)-\d+dpi)$", re.IGNORECASE
)


def output_path(
    source: Path,
    page_index: int,
    page_count: int,
    label: str,
    out_dir: Path | None,
    rel_parent: Path = Path(),
) -> Path:
    page_part = f"-p{page_index + 1}" if page_count > 1 else ""
    name = f"{source.stem}{page_part}@{label}.png"
    folder = out_dir / rel_parent if out_dir is not None else source.parent
    return folder / name


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)


def write_png(raster: Raster, path: Path, dpi: float | None) -> None:
    """Write an 8-bit PNG atomically: a failed write never leaves a file at `path`."""
    pixels = _to_uint8(raster.pixels)
    channels = pixels.shape[2]
    if raster.alpha is not None:
        pixels = np.concatenate([pixels, _to_uint8(raster.alpha)[..., None]], axis=2)
    mode = {(1, False): "L", (1, True): "LA", (3, False): "RGB", (3, True): "RGBA"}[
        (channels, raster.alpha is not None)
    ]
    img = Image.fromarray(pixels[..., 0] if pixels.shape[2] == 1 else pixels, mode=mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        img.save(tmp, format="PNG", **({"dpi": (dpi, dpi)} if dpi is not None else {}))
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
