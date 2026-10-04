"""Baseline engine: Lanczos-3 resampling in linear light."""

from __future__ import annotations

import numpy as np
from PIL import Image

from crisp.color import linear_to_srgb, srgb_to_linear
from crisp.types import Page, Raster, Target


def lanczos_resize(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize H x W x C float32 to `size` (w, h). No colour conversion, output is not clipped."""
    channels = [
        np.asarray(
            Image.fromarray(np.ascontiguousarray(arr[..., c]), mode="F").resize(
                size, Image.Resampling.LANCZOS
            ),
            dtype=np.float32,
        )
        for c in range(arr.shape[2])
    ]
    return np.stack(channels, axis=-1)


class BaselineEngine:
    name = "baseline"

    def upscale(self, page: Page, target: Target) -> Raster:
        raster = page.raster
        linear = srgb_to_linear(raster.pixels)
        if raster.alpha is None:
            out = lanczos_resize(linear, target.out_size)
            return Raster(np.clip(linear_to_srgb(out), 0, 1).astype(np.float32), None)
        alpha = raster.alpha[..., None]
        premult = lanczos_resize(linear * alpha, target.out_size)
        out_alpha = lanczos_resize(alpha, target.out_size)
        safe = np.where(out_alpha > 1e-6, out_alpha, 1.0)
        color = np.where(out_alpha > 1e-6, premult / safe, 0.0)
        pixels = np.clip(linear_to_srgb(np.clip(color, 0, 1)), 0, 1).astype(np.float32)
        return Raster(pixels, np.clip(out_alpha[..., 0], 0, 1).astype(np.float32))
