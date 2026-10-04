"""sRGB <-> linear-light conversion (IEC 61966-2-1)."""

import numpy as np


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    safe = np.maximum(x, 0.0)
    return np.where(safe <= 0.0031308, safe * 12.92, 1.055 * safe ** (1 / 2.4) - 0.055).astype(
        np.float32
    )
