"""Seeded degradation of ground-truth images, in spec 7.2 order."""

from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Degradation:
    name: str
    blur_sigma: float = 0.0  # in degraded-input pixels
    jpeg_quality: int | None = None
    noise_sigma: float = 0.0
    rotate_deg: float = 0.0
    lighting: float = 0.0


PRESETS = {
    "clean": Degradation("clean"),
    "blur": Degradation("blur", blur_sigma=1.0),
    "jpeg": Degradation("jpeg", blur_sigma=0.5, jpeg_quality=60),
    "scan": Degradation("scan", blur_sigma=0.8, noise_sigma=0.02, rotate_deg=0.5, lighting=0.15),
}


def _keep_channels(arr: np.ndarray, like: np.ndarray) -> np.ndarray:
    return arr[..., None] if arr.ndim == 2 and like.ndim == 3 else arr


def degrade(gt: np.ndarray, scale: int, d: Degradation, seed: int) -> np.ndarray:
    """Turn a ground-truth image (H x W x C) into its degraded, `scale`x smaller version."""
    img = gt.astype(np.float32)
    if d.blur_sigma > 0:
        img = _keep_channels(cv2.GaussianBlur(img, (0, 0), d.blur_sigma * scale), gt)
    h, w = img.shape[:2]
    img = _keep_channels(
        cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA), gt
    )
    if d.jpeg_quality is not None:
        u8 = np.clip(np.round(img * 255), 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(u8[..., 0] if u8.shape[2] == 1 else u8).save(
            buf, format="JPEG", quality=d.jpeg_quality
        )
        decoded = np.asarray(Image.open(io.BytesIO(buf.getvalue())), dtype=np.float32) / 255.0
        img = _keep_channels(decoded, gt)
    if d.noise_sigma > 0:
        rng = np.random.default_rng(seed)
        img = img + rng.normal(0.0, d.noise_sigma, img.shape).astype(np.float32)
    h, w = img.shape[:2]
    if d.rotate_deg:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), d.rotate_deg, 1.0)
        img = _keep_channels(
            cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
            gt,
        )
    if d.lighting:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
        img = img * (1.0 - d.lighting * r2)[..., None]
    return np.clip(img, 0.0, 1.0).astype(np.float32)
