"""Quality metrics for the benchmark. Images are H x W x C float32 in 0-1, equal shapes."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.morphology import skeletonize

from crisp.bench.scene import TextLabel

_WHITELIST = "0123456789.xRM"
_MIN_CROP_HEIGHT = 40  # judge-side normalisation, applied to every method equally


def _grey_u8(img: np.ndarray) -> np.ndarray:
    grey = img.mean(axis=2) if img.shape[2] > 1 else img[..., 0]
    return np.clip(np.round(grey * 255), 0, 255).astype(np.uint8)


def psnr(out: np.ndarray, gt: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(gt, out, data_range=1))


def ssim(out: np.ndarray, gt: np.ndarray) -> float:
    if out.shape[2] == 1:
        return float(structural_similarity(gt[..., 0], out[..., 0], data_range=1))
    return float(structural_similarity(gt, out, data_range=1, channel_axis=-1))


def _edges(img: np.ndarray) -> np.ndarray:
    return cv2.Canny(_grey_u8(img), 50, 150) > 0


def _distance_to(mask: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)


def edge_error(out: np.ndarray, gt: np.ndarray) -> float:
    """Symmetric chamfer distance between edge maps, in pixels."""
    e_out, e_gt = _edges(out), _edges(gt)
    if not e_out.any() or not e_gt.any():
        return float("nan")
    to_gt, to_out = _distance_to(e_gt), _distance_to(e_out)
    return float((to_gt[e_out].mean() + to_out[e_gt].mean()) / 2)


def _mean_stroke_width(img: np.ndarray) -> float:
    grey = _grey_u8(img)
    threshold, _ = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = grey <= threshold
    if not ink.any():
        return float("nan")
    skeleton = skeletonize(ink)
    if not skeleton.any():
        return float("nan")
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    return float(2 * dist[skeleton].mean())


def stroke_width_error(out: np.ndarray, gt: np.ndarray) -> float:
    """Relative error of the mean stroke width."""
    w_out, w_gt = _mean_stroke_width(out), _mean_stroke_width(gt)
    if np.isnan(w_out) or np.isnan(w_gt) or w_gt == 0:
        return float("nan")
    return abs(w_out - w_gt) / w_gt


def tesseract_available() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
    except Exception:  # noqa: BLE001  (not installed, not on PATH, or import failed)
        return False
    return True


def legibility(out: np.ndarray, labels: list[TextLabel]) -> dict[float, float] | None:
    """Fraction of labels Tesseract reads exactly, grouped by text height. None if unavailable."""
    if not tesseract_available():
        return None
    import pytesseract

    grey = _grey_u8(out)
    hits: dict[float, list[bool]] = {}
    for label in labels:
        x0, y0, x1, y1 = label.box
        pad = max(1, round(0.25 * (y1 - y0)))
        crop = grey[max(0, y0 - pad) : y1 + pad, max(0, x0 - pad) : x1 + pad]
        if crop.shape[0] < _MIN_CROP_HEIGHT:
            factor = _MIN_CROP_HEIGHT / crop.shape[0]
            crop = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
        crop = cv2.copyMakeBorder(crop, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        text = pytesseract.image_to_string(
            crop, config=f"--psm 7 -c tessedit_char_whitelist={_WHITELIST}"
        )
        hits.setdefault(label.height_in, []).append("".join(text.split()) == label.string)
    return {h: sum(v) / len(v) for h, v in hits.items()}
