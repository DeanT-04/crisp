"""Upscaling methods the benchmark compares. Each maps (H x W x C float32, scale) to the output."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from crisp.engines.baseline import lanczos_resize
from crisp.pipeline import upscale_page
from crisp.target import resolve_target
from crisp.types import Page, Raster, TargetSpec

Method = Callable[[np.ndarray, int], np.ndarray]

REALESRGAN_MODELS = ("realesrgan-x4plus", "realesrgan-x4plus-anime")


class MethodUnavailable(Exception):  # noqa: N818
    """A comparator that is installed but cannot run (e.g. no Vulkan GPU)."""


def _cv2_method(interpolation: int) -> Method:
    def method(img: np.ndarray, scale: int) -> np.ndarray:
        h, w = img.shape[:2]
        out = cv2.resize(img, (w * scale, h * scale), interpolation=interpolation)
        if out.ndim == 2:
            out = out[..., None]
        return np.clip(out, 0, 1).astype(np.float32)

    return method


nearest = _cv2_method(cv2.INTER_NEAREST)
bicubic = _cv2_method(cv2.INTER_CUBIC)


def lanczos(img: np.ndarray, scale: int) -> np.ndarray:
    """Lanczos-3 in sRGB, i.e. what most software does."""
    h, w = img.shape[:2]
    return np.clip(lanczos_resize(img, (w * scale, h * scale)), 0, 1).astype(np.float32)


def crisp_method(img: np.ndarray, scale: int) -> np.ndarray:
    page = Page(Path("<bench>"), 0, 1, Raster(img, None), None, False, None, "png")
    raster, _ = upscale_page(page, resolve_target(TargetSpec("scale", scale=scale), page))
    return raster.pixels


def passes_needed(scale: int) -> int:
    """Number of 4x Real-ESRGAN passes needed to reach at least `scale`."""
    passes = 1
    while 4**passes < scale:
        passes += 1
    return passes


def _exe(root: Path) -> Path:
    name = "realesrgan-ncnn-vulkan.exe" if sys.platform == "win32" else "realesrgan-ncnn-vulkan"
    return root / "bench" / "tools" / name


def realesrgan(model: str, root: Path) -> Method | None:
    exe = _exe(root)
    if not exe.exists():
        return None

    def method(img: np.ndarray, scale: int) -> np.ndarray:
        h, w = img.shape[:2]
        grey = img.shape[2] == 1
        rgb = np.repeat(img, 3, axis=2) if grey else img
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in.png"
            Image.fromarray(np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)).save(src)
            for i in range(passes_needed(scale)):
                dst = Path(tmp) / f"out{i}.png"
                try:
                    run = subprocess.run(
                        [str(exe), "-i", str(src), "-o", str(dst), "-n", model, "-s", "4"],
                        cwd=exe.parent,
                        capture_output=True,
                        text=True,
                    )
                except OSError as e:
                    raise MethodUnavailable(str(e)) from e
                if run.returncode != 0 or not dst.exists():
                    raise MethodUnavailable(run.stderr.strip() or f"exit code {run.returncode}")
                src = dst
            out = np.asarray(Image.open(src).convert("RGB"), dtype=np.float32) / 255.0
        out = cv2.resize(out, (w * scale, h * scale), interpolation=cv2.INTER_AREA)
        if grey:
            out = out.mean(axis=2, keepdims=True)
        return np.clip(out, 0, 1).astype(np.float32)

    return method


def available_methods(root: Path, ai: bool) -> dict[str, Method]:
    methods: dict[str, Method] = {
        "nearest": nearest,
        "bicubic": bicubic,
        "lanczos": lanczos,
        "crisp": crisp_method,
    }
    if ai:
        for model in REALESRGAN_MODELS:
            method = realesrgan(model, root)
            if method is not None:
                methods[model] = method
    return methods
