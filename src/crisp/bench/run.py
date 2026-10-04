"""Run the benchmark: generate, degrade, upscale with every method, score."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from crisp.bench import metrics
from crisp.bench.degrade import PRESETS, degrade
from crisp.bench.methods import MethodUnavailable, available_methods, nearest
from crisp.bench.scene import TextLabel, random_scene, render_scene

_CROP = 256


@dataclass(frozen=True)
class Config:
    name: str
    scenes: int
    size: tuple[int, int]
    scales: tuple[int, ...]
    presets: tuple[str, ...]
    ocr: bool
    ai: bool


QUICK = Config("quick", 4, (256, 192), (2, 4), ("clean", "blur"), ocr=False, ai=False)
FULL = Config("full", 24, (512, 384), (2, 4, 8), ("clean", "blur", "jpeg", "scan"), True, True)


@dataclass(frozen=True)
class Case:
    scene_seed: int
    scale: int
    preset: str


@dataclass
class CaseResult:
    case: Case
    method: str
    psnr: float
    ssim: float
    edge_error: float
    stroke_width_error: float
    legibility: dict[float, float] | None
    seconds: float


@dataclass
class Sample:
    case: Case
    crops: dict[str, np.ndarray]


@dataclass
class BenchRun:
    config: Config
    results: list[CaseResult]
    samples: list[Sample]
    unavailable: dict[str, str]


def build_cases(cfg: Config) -> list[Case]:
    return [
        Case(seed, scale, preset)
        for seed in range(cfg.scenes)
        for scale in cfg.scales
        for preset in cfg.presets
    ]


def _crop(img: np.ndarray, centre: tuple[int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = max(0, min(centre[0] - _CROP // 2, w - _CROP))
    y0 = max(0, min(centre[1] - _CROP // 2, h - _CROP))
    return img[y0 : y0 + _CROP, x0 : x0 + _CROP]


def _focus(labels: list[TextLabel], gt: np.ndarray) -> tuple[int, int]:
    if not labels:
        return gt.shape[1] // 2, gt.shape[0] // 2
    x0, y0, x1, y1 = min(labels, key=lambda lb: lb.height_in).box
    return (x0 + x1) // 2, (y0 + y1) // 2


def run_benchmark(
    cfg: Config, root: Path, on_case: Callable[[int, int], None] | None = None
) -> BenchRun:
    methods = available_methods(root, cfg.ai)
    use_ocr = cfg.ocr and metrics.tesseract_available()
    cases = build_cases(cfg)
    results: list[CaseResult] = []
    samples: list[Sample] = []
    unavailable: dict[str, str] = {}
    for index, case in enumerate(cases):
        scene = random_scene(case.scene_seed, *cfg.size)
        gt, labels = render_scene(scene, case.scale)
        seed = case.scene_seed * 1000 + case.scale * 10 + cfg.presets.index(case.preset)
        low = degrade(gt, case.scale, PRESETS[case.preset], seed)
        focus = _focus(labels, gt)
        crops: dict[str, np.ndarray] = {}
        if case.scene_seed == 0:
            crops["ground truth"] = _crop(gt, focus)
            crops["input"] = _crop(nearest(low, case.scale), focus)
        for name, method in list(methods.items()):
            started = time.perf_counter()
            try:
                out = method(low, case.scale)
            except MethodUnavailable as e:
                unavailable[name] = str(e)
                del methods[name]
                continue
            seconds = time.perf_counter() - started
            results.append(
                CaseResult(
                    case=case,
                    method=name,
                    psnr=metrics.psnr(out, gt),
                    ssim=metrics.ssim(out, gt),
                    edge_error=metrics.edge_error(out, gt),
                    stroke_width_error=metrics.stroke_width_error(out, gt),
                    legibility=metrics.legibility(out, labels) if use_ocr else None,
                    seconds=seconds,
                )
            )
            if case.scene_seed == 0:
                crops[name] = _crop(out, focus)
        if crops:
            samples.append(Sample(case, crops))
        if on_case:
            on_case(index + 1, len(cases))
    return BenchRun(cfg, results, samples, unavailable)


def _fmt_height(h: float) -> str:
    return str(int(h)) if float(h).is_integer() else str(h)


def aggregate(run: BenchRun) -> dict[str, float]:
    """Mean of every metric per (scale, preset, method); NaNs are ignored."""
    values: dict[str, list[float]] = {}
    for r in run.results:
        prefix = f"{r.case.scale}/{r.case.preset}/{r.method}"
        for metric in ("psnr", "ssim", "edge_error", "stroke_width_error", "seconds"):
            values.setdefault(f"{prefix}/{metric}", []).append(getattr(r, metric))
        if r.legibility:
            values.setdefault(f"{prefix}/legibility", []).append(
                float(np.mean(list(r.legibility.values())))
            )
            for h, frac in r.legibility.items():
                values.setdefault(f"{prefix}/legibility@{_fmt_height(h)}", []).append(frac)
    agg = {}
    for key, vals in values.items():
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            agg[key] = float(arr.mean())
    return agg
