"""Committed baseline scores and regression detection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Absolute tolerances. edge_error uses a relative tolerance instead.
TOLERANCES = {"psnr": 0.1, "ssim": 0.002, "legibility": 0.01, "stroke_width_error": 0.01}
EDGE_REL_TOLERANCE = 0.05
_HIGHER_IS_BETTER = {"psnr", "ssim", "legibility"}


@dataclass(frozen=True)
class Regression:
    key: str
    baseline: float
    current: float


def load_baseline(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_baseline(path: Path, config_name: str, agg: dict[str, float]) -> None:
    data = load_baseline(path)
    data[config_name] = agg
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compare(
    agg: dict[str, float], baseline_section: dict[str, float], method: str = "crisp"
) -> list[Regression]:
    """Metrics of `method` that got worse than the baseline by more than their tolerance."""
    regressions = []
    for key, base in baseline_section.items():
        parts = key.split("/")
        if len(parts) != 4 or parts[2] != method:
            continue
        metric = parts[3].split("@")[0]
        if metric in _HIGHER_IS_BETTER | {"edge_error", "stroke_width_error"} and key not in agg:
            regressions.append(Regression(key, base, float("nan")))  # metric disappeared
            continue
        if key not in agg:
            continue
        current = agg[key]
        if metric in _HIGHER_IS_BETTER:
            worse = current < base - TOLERANCES[metric]
        elif metric == "edge_error":
            worse = current > base * (1 + EDGE_REL_TOLERANCE)
        elif metric == "stroke_width_error":
            worse = current > base + TOLERANCES[metric]
        else:  # seconds and anything else is informational
            continue
        if worse:
            regressions.append(Regression(key, base, current))
    return regressions
