"""Refuse jobs that would not fit in memory or on disk."""

from __future__ import annotations

import shutil
from pathlib import Path

import psutil

from crisp.types import ResourceError, Target

_GB = 1024**3
_RAM_FRACTION = 0.25
_FLOAT_COPIES = 4  # working copies of the output array held at once


def check_resources(
    target: Target,
    channels: int,
    out_dir: Path,
    available_ram: int | None = None,
    free_disk: int | None = None,
    workers: int = 1,
) -> None:
    w, h = target.out_size
    working = w * h * channels * 4 * _FLOAT_COPIES
    ram = psutil.virtual_memory().available if available_ram is None else available_ram
    if working > ram * _RAM_FRACTION / max(1, workers):
        raise ResourceError(
            f"needs about {working / _GB:.1f} GB of memory, more than 25% of the "
            f"{ram / _GB:.1f} GB available (shared by {max(1, workers)} worker(s)); "
            "very large outputs need tiling, which arrives in a later version"
        )
    png_estimate = w * h * channels // 4
    if free_disk is None:
        probe = out_dir
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        free_disk = shutil.disk_usage(probe).free
    if png_estimate > free_disk:
        raise ResourceError(
            f"needs about {png_estimate / _GB:.1f} GB of disk space "
            f"but only {free_disk / _GB:.1f} GB is free"
        )
