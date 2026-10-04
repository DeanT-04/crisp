"""Core data types shared across crisp."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

Status = Literal["done", "skipped", "failed"]


@dataclass(eq=False)
class Raster:
    pixels: np.ndarray  # float32 HxWxC, C in {1,3}, 0-1
    alpha: np.ndarray | None = None  # float32 HxW, 0-1


@dataclass(eq=False)
class Page:
    source: Path
    index: int  # 0-based
    page_count: int
    raster: Raster | None  # None for vector PDF pages
    dpi: float | None
    has_vectors: bool
    size_pt: tuple[float, float] | None  # PDF page size in points, vector pages only
    source_format: str  # "png" | "jpeg" | "tiff" | "bmp" | "webp" | "pdf"

    @property
    def size_px(self) -> tuple[int, int]:
        """(w, h). Vector pages: size_pt rounded, i.e. pixels at 72 DPI."""
        if self.raster is not None:
            h, w = self.raster.pixels.shape[:2]
            return (w, h)
        assert self.size_pt is not None
        return (round(self.size_pt[0]), round(self.size_pt[1]))


@dataclass(frozen=True)
class TargetSpec:
    kind: Literal["scale", "dpi", "size"]
    scale: float | None = None
    dpi: int | None = None
    paper: str | None = None  # "A0".."A5", "Letter", "Tabloid"


@dataclass(frozen=True)
class Target:
    scale: float
    out_size: tuple[int, int]  # (w, h)
    out_dpi: float | None
    label: str  # "4x", "1.5x", "600dpi", "A3-300dpi"


@dataclass(frozen=True)
class Options:
    out_dir: Path | None = None
    overwrite: bool = False
    protected: frozenset[Path] = frozenset()  # resolved input paths that must never be written


@dataclass
class PageResult:
    page: int
    status: Status
    output: Path | None = None
    engine: str | None = None  # "baseline" | "vector"
    target: Target | None = None
    reason: str | None = None
    seconds: float = 0.0


@dataclass
class FileOutcome:
    source: Path
    pages: list[PageResult] = field(default_factory=list)
    error: str | None = None  # file-level failure (could not open)

    @property
    def status(self) -> Status:
        if self.error or any(p.status == "failed" for p in self.pages):
            return "failed"
        if self.pages and all(p.status == "skipped" for p in self.pages):
            return "skipped"
        return "done"


@dataclass
class BatchSummary:
    outcomes: list[FileOutcome]

    def counts(self) -> dict[str, int]:
        counts = {"done": 0, "skipped": 0, "failed": 0}
        for o in self.outcomes:
            counts[o.status] += 1
        return counts

    @property
    def exit_code(self) -> int:
        return 1 if self.counts()["failed"] else 0


class CrispError(Exception):
    """Message is shown to the user as-is."""


class UnsupportedInput(CrispError): ...


class TargetError(CrispError): ...


class ResourceError(CrispError): ...
