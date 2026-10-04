"""Engine interface: image + target in, bigger image out."""

from __future__ import annotations

from typing import Protocol

from crisp.types import Page, Raster, Target


class Engine(Protocol):
    name: str

    def upscale(self, page: Page, target: Target) -> Raster: ...
