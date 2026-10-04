"""Turn --scale / --dpi / --size into an exact output size."""

from __future__ import annotations

import re

from crisp.types import Page, Target, TargetError, TargetSpec

MAX_SCALE = 16.0

# Portrait (w, h) in millimetres.
PAPER_SIZES_MM = {
    "A0": (841.0, 1189.0),
    "A1": (594.0, 841.0),
    "A2": (420.0, 594.0),
    "A3": (297.0, 420.0),
    "A4": (210.0, 297.0),
    "A5": (148.0, 210.0),
    "Letter": (215.9, 279.4),
    "Tabloid": (279.4, 431.8),
}
_PAPER_BY_LOWER = {k.lower(): k for k in PAPER_SIZES_MM}
_SIZE_RE = re.compile(r"^([A-Za-z0-9]+)@(\d+)$")


def parse_target(scale: float | None, dpi: int | None, size: str | None) -> TargetSpec:
    if sum(x is not None for x in (scale, dpi, size)) > 1:
        raise ValueError("Use only one of --scale, --dpi or --size.")
    if size is not None:
        m = _SIZE_RE.match(size.strip())
        paper = _PAPER_BY_LOWER.get(m.group(1).lower()) if m else None
        if m is None or paper is None or int(m.group(2)) <= 0:
            raise ValueError("--size must look like A3@300 (paper: A0–A5, Letter, Tabloid).")
        return TargetSpec("size", dpi=int(m.group(2)), paper=paper)
    if dpi is not None:
        if dpi <= 0:
            raise ValueError("--dpi must be a positive number.")
        return TargetSpec("dpi", dpi=dpi)
    value = 2.0 if scale is None else float(scale)
    if not 1.0 < value <= MAX_SCALE:
        raise ValueError("--scale must be greater than 1 and at most 16.")
    return TargetSpec("scale", scale=value)


def resolve_target(spec: TargetSpec, page: Page) -> Target:
    # Vector pages are measured in points (reference resolution 72 DPI), rasters in pixels.
    base_w, base_h = page.size_pt if page.size_pt is not None else page.size_px
    if spec.kind == "scale":
        scale = float(spec.scale)
        out_dpi = page.dpi * scale if page.dpi else None
        label = f"{scale:g}x"
    elif spec.kind == "dpi":
        if not page.dpi:
            raise TargetError("input DPI unknown; use --scale instead")
        scale = spec.dpi / page.dpi
        out_dpi = float(spec.dpi)
        label = f"{spec.dpi}dpi"
    else:
        paper_w, paper_h = PAPER_SIZES_MM[spec.paper]
        if base_w > base_h:
            paper_w, paper_h = paper_h, paper_w
        scale = min(paper_w / 25.4 * spec.dpi / base_w, paper_h / 25.4 * spec.dpi / base_h)
        out_dpi = float(spec.dpi)
        label = f"{spec.paper}-{spec.dpi}dpi"
    if spec.kind != "scale":
        if scale <= 1.0:
            raise TargetError(f"already at or above the requested size (would need {scale:.2f}×)")
        if scale > MAX_SCALE:
            raise TargetError(f"would need {scale:.1f}×, above the 16× limit")
    out_size = (max(1, round(base_w * scale)), max(1, round(base_h * scale)))
    return Target(scale=scale, out_size=out_size, out_dpi=out_dpi, label=label)
