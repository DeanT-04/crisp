"""Synthetic engineering-drawing scenes with exact ground truth."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TEXT_HEIGHTS = (4, 6, 8, 12, 16, 24, 32, 40)

_CAP_RATIO = 0.72  # digit height as a fraction of the font size
_REF = 8  # reference magnification used to measure text extents
_MARGIN = 2.0  # input px kept clear around every label
_EDGE = 6.0  # input px kept between labels and the image edge


@dataclass(frozen=True)
class Line:
    x0: float
    y0: float
    x1: float
    y1: float
    width: float


@dataclass(frozen=True)
class Arc:
    cx: float
    cy: float
    r: float
    start_deg: float
    end_deg: float
    width: float


@dataclass(frozen=True)
class Text:
    x: float  # anchor ("la": left, ascender) in input px
    y: float
    height: float  # digit height in input px
    string: str


@dataclass(frozen=True)
class Scene:
    width: int
    height: int
    lines: tuple[Line, ...]
    arcs: tuple[Arc, ...]
    texts: tuple[Text, ...]


@dataclass(frozen=True)
class TextLabel:
    string: str
    height_in: float
    box: tuple[int, int, int, int]  # output px (x0, y0, x1, y1)


@cache
def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=max(1, size))


def _extent(string: str, height: float) -> tuple[float, float, float, float]:
    """Ink box (l, t, r, b) relative to the text anchor, in input px."""
    font = _font(round(height / _CAP_RATIO * _REF))
    left, top, right, bottom = font.getbbox(string, anchor="la")
    return (left / _REF, top / _REF, right / _REF, bottom / _REF)


def _random_string(rng: np.random.Generator, max_len: int) -> str:
    while True:
        kind = int(rng.integers(0, 5))
        if kind == 0:
            s = str(int(rng.integers(1, 999)))
        elif kind == 1:
            s = f"{int(rng.integers(1, 99))}.{int(rng.integers(0, 9))}"
        elif kind == 2:
            s = f"R{int(rng.integers(1, 50))}"
        elif kind == 3:
            s = f"M{int(rng.integers(3, 24))}x{rng.choice(['1', '1.5', '2'])}"
        else:
            s = f"{int(rng.integers(1, 99))}x{int(rng.integers(1, 99))}"
        if len(s) <= max_len:
            return s


def _hatch(x0: float, y0: float, x1: float, y1: float, spacing: float, width: float) -> list[Line]:
    lines = []
    c = x0 - y1
    while c < x1 - y0:
        lo, hi = max(x0, y0 + c), min(x1, y1 + c)
        if lo < hi:
            lines.append(Line(lo, lo - c, hi, hi - c, width))
        c += spacing
    return lines


def _overlaps(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def random_scene(seed: int, width: int, height: int) -> Scene:
    rng = np.random.default_rng(seed)
    uni = rng.uniform
    lines: list[Line] = []
    # Border
    for a, b in (
        ((2, 2), (width - 3, 2)),
        ((width - 3, 2), (width - 3, height - 3)),
        ((width - 3, height - 3), (2, height - 3)),
        ((2, height - 3), (2, 2)),
    ):
        lines.append(Line(*a, *b, 1.5))
    # Free lines: horizontal, vertical and angled
    for i in range(int(rng.integers(4, 8))):
        x0, y0 = uni(10, width - 10), uni(10, height - 10)
        if i % 3 == 0:
            x1, y1 = uni(10, width - 10), y0
        elif i % 3 == 1:
            x1, y1 = x0, uni(10, height - 10)
        else:
            x1, y1 = uni(10, width - 10), uni(10, height - 10)
        lines.append(Line(x0, y0, x1, y1, uni(0.5, 2.0)))
    # Hatched rectangle
    hw, hh = width * uni(0.12, 0.2), height * uni(0.12, 0.2)
    hx, hy = uni(10, width - 10 - hw), uni(10, height - 10 - hh)
    lines += [
        Line(hx, hy, hx + hw, hy, 1.0),
        Line(hx + hw, hy, hx + hw, hy + hh, 1.0),
        Line(hx + hw, hy + hh, hx, hy + hh, 1.0),
        Line(hx, hy + hh, hx, hy, 1.0),
    ]
    lines += _hatch(hx, hy, hx + hw, hy + hh, 5.0, 0.5)
    # Dimension lines: shaft plus two arrowheads of two strokes each
    for _ in range(2):
        xa = uni(12, width * 0.5)
        xb = uni(xa + width * 0.2, width - 12)
        y = uni(12, height - 12)
        lines.append(Line(xa, y, xb, y, 0.8))
        for x, sign in ((xa, 1), (xb, -1)):
            for ang in (20, -20):
                rad = math.radians(ang)
                lines.append(Line(x, y, x + sign * 6 * math.cos(rad), y + 6 * math.sin(rad), 0.8))
    # Arcs and circles
    arcs = []
    n_arcs = int(rng.integers(2, 5))
    for i in range(n_arcs):
        r = uni(min(width, height) * 0.06, min(width, height) * 0.2)
        cx, cy = uni(r + 8, width - r - 8), uni(r + 8, height - r - 8)
        start = 0.0 if i == 0 else float(uni(0, 180))
        end = 360.0 if i == 0 else start + float(uni(60, 270))
        arcs.append(Arc(cx, cy, r, start, end, uni(0.5, 2.0)))
    # Text, largest first so it is easier to place without overlap
    placed: list[tuple[float, float, float, float]] = []
    texts: list[Text] = []
    for h in sorted((t for t in TEXT_HEIGHTS if t <= height / 4), reverse=True):
        max_len = 4 if h >= 32 else 7
        for _ in range(500):
            s = _random_string(rng, max_len)
            left, top, right, bottom = _extent(s, h)
            lo_x, hi_x = _EDGE - left, width - _EDGE - right
            lo_y, hi_y = _EDGE - top, height - _EDGE - bottom
            if hi_x <= lo_x or hi_y <= lo_y:
                continue
            x, y = uni(lo_x, hi_x), uni(lo_y, hi_y)
            box = (x + left - _MARGIN, y + top - _MARGIN, x + right + _MARGIN, y + bottom + _MARGIN)
            if any(_overlaps(box, p) for p in placed):
                continue
            placed.append(box)
            texts.append(Text(float(x), float(y), float(h), s))
            break
    return Scene(width, height, tuple(lines), tuple(arcs), tuple(texts))


def render_scene(
    scene: Scene, scale: int, supersample: int = 4
) -> tuple[np.ndarray, list[TextLabel]]:
    """Render at `scale` x the input size. Returns H x W x 1 float32 (ink 0, paper 1) + labels."""
    k = scale * supersample
    size = (scene.width * k, scene.height * k)
    canvas = Image.new("L", size, 255)
    draw = ImageDraw.Draw(canvas)

    def px(width: float) -> int:
        return max(1, round(width * k))

    for ln in scene.lines:
        draw.line([(ln.x0 * k, ln.y0 * k), (ln.x1 * k, ln.y1 * k)], fill=0, width=px(ln.width))
    for a in scene.arcs:
        box = [(a.cx - a.r) * k, (a.cy - a.r) * k, (a.cx + a.r) * k, (a.cy + a.r) * k]
        draw.arc(box, a.start_deg, a.end_deg, fill=0, width=px(a.width))
    labels = []
    pad = round(1.5 * k)
    for t in scene.texts:
        font = _font(round(t.height / _CAP_RATIO * k))
        bx0, by0, bx1, by1 = draw.textbbox((t.x * k, t.y * k), t.string, font=font, anchor="la")
        draw.rectangle([bx0 - pad, by0 - pad, bx1 + pad, by1 + pad], fill=255)
        draw.text((t.x * k, t.y * k), t.string, fill=0, font=font, anchor="la")
        out_w, out_h = scene.width * scale, scene.height * scale
        box = (
            max(0, math.floor(bx0 / supersample)),
            max(0, math.floor(by0 / supersample)),
            min(out_w, math.ceil(bx1 / supersample)),
            min(out_h, math.ceil(by1 / supersample)),
        )
        labels.append(TextLabel(t.string, t.height, box))
    small = canvas.resize((scene.width * scale, scene.height * scale), Image.Resampling.BOX)
    return (np.asarray(small, dtype=np.float32) / 255.0)[..., None], labels
