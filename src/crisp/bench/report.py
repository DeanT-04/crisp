"""Self-contained HTML benchmark report (no external assets)."""

from __future__ import annotations

import base64
import io
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image

import crisp
from crisp.bench.baseline import Regression
from crisp.bench.run import BenchRun

# Columns: (title, metric key, formatter, best = "max" | "min" | None)
_COLUMNS = (
    ("PSNR ↑", "psnr", lambda v: f"{v:.2f}", "max"),
    ("SSIM ↑", "ssim", lambda v: f"{v:.4f}", "max"),
    ("Edge error ↓", "edge_error", lambda v: f"{v:.3f}", "min"),
    ("Stroke-width error ↓", "stroke_width_error", lambda v: f"{v:.3f}", "min"),
    ("Legibility ↑", "legibility", lambda v: f"{v * 100:.1f}%", "max"),
    ("s/img", "seconds", lambda v: f"{v:.3f}", None),
)

_CSS = """
:root{--navy:#0E1A2B;--cream:#FFF6E5;--gold:#F6B93B;--mint:#3DDC97;--red:#E8473F;--cyan:#5BC0EB}
body{background:var(--navy);color:var(--cream);font-family:system-ui,sans-serif;margin:2rem auto;
max-width:1100px;padding:0 1rem}
h1,h2,h3{color:var(--gold)}
table{border-collapse:collapse;margin:0.5rem 0 1.5rem;width:100%}
th,td{border-bottom:1px solid #2a3b55;padding:0.35rem 0.6rem;text-align:right}
th:first-child,td:first-child{text-align:left}
td.best{color:var(--mint);font-weight:700}
.banner{border:2px solid var(--red);padding:0.5rem 1rem;margin:1rem 0}
.banner h2{color:var(--red);margin:0.2rem 0}
.crops{display:flex;flex-wrap:wrap;gap:0.75rem;margin-bottom:1.5rem}
figure{margin:0;text-align:center}
figure img{width:256px;height:auto;image-rendering:pixelated;border:1px solid #2a3b55}
figcaption{color:var(--cyan);font-size:0.85rem}
.muted{color:#9fb0c8}
"""


def _png_uri(arr: np.ndarray) -> str:
    u8 = np.clip(np.round(arr * 255), 0, 255).astype(np.uint8)
    img = Image.fromarray(u8[..., 0] if u8.shape[2] == 1 else u8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _methods(run: BenchRun) -> list[str]:
    seen: list[str] = []
    for r in run.results:
        if r.method not in seen:
            seen.append(r.method)
    return seen


def _score_table(agg: dict[str, float], scale: int, preset: str, methods: list[str]) -> str:
    def key(method: str, metric: str) -> str:
        return f"{scale}/{preset}/{method}/{metric}"

    head = "".join(f"<th>{escape(title)}</th>" for title, *_ in _COLUMNS)
    rows = []
    best = {}
    for _, metric, _, rule in _COLUMNS:
        vals = [agg[key(m, metric)] for m in methods if key(m, metric) in agg]
        if vals and rule:
            best[metric] = max(vals) if rule == "max" else min(vals)
    for m in methods:
        cells = []
        for _, metric, fmt, _ in _COLUMNS:
            k = key(m, metric)
            if k not in agg:
                cells.append('<td class="muted">N/A</td>')
                continue
            cls = ' class="best"' if best.get(metric) == agg[k] else ""
            cells.append(f"<td{cls}>{fmt(agg[k])}</td>")
        rows.append(f"<tr><td>{escape(m)}</td>{''.join(cells)}</tr>")
    title = escape(f"{scale}× · {preset}")
    return f"<h3>{title}</h3><table><tr><th>Method</th>{head}</tr>{''.join(rows)}</table>"


def _height_table(agg: dict[str, float], scale: int, preset: str, methods: list[str]) -> str:
    prefix = f"{scale}/{preset}/"
    heights = sorted(
        {float(k.rsplit("@", 1)[1]) for k in agg if k.startswith(prefix) and "/legibility@" in k}
    )
    if not heights:
        return ""
    head = "".join(f"<th>{h:g} px</th>" for h in heights)
    rows = []
    for m in methods:
        cells = []
        for h in heights:
            v = agg.get(f"{prefix}{m}/legibility@{h:g}")
            cells.append('<td class="muted">N/A</td>' if v is None else f"<td>{v * 100:.0f}%</td>")
        rows.append(f"<tr><td>{escape(m)}</td>{''.join(cells)}</tr>")
    return (
        f"<h3>{escape(f'Legibility by text height · {scale}× · {preset}')}</h3>"
        f"<table><tr><th>Method</th>{head}</tr>{''.join(rows)}</table>"
    )


def write_report(
    run: BenchRun, agg: dict[str, float], regressions: list[Regression], path: Path
) -> Path:
    cfg = run.config
    methods = _methods(run)
    parts = [
        f"<h1>crisp benchmark · {escape(cfg.name)}</h1>",
        f'<p class="muted">crisp {escape(crisp.__version__)} · '
        f"{escape(datetime.now().strftime('%Y-%m-%d %H:%M'))} · {cfg.scenes} scenes of "
        f"{cfg.size[0]}×{cfg.size[1]}</p>",
    ]
    if run.unavailable:
        items = "".join(
            f"<li>{escape(n)}: {escape(why)}</li>" for n, why in run.unavailable.items()
        )
        parts.append(f'<p class="muted">Unavailable methods:</p><ul class="muted">{items}</ul>')
    if regressions:
        items = "".join(
            f"<li>{escape(r.key)}: baseline {r.baseline:.4g} → now {r.current:.4g}</li>"
            for r in regressions
        )
        parts.append(f'<div class="banner"><h2>Regressions</h2><ul>{items}</ul></div>')
    for scale in cfg.scales:
        for preset in cfg.presets:
            parts.append(_score_table(agg, scale, preset, methods))
            parts.append(_height_table(agg, scale, preset, methods))
    parts.append("<h2>Zoomed crops (smallest text)</h2>")
    for sample in run.samples:
        figures = "".join(
            f'<figure><img alt="{escape(name)}" src="{_png_uri(crop)}">'
            f"<figcaption>{escape(name)}</figcaption></figure>"
            for name, crop in sample.crops.items()
        )
        caption = escape(f"{sample.case.scale}× · {sample.case.preset}")
        parts.append(f"<h3>{caption}</h3><div class='crops'>{figures}</div>")
    html = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>crisp benchmark · {escape(cfg.name)}</title><style>{_CSS}</style></head>"
        f"<body>{''.join(parts)}</body></html>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
