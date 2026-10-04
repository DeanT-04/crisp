"""Process one file end to end: intake -> target -> engine -> write."""

from __future__ import annotations

import time
from pathlib import Path

from crisp.engines.baseline import BaselineEngine
from crisp.engines.vector import VectorEngine
from crisp.intake import open_file
from crisp.resources import check_resources
from crisp.target import resolve_target
from crisp.types import (
    CrispError,
    FileOutcome,
    Options,
    Page,
    PageResult,
    Raster,
    Target,
    TargetSpec,
)
from crisp.write import output_path, write_png

DEFAULT_OPTIONS = Options()


def upscale_page(page: Page, target: Target) -> tuple[Raster, str]:
    engine = VectorEngine() if page.has_vectors else BaselineEngine()
    return engine.upscale(page, target), engine.name


def process(
    path: Path,
    spec: TargetSpec,
    options: Options = DEFAULT_OPTIONS,
    rel_parent: Path = Path(),
) -> FileOutcome:
    """Never raises for per-file problems; they are reported in the outcome."""
    outcome = FileOutcome(source=path)
    try:
        pages = open_file(path)
    except CrispError as e:
        outcome.error = str(e)
        return outcome
    except Exception as e:  # noqa: BLE001
        outcome.error = f"unexpected error: {type(e).__name__}: {e}"
        return outcome
    self_path = path.resolve()
    for page in pages:
        outcome.pages.append(_process_page(page, spec, options, rel_parent, self_path))
    return outcome


def _process_page(
    page: Page, spec: TargetSpec, options: Options, rel_parent: Path, self_path: Path
) -> PageResult:
    result = PageResult(page=page.index, status="failed")
    started = time.perf_counter()
    try:
        target = resolve_target(spec, page)
        result.target = target
        out = output_path(
            page.source, page.index, page.page_count, target.label, options.out_dir, rel_parent
        )
        result.output = out
        resolved = out.resolve()
        if resolved == self_path or resolved in options.protected:
            result.reason = "would overwrite an input file"
        elif out.exists() and not options.overwrite:
            result.status = "skipped"
            result.reason = "output exists"
        else:
            channels = page.raster.pixels.shape[2] if page.raster is not None else 3
            check_resources(target, channels, out.parent)
            raster, engine = upscale_page(page, target)
            write_png(raster, out, target.out_dpi)
            result.status = "done"
            result.engine = engine
    except CrispError as e:
        result.reason = str(e)
    except Exception as e:  # noqa: BLE001
        result.reason = f"unexpected error: {type(e).__name__}: {e}"
    result.seconds = time.perf_counter() - started
    return result
