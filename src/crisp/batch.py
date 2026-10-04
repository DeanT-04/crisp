"""Find input files and process them, in parallel, with per-file isolation."""

from __future__ import annotations

import glob
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

from crisp.intake import SUPPORTED_EXTENSIONS
from crisp.pipeline import DEFAULT_OPTIONS, process
from crisp.types import BatchSummary, FileOutcome, Options, TargetSpec
from crisp.write import OUTPUT_PATTERN


@dataclass(frozen=True)
class InputItem:
    path: Path
    rel_parent: Path


def _supported(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS


def _not_crisp_output(p: Path) -> bool:
    return not OUTPUT_PATTERN.search(p.stem)


def discover_inputs(paths: list[Path], recursive: bool = False) -> list[InputItem]:
    """Expand files, folders and glob patterns. Keeps argument order, drops duplicates."""
    items: list[InputItem] = []
    for arg in paths:
        if arg.is_file():
            items.append(InputItem(arg, Path()))
        elif arg.is_dir():
            found = sorted((arg.rglob("*") if recursive else arg.iterdir()), key=str)
            items.extend(
                InputItem(p, p.parent.relative_to(arg) if recursive else Path())
                for p in found
                if _supported(p) and _not_crisp_output(p)
            )
        elif any(ch in str(arg) for ch in "*?["):
            matches = sorted(Path(m) for m in glob.glob(str(arg)))
            items.extend(
                InputItem(p, Path()) for p in matches if _supported(p) and _not_crisp_output(p)
            )
        else:
            raise FileNotFoundError(f"Path not found: {arg}")
    seen: set[Path] = set()
    unique = []
    for item in items:
        key = item.path.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return _drop_outputs_of_present_sources(unique)


def _drop_outputs_of_present_sources(items: list[InputItem]) -> list[InputItem]:
    """A shell-expanded glob also lists earlier crisp outputs; drop those whose source is given."""
    sources = {(i.path.resolve().parent, i.path.stem.lower()) for i in items}
    kept = []
    for item in items:
        m = OUTPUT_PATTERN.search(item.path.stem)
        if m and (item.path.resolve().parent, item.path.stem[: m.start()].lower()) in sources:
            continue
        kept.append(item)
    return kept


def _colliding_sources(items: list[InputItem], out_dir: Path | None) -> frozenset[Path]:
    groups: dict[tuple[Path, str], set[Path]] = {}
    for i in items:
        folder = out_dir / i.rel_parent if out_dir is not None else i.path.parent
        groups.setdefault((folder.resolve(), i.path.stem.lower()), set()).add(i.path.resolve())
    return frozenset(p for group in groups.values() if len(group) > 1 for p in group)


def _worker(item: InputItem, spec: TargetSpec, options: Options) -> FileOutcome:
    return process(item.path, spec, options, item.rel_parent)


def process_batch(
    items: list[InputItem],
    spec: TargetSpec,
    options: Options = DEFAULT_OPTIONS,
    jobs: int = 1,
    on_done: Callable[[FileOutcome], None] | None = None,
) -> BatchSummary:
    options = replace(
        options,
        protected=frozenset(i.path.resolve() for i in items),
        disambiguate=_colliding_sources(items, options.out_dir),
        jobs=max(1, min(jobs, len(items))),
    )
    outcomes: list[FileOutcome | None] = [None] * len(items)
    if jobs <= 1 or len(items) <= 1:
        for i, item in enumerate(items):
            outcomes[i] = _worker(item, spec, options)
            if on_done:
                on_done(outcomes[i])
    else:
        with ProcessPoolExecutor(max_workers=min(jobs, len(items))) as pool:
            futures = {pool.submit(_worker, item, spec, options): i for i, item in enumerate(items)}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    outcomes[i] = future.result()
                except Exception as e:  # noqa: BLE001  (BrokenProcessPool, unpicklable result)
                    outcomes[i] = FileOutcome(items[i].path, error=f"worker crashed: {e}")
                if on_done:
                    on_done(outcomes[i])
    return BatchSummary([o for o in outcomes if o is not None])
