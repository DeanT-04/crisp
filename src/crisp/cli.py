"""Command line interface."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from crisp.batch import discover_inputs, process_batch
from crisp.target import parse_target
from crisp.types import FileOutcome, Options

upscale_app = typer.Typer(add_completion=False, help="Upscale images and PDFs without AI.")
console = Console()
err_console = Console(stderr=True)


def _fail(message: str) -> typer.Exit:
    err_console.print(f"error: {message}", markup=False, highlight=False, soft_wrap=True)
    return typer.Exit(2)


def _reasons(outcome: FileOutcome) -> list[str]:
    if outcome.error:
        return [outcome.error]
    return [p.reason for p in outcome.pages if p.status == "failed" and p.reason]


@upscale_app.command()
def upscale(
    inputs: Annotated[list[Path], typer.Argument(help="Files, folders or glob patterns.")],
    scale: Annotated[float | None, typer.Option(help="Scale factor, e.g. 4 (default 2).")] = None,
    dpi: Annotated[int | None, typer.Option(help="Target DPI (needs known input DPI).")] = None,
    size: Annotated[str | None, typer.Option(help="Paper size, e.g. A3@300.")] = None,
    out: Annotated[Path | None, typer.Option(help="Output folder.")] = None,
    jobs: Annotated[int, typer.Option(help="Parallel workers.")] = os.cpu_count() or 1,
    recursive: Annotated[bool, typer.Option(help="Scan folders recursively.")] = False,
    overwrite: Annotated[bool, typer.Option(help="Replace existing crisp outputs.")] = False,
    verbose: Annotated[bool, typer.Option(help="Print one line per page.")] = False,
) -> None:
    try:
        spec = parse_target(scale, dpi, size)
    except ValueError as e:
        raise _fail(str(e)) from e
    try:
        items = discover_inputs(inputs, recursive)
    except FileNotFoundError as e:
        raise _fail(str(e)) from e
    if not items:
        raise _fail("No supported images found.")

    options = Options(out_dir=out, overwrite=overwrite)
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Upscaling", total=len(items))

        def on_done(outcome: FileOutcome) -> None:
            progress.advance(task)
            if not verbose:
                return
            for p in outcome.pages:
                label = p.target.label if p.target else "-"
                progress.console.print(
                    f"{outcome.source} p{p.page + 1}: {p.status} engine={p.engine} "
                    f"target={label} -> {p.output} ({p.seconds:.2f}s)",
                    markup=False,
                    highlight=False,
                    soft_wrap=True,
                )

        summary = process_batch(items, spec, options, jobs, on_done)

    counts = summary.counts()
    table = Table(show_header=True, header_style="bold")
    for column in ("Done", "Skipped", "Failed"):
        table.add_column(column, justify="right")
    table.add_row(str(counts["done"]), str(counts["skipped"]), str(counts["failed"]))
    console.print(table)
    for outcome in summary.outcomes:
        for reason in _reasons(outcome):
            console.print(
                f"FAILED {outcome.source}: {reason}", markup=False, highlight=False, soft_wrap=True
            )
    raise typer.Exit(summary.exit_code)


def main(argv: list[str] | None = None) -> None:
    upscale_app(args=argv, prog_name="crisp")
