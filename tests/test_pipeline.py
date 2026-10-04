import pytest
from PIL import Image

from crisp.engines.baseline import BaselineEngine
from crisp.pipeline import process
from crisp.types import Options, TargetSpec


def test_png_300dpi_scale4_writes_1200dpi(save_image):
    src = save_image("a.png", Image.new("RGB", (10, 8), "white"), dpi=(300, 300))
    out = process(src, TargetSpec("scale", scale=4))
    [pr] = out.pages
    assert (out.status, pr.engine) == ("done", "baseline")
    img = Image.open(pr.output)
    assert img.size == (40, 32) and img.info["dpi"][0] == pytest.approx(1200, abs=0.01)


def test_vector_pdf_uses_vector_engine(vector_pdf):
    assert process(vector_pdf, TargetSpec("scale", scale=2)).pages[0].engine == "vector"


def test_existing_output_skipped_unless_overwrite(save_image):
    src = save_image("a.png", Image.new("L", (4, 4)))
    process(src, TargetSpec("scale", scale=2))
    assert process(src, TargetSpec("scale", scale=2)).pages[0].reason == "output exists"
    assert process(src, TargetSpec("scale", scale=2), Options(overwrite=True)).status == "done"


def test_corrupt_file(tmp_path):
    (p := tmp_path / "x.png").write_bytes(b"junk")
    out = process(p, TargetSpec("scale", scale=2))
    assert out.status == "failed" and "could not read" in out.error


def test_never_overwrites_an_input(save_image):
    src = save_image("a.png", Image.new("L", (4, 4)))
    other = save_image("a@2x.png", Image.new("L", (9, 9)))
    before = other.read_bytes()
    out = process(
        src,
        TargetSpec("scale", scale=2),
        Options(overwrite=True, protected=frozenset({other.resolve()})),
    )
    assert out.pages[0].reason == "would overwrite an input file"
    assert other.read_bytes() == before


def test_dpi_unknown_fails_file(save_image):
    out = process(save_image("a.png", Image.new("L", (4, 4))), TargetSpec("dpi", dpi=600))
    assert out.status == "failed" and "--scale" in out.pages[0].reason


def test_unexpected_error_is_contained(save_image, monkeypatch):
    def boom(*a):
        raise RuntimeError("boom")

    monkeypatch.setattr(BaselineEngine, "upscale", boom)
    out = process(save_image("a.png", Image.new("L", (4, 4))), TargetSpec("scale", scale=2))
    assert out.pages[0].reason == "unexpected error: RuntimeError: boom"
