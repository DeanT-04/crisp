from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from crisp.types import Page, Raster


@pytest.fixture
def save_image(tmp_path):
    def fn(name: str, img: Image.Image, **save_kwargs) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path, **save_kwargs)
        return path

    return fn


@pytest.fixture
def make_page():
    def fn(w, h, dpi=None, channels=3, vector=False) -> Page:
        if vector:
            return Page(Path("v.pdf"), 0, 1, None, 72.0, True, (595.28, 841.89), "pdf")
        raster = Raster(np.zeros((h, w, channels), np.float32), None)
        return Page(Path("x.png"), 0, 1, raster, dpi, False, None, "png")

    return fn


def _canvas(path, pagesize=None, **kwargs):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    return canvas.Canvas(str(path), pagesize=pagesize or A4, **kwargs)


def _draw_vector_page(c):
    c.line(50, 50, 500, 700)
    c.rect(100, 100, 300, 200)
    c.drawString(120, 400, "100 mm")
    c.showPage()


@pytest.fixture
def vector_pdf(tmp_path) -> Path:
    c = _canvas(p := tmp_path / "vector.pdf")
    _draw_vector_page(c)
    c.save()
    return p


@pytest.fixture
def two_page_pdf(tmp_path) -> Path:
    c = _canvas(p := tmp_path / "two.pdf")
    _draw_vector_page(c)
    _draw_vector_page(c)
    c.save()
    return p


@pytest.fixture
def image_pdf(tmp_path) -> Path:
    from reportlab.lib.utils import ImageReader

    img = Image.new("RGB", (800, 600), (200, 30, 30))
    c = _canvas(p := tmp_path / "image.pdf", pagesize=(400, 300))
    c.drawImage(ImageReader(img), 0, 0, 400, 300)
    c.showPage()
    c.save()
    return p


@pytest.fixture
def encrypted_pdf(tmp_path) -> Path:
    c = _canvas(p := tmp_path / "locked.pdf", encrypt="secret")
    _draw_vector_page(c)
    c.save()
    return p


@pytest.fixture
def tiny_quick(monkeypatch):
    import crisp.bench.run as bench_run
    from tests.helpers import TINY

    monkeypatch.setattr(bench_run, "QUICK", TINY)


@pytest.fixture
def ocr_pdf(tmp_path) -> Path:
    """A scan (image covering the page) plus an invisible OCR text layer."""
    from reportlab.lib.utils import ImageReader

    img = Image.new("RGB", (800, 600), (200, 30, 30))
    c = _canvas(p := tmp_path / "ocr.pdf", pagesize=(400, 300))
    c.drawImage(ImageReader(img), 0, 0, 400, 300)
    text = c.beginText(50, 50)
    text.setTextRenderMode(3)
    text.textLine("hidden ocr text " * 8)
    c.drawText(text)
    c.showPage()
    c.save()
    return p
