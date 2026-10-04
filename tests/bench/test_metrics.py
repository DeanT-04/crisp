import numpy as np
import pytest

from crisp.bench import metrics
from crisp.bench.metrics import (
    edge_error,
    legibility,
    psnr,
    ssim,
    stroke_width_error,
    tesseract_available,
)
from crisp.bench.scene import Scene, Text, render_scene


def bar(x, width, size=(64, 64)):
    img = np.ones((*size, 1), np.float32)
    img[:, x : x + width] = 0
    return img


def test_psnr_ssim():
    a = np.full((16, 16, 1), 0.5, np.float32)
    assert psnr(a, a + 0.1) == pytest.approx(20.0, abs=1e-3)
    assert ssim(bar(20, 4), bar(20, 4)) == pytest.approx(1.0)


def test_edge_error():
    assert edge_error(bar(20, 4), bar(20, 4)) == 0
    # A single step edge shifted by 3 px: every edge pixel is 3 px from the other image's edge.
    assert edge_error(bar(23, 41), bar(20, 44)) == pytest.approx(3, abs=0.5)


def test_edge_error_nan_without_edges():
    assert np.isnan(edge_error(np.ones((16, 16, 1), np.float32), bar(5, 4, (16, 16))))


def test_stroke_width_error():
    assert stroke_width_error(bar(20, 6), bar(20, 6)) == pytest.approx(0, abs=1e-6)
    assert stroke_width_error(bar(20, 30), bar(20, 20)) == pytest.approx(0.5, abs=0.1)


def test_legibility_none_without_tesseract(monkeypatch):
    monkeypatch.setattr(metrics, "tesseract_available", lambda: False)
    assert metrics.legibility(bar(0, 1), []) is None


@pytest.mark.skipif(not tesseract_available(), reason="Tesseract not installed")
def test_legibility_reads_large_text():
    scene = Scene(200, 80, (), (), (Text(20, 20, 40, "1234"),))
    img, labels = render_scene(scene, 2)
    assert legibility(img, labels) == {40.0: 1.0}
    assert legibility(np.ones_like(img), labels) == {40.0: 0.0}
