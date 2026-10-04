from pathlib import Path

import numpy as np
import pytest

from crisp.bench.methods import (
    available_methods,
    bicubic,
    crisp_method,
    lanczos,
    nearest,
    passes_needed,
    realesrgan,
)
from crisp.pipeline import upscale_page
from crisp.target import resolve_target
from crisp.types import Page, Raster, TargetSpec

IMGS = [np.random.default_rng(0).random((12, 10, c), dtype=np.float32) for c in (1, 3)]


@pytest.mark.parametrize("method", [nearest, bicubic, lanczos, crisp_method])
@pytest.mark.parametrize("scale", [2, 4, 8])
def test_shapes(method, scale):
    for img in IMGS:
        out = method(img, scale)
        assert out.shape == (12 * scale, 10 * scale, img.shape[2])
        assert out.min() >= 0 and out.max() <= 1


def test_crisp_method_uses_pipeline():
    page = Page(Path("<bench>"), 0, 1, Raster(IMGS[1], None), None, False, None, "png")
    expected, engine = upscale_page(page, resolve_target(TargetSpec("scale", scale=2), page))
    assert engine == "baseline" and np.allclose(crisp_method(IMGS[1], 2), expected.pixels)


def test_passes_needed():
    assert [passes_needed(s) for s in (2, 4, 8, 16)] == [1, 1, 2, 2]


def test_realesrgan_absent(tmp_path):
    assert realesrgan("realesrgan-x4plus", tmp_path) is None
    assert list(available_methods(tmp_path, ai=True)) == ["nearest", "bicubic", "lanczos", "crisp"]
