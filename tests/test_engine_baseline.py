from pathlib import Path

import numpy as np
import pytest

from crisp.color import srgb_to_linear
from crisp.engines.baseline import BaselineEngine
from crisp.target import resolve_target
from crisp.types import Page, Raster, TargetSpec


def up(arr, scale, alpha=None):
    page = Page(Path("x.png"), 0, 1, Raster(arr, alpha), None, False, None, "png")
    return BaselineEngine().upscale(page, resolve_target(TargetSpec("scale", scale=scale), page))


def test_shape_and_range():
    out = up(np.random.default_rng(0).random((8, 10, 3), dtype=np.float32), 4)
    assert out.pixels.shape == (32, 40, 3) and out.pixels.dtype == np.float32
    assert out.pixels.min() >= 0 and out.pixels.max() <= 1


def test_constant_stays_constant():
    assert np.allclose(up(np.full((6, 6, 1), 0.5, np.float32), 3).pixels, 0.5, atol=1e-3)


def test_resamples_in_linear_light():
    checker = (np.indices((16, 16)).sum(0) % 2).astype(np.float32)[..., None]
    out = up(checker, 2).pixels
    assert srgb_to_linear(out).mean() == pytest.approx(0.5, abs=0.02)


def test_alpha_has_no_dark_fringe():
    rgb = np.zeros((8, 8, 3), np.float32)
    rgb[:, :4] = 1.0
    alpha = np.zeros((8, 8), np.float32)
    alpha[:, :4] = 1.0
    out = up(rgb, 4, alpha)
    assert out.alpha is not None and out.pixels[out.alpha > 0.01].min() >= 0.98
