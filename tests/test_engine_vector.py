import numpy as np

from crisp.engines.vector import VectorEngine
from crisp.intake import open_file
from crisp.target import resolve_target
from crisp.types import TargetSpec


def test_renders_vector_page_at_exact_size(vector_pdf):
    [page] = open_file(vector_pdf)
    target = resolve_target(TargetSpec("scale", scale=2), page)
    out = VectorEngine().upscale(page, target)
    assert out.pixels.shape == (1684, 1191, 3) and out.alpha is None
    assert out.pixels.min() < 0.2 and np.median(out.pixels) > 0.9
