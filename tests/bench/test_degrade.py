import numpy as np

from crisp.bench.degrade import PRESETS, degrade
from crisp.bench.scene import random_scene, render_scene

GT = render_scene(random_scene(0, 128, 96), 4)[0]


def test_shape_range_dtype():
    for d in PRESETS.values():
        low = degrade(GT, 4, d, seed=1)
        assert low.shape == (96, 128, 1) and low.dtype == np.float32
        assert low.min() >= 0 and low.max() <= 1


def test_clean_is_area_downscale_of_constant():
    flat = np.full((40, 40, 1), 0.3, np.float32)
    assert np.allclose(degrade(flat, 4, PRESETS["clean"], 0), 0.3, atol=1e-6)


def test_seeded():
    s = PRESETS["scan"]
    assert np.array_equal(degrade(GT, 4, s, 5), degrade(GT, 4, s, 5))
    assert not np.array_equal(degrade(GT, 4, s, 5), degrade(GT, 4, s, 6))


def test_blur_removes_high_frequencies():
    def hf(x):
        return np.abs(np.diff(x, axis=1)).mean()

    assert hf(degrade(GT, 4, PRESETS["blur"], 0)) < hf(degrade(GT, 4, PRESETS["clean"], 0))
