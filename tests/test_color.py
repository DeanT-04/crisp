import numpy as np
import pytest

from crisp.color import linear_to_srgb, srgb_to_linear


def test_roundtrip():
    x = np.linspace(0, 1, 1001, dtype=np.float32)
    assert np.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-5)


def test_known_values():
    assert float(srgb_to_linear(np.float32(0.5))) == pytest.approx(0.21404, abs=1e-4)
    assert float(srgb_to_linear(np.float32(0.04045))) == pytest.approx(0.0031308, abs=1e-6)
