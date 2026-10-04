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
