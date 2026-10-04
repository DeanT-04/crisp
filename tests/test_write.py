from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from crisp.types import Raster
from crisp.write import OUTPUT_PATTERN, output_path, write_png


def test_output_path():
    assert output_path(Path("d/a.png"), 0, 1, "4x", None) == Path("d/a@4x.png")
    assert output_path(Path("d/plan.pdf"), 2, 5, "600dpi", Path("out")) == Path(
        "out/plan-p3@600dpi.png"
    )
    assert output_path(Path("s/x/a.png"), 0, 1, "2x", Path("o"), Path("x")) == Path(
        "o/x/a@2x.png"
    )  # RF2


def test_output_pattern():
    for stem in ("a@4x", "a@1.5x", "plan-p3@600dpi", "a@A3-300dpi"):
        assert OUTPUT_PATTERN.search(stem)
    for stem in ("a", "me@home", "a@4"):
        assert not OUTPUT_PATTERN.search(stem)


@pytest.mark.parametrize(
    "channels,alpha,mode", [(3, False, "RGB"), (1, False, "L"), (3, True, "RGBA"), (1, True, "LA")]
)
def test_write_modes_and_dpi(tmp_path, channels, alpha, mode):
    r = Raster(
        np.full((4, 5, channels), 0.5, np.float32), np.ones((4, 5), np.float32) if alpha else None
    )
    write_png(r, p := tmp_path / "sub" / "o.png", 1200)
    img = Image.open(p)
    assert (img.mode, img.size) == (mode, (5, 4))
    assert img.info["dpi"][0] == pytest.approx(1200, abs=0.01)


def test_no_dpi(tmp_path):
    write_png(Raster(np.zeros((2, 2, 1), np.float32), None), p := tmp_path / "o.png", None)
    assert "dpi" not in Image.open(p).info


def test_failed_write_leaves_nothing(tmp_path, monkeypatch):  # RF3
    def broken_save(self, fp, *a, **k):
        Path(fp).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(Image.Image, "save", broken_save)
    with pytest.raises(OSError):
        write_png(Raster(np.zeros((2, 2, 1), np.float32), None), tmp_path / "o.png", None)
    assert list(tmp_path.iterdir()) == []
