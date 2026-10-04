import numpy as np
import pytest
from PIL import Image

from crisp.intake import open_file
from crisp.types import UnsupportedInput


def test_rgb_png(save_image):
    p = save_image("a.png", Image.new("RGB", (4, 3), (255, 0, 0)), dpi=(300, 300))
    [page] = open_file(p)
    assert page.raster.pixels.shape == (3, 4, 3) and page.raster.pixels.dtype == np.float32
    assert page.raster.alpha is None and page.dpi == pytest.approx(300, abs=0.01)
    assert (page.index, page.page_count, page.has_vectors, page.source_format) == (
        0,
        1,
        False,
        "png",
    )
    assert page.size_px == (4, 3)


def test_grayscale_single_channel(save_image):
    [page] = open_file(save_image("g.png", Image.new("L", (5, 5), 128)))
    assert page.raster.pixels.shape == (5, 5, 1)


def test_16bit_scaled(save_image):
    img = Image.fromarray(np.full((2, 2), 65535, np.uint16))
    [page] = open_file(save_image("h.png", img))
    assert page.raster.pixels.max() == pytest.approx(1.0)


def test_alpha_preserved(save_image):
    [page] = open_file(save_image("t.png", Image.new("RGBA", (3, 3), (0, 0, 0, 128))))
    assert page.raster.pixels.shape == (3, 3, 3)
    assert page.raster.alpha == pytest.approx(np.full((3, 3), 128 / 255), abs=1e-6)


def test_exif_rotation_applied(save_image):  # Review Focus 4
    exif = Image.Exif()
    exif[0x0112] = 6
    [page] = open_file(save_image("r.jpg", Image.new("RGB", (4, 2)), exif=exif))
    assert page.size_px == (2, 4)


def test_dpi_missing_or_tiny_is_unknown(save_image):
    assert open_file(save_image("n.png", Image.new("L", (2, 2))))[0].dpi is None
    assert open_file(save_image("s.png", Image.new("L", (2, 2)), dpi=(1, 1)))[0].dpi is None


def test_multipage_tiff(tmp_path):
    frames = [Image.new("L", (4, 4), v) for v in (0, 100, 200)]
    p = tmp_path / "m.tif"
    frames[0].save(p, save_all=True, append_images=frames[1:])
    pages = open_file(p)
    assert [pg.index for pg in pages] == [0, 1, 2] and all(pg.page_count == 3 for pg in pages)


def test_corrupt(tmp_path):
    (p := tmp_path / "x.png").write_bytes(b"not an image")
    with pytest.raises(UnsupportedInput, match="could not read"):
        open_file(p)


def test_unsupported_extension(tmp_path):
    (p := tmp_path / "x.gif").write_bytes(b"GIF89a")
    with pytest.raises(UnsupportedInput, match="unsupported file type"):
        open_file(p)
