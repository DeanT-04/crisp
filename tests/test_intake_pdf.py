import pytest

from crisp.intake import open_file
from crisp.types import UnsupportedInput


def test_vector_page(vector_pdf):
    [page] = open_file(vector_pdf)
    assert page.has_vectors and page.raster is None and page.dpi == 72.0
    assert page.size_pt == pytest.approx((595.28, 841.89), abs=0.1)
    assert page.size_px == (595, 842) and page.source_format == "pdf"


def test_image_only_page(image_pdf):
    [page] = open_file(image_pdf)
    assert not page.has_vectors and page.raster.pixels.shape == (600, 800, 3)
    assert page.dpi == pytest.approx(144, abs=0.5)


def test_pages_split(two_page_pdf):
    pages = open_file(two_page_pdf)
    assert [p.index for p in pages] == [0, 1] and pages[1].page_count == 2


def test_password_protected(encrypted_pdf):
    with pytest.raises(UnsupportedInput, match="password-protected"):
        open_file(encrypted_pdf)


def test_invisible_ocr_text_does_not_make_a_scan_vector(ocr_pdf):
    [page] = open_file(ocr_pdf)
    assert not page.has_vectors and page.raster.pixels.shape == (600, 800, 3)


def test_rotated_scan_is_extracted_upright(rotated_scan_pdf):
    [page] = open_file(rotated_scan_pdf)
    assert not page.has_vectors and page.raster.pixels.shape == (800, 600, 3)
    assert page.dpi == pytest.approx(144, abs=0.5)
    # Rotated 90 degrees clockwise: the scan's white top band is now on the right.
    assert page.raster.pixels[:, -50:].mean() > 0.9 and page.raster.pixels[:, :50].mean() < 0.1
