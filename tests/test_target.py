import pytest

from crisp.target import parse_target, resolve_target
from crisp.types import TargetError, TargetSpec


def test_parse():
    assert parse_target(None, None, None) == TargetSpec("scale", scale=2.0)
    assert parse_target(None, None, "a3@300") == TargetSpec("size", dpi=300, paper="A3")
    with pytest.raises(ValueError, match="only one of"):
        parse_target(4, 300, None)
    with pytest.raises(ValueError, match="greater than 1"):
        parse_target(1.0, None, None)
    with pytest.raises(ValueError, match="greater than 1"):
        parse_target(17, None, None)
    with pytest.raises(ValueError, match="A3@300"):
        parse_target(None, None, "B5@300")


def test_scale(make_page):
    t = resolve_target(TargetSpec("scale", scale=4), make_page(100, 50, dpi=300))
    assert (t.out_size, t.out_dpi, t.label) == ((400, 200), 1200, "4x")
    assert resolve_target(TargetSpec("scale", scale=1.5), make_page(10, 10)).label == "1.5x"
    assert resolve_target(TargetSpec("scale", scale=2), make_page(10, 10)).out_dpi is None


def test_dpi(make_page):
    t = resolve_target(TargetSpec("dpi", dpi=600), make_page(100, 50, dpi=150))
    assert (t.scale, t.out_size, t.out_dpi, t.label) == (4.0, (400, 200), 600, "600dpi")
    with pytest.raises(TargetError, match="--scale"):
        resolve_target(TargetSpec("dpi", dpi=600), make_page(10, 10))
    with pytest.raises(TargetError, match="already at or above"):
        resolve_target(TargetSpec("dpi", dpi=300), make_page(10, 10, dpi=600))
    with pytest.raises(TargetError, match="16×"):
        resolve_target(TargetSpec("dpi", dpi=600), make_page(10, 10, dpi=20))


def test_size_matches_orientation(make_page):
    t = resolve_target(TargetSpec("size", dpi=300, paper="A4"), make_page(1000, 600))
    assert (t.out_size, t.out_dpi, t.label) == ((3508, 2105), 300, "A4-300dpi")


def test_vector_page(make_page):
    t = resolve_target(TargetSpec("scale", scale=2), make_page(0, 0, vector=True))
    assert (t.out_size, t.out_dpi) == ((1191, 1684), 144)
