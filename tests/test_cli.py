from PIL import Image
from typer.testing import CliRunner

from crisp.cli import upscale_app


def invoke(args):
    return CliRunner().invoke(upscale_app, args)


def test_upscales(save_image):
    src = save_image("a.png", Image.new("L", (4, 4)))
    r = invoke([str(src), "--scale", "4", "--jobs", "1"])
    assert r.exit_code == 0 and (src.parent / "a@4x.png").exists()


def test_conflicting_size_options(save_image):
    r = invoke([str(save_image("a.png", Image.new("L", (4, 4)))), "--scale", "4", "--dpi", "300"])
    assert r.exit_code == 2 and "only one of" in r.output


def test_bad_size(save_image):
    assert (
        invoke([str(save_image("a.png", Image.new("L", (4, 4)))), "--size", "B9@300"]).exit_code
        == 2
    )


def test_empty_folder(tmp_path):
    r = invoke([str(tmp_path)])
    assert r.exit_code == 2 and "No supported images found" in r.output


def test_missing_path(tmp_path):
    assert invoke([str(tmp_path / "nope.png")]).exit_code == 2


def test_failure_reported_with_exit_1(save_image, tmp_path):
    good = save_image("a.png", Image.new("L", (4, 4)))
    (bad := tmp_path / "bad.png").write_bytes(b"junk")
    r = invoke([str(good), str(bad), "--jobs", "1"])
    assert r.exit_code == 1 and "bad.png" in r.output and "could not read" in r.output
