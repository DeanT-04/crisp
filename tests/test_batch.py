from pathlib import Path

import pytest
from PIL import Image

from crisp.batch import discover_inputs, process_batch
from crisp.types import Options, TargetSpec


@pytest.fixture
def tree(tmp_path):
    for rel in ("a.png", "b.JPG", "sub/c.png", "a@2x.png"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        fmt = "PNG" if rel.lower().endswith("png") else "JPEG"
        Image.new("L", (4, 4)).save(tmp_path / rel, format=fmt)
    (tmp_path / "notes.txt").write_text("x")
    return tmp_path


def names(items):
    return sorted(i.path.name for i in items)


def test_folder_scan_one_level_skips_crisp_outputs(tree):  # RF1
    assert names(discover_inputs([tree])) == ["a.png", "b.JPG"]


def test_explicit_crisp_output_still_processed(tree):
    assert names(discover_inputs([tree / "a@2x.png"])) == ["a@2x.png"]


def test_recursive_sets_rel_parent(tree):
    items = discover_inputs([tree], recursive=True)
    assert {i.path.name: i.rel_parent for i in items}["c.png"] == Path("sub")


def test_glob_expanded_by_crisp(tree):  # RF5 (and RF1 for globs)
    assert names(discover_inputs([tree / "*.png"])) == ["a.png"]


def test_missing_and_duplicates(tree):
    with pytest.raises(FileNotFoundError):
        discover_inputs([tree / "nope.png"])
    assert len(discover_inputs([tree / "a.png", tree / "a.png"])) == 1


@pytest.mark.parametrize("jobs", [1, 2])
def test_batch_isolates_failures(tree, jobs):
    (tree / "a@2x.png").unlink()  # the fixture's crisp output would make a.png "skipped"
    (tree / "bad.png").write_bytes(b"junk")
    items = discover_inputs([tree / "a.png", tree / "bad.png", tree / "b.JPG"])
    seen = []
    s = process_batch(items, TargetSpec("scale", scale=2), jobs=jobs, on_done=seen.append)
    assert [o.source.name for o in s.outcomes] == ["a.png", "bad.png", "b.JPG"]
    assert s.counts() == {"done": 2, "skipped": 0, "failed": 1}
    assert s.exit_code == 1 and len(seen) == 3


def test_recursive_out_mirrors_folders(tmp_path):  # RF2
    for d in ("x", "y"):
        (tmp_path / "in" / d).mkdir(parents=True)
        Image.new("L", (4, 4)).save(tmp_path / "in" / d / "a.png")
    items = discover_inputs([tmp_path / "in"], recursive=True)
    process_batch(items, TargetSpec("scale", scale=2), Options(out_dir=tmp_path / "o"))
    assert (tmp_path / "o/x/a@2x.png").exists() and (tmp_path / "o/y/a@2x.png").exists()
