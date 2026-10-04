import numpy as np

from crisp.bench.scene import TEXT_HEIGHTS, random_scene, render_scene


def test_deterministic():
    a, _ = render_scene(random_scene(7, 256, 192), 2)
    b, _ = render_scene(random_scene(7, 256, 192), 2)
    assert np.array_equal(a, b)


def test_contents():
    s = random_scene(1, 256, 192)
    assert sorted(t.height for t in s.texts) == list(TEXT_HEIGHTS)
    assert all(t.string and set(t.string) <= set("0123456789.xRM") for t in s.texts)
    assert len(s.lines) >= 6 and len(s.arcs) >= 2


def test_render_shape_ink_and_boxes():
    img, labels = render_scene(random_scene(3, 256, 192), 4)
    assert img.shape == (768, 1024, 1) and img.dtype == np.float32
    assert 0.01 < (img < 0.5).mean() < 0.4
    assert len(labels) == len(TEXT_HEIGHTS)
    for i, a in enumerate(labels):
        x0, y0, x1, y1 = a.box
        assert 0 <= x0 < x1 <= 1024 and 0 <= y0 < y1 <= 768
        for b in labels[i + 1 :]:
            assert (
                a.box[2] <= b.box[0]
                or b.box[2] <= a.box[0]
                or a.box[3] <= b.box[1]
                or b.box[3] <= a.box[1]
            )
