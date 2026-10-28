import math

import numpy as np
import pytest

from govkadr import geometry as g
from govkadr.identity import umeyama
from govkadr.spec import PHOTO, mm_to_px, px_to_mm


def test_photo_spec_matches_600_dpi():
    assert (PHOTO.width_px, PHOTO.height_px) == (826, 1062)
    assert PHOTO.head_target_px == pytest.approx(805.5, abs=0.1)
    assert px_to_mm(mm_to_px(34.1)) == pytest.approx(34.1)
    # The 70-80 % rule of gov.pl is the 31.5-36.0 mm head-height window.
    assert PHOTO.head_min_mm == pytest.approx(0.7 * PHOTO.height_mm)
    assert PHOTO.head_max_mm == pytest.approx(0.8 * PHOTO.height_mm)


def test_resample_scale_maps_pixel_edges():
    m = g.resample_scale(0.5)
    # Edges scale, centres shift: the left edge -0.5 stays at -0.5, centre 0 of a 2x2 block -> 0.
    assert g.apply(m, [[-0.5, -0.5]])[0] == pytest.approx([-0.5, -0.5])
    assert g.apply(m, [[0.5, 0.5]])[0] == pytest.approx([0.0, 0.0])


def test_compose_rotation_inverse_roundtrip():
    m = g.translate(10, -4) @ g.rotate(math.radians(3.3)) @ g.scale(0.27)
    pts = np.array([[1.0, 2.0], [400.0, -30.0]])
    back = g.apply(np.linalg.inv(m), g.apply(m, pts))
    assert back == pytest.approx(pts)


def test_warp_identity_and_minification_preserve_mean():
    rng = np.random.default_rng(0)
    img = (rng.random((200, 300, 3)) * 255).astype(np.uint8)
    same = g.warp(img, np.eye(3), (300, 200))
    assert np.abs(same.astype(int) - img).max() <= 1
    small = g.warp(img, g.resample_scale(0.25), (75, 50))
    assert small.mean() == pytest.approx(img.mean(), abs=1.5)


def test_coverage_marks_pixels_outside_source():
    cov = g.coverage((100, 100), g.translate(50, 0), (200, 100))
    assert cov[:, :50].sum() == 0
    assert cov[:, 50:150].all()
    assert not cov[:, 150:].any()


def test_umeyama_recovers_similarity():
    src = np.array([[0, 0], [10, 0], [0, 10], [7, 3], [2, 9]], float)
    m = g.translate(5, 7) @ g.rotate(0.4) @ g.scale(1.7)
    est = umeyama(src, g.apply(m, src))
    assert est == pytest.approx(m, abs=1e-9)
