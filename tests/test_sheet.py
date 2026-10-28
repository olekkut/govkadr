import numpy as np
import pytest

from govkadr import sheet
from govkadr.spec import PHOTO, SHEET_4x6, SHEET_10x15


@pytest.mark.parametrize("spec", [SHEET_10x15, SHEET_4x6])
def test_layout_fits_with_symmetric_margins(spec):
    tiles = sheet.layout(spec)
    assert len(tiles) == spec.cols * spec.rows
    xs = sorted({x for x, _ in tiles})
    ys = sorted({y for _, y in tiles})
    assert abs(xs[0] - (spec.width_px - xs[-1] - PHOTO.width_px)) <= 1
    assert abs(ys[0] - (spec.height_px - ys[-1] - PHOTO.height_px)) <= 1
    assert all(b - a > PHOTO.width_px for a, b in zip(xs, xs[1:], strict=False))


def _photo(value):
    return np.full((PHOTO.height_px, PHOTO.width_px, 3), value, np.uint8)


@pytest.mark.parametrize("spec", [SHEET_10x15, SHEET_4x6])
def test_tiles_are_pixel_exact_and_marks_stay_outside(spec):
    out = sheet.build(_photo(40), spec)
    assert out.shape == (spec.height_px, spec.width_px, 3)
    for x, y in sheet.layout(spec):
        assert (out[y : y + PHOTO.height_px, x : x + PHOTO.width_px] == 40).all()
    grey = (out == spec.mark_gray).all(axis=2)
    assert grey.any(), "crop marks missing"
    for x, y in sheet.layout(spec):
        assert not grey[y : y + PHOTO.height_px, x : x + PHOTO.width_px].any()


def test_list_of_photos_fills_tiles_in_order_and_leaves_rest_blank():
    out = sheet.build([_photo(10), _photo(20)], SHEET_10x15)
    tiles = sheet.layout(SHEET_10x15)
    (x0, y0), (x1, y1), (x2, y2) = tiles[:3]
    assert out[y0 + 5, x0 + 5, 0] == 10
    assert out[y1 + 5, x1 + 5, 0] == 20
    assert out[y2 + 5, x2 + 5, 0] == 255
