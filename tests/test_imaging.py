import math

import cv2
import numpy as np
import pytest

from govkadr import fill
from govkadr import photometric as ph
from govkadr import virtual_studio as vs
from govkadr.framing import find_crown


def test_srgb_linear_roundtrip():
    x = np.linspace(0, 1, 101, dtype=np.float32)
    assert ph.linear_to_srgb(ph.srgb_to_linear(x)) == pytest.approx(x, abs=1e-5)


def test_composite_on_white_gives_pure_white_background():
    fg = np.full((10, 10, 3), 0.2, np.float32)
    alpha = np.zeros((10, 10), np.float32)
    alpha[3:7, 3:7] = 1
    out = ph.composite_on_white(fg, alpha)
    assert (out[0, 0] == 1.0).all()
    assert out[5, 5] == pytest.approx([0.2, 0.2, 0.2])


def test_find_crown_is_subpixel_and_ignores_thin_strands():
    h, w = 200, 200
    alpha = np.zeros((h, w), np.float32)
    cv2.circle(alpha, (100, 120), 60, 1.0, -1)
    alpha[20:60, 100] = 1.0  # a single 1-px flyaway strand above the head
    crown = find_crown(alpha, np.ones((h, w), bool), 40, 160, opening_px=5)
    assert crown is not None
    assert crown.y_roi == pytest.approx(60, abs=1.5)
    assert not crown.touches_source_top


def test_lr_asymmetry_fit_detects_side_light():
    h, w = 120, 120
    xs = np.arange(w, dtype=np.float32)[None, :].repeat(h, 0)
    ramp = np.exp(0.4 * (xs - 60) / 30)  # brighter on the right
    lin = np.dstack([0.3 * ramp] * 3).astype(np.float32)
    skin = np.zeros((h, w), bool)
    skin[20:100, 30:90] = True
    coeffs, info = ph.fit_lr_asymmetry(lin, skin, axis_x=60, eye_y=60, face_w=60, face_h=40)
    assert info["applied"]
    corr = ph.Correction(lr_coeffs=coeffs, lr_frame=(60, 60, 30, 40))
    fixed = lin * corr.lr_field((h, w))
    assert fixed[60, 40, 0] / fixed[60, 80, 0] == pytest.approx(1.0, abs=0.08)


def test_virtual_studio_clamshell_is_left_right_symmetric():
    gx, gy = np.meshgrid(np.linspace(-0.8, 0.8, 41), np.linspace(-0.8, 0.8, 41))
    n = vs._normalize(np.dstack([gx, gy, np.ones_like(gx)]))
    e = vs.irradiance(n.astype(np.float32), vs.PRESETS["clamshell"])
    assert e == pytest.approx(e[:, ::-1], rel=1e-4)
    loop = vs.irradiance(n.astype(np.float32), vs.PRESETS["loop"])
    assert loop[20, 5] > loop[20, 35]  # key light from the image's left


def test_sh_basis_constant_term():
    n = np.array([[0.0, 0.0, 1.0]])
    assert vs.sh_basis(n)[0, 0] == pytest.approx(0.282095)
    assert vs.sh_basis(n).shape == (1, 9)


def test_fill_alpha_follows_generated_subject():
    fill_srgb = np.ones((20, 20, 3), np.float32)
    fill_srgb[10:, :] = 0.1  # generated dark jacket in the lower half
    mask = np.zeros((20, 20), bool)
    mask[:, 15:] = True
    alpha = np.full((20, 20), 0.5, np.float32)
    a = fill.alpha_from_fill(fill_srgb, alpha, mask, feather_px=1)
    assert a[2, 18] == pytest.approx(0.0, abs=0.05)
    assert a[17, 18] == pytest.approx(1.0, abs=0.05)
    assert a[5, 2] == pytest.approx(0.5, abs=1e-3)  # untouched far from the mask


def test_session_parsing_handles_foreign_names():
    assert fill._session_and_time("PXL_20261027_115316389") == ("20261027", 11 * 3600 + 53 * 60 + 16)
    assert fill._session_and_time("my holiday photo") == ("", 0)


def test_degrees_helper_consistency():
    assert math.degrees(math.radians(2.5)) == pytest.approx(2.5)
