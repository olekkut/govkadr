"""Non-destructive photometric normalisation (strict-gov mode).

Every correction is a *smooth multiplicative field* in linear light: a global per-channel white
balance gain, a global exposure gain, and a very low-order left/right shading field. Multiplying
by a field that varies over tens of millimetres cannot create, move or remove pores, moles,
scars or edges - local contrast ratios are preserved exactly - so facial texture and geometry stay
100 % original while casts and uneven lighting are removed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .landmarks import BROW_A, BROW_B, EYE_A, EYE_B, FACE_OVAL, LIPS_OUTER, NOSTRILS, iris_centres, iris_radius

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((np.maximum(x, 0.0) + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, None)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def luminance(lin: np.ndarray) -> np.ndarray:
    return lin @ _LUMA


def _poly(shape, pts, ids) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(pts[list(ids)] * 4).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=2)
    return mask


def _morph(mask: np.ndarray, op: int, radius: float) -> np.ndarray:
    k = max(1, int(round(radius))) * 2 + 1
    return cv2.morphologyEx(mask, op, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


@dataclass
class FaceMasks:
    skin: np.ndarray
    sclera: np.ndarray
    oval: np.ndarray
    face_w: float


def face_masks(pts: np.ndarray, alpha: np.ndarray, lin: np.ndarray) -> FaceMasks:
    """Skin and sclera masks from the mesh, in the coordinates of ``pts``."""
    shape = alpha.shape
    oval_pts = pts[list(FACE_OVAL)]
    fw = float(oval_pts[:, 0].max() - oval_pts[:, 0].min())
    oval = _poly(shape, pts, FACE_OVAL)
    excl = np.zeros(shape, np.uint8)
    for ids in (EYE_A, EYE_B, BROW_A, BROW_B, LIPS_OUTER, NOSTRILS):
        excl |= _poly(shape, pts, ids)
    excl = cv2.dilate(excl, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.05 * fw) | 1,) * 2))
    skin = cv2.erode(oval, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.06 * fw) | 1,) * 2)) & (1 - excl)
    skin = skin.astype(bool) & (alpha > 0.98)
    y = luminance(lin)
    if skin.sum() > 100:
        med = np.median(y[skin])
        # Drop hair fringe, stubble, brows that leak into the oval, and specular highlights.
        skin &= (y > 0.45 * med) & (y < np.percentile(y[skin], 99.0))

    eyes = (_poly(shape, pts, EYE_A) | _poly(shape, pts, EYE_B)).astype(np.uint8)
    eyes = cv2.erode(eyes, np.ones((3, 3), np.uint8))
    iris = np.zeros(shape, np.uint8)
    r = iris_radius(pts) * 1.15
    for c in iris_centres(pts):
        cv2.circle(iris, (int(round(c[0] * 4)), int(round(c[1] * 4))), int(round(r * 4)), 1, -1, lineType=cv2.LINE_8, shift=2)
    sclera = eyes.astype(bool) & ~iris.astype(bool)
    if sclera.sum() > 20:
        ys = y[sclera]
        sclera &= (y > np.percentile(ys, 55)) & (y < np.percentile(ys, 98))
    return FaceMasks(skin=skin, sclera=sclera, oval=oval.astype(bool), face_w=fw)


@dataclass
class Correction:
    wb_gains: np.ndarray = field(default_factory=lambda: np.ones(3, np.float32))
    exposure_gain: float = 1.0
    lr_coeffs: np.ndarray | None = None  # (3, 3) odd-polynomial coefficients per channel
    lr_frame: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)  # axis_x, eye_y, half_width, face_h
    report: dict = field(default_factory=dict)

    def lr_field(self, shape: tuple[int, int]) -> np.ndarray:
        """Per-channel multiplicative left/right shading gain, (H, W, 3)."""
        if self.lr_coeffs is None:
            return np.ones((*shape, 3), np.float32)
        ax, ey, hw, fh = self.lr_frame
        u = np.clip((np.arange(shape[1], dtype=np.float32) - ax) / hw, -1.4, 1.4)[None, :]
        v = np.clip((np.arange(shape[0], dtype=np.float32) - ey) / fh, -1.0, 1.6)[:, None]
        basis = (u + 0 * v, u * v, u**3 + 0 * v)
        out = np.empty((*shape, 3), np.float32)
        for c in range(3):
            a = sum(self.lr_coeffs[c, i] * basis[i] for i in range(3))
            out[..., c] = np.exp(-np.clip(a, -0.35, 0.35))
        return out

    def apply(self, lin: np.ndarray) -> np.ndarray:
        return lin * (self.wb_gains * self.exposure_gain)[None, None, :] * self.lr_field(lin.shape[:2])


def _robust_rgb(lin: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    if mask.sum() < 50:
        return None
    px = lin[mask]
    y = px @ _LUMA
    keep = (y > np.percentile(y, 20)) & (y < np.percentile(y, 99))
    return px[keep].mean(axis=0) if keep.sum() >= 20 else None


def estimate_white_balance(fg_lin, img_lin, alpha, coverage, masks: FaceMasks, chin_y: float, d_px: float, strength: float = 0.8) -> tuple[np.ndarray, dict]:
    """Illuminant from achromatic references: a white shirt (weight 3) and the wall (weight 1).

    Each reference votes for an illuminant chromaticity; votes are averaged in log space with
    weights, then ``strength`` of the cast is removed (the phone's AWB already did most of the
    work). Gains are luminance-normalised. With no usable reference the camera's AWB is kept.

    The sclera is measured but deliberately *not* voted: on the test set it read 0.14 log-units
    redder than a white shirt in the same frame (real sclera, caruncle and lid margin are pinkish),
    and using it greyed the skin of every photo without a white shirt.
    """
    refs: dict[str, tuple[np.ndarray, float]] = {}
    h, w = alpha.shape
    rows = np.arange(h)[:, None]
    sat = (fg_lin.max(axis=2) - fg_lin.min(axis=2)) / np.maximum(fg_lin.max(axis=2), 1e-4)
    y = luminance(fg_lin)
    shirt = (alpha > 0.98) & (rows > chin_y + 0.25 * d_px) & (y > 0.22) & (sat < 0.25)
    shirt = _morph(shirt.astype(np.uint8), cv2.MORPH_OPEN, 3).astype(bool)
    if shirt.mean() > 0.004:
        rgb = _robust_rgb(fg_lin, shirt)
        if rgb is not None:
            refs["shirt"] = (rgb, 3.0)
    sclera = _robust_rgb(fg_lin, masks.sclera)
    wall = (alpha < 0.01) & coverage
    wall = _morph(wall.astype(np.uint8), cv2.MORPH_ERODE, 6).astype(bool)
    rgb = _robust_rgb(img_lin, wall)
    if rgb is not None:
        chroma = np.log(rgb / rgb.mean())
        # A wall is only a usable white reference if it is close to neutral to begin with.
        if np.abs(chroma).max() < 0.25:
            refs["wall"] = (rgb, 1.0)
    report = {k: {"rgb_linear": [round(float(c), 4) for c in v[0]], "weight": v[1]} for k, v in refs.items()}
    if sclera is not None:
        report["sclera"] = {"rgb_linear": [round(float(c), 4) for c in sclera], "weight": 0.0, "note": "diagnostic only (not achromatic)"}
    if not refs:
        return np.ones(3, np.float32), {"references": report, "gains": [1.0, 1.0, 1.0], "note": "no neutral reference found"}
    logs = np.array([np.log(v[0] / np.exp(np.log(v[0]).mean())) for v in refs.values()])
    wts = np.array([v[1] for v in refs.values()])
    cast = (logs * wts[:, None]).sum(axis=0) / wts.sum()
    gains = np.exp(-strength * cast)
    gains = np.clip(gains / float(_LUMA @ gains), 0.8, 1.25).astype(np.float32)
    return gains, {"references": report, "cast_log_rgb": [round(float(c), 4) for c in cast], "strength": strength, "gains": [round(float(g), 4) for g in gains]}


def fit_lr_asymmetry(lin: np.ndarray, skin: np.ndarray, axis_x: float, eye_y: float, face_w: float, face_h: float) -> tuple[np.ndarray | None, dict]:
    """Odd (left/right antisymmetric) low-frequency log-shading per channel, fitted on skin.

    A face is close to bilaterally symmetric in albedo, so the antisymmetric part of its smoothed
    log-intensity is lighting: a window on one side, a lamp on the other. It is modelled as
    a1*u + a2*u*v + a3*u^3 (u: horizontal offset from the facial axis, v: vertical offset from the
    eye line) and later divided out. Symmetric shading (nose, cheekbones, top light) is untouched.
    """
    h, w = skin.shape
    sigma = 0.06 * face_w
    m = skin.astype(np.float32)
    logi = np.log(np.maximum(lin, 1e-4))
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    xs = np.arange(w, dtype=np.float32)
    mirror_x = (2 * axis_x - xs).astype(np.float32)
    map_x = np.repeat(mirror_x[None, :], h, axis=0)
    map_y = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
    den_m = cv2.remap(den, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
    valid = (den > 0.35) & (den_m > 0.35) & skin & (xs[None, :] < axis_x)
    if valid.sum() < 500:
        return None, {"applied": False, "note": "not enough mirrored skin"}
    hw = 0.5 * face_w
    yy, xx = np.nonzero(valid)
    sel = np.linspace(0, len(yy) - 1, min(len(yy), 20000)).astype(int)
    yy, xx = yy[sel], xx[sel]
    u = (xx - axis_x) / hw
    v = (yy - eye_y) / face_h
    design = np.column_stack([u, u * v, u**3])
    coeffs = np.zeros((3, 3), np.float64)
    asym_db = []
    for c in range(3):
        s = cv2.GaussianBlur(logi[..., c] * m, (0, 0), sigma) / np.maximum(den, 1e-6)
        s_m = cv2.remap(s, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
        a = 0.5 * (s - s_m)  # odd part, evaluated on the image-left half
        coeffs[c] = np.linalg.lstsq(design, a[yy, xx], rcond=None)[0]
        asym_db.append(float(np.median(a[yy, xx]) * 2 * 20 / math.log(10)))
    left_right_ev = [round(float(-2 * (coeffs[c, 0] + coeffs[c, 2]) / math.log(2)), 3) for c in range(3)]
    return coeffs, {
        "applied": True,
        "model": "a1*u + a2*u*v + a3*u^3 (odd in u), per linear RGB channel",
        "coefficients": np.round(coeffs, 5).tolist(),
        "left_minus_right_ev_at_cheeks": left_right_ev,  # >0: image-left brighter
    }


def exposure_gain(lin: np.ndarray, skin: np.ndarray, lo: float = 0.42, hi: float = 0.80) -> tuple[float, dict]:
    """Only correct clearly wrong exposure: median skin value outside [lo, hi] (sRGB-encoded)."""
    if skin.sum() < 100:
        return 1.0, {"applied": False}
    y = luminance(lin)[skin]
    med = float(np.median(y))
    med_srgb = float(linear_to_srgb(np.array(med)))
    gain = 1.0
    if med_srgb < lo:
        gain = float(srgb_to_linear(np.array(lo))) / med
    elif med_srgb > hi:
        gain = float(srgb_to_linear(np.array(hi))) / med
    # Never push the brightest 0.5 % of skin into clipping.
    p995 = float(np.percentile(lin[skin].max(axis=1), 99.5))
    if p995 * gain > 0.97:
        gain = min(gain, 0.97 / max(p995, 1e-4)) if gain > 1 else gain
    gain = float(np.clip(gain, 0.6, 1.8))
    return gain, {"skin_median_srgb_before": round(med_srgb, 4), "gain": round(gain, 4), "ev": round(math.log2(gain), 3)}


def normalise(fg_srgb, img_srgb, alpha, coverage, pts, chin_y, d_px, axis_x, eye_y, face_h, wb_strength=0.8) -> tuple[np.ndarray, Correction, FaceMasks]:
    """Full strict-gov photometric chain on the decontaminated foreground; returns linear RGB."""
    fg_lin = srgb_to_linear(fg_srgb)
    img_lin = srgb_to_linear(img_srgb)
    masks = face_masks(pts, alpha, fg_lin)
    corr = Correction()
    corr.wb_gains, corr.report["white_balance"] = estimate_white_balance(fg_lin, img_lin, alpha, coverage, masks, chin_y, d_px, wb_strength)
    lin = fg_lin * corr.wb_gains
    corr.lr_coeffs, corr.report["left_right_balance"] = fit_lr_asymmetry(lin, masks.skin, axis_x, eye_y, masks.face_w, face_h)
    corr.lr_frame = (axis_x, eye_y, 0.5 * masks.face_w, face_h)
    lin = lin * corr.lr_field(lin.shape[:2])
    corr.exposure_gain, corr.report["exposure"] = exposure_gain(lin, masks.skin)
    lin = lin * corr.exposure_gain
    y = luminance(lin)[masks.skin]
    corr.report["skin_median_srgb_after"] = round(float(linear_to_srgb(np.array(np.median(y)))), 4) if y.size else None
    return lin, corr, masks


def composite_on_white(fg_srgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Blend in sRGB, matching how matting networks are trained (Composition-1k convention)."""
    a = alpha[..., None]
    return np.clip(fg_srgb * a + (1.0 - a), 0.0, 1.0)
