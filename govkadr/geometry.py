"""2-D affine algebra and anti-aliased resampling.

Conventions: 3x3 homogeneous matrices mapping *source* to *destination* coordinates, y pointing
down, and pixel centres at integer coordinates (OpenCV's convention). Composing matrices and
warping once avoids the cumulative blur of rotate-then-scale-then-crop pipelines.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def translate(tx: float, ty: float) -> np.ndarray:
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def scale(sx: float, sy: float | None = None) -> np.ndarray:
    return np.diag([sx, sx if sy is None else sy, 1.0])


def rotate(theta: float) -> np.ndarray:
    """Rotation by ``theta`` radians in y-down coordinates (positive = clockwise on screen)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def resample_scale(sx: float, sy: float | None = None) -> np.ndarray:
    """Map pixel coordinates of an image to those of the same image resized by (sx, sy).

    Pixel edges, not centres, scale: x' = (x + 0.5) * sx - 0.5.
    """
    sy = sx if sy is None else sy
    return translate(-0.5, -0.5) @ scale(sx, sy) @ translate(0.5, 0.5)


def apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    flat = pts.reshape(-1, 2)
    out = flat @ m[:2, :2].T + m[:2, 2]
    return out.reshape(pts.shape)


def linear_scale(m: np.ndarray) -> float:
    return math.sqrt(abs(np.linalg.det(m[:2, :2])))


def warp(
    src: np.ndarray,
    m: np.ndarray,
    size: tuple[int, int],
    interp: int = cv2.INTER_LANCZOS4,
    border: int = cv2.BORDER_REPLICATE,
    border_value: float | tuple = 0,
) -> np.ndarray:
    """Resample ``src`` through ``m`` (src -> dst) into an image of ``size`` = (width, height).

    Only the source window that reaches the output is touched, and when ``m`` minifies, that window
    is first box-filtered (INTER_AREA) to ~1.25x the output density so the interpolation kernel
    never aliases fine texture (hair, skin pores) into moire.
    """
    w_out, h_out = size
    h, w = src.shape[:2]
    inv = np.linalg.inv(m)
    corners = apply(inv, [[-0.5, -0.5], [w_out - 0.5, -0.5], [-0.5, h_out - 0.5], [w_out - 0.5, h_out - 0.5]])
    s = linear_scale(m)
    pad = 8.0 / min(s, 1.0) + 2.0
    x0 = int(np.clip(math.floor(corners[:, 0].min() - pad), 0, w - 1))
    y0 = int(np.clip(math.floor(corners[:, 1].min() - pad), 0, h - 1))
    x1 = int(np.clip(math.ceil(corners[:, 0].max() + pad), x0 + 1, w))
    y1 = int(np.clip(math.ceil(corners[:, 1].max() + pad), y0 + 1, h))
    src = src[y0:y1, x0:x1]
    m = m @ translate(x0, y0)
    if s < 0.8:
        f = s * 1.25
        cw, ch = src.shape[1], src.shape[0]
        nw, nh = max(1, round(cw * f)), max(1, round(ch * f))
        src = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
        m = m @ np.linalg.inv(resample_scale(nw / cw, nh / ch))
    return cv2.warpAffine(src, m[:2], (w_out, h_out), flags=interp, borderMode=border, borderValue=border_value)


def coverage(src_shape: tuple[int, int], m: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Boolean mask of output pixels whose pre-image lies inside the source frame."""
    h, w = src_shape[:2]
    w_out, h_out = size
    xs, ys = np.meshgrid(np.arange(w_out, dtype=np.float64), np.arange(h_out, dtype=np.float64))
    inv = np.linalg.inv(m)
    sx = inv[0, 0] * xs + inv[0, 1] * ys + inv[0, 2]
    sy = inv[1, 0] * xs + inv[1, 1] * ys + inv[1, 2]
    return (sx >= -0.5) & (sx <= w - 0.5) & (sy >= -0.5) & (sy <= h - 0.5)
