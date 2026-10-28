"""Crown/chin measurement and the single source -> 35x45 mm canvas transform.

Frames are related by composed affine matrices:

* *rotated frame*: source rotated by -eye_angle about the inter-pupillary midpoint (unit scale), so
  the eye line is horizontal and lies on y = 0;
* *ROI*: the rotated frame stretched onto the matting network input (axis-aligned, no rotation);
* *output*: the 826x1062 canvas; the crown sits ``top_margin`` below the top edge, the facial
  mid-sagittal axis on the vertical centre line and chin-to-crown spans ``head_target`` exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from . import geometry as g
from .landmarks import CHIN, FACE_OVAL, MIDLINE, FaceObservation, iris_centres
from .spec import PHOTO, PhotoSpec


def rotated_frame(obs: FaceObservation) -> np.ndarray:
    left, right = iris_centres(obs.points)
    mid = 0.5 * (left + right)
    return g.rotate(-obs.eye_angle_rad) @ g.translate(-mid[0], -mid[1])


def eye_to_chin(obs: FaceObservation) -> float:
    return float(g.apply(rotated_frame(obs), obs.points[CHIN, :2])[1])


def roi_transform(obs: FaceObservation, height_px: int, top_extra: float = 0.0) -> tuple[np.ndarray, tuple[int, int], tuple]:
    """Matting ROI in rotated-frame units of D (eye line -> chin distance).

    The canvas spans roughly [-1.35 D, +1.5 D] vertically and +-1.15 D horizontally for typical
    heads, so the ROI leaves ~0.6 D of slack on each side for hair volume and framing changes.
    """
    d = eye_to_chin(obs)
    x0, x1 = -1.6 * d, 1.6 * d
    y0, y1 = -(2.0 + top_extra) * d, 2.1 * d
    w_px = int(round(height_px * (x1 - x0) / (y1 - y0) / 32.0)) * 32
    m = g.translate(-0.5, -0.5) @ g.scale(w_px / (x1 - x0), height_px / (y1 - y0)) @ g.translate(-x0, -y0) @ rotated_frame(obs)
    return m, (w_px, height_px), (x0, x1, y0, y1)


@dataclass
class CrownResult:
    y_roi: float  # sub-pixel row of the alpha = 0.5 crossing at the top of the head
    x_roi: float
    touches_roi_top: bool
    touches_source_top: bool


def find_crown(alpha: np.ndarray, coverage: np.ndarray, x_lo: float, x_hi: float, opening_px: int) -> CrownResult | None:
    """Highest point of the head silhouette inside the column band [x_lo, x_hi].

    A morphological opening first removes isolated flyaway strands so the crown reflects the hair
    *volume*, not a single backlit hair. The 0.5-crossing is interpolated to sub-pixel precision.
    """
    h, w = alpha.shape
    mask = (alpha > 0.5).astype(np.uint8)
    k = max(3, opening_px | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    c0, c1 = max(0, int(math.floor(x_lo))), min(w, int(math.ceil(x_hi)) + 1)
    band = mask[:, c0:c1]
    has = band.any(axis=0)
    if not has.any():
        return None
    first = np.where(has, band.argmax(axis=0), h)
    col = int(np.argmin(first))
    row = int(first[col])
    x = c0 + col
    y = float(row)
    if row > 0:
        a_prev, a_cur = float(alpha[row - 1, x]), float(alpha[row, x])
        if a_cur > a_prev:
            y = row - 1 + (0.5 - a_prev) / (a_cur - a_prev)
    # The head is cut by the photo's top edge if, in some band column, the silhouette starts
    # right at the first row that still maps inside the source image.
    cov = coverage[:, c0:c1]
    first_cov = np.where(cov.any(axis=0), cov.argmax(axis=0), h)
    touches_source = bool(np.any(has & (first_cov > 0) & (first <= first_cov + 1)))
    return CrownResult(y_roi=y, x_roi=float(x), touches_roi_top=row <= 1, touches_source_top=touches_source)


@dataclass
class Framing:
    m_out: np.ndarray  # source -> output canvas
    scale: float  # output px per source px
    crown_rot: tuple[float, float]  # crown (x, y) in the rotated frame
    chin_rot: tuple[float, float]
    axis_x_rot: float
    head_src_px: float

    def to_dict(self) -> dict:
        return {
            "m_out": self.m_out.tolist(),
            "scale": self.scale,
            "crown_rot": list(self.crown_rot),
            "chin_rot": list(self.chin_rot),
            "axis_x_rot": self.axis_x_rot,
            "head_src_px": self.head_src_px,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Framing:
        return cls(np.array(d["m_out"]), d["scale"], tuple(d["crown_rot"]), tuple(d["chin_rot"]), d["axis_x_rot"], d["head_src_px"])


def face_axis_x(points_2d: np.ndarray) -> float:
    """x of the facial mid-sagittal axis: median over nasion, nose bridge, philtrum, lips and chin."""
    return float(np.median(points_2d[list(MIDLINE), 0]))


def face_width(points_2d: np.ndarray) -> float:
    oval = points_2d[list(FACE_OVAL)]
    return float(oval[:, 0].max() - oval[:, 0].min())


def solve(obs: FaceObservation, crown_rot: tuple[float, float], spec: PhotoSpec = PHOTO) -> Framing:
    r = rotated_frame(obs)
    pts_rot = g.apply(r, obs.points[:, :2])
    chin = pts_rot[CHIN]
    axis = face_axis_x(pts_rot)
    head = float(chin[1] - crown_rot[1])
    s = spec.head_target_px / head
    m_out = g.translate((spec.width_px - 1) / 2.0, spec.top_margin_target_px - 0.5) @ g.scale(s) @ g.translate(-axis, -crown_rot[1]) @ r
    return Framing(m_out, s, (float(crown_rot[0]), float(crown_rot[1])), (float(chin[0]), float(chin[1])), axis, head)


def canvas_inside(m_out: np.ndarray, m_roi: np.ndarray, roi_size: tuple[int, int], spec: PhotoSpec = PHOTO) -> bool:
    """Whether every canvas pixel has matte coverage."""
    corners = [[-0.5, -0.5], [spec.width_px - 0.5, -0.5], [-0.5, spec.height_px - 0.5], [spec.width_px - 0.5, spec.height_px - 0.5]]
    pts = g.apply(m_roi @ np.linalg.inv(m_out), corners)
    w, h = roi_size
    return bool((pts[:, 0] >= -0.5).all() and (pts[:, 0] <= w - 0.5).all() and (pts[:, 1] >= -0.5).all() and (pts[:, 1] <= h - 0.5).all())
