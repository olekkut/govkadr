"""Diagnostic sheet in the style of biometric enrolment scanners.

Three panels: the source photo with the detected tilt and the 35x45 crop window; the delivered
photo annotated with the 478-point mesh, iris fits, eye axis, crown-to-chin box, the 31.5-36.0 mm
compliance zone and centring guides; and a PASS/WARN/FAIL table of the re-measured checks.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks.python.vision import FaceLandmarksConnections
from PIL import Image, ImageDraw, ImageFont

from . import geometry as g
from .framing import face_width
from .landmarks import CHIN, FaceObservation, iris_centres, iris_radius
from .spec import PHOTO, mm_to_px

_FONTS = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
_TESS = [(c.start, c.end) for c in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION]

C_MESH = (0, 190, 255)
C_IRIS = (255, 214, 0)
C_EYE = (0, 220, 120)
C_HEAD = (255, 64, 160)
C_ZONE = (0, 200, 90)
C_AXIS = (0, 150, 255)
C_GUIDE = (110, 110, 110)
STATUS_RGB = {"PASS": (34, 150, 70), "WARN": (230, 140, 0), "FAIL": (210, 45, 45), "PASS_WITH_WARNINGS": (230, 140, 0)}


def _font(size: int, mono: bool = False) -> ImageFont.FreeTypeFont:
    names = ("consola.ttf", "cour.ttf") if mono else ("segoeui.ttf", "arial.ttf")
    for n in names:
        try:
            return ImageFont.truetype(str(_FONTS / n), size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _pt(p) -> tuple[int, int]:
    return int(round(p[0] * 16)), int(round(p[1] * 16))


def _line(img, a, b, color, th=1):
    cv2.line(img, _pt(a), _pt(b), color, th, cv2.LINE_AA, shift=4)


def _dashed(img, a, b, color, th=1, dash=10.0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(1, int(np.linalg.norm(b - a) / dash))
    for i in range(0, n, 2):
        _line(img, a + (b - a) * i / n, a + (b - a) * min(i + 1, n) / n, color, th)


def _arrow(img, a, b, color, th=2):
    cv2.arrowedLine(img, _pt(a), _pt(b), color, th, cv2.LINE_AA, shift=4, tipLength=0.04)
    cv2.arrowedLine(img, _pt(b), _pt(a), color, th, cv2.LINE_AA, shift=4, tipLength=0.04)


def _text(draw: ImageDraw.ImageDraw, xy, text, fill, size=22, mono=False, anchor="la", bg=None):
    font = _font(size, mono)
    if bg is not None:
        box = draw.textbbox(xy, text, font=font, anchor=anchor)
        draw.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=bg)
    draw.text(xy, text, fill=fill, font=font, anchor=anchor)


def annotate_output(out_rgb: np.ndarray, alpha: np.ndarray, obs: FaceObservation, meas: dict, zoom: float = 1.5) -> np.ndarray:
    big = cv2.resize(out_rgb, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    h, w = big.shape[:2]
    P = lambda p: (np.asarray(p, float) + 0.5) * zoom - 0.5  # noqa: E731
    pts = P(obs.points[:, :2])
    m = meas["measurements"]
    crown_y, chin_y = P([0, m["crown_y_px"]])[1], P([0, m["chin_y_px"]])[1]
    axis_x = P([m["face_axis_x_px"], 0])[0]
    fw = face_width(pts)

    layer = big.copy()
    # Compliance zone: where the crown may sit given the measured chin (31.5-36.0 mm).
    z_top, z_bot = chin_y - mm_to_px(PHOTO.head_max_mm) * zoom, chin_y - mm_to_px(PHOTO.head_min_mm) * zoom
    cv2.rectangle(layer, (0, int(z_top)), (w - 1, int(z_bot)), C_ZONE, -1)
    # ISO eye-line band (50-70 % of the height from the bottom).
    e_top, e_bot = h * (1 - PHOTO.eye_line_max_frac), h * (1 - PHOTO.eye_line_min_frac)
    cv2.rectangle(layer, (0, int(e_top)), (int(0.06 * w), int(e_bot)), C_AXIS, -1)
    big = cv2.addWeighted(layer, 0.18, big, 0.82, 0)

    mesh = big.copy()
    for a, b in _TESS:
        _line(mesh, pts[a], pts[b], C_MESH, 1)
    big = cv2.addWeighted(mesh, 0.35, big, 0.65, 0)
    for p in pts[:468]:
        cv2.circle(big, _pt(p), 1 * 16, C_MESH, -1, cv2.LINE_AA, shift=4)

    left, right = iris_centres(pts)
    r = iris_radius(pts)
    for c in (left, right):
        cv2.circle(big, _pt(c), int(r * 16), C_IRIS, 2, cv2.LINE_AA, shift=4)
        _line(big, c - [r * 0.5, 0], c + [r * 0.5, 0], C_IRIS, 1)
        _line(big, c - [0, r * 0.5], c + [0, r * 0.5], C_IRIS, 1)
    d = right - left
    ext = d / np.linalg.norm(d) * fw * 0.45
    _line(big, left - ext, right + ext, C_EYE, 2)

    # Crown-to-chin box and lines.
    x0, x1 = axis_x - 0.62 * fw, axis_x + 0.62 * fw
    cv2.rectangle(big, _pt((x0, crown_y)), _pt((x1, chin_y)), C_HEAD, 2, cv2.LINE_AA, shift=4)
    _dashed(big, (0, crown_y), (w, crown_y), C_HEAD, 1)
    _dashed(big, (0, chin_y), (w, chin_y), C_HEAD, 1)
    _dashed(big, (0, z_top), (w, z_top), C_ZONE, 2, 14)
    _dashed(big, (0, z_bot), (w, z_bot), C_ZONE, 2, 14)
    dim_x = min(x1 + 0.08 * fw, w - 150)
    _arrow(big, (dim_x, crown_y), (dim_x, chin_y), C_HEAD, 2)
    _arrow(big, (0.5 * (x0 + axis_x), -0.5), (0.5 * (x0 + axis_x), crown_y), C_GUIDE, 1)
    # Centring: canvas centre (dashed grey) vs facial axis (solid blue).
    _dashed(big, ((w - 1) / 2, 0), ((w - 1) / 2, h), C_GUIDE, 1, 8)
    _line(big, (axis_x, crown_y), (axis_x, h - 1), C_AXIS, 1)
    for p in (pts[CHIN],):
        cv2.drawMarker(big, (int(p[0]), int(p[1])), C_HEAD, cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)

    pil = Image.fromarray(big)
    dr = ImageDraw.Draw(pil)
    lab = (255, 255, 255)
    # Head-height label at ~30 % of the head (forehead level), clear of the eye-line label.
    label_y = crown_y + 0.3 * (chin_y - crown_y)
    _text(dr, (dim_x + 10, label_y - 2), f'{m["head_height_mm"]:.2f} mm', C_HEAD, 24, True, "ld", bg=lab)
    _text(dr, (dim_x + 10, label_y + 6), f'{m["head_height_px"]:.0f} px', C_HEAD, 24, True, "la", bg=lab)
    _text(dr, (0.5 * (x0 + axis_x) + 8, 0.5 * crown_y), f'{m["top_margin_mm"]:.2f} mm', C_GUIDE, 20, True, "lm", bg=lab)
    _text(dr, (w - 8, z_top + 4), "max 36.0 mm", C_ZONE, 18, True, "ra", bg=lab)
    _text(dr, (w - 8, z_bot - 4), "min 31.5 mm", C_ZONE, 18, True, "rd", bg=lab)
    _text(dr, (max(8.0, left[0] - ext[0]), left[1] - 8), f'eye line {m["roll_residual_deg"]:+.2f}°', C_EYE, 20, True, "ld", bg=lab)
    _text(dr, (axis_x + 8, h - 16), f'axis {m["horizontal_offset_mm"]:+.2f} mm', C_AXIS, 20, True, "ld", bg=lab)
    _text(dr, (8, e_top + 4), "ISO eye zone", C_AXIS, 16, False, "la", bg=lab)
    return np.asarray(pil)


def annotate_source(src_small: np.ndarray, scale_small: float, src_obs: FaceObservation, m_out: np.ndarray) -> np.ndarray:
    img = src_small.copy()
    s = g.resample_scale(scale_small)
    pts = g.apply(s, src_obs.points[:, :2])
    for p in pts[::3]:
        cv2.circle(img, _pt(p), 16, C_MESH, -1, cv2.LINE_AA, shift=4)
    left, right = iris_centres(pts)
    d = (right - left) / np.linalg.norm(right - left)
    span = 1.6 * np.linalg.norm(right - left)
    _line(img, left - d * span, right + d * span, C_EYE, 2)
    _dashed(img, left - [span, 0], right + [span, 0], (255, 255, 255), 1)
    corners = [[-0.5, -0.5], [PHOTO.width_px - 0.5, -0.5], [PHOTO.width_px - 0.5, PHOTO.height_px - 0.5], [-0.5, PHOTO.height_px - 0.5]]
    quad = g.apply(s @ np.linalg.inv(m_out), corners)
    cv2.polylines(img, [np.round(quad * 16).astype(np.int32)], True, C_HEAD, 2, cv2.LINE_AA, shift=4)
    pil = Image.fromarray(img)
    dr = ImageDraw.Draw(pil)
    _text(dr, (right[0] + d[0] * span + 6, right[1] + d[1] * span), f"tilt {src_obs.eye_angle_deg:+.2f}°", C_EYE, 20, True, "lm", bg=(255, 255, 255))
    _text(dr, tuple(quad[0] + [6, 6]), "35x45 mm crop", C_HEAD, 20, False, "la", bg=(255, 255, 255))
    return np.asarray(pil)


def report_panel(height: int, width: int, title: str, subtitle: str, meas: dict, extra: list[tuple[str, str]]) -> np.ndarray:
    pil = Image.new("RGB", (width, height), (250, 250, 252))
    dr = ImageDraw.Draw(pil)
    verdict = meas["verdict"]
    _text(dr, (28, 24), title, (20, 20, 30), 30)
    _text(dr, (28, 66), subtitle, (90, 90, 100), 18)
    dr.rounded_rectangle((28, 100, width - 28, 150), 10, fill=STATUS_RGB[verdict])
    _text(dr, (width // 2, 125), verdict.replace("_", " "), (255, 255, 255), 28, False, "mm")
    y = 172
    for name, c in meas["checks"].items():
        dr.rounded_rectangle((28, y, 98, y + 26), 6, fill=STATUS_RGB[c["status"]])
        _text(dr, (63, y + 13), c["status"], (255, 255, 255), 15, True, "mm")
        val = c["value"] if isinstance(c["value"], str) else str(c["value"])
        _text(dr, (110, y + 1), name.replace("_", " "), (30, 30, 40), 18)
        _text(dr, (110, y + 24), f"{val[:46]}   [{c['requirement'][:44]}]", (100, 100, 110), 14, True)
        y += 50
    y += 6
    for k, v in extra:
        _text(dr, (28, y), f"{k}: {v}"[:80], (40, 40, 60), 16, True)
        y += 24
    return np.asarray(pil)


DIAG_ZOOM = 1.5
DIAG_HEIGHT = round(PHOTO.height_px * DIAG_ZOOM)


def source_thumbnail(source_rgb: np.ndarray, height: int = DIAG_HEIGHT) -> tuple[np.ndarray, float]:
    s = height / source_rgb.shape[0]
    small = g.warp(source_rgb, g.resample_scale(s), (round(source_rgb.shape[1] * s), height), interp=cv2.INTER_LINEAR)
    return small, s


def compose(src_small: np.ndarray, src_scale: float, src_obs: FaceObservation, m_out: np.ndarray, out_rgb: np.ndarray, alpha: np.ndarray, out_obs: FaceObservation, meas: dict, title: str, subtitle: str, extra: list[tuple[str, str]]) -> np.ndarray:
    right = annotate_output(out_rgb, alpha, out_obs, meas, DIAG_ZOOM)
    hgt = right.shape[0]
    left = annotate_source(src_small[:hgt], src_scale, src_obs, m_out)
    panel = report_panel(hgt, 900, title, subtitle, meas, extra)
    gap = np.full((hgt, 12, 3), 255, np.uint8)
    return np.hstack([left, gap, right, gap, panel])
