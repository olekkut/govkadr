"""478-point face mesh (MediaPipe Face Landmarker, iris-refined) with iterative roll estimation.

The landmark network sees faces through a 256x256 crop and is most accurate on upright faces, so
after a first pass on the whole frame the face is re-cropped, de-rotated by the current eye-line
estimate and re-measured until the residual eye-line tilt converges (typically 2-3 passes). The
final landmarks therefore come from an upright, tightly framed view of the face.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from . import geometry as g

IRIS_A = (468, 469, 470, 471, 472)  # centre + 4 limbus points (subject's right eye)
IRIS_B = (473, 474, 475, 476, 477)  # subject's left eye
CHIN = 152
# Landmarks on the facial mid-sagittal line: nasion, nose bridge, philtrum, lips, chin.
MIDLINE = (9, 8, 168, 6, 197, 195, 5, 4, 1, 2, 164, 0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 152)
FACE_OVAL = (
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
)
EYE_A = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
EYE_B = (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466)
BROW_A = (70, 63, 105, 66, 107, 55, 65, 52, 53, 46)
BROW_B = (300, 293, 334, 296, 336, 285, 295, 282, 283, 276)
LIPS_OUTER = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146)
NOSTRILS = (98, 97, 2, 326, 327, 294, 278, 344, 440, 275, 4, 45, 220, 115, 48, 64)


class NoFaceError(RuntimeError):
    pass


@dataclass
class FaceObservation:
    points: np.ndarray  # (478, 3): x, y in source pixels, z relative depth in source-pixel units
    eye_angle_rad: float  # tilt of the inter-pupillary line in the source (y-down, + = clockwise)
    blendshapes: dict[str, float]
    pose_deg: dict[str, float]
    faces_detected: int
    passes: int
    angle_history_deg: list[float] = field(default_factory=list)
    angle_stderr_deg: float = float("nan")  # standard error of the eye-line angle over TTA samples
    tta_samples: int = 0

    @property
    def iris_centres(self) -> tuple[np.ndarray, np.ndarray]:
        return iris_centres(self.points)

    @property
    def eye_angle_deg(self) -> float:
        return math.degrees(self.eye_angle_rad)


def iris_centres(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Iris centres ordered image-left, image-right.

    Each centre averages the network's centre point with the centroid of the four limbus points,
    which halves the jitter of either estimate on its own.
    """

    def centre(ids):
        return 0.5 * pts[ids[0], :2] + 0.5 * pts[list(ids[1:]), :2].mean(axis=0)

    a, b = centre(IRIS_A), centre(IRIS_B)
    return (a, b) if a[0] <= b[0] else (b, a)


def iris_radius(pts: np.ndarray) -> float:
    radii = []
    for ids in (IRIS_A, IRIS_B):
        c = pts[ids[0], :2]
        radii.append(np.linalg.norm(pts[list(ids[1:]), :2] - c, axis=1).mean())
    return float(np.mean(radii))


def eye_line_angle(pts: np.ndarray) -> float:
    left, right = iris_centres(pts)
    d = right - left
    return math.atan2(d[1], d[0])


def _pose_from_matrix(mat: np.ndarray | None) -> dict[str, float]:
    if mat is None:
        return {}
    r = np.asarray(mat, dtype=np.float64)[:3, :3]
    r = r / np.linalg.norm(r, axis=0, keepdims=True)
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2])))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return {"yaw": yaw, "pitch": pitch, "roll": roll}


class FaceAnalyzer:
    def __init__(self, model_path: Path):
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=4,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(opts)

    def close(self) -> None:
        self._landmarker.close()

    def detect(self, rgb: np.ndarray) -> list[tuple[np.ndarray, dict[str, float], np.ndarray | None]]:
        """All faces in ``rgb`` as (points in pixel-centre coords, blendshapes, pose matrix)."""
        h, w = rgb.shape[:2]
        res = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        faces = []
        for i, lms in enumerate(res.face_landmarks):
            pts = np.array([[p.x * w - 0.5, p.y * h - 0.5, p.z * w] for p in lms], dtype=np.float64)
            blend = {c.category_name: float(c.score) for c in res.face_blendshapes[i]} if res.face_blendshapes else {}
            mat = np.asarray(res.facial_transformation_matrixes[i]) if res.facial_transformation_matrixes else None
            faces.append((pts, blend, mat))
        return faces

    def _upright_pass(self, img, pts, angle, crop_px, zoom=1.0, extra_rot=0.0):
        """Re-detect on a crop de-rotated by ``angle``; returns source-space landmarks or None."""
        left, right = iris_centres(pts)
        mid = 0.5 * (left + right)
        s = zoom * crop_px / (6.0 * float(np.linalg.norm(right - left)))
        m = g.translate(crop_px / 2 - 0.5, crop_px * 0.45 - 0.5) @ g.scale(s) @ g.rotate(-(angle + extra_rot)) @ g.translate(-mid[0], -mid[1])
        found = self.detect(g.warp(img, m, (crop_px, crop_px), interp=cv2.INTER_CUBIC))
        if not found:
            return None
        centre = np.array([crop_px / 2, crop_px * 0.45])
        pts_c, blend, mat = min(found, key=lambda f: np.linalg.norm(0.5 * np.add(*iris_centres(f[0])) - centre))
        return np.column_stack([g.apply(np.linalg.inv(m), pts_c[:, :2]), pts_c[:, 2] / s]), blend, mat

    def analyze(self, img: np.ndarray, crop_px: int = 1024, passes: int = 2) -> FaceObservation:
        h, w = img.shape[:2]
        s0 = min(1.0, 1600.0 / max(h, w))
        size0 = (max(1, round(w * s0)), max(1, round(h * s0)))
        m0 = g.resample_scale(size0[0] / w, size0[1] / h)
        faces = self.detect(g.warp(img, m0, size0, interp=cv2.INTER_LINEAR))
        if not faces:
            raise NoFaceError("no face found")
        faces_detected = len(faces)
        # The subject is the largest face (by inter-ocular distance).
        pts, blend, mat = max(faces, key=lambda f: np.linalg.norm(np.subtract(*iris_centres(f[0]))))
        pts = np.column_stack([g.apply(np.linalg.inv(m0), pts[:, :2]), pts[:, 2] / s0])
        angle = eye_line_angle(pts)
        history = [math.degrees(angle)]
        for _ in range(passes):
            res = self._upright_pass(img, pts, angle, crop_px)
            if res is None:
                break
            pts, blend, mat = res
            angle = eye_line_angle(pts)
            history.append(math.degrees(angle))
        # Test-time augmentation: the iris network jitters by ~0.1-0.2 deg between crops, so the
        # final mesh is the per-landmark median over 9 zoom/rotation-perturbed upright crops.
        samples = []
        for zoom in (0.92, 1.0, 1.08):
            for rot in (-1.5, 0.0, 1.5):
                res = self._upright_pass(img, pts, angle, crop_px, zoom, math.radians(rot))
                if res is not None:
                    samples.append(res[0])
        if samples:
            stack = np.stack(samples)
            pts = np.median(stack, axis=0)
            angle = eye_line_angle(pts)
            spread = float(np.std([math.degrees(eye_line_angle(p)) for p in stack]) / math.sqrt(len(samples)))
        else:
            spread = float("nan")
        history.append(math.degrees(angle))
        obs = FaceObservation(
            points=pts,
            eye_angle_rad=angle,
            blendshapes=blend,
            pose_deg=_pose_from_matrix(mat),
            faces_detected=faces_detected,
            passes=len(history) - 1,
            angle_history_deg=history,
        )
        obs.angle_stderr_deg = spread
        obs.tta_samples = len(samples)
        return obs
