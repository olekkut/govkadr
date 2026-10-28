"""Generative edge fill: complete the subject where the 35x45 crop runs past the original photo.

Only pixels that are both outside the source frame and inside the (geometrically extended)
silhouette are generated, never the face. A well-framed photo of the same person from the same
session is used as the reference image, so the jacket, shirt and tie continue correctly. The
strict-gov photo is never filled (gov.pl forbids altering the photo); virtual-studio, studio-ai
and studio-flux use the filled subject.
"""

from __future__ import annotations

import re

import cv2
import numpy as np

from . import geometry as g
from .landmarks import FACE_OVAL

MIN_FILL_PX = 400  # at working resolution (~100 output pixels); smaller gaps are not worth a model load


def fill_mask(alpha: np.ndarray, coverage: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Pixels to generate: outside the photo, near the (extended) silhouette, away from the face.

    The band reaches 40 px beyond the extended silhouette so the generator, not the crude
    extension, decides where a sloping shoulder ends (see ``alpha_from_fill``).
    """
    near = cv2.dilate((alpha > 0.02).astype(np.uint8), np.ones((81, 81), np.uint8)).astype(bool)
    mask = (~coverage) & near
    face = np.zeros(alpha.shape, np.uint8)
    cv2.fillPoly(face, [np.round(pts[list(FACE_OVAL)]).astype(np.int32)], 1)
    face = cv2.dilate(face, np.ones((41, 41), np.uint8)).astype(bool)
    return mask & ~face


def _session_and_time(stem: str) -> tuple[str, int]:
    """(session, seconds of day) from camera names like PXL_20261027_115316389; ('', 0) otherwise."""
    m = re.search(r"(\d{8})_(\d{2})(\d{2})(\d{2})", stem)
    if not m:
        return "", 0
    return m.group(1), int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4))


def pick_reference(job, candidates) -> object | None:
    """Best-framed other photo from the same session (same date), nearest in capture time.

    Photos whose names carry no date are treated as one session.
    """
    session, t = _session_and_time(job.stem)

    def score(c):
        cov = c.report.get("strict-gov", {}).get("coverage", {})
        return (cov.get("subject_rows_filled", 99) + cov.get("subject_cols_filled", 99), abs(_session_and_time(c.stem)[1] - t))

    pool = [
        c
        for c in candidates
        if c is not job and _session_and_time(c.stem)[0] == session and c.report.get("strict-gov", {}).get("verdict") in ("PASS", "PASS_WITH_WARNINGS")
    ]
    pool = [c for c in pool if c.report["strict-gov"].get("coverage", {}).get("subject_rows_filled", 99) == 0]
    return min(pool, key=score) if pool else None


def blend(lin: np.ndarray, fill_lin: np.ndarray, mask: np.ndarray, feather_px: int = 12) -> np.ndarray:
    """Feathered composite of generated pixels into the mask (and a thin seam band around it)."""
    grown = cv2.dilate(mask.astype(np.uint8), np.ones((2 * feather_px + 1,) * 2, np.uint8))
    w = cv2.GaussianBlur(grown.astype(np.float32), (0, 0), feather_px / 2)
    w = np.maximum(w, mask.astype(np.float32))[..., None]
    return lin * (1 - w) + fill_lin * w


def coverage_for(job, k: int, size: tuple[int, int], m_out: np.ndarray) -> np.ndarray:
    src = job.report["source"]
    return g.coverage((src["height"], src["width"]), g.resample_scale(k) @ m_out, size)


def alpha_from_fill(fill_srgb: np.ndarray, alpha: np.ndarray, mask: np.ndarray, feather_px: int = 6) -> np.ndarray:
    """Matte of the generated strip, read from the render itself.

    The generator was asked for a plain white background, so anything clearly darker than white
    is subject. Near-white clothing is classed as background, but it composites to the same white.
    Replaces the nearest-pixel extension, which smeared a sloping shoulder edge into a grey band.
    """
    d = (1.0 - fill_srgb.min(axis=2)).astype(np.float32)
    a_fill = np.clip((d - 0.04) / 0.12, 0.0, 1.0)
    a_fill = cv2.GaussianBlur(a_fill, (0, 0), 1.0)
    w = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), feather_px)
    w = np.where(mask, 1.0, w)  # fully generated inside the mask, feathered just outside it
    return (alpha * (1 - w) + a_fill * w).astype(np.float32)
