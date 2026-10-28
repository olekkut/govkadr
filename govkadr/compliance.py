"""Independent re-measurement of a finished photo against the specification.

Nothing here trusts the numbers the renderer aimed for: the face mesh is re-detected on the
826x1062 output, the crown is re-extracted from the output matte, and the file is re-read from
disk, so a PASS means the delivered pixels and metadata comply.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .framing import face_axis_x, face_width, find_crown
from .landmarks import CHIN, FaceAnalyzer, FaceObservation, eye_line_angle, iris_centres
from .spec import PHOTO, PhotoSpec, px_to_mm


def _check(status: bool, value, requirement: str, soft: bool = False) -> dict:
    return {"status": "PASS" if status else ("WARN" if soft else "FAIL"), "value": value, "requirement": requirement}


def background_stats(img: np.ndarray, alpha: np.ndarray) -> dict:
    """Uniformity of everything the matte calls background, measured on the decoded JPEG.

    Pixels within 2 px of the silhouette are excluded (they legitimately blend with hair); JPEG
    ringing can still nudge a few pixels in edge blocks to 253-254, hence the "visually white"
    (>= 250 in every channel) companion statistic.
    """
    bg = cv2.erode((alpha < 0.002).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    px = img[bg].astype(np.int16)
    pure = np.all(px == 255, axis=1)
    near = np.all(px >= 250, axis=1)
    return {
        "background_fraction": round(float(bg.mean()), 4),
        "pure_white_fraction": round(float(pure.mean()), 6) if px.size else None,
        "visually_white_fraction": round(float(near.mean()), 6) if px.size else None,
        "mean_rgb": [round(float(v), 3) for v in px.mean(axis=0)] if px.size else None,
        "std_rgb": [round(float(v), 4) for v in px.std(axis=0)] if px.size else None,
        "min_rgb": [int(v) for v in px.min(axis=0)] if px.size else None,
        "top_row_white": bool(np.all(img[0] == 255)),
    }


def measure(
    img: np.ndarray,
    alpha: np.ndarray,
    analyzer: FaceAnalyzer,
    file_info: dict,
    source_obs: FaceObservation,
    coverage_issue_px: dict,
    crown_cut: bool,
    spec: PhotoSpec = PHOTO,
) -> tuple[dict, FaceObservation]:
    obs = analyzer.analyze(img)
    pts = obs.points[:, :2]
    left, right = iris_centres(pts)
    eye_y = 0.5 * (left[1] + right[1])
    axis = face_axis_x(pts)
    fw = face_width(pts)
    crown = find_crown(alpha, np.ones(alpha.shape, bool), axis - 0.55 * fw, axis + 0.55 * fw, 3)
    chin_y = float(pts[CHIN, 1])
    crown_y = crown.y_roi if crown else float("nan")
    head_px = chin_y - crown_y
    head_mm = px_to_mm(head_px)
    top_mm = px_to_mm(crown_y + 0.5)
    offset_mm = px_to_mm(axis - (spec.width_px - 1) / 2.0)
    roll = math.degrees(eye_line_angle(pts))
    eye_frac = (spec.height_px - 0.5 - eye_y) / spec.height_px
    bg = background_stats(img, alpha)
    bs = obs.blendshapes
    blink = max(bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0))
    gaze = max(
        bs.get(k, 0.0)
        for k in (
            "eyeLookInLeft",
            "eyeLookInRight",
            "eyeLookOutLeft",
            "eyeLookOutRight",
            "eyeLookUpLeft",
            "eyeLookUpRight",
            "eyeLookDownLeft",
            "eyeLookDownRight",
        )
    )
    smile = max(bs.get("mouthSmileLeft", 0.0), bs.get("mouthSmileRight", 0.0))
    jaw = bs.get("jawOpen", 0.0)
    yaw, pitch = source_obs.pose_deg.get("yaw", float("nan")), source_obs.pose_deg.get("pitch", float("nan"))

    face = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    oval_box = pts[:, :2]
    x0, y0 = np.clip(oval_box.min(axis=0).astype(int), 0, None)
    x1, y1 = oval_box.max(axis=0).astype(int)
    sharp = float(cv2.Laplacian(face[y0:y1, x0:x1], cv2.CV_64F).var())

    measurements = {
        "roll_residual_deg": round(roll, 3),
        "roll_residual_stderr_deg": round(obs.angle_stderr_deg, 3),
        "crown_y_px": round(crown_y, 2),
        "chin_y_px": round(chin_y, 2),
        "head_height_px": round(head_px, 1),
        "head_height_mm": round(head_mm, 2),
        "head_height_fraction": round(head_mm / spec.height_mm, 4),
        "top_margin_mm": round(top_mm, 2),
        "face_axis_x_px": round(axis, 2),
        "horizontal_offset_mm": round(offset_mm, 2),
        "eye_line_from_bottom_fraction": round(eye_frac, 4),
        "inter_pupillary_px": round(float(np.linalg.norm(right - left)), 1),
        "face_sharpness_laplacian_var": round(sharp, 1),
        "background": bg,
        "blendshapes": {"eye_blink_max": round(blink, 3), "jaw_open": round(jaw, 3), "smile_max": round(smile, 3), "gaze_offaxis_max": round(gaze, 3)},
        "pose_source_deg": {k: round(v, 2) for k, v in source_obs.pose_deg.items()},
    }
    checks = {
        "pixel_dimensions": _check(
            (file_info["width_px"], file_info["height_px"]) == (spec.width_px, spec.height_px),
            f"{file_info['width_px']}x{file_info['height_px']}",
            f"{spec.width_px}x{spec.height_px} px (35x45 mm @ 600 dpi)",
        ),
        "dpi_metadata": _check(
            file_info["jfif_dpi"] == [600.0, 600.0] and file_info["exif_dpi"] == [600.0, 600.0],
            {"jfif": file_info["jfif_dpi"], "exif": file_info["exif_dpi"]},
            "600 dpi in JFIF and EXIF",
        ),
        "file_size": _check(file_info["file_bytes"] <= spec.max_file_bytes, file_info["file_bytes"], "<= 2.5 MB (gov.pl upload)"),
        "online_min_resolution": _check(
            file_info["width_px"] >= spec.min_px_passport[0] and file_info["height_px"] >= spec.min_px_passport[1],
            f"{file_info['width_px']}x{file_info['height_px']}",
            "ID >= 492x633, passport >= 768x1004",
        ),
        "colour_profile": _check(file_info["icc_profile"] is not None and "sRGB" in file_info["icc_profile"], file_info["icc_profile"], "embedded sRGB ICC"),
        "head_height": _check(
            spec.head_min_mm <= head_mm <= spec.head_max_mm,
            f"{head_mm:.2f} mm / {head_px:.0f} px",
            f"{spec.head_min_mm}-{spec.head_max_mm} mm (target {spec.head_target_mm})",
        ),
        "top_margin": _check(
            spec.top_margin_min_mm - 0.1 <= top_mm <= spec.top_margin_max_mm + 0.1,
            f"{top_mm:.2f} mm",
            f"{spec.top_margin_min_mm}-{spec.top_margin_max_mm} mm above crown",
            soft=True,
        ),
        "horizontal_centering": _check(abs(offset_mm) <= spec.center_tolerance_mm, f"{offset_mm:+.2f} mm", f"+/-{spec.center_tolerance_mm} mm (target 0.0)"),
        "eye_line_level": _check(abs(roll) <= 0.5, f"{roll:+.3f} deg", "head not tilted (residual |roll| <= 0.5 deg)"),
        "background_uniform_white": _check(
            (bg["pure_white_fraction"] or 0) >= 0.995 and (bg["visually_white_fraction"] or 0) >= 0.999 and bg["top_row_white"],
            f"{100 * (bg['pure_white_fraction'] or 0):.2f} % #FFFFFF, {100 * (bg['visually_white_fraction'] or 0):.3f} % >= 250",
            ">= 99.5 % exactly #FFFFFF, >= 99.9 % >= 250, clear top edge",
        ),
        "crown_in_source": _check(not crown_cut, "cut by photo edge" if crown_cut else "fully visible", "whole head visible in the source photo"),
        "source_coverage": _check(
            coverage_issue_px.get("subject_rows_filled", 0) <= 3, coverage_issue_px, "subject fully inside the source photo (no edge fill)", soft=True
        ),
        "single_face": _check(source_obs.faces_detected == 1, source_obs.faces_detected, "only the photographed person in frame", soft=True),
        "eyes_open": _check(blink < 0.5, round(blink, 3), "eyes naturally open (blink score < 0.5)", soft=True),
        "mouth_closed": _check(jaw < 0.12, round(jaw, 3), "mouth closed (jawOpen < 0.12)", soft=True),
        "neutral_expression": _check(smile < 0.45, round(smile, 3), "neutral expression (smile < 0.45)", soft=True),
        "gaze_at_camera": _check(gaze < 0.55, round(gaze, 3), "looking at the lens (eyeLook* < 0.55)", soft=True),
        "frontal_pose": _check(abs(yaw) <= 5 and abs(pitch) <= 5, f"yaw {yaw:+.1f}, pitch {pitch:+.1f} deg", "|yaw|, |pitch| <= 5 deg (ICAO)", soft=True),
        "eye_line_height": _check(
            spec.eye_line_min_frac <= eye_frac <= spec.eye_line_max_frac,
            f"{100 * eye_frac:.1f} % from bottom",
            "50-70 % of height (ISO/IEC 19794-5, informational)",
            soft=True,
        ),
    }
    statuses = [c["status"] for c in checks.values()]
    verdict = "FAIL" if "FAIL" in statuses else ("PASS_WITH_WARNINGS" if "WARN" in statuses else "PASS")
    return {"verdict": verdict, "checks": checks, "measurements": measurements, "file": file_info}, obs
