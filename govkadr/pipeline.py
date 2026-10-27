"""Stage-major orchestration.

Each network is loaded once, run over every photo, and unloaded before the next one is loaded, so
peak memory is that of the largest single model (never two at once):

    analyze        MediaPipe face mesh + iterative roll (CPU)
    matting        BiRefNet alpha, crown detection, framing solve (GPU)
    strict         strict-gov render, compliance, sheets, diagnostics (CPU)
    studio         IC-Light render only (GPU)
    studio-finish  register render, transfer lighting, compliance, sheets (CPU)
    identity       ArcFace drift audit (CPU / ONNX Runtime)

The CLI runs each stage in its own process. Intermediate results live in
``<output>/.cache/<photo>/`` so stages resume per photo and can be re-run alone.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import framing, imageio, overlay, resources, sheet
from . import geometry as g
from . import photometric as ph
from .compliance import measure
from .landmarks import CHIN, FaceAnalyzer, FaceObservation, NoFaceError, iris_centres
from .jobs import Job, make_jobs, write_json
from .spec import PHOTO, SHEET_4x6, SHEET_10x15

K = 2  # supersampling of the working canvas relative to the 826x1062 output
MODES = ("strict-gov", "studio-ai")


def _save_obs(path: Path, obs: FaceObservation) -> None:
    write_json(path, {k: getattr(obs, k) for k in ("points", "eye_angle_rad", "blendshapes", "pose_deg", "faces_detected", "passes", "angle_history_deg", "angle_stderr_deg", "tta_samples")})


def _load_obs(path: Path) -> FaceObservation:
    d = json.loads(path.read_text(encoding="utf-8"))
    obs = FaceObservation(np.array(d["points"]), d["eye_angle_rad"], d["blendshapes"], d["pose_deg"], d["faces_detected"], d["passes"], d["angle_history_deg"])
    obs.angle_stderr_deg, obs.tta_samples = d["angle_stderr_deg"], d["tta_samples"]
    return obs


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- 1. analyze
def stage_analyze(jobs: list[Job], analyzer: FaceAnalyzer) -> None:
    for job in jobs:
        if job.failed or job.done("analyze"):
            continue
        try:
            img, meta = imageio.load_srgb(job.src)
            obs = analyzer.analyze(img)
            _save_obs(job.cache / "face.json", obs)
            left, right = iris_centres(obs.points)
            job.save("source", meta)
            job.save("analysis", {
                "faces_detected": obs.faces_detected,
                "eye_line_tilt_deg": round(obs.eye_angle_deg, 3),
                "eye_line_tilt_stderr_deg": round(obs.angle_stderr_deg, 3),
                "applied_rotation_deg_ccw": round(obs.eye_angle_deg, 3),
                "estimate_history_deg": [round(a, 3) for a in obs.angle_history_deg],
                "inter_pupillary_px_source": round(float(np.linalg.norm(right - left)), 1),
                "pose_deg": {k: round(v, 2) for k, v in obs.pose_deg.items()},
            })
            _log(f"analyze  {job.stem}: tilt {obs.eye_angle_deg:+.2f} deg (+/-{obs.angle_stderr_deg:.2f}), IPD {np.linalg.norm(right - left):.0f}px")
        except NoFaceError as e:
            job.fail(f"analyze: {e}")
            _log(f"analyze  {job.stem}: FAILED {e}")


# --------------------------------------------------------------------------- 2. matting
def stage_matting(jobs: list[Job]) -> None:
    from .matting import Matting, height_for_budget

    budget = resources.gpu_budget_gb()
    height = height_for_budget(budget - 0.45, 3.2 / 4.1)  # 0.45 GB: weights + CUDA context
    _log(f"matting: GPU budget {budget:.2f} GB -> ROI height {height}px ({resources.snapshot()})")
    model = Matting()
    for job in jobs:
        if job.failed or job.done("matting"):
            continue
        try:
            img, _ = imageio.load_srgb(job.src)
            obs = _load_obs(job.cache / "face.json")
            r3 = framing.rotated_frame(obs)
            top_extra, notes = 0.0, []
            for _attempt in range(4):
                m_roi, (w, h), _ = framing.roi_transform(obs, height, top_extra)
                # Outside the photo, pad with the wall colour: edge-replicated streaks confused the
                # network into cutting off a jacket shoulder next to them. The silhouette is then
                # extended geometrically into the padding so the subject's outline stays continuous.
                cov = g.coverage(img.shape, m_roi, (w, h))
                wall = tuple(int(v) for v in np.median(np.concatenate([img[0], img[:, 0], img[:, -1]]), axis=0))
                roi = g.warp(img, m_roi, (w, h), border=cv2.BORDER_CONSTANT, border_value=wall)
                alpha = model(roi)
                if not cov.all():
                    # Each pixel outside the photo takes the matte value of the nearest pixel inside it
                    # (from 3 px in, past the padding seam), so a shoulder cut by the frame continues
                    # straight out at full opacity.
                    inner = cv2.erode(cov.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
                    _, labels = cv2.distanceTransformWithLabels((~inner).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
                    lut = np.zeros(labels.max() + 1, np.float32)
                    lut[labels[inner]] = alpha[inner]
                    # Only the body continues past the frame: above the eye line the head ends, so
                    # hair touching the photo's top edge must not be stretched up into the margin.
                    eye_row = g.apply(m_roi, 0.5 * np.add(*iris_centres(obs.points)))[1]
                    below = np.arange(h)[:, None] > eye_row
                    alpha = np.where(inner | ~below, alpha, lut[labels])
                pts = g.apply(m_roi, obs.points[:, :2])
                axis, fw = framing.face_axis_x(pts), framing.face_width(pts)
                crown = framing.find_crown(alpha, cov, axis - 0.55 * fw, axis + 0.55 * fw, max(3, round(0.004 * h)))
                if crown is None:
                    raise RuntimeError("no head silhouette in the matte")
                if crown.touches_roi_top and not crown.touches_source_top:
                    top_extra += 0.8
                    notes.append("hair reached ROI top: ROI extended")
                    continue
                crown_rot = g.apply(r3 @ np.linalg.inv(m_roi), [crown.x_roi, crown.y_roi])
                fr = framing.solve(obs, tuple(crown_rot))
                if not framing.canvas_inside(fr.m_out, m_roi, (w, h)):
                    top_extra += 0.5
                    notes.append("canvas exceeded ROI: ROI extended")
                    continue
                break
            np.savez_compressed(job.cache / "matte.npz", alpha=alpha.astype(np.float16), m_roi=m_roi)
            write_json(job.cache / "framing.json", {**fr.to_dict(), "crown_cut": crown.touches_source_top})
            job.save("matting", {
                "model": model.name,
                "roi_px": [w, h],
                **model.last,
                "crown_touches_photo_edge": crown.touches_source_top,
                "notes": notes,
            })
            job.save("framing", {
                "scale_output_px_per_source_px": round(fr.scale, 5),
                "head_height_source_px": round(fr.head_src_px, 1),
                "crown_above_eye_line_source_px": round(-fr.crown_rot[1], 1),
                "chin_below_eye_line_source_px": round(fr.chin_rot[1], 1),
            })
            _log(f"matting  {job.stem}: {w}x{h} in {model.last.get('seconds')}s, head {fr.head_src_px:.0f}px src, scale {fr.scale:.4f}" + (" CROWN CUT" if crown.touches_source_top else ""))
        except Exception as e:  # noqa: BLE001 - one bad photo must not stop the batch
            job.fail(f"matting: {e}")
            _log(f"matting  {job.stem}: FAILED {e}\n{traceback.format_exc()}")
    model.close()


# --------------------------------------------------------------------------- 3. strict-gov
def _render_inputs(job: Job, img: np.ndarray):
    obs = _load_obs(job.cache / "face.json")
    fj = json.loads((job.cache / "framing.json").read_text(encoding="utf-8"))
    fr = framing.Framing.from_dict(fj)
    mz = np.load(job.cache / "matte.npz")
    size = (PHOTO.width_px * K, PHOTO.height_px * K)
    m_work = g.resample_scale(K) @ fr.m_out
    work = g.warp(img, m_work, size).astype(np.float32) / 255.0
    from .matting import clean_alpha

    alpha = g.warp(mz["alpha"].astype(np.float32), m_work @ np.linalg.inv(mz["m_roi"]), size, interp=cv2.INTER_CUBIC)
    alpha = clean_alpha(np.clip(alpha, 0.0, 1.0))
    cov = g.coverage(img.shape, m_work, size)
    return obs, fr, fj["crown_cut"], m_work, work, alpha, cov


def _downsample(x: np.ndarray) -> np.ndarray:
    return cv2.resize(x, (PHOTO.width_px, PHOTO.height_px), interpolation=cv2.INTER_AREA)


def _deliver(job: Job, mode: str, comp_work: np.ndarray, alpha_work: np.ndarray, analyzer, obs, fr, crown_cut, coverage_info, src_small, src_scale, extra) -> dict:
    """Downsample, save, re-measure, and write diagnostics + print sheets for one mode."""
    out = imageio.to_uint8(_downsample(comp_work))
    alpha_out = _downsample(alpha_work)
    photo_path = job.path(mode, "35x45mm_600dpi")
    imageio.save_jpeg(photo_path, out, quality=PHOTO.jpeg_quality)
    delivered = np.asarray(Image.open(photo_path).convert("RGB"))
    info = imageio.inspect_output(photo_path)
    meas, out_obs = measure(delivered, alpha_out, analyzer, info, obs, coverage_info, crown_cut)
    diag = overlay.compose(
        src_small, src_scale, obs, fr.m_out, delivered, alpha_out, out_obs, meas,
        f"govkadr · {mode}", f"{job.src.name} · 35x45 mm @ 600 dpi · gov.pl / ICAO 9303", extra,
    )
    imageio.save_jpeg(job.path(mode, "diagnostic"), diag, quality=90)
    for spec in (SHEET_10x15, SHEET_4x6):
        imageio.save_jpeg(job.path(mode, spec.name), sheet.build(delivered, spec), quality=95)
    meas["files"] = {
        "photo": str(photo_path),
        "diagnostic": str(job.path(mode, "diagnostic")),
        "sheet_10x15cm": str(job.path(mode, SHEET_10x15.name)),
        "sheet_4x6in": str(job.path(mode, SHEET_4x6.name)),
    }
    return meas


def stage_strict(jobs: list[Job], analyzer: FaceAnalyzer) -> None:
    from .matting import estimate_foreground

    for job in jobs:
        if job.failed or job.done("strict"):
            continue
        try:
            t0 = time.perf_counter()
            img, _ = imageio.load_srgb(job.src)
            obs, fr, crown_cut, m_work, work, alpha, cov = _render_inputs(job, img)
            src_small, src_scale = overlay.source_thumbnail(img)
            del img
            filled = (~cov) & (alpha > 0.5)
            coverage_info = {
                "subject_rows_filled": int(np.ceil(filled.any(axis=1).sum() / K)),
                "subject_cols_filled": int(np.ceil(filled.any(axis=0).sum() / K)),
                "canvas_outside_photo_pct": round(100 * float((~cov).mean()), 2),
            }
            fg = estimate_foreground(work, alpha)
            pts = g.apply(m_work, obs.points[:, :2])
            left, right = iris_centres(pts)
            eye_y = 0.5 * (left[1] + right[1])
            chin_y = float(pts[CHIN, 1])
            d = chin_y - eye_y
            axis = framing.face_axis_x(pts)
            lin, corr, masks = ph.normalise(fg, work, alpha, cov, pts, chin_y, d, axis, eye_y, d)
            fg_srgb = ph.linear_to_srgb(np.clip(lin, 0.0, 1.0))
            comp = ph.composite_on_white(fg_srgb, alpha)
            extra = [
                ("rotation applied", f"{obs.eye_angle_deg:+.2f} deg CCW (+/-{obs.angle_stderr_deg:.2f})"),
                ("scale", f"{fr.scale:.4f} out px / src px"),
                ("white balance gains", str(corr.report["white_balance"]["gains"])),
                ("L/R balance (EV, l-r)", str(corr.report["left_right_balance"].get("left_minus_right_ev_at_cheeks"))),
                ("exposure", str(corr.report["exposure"].get("ev", 0.0)) + " EV"),
            ]
            meas = _deliver(job, "strict-gov", comp, alpha, analyzer, obs, fr, crown_cut, coverage_info, src_small, src_scale, extra)
            meas["photometric"] = corr.report
            meas["coverage"] = coverage_info
            meas["seconds"] = round(time.perf_counter() - t0, 1)
            job.save("strict-gov", meas)
            np.savez_compressed(job.cache / "subject.npz", lin=lin.astype(np.float16), alpha=alpha.astype(np.float16), pts=pts)
            np.savez_compressed(job.cache / "thumb.npz", img=src_small, scale=src_scale)
            m = meas["measurements"]
            _log(f"strict   {job.stem}: {meas['verdict']} head {m['head_height_mm']}mm offset {m['horizontal_offset_mm']:+.2f}mm roll {m['roll_residual_deg']:+.3f} ({meas['seconds']}s)")
        except Exception as e:  # noqa: BLE001
            job.fail(f"strict-gov: {e}")
            _log(f"strict   {job.stem}: FAILED {e}\n{traceback.format_exc()}")


# --------------------------------------------------------------------------- 4. studio-ai
def _gain_visual(gain: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """log2 luminance gain as a diverging map: blue = darkened, red = brightened, white = 0 EV."""
    ev = np.log2(np.maximum(ph.luminance(gain), 1e-3))
    t = np.clip(ev / 1.0, -1, 1)[..., None]
    red, blue, white = np.array([0.84, 0.19, 0.15]), np.array([0.13, 0.4, 0.75]), np.ones(3)
    vis = np.where(t > 0, white + (red - white) * t, white + (blue - white) * (-t))
    a = alpha[..., None]
    return (vis * a + 0.92 * (1 - a)).astype(np.float32)


def _subject(job: Job):
    """Subject layer for the relighting modes: the edge-filled one when it exists."""
    f = job.cache / "subject_filled.npz"
    return np.load(f if f.exists() else job.cache / "subject.npz")


def stage_fill(jobs: list[Job]) -> None:
    """Generative edge fill (FLUX.2 [klein] inpainting) where the crop runs past the photo."""
    from . import fill as fl

    size = (PHOTO.width_px * K, PHOTO.height_px * K)
    todo = []
    for job in jobs:
        if job.failed or job.done("fill") or not job.done("strict"):
            continue
        z = np.load(job.cache / "subject.npz")
        fr = framing.Framing.from_dict(json.loads((job.cache / "framing.json").read_text(encoding="utf-8")))
        mask = fl.fill_mask(z["alpha"].astype(np.float32), fl.coverage_for(job, K, size, fr.m_out), z["pts"])
        if mask.sum() < fl.MIN_FILL_PX:
            write_json(job.cache / "fill.json", {"applied": False, "pixels": int(mask.sum())})
            (job.cache / "subject_filled.npz").unlink(missing_ok=True)
            continue
        todo.append((job, mask))
    if not todo:
        return
    need = float(os.environ.get("GOVKADR_MIN_FREE_COMMIT_FLUX_GB", "9.2"))
    for _ in range(30):
        if resources.free_commit_gb() >= need:
            break
        _log(f"fill: waiting for memory headroom (free commit {resources.free_commit_gb():.2f} GB, need {need})")
        time.sleep(10)
    if resources.free_commit_gb() < need and os.environ.get("GOVKADR_ALLOW_LOW_MEMORY") != "1":
        for job, mask in todo:
            write_json(job.cache / "fill.json", {"applied": False, "pixels": int(mask.sum()), "skipped": "not enough free memory"})
        _log("fill: skipped, not enough free memory")
        return
    from .flux_relight import FluxFiller

    _log(f"fill: {len(todo)} photo(s) need edge fill; loading FLUX.2 [klein] inpainting ({resources.snapshot()})")
    filler = FluxFiller()
    for job, mask in todo:
        try:
            z = np.load(job.cache / "subject.npz")
            lin, alpha, pts = z["lin"].astype(np.float32), z["alpha"].astype(np.float32), z["pts"]
            image = imageio.to_uint8(ph.composite_on_white(ph.linear_to_srgb(np.clip(lin, 0, 1)), alpha))
            seam = cv2.dilate(mask.astype(np.uint8), np.ones((17, 17), np.uint8))
            # FLUX also conditions on the input image: replace the edge-replicated streaks under the
            # mask with a smooth extrapolation, or it copies their pattern into the fill.
            small = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            small_mask = cv2.resize(seam, small.shape[1::-1], interpolation=cv2.INTER_NEAREST)
            smooth = cv2.resize(cv2.inpaint(small, small_mask, 9, cv2.INPAINT_TELEA), image.shape[1::-1], interpolation=cv2.INTER_CUBIC)
            image = np.where(seam[..., None] > 0, smooth, image)
            # Search every processed photo, not just the ones selected with --only.
            ref_job = fl.pick_reference(job, [c.load() for c in make_jobs(job.src.parent, job.cache.parent.parent)])
            ref = np.asarray(Image.open(ref_job.path("strict-gov", "35x45mm_600dpi")).convert("RGB")) if ref_job else None
            out, info = filler.fill(image, seam * 255, ref)
            fill_lin = ph.srgb_to_linear(cv2.resize(out, size, interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0)
            lin_f = fl.blend(lin, fill_lin, mask)
            alpha_f = fl.alpha_from_fill(ph.linear_to_srgb(fill_lin), alpha, mask)
            np.savez_compressed(job.cache / "subject_filled.npz", lin=lin_f.astype(np.float16), alpha=alpha_f.astype(np.float16), pts=pts)
            info.update({
                "applied": True,
                "pixels": int(mask.sum()),
                "area_mm2_on_print": round(float(mask.sum()) / (K * K) / (PHOTO.width_px / PHOTO.width_mm) ** 2, 1),
                "reference_photo": ref_job.src.name if ref_job else None,
            })
            write_json(job.cache / "fill.json", info)
            _log(f"fill     {job.stem}: {info['area_mm2_on_print']} mm2 generated, reference {info['reference_photo']}, {info['seconds']}s")
        except Exception as e:  # noqa: BLE001
            write_json(job.cache / "fill.json", {"applied": False, "error": str(e)})
            _log(f"fill     {job.stem}: FAILED {e}")
            raise SystemExit(3) from e
    filler.close()


def _fill_note(job: Job) -> tuple[str, str] | None:
    f = job.cache / "fill.json"
    if not f.exists():
        return None
    info = json.loads(f.read_text(encoding="utf-8"))
    if not info.get("applied"):
        return None
    return ("edge fill (generative)", f"{info['area_mm2_on_print']} mm2 of clothing generated, ref. {info['reference_photo']}")


def stage_virtual(jobs: list[Job], analyzer: FaceAnalyzer, preset_name: str = "clamshell") -> None:
    """virtual-studio: physically based relighting with virtual softboxes (CPU, no generative model)."""
    from . import virtual_studio as vs

    preset = vs.PRESETS[preset_name]
    for job in jobs:
        if job.failed or job.done("virtual") or not job.done("strict"):
            continue
        try:
            t0 = time.perf_counter()
            z = _subject(job)
            lin, alpha, pts = z["lin"].astype(np.float32), z["alpha"].astype(np.float32), z["pts"]
            th = np.load(job.cache / "thumb.npz")
            obs = _load_obs(job.cache / "face.json")
            fj = json.loads((job.cache / "framing.json").read_text(encoding="utf-8"))
            fr = framing.Framing.from_dict(fj)
            pts3 = np.column_stack([pts, obs.points[:, 2] * fr.scale * K])  # mesh depth in working px
            masks = ph.face_masks(pts, alpha, lin)
            geo = vs.build_geometry(lin, alpha, pts3, masks.skin, masks.face_w)
            tiles, infos, chosen = [_downsample(ph.composite_on_white(ph.linear_to_srgb(np.clip(lin, 0, 1)), alpha))], {}, None
            for p in vs.PRESETS.values():
                out_lin, gain, info = vs.relight(lin, alpha, geo, p)
                comp = ph.composite_on_white(ph.linear_to_srgb(np.clip(out_lin, 0, 1)), alpha)
                tiles.append(_downsample(comp))
                infos[p.name] = info
                if p.name == preset.name:
                    chosen = (comp, gain)
            a_small = cv2.resize(alpha, geo.normals.shape[1::-1], interpolation=cv2.INTER_AREA)
            tiles.append(_downsample(cv2.resize(vs.normals_visual(geo.normals, a_small), alpha.shape[::-1], interpolation=cv2.INTER_LINEAR)))
            comp, gain = chosen
            extra = [
                ("preset", f"{preset.name}: {preset.description}"),
                ("geometry", "MediaPipe mesh normals + inflated matte silhouette"),
                ("old light (SH2 fit R^2)", str(infos[preset.name]["old_light_fit_r2"])),
                ("skin gain EV p5/50/95", str(infos[preset.name]["skin_gain_ev_p5_p50_p95"])),
                ("note", "lighting is computed, not generated; ID suitability depends on preset"),
            ] + ([n] if (n := _fill_note(job)) else [])
            coverage = job.report.get("strict-gov", {}).get("coverage", {})
            meas = _deliver(job, "virtual-studio", comp, alpha, analyzer, obs, fr, fj["crown_cut"], coverage, th["img"], float(th["scale"]), extra)
            imageio.save_jpeg(job.path("virtual-studio", "presets"), np.hstack(tiles), quality=90)
            meas["relight"] = {"preset": preset.name, "presets": infos}
            meas["files"]["presets"] = str(job.path("virtual-studio", "presets"))
            meas["seconds"] = round(time.perf_counter() - t0, 1)
            job.save("virtual-studio", meas)
            _log(f"virtual  {job.stem}: {meas['verdict']}, {preset.name} skin EV {infos[preset.name]['skin_gain_ev_p5_p50_p95']}, SH R2 {infos[preset.name]['old_light_fit_r2']}")
        except Exception as e:  # noqa: BLE001
            job.save("virtual-studio", {"verdict": "ERROR", "error": str(e)})
            _log(f"virtual  {job.stem}: FAILED {e}" + chr(10) + traceback.format_exc())


def stage_studio(jobs: list[Job]) -> None:
    """GPU half of studio-ai: IC-Light renders only (registration and QC run in another process)."""
    from .relight import StudioRelighter

    todo = [j for j in jobs if not j.failed and j.done("strict") and not j.done("studio")]
    if not todo:
        return
    # The IC-Light process peaks at ~4.6 GB of commit (torch + CUDA libraries ~2.3 GB, weights,
    # activations). Running it with less headroom pushes Windows to its commit limit, where *other*
    # applications' allocations start failing - so wait for headroom, and by default skip rather
    # than gamble with the user's session.
    need = float(os.environ.get("GOVKADR_MIN_FREE_COMMIT_GB", "5.2"))
    for _ in range(30):
        if resources.free_commit_gb() >= need:
            break
        _log(f"studio: waiting for memory headroom (free commit {resources.free_commit_gb():.2f} GB, need {need})")
        time.sleep(10)
    free = resources.free_commit_gb()
    if free < need and os.environ.get("GOVKADR_ALLOW_LOW_MEMORY") != "1":
        reason = (
            f"skipped: only {free:.1f} GB of free system commit, IC-Light needs ~{need} GB (measured peak 8.7 GB). Close memory-heavy "
            "applications or raise the Windows page-file limit, or re-run with --allow-low-memory."
        )
        for job in todo:
            (job.cache / "render_error.txt").write_text(reason, encoding="utf-8")
        _log(f"studio: {reason}")
        return
    _log(f"studio: loading IC-Light ({resources.snapshot()})")
    relighter = StudioRelighter()
    _log(f"studio: loaded ({resources.snapshot()})")
    for job in todo:
        for attempt in range(2):
            try:
                z = _subject(job)
                lin, alpha = z["lin"].astype(np.float32), z["alpha"].astype(np.float32)
                fg = ph.linear_to_srgb(np.clip(lin, 0, 1))
                grey = imageio.to_uint8(fg * alpha[..., None] + (127.0 / 255.0) * (1 - alpha[..., None]))
                del z, lin, alpha, fg
                render, info = relighter.relight(grey)
                cv2.imwrite(str(job.cache / "render.png"), cv2.cvtColor(render, cv2.COLOR_RGB2BGR))
                write_json(job.cache / "render.json", info)
                _log(f"studio   {job.stem}: rendered in {info['seconds']}s, peak VRAM {info['peak_vram_gb']} GB, free commit {resources.free_commit_gb():.2f} GB")
                break
            except (MemoryError, RuntimeError) as e:
                if attempt == 0 and (isinstance(e, MemoryError) or "memory" in str(e).lower()):
                    _log(f"studio   {job.stem}: out of memory, retrying once after cleanup")
                    import gc

                    import torch

                    gc.collect()
                    torch.cuda.empty_cache()
                    time.sleep(5)
                    continue
                (job.cache / "render_error.txt").write_text(str(e), encoding="utf-8")
                _log(f"studio   {job.stem}: FAILED {e}")
                raise SystemExit(3) from e  # CUDA errors are sticky: let the parent restart the process
    relighter.close()


def stage_flux(jobs: list[Job]) -> None:
    """GPU half of studio-flux: FLUX.2 [klein] renders only."""
    from .flux_relight import FluxRelighter

    todo = [j for j in jobs if not j.failed and j.done("strict") and not j.done("flux")]
    if not todo:
        return
    # Measured with the streamed loader: see ARCHITECTURE.md section 2.8.
    need = float(os.environ.get("GOVKADR_MIN_FREE_COMMIT_FLUX_GB", "9.2"))
    for _ in range(30):
        if resources.free_commit_gb() >= need:
            break
        _log(f"flux: waiting for memory headroom (free commit {resources.free_commit_gb():.2f} GB, need {need})")
        time.sleep(10)
    free = resources.free_commit_gb()
    if free < need and os.environ.get("GOVKADR_ALLOW_LOW_MEMORY") != "1":
        reason = (
            f"skipped: only {free:.1f} GB of free system commit, FLUX.2 [klein] needs ~{need} GB (measured peak 8.7 GB). Close memory-heavy "
            "applications or raise the Windows page-file limit, or re-run with --allow-low-memory."
        )
        for job in todo:
            (job.cache / "flux_error.txt").write_text(reason, encoding="utf-8")
        _log(f"flux: {reason}")
        return
    _log(f"flux: loading FLUX.2 [klein] 4B ({resources.snapshot()})")
    relighter = FluxRelighter()
    _log(f"flux: loaded ({resources.snapshot()})")
    for job in todo:
        try:
            z = _subject(job)
            lin, alpha = z["lin"].astype(np.float32), z["alpha"].astype(np.float32)
            ref = imageio.to_uint8(ph.composite_on_white(ph.linear_to_srgb(np.clip(lin, 0, 1)), alpha))
            del z, lin, alpha
            render, info = relighter.relight(ref)
            cv2.imwrite(str(job.cache / "flux.png"), cv2.cvtColor(render, cv2.COLOR_RGB2BGR))
            write_json(job.cache / "flux.json", info)
            _log(f"flux     {job.stem}: rendered in {info['seconds']}s, peak VRAM {info['peak_vram_gb']} GB")
        except Exception as e:  # noqa: BLE001
            (job.cache / "flux_error.txt").write_text(str(e), encoding="utf-8")
            _log(f"flux     {job.stem}: FAILED {e}")
            raise SystemExit(3) from e  # CUDA errors are sticky: let the parent restart the process
    relighter.close()


def stage_studio_finish(jobs: list[Job], analyzer: FaceAnalyzer, mode: str = "studio-ai", tag: str = "render", stage: str = "studio-finish") -> None:
    """CPU half of a generative mode: register the render to the photo, transfer lighting, deliver, QC."""
    from .identity import FaceIdentity, cosine
    from .relight import register_render, transfer_lighting

    ident = None
    for job in jobs:
        if job.failed or job.done(stage) or not job.done("strict"):
            continue
        if reason := job.skipped_studio(tag):
            job.save(mode, {"verdict": "SKIPPED", "error": reason})
            continue
        if not (job.cache / f"{tag}.png").exists():
            err = job.cache / f"{tag}_error.txt"  # the render is deleted after use: re-run its GPU stage
            job.save(mode, {"verdict": "ERROR", "error": err.read_text(encoding="utf-8") if err.exists() else "no generative render"})
            continue
        try:
            t0 = time.perf_counter()
            info = json.loads((job.cache / f"{tag}.json").read_text(encoding="utf-8"))
            z = _subject(job)
            lin, alpha, pts = z["lin"].astype(np.float32), z["alpha"].astype(np.float32), z["pts"]
            th = np.load(job.cache / "thumb.npz")
            obs = _load_obs(job.cache / "face.json")
            fj = json.loads((job.cache / "framing.json").read_text(encoding="utf-8"))
            fr = framing.Framing.from_dict(fj)
            render = cv2.cvtColor(cv2.imread(str(job.cache / f"{tag}.png")), cv2.COLOR_BGR2RGB)
            # Identity drift of the raw render, recorded as a number only (the image is never kept).
            ident = ident or FaceIdentity()
            e_a, _ = ident.embed(np.asarray(Image.open(job.path("strict-gov", "35x45mm_600dpi")).convert("RGB")))
            e_r, _ = ident.embed(render)
            info["identity_cos_strict_vs_raw_render"] = round(cosine(e_a, e_r), 4) if e_r is not None else None
            h, w = alpha.shape
            to_work = g.resample_scale(w / render.shape[1], h / render.shape[0])
            render_work = cv2.resize(render, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
            found = analyzer.detect(render)
            if found:
                rpts = g.apply(to_work, max(found, key=lambda f: np.ptp(f[0][:, 0]))[0][:, :2])
                aligned = register_render(render_work, rpts, pts, (w, h))
                shift = float(np.linalg.norm(rpts[:468] - pts[:468], axis=1).mean()) / 2.0
                info["registration"] = {"method": "thin-plate spline on the 478-point meshes", "mean_landmark_shift_px": round(shift, 1)}
            else:
                aligned = render_work
                info["registration"] = {"method": "none (no face found in render)"}
            masks = ph.face_masks(pts, alpha, lin)
            studio_lin, gain, tinfo = transfer_lighting(lin, ph.srgb_to_linear(aligned), masks.skin, masks.face_w, framing.face_axis_x(pts))
            info.update(tinfo)
            comp = ph.composite_on_white(ph.linear_to_srgb(np.clip(studio_lin, 0, 1)), alpha)
            lum_ev = np.log2(np.maximum(ph.luminance(gain), 1e-3))[alpha > 0.98]
            info["gain_ev_p5_p50_p95"] = [round(float(v), 3) for v in np.percentile(lum_ev, [5, 50, 95])]
            extra = [
                ("relight", info["model"]),
                ("sampler", f"{info.get('sampler', 'flow matching')}, {info['steps']} steps, cfg {info.get('cfg', info.get('guidance'))}, seed {info['seed']}"),
                ("raw render identity (cos vs strict)", str(info["identity_cos_strict_vs_raw_render"])),
                ("registration", info["registration"]["method"]),
                ("lighting field EV p5/50/95", str(info["gain_ev_p5_p50_p95"])),
                ("chroma gain", str(info["chroma_gain_rgb"])),
                ("note", "R&D output - generative relighting is not for official ID"),
            ] + ([n] if (n := _fill_note(job)) else [])
            coverage = job.report.get("strict-gov", {}).get("coverage", {})
            meas = _deliver(job, mode, comp, alpha, analyzer, obs, fr, fj["crown_cut"], coverage, th["img"], float(th["scale"]), extra)
            fg = ph.linear_to_srgb(np.clip(lin, 0, 1))
            strip = np.hstack([
                _downsample(ph.composite_on_white(fg, alpha)),
                _downsample(comp),
                _downsample(_gain_visual(gain, alpha)),
            ])
            imageio.save_jpeg(job.path(mode, "comparison"), strip, quality=90)
            meas["relight"] = info
            meas["files"]["comparison"] = str(job.path(mode, "comparison"))
            meas["seconds"] = round(time.perf_counter() - t0, 1)
            job.save(mode, meas)
            # The diffusion render is only a lighting source; never keep it as an image.
            (job.cache / f"{tag}.png").unlink(missing_ok=True)
            _log(
                f"{mode}  {job.stem}: {meas['verdict']}, raw render cos {info['identity_cos_strict_vs_raw_render']}, "
                f"lighting EV p5/50/95 {info['gain_ev_p5_p50_p95']}, chroma {info['chroma_gain_rgb']}"
            )
        except Exception as e:  # noqa: BLE001
            job.save(mode, {"verdict": "ERROR", "error": str(e)})
            _log(f"{mode}  {job.stem}: FAILED {e}" + chr(10) + traceback.format_exc())


# --------------------------------------------------------------------------- 5. identity
def stage_identity(jobs: list[Job]) -> dict:
    from .identity import FaceIdentity, cosine

    fi = FaceIdentity()
    src_emb: dict[str, np.ndarray] = {}
    for job in jobs:
        if job.failed and "strict" not in (job.failed or ""):
            continue
        img, _ = imageio.load_srgb(job.src)
        e_src, info_src = fi.embed(img)
        del img
        if e_src is None:
            continue
        src_emb[job.stem] = e_src
        res = {"model": "InsightFace buffalo_l: SCRFD-10GF + ArcFace R50 (WebFace600K)", "source_detection": info_src}
        for key, mode, kind in (("strict_gov", "strict-gov", "35x45mm_600dpi"), ("studio_ai", "studio-ai", "35x45mm_600dpi"), ("virtual_studio", "virtual-studio", "35x45mm_600dpi"), ("studio_flux", "studio-flux", "35x45mm_600dpi")):
            p = job.path(mode, kind)
            if p.exists():
                e, _ = fi.embed(np.asarray(Image.open(p).convert("RGB")))
                res[f"cosine_source_vs_{key}"] = round(cosine(e_src, e), 4) if e is not None else None
        if "cosine_source_vs_strict_gov" in res and "cosine_source_vs_studio_ai" in res:
            ea, _ = fi.embed(np.asarray(Image.open(job.path("strict-gov", "35x45mm_600dpi")).convert("RGB")))
            eb, _ = fi.embed(np.asarray(Image.open(job.path("studio-ai", "35x45mm_600dpi")).convert("RGB")))
            res["cosine_strict_gov_vs_studio_ai"] = round(cosine(ea, eb), 4)
        job.save("identity", res)
        _log(f"identity {job.stem}: " + ", ".join(f"{k.replace('cosine_', '')}={v}" for k, v in res.items() if k.startswith("cosine")))
    stems = list(src_emb)
    baseline: dict = {}
    if len(stems) >= 2:
        e = np.stack([src_emb[s] for s in stems])
        sim = e @ e.T
        pairs = sim[np.triu_indices(len(stems), 1)]
        pairs = pairs[pairs > 0.35]  # same-identity pairs only (ArcFace impostor pairs sit below ~0.3)
        if pairs.size:
            baseline = {
                "description": "ArcFace cosine between different raw photos of the same person = natural shot-to-shot variation",
                "pairs": int(pairs.size),
                "min": round(float(pairs.min()), 4),
                "p5": round(float(np.percentile(pairs, 5)), 4),
                "median": round(float(np.median(pairs)), 4),
                "max": round(float(pairs.max()), 4),
            }
        for i, s in enumerate(stems):
            job = next(j for j in jobs if j.stem == s)
            res = job.report["identity"]
            res["closest_other_raw_shot_cosine"] = round(float(np.delete(sim[i], i).max()), 4)
            b = res.get("cosine_source_vs_studio_ai")
            if b is not None and baseline:
                res["studio_ai_drift"] = round(1 - b, 4)
                res["studio_ai_drift_vs_natural_variation"] = (
                    "smaller than the drift between two real shots (median)" if b >= baseline["median"] else "larger than the median drift between two real shots"
                )
            job.save("identity", res)
    if jobs:
        write_json(jobs[0].cache.parent / "identity_baseline.json", baseline)
    return baseline
