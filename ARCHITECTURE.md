# govkadr — architecture

Local pipeline that turns phone portraits into 35 × 45 mm biometric photos for Polish ID cards and
passports (gov.pl *Zdjęcie do dowodu lub paszportu*, ICAO Doc 9303 Part 3), in four modes:

| Mode | What it may change | Intended use |
|---|---|---|
| `strict-gov` | geometry (one similarity transform), background, *smooth* multiplicative exposure / white-balance / left-right fields | submission-safe output |
| `virtual-studio` | strict-gov **plus** a physically computed softbox lighting field (§2.7); no generative model | studio portraits; `clamshell` / `high-key` presets are ID-style |
| `studio-ai` | strict-gov **plus** a relighting field taken from IC-Light (SD 1.5), constrained and audited | R&D, **not** for official submission |
| `studio-flux` | the same with FLUX.2 [klein] 4B as the lighting source (§2.8) | R&D, **not** for official submission |

Every number below was measured on the target machine (RTX 5060 Laptop 8 GB, Ryzen 7 260, Windows 11,
Pixel 8 Pro test set) unless stated otherwise.

---

## 1. Platform findings that shaped the design

**Blackwell (sm_120).** PyTorch 2.14.1 + CUDA 13.0 wheels (`download.pytorch.org/whl/cu130`) ship native
`sm_120` kernels (`torch.cuda.get_arch_list()` contains `sm_120`); fp16, bf16 and SDPA all verified.
Python is pinned to 3.12 (best wheel coverage for MediaPipe, numba, onnxruntime).

**On Windows, VRAM costs system commit, one-to-one.** Under WDDM every CUDA allocation is backed by
system commit charge so the OS can page it out (measured: allocating 3 GB on the GPU reduced free
system commit by 3.0 GB). The binding constraint is therefore `min(free VRAM, free commit)`, not
8 GB. On this machine the commit limit is 35.3 GB (31 GB RAM + a page file manually capped at 4 GB)
and other applications hold ~30 GB of it, which left **4–6 GB of free commit** during development.
Fixed costs inside one Python process:

| Item | Commit |
|---|---|
| `import numpy` (OpenBLAS pre-allocates per-thread buffers, 16 threads) | 0.49 GB → **0.11 GB** with `OPENBLAS_NUM_THREADS=4` |
| `import torch` (CUDA build) | ~1.0 GB |
| CUDA context | 0.27 GB |
| cuDNN kernel libraries, loaded on the first convolution | ~1.0 GB |

Everything in §6 follows from this.

**Input colour.** All Pixel 8 Pro JPEGs are tagged *Display P3*. They are converted to sRGB through
LittleCMS (relative colorimetric) on load; ignoring the profile would desaturate skin tones.
Outputs carry an embedded sRGB profile.

---

## 2. Pipeline

```
photo ──► analyze ──► matting ──► strict ──────────────► strict-gov photo, sheets, diagnostics
          (CPU)       (GPU)       (CPU)        │
                                               └► studio ──► studio-finish ──► studio-ai photo, …
                                                  (GPU)      (CPU)
all outputs ──► identity (CPU, ONNX Runtime) ──► report.json, summary.{json,md}, index.html
```

Each stage runs **in its own Python process**, **stage-major**: a model is loaded once, applied to
every photo, then the process exits and returns all of its memory. Intermediate results live in
`output/.cache/<photo>/`; stages resume per photo, and the orchestrator restarts a stage process
that dies (a CUDA error caused by the OS running out of commit is sticky for the process).

### 2.1 analyze — landmarks and roll (MediaPipe Face Landmarker)

* **Model:** MediaPipe Face Landmarker (`face_landmarker.task`, float16, Apache-2.0): BlazeFace
  detector + 478-point mesh including the 10 iris points, 52 blendshapes and a head-pose matrix.
  Chosen over InsightFace 2d106 / 3D68 and dlib-68 because it is the only one that gives iris
  centres, expression blendshapes (eye blink, jaw open, smile, gaze) *and* pose from one CPU pass
  (~50 ms).
* **Precision:** the mesh network sees the face through a 256 × 256 crop and its iris estimates
  jitter by 0.1–0.2° between crops. So: (1) a pass on the whole frame; (2) two passes on a
  1024 px crop that is *de-rotated* by the current eye-line estimate (the network is most
  accurate on upright faces); (3) test-time augmentation: 9 upright crops (zoom 0.92/1.0/1.08 ×
  rotation −1.5/0/+1.5°), per-landmark median. The standard error of the eye-line angle over
  the TTA set is reported; it was 0.04–0.07° on the test photos.
* **Iris centre** = mean of the network's centre point and the centroid of the 4 limbus points.
* **Roll** = angle of the line through the two iris centres; it is removed by the framing
  transform (§2.3). The report states it as `applied_rotation_deg_ccw` (positive = the image was
  rotated counter-clockwise).

### 2.2 matting — alpha, crown and framing (BiRefNet)

* **Model:** `ZhengPeng7/BiRefNet_dynamic-matting` (Swin-L, MIT licence), fp16. It was trained at
  *dynamic* resolutions (256–2304 px), so it takes the head-and-shoulders ROI at its true aspect
  ratio instead of a stretched square.

  | Candidate | Verdict |
  |---|---|
  | BiRefNet_HR-matting @ 2048² | best for arbitrary scenes, but 3.95 GB peak; square input wastes pixels on background |
  | **BiRefNet_dynamic-matting** | same family, any aspect; 1.34 / 1.83 / 2.42 GB peak at 1024 / 1280 / 1536 px height |
  | RMBG-2.0 (BiRefNet-based) | CC BY-NC weights |
  | MODNet, U²-Net, rembg models | visibly worse on hair |
  | ViTMatte | needs a trimap (would need a second model to produce it) |
  | SAM 2 | segmentation masks, not alpha mattes |

* **ROI:** in a frame rotated so the eye line is horizontal, ±1.6 D wide and from 2.0 D above to
  2.1 D below the eye line (D = eye-line-to-chin distance), rendered with one anti-aliased warp.
  Nearly every input pixel lands on the subject, so the matte is as dense as the final photo or
  denser. Input height (1024–1536 px) is chosen from the live memory budget; the ROI grows
  automatically if the hair reaches its top or the final canvas would fall outside it.
* **Crown (top of head *including hair*)**: the highest point of the silhouette (`alpha > 0.5`)
  within ±0.55 face widths of the facial axis, after a morphological opening that removes
  isolated flyaway strands, interpolated to sub-pixel precision. Landmark 10 (the top of the
  mesh) sits at the hairline and would cut off the hair volume, so it is not used. If the
  silhouette touches the photo's own top edge, the report flags `crown_in_source: FAIL`.
* **Chin:** mesh landmark 152 (menton), from the TTA median mesh.
* **Clean-up:** the sigmoid's near-0/near-1 tails are snapped (so the background becomes *exactly*
  `#FFFFFF`, not 254), and components not connected to the subject are dropped.

### 2.3 Geometry — one transform, one resampling

All coordinate frames are related by 3 × 3 matrices with pixel centres at integer coordinates. The
final source → canvas transform is

```
M_out = T(cx, top_margin) · S(s) · T(−axis_x, −crown_y) · R(−θ) · T(−eye_mid)
s     = 805.5 px / (chin_y − crown_y)          # 34.1 mm at 600 dpi
cx    = (826 − 1) / 2                          # facial axis on the vertical centre line
top   = 3.25 mm = 76.8 px                      # crown → top edge
```

where `axis_x` is the median x of 24 mid-sagittal landmarks (nasion, nose bridge, philtrum, lips,
chin) after de-rotation. Rotation, scale and crop are composed and the photo is resampled **once**
onto a 2× supersampled working canvas (1652 × 2124). When the transform minifies, the source
window is first box-filtered (`INTER_AREA`) to ~1.25× the output density, so Lanczos never
aliases hair or skin texture. The final 826 × 1062 image is an exact 2 × 2 box average of the
working canvas.

### 2.4 strict — deterministic photometry and delivery

1. **Foreground colour estimation** (Germer et al., multi-level, via `pymatting`): solves
   `I = αF + (1−α)B` for F, so the old wall colour cannot halo through semi-transparent hair.
2. **Photometric normalisation** — all in linear light, all *smooth multiplicative fields*.
   Multiplying by a field that varies over centimetres cannot create, move or remove pores,
   moles, scars or edges, because local contrast ratios are preserved exactly:
   * **White balance:** illuminant voted by achromatic references: a white shirt (weight 3, if
     found below the chin) and the wall (weight 1, only if it is already near-neutral). 80 % of
     the estimated cast is removed (the phone's AWB already did most of the work); gains are
     luminance-normalised and clamped to [0.8, 1.25]; with no reference the AWB is kept. The
     sclera is measured but *not* used: on this test set it read 0.14 log-units redder than a white
     shirt in the same frame (real sclera, caruncle and lid margin are pinkish), and a first version
     that trusted it turned the skin grey in every photo without a white shirt.
   * **Left/right balance:** faces are close to bilaterally symmetric in albedo, so the
     antisymmetric part of the smoothed log-intensity of the skin is lighting (a window on one
     side). It is fitted per channel as `a1·u + a2·u·v + a3·u³` (odd in u = offset from the facial
     axis) and divided out, clamped to ±0.5 EV. Symmetric shading (nose, cheekbones, top light)
     is untouched. Per-channel fitting also removes one-sided colour casts (mixed daylight and
     tungsten light).
   * **Exposure:** changed only if the median skin value falls outside [0.42, 0.80] (sRGB
     encoded), and never so far that the brightest 0.5 % of skin clips.
3. **Compositing** onto `#FFFFFF` in sRGB (the convention matting networks are trained with).
4. **Delivery:** 826 × 1062 JPEG (q95, 4:4:4), 600 dpi in both JFIF and EXIF, embedded sRGB ICC,
   no camera or GPS metadata; two print sheets; a diagnostic sheet; `report.json`.

### 2.5 studio-ai — generative relighting, constrained

**Generator: IC-Light** (lllyasviel, ICLR 2025; `iclight_sd15_fc` offsets on Realistic Vision 5.1,
the base model its weights were trained against). It is an SD 1.5 UNet whose input is widened to
take the VAE latent of the matted subject, and it re-renders the subject under the light described
by a prompt and an initial light-map latent.

| Option considered | Why not (as the core) |
|---|---|
| FLUX.1 Kontext [dev] | 12 B parameters; even at Q4 + offload it exceeds the commit budget here; gated, non-commercial |
| Qwen-Image-Edit | 20 B — does not fit |
| LivePortrait pose/gaze retargeting | 512 px face crop is below the ~800 px head height the 35 × 45 output needs; the test faces are already frontal (\|yaw\|, \|pitch\| < 3° except one) |
| InstantID / PuLID identity synthesis | regenerates the face: maximal identity drift for no biometric benefit |
| SDXL attire inpainting | half the test set already wears a suit; no measurable value for the audit |

**Measured behaviour.** On its own IC-Light re-draws the face (slimmer jaw, different iris colour,
moles removed). In one test it turned a red tie black: colour words in the negative prompt
("oversaturated", "colour cast") steer classifier-free guidance towards desaturation, so the
negative prompt is IC-Light's own default. During development the unmodified render scored
**ArcFace cosine 0.61–0.76** against the photo, *below* every pair of real shots of the same
person. It is therefore used only as a *lighting* oracle, held in memory and in a temporary cache
file that is deleted as soon as studio-finish has used it. The render is never saved or shown:

1. **Render** at 512 × 656 (fp8 weights, 25 DPM++ 2M SDE Karras steps, CFG 2.0, fixed seed).
2. **Register** the render onto the photo: the mesh is re-detected on the render and a thin-plate
   spline through ~160 landmark pairs (anchored to identity at the canvas border) warps it back
   onto the real face geometry, so shading of the generated jaw does not land on the real cheeks.
3. **Measure the lighting field on skin only:** `log(Y_render / Y_photo)`, median-filtered
   (removes the moles and pores the generator dropped), normalised-convolution smoothed at
   3.5 % of the face width and extended over hair, neck and clothes with a 25 % wide version.
   Clothing never contributes, so a recoloured tie cannot leak in.
4. **Symmetrise it** about the facial axis. Biometric photos need even light; the diffusion
   model happily invents a side key light (in testing it put −0.6 / +1.0 EV across the face).
   Keeping only the symmetric part retains its fill and shadow softening.
5. **Colour**, in two parts. Global: half of the render's white-balance shift on skin, capped at ±8 %.
   Local: a zero-mean skin chroma field, median-filtered and smoothed like the lighting field,
   capped at ±30 % and *not* symmetrised. Mixed daylight/tungsten casts differed by 0.2–0.45
   log-units between the two halves of a face in the test set, and this is what evens out the cyan
   blotches the phone left in photos #19–23. Moles, freckles, lips and clothing cannot be
   recoloured, because the field is measured on median-filtered skin only.
6. Multiply the strict-gov foreground by these fields, so every edge and pore is still the camera's.

The before/after and the applied lighting field (as a diverging map) are saved side by side in
`*_studio-ai_comparison.jpg`.

### 2.6 identity — biometric drift audit

* InsightFace `buffalo_l`: SCRFD-10GF detector (5 keypoints) + **ArcFace ResNet-50 (WebFace600K)**,
  on ONNX Runtime **CPU**, so it never competes with the generative stage for VRAM. Faces are
  aligned to the canonical 112 × 112 ArcFace template with a least-squares similarity
  (Umeyama), as the reference implementation does. Note: the buffalo_l weights are licensed for
  non-commercial research use.
* Reported per photo: cosine(source, strict-gov), cosine(source, studio-ai),
  cosine(strict-gov, studio-ai), and the closest *other real shot* of the same person.
* **Calibration baseline:** the pairwise cosine between *different raw photos* of the same person
  in the batch (natural shot-to-shot variation). A drift is reported as smaller or larger than the
  median of that distribution, which is a more meaningful yardstick than an absolute threshold.

---

### 2.7 virtual-studio — physically based relighting (no generative model)

1. **Geometry.** Per-pixel surface normals: the MediaPipe mesh carries real depth (its z is in
   the same units as x/y), so a Delaunay triangulation of its 468 vertices gives area-weighted
   vertex normals, interpolated per pixel. Hair, neck and shoulders get normals of the matte
   "inflated" into a rounded relief (distance transform → circular profile). The two are
   feathered together and lightly smoothed. Neural normal estimators (e.g. Sapiens, CC BY-NC)
   were not needed: soft studio light depends only on large-scale shape.
2. **Old light.** The photo's lighting is fitted as 2nd-order spherical harmonics of the normal
   (9 coefficients; Ramamoorthi & Hanrahan 2001), on skin luminance that is normalised-blurred so
   pores and stubble do not count. On the test photos R² = 0.07–0.55: after strict-gov's
   left/right balancing little structured light is left to explain, which is expected.
3. **New light.** Diffuse irradiance from area lights, each softbox sampled as a disc (Fibonacci
   spiral, 24 samples), plus a white-studio bounce term:

   | Preset | Setup | ID-style |
   |---|---|---|
   | clamshell | 56°-wide key 30° above the lens + reflector 28° below, bounce 0.45 | yes (default) |
   | high-key | 80°-wide frontal key + two side fills at ±60°, bounce 0.7 | yes |
   | loop | key 40° left / 35° up, weak fill, rim light from behind | no |

4. **Apply** new / old irradiance (old light removed 80 % on the face, 40 % on hair/body where the
   fit extrapolates), normalised to unchanged median skin exposure. ID presets may darken by at
   most 0.6 EV (no contouring); loop may move ±1.2 EV. Runs in ~4 s per photo on the CPU.

Self-shadowing is not ray-traced; with large soft sources it is faint. Baked-in hard shadows are
softened only as far as the fit explains them.

### 2.8 studio-flux — FLUX.2 [klein] 4B as the lighting source

FLUX.2 [klein] 4B (Black Forest Labs, Apache-2.0) is a distilled 4-step rectified-flow
transformer with a Qwen3-4B text encoder that edits a reference image from an instruction. The
model card asks for ~13 GB of VRAM; on the 8 GB card:

* the 8 GB text encoder runs once on the CPU and the prompt embedding is cached on disk;
* the transformer is stored on the GPU in float8 e4m3 (3.8 GB) and computed in bfloat16. Its bf16
  weights are zero-copy views of the read-only file mapping, cast to fp8 and moved to the GPU;
  `from_pretrained` would first copy all 7.75 GB into RAM. That ~13 GB load peak crashed the stage
  natively when only 10 GB was free;
* the VAE runs in bfloat16 on the GPU.

Measured with the streamed loader: 42 s to load (6.7 GB process commit), 10.6–14 s per
768 × 992 render, 5.56 GB peak VRAM, 8.7 GB process commit at peak. The stage waits for 9.2 GB
of free commit (measured peak plus margin) and otherwise skips itself. The render is used
exactly like IC-Light's (registered, skin-only, symmetrised lighting and colour field), except that
global exposure (±0.5 EV) and local shaping (±0.75 EV) are now limited separately. FLUX renders
about 1 EV brighter overall, and a single clamp flattened the shading into a uniform lift.
Its raw render is far more faithful than IC-Light's: ArcFace cosine to the strict-gov photo of
0.82–0.93 on test photos, against ~0.69. It still alters hair tone, lip colour and fabric.

### 2.9 Edges outside the original photo: matting padding and generative fill

When the 35×45 framing extends past the original photo's border, two things are needed:

* **Matting.** The ROI used to be padded by edge replication. The striped bands this produces
  made BiRefNet cut a jacket shoulder off along a diagonal (a white wedge in the bottom-left corner
  of two telephoto shots). The padding is now the wall colour (median of the photo border), and the
  matte outside the photo is continued by copying the value of the nearest pixel 3 px inside it
  (`distanceTransformWithLabels`). A diffusion-style inpaint of the matte only reached 0.58 opacity
  there.
* **Fill (`fill` stage, FLUX.2 [klein] inpainting).** Mask = outside the photo ∧ within 40 px of
  the extended silhouette ∧ not within ~20 px of the face oval. The matte of the generated strip
  is then read from the render itself: anything clearly darker than the requested white
  background is subject. The nearest-pixel extension had smeared a sloping shoulder edge into a
  grey band. The prompt names no garment; a first version said "suit jacket, shirt, tie" and
  painted a collar and tie onto a T-shirt. Strips under ~100 output px are skipped. The
  streaks under the mask are replaced by a smooth extrapolation first, because FLUX also conditions
  on the input image and otherwise copies their pattern. The `image_reference` is the best-framed
  photo of the same session (no truncation, nearest capture time), so the jacket, shirt and tie
  continue with the right fabric. The result is feather-blended into the subject layer
  (`subject_filled.npz`) used by virtual-studio, studio-ai and studio-flux. strict-gov stays
  unfilled and keeps its `source_coverage` warning. The generated area is reported in mm² of print.

### 2.10 Print manager

`govkadr gui` runs a standard-library HTTP server on 127.0.0.1 with one page. It reads the
per-photo reports, offers every mode's photo with its verdict and identity score, and composes a
4 × 2 sheet from any mix of photos (`sheet.build` accepts a list of tiles). It writes 600 dpi sheets
plus a JSON manifest to `output/prints/`. File serving is restricted to images and JSON under
`output/`, and path traversal is rejected.

## 3. Memory strategy (RTX 5060 8 GB, ~5 GB free commit)

| Measure | Effect |
|---|---|
| One process per stage, stage-major | peak = largest single model; memory fully returned between stages |
| Orchestrator imports only stdlib | holds no numpy / OpenCV / MediaPipe / torch while the GPU stages run |
| `OPENBLAS_NUM_THREADS=4` | −0.75 GB per process (numpy and scipy each bundle OpenBLAS) |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | freed blocks (and their commit) go back to the driver promptly |
| BiRefNet fp16, input height from live budget | 1.34–2.42 GB instead of 3.95 GB (HR @ 2048²) |
| IC-Light UNet stored as **float8 e4m3**, computed in fp16 (diffusers layerwise casting); `conv_in` kept fp16 because its extra channels carry the subject latent | 1.6 → 0.8 GB weights, visually identical render |
| Weights streamed from a read-only memory map straight to the GPU (custom safetensors reader) | no full CPU copy; also sidesteps a safetensors 0.8 + torch 2.14 access violation on files without `format: pt` metadata |
| SDPA attention (`AttnProcessor2_0`) | the classic processor needed a 1.8 GB attention matrix |
| Classifier-free guidance as two sequential batch-1 passes | half the UNet activations |
| fp16 VAE (bf16 convolutions have no precompiled cuDNN kernels for sm_120: 16 s JIT) | first call 1.3 s instead of 16.6 s |
| Prompt embeddings cached on disk | the GPU process never loads CLIP / transformers after the first run |
| ArcFace / SCRFD on CPU | zero VRAM |
| Pre-flight wait (≤ 5 min) for ≥ 5.2 GB free commit; otherwise studio-ai is **skipped** (verdict `SKIPPED`) unless `--allow-low-memory` | never pushes Windows to its commit limit, where *other* applications' allocations fail |
| Automatic stage restart, per-photo resume | a sticky CUDA error in one stage process costs one restart, not the batch |

Peaks: matting ≈ 1.3–2.4 GB VRAM; IC-Light 1.75 GB VRAM and ~4.6 GB process commit.
During the reference run, other applications left only ~4.2 GB of free commit; the IC-Light
stage (run before the skip safeguard existed) drove it to 0.05 GB and needed one in-process retry.
**Recommendation:** raise the page-file maximum (currently 4 GB, e.g. to "system managed") or
close large browser sessions before running studio-ai; that also allows a larger matting ROI.

---

## 4. Verification

Compliance is re-measured on the delivered file, not taken from what the renderer aimed for:
the mesh is re-detected on the 826 × 1062 JPEG, the crown is re-extracted from the output matte,
and the file is re-read from disk.

| Check | Rule |
|---|---|
| pixel dimensions | 826 × 1062 |
| DPI metadata | 600 in JFIF **and** EXIF |
| file size | ≤ 2.5 MB (gov.pl upload) |
| online minimum resolution | ID ≥ 492 × 633, passport ≥ 768 × 1004 |
| colour profile | embedded sRGB |
| head height (chin → crown incl. hair) | 31.5–36.0 mm (target 34.1 mm / 805 px) |
| top margin | 3.0–3.5 mm (soft) |
| horizontal centring | facial axis within ±2.1 mm of the centre line |
| eye line level | residual roll ≤ 0.5° |
| background | ≥ 99.5 % of background pixels exactly `#FFFFFF`, ≥ 99.9 % ≥ 250, clear top edge |
| crown in source | the head is not cut by the photo's top edge |
| source coverage | no part of the subject had to be filled from beyond the photo edge (soft) |
| single face, eyes open, mouth closed, neutral expression, gaze, \|yaw\| and \|pitch\| ≤ 5°, eye line at 50–70 % of the height | soft (WARN), from blendshapes and the pose matrix |

## 5. Print sheets

`*_sheet_10x15cm.jpg` is exactly what was specified: 3543 × 2362 px = 150 × 100 mm at 600 dpi,
4 × 2 tiles pasted pixel-for-pixel, with 1 px grey (200) crop marks drawn only in the gutters,
kept 8 px (0.34 mm) away from the photos. Most "10 × 15" kiosk paper is in fact 4 × 6 in
(152.4 × 101.6 mm); printing a 150 × 100 mm file "fit to page" there scales it by **+1.6 %**
(35.6 × 45.7 mm per photo). `*_sheet_4x6in.jpg` (3600 × 2400 px) is laid out for that paper, so
the photos come out at exactly 35 × 45 mm. Use the one that matches the paper and print at
100 % / "no cropping".

## 6. Results on the test set (32 Pixel 8 Pro photos, both modes)

Re-measured on the delivered files (`output/summary.md` has the per-photo table):

| Metric | Result |
|---|---|
| strict-gov verdicts | 27 PASS, 5 PASS_WITH_WARNINGS, 0 FAIL |
| warnings | 3 × camera below eye level (pitch −5.7° to −7.6°), 2 × shoulders cut by the photo's bottom edge (12 and 19 rows edge-filled) |
| head height (chin → crown incl. hair) | 34.18 ± 0.05 mm, range 34.09–34.29 mm (target 34.1, legal 31.5–36.0) |
| top margin | 3.23 ± 0.05 mm (3.10–3.30) |
| horizontal offset of the facial axis | 0.003 ± 0.021 mm, max 0.08 mm (tolerance ±2.1) |
| roll corrected / residual | −4.15° … +2.27° corrected; residual 0.016 ± 0.062°, max 0.20° |
| background | 99.54–99.74 % of background pixels exactly `#FFFFFF` after JPEG; 100 % ≥ 250 |
| file size | 158–260 KB, 600 dpi JFIF + EXIF, sRGB |
| ArcFace cos(source, strict-gov) | **0.988** ± 0.003 (min 0.981) |
| ArcFace cos(source, studio-ai) | **0.897** ± 0.012 (0.869–0.917) |
| ArcFace cos(source, unmodified IC-Light render; development measurement, no longer kept) | 0.689 ± 0.037 (0.610–0.760) |
| shot-to-shot baseline (496 pairs of different real photos) | median 0.830, min 0.714, max 0.983 |

Reading the audit: strict-gov is effectively identity-neutral. studio-ai drifts less than two real
photos of the same person usually differ (0.897 > median 0.830). The unmodified generative render drifted
*more* than any pair of real photos (0.689 < min 0.714), which is why it is used only as a
lighting oracle and then discarded.

**All four modes** (same 32 photos; ArcFace cosine of each delivered photo to its source, and of
each generative mode's raw render to the strict-gov photo, recorded as a number only):

| Mode | Verdicts | cos(source, delivered) | raw render cos |
|---|---|---|---|
| strict-gov | 27 PASS, 5 WARN | 0.988 (0.981–0.992) | – |
| virtual-studio (clamshell) | 27 PASS, 5 WARN | 0.906 (0.870–0.942) | – (no generative model) |
| studio-ai (IC-Light) | 26 PASS, 5 WARN, 1 FAIL | 0.891 (0.866–0.922) | 0.694 (0.624–0.758) |
| studio-flux (FLUX.2 klein) | 27 PASS, 5 WARN | **0.949** (0.921–0.964) | **0.910** (0.817–0.950) |

The one FAIL is a background at 99.50 % exactly `#FFFFFF`, just under this project's own 99.5 %
threshold (JPEG ringing beside brightened hair); 99.999 % of it is ≥ 250.

The virtual-studio row is a useful calibration. It generates no pixels, yet its score drops to
0.906 simply because the light changed, so ArcFace reacts to lighting as well as identity. Part
of every relit mode's drop is therefore lighting, not a changed face. The raw-render column is
where real redrawing shows: FLUX.2 [klein] stays close to the person (0.91), IC-Light does not (0.69).

Timing (reference run, all 32 photos): analyze ~1.5 s, matting ~0.4 s (after a ~25 s model load),
strict ~7 s, IC-Light ~7 s, studio-finish ~4 s, identity ~3 s per photo; ~21 min end to end,
including a 5 min wait for memory headroom.

The best sources were the 5× telephoto (110 mm-equivalent) shots at 1.5–2 m. The 24 mm
close-ups and 21 mm selfies pass the geometric checks but show wide-angle perspective
distortion (enlarged nose and jaw, hidden ears), which no compliant pipeline may correct.

## 7. Limitations

* The chin is the mesh's landmark 152. A beard or a strong double chin can move the true menton by
  a few pixels (≈ 0.1–0.3 mm), which is well inside the 4.5 mm head-height tolerance.
* White balance assumes the shirt, sclera or wall is achromatic; a strongly coloured wall is
  ignored automatically, but a cream shirt would bias the result slightly warm.
* `studio-ai` is generative by design and is **not** valid for official submission. Its drift is
  quantified per photo in `report.json`.
* The identity baseline assumes every photo in a batch shows the same person (pairs below cosine
  0.35 are excluded as impostor pairs).
