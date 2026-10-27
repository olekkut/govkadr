"""studio-ai mode: IC-Light neural relighting + frequency-separated texture transfer.

IC-Light (lllyasviel, ICLR 2025) is a Stable Diffusion 1.5 UNet whose input is widened to take a
VAE latent of the matted subject; it re-renders the subject under the lighting described by the
prompt and an initial "light map" latent. On its own it also re-draws the face (jaw, eyes, moles:
ArcFace cosine ~0.65 to the photo), so its render is used only as a *lighting oracle*:

    render  -> register onto the photo's face mesh (thin-plate spline)
            -> log(Y_render / Y_photo) on skin, median-filtered, smoothed, extended off-face
            -> symmetrised about the facial axis, clamped to +-0.75 EV, + half the colour shift
    studio = photo * field

Every edge and pore in the result is the camera's; the raw render is kept for the drift audit.

Memory plan (Windows backs VRAM with system commit): weights stream from a read-only memory map
straight to the GPU, the UNet is stored in float8 and computed in fp16, CFG runs as two batch-1
passes, prompt embeddings are cached on disk, and this module runs in its own process. Peak is
~1.75 GB of VRAM at 512x656.
"""

from __future__ import annotations

import gc
import time

import cv2
import numpy as np
import torch

from . import weights

PROMPT = "portrait photo of a person, soft frontal softbox light, bright studio, best quality"
# IC-Light's own default negative prompt. Adding colour words here ("oversaturated", "colour cast")
# makes classifier-free guidance desaturate clothing - a red tie came out black in testing.
NEGATIVE = "lowres, bad anatomy, bad hands, cropped, worst quality"


def softbox_light_map(w: int, h: int) -> np.ndarray:
    """Initial latent image: a large bright source above the lens falling off to the sides.

    IC-Light reads light direction from the brightness layout of the initial image (its demo uses
    linear gradients for left/right/top light). This map models a big frontal softbox slightly
    above eye level plus ambient fill from a white studio.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = (xx / (w - 1) - 0.5) / 0.55, (yy / (h - 1) - 0.28) / 0.75
    lum = 150 + 105 * np.exp(-(u**2 + v**2))
    return np.repeat(lum[..., None], 3, axis=2).clip(0, 255).astype(np.uint8)


def prompt_embeddings(base) -> tuple[torch.Tensor, torch.Tensor]:
    """CLIP embeddings of the fixed prompt pair, computed once on the CPU and cached on disk.

    Keeps the text encoder (and the transformers import) out of the GPU process on later runs.
    """
    import hashlib

    key = hashlib.sha1(f"{PROMPT}\0{NEGATIVE}\0{weights.SD15_REALISTIC_VISION.revision}".encode()).hexdigest()[:12]
    path = weights.MODELS_DIR / "cache" / f"iclight_prompts_{key}.pt"
    if path.exists():
        cond, uncond = torch.load(path)
        return cond, uncond
    from transformers import CLIPTextModel, CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=torch.float32).eval()
    with torch.inference_mode():
        cond, uncond = (
            te(tok(t, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids).last_hidden_state.half()
            for t in (PROMPT, NEGATIVE)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save((cond, uncond), path)
    del te
    gc.collect()
    return cond, uncond


class StudioRelighter:
    def __init__(self, device: str = "cuda", seed: int = 20261027, fp8_weights: bool = True):
        from accelerate import init_empty_weights
        from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, UNet2DConditionModel
        self.device = torch.device(device)
        self.seed = seed
        base = weights.fetch_hf(weights.SD15_REALISTIC_VISION)
        offsets = weights.fetch_hf(weights.IC_LIGHT) / "iclight_sd15_fc.safetensors"

        with init_empty_weights():
            unet = UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(base, subfolder="unet"))
        # Tensor-by-tensor from the memory-mapped file straight to the GPU: no full CPU copy.
        state = {k: t.to(self.device, torch.float16) for k, t in weights.read_safetensors(base / "unet" / "diffusion_pytorch_model.safetensors")}
        unet.load_state_dict(state, assign=True)
        del state
        with torch.no_grad():
            # Widen conv_in to 8 channels: 4 noisy latent + 4 subject latent (zero-initialised extra
            # weights, as in IC-Light), then add the learned offsets parameter by parameter.
            old = unet.conv_in
            new = torch.nn.Conv2d(8, old.out_channels, old.kernel_size, old.stride, old.padding, device=self.device, dtype=torch.float16)
            new.weight.zero_()
            new.weight[:, :4].copy_(old.weight)
            new.bias = old.bias
            unet.conv_in = new
            params = dict(unet.named_parameters())
            params.update(dict(unet.named_buffers()))
            for k, t in weights.read_safetensors(offsets):
                params[k].add_(t.to(self.device, params[k].dtype))
        from diffusers.models.attention_processor import AttnProcessor2_0

        # Memory-efficient SDPA attention (never materialises the N x N attention matrix).
        self.unet = unet.eval()
        self.unet.set_attn_processor(AttnProcessor2_0())
        if fp8_weights:
            # Weights stored as float8 e4m3 and up-cast per layer at compute time: 1.6 -> 0.8 GB.
            # Only the low-frequency lighting of this render is used, so fp8 rounding is invisible.
            # conv_in stays fp16: its extra input channels carry the subject latent (the conditioning).
            self.unet.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn,
                compute_dtype=torch.float16,
                skip_modules_pattern=("conv_in", "conv_out", "norm", "time_embed"),
            )
            torch.cuda.empty_cache()
        # fp16 VAE: bf16 convolutions have no precompiled cuDNN kernels for sm_120 (16 s JIT on
        # first use); _decode falls back to fp32 if fp16 ever overflows.
        self.vae = AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=torch.float16).to(self.device).eval()
        self.vae.set_attn_processor(AttnProcessor2_0())
        self.cond, self.uncond = (e.to(self.device) for e in prompt_embeddings(base))
        # IC-Light's own sampler settings (note: default *linear* beta schedule, as in its demo).
        self.scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True, steps_offset=1
        )

    def close(self) -> None:
        del self.unet, self.vae, self.cond, self.uncond
        gc.collect()
        torch.cuda.empty_cache()

    def _to_latent(self, rgb_u8: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(rgb_u8).float().div(127.0).sub(1.0).permute(2, 0, 1)[None]  # IC-Light normalisation
        lat = self.vae.encode(x.to(self.device, self.vae.dtype)).latent_dist.mode()
        return (lat * self.vae.config.scaling_factor).to(torch.float16)

    def _decode(self, lat: torch.Tensor) -> np.ndarray:
        x = self.vae.decode(lat.to(self.vae.dtype) / self.vae.config.scaling_factor).sample
        if not torch.isfinite(x).all():
            self.vae.to(torch.float32)
            x = self.vae.decode(lat.float() / self.vae.config.scaling_factor).sample
            self.vae.to(torch.float16)
        return (x[0].float().permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5).clip(0, 255).astype(np.uint8)

    def _img2img(self, init: torch.Tensor, concat: torch.Tensor, strength: float, steps: int, cfg: float, gen: torch.Generator) -> torch.Tensor:
        sch = self.scheduler
        n = int(round(steps / strength))
        sch.set_timesteps(n, device=self.device)
        t_start = max(n - min(int(n * strength), n), 0)
        timesteps = sch.timesteps[t_start * sch.order :]
        if hasattr(sch, "set_begin_index"):
            sch.set_begin_index(t_start * sch.order)
        noise = torch.randn(init.shape, generator=gen, device=self.device, dtype=torch.float32).to(init.dtype)
        lat = sch.add_noise(init, noise, timesteps[:1])
        for t in timesteps:
            x = torch.cat([sch.scale_model_input(lat, t), concat], dim=1)
            # Conditional and unconditional passes run one after another (batch 1) instead of as
            # a batch of 2: half the activation memory for ~10 % more time.
            eu = self.unet(x, t, encoder_hidden_states=self.uncond).sample
            ec = self.unet(x, t, encoder_hidden_states=self.cond).sample
            lat = sch.step(eu + cfg * (ec - eu), t, lat, generator=gen).prev_sample
        return lat

    @torch.inference_mode()
    def relight(self, subject_on_grey: np.ndarray, size: tuple[int, int] = (512, 656), steps: int = 25, cfg: float = 2.0, denoise: float = 0.9) -> tuple[np.ndarray, dict]:
        """Relit render of a subject composited on 50 % grey (IC-Light's input convention)."""
        w, h = size
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        fg = cv2.resize(subject_on_grey, (w, h), interpolation=cv2.INTER_AREA)
        gen = torch.Generator(device=self.device).manual_seed(self.seed)
        concat = self._to_latent(fg)
        init = self._to_latent(softbox_light_map(w, h))
        lat = self._img2img(init, concat, denoise, steps, cfg, gen)
        out = self._decode(lat)
        info = {
            "model": "IC-Light fc (SD1.5, Realistic Vision 5.1 base)",
            "size": [w, h],
            "steps": steps,
            "cfg": cfg,
            "denoise": denoise,
            "sampler": "DPM++ 2M SDE Karras",
            "seed": self.seed,
            "seconds": round(time.perf_counter() - t0, 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
        return out, info


def register_render(render: np.ndarray, render_pts: np.ndarray, ref_pts: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Warp the generated render onto the real face geometry (thin-plate spline on the meshes).

    The diffusion model re-draws the face (jaw, eye shape); before its shading can be transferred
    it must be registered to the photo, or the lighting of a slimmer generated jaw would land on
    the real cheeks. Canvas-border anchors pin the warp to identity far from the face.
    """
    from scipy.interpolate import RBFInterpolator

    w, h = size
    ids = np.r_[0:468:3, 468, 473]
    src = ref_pts[ids]
    disp = render_pts[ids] - src
    border = np.array([[x, y] for x in np.linspace(0, w - 1, 6) for y in (0, h - 1)] + [[x, y] for y in np.linspace(0, h - 1, 7)[1:-1] for x in (0, w - 1)], float)
    src = np.vstack([src, border])
    disp = np.vstack([disp, np.zeros_like(border)])
    rbf = RBFInterpolator(src, disp, kernel="thin_plate_spline", smoothing=1.0)
    step = 16
    gy, gx = np.mgrid[0:h:step, 0:w:step]
    coarse = rbf(np.column_stack([gx.ravel(), gy.ravel()]).astype(float)).reshape(gy.shape[0], gy.shape[1], 2)
    dx = cv2.resize(coarse[..., 0].astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(coarse[..., 1].astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(render, xs + dx, ys + dy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _nc_blur(values: np.ndarray, weight: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Normalised convolution: weighted Gaussian average that fills holes from valid neighbours."""
    num = cv2.GaussianBlur(values * weight, (0, 0), sigma)
    den = cv2.GaussianBlur(weight, (0, 0), sigma)
    return num / np.maximum(den, 1e-6), den


def transfer_lighting(
    orig_lin: np.ndarray,
    render_lin: np.ndarray,
    skin: np.ndarray,
    face_w: float,
    axis_x: float,
    max_ev: float = 0.75,
    chroma_strength: float = 0.5,
    chroma_limit: float = 0.08,
    local_chroma_limit: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Relighting field measured on skin only, then extended smoothly over hair, neck and clothes.

    * luminance: log(Y_render / Y_photo) on skin, median-filtered (removes moles/pores that the
      generator dropped), normalised-convolution smoothed at ~3.5 % of the face width, with a wide
      (~25 %) version filling everything away from the face;
    * symmetry: the field is averaged with its mirror image about the facial axis. Biometric photos
      need even illumination, and the diffusion model happily invents a side key light; keeping
      only the symmetric part retains its fill and shadow softening but no left/right modelling;
    * colour: the render's global white balance (median chroma ratio over skin) at
      ``chroma_strength``, capped at +-``chroma_limit``, plus a local zero-mean skin chroma field
      (capped at +-``local_chroma_limit``) that evens out mixed-light colour blotches.
    The photo is multiplied by this field, so every edge and pore in the result is the camera's.
    """
    luma = np.array([0.2126, 0.7152, 0.0722], np.float32)
    q = 4  # work at quarter resolution: the field is smooth by construction
    h, w = skin.shape
    small = (w // q, h // q)
    yo = cv2.resize(orig_lin @ luma, small, interpolation=cv2.INTER_AREA)
    yr = cv2.resize(render_lin @ luma, small, interpolation=cv2.INTER_AREA)
    m = cv2.resize(skin.astype(np.float32), small, interpolation=cv2.INTER_AREA) > 0.99
    lr = np.log(np.maximum(yr, 1e-4)) - np.log(np.maximum(yo, 1e-4))
    lr = cv2.medianBlur(np.clip(lr, -3, 3).astype(np.float32), 5)
    wgt = m.astype(np.float32)
    fine, den_f = _nc_blur(lr, wgt, 0.035 * face_w / q)
    wide, den_w = _nc_blur(lr, wgt, 0.25 * face_w / q)
    glob = float(np.median(lr[m])) if m.any() else 0.0
    wide = np.where(den_w > 1e-3, wide, glob)
    t = np.clip(den_f / 0.35, 0.0, 1.0)
    field = (t * fine + (1 - t) * wide).astype(np.float32)
    ax = (axis_x + 0.5) / q - 0.5
    map_x = np.repeat((2 * ax - np.arange(small[0], dtype=np.float32))[None, :], small[1], axis=0)
    map_y = np.repeat(np.arange(small[1], dtype=np.float32)[:, None], small[0], axis=1)
    mirrored = cv2.remap(field, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    asym = float(np.abs(0.5 * (field - mirrored))[m].mean() / np.log(2)) if m.any() else 0.0
    field = 0.5 * (field + mirrored)
    field = np.clip(field, -max_ev * np.log(2), max_ev * np.log(2))
    gain_y = np.exp(cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC))

    # Colour, split in two: the render's global white balance (gentle), and a local zero-mean skin
    # chroma field that evens out colour blotches - e.g. the cyan patches mixed daylight/tungsten
    # light leaves on one side of a face - but, measured on median-filtered skin like the lighting
    # field, cannot recolour moles, freckles, lips or clothing.
    def log_chroma(img):
        small_img = cv2.resize(img, small, interpolation=cv2.INTER_AREA)
        return np.log(np.maximum(small_img, 1e-4)) - np.log(np.maximum(small_img @ luma, 1e-4))[..., None]

    cr = log_chroma(render_lin) - log_chroma(orig_lin)
    glob_c = np.median(cr[m], axis=0) if m.any() else np.zeros(3, np.float32)
    local = np.empty_like(cr)
    for c in range(3):
        ch = cv2.medianBlur(np.clip(cr[..., c] - glob_c[c], -1, 1).astype(np.float32), 5)
        f_c, _ = _nc_blur(ch, wgt, 0.035 * face_w / q)
        w_c, d_c = _nc_blur(ch, wgt, 0.25 * face_w / q)
        local[..., c] = t * f_c + (1 - t) * np.where(d_c > 1e-3, w_c, 0.0)
    # Mixed daylight/tungsten casts differ by 0.2-0.45 log-units between the two halves of a face
    # (measured on the test set), so the local field may follow the render up to +-30 %.
    lim_g, lim_l = np.log1p(chroma_limit), np.log1p(local_chroma_limit)
    log_gain_c = np.clip(chroma_strength * glob_c, -lim_g, lim_g) + np.clip(local, -lim_l, lim_l)
    gain_c = np.exp(log_gain_c)
    gain_c /= np.maximum(gain_c @ luma, 1e-4)[..., None]  # chroma only: leave luminance to gain_y
    gain_c = cv2.resize(gain_c.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    gain = gain_y[..., None] * gain_c
    global_gain = np.exp(np.clip(chroma_strength * glob_c, -lim_g, lim_g))
    global_gain /= float(luma @ global_gain)
    info = {
        "chroma_gain_rgb": [round(float(c), 4) for c in global_gain],
        "render_chroma_ratio_rgb": [round(float(c), 4) for c in np.exp(glob_c)],
        "local_chroma_p95_abs_log": [round(float(v), 4) for v in np.percentile(np.abs(np.clip(local[m], -lim_l, lim_l)), 95, axis=0)] if m.any() else None,
        "global_luma_ev": round(glob / np.log(2), 3),
        "removed_left_right_component_ev": round(asym, 3),
    }
    return orig_lin * gain, gain, info
