"""Trimap-free alpha matting (BiRefNet) and colour decontamination.

BiRefNet_dynamic-matting (Swin-L, MIT licence) was trained at dynamic resolutions (256-2304 px),
so it accepts the head-and-shoulders ROI at its true aspect ratio: every input pixel lands on the
subject's region and the matte is as dense as the final 826x1062 photo or denser, keeping
individual hair strands. Its input height is chosen from the live memory budget (see
``resources``). Foreground colours are then re-estimated (multi-level solver of Germer et al.)
so the former wall colour does not halo through semi-transparent hair on the white background.
"""

from __future__ import annotations

import gc
import time

import cv2
import numpy as np

from . import weights

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Measured on the RTX 5060 Laptop (fp16): model + activations as a function of input megapixels.
_GB_FIXED, _GB_PER_MP = 0.95, 1.30


def height_for_budget(budget_gb: float, aspect_w_over_h: float) -> int:
    """Largest multiple-of-128 input height (1024-1536) whose estimated footprint fits."""
    for h in (1536, 1408, 1280, 1152, 1024):
        mp = h * h * aspect_w_over_h / 1e6
        if _GB_FIXED + _GB_PER_MP * mp <= budget_gb:
            return h
    return 1024


class Matting:
    """Alpha matting network with an OOM ladder: it shrinks the input instead of failing."""

    def __init__(self, asset: weights.HfAsset = weights.BIREFNET_DYNAMIC_MATTING, device: str = "cuda"):
        import torch
        from transformers import AutoModelForImageSegmentation

        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        torch.set_float32_matmul_precision("high")
        path = weights.fetch_hf(asset)
        self.model = AutoModelForImageSegmentation.from_pretrained(str(path), trust_remote_code=True, torch_dtype=self.dtype)
        self.model.to(self.device).eval()
        gc.collect()
        self.name = asset.repo_id
        self.last: dict = {}

    def close(self) -> None:
        import torch

        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """Alpha matte (float32 in [0,1]) for an sRGB uint8 image whose sides are multiples of 32."""
        import torch

        h, w = rgb.shape[:2]
        scale = 1.0
        while True:
            sw, sh = int(round(w * scale / 32)) * 32, int(round(h * scale / 32)) * 32
            x = rgb if (sw, sh) == (w, h) else cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
            try:
                t0 = time.perf_counter()
                alpha = self._run(x)
                self.last = {"input_px": [sw, sh], "seconds": round(time.perf_counter() - t0, 3)}
                break
            except (torch.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower() or sh <= 768:
                    raise
                torch.cuda.empty_cache()
                scale *= 0.85
                print(f"[matting] CUDA OOM at {sw}x{sh}, retrying smaller")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if alpha.shape != (h, w):
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_CUBIC)
        return np.clip(alpha, 0.0, 1.0)

    def _run(self, rgb: np.ndarray) -> np.ndarray:
        import torch

        x = (rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(self.device, self.dtype)
        with torch.inference_mode():
            pred = self.model(t)[-1].float().sigmoid()
        if self.device == "cuda":
            torch.cuda.synchronize()
        return pred[0, 0].cpu().numpy()


def clean_alpha(alpha: np.ndarray, lo: float = 0.02, hi: float = 0.98) -> np.ndarray:
    """Snap the sigmoid's near-0/near-1 tails and drop blobs not attached to the subject.

    Sigmoid mattes are never exactly 0, which would leave a faint grey veil instead of #FFFFFF;
    stray detections (a towel, a door frame) are removed by keeping only the largest component.
    """
    a = np.clip((alpha - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((a > 0).astype(np.uint8), connectivity=8)
    if n > 2:
        main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        a[(labels != main) & (labels != 0)] = 0.0
    return a


def estimate_foreground(img: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Solve for the un-premultiplied foreground colour F in I = aF + (1-a)B (sRGB floats)."""
    from pymatting import estimate_foreground_ml

    fg = estimate_foreground_ml(img.astype(np.float64), alpha.astype(np.float64), regularization=1e-5, n_small_iterations=10, n_big_iterations=2)
    return np.clip(fg, 0.0, 1.0).astype(np.float32)
