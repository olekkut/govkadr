"""Model weight registry: every network the pipeline uses, pinned to an exact upstream revision.

Weights live under ``<repo>/models`` (git-ignored) so the project is self-contained and can be
wiped with a single ``rm -rf models``.  Hugging Face assets are pinned by commit SHA, plain URL
assets by SHA-256 (verified after every download).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("GOVKADR_MODELS", REPO_ROOT / "models"))
HF_CACHE = MODELS_DIR / "hf"


@dataclass(frozen=True)
class UrlAsset:
    url: str
    sha256: str
    filename: str
    extract: tuple[str, ...] = ()  # members to pull out of a zip archive


@dataclass(frozen=True)
class HfAsset:
    repo_id: str
    revision: str
    allow_patterns: tuple[str, ...] = field(default_factory=tuple)


FACE_LANDMARKER = UrlAsset(
    url="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    sha256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    filename="face_landmarker.task",
)
INSIGHTFACE_BUFFALO_L = UrlAsset(
    url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    sha256="80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
    filename="buffalo_l.zip",
    extract=("det_10g.onnx", "w600k_r50.onnx"),
)
BIREFNET_HR_MATTING = HfAsset("ZhengPeng7/BiRefNet_HR-matting", "5d6b6f8adcb5b417c871b1d84ceaae9871355b7f")
BIREFNET_DYNAMIC_MATTING = HfAsset("ZhengPeng7/BiRefNet_dynamic-matting", "074df545be87034e74a96bf71566ecbbc4c15f0a")
BIREFNET_MATTING = HfAsset("ZhengPeng7/BiRefNet-matting", "eccde0a8cbdce7ac5fecfeb06340fe7b949e85d9")
IC_LIGHT = HfAsset("lllyasviel/ic-light", "9cad1878695f546a7fb9eaca14e2a89131ba5ffe", ("iclight_sd15_fc.safetensors",))
# IC-Light's offsets were trained against Realistic Vision 5.1; the official demo loads this exact repo.
SD15_REALISTIC_VISION = HfAsset(
    "stablediffusionapi/realistic-vision-v51",
    "19e3643d7d963c156d01537188ec08f0b79a514a",
    (
        "model_index.json",
        "scheduler/*",
        "tokenizer/*",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ),
)


# FLUX.2 [klein] 4B (Apache-2.0): 4B rectified-flow transformer + Qwen3-4B text encoder, with
# reference-image editing. The single-file checkpoint duplicates transformer/ and is skipped.
FLUX2_KLEIN_4B = HfAsset(
    "black-forest-labs/FLUX.2-klein-4B",
    "e7b7dc27f91deacad38e78976d1f2b499d76a294",
    ("model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/*", "transformer/*", "vae/*"),
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_url(asset: UrlAsset) -> Path:
    """Download (once) and verify a URL asset. Returns the file, or the extraction dir for zips."""
    target = MODELS_DIR / asset.filename
    out_dir = MODELS_DIR / Path(asset.filename).stem
    if asset.extract and all((out_dir / m).exists() for m in asset.extract):
        return out_dir
    if not asset.extract and target.exists():
        return target
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        tmp = target.with_suffix(target.suffix + ".part")
        print(f"[weights] downloading {asset.url}")
        with urllib.request.urlopen(asset.url) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        tmp.replace(target)
    digest = _sha256(target)
    if digest != asset.sha256:
        target.unlink()
        raise RuntimeError(f"SHA-256 mismatch for {asset.url}: got {digest}, expected {asset.sha256}")
    if not asset.extract:
        return target
    out_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(target) as z:
        for member in z.namelist():
            name = Path(member).name
            if name in asset.extract:
                (out_dir / name).write_bytes(z.read(member))
    target.unlink()
    return out_dir


def fetch_hf(asset: HfAsset) -> Path:
    """Snapshot a pinned Hugging Face revision into ``models/hf/<org>--<name>`` and return it.

    A plain ``local_dir`` (no blob symlinks) because creating symlinks on Windows needs
    Developer Mode or admin rights.
    """
    from huggingface_hub import snapshot_download

    local_dir = HF_CACHE / asset.repo_id.replace("/", "--")
    return Path(
        snapshot_download(
            repo_id=asset.repo_id,
            revision=asset.revision,
            allow_patterns=list(asset.allow_patterns) or None,
            local_dir=local_dir,
        )
    )


def read_safetensors(path: Path):
    """Yield (name, CPU tensor) one at a time from a .safetensors file.

    A minimal reader (8-byte header length, JSON header, raw little-endian buffers) used instead of
    ``safetensors.safe_open``: safetensors 0.8 + torch 2.14 hit an access violation on files
    without ``format: pt`` metadata (IC-Light's). Streaming keeps peak CPU memory at one tensor.
    """
    import json
    import struct
    import warnings

    import numpy as np
    import torch

    dtypes = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32, "I64": torch.int64, "I32": torch.int32, "U8": torch.uint8}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    base = 8 + n
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        # Zero-copy view of the read-only mapping: file-backed pages cost no commit charge.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # "array is not writable": we only read
            yield name, torch.from_numpy(mm[base + start : base + end]).view(dtypes[info["dtype"]]).reshape(info["shape"])
    del mm


def fetch_all(include_generative: bool = True, include_flux: bool = False) -> None:
    fetch_url(FACE_LANDMARKER)
    fetch_url(INSIGHTFACE_BUFFALO_L)
    fetch_hf(BIREFNET_DYNAMIC_MATTING)
    if include_generative:
        fetch_hf(IC_LIGHT)
        fetch_hf(SD15_REALISTIC_VISION)
    if include_flux:
        fetch_hf(FLUX2_KLEIN_4B)
