"""studio-flux mode: FLUX.2 [klein] 4B (Apache-2.0) as a larger relighting oracle.

FLUX.2 [klein] is a 4B rectified-flow transformer with a Qwen3-4B text encoder that edits a
reference image from an instruction, in 4 distilled steps. It is used exactly like IC-Light in
studio-ai: its render is registered to the real face, only its smooth lighting / colour field is
transferred (relight.transfer_lighting), and the render is then discarded.

Memory plan for an 8 GB card (the model card asks for ~13 GB):
* the 8 GB text encoder runs once on the CPU; the prompt embedding is cached on disk;
* the transformer is stored on the GPU in float8 e4m3 (3.9 GB) and computed in bfloat16;
* the VAE stays in bfloat16 on the GPU.
"""

from __future__ import annotations

import gc
import hashlib
import time

import numpy as np
import torch
from PIL import Image

from . import weights

PROMPT = (
    "Relight this portrait as a professional studio photo: a large soft frontal softbox slightly above "
    "the camera, gentle fill light, even illumination on both sides of the face, neutral white balance, "
    "plain white background. Keep the person, face shape, eyes, skin texture, hair, clothing and "
    "expression exactly the same."
)


def prompt_embeddings(base) -> torch.Tensor:
    """Qwen3 embedding of the fixed prompt, computed once on the CPU and cached on disk."""
    key = hashlib.sha1(f"{PROMPT}\0{weights.FLUX2_KLEIN_4B.revision}".encode()).hexdigest()[:12]
    path = weights.MODELS_DIR / "cache" / f"flux2_klein_prompt_{key}.pt"
    if path.exists():
        return torch.load(path)
    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(base, transformer=None, vae=None, torch_dtype=torch.bfloat16)
    with torch.inference_mode():
        out = pipe.encode_prompt(PROMPT, device=torch.device("cpu"))
    emb = (out[0] if isinstance(out, tuple) else out).to(torch.bfloat16)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb, path)
    del pipe
    gc.collect()
    return emb


class FluxRelighter:
    def __init__(self, seed: int = 20261027):
        from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Transformer2DModel

        self.seed = seed
        base = weights.fetch_hf(weights.FLUX2_KLEIN_4B)
        self.prompt_embeds = prompt_embeddings(base).to("cuda")
        from accelerate import init_empty_weights

        # from_pretrained would first copy all 7.75 GB into RAM (~13 GB peak with the fp8 copy).
        # Instead: bf16 tensors that are zero-copy views of the read-only file mapping (no commit),
        # cast to fp8 (3.8 GB), then moved to the GPU module by module.
        with init_empty_weights():
            transformer = Flux2Transformer2DModel.from_config(Flux2Transformer2DModel.load_config(base, subfolder="transformer"))
        state = dict(weights.read_safetensors(base / "transformer" / "diffusion_pytorch_model.safetensors"))
        transformer.load_state_dict(state, assign=True, strict=True)
        del state
        transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
        transformer.to("cuda")
        gc.collect()
        vae = AutoencoderKLFlux2.from_pretrained(base, subfolder="vae", torch_dtype=torch.bfloat16).to("cuda")
        self.pipe = Flux2KleinPipeline.from_pretrained(
            base, transformer=transformer, vae=vae, text_encoder=None, tokenizer=None, torch_dtype=torch.bfloat16
        )
        torch.cuda.empty_cache()

    def close(self) -> None:
        del self.pipe, self.prompt_embeds
        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def relight(self, subject_rgb: np.ndarray, size: tuple[int, int] = (768, 992), steps: int = 4) -> tuple[np.ndarray, dict]:
        w, h = size
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        ref = Image.fromarray(subject_rgb).resize((w, h), Image.LANCZOS)
        out = self.pipe(
            image=ref,
            prompt_embeds=self.prompt_embeds,
            height=h,
            width=w,
            num_inference_steps=steps,
            guidance_scale=1.0,
            generator=torch.Generator("cuda").manual_seed(self.seed),
        ).images[0]
        info = {
            "model": "FLUX.2 [klein] 4B (fp8 storage, bf16 compute)",
            "size": [w, h],
            "steps": steps,
            "guidance": 1.0,
            "seed": self.seed,
            "seconds": round(time.perf_counter() - t0, 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
        return np.asarray(out.convert("RGB")), info
