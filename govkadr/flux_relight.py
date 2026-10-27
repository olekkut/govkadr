"""FLUX.2 [klein] 4B (Apache-2.0): relighting oracle for studio-flux, and edge filling.

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


# Deliberately names no garment: an earlier prompt mentioning "suit jacket, shirt, tie" made FLUX
# paint a collar and tie onto a T-shirt.
FILL_PROMPT = (
    "Seamlessly complete the missing edge of this photo by continuing exactly what is already there: the "
    "same clothing, fabric, colour, folds and lighting as in the photo and in the reference photo of the "
    "same person. Do not add any new clothing, accessories or objects. Plain white background."
)


def prompt_embeddings(base, prompt: str = PROMPT) -> torch.Tensor:
    """Qwen3 embedding of a fixed prompt, computed once on the CPU and cached on disk."""
    key = hashlib.sha1(f"{prompt}\0{weights.FLUX2_KLEIN_4B.revision}".encode()).hexdigest()[:12]
    path = weights.MODELS_DIR / "cache" / f"flux2_klein_prompt_{key}.pt"
    if path.exists():
        return torch.load(path)
    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(base, transformer=None, vae=None, torch_dtype=torch.bfloat16)
    with torch.inference_mode():
        out = pipe.encode_prompt(prompt, device=torch.device("cpu"))
    emb = (out[0] if isinstance(out, tuple) else out).to(torch.bfloat16)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb, path)
    del pipe
    gc.collect()
    return emb


def load_flux(pipeline_cls_name: str = "Flux2KleinPipeline"):
    """FLUX.2 [klein] pipeline without text encoder: fp8 transformer streamed from the file mapping.

    ``from_pretrained`` would first copy all 7.75 GB into RAM (~13 GB peak with the fp8 copy).
    Instead the bf16 tensors are zero-copy views of the read-only file mapping (no commit), cast
    to fp8 (3.8 GB), then moved to the GPU.
    """
    import diffusers
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel

    base = weights.fetch_hf(weights.FLUX2_KLEIN_4B)
    with init_empty_weights():
        transformer = Flux2Transformer2DModel.from_config(Flux2Transformer2DModel.load_config(base, subfolder="transformer"))
    state = dict(weights.read_safetensors(base / "transformer" / "diffusion_pytorch_model.safetensors"))
    transformer.load_state_dict(state, assign=True, strict=True)
    del state
    transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    transformer.to("cuda")
    gc.collect()
    vae = AutoencoderKLFlux2.from_pretrained(base, subfolder="vae", torch_dtype=torch.bfloat16).to("cuda")
    pipe = getattr(diffusers, pipeline_cls_name).from_pretrained(
        base, transformer=transformer, vae=vae, text_encoder=None, tokenizer=None, torch_dtype=torch.bfloat16
    )
    torch.cuda.empty_cache()
    return pipe, base


class FluxRelighter:
    def __init__(self, seed: int = 20261027):
        self.seed = seed
        self.pipe, base = load_flux()
        self.prompt_embeds = prompt_embeddings(base).to("cuda")

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


class FluxFiller:
    """Generative completion of the image edge where the crop runs past the original photo.

    FLUX.2 [klein] inpainting fills only the masked pixels; a well-framed photo of the same person
    from the same session is passed as ``image_reference`` so the jacket, shirt and tie continue
    with the right fabric and colour.
    """

    def __init__(self, seed: int = 20261027):
        self.seed = seed
        self.pipe, base = load_flux("Flux2KleinInpaintPipeline")
        self.prompt_embeds = prompt_embeddings(base, FILL_PROMPT).to("cuda")

    def close(self) -> None:
        del self.pipe, self.prompt_embeds
        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def fill(self, image: np.ndarray, mask: np.ndarray, reference: np.ndarray | None, size: tuple[int, int] = (768, 992), steps: int = 4) -> tuple[np.ndarray, dict]:
        w, h = size
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        kwargs = {}
        if reference is not None:
            kwargs["image_reference"] = [Image.fromarray(reference).resize((w, h), Image.LANCZOS)]
        out = self.pipe(
            image=Image.fromarray(image).resize((w, h), Image.LANCZOS),
            mask_image=Image.fromarray(mask).resize((w, h), Image.NEAREST),
            prompt_embeds=self.prompt_embeds,
            height=h,
            width=w,
            strength=1.0,
            num_inference_steps=steps,
            guidance_scale=1.0,
            generator=torch.Generator("cuda").manual_seed(self.seed),
            **kwargs,
        ).images[0]
        info = {
            "model": "FLUX.2 [klein] 4B inpainting (fp8 storage, bf16 compute)",
            "size": [w, h],
            "steps": steps,
            "seed": self.seed,
            "reference_used": reference is not None,
            "seconds": round(time.perf_counter() - t0, 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
        return np.asarray(out.convert("RGB")), info
