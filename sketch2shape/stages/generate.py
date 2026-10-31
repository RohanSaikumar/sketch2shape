"""Stage 3: turn the line art into product-style photos with SDXL and a scribble ControlNet.

The ControlNet keeps the sketch's outline and layout; the prompt asks for one centred object on a
plain background in soft light, which is what the 3D model reconstructs best from. Seeds are fixed
per round, so a run is repeatable.
"""
from __future__ import annotations

from PIL import Image

from ..contracts import Candidate


def generate(models, sketch, intent, control_scale: float, n: int, round_idx: int, store, cfg):
    """Returns (candidates, number actually generated). Drops to one image if two do not fit in memory."""
    import torch
    g = cfg.generate
    pipe = models.sdxl()
    control = Image.open(sketch.control_path).convert("RGB").resize((int(g.size), int(g.size)))
    seed = int(g.seed) + 1000 * round_idx
    device = f"cuda:{models.gpu('sdxl')}"
    while True:
        try:
            images = pipe(
                intent.prompt, negative_prompt=intent.negative_prompt, image=control,
                controlnet_conditioning_scale=float(control_scale),
                num_inference_steps=int(g.steps), guidance_scale=float(g.guidance_scale),
                num_images_per_prompt=n, width=int(g.size), height=int(g.size),
                generator=torch.Generator(device).manual_seed(seed),
            ).images
            break
        except torch.cuda.OutOfMemoryError:
            models.free_memory()
            if n == 1:
                raise
            n = 1
    candidates = []
    for index, image in enumerate(images):
        path = store.save_image(image, f"05_candidate_r{round_idx}_{index}.png")
        candidates.append(Candidate(path=path, seed=seed, index=index, round=round_idx,
                                    control_scale=round(float(control_scale), 2)))
    return candidates, n
