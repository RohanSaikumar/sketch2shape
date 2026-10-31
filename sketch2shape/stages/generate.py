"""Stage 3: turn the line art into product-style photos with SDXL and a scribble ControlNet.

The ControlNet keeps the sketch's outline and layout; the prompt asks for one centred object on a
plain background in soft light, which is what the 3D model reconstructs best from. Seeds are fixed
per round, so a run is repeatable. The model runs in a GPU worker process (see models.py).
"""
from __future__ import annotations

from pathlib import Path

from ..contracts import Candidate


def generate(models, sketch, intent, control_scale: float, n: int, round_idx: int, store, cfg):
    """Returns (candidates, number actually generated). The worker drops to one image if two do not fit."""
    g = cfg.generate
    seed = int(g.seed) + 1000 * round_idx
    result = models.run_worker(
        "generate", "sdxl", store,
        base_model=g.base_model, controlnet=g.controlnet, vae=g.vae,
        control_path=str(sketch.control_path), prompt=intent.prompt, negative_prompt=intent.negative_prompt,
        control_scale=float(control_scale), n=int(n), steps=int(g.steps),
        guidance_scale=float(g.guidance_scale), size=int(g.size), seed=seed,
        out_dir=str(store.dir), name_prefix=f"05_candidate_r{round_idx}_",
    )
    candidates = [Candidate(path=Path(p), seed=seed, index=i, round=round_idx,
                            control_scale=round(float(control_scale), 2))
                  for i, p in enumerate(result["paths"])]
    return candidates, int(result["n"])
