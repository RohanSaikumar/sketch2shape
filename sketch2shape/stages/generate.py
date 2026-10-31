"""Stage 3: turn the line art into realistic studio photos with SDXL and a scribble ControlNet.

The ControlNet keeps the sketch's outline and layout; the prompt asks for one centred object on a
plain background in soft light, which is what the 3D model reconstructs best from. A rough doodle
(as judged by the naming call) gets lighter control that stops part-way, so the sketch sets the layout
and the model draws a real-looking object. Seeds are fixed per round, so a run is repeatable. The
model runs in a GPU worker process (see models.py).
"""
from __future__ import annotations

from pathlib import Path

from ..contracts import Candidate


def model_job(cfg) -> dict:
    """The worker fields that choose the SDXL model (the app loads it with these before the first run)."""
    g = cfg.generate
    return {"base_model": g.base_model, "controlnet": g.controlnet, "vae": g.vae}


def sketch_control(intent, control_scale: float, round_idx: int, g) -> tuple[float, float]:
    """(ControlNet scale, share of the steps it guides) for one round. A rough doodle gets the lighter
    generate.rough settings in its first round; if the fidelity gate rejects that image, retry rounds
    follow the lines on every step again."""
    if round_idx == 0 and intent.drawing == "rough":
        return min(float(control_scale), float(g.rough.control_scale)), float(g.rough.control_end)
    return float(control_scale), 1.0


def generate(models, sketch, intent, control_scale: float, n: int, round_idx: int, store, cfg):
    """Returns (candidates, number actually generated). The worker drops to one image if two do not fit."""
    g = cfg.generate
    seed = int(g.seed) + 1000 * round_idx
    control_scale, control_end = sketch_control(intent, control_scale, round_idx, g)
    result = models.run_worker(
        "generate", "sdxl", store, **model_job(cfg),
        control_path=str(sketch.control_path), prompt=intent.prompt, negative_prompt=intent.negative_prompt,
        control_scale=control_scale, control_end=control_end, n=int(n), steps=int(g.steps),
        guidance_scale=float(g.guidance_scale), size=int(g.size), seed=seed,
        out_dir=str(store.dir), name_prefix=f"05_candidate_r{round_idx}_",
    )
    candidates = [Candidate(path=Path(p), seed=seed, index=i, round=round_idx,
                            control_scale=round(control_scale, 2))
                  for i, p in enumerate(result["paths"])]
    return candidates, int(result["n"])
