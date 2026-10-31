"""GPU worker: runs one heavy model job in its own process, then exits.

The pipeline starts a fresh worker for each SDXL, shape and texture step, pinned to one GPU through
CUDA_VISIBLE_DEVICES. A process gives all of its memory back when it exits, so the big models never
share CPU RAM. (Keeping them in one long-lived process ran out of Kaggle's 30 GB.) Jobs and results
are small JSON files; images and meshes are passed as file paths.

    python -m sketch2shape.worker job.json result.json
"""
from __future__ import annotations

import json
import platform
import sys
import time
import traceback
from pathlib import Path


def _peak_gb() -> float:
    import torch
    return round(torch.cuda.max_memory_reserved() / 2**30, 2) if torch.cuda.is_available() else 0.0


def save_atomically(target, write) -> None:
    """Write via a temporary name in the same folder, then rename, so the app never reads half a file.
    `write` receives the temporary path, which keeps the target's extension (trimesh/PIL pick the format from it)."""
    target = Path(target)
    temporary = target.with_name(f"tmp_{target.name}")
    write(temporary)
    temporary.replace(target)


def task_ping(job: dict) -> dict:
    return {"python": platform.python_version()}


def task_generate(job: dict) -> dict:
    """SDXL + scribble ControlNet: n product-style images that follow the line art."""
    import torch
    from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetPipeline
    from PIL import Image

    started = time.time()
    controlnet = ControlNetModel.from_pretrained(job["controlnet"], torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(job["vae"], torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        job["base_model"], controlnet=controlnet, vae=vae,
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
    if hasattr(pipe.vae, "enable_slicing"):        # newer diffusers removed pipe.enable_vae_slicing()
        pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    load_s = time.time() - started

    size = int(job["size"])
    control = Image.open(job["control_path"]).convert("RGB").resize((size, size))
    n = int(job["n"])
    started = time.time()
    while True:
        try:
            images = pipe(
                job["prompt"], negative_prompt=job["negative_prompt"], image=control,
                controlnet_conditioning_scale=float(job["control_scale"]),
                num_inference_steps=int(job["steps"]), guidance_scale=float(job["guidance_scale"]),
                num_images_per_prompt=n, width=size, height=size,
                generator=torch.Generator("cuda").manual_seed(int(job["seed"])),
            ).images
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if n == 1:
                raise
            n = 1                                   # two images did not fit: make one
    paths = []
    for index, image in enumerate(images):
        path = Path(job["out_dir"]) / f"{job['name_prefix']}{index}.png"
        save_atomically(path, image.save)
        paths.append(str(path))
    return {"paths": paths, "n": n, "load_s": round(load_s, 1),
            "run_s": round(time.time() - started, 1), "peak_gb": _peak_gb()}


def task_shape(job: dict) -> dict:
    """Hunyuan3D-2 shape (turbo + FlashVDM by default) plus its floater/degenerate-face/face-count clean-up."""
    import torch
    from hy3dgen.shapegen import (DegenerateFaceRemover, FaceReducer, FloaterRemover,
                                  Hunyuan3DDiTFlowMatchingPipeline)
    from PIL import Image

    started = time.time()
    try:
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            job["model"], subfolder=job["subfolder"], use_safetensors=True)
        if job["flashvdm"]:
            pipe.enable_flashvdm()
        steps, variant = int(job["steps"]), job["subfolder"]
    except Exception as exc:                        # turbo weights unavailable: the base model needs ~50 steps
        print(f"[worker] {job['subfolder']} failed to load ({exc!r}); using the base shape model", flush=True)
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(job["model"])
        steps, variant = 50, "hunyuan3d-dit-v2-0"
    load_s = time.time() - started

    started = time.time()
    mesh = pipe(image=Image.open(job["image"]).convert("RGBA"), num_inference_steps=steps,
                generator=torch.manual_seed(int(job["seed"])))[0]
    for step in (FloaterRemover(), DegenerateFaceRemover(), FaceReducer()):
        mesh = step(mesh)
    save_atomically(job["out"], mesh.export)
    return {"variant": variant, "steps": steps, "faces": int(len(mesh.faces)), "load_s": round(load_s, 1),
            "run_s": round(time.time() - started, 1), "peak_gb": _peak_gb()}


def task_paint(job: dict) -> dict:
    """Hunyuan3D-2 texture for a shape GLB, retrying with CPU offload if the GPU runs out of memory."""
    import torch
    import trimesh
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    from PIL import Image

    started = time.time()
    pipe = Hunyuan3DPaintPipeline.from_pretrained(job["model"])
    load_s = time.time() - started

    image = Image.open(job["image"]).convert("RGBA")
    mode = "hunyuan paint"
    started = time.time()
    try:
        textured = pipe(trimesh.load(job["mesh"], force="mesh"), image=image)
    except torch.cuda.OutOfMemoryError:
        if not job["cpu_offload_on_oom"]:
            raise
        torch.cuda.empty_cache()
        pipe.enable_model_cpu_offload()
        textured = pipe(trimesh.load(job["mesh"], force="mesh"), image=image)
        mode = "hunyuan paint (CPU offload)"
    save_atomically(job["out"], textured.export)
    return {"mode": mode, "load_s": round(load_s, 1), "run_s": round(time.time() - started, 1),
            "peak_gb": _peak_gb()}


TASKS = {"ping": task_ping, "generate": task_generate, "shape": task_shape, "paint": task_paint}


def main(argv=None) -> int:
    job_path, result_path = (argv if argv is not None else sys.argv[1:])[:2]
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    try:
        result = {"ok": True, **TASKS[job["task"]](job)}
        try:
            import resource                         # peak CPU RAM of this worker (Linux reports KB)
            result["max_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
        except ImportError:
            pass
    except Exception as exc:
        result = {"ok": False, "error": repr(exc), "traceback": traceback.format_exc()[-3000:]}
        print(result["traceback"], file=sys.stderr, flush=True)
    Path(result_path).write_text(json.dumps(result), encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
