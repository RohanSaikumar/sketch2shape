"""GPU worker: runs heavy model jobs in its own process.

The pipeline runs each SDXL, shape and texture job in a worker process pinned to one GPU through
CUDA_VISIBLE_DEVICES. A process gives all of its memory back when it exits, so the big models never
share CPU RAM. (Keeping them in one long-lived process ran out of Kaggle's 30 GB.) Jobs and results
are small JSON files; images and meshes are passed as file paths.

    python -m sketch2shape.worker job.json result.json
        load the model, run the one job, write result.json, exit
    python -m sketch2shape.worker --serve load.json ready.json
        load the model, write ready.json, then run each "job.json<TAB>result.json" line read from
        stdin, writing each result file when its job ends, until stdin is closed

The serving form keeps SDXL loaded between runs and lets the shape model load while the image is still
being made (see models.py). A job stops early when the file named by its "cancel_path" appears.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path


class Cancelled(Exception):
    """The job's cancel file appeared."""


def _peak_gb() -> float:
    import torch
    return round(torch.cuda.max_memory_reserved() / 2**30, 2) if torch.cuda.is_available() else 0.0


def _reset_peak() -> None:
    """Measure each job's GPU peak on its own when one worker runs several jobs."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def save_atomically(target, write) -> None:
    """Write via a temporary name in the same folder, then rename, so the app never reads half a file.
    `write` receives the temporary path, which keeps the target's extension (trimesh/PIL pick the format from it)."""
    target = Path(target)
    temporary = target.with_name(f"tmp_{target.name}")
    write(temporary)
    temporary.replace(target)


def write_json(path, data: dict) -> None:
    save_atomically(path, lambda temporary: Path(temporary).write_text(json.dumps(data), encoding="utf-8"))


def cancel_check(job: dict):
    """A function that tells whether this job has been cancelled (its cancel file exists)."""
    path = job.get("cancel_path")
    return (lambda: os.path.exists(path)) if path else (lambda: False)


# ------------------------------------------------------------------ tasks: load the model, run one job

def load_ping(job: dict):
    return None


def run_ping(state, job: dict) -> dict:
    """For tests: waits job["seconds"] (default 0), stopping early if cancelled."""
    cancelled, end = cancel_check(job), time.time() + float(job.get("seconds", 0))
    while time.time() < end:
        if cancelled():
            raise Cancelled()
        time.sleep(0.05)
    return {"python": platform.python_version()}


def load_generate(job: dict):
    import torch
    from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetPipeline

    controlnet = ControlNetModel.from_pretrained(job["controlnet"], torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(job["vae"], torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        job["base_model"], controlnet=controlnet, vae=vae,
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
    if hasattr(pipe.vae, "enable_slicing"):        # newer diffusers removed pipe.enable_vae_slicing()
        pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def run_generate(pipe, job: dict) -> dict:
    """SDXL + scribble ControlNet: n product-style images that follow the line art."""
    import inspect

    import torch
    from PIL import Image

    cancelled = cancel_check(job)
    if cancelled():                                 # cancelled while it waited for the model to load
        raise Cancelled()
    _reset_peak()
    size = int(job["size"])
    control = Image.open(job["control_path"]).convert("RGB").resize((size, size))
    n = int(job["n"])
    extra = {}
    if "callback_on_step_end" in inspect.signature(pipe.__call__).parameters:
        def stop_if_cancelled(pipeline, step, timestep, kwargs):
            if cancelled():
                pipeline._interrupt = True          # diffusers skips the remaining steps
            return kwargs
        extra["callback_on_step_end"] = stop_if_cancelled
    while True:
        try:
            images = pipe(
                job["prompt"], negative_prompt=job["negative_prompt"], image=control,
                controlnet_conditioning_scale=float(job["control_scale"]),
                control_guidance_end=float(job.get("control_end", 1.0)),   # share of the steps the sketch guides
                num_inference_steps=int(job["steps"]), guidance_scale=float(job["guidance_scale"]),
                num_images_per_prompt=n, width=size, height=size,
                generator=torch.Generator("cuda").manual_seed(int(job["seed"])), **extra,
            ).images
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if n == 1:
                raise
            n = 1                                   # two images did not fit: make one
    if cancelled():
        raise Cancelled()
    paths = []
    for index, image in enumerate(images):
        path = Path(job["out_dir"]) / f"{job['name_prefix']}{index}.png"
        save_atomically(path, image.save)
        paths.append(str(path))
    return {"paths": paths, "n": n, "peak_gb": _peak_gb()}


def load_shape(job: dict):
    """Hunyuan3D-2 shape model (turbo + FlashVDM by default). Returns (pipeline, variant, steps or None)."""
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    try:
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            job["model"], subfolder=job["subfolder"], use_safetensors=True)
        if job["flashvdm"]:
            pipe.enable_flashvdm()
        return pipe, job["subfolder"], None
    except Exception as exc:                        # turbo weights unavailable: the base model needs ~50 steps
        print(f"[worker] {job['subfolder']} failed to load ({exc!r}); using the base shape model", flush=True)
        return Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(job["model"]), "hunyuan3d-dit-v2-0", 50


def run_shape(state, job: dict) -> dict:
    """One shape plus its floater/degenerate-face/face-count clean-up."""
    import torch
    from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover
    from PIL import Image

    pipe, variant, base_steps = state
    steps = base_steps or int(job["steps"])
    _reset_peak()
    mesh = pipe(image=Image.open(job["image"]).convert("RGBA"), num_inference_steps=steps,
                octree_resolution=int(job.get("octree_resolution", 380)),
                generator=torch.manual_seed(int(job["seed"])))[0]
    for step in (FloaterRemover(), DegenerateFaceRemover(), FaceReducer()):
        mesh = step(mesh)
    save_atomically(job["out"], mesh.export)
    return {"variant": variant, "steps": steps, "faces": int(len(mesh.faces)), "peak_gb": _peak_gb()}


def load_paint(job: dict):
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    return Hunyuan3DPaintPipeline.from_pretrained(job["model"])


def run_paint(pipe, job: dict) -> dict:
    """Hunyuan3D-2 texture for a shape GLB, retrying with CPU offload if the GPU runs out of memory."""
    import torch
    import trimesh
    from PIL import Image

    _reset_peak()
    image = Image.open(job["image"]).convert("RGBA")
    mode = "hunyuan paint"
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
    return {"mode": mode, "peak_gb": _peak_gb()}


TASKS = {"ping": (load_ping, run_ping), "generate": (load_generate, run_generate),
         "shape": (load_shape, run_shape), "paint": (load_paint, run_paint)}


# ------------------------------------------------------------------ running jobs

def failure(exc: Exception) -> dict:
    result = {"ok": False, "error": repr(exc), "traceback": traceback.format_exc()[-3000:]}
    print(result["traceback"], file=sys.stderr, flush=True)
    return result


def load_model(job: dict):
    """(run function, loaded model, seconds the load took) for the job's task."""
    load, run = TASKS[job["task"]]
    started = time.time()
    model = load(job)
    return run, model, round(time.time() - started, 1)


def run_one(run, model, job: dict) -> dict:
    started = time.time()
    try:
        result = {"ok": True, **run(model, job)}
    except Cancelled:
        result = {"ok": False, "error": "cancelled", "cancelled": True}
    except Exception as exc:
        result = failure(exc)
    result["run_s"] = round(time.time() - started, 1)
    try:
        import resource                         # peak CPU RAM of this worker so far (Linux reports KB)
        result["max_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    except ImportError:
        pass
    return result


def serve(load_path: Path, ready_path: Path) -> int:
    try:
        run, model, load_s = load_model(json.loads(load_path.read_text(encoding="utf-8")))
    except Exception as exc:
        write_json(ready_path, failure(exc))
        return 1
    write_json(ready_path, {"ok": True, "load_s": load_s})
    while True:
        line = sys.stdin.readline()
        if not line.strip():                    # stdin closed: the app or the run is done with this worker
            return 0
        job_path, result_path = line.rstrip("\n").split("\t")
        try:
            result = run_one(run, model, json.loads(Path(job_path).read_text(encoding="utf-8")))
        except Exception as exc:                # e.g. an unreadable job file
            result = failure(exc)
        write_json(result_path, result)


def main(argv=None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if args[:1] == ["--serve"]:
        return serve(Path(args[1]), Path(args[2]))
    job_path, result_path = args[:2]
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    try:
        run, model, load_s = load_model(job)
    except Exception as exc:
        result = failure(exc)
    else:
        result = {**run_one(run, model, job), "load_s": load_s}
    Path(result_path).write_text(json.dumps(result), encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
