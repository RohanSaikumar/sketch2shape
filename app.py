"""Gradio app: draw or upload a sketch, add an optional description, get a textured 3D model.

While a model is being made, the page shows what is happening right now: a progress bar weighted by
how long each step takes, the current step with its elapsed and usual time, a checklist of finished
steps, and each intermediate result as soon as it exists (cleaned sketch, candidate images with
scores, cut-out, then the untextured 3D shape, which can be rotated while the texture is painted).

Run:  python app.py            (local)
      python app.py --share    (public gradio.live link, e.g. from a Kaggle notebook)
"""
from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path

import gradio as gr
from PIL import Image

from sketch2shape import Sketch2Shape, load_config
from sketch2shape.models import relaunch_with_lean_malloc, stop_leftover_workers

# step: (what the viewer is told, usual seconds on Kaggle's T4s, progress at start, progress at end)
STEPS = {
    "preprocess": ("Cleaning up the sketch", 2, 0.00, 0.02),
    "understand": ("Working out what the object is", 10, 0.02, 0.05),
    "generate": ("Generating product-style images that follow your lines (SDXL)", 110, 0.05, 0.35),
    "score": ("Checking which image follows the sketch best", 20, 0.35, 0.40),
    "prepare": ("Cutting the object out of the chosen image", 5, 0.40, 0.42),
    "shape": ("Building the 3D shape (Hunyuan3D-2)", 85, 0.42, 0.65),
    "check_shape": ("Checking the mesh", 5, 0.65, 0.67),
    "texture": ("Painting the texture (Hunyuan3D-2)", 115, 0.67, 0.95),
    "check_texture": ("Checking the texture", 5, 0.95, 0.97),
    "export": ("Saving the GLB and OBJ files", 5, 0.97, 1.00),
}


def thumbnail(path: Path, size: int = 384) -> str | None:
    """A small JPEG of an intermediate image (~25 KB instead of ~1 MB), so the live view works on slow links.
    Returns None if the image cannot be read yet (e.g. still being written); the next refresh retries."""
    target = path.parent / "thumbs" / (path.stem + ".jpg")
    if not target.exists():
        try:
            image = Image.open(path).convert("RGBA")
        except (OSError, SyntaxError, ValueError):
            return None
        target.parent.mkdir(exist_ok=True)
        flat = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image).convert("RGB")
        flat.thumbnail((size, size))
        flat.save(target, quality=82)
    return str(target)


def gallery_items(run_dir: Path, manifest: dict) -> list[tuple[str, str]]:
    found = []
    if (run_dir / "02_lineart.png").exists():
        found.append((run_dir / "02_lineart.png", "cleaned sketch"))
    scored = {c["path"]: c for c in manifest.get("candidates", [])}
    chosen = manifest.get("chosen_candidate")
    for path in sorted(run_dir.glob("05_candidate_r*_[0-9].png")):
        c = scored.get(path.name)
        label = "scoring..." if c is None else f"score {c['total']:.2f}" + ("  (chosen)" if path.name == chosen else "")
        found.append((path, label))
    if (run_dir / "06_prepared.png").exists():
        found.append((run_dir / "06_prepared.png", "input to 3D"))
    items = [(thumbnail(path), label) for path, label in found]
    return [(thumb, label) for thumb, label in items if thumb]


def run_summary(manifest: dict) -> dict:
    intent = manifest.get("intent") or {}
    return {
        "object": intent.get("label"),
        "named_by": intent.get("source"),
        "prompt_used": intent.get("prompt"),
        "result": manifest.get("result"),
        "export": manifest.get("export"),
        "timings_s": manifest.get("timings_s"),
        "workers": manifest.get("workers"),
        "score_timings_s": manifest.get("score_timings_s"),
        "warnings": manifest.get("warnings"),
        "run_folder": manifest.get("run_id"),
    }


def status_text(done: list[str], current: tuple[str, float] | None, total_s: float) -> str:
    lines = [f"**Elapsed: {int(total_s) // 60}:{int(total_s) % 60:02d}**"]
    if current:
        label, usual = STEPS[current[0]][:2]
        lines.append(f"Now: {label}: {int(time.monotonic() - current[1])} s (usually about {usual} s)")
    lines += [f"- done: {line}" for line in done]
    return "\n\n".join(lines[:2]) + ("\n\n" + "\n".join(lines[2:]) if len(lines) > 2 else "")


def model_maker(pipeline: Sketch2Shape):
    """The button handler: runs the pipeline in a thread and streams progress and partial results."""
    def make_model(canvas, upload, prompt, progress=gr.Progress()):
        image, source = (upload, "upload") if upload is not None else (canvas, "canvas")
        ctx = pipeline.new_context(image, prompt or "", source)
        run_dir = Path(ctx.store.dir)
        events: queue.Queue = queue.Queue()
        holder = {}

        def work():
            holder["result"] = pipeline.finish(ctx, pipeline.drive(ctx, lambda *e: events.put(e)))

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        started = time.monotonic()
        done, current, fraction = [], None, 0.0
        shown_items, shown_shape = 0, None
        skip = gr.update()

        while worker.is_alive() or not events.empty():
            changed = False
            try:
                state, outcome, _ = events.get(timeout=1.0)
                if outcome == "start":
                    current = (state, time.monotonic())
                else:
                    took = time.monotonic() - current[1] if current else 0.0
                    done.append(f"{STEPS[state][0]} ({outcome}, {took:.0f} s)")
                    fraction = max(fraction, STEPS[state][3])
                    current = None
                changed = True
            except queue.Empty:
                pass
            if current:
                label, usual, low, high = STEPS[current[0]]
                within = min(0.95, (time.monotonic() - current[1]) / usual)
                fraction = max(fraction, low + (high - low) * within)
                progress(fraction, desc=f"{label}: {int(time.monotonic() - current[1])} s of about {usual} s")

            try:                            # the live view must never stop the run
                items = gallery_items(run_dir, ctx.store.manifest)
                gallery_update = items if (changed or len(items) != shown_items) else skip
                shown_items = len(items)
            except Exception:
                gallery_update = skip
            shapes = sorted(run_dir.glob("07_shape_seed*.glb"))
            viewer_update = skip
            if shapes and shapes[-1] != shown_shape:
                shown_shape = shapes[-1]
                viewer_update = gr.update(value=str(shown_shape), label="3D shape (texture still in progress)")
            yield (viewer_update, skip, gallery_update, skip,
                   status_text(done, current, time.monotonic() - started))

        result = holder.get("result")
        if result is None:
            yield (skip, skip, skip, skip, f"**Something went wrong.** See runs/{run_dir.name}/manifest.json.")
            return
        files = [str(p) for p in (result.glb_path, result.obj_zip_path) if p]
        status = ("**Done.** " if result.ok else "**Could not make a model.** ") + result.message
        warnings = result.manifest.get("warnings") or []
        if result.ok and warnings:
            status += "\n\nNotes: " + " ".join(warnings[:3])
        status += "\n\n" + "\n".join(f"- done: {line}" for line in done)
        viewer = gr.update(value=str(result.glb_path), label="3D model (drag to rotate)") if result.glb_path else skip
        yield (viewer, files or None, gallery_items(run_dir, result.manifest), run_summary(result.manifest), status)

    return make_model


def build_app(pipeline: Sketch2Shape) -> gr.Blocks:
    make_model = model_maker(pipeline)
    with gr.Blocks(title="Sketch2Shape") as demo:
        gr.Markdown("# Sketch2Shape\nDraw a sketch or upload one, optionally describe it, and get a textured "
                    "3D model you can download as .glb or .obj. On Kaggle's free T4 GPUs a model takes "
                    "about 5-7 minutes; each step and its result appear below as soon as they are ready.")
        with gr.Row():
            with gr.Column():
                with gr.Tab("Draw"):
                    canvas = gr.Sketchpad(label="Sketch (brush, eraser, undo and clear are in the toolbar)",
                                          type="pil",
                                          brush=gr.Brush(colors=["#000000"], color_mode="fixed", default_size=4))
                with gr.Tab("Upload"):
                    upload = gr.Image(label="Sketch image (clear it to use the drawing instead)", type="pil")
                prompt = gr.Textbox(label="Description (optional)",
                                    placeholder="e.g. a matte red ceramic mug, or a wooden chair")
                button = gr.Button("Make 3D model", variant="primary")
                status = gr.Markdown()
            with gr.Column():
                viewer = gr.Model3D(label="3D model (drag to rotate)", clear_color=(1, 1, 1, 1), height=440)
                files = gr.File(label="Downloads", file_count="multiple")
        gallery = gr.Gallery(label="Steps so far", columns=5, height=240)
        with gr.Accordion("Run details", open=False):
            details = gr.JSON(label="Run summary")
        button.click(make_model, [canvas, upload, prompt], [viewer, files, gallery, details, status])
    return demo


def main() -> None:
    relaunch_with_lean_malloc()
    stop_leftover_workers()
    parser = argparse.ArgumentParser(description="Sketch2Shape web app")
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    parser.add_argument("--config", help="path to a config.yaml")
    parser.add_argument("--port", type=int, default=None, help="default: the first free port from 7860")
    args = parser.parse_args()
    cfg = load_config(args.config)
    runs = Path(cfg.runs_dir).resolve()
    runs.mkdir(parents=True, exist_ok=True)
    demo = build_app(Sketch2Shape(cfg))
    demo.queue(default_concurrency_limit=1).launch(share=args.share, server_port=args.port,
                                                   allowed_paths=[str(runs)])


if __name__ == "__main__":
    main()
