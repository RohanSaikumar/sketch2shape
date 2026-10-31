"""Gradio app: draw or upload a sketch, add an optional description, get a textured 3D model.

While a model is being made, a progress panel pinned to the top of the page shows what is happening
right now: a bar weighted by how long each step takes, the current step with its elapsed and usual
time, and the finished steps. Each intermediate result appears as soon as it exists (cleaned sketch,
candidate image with its score, cut-out, then the untextured 3D shape, which can be rotated while the
texture is painted). The button is disabled while a model is being made, and only one model is made
at a time, even if the page is reloaded mid-run. Cancel stops the run: GPU work stops at once, a step
on the CPU finishes first. SDXL and the CPU models start loading when the app starts.

Run:  python app.py            (local)
      python app.py --share    (public gradio.live link, e.g. from a Kaggle notebook)
"""
from __future__ import annotations

import argparse
import html
import json
import inspect
import queue
import threading
import time
from pathlib import Path

import gradio as gr
from PIL import Image

from sketch2shape import Sketch2Shape, load_config
from sketch2shape.models import relaunch_with_lean_malloc, stop_leftover_workers

ROOT = Path(__file__).resolve().parent
EXAMPLES = sorted((ROOT / "examples" / "sketches").glob("*.png"))

# step: (what the viewer is told, usual seconds on Kaggle's T4s, progress at start, progress at end).
# With SDXL kept loaded and the shape model loaded early, a run is ~4 minutes, most of it the texture.
STEPS = {
    "preprocess": ("Cleaning up the sketch", 2, 0.00, 0.01),
    "understand": ("Working out what the object is", 5, 0.01, 0.03),
    "generate": ("Generating a realistic image that follows your lines (SDXL)", 25, 0.03, 0.14),
    "score": ("Checking the image follows the sketch", 30, 0.14, 0.27),
    "prepare": ("Cutting the object out of the chosen image", 2, 0.27, 0.28),
    "shape": ("Building the 3D shape (Hunyuan3D-2)", 40, 0.28, 0.44),
    "check_shape": ("Checking the mesh", 2, 0.44, 0.45),
    "texture": ("Painting the texture (Hunyuan3D-2)", 130, 0.45, 0.96),
    "check_texture": ("Checking the texture", 2, 0.96, 0.97),
    "export": ("Saving the GLB and OBJ files", 4, 0.97, 1.00),
}

# Keeps the progress panel in view while the page scrolls. Older Gradio takes css in Blocks(), 6.x in launch().
CSS = """
.gradio-container { overflow: visible !important; }  /* 'hidden' here disables sticky positioning */
#s2s-status { position: sticky; top: 0; z-index: 100; background: var(--background-fill-primary); padding-bottom: 6px; }
"""
BLOCKS_TAKES_CSS = "css" in inspect.signature(gr.Blocks.__init__).parameters

# Only one model is made at a time. Gradio's queue enforces that per request, but a page reload cancels the
# request while the pipeline thread keeps running; this guard stops a second run from starting beside it.
BUSY = {"thread": None, "started": 0.0, "run": "", "ctx": None}
BUSY_LOCK = threading.Lock()


def run_in_progress() -> bool:
    with BUSY_LOCK:
        return BUSY["thread"] is not None and BUSY["thread"].is_alive()


# ------------------------------------------------------------------ pieces of the page

def panel(fraction: float, title: str, detail: str = "", done: list[str] | None = None,
          state: str = "running") -> str:
    """The progress panel: title, percentage, bar, one detail line and the finished steps."""
    pct = max(0, min(100, int(round(fraction * 100))))
    colour = {"running": "var(--color-accent)", "done": "#2e9e5b", "error": "#d64545",
              "idle": "var(--neutral-400)"}[state]
    steps = "".join(f"<div>&#10003; {html.escape(line)}</div>" for line in (done or []))
    return (
        '<div style="padding:10px 14px;border:1px solid var(--border-color-primary);border-radius:10px;'
        'background:var(--background-fill-secondary)">'
        '<div style="display:flex;justify-content:space-between;gap:12px;font-weight:600;margin-bottom:6px">'
        f'<span>{html.escape(title)}</span><span>{pct}%</span></div>'
        '<div style="height:12px;border-radius:6px;background:var(--neutral-200);overflow:hidden">'
        f'<div style="height:100%;width:{pct}%;background:{colour};transition:width .8s ease"></div></div>'
        f'<div style="margin-top:6px;font-size:.93em">{detail}</div>'
        f'<div style="margin-top:4px;font-size:.85em;opacity:.8;columns:2 260px">{steps}</div>'
        '</div>'
    )


IDLE = panel(0.0, "Ready", "Draw a sketch or upload one, optionally describe it, then click "
             "<b>Make 3D model</b>. A model takes about 4-5 minutes on Kaggle's free T4 GPUs "
             "(the first one after the app starts a little longer).", state="idle")


def models_line(pipeline: Sketch2Shape) -> tuple[str, bool]:
    """The model status line under the progress panel, and whether it is still changing (SDXL loading)."""
    status = getattr(pipeline.models, "sdxl_status", None)
    if status is None:
        return "", False
    state, text = status()
    colour = {"loading": "#d98e04", "ready": "#2e9e5b", "off": "var(--neutral-400)", "failed": "#d64545"}[state]
    return (f'<div style="font-size:.88em;padding:0 14px"><span style="color:{colour}">&#9679;</span> '
            f'{html.escape(text)}</div>'), state == "loading"


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
        "drawing": intent.get("drawing") or "not judged (no LLM)",
        "prompt_used": intent.get("prompt"),
        "candidates": [{key: c.get(key) for key in ("round", "control_scale", "total", "scores")}
                       for c in manifest.get("candidates") or []],
        "result": manifest.get("result"),
        "export": manifest.get("export"),
        "timings_s": manifest.get("timings_s"),
        "workers": manifest.get("workers"),
        "score_timings_s": manifest.get("score_timings_s"),
        "warnings": manifest.get("warnings"),
        "run_folder": manifest.get("run_id"),
    }


def past_runs(runs_dir: Path) -> list[tuple[str, str]]:
    """Finished runs with a model, newest first, as (label, run folder name) for a dropdown."""
    choices = []
    for manifest_path in sorted(Path(runs_dir).glob("*/manifest.json"), reverse=True):
        run_dir = manifest_path.parent
        if not (run_dir / "model.glb").exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        name = (manifest.get("intent") or {}).get("label") or "object"
        prompt = (manifest.get("inputs") or {}).get("prompt") or ""
        took = (manifest.get("result") or {}).get("total_s")
        label = f"{manifest.get('started', run_dir.name)} | {name}" + (f" | \"{prompt[:30]}\"" if prompt else "") \
            + (f" | {took:.0f} s" if took else "")
        choices.append((label, run_dir.name))
    return choices


def show_past_run(runs_dir: Path, run_name: str | None):
    """Model, images and downloads of an earlier run."""
    if not run_name:
        return None, None, []
    run_dir = Path(runs_dir) / run_name
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    files = [str(p) for p in (run_dir / "model.glb", run_dir / "model_obj.zip") if p.exists()]
    glb = run_dir / "model.glb"
    return (str(glb) if glb.exists() else None), (files or None), gallery_items(run_dir, manifest)


def has_strokes(canvas) -> bool:
    """True if the sketchpad holds anything drawn (any dark pixel once flattened onto white)."""
    image = canvas.get("composite") if isinstance(canvas, dict) else canvas
    if image is None:
        return False
    try:
        rgba = image.convert("RGBA") if isinstance(image, Image.Image) else Image.fromarray(image).convert("RGBA")
        flat = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("L")
        return flat.getextrema()[0] < 200
    except Exception:
        return True                         # unknown format: let preprocessing decide


def blank_canvas() -> dict:
    """A white drawing surface; some Gradio versions otherwise show a black, transparent canvas."""
    return {"background": Image.new("RGB", (800, 800), "white"), "layers": [], "composite": None}


def pick_input(mode: str, canvas, upload):
    """The open tab decides which input is used; if that one is empty, fall back to the other."""
    if mode == "upload":
        return (upload, "upload") if upload is not None else (canvas, "canvas")
    if has_strokes(canvas) or upload is None:
        return canvas, "canvas"
    return upload, "upload"


def elapsed_text(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


# ------------------------------------------------------------------ the button handler

def model_maker(pipeline: Sketch2Shape):
    """Runs the pipeline in a thread and streams progress and partial results to the page."""
    def make_model(canvas, upload, prompt, mode="canvas"):
        skip = gr.update()
        with BUSY_LOCK:
            running = BUSY["thread"]
            if running is not None and running.is_alive():
                yield (skip, skip, skip, skip,
                       panel(0.0, "Already making a model",
                             f"A model started {elapsed_text(time.monotonic() - BUSY['started'])} ago "
                             f"(run {html.escape(BUSY['run'])}) is still being made. Try again when it "
                             "finishes, or stop it with <b>Cancel</b>; only one model is made at a time.",
                             state="error"))
                return
            image, source = pick_input(mode, canvas, upload)
            ctx = pipeline.new_context(image, prompt or "", source)
            run_dir = Path(ctx.store.dir)
            events: queue.Queue = queue.Queue()
            holder = {}

            def work():
                holder["result"] = pipeline.finish(ctx, pipeline.drive(ctx, lambda *e: events.put(e)))

            worker = threading.Thread(target=work, daemon=True)
            BUSY.update(thread=worker, started=time.monotonic(), run=run_dir.name, ctx=ctx)
            worker.start()

        started = time.monotonic()
        used = "your drawing" if source == "canvas" else "the uploaded image"
        # clear the previous model's results as the new run starts
        yield (gr.update(value=None, label="3D model (drag to rotate)"), None, [], None,
               panel(0.0, "Starting", f"Using {used}."))
        done, current, fraction = [], None, 0.0
        shown_items, shown_shape = 0, None

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
            title, detail = "Working", f"Elapsed {elapsed_text(time.monotonic() - started)}"
            if current:
                label, usual, low, high = STEPS[current[0]]
                spent = time.monotonic() - current[1]
                fraction = max(fraction, low + (high - low) * min(0.95, spent / usual))
                title = label
                detail = f"{int(spent)} s of about {usual} s &middot; elapsed {elapsed_text(time.monotonic() - started)}"
            if ctx.cancelled:
                title, detail = "Cancelling...", "Stopping the current step (a step on the CPU finishes first)."

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
            yield viewer_update, skip, gallery_update, skip, panel(fraction, title, detail, done)

        result = holder.get("result")
        if result is None:
            yield (skip, skip, skip, skip, panel(fraction, "Something went wrong",
                   f"See runs/{html.escape(run_dir.name)}/manifest.json.", done, state="error"))
            return
        files = [str(p) for p in (result.glb_path, result.obj_zip_path) if p]
        warnings = result.manifest.get("warnings") or []
        if result.ok:
            detail = (f"{html.escape(result.message)} Download the model below, or drag the viewer to rotate it."
                      + (f"<br><i>Notes:</i> {html.escape(' '.join(warnings[:3]))}" if warnings else ""))
            final = panel(1.0, "Done", detail, done, state="done")
        elif ctx.cancelled:
            final = panel(fraction, "Cancelled", "The run was stopped. Click <b>Make 3D model</b> to start "
                          "another one.", done, state="idle")
        else:
            final = panel(fraction, "Could not make a model", html.escape(result.message), done, state="error")
        viewer = gr.update(value=str(result.glb_path), label="3D model (drag to rotate)") if result.glb_path else skip
        yield viewer, files or None, gallery_items(run_dir, result.manifest), run_summary(result.manifest), final

    return make_model


# ------------------------------------------------------------------ the page

def build_app(pipeline: Sketch2Shape) -> gr.Blocks:
    make_model = model_maker(pipeline)
    blocks_args = {"css": CSS} if BLOCKS_TAKES_CSS else {}
    with gr.Blocks(title="Sketch2Shape", **blocks_args) as demo:
        gr.Markdown("# Sketch2Shape\nDraw a sketch or upload one, optionally describe it, and get a textured "
                    "3D model you can download as .glb or .obj.")
        status = gr.HTML(IDLE, elem_id="s2s-status")
        loaded = gr.HTML(models_line(pipeline)[0])
        timer = gr.Timer(2.0)                   # refreshes the line while SDXL loads, then stops
        mode = gr.State("canvas")
        with gr.Row():
            with gr.Column():
                with gr.Tab("Draw") as draw_tab:
                    canvas = gr.Sketchpad(value=blank_canvas(), type="pil",
                                          label="Sketch (brush, eraser, undo and clear are in the toolbar)",
                                          brush=gr.Brush(colors=["#000000"], color_mode="fixed", default_size=4))
                with gr.Tab("Upload") as upload_tab:
                    upload = gr.Image(label="Sketch image (photo, scan or drawing)", type="pil")
                    if EXAMPLES:
                        gr.Examples(examples=[[str(p)] for p in EXAMPLES], inputs=[upload],
                                    label="Or pick an example sketch")
                prompt = gr.Textbox(label="Description (optional)",
                                    placeholder="e.g. a matte red ceramic mug, or a wooden chair. "
                                                "Leave empty to let the app decide.")
                with gr.Row():
                    button = gr.Button("Make 3D model", variant="primary", scale=3)
                    stop = gr.Button("Cancel", variant="stop", interactive=False, scale=1)
            with gr.Column():
                viewer = gr.Model3D(label="3D model (drag to rotate)", clear_color=(1, 1, 1, 1), height=440)
                files = gr.File(label="Downloads", file_count="multiple")
        gallery = gr.Gallery(label="Steps so far", columns=5, height=240)
        with gr.Accordion("Run details", open=False):
            details = gr.JSON(label="Run summary")
        runs_dir = Path(pipeline.cfg.runs_dir)
        with gr.Accordion("Previous models (kept while this app runs)", open=False):
            with gr.Row():
                history = gr.Dropdown(choices=past_runs(runs_dir), label="Pick an earlier run", scale=4)
                refresh = gr.Button("Refresh list", scale=1)
            with gr.Row():
                old_viewer = gr.Model3D(label="Earlier model (drag to rotate)", clear_color=(1, 1, 1, 1), height=360)
                old_files = gr.File(label="Its downloads", file_count="multiple")
            old_gallery = gr.Gallery(label="Its steps", columns=5, height=200)

        def cancel_run():
            with BUSY_LOCK:
                running, ctx = BUSY["thread"], BUSY["ctx"]
            if running is None or not running.is_alive() or ctx is None:
                return gr.update()
            pipeline.cancel(ctx)
            return gr.update(interactive=False, value="Cancelling...")

        def refresh_models_line():
            line, changing = models_line(pipeline)
            return line, gr.Timer(active=changing)

        def idle_buttons():                 # Cancel stays usable if another tab's run is still going
            return (gr.update(interactive=True, value="Make 3D model"),
                    gr.update(interactive=run_in_progress(), value="Cancel"))

        draw_tab.select(lambda: "canvas", None, mode)
        upload_tab.select(lambda: "upload", None, mode)
        (button.click(lambda: (gr.update(interactive=False, value="Working... please wait"),
                               gr.update(interactive=True, value="Cancel")), None, [button, stop], queue=False)
               .then(make_model, [canvas, upload, prompt, mode], [viewer, files, gallery, details, status],
                     show_progress="hidden")
               .then(idle_buttons, None, [button, stop], queue=False)
               .then(lambda: gr.update(choices=past_runs(runs_dir)), None, history, queue=False)
               .then(refresh_models_line, None, [loaded, timer], queue=False))
        stop.click(cancel_run, None, stop, queue=False)
        timer.tick(refresh_models_line, None, [loaded, timer], queue=False)
        demo.load(lambda: gr.update(interactive=run_in_progress()), None, stop, queue=False)
        demo.load(refresh_models_line, None, [loaded, timer], queue=False)
        refresh.click(lambda: gr.update(choices=past_runs(runs_dir)), None, history, queue=False)
        history.change(lambda name: show_past_run(runs_dir, name), history, [old_viewer, old_files, old_gallery],
                       queue=False)
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
    pipeline = Sketch2Shape(cfg)
    pipeline.models.warm_up()               # SDXL and the CPU models load while the page opens
    demo = build_app(pipeline)
    launch_args = {} if BLOCKS_TAKES_CSS else {"css": CSS}
    demo.queue(default_concurrency_limit=1).launch(share=args.share, server_port=args.port,
                                                   allowed_paths=[str(runs), str(ROOT / "examples")],
                                                   **launch_args)


if __name__ == "__main__":
    main()
