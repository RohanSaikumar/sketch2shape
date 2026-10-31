"""Gradio app: draw or upload a sketch, add an optional description, get a textured 3D model.

Run:  python app.py            (local)
      python app.py --share    (public gradio.live link, e.g. from a Kaggle notebook)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from sketch2shape import Sketch2Shape, load_config

PROGRESS = {"preprocess": 0.05, "understand": 0.1, "generate": 0.3, "score": 0.35, "prepare": 0.4,
            "shape": 0.55, "check_shape": 0.6, "texture": 0.85, "check_texture": 0.9, "export": 1.0}


def gallery_items(result) -> list[tuple[str, str]]:
    run_dir = Path(result.run_dir)
    items = []
    if (run_dir / "02_lineart.png").exists():
        items.append((str(run_dir / "02_lineart.png"), "cleaned sketch"))
    chosen = result.manifest.get("chosen_candidate")
    for c in result.manifest.get("candidates", []):
        label = f"score {c['total']:.2f}" + ("  (chosen)" if c["path"] == chosen else "")
        items.append((str(run_dir / c["path"]), label))
    if (run_dir / "06_prepared.png").exists():
        items.append((str(run_dir / "06_prepared.png"), "input to 3D"))
    return items


def run_summary(manifest: dict) -> dict:
    intent = manifest.get("intent") or {}
    return {
        "object": intent.get("label"),
        "named_by": intent.get("source"),
        "prompt_used": intent.get("prompt"),
        "result": manifest.get("result"),
        "export": manifest.get("export"),
        "timings_s": manifest.get("timings_s"),
        "warnings": manifest.get("warnings"),
        "run_folder": manifest.get("run_id"),
    }


def build_app(pipeline: Sketch2Shape) -> gr.Blocks:
    def make_model(canvas, upload, prompt, progress=gr.Progress()):
        image, source = (upload, "upload") if upload is not None else (canvas, "canvas")

        def on_event(state: str, outcome: str, elapsed: float) -> None:
            progress(PROGRESS.get(state, 0.0), desc=f"{state.replace('_', ' ')}: {outcome} ({elapsed:.0f} s)")

        result = pipeline.run(image, prompt or "", source, on_event)
        files = [str(p) for p in (result.glb_path, result.obj_zip_path) if p]
        status = ("**Done.** " if result.ok else "**Could not make a model.** ") + result.message
        warnings = result.manifest.get("warnings") or []
        if result.ok and warnings:
            status += "\n\nNotes: " + " ".join(warnings[:3])
        return (str(result.glb_path) if result.glb_path else None, files or None,
                gallery_items(result), run_summary(result.manifest), status)

    with gr.Blocks(title="Sketch2Shape") as demo:
        gr.Markdown("# Sketch2Shape\nDraw a sketch or upload one, optionally describe it, "
                    "and get a textured 3D model you can download as .glb or .obj.")
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
        with gr.Accordion("How it was made", open=False):
            gallery = gr.Gallery(label="Steps", columns=4, height=280)
            details = gr.JSON(label="Run summary")
        button.click(make_model, [canvas, upload, prompt], [viewer, files, gallery, details, status])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Sketch2Shape web app")
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    parser.add_argument("--config", help="path to a config.yaml")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    cfg = load_config(args.config)
    runs = Path(cfg.runs_dir).resolve()
    runs.mkdir(parents=True, exist_ok=True)
    demo = build_app(Sketch2Shape(cfg))
    demo.queue(default_concurrency_limit=1).launch(share=args.share, server_port=args.port,
                                                   allowed_paths=[str(runs)])


if __name__ == "__main__":
    main()
