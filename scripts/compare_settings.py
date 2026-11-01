"""Compare image settings on the same sketches, so defaults are chosen from evidence instead of guesses.

Stop the app first: it keeps SDXL loaded on GPU 0. Each sketch is cleaned up and named once; then every
variant below makes two images from it with the same seeds, and the vision-LLM check marks the usable one
(green border) or rejects both (red). One sheet per sketch, a row per variant, goes to --out. With
--paint, the first sketch's "now" image is also built in 3D and textured twice, with and without
Hunyuan's light-removal (delight) step; both models are saved as runs named compare-..., so they can be
rotated side by side in the app's "Previous models".

    python scripts/compare_settings.py --last 3            # the sketches of the last 3 app runs
    python scripts/compare_settings.py --images examples/sketches/mug.png photo.jpg --paint
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap
import time
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sketch2shape import load_config  # noqa: E402
from sketch2shape.contracts import Candidate  # noqa: E402
from sketch2shape.models import ModelManager, gpu_count, relaunch_with_lean_malloc  # noqa: E402
from sketch2shape.runstore import RunStore  # noqa: E402
from sketch2shape.stages.generate import model_job, sketch_control  # noqa: E402
from sketch2shape.stages.image_check import check_images  # noqa: E402
from sketch2shape.stages.preprocess import preprocess, turn_upright  # noqa: E402
from sketch2shape.stages.understand import build_prompt, understand, with_article  # noqa: E402

# the settings of the 1 Nov morning runs (e.g. the good headphones), before any prompt changes
OLD_SUFFIX = ("single object, centered, whole object in frame, product photo, plain white background, "
              "soft even studio lighting, high detail")
OLD_NEGATIVE = ("multiple objects, cropped, text, watermark, cast shadow, ground, scene, cluttered background, "
                "blurry, sketch, line art, drawing, hands, people, liquid")
PRODUCT_SUFFIX = ("single object with real volume, plain white background, centered, entire object visible, "
                  "product photo, soft even studio lighting, high detail")


def variants(intent, typed: str, cfg) -> dict:
    """name -> (prompt, negative prompt, ControlNet scale, share of steps it guides)."""
    g = cfg.generate
    now_scale, now_end = sketch_control(intent, float(g.control_scale), False, g)
    if typed:
        old_subject = typed if not intent.label or intent.label in typed.lower() else f"{with_article(intent.label)}, {typed}"
    else:
        old_subject = with_article(intent.label) if intent.label else "a single everyday object"
    no_realism_neg = intent.negative_prompt.replace(f", {g.realism.negative_prompt}", "")
    firmer = {"scene": (0.5, 0.6)}.get(intent.kind) or ((0.7, 0.8) if intent.drawing == "rough" else (0.85, 1.0))
    return {
        "before (1 Nov morning)": (f"{old_subject}, {OLD_SUFFIX}", OLD_NEGATIVE, 0.7, 1.0),
        "now": (intent.prompt, intent.negative_prompt, now_scale, now_end),
        "now, product-photo look": (intent.prompt.replace(f"{g.realism.prompt}, ", "").replace(g.prompt_suffix, PRODUCT_SUFFIX),
                                    no_realism_neg, now_scale, now_end),
        "now, no GPT colours": (build_prompt(intent.label, intent.details, typed, cfg, "", intent.kind),
                                intent.negative_prompt, now_scale, now_end),
        "now, firmer sketch control": (intent.prompt, intent.negative_prompt, *firmer),
    }


def sketches_from(args, cfg) -> list[tuple[Path, str, str]]:
    """(image, typed description, source) for each sketch to compare."""
    if args.images:
        return [(Path(p), args.prompt, "upload") for p in args.images]
    runs = sorted((d for d in Path(cfg.runs_dir).iterdir()
                   if d.is_dir() and (d / "01_input.png").exists() and not d.name.startswith("compare")),
                  reverse=True)[:args.last]
    found = []
    for run in runs:
        inputs = json.loads((run / "manifest.json").read_text(encoding="utf-8")).get("inputs", {}) \
            if (run / "manifest.json").exists() else {}
        found.append((run / "01_input.png", inputs.get("prompt", ""), inputs.get("source", "upload")))
    return found


def sketch_type(image: Image.Image, source: str, drawing: str) -> str:
    """Which of three kinds of input this is, so the comparison can show which settings suit each: a photo of
    a drawing on paper (grey, uneven background), a rough doodle (e.g. drawn with a mouse) or a clean drawing."""
    import numpy as np
    flat = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image.convert("RGBA")).convert("L")
    if source == "upload" and (np.asarray(flat) > 235).mean() < 0.5:
        return "paper photo"
    return "rough doodle" if drawing == "rough" else "clean drawing"


def tile(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    flat = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image).convert("RGB")
    flat.thumbnail((size, size))
    return flat


def sheet(sketch_path: Path, rows: list[dict], target: Path, size: int = 300) -> None:
    """A row per variant: name, control and verdict on top, then the sketch and the two images."""
    text_h = 64
    out = Image.new("RGB", (size * 3 + 40, len(rows) * (size + text_h + 10)), "white")
    draw = ImageDraw.Draw(out)
    for i, row in enumerate(rows):
        y = i * (size + text_h + 10)
        head = f"{row['name']}  |  control {row['scale']:.2f} for {row['end']:.0%} of steps  |  {row['verdict']}"
        draw.text((10, y + 4), head, fill="black")
        draw.text((10, y + 20), "\n".join(textwrap.wrap(row["prompt"], 150)[:3]), fill=(90, 90, 90))
        out.paste(tile(sketch_path, size), (10, y + text_h))
        for j, (path, judged) in enumerate(row["images"]):
            x = 10 + (j + 1) * (size + 10)
            out.paste(tile(path, size), (x, y + text_h))
            colour = {"picked": (46, 158, 91), "rejected": (214, 69, 69)}.get(judged)
            if colour:
                draw.rectangle([x, y + text_h, x + size - 1, y + text_h + size - 1], outline=colour, width=6)
    out.save(target, quality=88)


def paint_test(models, cfg, sketch_store, intent, picked: Path, typed: str, stamp: str) -> None:
    """Shape once, then texture with and without delight; each saved as a run for the app's Previous models."""
    import trimesh

    from sketch2shape.stages.export import export_mesh
    from sketch2shape.stages.prepare import prepare
    from sketch2shape.stages.reconstruct import shape_model_job
    from sketch2shape.stages.validate import as_trimesh

    r = cfg.reconstruct
    candidate = Candidate(path=picked, seed=int(cfg.generate.seed), index=0, round=0, control_scale=0.0)
    prepared = prepare(models, candidate, sketch_store, cfg)
    shape_path = sketch_store.path("07_shape.glb")
    models.run_worker("shape", "hunyuan", sketch_store, **shape_model_job(cfg), steps=int(r.shape_steps),
                      octree_resolution=int(r.octree_resolution), image=str(prepared.rgba_path),
                      seed=int(r.seed), out=str(shape_path))
    for delight in (True, False):
        label = "on" if delight else "off"
        out = sketch_store.path(f"09_textured_delight_{label}.glb")
        started = time.monotonic()
        models.run_worker("paint", "paint", sketch_store, model=r.model, mesh=str(shape_path),
                          image=str(prepared.rgba_path), out=str(out), cpu_offload_on_oom=True, delight=delight)
        run = RunStore(cfg.runs_dir, f"compare-{stamp}-delight-{label}")
        for name in ("01_input.png", "02_lineart.png", "06_prepared.png"):
            if sketch_store.path(name).exists():
                shutil.copy(sketch_store.path(name), run.path(name))
        export_mesh(as_trimesh(trimesh.load(out, force="scene")), run, cfg)
        run.set("intent", {"label": f"{intent.label} (delight {label})", "source": intent.source})
        run.set("inputs", {"prompt": typed, "source": "compare"})
        run.set("result", {"ok": True, "message": f"compare_settings.py, delight {label}",
                           "total_s": round(time.monotonic() - started, 1)})
        run.write()
        print(f"  textured, delight {label}: {run.dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", nargs="*", help="sketch images to compare")
    parser.add_argument("--prompt", default="", help="typed description used with --images")
    parser.add_argument("--last", type=int, default=3, help="otherwise: the input sketches of the last N app runs")
    parser.add_argument("--paint", action="store_true", help="also compare texturing with and without delight")
    parser.add_argument("--out", default="/kaggle/working/compare" if Path("/kaggle").exists() else "compare")
    parser.add_argument("--config")
    args = parser.parse_args()
    relaunch_with_lean_malloc()
    if gpu_count() == 0:
        sys.exit("No NVIDIA GPU found.")

    keep = ["generate", "paint"] if args.paint else ["generate"]
    cfg = load_config(args.config, overrides={"workers.keep_loaded": keep})
    g = cfg.generate
    models = ModelManager(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    summary = []
    for number, (image_path, typed, source) in enumerate(sketches_from(args, cfg)):
        store = RunStore(out / stamp, f"sketch{number}")
        with Image.open(image_path) as image:
            original = image.copy()
        sketch = preprocess(original, store, cfg, source)
        intent = understand(models, sketch, typed, cfg, store)
        turn_upright(sketch, intent.up)
        kind_of_input = sketch_type(original, source, intent.drawing)
        print(f"\n[{image_path}] {intent.label} | {kind_of_input} | drawing {intent.drawing or '?'} | "
              f"kind {intent.kind or '?'} | up {intent.up or '?'} | typed '{typed}'", flush=True)
        rows, picked_now = [], None
        for name, (prompt, negative, scale, end) in variants(intent, typed, cfg).items():
            started = time.monotonic()
            prefix = f"{name.split()[0].strip(',')}_{len(rows)}_"
            result = models.run_worker("generate", "sdxl", store, **model_job(cfg), control_path=str(sketch.control_path),
                                       prompt=prompt, negative_prompt=negative, control_scale=scale, control_end=end,
                                       n=2, steps=int(g.steps), guidance_scale=float(g.guidance_scale),
                                       size=int(g.size), seed=int(g.seed), out_dir=str(store.dir), name_prefix=prefix)
            found = [Candidate(path=Path(p), seed=int(g.seed), index=i, round=0, control_scale=scale)
                     for i, p in enumerate(result["paths"])]
            verdict = check_images(sketch, intent, found, cfg, store)
            if verdict is None:
                said = "check did not run"
            elif verdict["best"] is None:
                said = f"GPT rejected both: {verdict['problems'][:80]}"
            else:
                said = f"GPT picked {'AB'[found.index(verdict['best'])]}"
            if name == "now":
                picked_now = (verdict or {}).get("best") or found[0]
            rows.append({"name": name, "prompt": prompt, "scale": scale, "end": end, "verdict": said,
                         "images": [(c.path, c.judged) for c in found]})
            print(f"  {name:28s} {time.monotonic() - started:5.1f} s  {said}", flush=True)
        target = out / f"{stamp}_sketch{number}_{(intent.label or 'object').replace(' ', '_')}.jpg"
        sheet(sketch.lineart_path, rows, target)
        print(f"  sheet: {target}", flush=True)
        summary.append({"sketch": str(image_path), "label": intent.label, "input": kind_of_input,
                        "drawing": intent.drawing,
                        "kind": intent.kind, "up": intent.up, "sheet": str(target),
                        "rows": [{k: v for k, v in row.items() if k != "images"} for row in rows]})
        if args.paint and number == 0 and picked_now is not None:
            paint_test(models, cfg, store, intent, picked_now.path, typed, stamp)
    (out / f"{stamp}_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nSheets and summary in {out}", flush=True)


if __name__ == "__main__":
    main()
