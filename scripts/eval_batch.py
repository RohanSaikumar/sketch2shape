"""Run every sketch in a folder through the pipeline and write the results table used in the README.

    python scripts/eval_batch.py examples/sketches --prompts examples/prompts.json --out results

prompts.json (optional) maps a file name to a description, e.g. {"mug.png": "a white ceramic mug"}.
Writes results/results.md, results/results.json and a copy of each GLB.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sketch2shape import Sketch2Shape, load_config  # noqa: E402
from sketch2shape.models import relaunch_with_lean_malloc  # noqa: E402

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp"}


def contact_sheet(run_dir: Path, manifest: dict, target: Path, tile: int = 256) -> None:
    """One strip per sketch: input, cleaned sketch, every candidate with its score (* = chosen), input to 3D."""
    chosen = manifest.get("chosen_candidate")
    items = [(run_dir / "01_input.png", "input"), (run_dir / "02_lineart.png", "cleaned")]
    items += [(run_dir / c["path"], f"{c['total']:.2f}" + (" *" if c["path"] == chosen else ""))
              for c in manifest.get("candidates", [])]
    items.append((run_dir / "06_prepared.png", "to 3D"))
    items = [(p, label) for p, label in items if p.exists()]
    sheet = Image.new("RGB", (tile * len(items), tile + 22), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (path, label) in enumerate(items):
        image = Image.open(path).convert("RGBA")
        flat = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image).convert("RGB")
        sheet.paste(flat.resize((tile, tile)), (i * tile, 22))
        draw.text((i * tile + 6, 5), label, fill="black")
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", nargs="?", default="examples/sketches")
    parser.add_argument("--prompts", help="JSON file mapping sketch file names to descriptions")
    parser.add_argument("--out", default="results")
    parser.add_argument("--config")
    args = parser.parse_args()
    relaunch_with_lean_malloc()

    import torch
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU found. On Kaggle: Session options, Accelerator 'GPU T4 x2', then run again.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8")) if args.prompts else {}
    cfg = load_config(args.config, {"runs_dir": str(out / "runs")})
    pipeline = Sketch2Shape(cfg)

    rows = []
    sketches = sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in IMAGE_TYPES)
    for sketch in sketches:
        print(f"== {sketch.name}", flush=True)
        result = pipeline.run(Image.open(sketch), prompt=prompts.get(sketch.name, ""), source="upload")
        m = result.manifest
        intent, res, export = m.get("intent") or {}, m.get("result") or {}, m.get("export") or {}
        best = max(m.get("candidates", []), key=lambda c: c["total"], default={})
        if result.glb_path:
            shutil.copy(result.glb_path, out / f"{sketch.stem}.glb")
        contact_sheet(Path(result.run_dir), m, out / "sheets" / f"{sketch.stem}.png")
        rows.append({
            "sketch": sketch.name, "prompt": prompts.get(sketch.name, ""), "ok": result.ok,
            "object": intent.get("label"), "named_by": intent.get("source"),
            "fidelity": best.get("total"), "rounds": res.get("generation_rounds"),
            "shape_attempts": res.get("shape_attempts"), "texture": res.get("texture_mode"),
            "faces": export.get("faces"), "seconds": res.get("total_s"),
            "warnings": len(m.get("warnings", [])), "run": m.get("run_id"),
        })
        print(json.dumps(rows[-1]), flush=True)

    (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    header = ["sketch", "prompt", "ok", "object", "named_by", "fidelity", "rounds", "texture", "faces", "seconds"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(r[h]) if r[h] not in (None, "") else "-" for h in header) + " |" for r in rows]
    times = [r["seconds"] for r in rows if r["ok"] and r["seconds"]]
    if times:
        lines.append("")
        lines.append(f"Successful runs: {len(times)}/{len(rows)}. Median time: {statistics.median(times):.0f} s, "
                     f"slowest: {max(times):.0f} s.")
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
