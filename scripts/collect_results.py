"""Collect the results of finished app runs for the README: one table row and one picture strip per run.

Reads runs/<id>/manifest.json (and the step images next to it); runs nothing on the GPU, so it can run
while the app is open.

    python scripts/collect_results.py --last 8            # the 8 newest finished runs
    python scripts/collect_results.py --runs 20261101-063012-ab12cd 20261101-064501-cd34ef

Writes results/app_runs.md, results/app_runs.json and results/sheets/<run>.png, and prints the table.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_batch import contact_sheet  # noqa: E402

from sketch2shape import load_config  # noqa: E402


def finished_runs(runs_dir: Path, wanted: list[str], last: int) -> list[Path]:
    """Run folders with a model, newest first (skipping the comparison script's runs)."""
    if wanted:
        return [runs_dir / name for name in wanted]
    found = [d for d in runs_dir.iterdir() if d.is_dir() and not d.name.startswith(("_", "compare"))
             and (d / "manifest.json").exists() and (d / "model.glb").exists()]
    return sorted(found, reverse=True)[:last]


def row_for(run: Path) -> dict:
    m = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    intent, res, inputs = m.get("intent") or {}, m.get("result") or {}, m.get("inputs") or {}
    timings = m.get("timings_s") or {}
    chosen = next((c for c in m.get("candidates", []) if c.get("path") == m.get("chosen_candidate")), {})
    return {
        "run": run.name, "object": intent.get("label"), "typed": inputs.get("prompt") or "",
        "input": inputs.get("source"), "drawing": intent.get("drawing") or "-", "kind": intent.get("kind") or "-",
        "named_by": intent.get("source"), "line_match": chosen.get("total"),
        "image_check": "picked" if chosen.get("judged") == "picked" else (chosen.get("judged") or "-"),
        "rounds": res.get("generation_rounds"), "texture": res.get("texture_mode"),
        "seconds": res.get("total_s"), "image_s": timings.get("generate"), "shape_s": timings.get("shape"),
        "texture_s": timings.get("texture"), "ok": res.get("ok"), "warnings": len(m.get("warnings", [])),
        "manifest": m,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="*", default=[], help="run folder names (default: the newest)")
    parser.add_argument("--last", type=int, default=8)
    parser.add_argument("--out", default="results")
    parser.add_argument("--config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out)
    (out / "sheets").mkdir(parents=True, exist_ok=True)

    rows = []
    for run in finished_runs(Path(cfg.runs_dir), args.runs, args.last):
        row = row_for(run)
        contact_sheet(run, row.pop("manifest"), out / "sheets" / f"{run.name}.png")
        rows.append(row)
    (out / "app_runs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    header = ["run", "object", "typed", "input", "drawing", "kind", "image_check", "line_match", "rounds",
              "texture", "image_s", "shape_s", "texture_s", "seconds"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(r[h]) if r[h] not in (None, "") else "-" for h in header) + " |" for r in rows]
    times = [r["seconds"] for r in rows if r["ok"] and r["seconds"]]
    if times:
        lines += ["", f"Finished runs: {len(times)}. Median time {statistics.median(times):.0f} s, "
                      f"fastest {min(times):.0f} s, slowest {max(times):.0f} s."]
    (out / "app_runs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nPicture strips (input, cleaned sketch, images with scores, input to 3D): {out / 'sheets'}")


if __name__ == "__main__":
    main()
