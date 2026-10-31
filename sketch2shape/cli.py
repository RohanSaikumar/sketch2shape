"""Command line: python -m sketch2shape.cli path/to/sketch.png --prompt "a red ceramic mug" """
from __future__ import annotations

import argparse
import json
import sys

from PIL import Image

from .config import load_config
from .pipeline import Sketch2Shape


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Turn a 2D sketch into a textured 3D model (.glb and .obj).")
    parser.add_argument("sketch", help="sketch image (png or jpg)")
    parser.add_argument("--prompt", default="", help="optional description: object, material, colour, style")
    parser.add_argument("--config", help="path to a config.yaml (default: the one in the repo)")
    parser.add_argument("--candidates", type=int, help="images generated per round (1 is faster)")
    parser.add_argument("--no-texture", action="store_true", help="skip texturing; use projected colours")
    parser.add_argument("--runs-dir", help="where run folders are written (default: runs/)")
    args = parser.parse_args(argv)

    overrides = {}
    if args.candidates:
        overrides["generate.candidates"] = args.candidates
    if args.no_texture:
        overrides["reconstruct.texture"] = False
    if args.runs_dir:
        overrides["runs_dir"] = args.runs_dir
    cfg = load_config(args.config, overrides)

    def on_event(state: str, outcome: str, elapsed: float) -> None:
        print(f"[{elapsed:6.1f} s] {state:<14} -> {outcome}", flush=True)

    result = Sketch2Shape(cfg).run(Image.open(args.sketch), prompt=args.prompt, source="upload", on_event=on_event)
    m = result.manifest
    print(json.dumps({
        "ok": result.ok,
        "message": result.message,
        "glb": str(result.glb_path) if result.glb_path else None,
        "obj_zip": str(result.obj_zip_path) if result.obj_zip_path else None,
        "object": (m.get("intent") or {}).get("label"),
        "texture": (m.get("result") or {}).get("texture_mode"),
        "total_s": (m.get("result") or {}).get("total_s"),
        "warnings": m.get("warnings"),
        "run_dir": str(result.run_dir),
    }, indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
