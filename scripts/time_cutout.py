"""Why does the cut-out take ~28 s per image in the app on Kaggle? Hunyuan's BackgroundRemover calls rembg's
new_session() without a model name, so the model is whatever the installed rembg defaults to: u2net in
older versions, BRIA RMBG-2.0 (a large 1024 px model, CC BY-NC 4.0) in newer ones. This prints the
rembg version, its default model and the onnxruntime devices, then times a cut-out in a fresh process
with the app's settings, for the default model and for u2net.

    python scripts/time_cutout.py [image.png]     # default: the chosen image of the newest run
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def child(image_path: str, model: str) -> None:
    import inspect
    from importlib import metadata

    import onnxruntime
    from PIL import Image
    from rembg import new_session, remove

    default = inspect.signature(new_session).parameters["model_name"].default
    started = time.time()
    session = new_session() if model == "default" else new_session(model)
    load_s = time.time() - started
    remove(Image.new("RGB", (64, 64), "white"), session=session)     # warm-up, as the app does at start
    image = Image.open(image_path).convert("RGB")
    times = []
    for _ in range(2):
        started = time.time()
        remove(image, session=session, bgcolor=[255, 255, 255, 0])   # the call Hunyuan's remover makes
        times.append(round(time.time() - started, 1))
    print(json.dumps({"model": default if model == "default" else model, "load_s": round(load_s, 1),
                      "per_image_s": times, "rembg": metadata.version("rembg"), "ort": onnxruntime.__version__,
                      "devices": onnxruntime.get_available_providers(),
                      "cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()}))


def newest_image() -> Path:
    from sketch2shape import load_config
    runs = sorted((d for d in Path(load_config().runs_dir).iterdir()
                   if d.is_dir() and (d / "manifest.json").exists()), reverse=True)
    for run in runs:
        chosen = json.loads((run / "manifest.json").read_text(encoding="utf-8")).get("chosen_candidate")
        if chosen and (run / chosen).exists():
            return run / chosen
    sys.exit("No run with a generated image found; pass an image path.")


def main() -> None:
    if sys.argv[1:2] == ["--child"]:
        child(sys.argv[2], sys.argv[3])
        return
    from sketch2shape.models import LEAN_MALLOC
    image = Path(sys.argv[1]) if len(sys.argv) > 1 else newest_image()
    print(f"image: {image}", flush=True)
    for model in ("default", "u2net"):              # "default" = what the app uses today
        out = subprocess.run([sys.executable, __file__, "--child", str(image), model],
                             env={**os.environ, **LEAN_MALLOC}, capture_output=True, text=True, cwd=ROOT)
        lines = [line for line in out.stdout.splitlines() if line.startswith("{")]
        print(f"{model:8s} {lines[-1] if lines else 'failed: ' + out.stderr[-400:]}", flush=True)


if __name__ == "__main__":
    main()
