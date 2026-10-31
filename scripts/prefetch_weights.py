"""Download every model the pipeline uses, so the first run does not wait for downloads.

    python scripts/prefetch_weights.py

Run at the end of scripts/setup.sh. Only the files the pipeline loads are fetched (SDXL fp16 weights,
the ControlNet, the fp16-safe VAE, CLIP, the Hunyuan3D-2 subfolders in config.yaml, and rembg's
background-removal model). A failure here only means that model downloads on first use instead.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sketch2shape import load_config  # noqa: E402


def main() -> None:
    from huggingface_hub import snapshot_download
    cfg = load_config()
    g, r, u = cfg.generate, cfg.reconstruct, cfg.understand
    shape = r.shape_subfolder
    downloads = [
        (g.base_model, ["*.json", "*.txt", "tokenizer*/*", "*fp16.safetensors"]),
        (g.controlnet, ["*.json", "diffusion_pytorch_model.safetensors"]),
        (g.vae, ["*.json", "diffusion_pytorch_model.safetensors"]),
        (u.clip_model, ["*.json", "*.txt", "*.safetensors"]),
        (r.model, [f"{shape}/*", f"{shape.replace('-dit-', '-vae-')}/*",
                   "hunyuan3d-paint-v2-0-turbo/*", "hunyuan3d-delight-v2-0/*"]),
    ]
    for repo, patterns in downloads:
        started = time.time()
        try:
            snapshot_download(repo, allow_patterns=patterns)
            print(f"== {repo}: ready ({time.time() - started:.0f} s)", flush=True)
        except Exception as exc:
            print(f"== {repo}: not prefetched ({exc!r}); it will download on first use", flush=True)
    started = time.time()
    try:
        from rembg import new_session
        new_session()
        print(f"== rembg background-removal model: ready ({time.time() - started:.0f} s)", flush=True)
    except Exception as exc:
        print(f"== rembg model not prefetched ({exc!r})", flush=True)


if __name__ == "__main__":
    main()
