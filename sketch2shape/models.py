"""Model access for the pipeline.

The three heavy GPU steps (SDXL, Hunyuan3D-2 shape, Hunyuan3D-2 texture) each run in a fresh worker
process (sketch2shape/worker.py) pinned to its GPU with CUDA_VISIBLE_DEVICES. A worker loads its
model, does one job, writes its output files and exits, giving all of its memory back. Keeping the
models in one long-lived process ran out of Kaggle's 30 GB of CPU RAM; separate processes are the
memory pattern the Kaggle trial ran without problems, at the cost of about a minute of model loading
per run. It also means a crash inside one model cannot take down the app.

The small models stay in this process on the CPU: CLIP (naming and scoring) and background removal.
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class WorkerError(RuntimeError):
    """A GPU worker process failed; the message carries its error and the log path."""


class ModelManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self._clip = None
        self._bg = None

    # --------------------------------------------------------- GPU workers

    def gpu_index(self, role: str) -> int:
        return int(getattr(self.cfg.gpus, role))

    def run_worker(self, task: str, role: str, store, **job) -> dict:
        """Run one job in a fresh process on the GPU for `role` ("sdxl" or "hunyuan") and return its result."""
        tag = f"{task}_{uuid.uuid4().hex[:6]}"
        job_path = store.path(f"worker_{tag}_job.json")
        result_path = store.path(f"worker_{tag}_result.json")
        log_path = store.path(f"worker_{tag}.log")
        job_path.write_text(json.dumps({"task": task, **job}, default=str), encoding="utf-8")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.gpu_index(role)), PYTHONUNBUFFERED="1")
        try:
            with open(log_path, "w", encoding="utf-8") as log:
                subprocess.run([sys.executable, "-m", "sketch2shape.worker", str(job_path), str(result_path)],
                               env=env, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                               timeout=float(self.cfg.worker_timeout_s))
        except subprocess.TimeoutExpired:
            raise WorkerError(f"{task} worker timed out after {self.cfg.worker_timeout_s} s (log: {log_path})")
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
        if not result or not result.get("ok"):
            detail = (result or {}).get("error") or tail(log_path)
            raise WorkerError(f"{task} worker failed: {detail} (log: {log_path})")
        store.manifest.setdefault("workers", []).append(
            {"task": task, **{k: result[k] for k in ("load_s", "run_s", "peak_gb") if k in result}})
        return result

    # ------------------------------------------------- small models on CPU

    def clip(self):
        if self._clip is None:
            from transformers import CLIPModel, CLIPProcessor
            name = self.cfg.understand.clip_model
            self._clip = (CLIPModel.from_pretrained(name).eval(), CLIPProcessor.from_pretrained(name))
        return self._clip

    def bg_remover(self):
        if self._bg is None:
            from hy3dgen.rembg import BackgroundRemover
            self._bg = BackgroundRemover()
        return self._bg

    def free_memory(self) -> None:
        gc.collect()


def tail(path: Path, lines: int = 15) -> str:
    try:
        return "".join(Path(path).read_text(encoding="utf-8", errors="replace").splitlines(True)[-lines:])
    except OSError:
        return "no log"
