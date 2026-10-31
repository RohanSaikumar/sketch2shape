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
        self._text_cache: dict = {}

    # --------------------------------------------------------- GPU workers

    def gpu_index(self, role: str) -> int:
        """The configured GPU for this role, or GPU 0 when the machine has fewer GPUs (e.g. one P100)."""
        index = int(getattr(self.cfg.gpus, role))
        return index if index < gpu_count() else 0

    def run_worker(self, task: str, role: str, store, **job) -> dict:
        """Run one job in a fresh process on the GPU for `role` ("sdxl" or "hunyuan") and return its result."""
        tag = f"{task}_{uuid.uuid4().hex[:6]}"
        job_path = store.path(f"worker_{tag}_job.json")
        result_path = store.path(f"worker_{tag}_result.json")
        log_path = store.path(f"worker_{tag}.log")
        job_path.write_text(json.dumps({"task": task, **job}, default=str), encoding="utf-8")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.gpu_index(role)), PYTHONUNBUFFERED="1",
                   **LEAN_MALLOC)
        free_memory()                       # give this process's spare memory back before a worker loads
        with open(log_path, "w", encoding="utf-8") as log:
            # own process group, so anything the worker starts is cleaned up with it
            proc = subprocess.Popen([sys.executable, "-m", "sketch2shape.worker", str(job_path), str(result_path)],
                                    env=env, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                proc.wait(timeout=float(self.cfg.worker_timeout_s))
            except subprocess.TimeoutExpired:
                raise WorkerError(f"{task} worker timed out after {self.cfg.worker_timeout_s} s (log: {log_path})")
            finally:
                kill_group(proc)
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
        if not result or not result.get("ok"):
            killed = " (killed: probably out of CPU RAM)" if proc.returncode == -9 else ""
            detail = (result or {}).get("error") or tail(log_path)
            raise WorkerError(f"{task} worker failed{killed}: {detail} (log: {log_path})")
        store.manifest.setdefault("workers", []).append(
            {"task": task, **{k: result[k] for k in ("load_s", "run_s", "peak_gb", "max_rss_gb") if k in result}})
        return result

    # ------------------------------------------------- small models on CPU

    def clip(self):
        if self._clip is None:
            from transformers import CLIPModel, CLIPProcessor
            name = self.cfg.understand.clip_model
            self._clip = (CLIPModel.from_pretrained(name).eval(), CLIPProcessor.from_pretrained(name))
        return self._clip

    def clip_text_features(self, texts: list[str]):
        """Normalised CLIP text embeddings, computed once per text and reused (CPU text encoding is slow)."""
        import torch
        model, processor = self.clip()
        missing = [t for t in dict.fromkeys(texts) if t not in self._text_cache]
        if missing:
            inputs = processor(text=missing, return_tensors="pt", padding=True)
            with torch.no_grad():       # text tower + projection: stable across transformers versions
                pooled = model.text_model(input_ids=inputs["input_ids"],
                                          attention_mask=inputs["attention_mask"]).pooler_output
                features = model.text_projection(pooled)
            features = features / features.norm(dim=-1, keepdim=True)
            self._text_cache.update(zip(missing, features))
        return torch.stack([self._text_cache[t] for t in texts])

    def clip_image_features(self, images):
        import torch
        model, processor = self.clip()
        with torch.no_grad():
            pixels = processor(images=images, return_tensors="pt")["pixel_values"]
            features = model.visual_projection(model.vision_model(pixel_values=pixels).pooler_output)
        return features / features.norm(dim=-1, keepdim=True)

    def bg_remover(self):
        if self._bg is None:
            from hy3dgen.rembg import BackgroundRemover
            self._bg = BackgroundRemover()
        return self._bg

    def free_memory(self) -> None:
        free_memory()


# glibc keeps freed memory inside the process by default; PyTorch model loading frees a lot of it, so a
# worker's RAM keeps climbing. These limits return large freed blocks to the system straight away.
LEAN_MALLOC = {"MALLOC_ARENA_MAX": "2", "MALLOC_MMAP_THRESHOLD_": "131072", "MALLOC_TRIM_THRESHOLD_": "131072"}


def relaunch_with_lean_malloc() -> None:
    """Restart the current script with LEAN_MALLOC set (the allocator reads it only at start-up). Linux only."""
    if sys.platform.startswith("linux") and any(os.environ.get(k) != v for k, v in LEAN_MALLOC.items()):
        os.execve(sys.executable, [sys.executable, *sys.argv],
                  {**os.environ, **LEAN_MALLOC, "PYTHONUNBUFFERED": "1"})


def stop_leftover_workers() -> None:
    """Stop GPU workers left running by an app or batch that was interrupted (they hold GPU and RAM). Linux."""
    if sys.platform.startswith("linux"):
        # "[.]" so the pattern never matches a shell whose own command line contains it
        found = subprocess.run(["pkill", "-9", "-f", "sketch2shape[.]worker"], capture_output=True).returncode == 0
        if found:
            print("Stopped GPU worker(s) left over from an interrupted run.", flush=True)


def free_memory() -> None:
    """Collect garbage and hand freed heap memory back to the operating system (Linux)."""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


_GPU_COUNT = None


def gpu_count() -> int:
    """Number of NVIDIA GPUs, from nvidia-smi so this process never opens a CUDA context."""
    global _GPU_COUNT
    if _GPU_COUNT is None:
        try:
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
            _GPU_COUNT = sum(line.startswith("GPU ") for line in out.splitlines())
        except (OSError, subprocess.SubprocessError):
            _GPU_COUNT = 0
    return _GPU_COUNT


def kill_group(proc: subprocess.Popen) -> None:
    """Stop the worker and anything it started (Linux; elsewhere just the worker)."""
    if proc.poll() is None:
        proc.kill()
        proc.wait()
    if hasattr(os, "killpg"):
        try:
            os.killpg(proc.pid, 9)
        except (ProcessLookupError, PermissionError):
            pass


def rss_gb() -> float | None:
    """This process's resident memory in GB (Linux), for the run log."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 2**20, 2)
    except OSError:
        pass
    return None


def tail(path: Path, lines: int = 15) -> str:
    try:
        return "".join(Path(path).read_text(encoding="utf-8", errors="replace").splitlines(True)[-lines:])
    except OSError:
        return "no log"
