"""Model access for the pipeline.

The three heavy GPU steps (SDXL, Hunyuan3D-2 shape, Hunyuan3D-2 texture) run in worker processes
(sketch2shape/worker.py) pinned to their GPU with CUDA_VISIBLE_DEVICES, so the big models never share
CPU RAM: keeping them in one long-lived process ran out of Kaggle's 30 GB, and a crash inside one model
cannot take down the app. Loading the models took about half of every run, so with two GPUs:

  - SDXL stays loaded between runs in a worker of its own on GPU 0, which nothing else uses;
  - the shape model starts loading on GPU 1 while the image is still being made; that worker exits after
    its one job, so the texture model then has the GPU and the RAM to itself;
  - the texture model loads in a fresh worker that exits when it is done, as before.

A worker loads early only when no other worker is loading and enough RAM is free, because each Hunyuan
model briefly needs about 15 GB of RAM while it loads. With one GPU every job uses a fresh worker.
Same models, settings and seeds either way, so the output does not change.

The small models stay in this process on the CPU: CLIP (naming and scoring) and background removal.
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The job fields that choose the model; the rest describe one job, so a loaded worker can take any job
# whose model fields match.
LOAD_KEYS = {"ping": (), "generate": ("base_model", "controlnet", "vae"),
             "shape": ("model", "subfolder", "flashvdm"), "paint": ("model",)}


class WorkerError(RuntimeError):
    """A GPU worker failed or was cancelled; the message carries its error and the log path."""


class ServingWorker:
    """A worker process that loads its model once, then runs each job it is sent (worker.py --serve)."""

    def __init__(self, task: str, gpu: int, load: dict, folder: Path):
        tag = f"{task}_{uuid.uuid4().hex[:6]}"
        self.task, self.load, self.jobs = task, load, 0
        self.ready_path = Path(folder) / f"serve_{tag}_ready.json"
        self.log_path = Path(folder) / f"serve_{tag}.log"
        self.cancel_path: Path | None = None
        load_path = Path(folder) / f"serve_{tag}_load.json"
        load_path.write_text(json.dumps({"task": task, **load}, default=str), encoding="utf-8")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", **LEAN_MALLOC)
        free_memory()                       # give this process's spare memory back before a worker loads
        with open(self.log_path, "w", encoding="utf-8") as log:
            # own process group, so anything the worker starts is cleaned up with it
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "sketch2shape.worker", "--serve", str(load_path), str(self.ready_path)],
                env=env, cwd=PROJECT_ROOT, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                text=True, start_new_session=True)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def loading(self) -> bool:
        return self.alive() and not self.ready_path.exists()

    def load_s(self) -> float | None:
        try:
            return json.loads(self.ready_path.read_text(encoding="utf-8")).get("load_s")
        except (OSError, ValueError):
            return None

    def run(self, job: dict, store, timeout_s: float, cancel: threading.Event, last: bool = False) -> dict:
        """Send one job and wait for its result, including any loading still under way. last=True closes the
        worker's input, so it exits after this job and frees its GPU."""
        tag = f"{self.task}_{uuid.uuid4().hex[:6]}"
        job_path, result_path = store.path(f"worker_{tag}_job.json"), store.path(f"worker_{tag}_result.json")
        self.cancel_path = store.path(f"worker_{tag}.cancel")
        job_path.write_text(json.dumps({"task": self.task, **job, "cancel_path": str(self.cancel_path)},
                                       default=str), encoding="utf-8")
        try:
            self.proc.stdin.write(f"{job_path}\t{result_path}\n")
            self.proc.stdin.flush()
            if last:
                self.proc.stdin.close()
        except (OSError, ValueError):       # the worker has already exited; the loop below says why
            pass
        sent = time.monotonic()
        while not result_path.exists():
            if cancel.is_set():
                self.interrupt()
                raise WorkerError("Cancelled.")
            if not self.alive() and not result_path.exists():
                raise WorkerError(self.failure())
            if time.monotonic() - sent > timeout_s:
                raise WorkerError(f"{self.task} worker timed out after {timeout_s:.0f} s (log: {self.log_path})")
            time.sleep(0.2)
        if cancel.is_set():                 # the job ended because it was cancelled, or just as it was
            raise WorkerError("Cancelled.")
        self.jobs += 1
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["waited_s"] = round(time.monotonic() - sent, 1)
        return result

    def interrupt(self) -> None:
        """Ask the job in progress to stop at its next step; SDXL stops within a second and stays loaded."""
        if self.cancel_path is not None:
            try:
                self.cancel_path.touch()
            except OSError:
                pass

    def failure(self) -> str:
        try:
            detail = json.loads(self.ready_path.read_text(encoding="utf-8")).get("error")
        except (OSError, ValueError):
            detail = None
        killed = " (killed: probably out of CPU RAM)" if self.proc.returncode == -9 else ""
        return f"{self.task} worker failed{killed}: {detail or tail(self.log_path)} (log: {self.log_path})"

    def stop(self, wait_s: float = 30.0) -> None:
        """Close the worker's input so it exits after any job in progress, and kill it after wait_s."""
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            pass
        kill_group(self.proc)


class ModelManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self._clip = None
        self._bg = None
        self._text_cache: dict = {}
        self._small_lock = threading.Lock()     # the CPU models load once even if warm-up and a run overlap
        self._lock = threading.RLock()          # guards the worker tables below
        self._kept: dict[str, ServingWorker] = {}       # task -> worker that stays loaded between runs
        self._early: dict[str, ServingWorker] = {}      # task -> worker loaded early for a later step of this run
        self._waiting: dict[str, tuple] = {}            # task -> (role, store, load) to start once it is safe
        self._fresh: set[subprocess.Popen] = set()      # one-job workers running now
        self._cancel = threading.Event()

    # --------------------------------------------------------- GPU workers

    def gpu_index(self, role: str) -> int:
        """The configured GPU for this role, or GPU 0 when the machine has fewer GPUs (e.g. one P100)."""
        index = int(getattr(self.cfg.gpus, role))
        return index if index < gpu_count() else 0

    def separate_gpus(self) -> bool:
        """SDXL and Hunyuan on different GPUs (Kaggle's T4 x2): only then may a worker stay loaded or load early."""
        return self.gpu_index("sdxl") != self.gpu_index("hunyuan")

    def run_worker(self, task: str, role: str, store, **job) -> dict:
        """Run one job on the GPU for `role` ("sdxl" or "hunyuan") and return its result: in the worker
        loaded early or kept loaded for this task if there is one, otherwise in a fresh worker process."""
        if self._cancel.is_set():
            raise WorkerError("Cancelled.")
        load = {key: job[key] for key in LOAD_KEYS.get(task, ()) if key in job}
        timeout = float(self.cfg.worker_timeout_s)
        with self._lock:
            self._waiting.pop(task, None)       # its step has come: too late to load it early
            early = self._early.get(task)
        if early is not None:
            try:
                if early.alive() and early.load == load:
                    return self._record(store, task, "loaded early", early,
                                        early.run(job, store, timeout, self._cancel, last=True))
            finally:
                with self._lock:
                    self._early.pop(task, None)
                early.stop(wait_s=0 if self._cancel.is_set() else 30)   # its GPU must be free for the next step
        if task in self.cfg.workers.keep_loaded and self.separate_gpus():
            kept = self._kept_worker(task, role, load)
            try:
                return self._record(store, task, "kept loaded", kept, kept.run(job, store, timeout, self._cancel))
            finally:
                if not kept.alive():            # crashed: a new one is started next time
                    with self._lock:
                        if self._kept.get(task) is kept:
                            del self._kept[task]
        return self._record(store, task, "fresh", None, self._run_fresh(task, role, store, job))

    def _record(self, store, task: str, how: str, worker, result: dict) -> dict:
        """Check a job's result and add it to the run's worker log."""
        if not result.get("ok"):
            log = worker.log_path if worker is not None else store.dir
            raise WorkerError(f"{task} worker failed: {result.get('error')} (log: {log})")
        entry = {"task": task, "how": how}
        if worker is not None:                  # the model's load time is counted once, on its first job
            entry["load_s"] = worker.load_s() if worker.jobs == 1 else 0.0
        entry.update({k: result[k] for k in ("load_s", "waited_s", "run_s", "peak_gb", "max_rss_gb") if k in result})
        store.manifest.setdefault("workers", []).append(entry)
        return result

    def _run_fresh(self, task: str, role: str, store, job: dict) -> dict:
        """One job in a new worker process that loads the model, runs the job and exits."""
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
        with self._lock:
            self._fresh.add(proc)
        try:
            deadline = time.monotonic() + float(self.cfg.worker_timeout_s)
            while proc.poll() is None:
                if self._cancel.is_set():
                    raise WorkerError("Cancelled.")
                if time.monotonic() > deadline:
                    raise WorkerError(f"{task} worker timed out after {self.cfg.worker_timeout_s} s (log: {log_path})")
                time.sleep(0.2)
        finally:
            with self._lock:
                self._fresh.discard(proc)
            kill_group(proc)
        if self._cancel.is_set():           # killed by cancel() between two checks of the loop above
            raise WorkerError("Cancelled.")
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
        if not result or not result.get("ok"):
            killed = " (killed: probably out of CPU RAM)" if proc.returncode == -9 else ""
            detail = (result or {}).get("error") or tail(log_path)
            raise WorkerError(f"{task} worker failed{killed}: {detail} (log: {log_path})")
        return result

    def _kept_worker(self, task: str, role: str, load: dict) -> ServingWorker:
        with self._lock:
            kept = self._kept.get(task)
            if kept is not None and (not kept.alive() or kept.load != load):
                kept.stop(wait_s=0)
                kept = None
            if kept is None:
                folder = Path(self.cfg.runs_dir) / "_workers"   # outlives runs; past_runs() skips it
                folder.mkdir(parents=True, exist_ok=True)
                kept = self._kept[task] = ServingWorker(task, self.gpu_index(role), load, folder)
            return kept

    def load_early(self, task: str, role: str, store, **load) -> None:
        """Start loading the model for a later step of this run on its own idle GPU, as soon as no other
        worker is loading and enough RAM is free; run_worker() then hands the step's job to it."""
        if role == "sdxl" or not self.separate_gpus():
            return
        with self._lock:
            if task in self._early or task in self._waiting:
                return
            self._waiting[task] = (role, store, load)
        threading.Thread(target=self._start_waiting, daemon=True).start()

    def _start_waiting(self) -> None:
        while not self._cancel.is_set():
            with self._lock:
                if not self._waiting:
                    return
                busy = bool(self._fresh) or any(w.loading() for w in [*self._kept.values(), *self._early.values()])
                free = free_ram_gb()
                if not busy and (free is None or free >= float(self.cfg.workers.min_free_ram_gb)):
                    for task, (role, store, load) in self._waiting.items():
                        self._early[task] = ServingWorker(task, self.gpu_index(role), load, store.dir)
                        store.manifest.setdefault("loaded_early", []).append({"task": task, "free_ram_gb": free})
                    self._waiting.clear()
                    return
            time.sleep(0.5)

    def warm_up(self) -> None:
        """At app start: begin loading SDXL into its kept worker, and the CPU models in a background thread,
        so the first run does not wait for them."""
        from .stages.generate import model_job

        if "generate" in self.cfg.workers.keep_loaded and self.separate_gpus():
            self._kept_worker("generate", "sdxl", model_job(self.cfg))
        threading.Thread(target=self._warm_small_models, daemon=True).start()

    def _warm_small_models(self) -> None:
        from PIL import Image

        from .stages.understand import with_article
        from .vocab import OBJECTS
        try:
            blank = Image.new("RGB", (64, 64), "white")
            self.bg_remover()(blank)                    # the first cut-out also sets up onnxruntime
            self.clip_image_features([blank])
            for template in ("a photo of {}", "a black and white sketch of {}"):   # scoring; naming without a key
                self.clip_text_features([template.format(with_article(name)) for name in OBJECTS])
        except Exception as exc:                        # only a head start: a run loads what it needs anyway
            print(f"[warm-up] CPU models not preloaded: {exc!r}", flush=True)

    def new_run(self) -> None:
        self._cancel.clear()

    def end_run(self) -> None:
        """Stop workers loaded early for steps the run never reached (it failed or was cancelled first)."""
        with self._lock:
            self._waiting.clear()
            early = list(self._early.values())
            self._early.clear()
        for worker in early:
            worker.stop(wait_s=0)

    def cancel(self) -> None:
        """Stop the current run's GPU work now (called from another thread). Workers started for this run
        are killed; the kept SDXL worker only drops its current job and stays loaded."""
        self._cancel.set()
        with self._lock:
            self._waiting.clear()
            early, fresh, kept = list(self._early.values()), list(self._fresh), list(self._kept.values())
        for worker in early:
            worker.stop(wait_s=0)
        for proc in fresh:
            kill_group(proc)
        for worker in kept:
            worker.interrupt()

    # ------------------------------------------------- small models on CPU

    def clip(self):
        with self._small_lock:
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
        with self._small_lock:
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


def free_ram_gb() -> float | None:
    """RAM this container can still use, in GB: its cgroup limit minus its working set (reclaimable file
    cache not counted), else MemAvailable. None where neither can be read (e.g. Windows)."""
    cgroups = (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.stat",
                "inactive_file"),
               ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes",
                "/sys/fs/cgroup/memory/memory.stat", "total_inactive_file"))
    for limit_file, usage_file, stat_file, inactive in cgroups:
        try:
            limit = Path(limit_file).read_text().strip()
            if limit == "max" or int(limit) >= 2**60:
                continue
            stat = dict(line.split()[:2] for line in Path(stat_file).read_text().splitlines() if line.strip())
            used = int(Path(usage_file).read_text()) - int(stat.get(inactive, 0))
            return round((int(limit) - used) / 2**30, 1)
        except (OSError, ValueError):
            continue
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return round(int(line.split()[1]) / 2**20, 1)
    except OSError:
        pass
    return None


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
