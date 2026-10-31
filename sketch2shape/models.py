"""Loads each model once, on the GPU chosen in config.yaml, and keeps GPU memory within a T4's 15 GB.

Measured on Kaggle's two T4s: SDXL + ControlNet generating two images peaks at 13.4 GB, so GPU 0 holds
SDXL and CLIP only. On GPU 1, Hunyuan3D-2's shape model peaks at up to 9.4 GB and its texture model at
12.8 GB, so the two cannot sit there together: they take turns, and the idle one waits in CPU memory
(a few seconds to move, against about a minute to reload from disk). If a model cannot be moved it is
released and reloaded when next needed.

On a single GPU, SDXL moves to the CPU before the 3D models load, and the 3D models are released
before SDXL runs again. Heavy imports happen inside the methods so the rest of the package (and the
offline tests) works without a GPU.
"""
from __future__ import annotations

import gc
from contextlib import contextmanager


class ModelManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self._sdxl = None
        self._clip = None
        self._bg = None
        self._shape = None
        self._paint = None
        self._parked: set[str] = set()       # 3D models currently waiting in CPU memory
        self._paint_offloaded = False
        self.shape_steps = int(cfg.reconstruct.shape_steps)
        self.shape_variant = None

    # ------------------------------------------------------------ placement

    def gpu(self, role: str) -> int:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU found. Sketch2Shape needs an NVIDIA GPU (tested on Kaggle 'GPU T4 x2').")
        index = int(getattr(self.cfg.gpus, role))
        return index if index < torch.cuda.device_count() else 0

    @property
    def shared_gpu(self) -> bool:
        return self.gpu("sdxl") == self.gpu("hunyuan")

    @contextmanager
    def on_hunyuan_gpu(self):
        """Hunyuan3D-2's texture code allocates on the current CUDA device, so make that the 3D GPU."""
        import torch
        with torch.cuda.device(self.gpu("hunyuan")):
            yield

    def free_memory(self) -> None:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ---------------------------------------------------------- 2D models

    def sdxl(self):
        if self._sdxl is None:
            import torch
            from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetPipeline
            g = self.cfg.generate
            controlnet = ControlNetModel.from_pretrained(g.controlnet, torch_dtype=torch.float16)
            vae = AutoencoderKL.from_pretrained(g.vae, torch_dtype=torch.float16)
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                g.base_model, controlnet=controlnet, vae=vae,
                torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
            if hasattr(pipe.vae, "enable_slicing"):    # newer diffusers removed pipe.enable_vae_slicing()
                pipe.vae.enable_slicing()
            pipe.set_progress_bar_config(disable=True)
            self._sdxl = pipe
        if self.shared_gpu:
            self._release_3d()
        self._sdxl.to(f"cuda:{self.gpu('sdxl')}")
        return self._sdxl

    def clip(self):
        if self._clip is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            name = self.cfg.understand.clip_model
            model = CLIPModel.from_pretrained(name, torch_dtype=torch.float16).to(f"cuda:{self.gpu('sdxl')}").eval()
            self._clip = (model, CLIPProcessor.from_pretrained(name))
        return self._clip

    def bg_remover(self):
        if self._bg is None:
            from hy3dgen.rembg import BackgroundRemover
            self._bg = BackgroundRemover()
        return self._bg

    # ---------------------------------------------------------- 3D models

    def shape(self):
        if self.shared_gpu:
            self._offload_sdxl()
        self._evict("paint")
        if self._shape is None:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            r = self.cfg.reconstruct
            device = f"cuda:{self.gpu('hunyuan')}"
            with self.on_hunyuan_gpu():
                try:
                    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                        r.model, subfolder=r.shape_subfolder, use_safetensors=True, device=device)
                    if r.flashvdm:
                        pipe.enable_flashvdm()
                    self.shape_steps, self.shape_variant = int(r.shape_steps), r.shape_subfolder
                except Exception as exc:   # turbo weights unavailable: the base model needs ~50 steps
                    print(f"[models] {r.shape_subfolder} failed to load ({exc!r}); using the base shape model")
                    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(r.model, device=device)
                    self.shape_steps, self.shape_variant = 50, "hunyuan3d-dit-v2-0"
            self._shape = pipe
        elif "shape" in self._parked:
            self._move("shape", f"cuda:{self.gpu('hunyuan')}")
        return self._shape

    def paint(self):
        if self.shared_gpu:
            self._offload_sdxl()
        self._evict("shape")
        if self._paint is None:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            with self.on_hunyuan_gpu():
                self._paint = Hunyuan3DPaintPipeline.from_pretrained(self.cfg.reconstruct.model)
        elif "paint" in self._parked:
            self._move("paint", f"cuda:{self.gpu('hunyuan')}")
        return self._paint

    def paint_offload(self) -> None:
        """Keep texture sub-models on the CPU and move each to the GPU only while it runs."""
        with self.on_hunyuan_gpu():
            try:
                self._paint.enable_model_cpu_offload(gpu_id=self.gpu("hunyuan"))
            except TypeError:
                self._paint.enable_model_cpu_offload()
        self._paint_offloaded = True

    def _evict(self, name: str) -> None:
        """Make room on the 3D GPU: park the named model in CPU memory, or release it if it cannot move."""
        if getattr(self, f"_{name}") is None or name in self._parked:
            return
        if name == "paint" and self._paint_offloaded:     # already lives on the CPU between calls
            return
        if self.cfg.reconstruct.park_on_cpu:
            try:
                self._move(name, "cpu")
                self._parked.add(name)
                self.free_memory()
                return
            except Exception as exc:
                print(f"[models] could not move the {name} model to the CPU ({exc!r}); releasing it instead")
        setattr(self, f"_{name}", None)
        self.free_memory()

    def _move(self, name: str, device: str) -> None:
        if name == "shape":
            self._shape.to(device)
        else:
            parts = [getattr(m, "pipeline", None) for m in getattr(self._paint, "models", {}).values()]
            parts = [p for p in parts if p is not None]
            if not parts:
                raise RuntimeError("no movable parts found in the texture pipeline")
            for part in parts:
                part.to(device)
        if device == "cpu":
            self._parked.add(name)
        else:
            self._parked.discard(name)

    def _release_3d(self) -> None:
        if self._shape is not None or self._paint is not None:
            self._shape = None
            self._paint = None
            self._parked.clear()
            self._paint_offloaded = False
            self.free_memory()

    def _offload_sdxl(self) -> None:
        if self._sdxl is not None:
            self._sdxl.to("cpu")
            self.free_memory()
