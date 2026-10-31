"""Small records passed between pipeline stages. Images and meshes stay on disk; these hold paths."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SketchAsset:
    lineart_path: Path      # black strokes on white, square, centred
    control_path: Path      # the image fed to the ControlNet
    mask_path: Path         # filled silhouette of the sketch
    stroke_ratio: float     # share of stroke pixels in the original input
    source: str             # "canvas" or "upload"


@dataclass
class Intent:
    label: str | None       # what the object is, if known
    confidence: float
    source: str             # "llm", "clip" or "none"
    prompt: str
    negative_prompt: str
    details: str = ""
    drawing: str = ""       # "clean" or "rough" (the LLM's judgement of the sketch); "" when unknown
    kind: str = ""          # "object", or "scene" (made as a miniature diorama); "" when unknown
    up: str = ""            # which edge the top of the drawing pointed to before it was turned upright


@dataclass
class Candidate:
    path: Path
    seed: int
    index: int
    round: int
    control_scale: float
    scores: dict = field(default_factory=dict)
    total: float = 0.0
    cutout_path: Path | None = None
    judged: str = ""        # the vision-LLM image check: "picked", "rejected", or "" when it did not run


@dataclass
class PreparedImage:
    rgba_path: Path
    mask_ratio: float
    method: str             # "rembg" or "colour-threshold"


@dataclass
class MeshReport:
    ok: bool
    checks: dict
    problems: list[str]


@dataclass
class ExportBundle:
    glb_path: Path
    obj_zip_path: Path | None
    reload_ok: bool
    faces: int
    textured: bool
