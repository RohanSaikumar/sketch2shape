"""The run controller: a small explicit state machine that calls the stages and owns every retry decision.

Stages never call each other and never decide what runs next. Each handler below runs one stage and
returns an outcome word; TRANSITIONS maps (state, outcome) to the next state. Loops are bounded by
attempt counters and by a global time budget, and once any mesh exists a failure exports the best
result so far instead of returning nothing.
"""
from __future__ import annotations

import platform
import time
import traceback
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from .config import load_config
from .contracts import Candidate, ExportBundle, Intent, PreparedImage, SketchAsset
from .models import ModelManager, free_memory, rss_gb
from .runstore import RunStore
from .stages.export import export_mesh
from .stages.generate import generate
from .stages.prepare import prepare
from .stages.preprocess import EmptySketchError, preprocess
from .stages.reconstruct import make_shape, paint_mesh, project_colours
from .stages.score import score_candidates
from .stages.understand import understand
from .stages.validate import check_mesh, drop_fragments


class State(str, Enum):
    PREPROCESS = "preprocess"
    UNDERSTAND = "understand"
    GENERATE = "generate"
    SCORE = "score"
    PREPARE = "prepare"
    SHAPE = "shape"
    CHECK_SHAPE = "check_shape"
    TEXTURE = "texture"
    CHECK_TEXTURE = "check_texture"
    EXPORT = "export"
    DONE = "done"
    FAILED = "failed"


TERMINAL = {State.DONE, State.FAILED}

TRANSITIONS: dict[tuple[State, str], State] = {
    (State.PREPROCESS, "ok"): State.UNDERSTAND,
    (State.PREPROCESS, "empty"): State.FAILED,
    (State.UNDERSTAND, "ok"): State.GENERATE,
    (State.GENERATE, "ok"): State.SCORE,
    (State.SCORE, "pass"): State.PREPARE,
    (State.SCORE, "retry"): State.GENERATE,        # gate 1: new seeds, stronger sketch conditioning
    (State.SCORE, "give_up"): State.PREPARE,       # rounds or time spent: best candidate, flagged
    (State.PREPARE, "ok"): State.SHAPE,
    (State.SHAPE, "ok"): State.CHECK_SHAPE,
    (State.SHAPE, "retry"): State.SHAPE,
    (State.SHAPE, "use_best"): State.TEXTURE,
    (State.SHAPE, "error"): State.FAILED,
    (State.CHECK_SHAPE, "pass"): State.TEXTURE,
    (State.CHECK_SHAPE, "retry"): State.SHAPE,     # gate 2: new seed
    (State.CHECK_SHAPE, "accept_flagged"): State.TEXTURE,
    (State.TEXTURE, "ok"): State.CHECK_TEXTURE,
    (State.TEXTURE, "fallback"): State.EXPORT,     # colours projected from the image instead
    (State.CHECK_TEXTURE, "pass"): State.EXPORT,
    (State.CHECK_TEXTURE, "fallback"): State.EXPORT,
    (State.EXPORT, "ok"): State.DONE,
}


@dataclass
class RunContext:
    store: RunStore
    image: object
    prompt: str
    source: str
    budget_s: float
    control_scale: float
    candidates_n: int
    started: float = field(default_factory=time.monotonic)
    sketch: SketchAsset | None = None
    intent: Intent | None = None
    candidates: list[Candidate] = field(default_factory=list)
    best: Candidate | None = None
    prepared: PreparedImage | None = None
    rounds: int = 0
    shape_attempts: int = 0
    mesh: object = None                 # current untextured shape
    best_shape: object = None
    best_shape_problems: int = 10**6
    textured: object = None
    final_mesh: object = None
    texture_mode: str = "none"
    bundle: ExportBundle | None = None
    message: str = ""

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def over_budget(self) -> bool:
        return self.elapsed() > self.budget_s


@dataclass
class RunResult:
    ok: bool
    message: str
    run_dir: Path
    glb_path: Path | None
    obj_zip_path: Path | None
    manifest: dict


EventCallback = Callable[[str, str, float], None]


class Sketch2Shape:
    def __init__(self, cfg=None, models=None):
        self.cfg = cfg or load_config()
        self.models = models or ModelManager(self.cfg)
        self.handlers = {
            State.PREPROCESS: self._preprocess,
            State.UNDERSTAND: self._understand,
            State.GENERATE: self._generate,
            State.SCORE: self._score,
            State.PREPARE: self._prepare,
            State.SHAPE: self._shape,
            State.CHECK_SHAPE: self._check_shape,
            State.TEXTURE: self._texture,
            State.CHECK_TEXTURE: self._check_texture,
            State.EXPORT: self._export,
        }

    # ------------------------------------------------------------------ driver

    def new_context(self, image, prompt: str = "", source: str = "upload") -> RunContext:
        store = RunStore(self.cfg.runs_dir)
        ctx = RunContext(
            store=store, image=image, prompt=(prompt or "").strip(), source=source,
            budget_s=float(self.cfg.time_budget_s),
            control_scale=float(self.cfg.generate.control_scale),
            candidates_n=int(self.cfg.generate.candidates),
        )
        store.set("inputs", {"prompt": ctx.prompt, "source": source})
        store.set("config", self.cfg.raw)
        store.set("environment", environment_info())
        return ctx

    def run(self, image, prompt: str = "", source: str = "upload",
            on_event: EventCallback | None = None) -> RunResult:
        ctx = self.new_context(image, prompt, source)
        return self.finish(ctx, self.drive(ctx, on_event))

    def finish(self, ctx: RunContext, final: State) -> RunResult:
        """Record the outcome in manifest.json and build the result."""
        ok = final == State.DONE
        if ok and not ctx.message:
            ctx.message = f"Done in {ctx.elapsed():.0f} s."
        ctx.store.set("result", {
            "ok": ok, "message": ctx.message, "total_s": round(ctx.elapsed(), 1),
            "texture_mode": ctx.texture_mode, "generation_rounds": ctx.rounds + 1 if ctx.candidates else 0,
            "shape_attempts": ctx.shape_attempts,
        })
        ctx.store.write()
        return RunResult(
            ok=ok, message=ctx.message, run_dir=ctx.store.dir,
            glb_path=ctx.bundle.glb_path if ctx.bundle else None,
            obj_zip_path=ctx.bundle.obj_zip_path if ctx.bundle else None,
            manifest=ctx.store.manifest,
        )

    def drive(self, ctx: RunContext, on_event: EventCallback | None = None, max_steps: int = 40) -> State:
        """Run the state machine. on_event(state, outcome, elapsed) is called with outcome "start" when a
        step begins and with its outcome when it ends, so a UI can show what is happening right now."""
        state = State.PREPROCESS
        for _ in range(max_steps):
            if state in TERMINAL:
                return state
            if on_event:
                on_event(state.value, "start", ctx.elapsed())
            started = time.monotonic()
            try:
                outcome = self.handlers[state](ctx)
            except Exception as exc:  # an unexpected stage failure; the recovery rule below decides what next
                outcome = "error"
                ctx.store.warn(f"{state.value} failed: {exc!r}")
                ctx.store.manifest.setdefault("errors", []).append(
                    {"state": state.value, "traceback": traceback.format_exc()[-3000:]})
            seconds = time.monotonic() - started
            free_memory()                   # keep the main process small for the next GPU worker
            ctx.store.add_time(state.value, seconds)
            nxt = TRANSITIONS.get((state, outcome)) or self._recover(ctx, state)
            ctx.store.manifest["transitions"].append(
                {"state": state.value, "outcome": outcome, "next": nxt.value, "seconds": round(seconds, 2),
                 "main_rss_gb": rss_gb()})
            if on_event:
                on_event(state.value, outcome, ctx.elapsed())
            state = nxt
        ctx.message = ctx.message or "Stopped after too many steps."
        return State.FAILED

    def _recover(self, ctx: RunContext, state: State) -> State:
        """No planned transition: export whatever mesh exists, otherwise fail with a clear message."""
        if state != State.EXPORT and (ctx.final_mesh is not None or ctx.mesh is not None):
            if ctx.final_mesh is None:
                ctx.final_mesh = self._projected(ctx, "an earlier step failed")
            return State.EXPORT
        ctx.message = ctx.message or f"The run failed during '{state.value}'. See manifest.json in {ctx.store.dir}."
        return State.FAILED

    # ---------------------------------------------------------------- handlers

    def _preprocess(self, ctx: RunContext) -> str:
        try:
            ctx.sketch = preprocess(ctx.image, ctx.store, self.cfg, ctx.source)
        except EmptySketchError as exc:
            ctx.message = str(exc)
            return "empty"
        ctx.store.set("sketch", asdict(ctx.sketch))
        return "ok"

    def _understand(self, ctx: RunContext) -> str:
        ctx.intent = understand(self.models, ctx.sketch, ctx.prompt, self.cfg, ctx.store)
        if ctx.intent.label is None and not ctx.prompt:
            g = self.cfg.generate      # unknown object and no prompt: lean harder on the lines
            ctx.control_scale = min(g.max_control_scale, ctx.control_scale + g.control_scale_step)
        ctx.store.set("intent", asdict(ctx.intent))
        return "ok"

    def _generate(self, ctx: RunContext) -> str:
        fresh, used_n = generate(self.models, ctx.sketch, ctx.intent, ctx.control_scale,
                                 ctx.candidates_n, ctx.rounds, ctx.store, self.cfg)
        if used_n < ctx.candidates_n:
            ctx.store.warn(f"Generated {used_n} candidate(s) instead of {ctx.candidates_n} to fit in GPU memory.")
            ctx.candidates_n = used_n
        ctx.candidates.extend(fresh)
        return "ok"

    def _score(self, ctx: RunContext) -> str:
        score_candidates(self.models, ctx.sketch, ctx.intent,
                         [c for c in ctx.candidates if not c.scores], self.cfg, ctx.store)
        ctx.best = max(ctx.candidates, key=lambda c: c.total)
        ctx.store.set("candidates", [
            {"path": c.path.name, "round": c.round, "seed": c.seed, "index": c.index,
             "control_scale": c.control_scale, "scores": c.scores, "total": c.total}
            for c in ctx.candidates])
        ctx.store.set("chosen_candidate", ctx.best.path.name)
        s, g = self.cfg.score, self.cfg.generate
        if ctx.best.total >= s.threshold:
            return "pass"
        if ctx.rounds < s.max_rounds and not ctx.over_budget():
            ctx.rounds += 1
            ctx.control_scale = min(g.max_control_scale, ctx.control_scale + g.control_scale_step)
            return "retry"
        ctx.store.warn(f"No candidate reached the sketch-fidelity threshold ({s.threshold}); "
                       f"using the best one ({ctx.best.total:.2f}).")
        return "give_up"

    def _prepare(self, ctx: RunContext) -> str:
        ctx.prepared = prepare(self.models, ctx.best, ctx.store, self.cfg)
        ctx.store.set("prepared", asdict(ctx.prepared))
        return "ok"

    def _shape(self, ctx: RunContext) -> str:
        r = self.cfg.reconstruct
        seed = int(r.seed) + ctx.shape_attempts * 7919
        ctx.shape_attempts += 1
        try:
            ctx.mesh = make_shape(self.models, ctx.prepared, seed, ctx.store, self.cfg)
        except Exception as exc:
            ctx.store.warn(f"Shape attempt {ctx.shape_attempts} (seed {seed}) failed: {exc}")
            if ctx.shape_attempts < r.max_shape_attempts and not ctx.over_budget():
                return "retry"
            if ctx.best_shape is not None:      # an earlier attempt produced a shape: keep it
                ctx.mesh = ctx.best_shape
                return "use_best"
            ctx.message = "3D shape generation failed. See manifest.json for the error."
            return "error"
        ctx.store.manifest.setdefault("shape_seeds", []).append(seed)
        return "ok"

    def _check_shape(self, ctx: RunContext) -> str:
        report = check_mesh(ctx.mesh, self.cfg)
        if "loose fragments" in report.problems:
            ctx.mesh = drop_fragments(ctx.mesh, self.cfg.validate.max_fragment_share)
            report = check_mesh(ctx.mesh, self.cfg)
        ctx.store.manifest.setdefault("shape_checks", []).append(asdict(report))
        if len(report.problems) < ctx.best_shape_problems:
            ctx.best_shape, ctx.best_shape_problems = ctx.mesh, len(report.problems)
        if report.ok:
            return "pass"
        if ctx.shape_attempts < self.cfg.reconstruct.max_shape_attempts and not ctx.over_budget():
            return "retry"
        ctx.mesh = ctx.best_shape
        ctx.store.warn(f"Shape still has problems after {ctx.shape_attempts} attempt(s): "
                       f"{', '.join(report.problems)}. Exporting the best one.")
        return "accept_flagged"

    def _texture(self, ctx: RunContext) -> str:
        r = self.cfg.reconstruct
        if not r.texture or ctx.over_budget():
            ctx.final_mesh = self._projected(ctx, "texturing is off" if not r.texture else "the time budget was spent")
            return "fallback"
        try:
            ctx.textured, ctx.texture_mode = paint_mesh(self.models, ctx.mesh, ctx.prepared, ctx.store, self.cfg)
        except Exception as exc:
            ctx.final_mesh = self._projected(ctx, f"texturing failed ({exc})")
            return "fallback"
        return "ok"

    def _check_texture(self, ctx: RunContext) -> str:
        report = check_mesh(ctx.textured, self.cfg, expect_texture=True)
        ctx.store.set("texture_check", asdict(report))
        if report.ok:
            ctx.final_mesh = ctx.textured
            return "pass"
        ctx.final_mesh = self._projected(ctx, f"the textured model failed checks ({', '.join(report.problems)})")
        return "fallback"

    def _export(self, ctx: RunContext) -> str:
        ctx.bundle = export_mesh(ctx.final_mesh, ctx.store, self.cfg)
        ctx.store.set("export", asdict(ctx.bundle))
        if not ctx.bundle.reload_ok:
            ctx.store.warn("The exported GLB did not reload with the same geometry; check it in a 3D viewer.")
        return "ok"

    def _projected(self, ctx: RunContext, reason: str):
        ctx.store.warn(f"Used colours projected from the image because {reason}.")
        ctx.texture_mode = "projected colours"
        return project_colours(ctx.mesh, ctx.prepared.rgba_path, self.cfg.export.flip_projection_x)


def environment_info() -> dict:
    info = {"python": platform.python_version()}
    for name in ("torch", "diffusers", "transformers", "trimesh", "gradio", "anthropic"):
        try:
            info[name] = __import__(name).__version__
        except Exception:
            info[name] = None
    try:
        import torch
        if torch.cuda.is_available():
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return info
