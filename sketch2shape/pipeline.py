"""The run controller: a small explicit state machine that calls the stages and owns every retry decision.

Stages never call each other and never decide what runs next. Each handler below runs one stage and
returns an outcome word; TRANSITIONS maps (state, outcome) to the next state. Loops are bounded by
attempt counters and by a global time budget, and once any mesh exists a failure exports the best
result so far instead of returning nothing.
"""
from __future__ import annotations

import platform
import subprocess
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from .config import load_config
from .contracts import Candidate, ExportBundle, Intent, PreparedImage, SketchAsset
from .llm import provider
from .models import ModelManager, free_memory, rss_gb
from .runstore import RunStore
from .stages.export import export_mesh
from .stages.generate import generate
from .stages.image_check import check_images
from .stages.prepare import prepare
from .stages.preprocess import EmptySketchError, preprocess, turn_upright
from .stages.reconstruct import make_shape, paint_mesh, project_colours, shape_model_job
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
    strict: bool = False                # an image drifted from the lines: follow them on every step
    negative_base: str = ""             # the negative prompt before the image check's "avoid" words
    shape_attempts: int = 0
    mesh: object = None                 # current untextured shape
    best_shape: object = None
    best_shape_problems: int = 10**6
    textured: object = None
    final_mesh: object = None
    texture_mode: str = "none"
    bundle: ExportBundle | None = None
    message: str = ""
    cancel: threading.Event = field(default_factory=threading.Event)   # set by Sketch2Shape.cancel()

    @property
    def cancelled(self) -> bool:
        return self.cancel.is_set()

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
        self._models_call("new_run")
        store = RunStore(self.cfg.runs_dir)
        ctx = RunContext(
            store=store, image=image, prompt=(prompt or "").strip(), source=source,
            budget_s=float(self.cfg.time_budget_s),
            control_scale=float(self.cfg.generate.control_scale),
            # two images give the vision-LLM check a choice; without a key the line scores pick, and
            # a second cut-out on the CPU would cost ~30 s for little gain
            candidates_n=int(self.cfg.generate.candidates) if provider(self.cfg) else 1,
        )
        store.set("inputs", {"prompt": ctx.prompt, "source": source})
        store.set("config", self.cfg.raw)
        store.set("environment", environment_info())
        return ctx

    def run(self, image, prompt: str = "", source: str = "upload",
            on_event: EventCallback | None = None) -> RunResult:
        ctx = self.new_context(image, prompt, source)
        return self.finish(ctx, self.drive(ctx, on_event))

    def cancel(self, ctx: RunContext) -> None:
        """Stop a run as soon as possible; called from another thread (the app's Cancel button). GPU work
        stops at once; a step running on the CPU finishes first, then the run ends without exporting."""
        ctx.cancel.set()
        self._models_call("cancel")

    def _models_call(self, name: str) -> None:
        """Call an optional ModelManager hook (test doubles may not have it)."""
        hook = getattr(self.models, name, None)
        if hook is not None:
            hook()

    def finish(self, ctx: RunContext, final: State) -> RunResult:
        """Record the outcome in manifest.json and build the result."""
        self._models_call("end_run")        # stop workers loaded early for steps this run never reached
        ok = final == State.DONE
        if ok and not ctx.message:
            ctx.message = f"Done in {ctx.elapsed():.0f} s."
        ctx.store.set("result", {
            "ok": ok, "message": ctx.message, "total_s": round(ctx.elapsed(), 1),
            "texture_mode": ctx.texture_mode, "generation_rounds": ctx.rounds + 1 if ctx.candidates else 0,
            "shape_attempts": ctx.shape_attempts, "cancelled": ctx.cancelled,
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
        step begins and with its outcome when it ends, so a UI can show what is happening right now.
        A cancelled run ends after the current step with the outcome "cancelled", without exporting."""
        state = State.PREPROCESS
        ctx.store.set("setup_s", round(ctx.elapsed(), 2))     # time before the first step
        for _ in range(max_steps):
            if state in TERMINAL:
                return state
            if ctx.cancelled:
                ctx.message = "Cancelled."
                return State.FAILED
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
            cleanup_s = time.monotonic() - started - seconds
            ctx.store.add_time(state.value, seconds)
            if ctx.cancelled:
                outcome, nxt = "cancelled", State.FAILED
                ctx.message = "Cancelled."
            else:
                nxt = TRANSITIONS.get((state, outcome)) or self._recover(ctx, state)
            ctx.store.manifest["transitions"].append(
                {"state": state.value, "outcome": outcome, "next": nxt.value, "seconds": round(seconds, 2),
                 "cleanup_s": round(cleanup_s, 2), "main_rss_gb": rss_gb()})
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
        load_early = getattr(self.models, "load_early", None)
        if load_early is not None and self.cfg.workers.load_shape_early:
            # the shape model's GPU is idle while the image is made, so it starts loading now
            load_early("shape", "hunyuan", ctx.store, **shape_model_job(self.cfg))
        return "ok"

    def _understand(self, ctx: RunContext) -> str:
        ctx.intent = understand(self.models, ctx.sketch, ctx.prompt, self.cfg, ctx.store)
        ctx.negative_base = ctx.intent.negative_prompt
        turns = turn_upright(ctx.sketch, ctx.intent.up)
        if turns:
            ctx.store.warn(f"The drawing was {'upside down' if turns == 2 else 'sideways'}; turned it upright.")
        if ctx.intent.label is None and not ctx.prompt:
            g = self.cfg.generate      # unknown object and no prompt: lean harder on the lines
            ctx.control_scale = min(g.max_control_scale, ctx.control_scale + g.control_scale_step)
        ctx.store.set("intent", asdict(ctx.intent))
        return "ok"

    def _generate(self, ctx: RunContext) -> str:
        fresh, used_n = generate(self.models, ctx.sketch, ctx.intent, ctx.control_scale,
                                 ctx.candidates_n, ctx.rounds, ctx.store, self.cfg, strict=ctx.strict)
        if used_n < ctx.candidates_n:
            ctx.store.warn(f"Generated {used_n} candidate(s) instead of {ctx.candidates_n} to fit in GPU memory.")
            ctx.candidates_n = used_n
        ctx.candidates.extend(fresh)
        return "ok"

    def _score(self, ctx: RunContext) -> str:
        """Gate 1. The vision LLM (when there is a key) picks the usable image of this round, or rejects them
        all; the line scores then check the pick follows the sketch. A rejection retries with new seeds and
        the check's words added to the negative prompt; an image that drifted from the lines retries with
        stronger sketch control. A scene's diorama is judged by the check alone."""
        fresh = [c for c in ctx.candidates if c.round == ctx.rounds and not c.scores and not c.judged]
        verdict = check_images(ctx.sketch, ctx.intent, fresh, self.cfg, ctx.store)
        score_candidates(self.models, ctx.sketch, ctx.intent,
                         [c for c in fresh if c.judged != "rejected"], self.cfg, ctx.store)
        usable = [c for c in ctx.candidates if c.judged != "rejected" and c.scores]
        ctx.best = max(usable, key=lambda c: c.total) if usable else None
        s, g = self.cfg.score, self.cfg.generate
        scene = ctx.intent.kind == "scene"
        if ctx.best is not None and (ctx.best.total >= s.threshold or (scene and ctx.best.judged == "picked")):
            return self._log_candidates(ctx, "pass")
        if ctx.rounds < s.max_rounds and not ctx.over_budget():
            ctx.rounds += 1
            if verdict is not None and verdict["best"] is None:     # all rejected: avoid what the check saw
                if verdict["avoid"]:            # replaces earlier ones, so the prompt stays within 77 tokens
                    ctx.intent.negative_prompt = f"{verdict['avoid']}, {ctx.negative_base or ctx.intent.negative_prompt}"
            elif not scene:                                         # drifted from the lines: follow them
                ctx.strict = True
                ctx.control_scale = min(g.max_control_scale, ctx.control_scale + g.control_scale_step)
            return self._log_candidates(ctx, "retry")
        if ctx.best is None:            # every image was rejected: fall back to the line scores alone
            score_candidates(self.models, ctx.sketch, ctx.intent,
                             [c for c in ctx.candidates if not c.scores], self.cfg, ctx.store)
            ctx.best = max(ctx.candidates, key=lambda c: c.total)
            ctx.store.warn("The image check rejected every image; using the one that follows the sketch best.")
        else:
            ctx.store.warn(f"No image reached the sketch-fidelity threshold ({s.threshold}); "
                           f"using the best one ({ctx.best.total:.2f}).")
        return self._log_candidates(ctx, "give_up")

    def _log_candidates(self, ctx: RunContext, outcome: str) -> str:
        ctx.store.set("candidates", [
            {"path": c.path.name, "round": c.round, "seed": c.seed, "index": c.index,
             "control_scale": c.control_scale, "judged": c.judged, "scores": c.scores, "total": c.total}
            for c in ctx.candidates])
        if ctx.best is not None:
            ctx.store.set("chosen_candidate", ctx.best.path.name)
        return outcome

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


_ENVIRONMENT: dict | None = None


def environment_info() -> dict:
    """Python, package versions and GPU names. Read once per process from package metadata and nvidia-smi:
    importing the packages and asking torch for the GPUs cost the first run of an app up to ~20 s and
    opened a CUDA context in this process."""
    global _ENVIRONMENT
    if _ENVIRONMENT is None:
        from importlib import metadata
        info = {"python": platform.python_version()}
        for name in ("torch", "diffusers", "transformers", "trimesh", "gradio", "openai", "anthropic"):
            try:
                info[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                info[name] = None
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=30).stdout
            info["gpus"] = [line.strip() for line in out.splitlines() if line.strip()]
        except (OSError, subprocess.SubprocessError):
            pass
        _ENVIRONMENT = info
    return dict(_ENVIRONMENT)
