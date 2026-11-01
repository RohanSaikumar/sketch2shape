"""Tests that run without a GPU: preprocessing, scoring maths, mesh checks, export and the state machine.

    python -m pytest -q tests
"""
from __future__ import annotations

import zipfile

import cv2
import numpy as np
import pytest
import trimesh
from PIL import Image, ImageDraw

from sketch2shape.config import load_config
from sketch2shape.pipeline import TERMINAL, TRANSITIONS, Sketch2Shape, State
from sketch2shape.runstore import RunStore
from sketch2shape.stages.export import export_mesh
from sketch2shape.stages.imageops import fill_holes
from sketch2shape.stages.preprocess import EmptySketchError, preprocess
from sketch2shape.stages.reconstruct import project_colours
from sketch2shape.stages.score import iou, stroke_fidelity
from sketch2shape.stages.validate import as_trimesh, check_mesh, drop_fragments, is_textured


@pytest.fixture
def cfg(tmp_path):
    return load_config(overrides={"runs_dir": str(tmp_path / "runs")})


@pytest.fixture
def store(tmp_path):
    return RunStore(tmp_path / "runs")


def mug_drawing(size=512, background=(0, 0, 0, 0)) -> Image.Image:
    """Black strokes on a transparent canvas, like the Gradio sketchpad returns."""
    img = Image.new("RGBA", (size, size), background)
    d = ImageDraw.Draw(img)
    s = size / 1024
    def arc(cx, cy, rx, ry, a0, a1):
        d.arc([(cx - rx) * s, (cy - ry) * s, (cx + rx) * s, (cy + ry) * s], a0, a1, fill=(0, 0, 0, 255), width=7)
    arc(480, 300, 210, 60, 0, 360)
    d.line([(270 * s, 300 * s), (290 * s, 780 * s)], fill=(0, 0, 0, 255), width=7)
    d.line([(690 * s, 300 * s), (670 * s, 780 * s)], fill=(0, 0, 0, 255), width=7)
    arc(480, 780, 190, 55, 0, 180)
    arc(680, 540, 120, 150, -90, 90)
    return img


# ---------------------------------------------------------------- preprocess

def test_preprocess_canvas_drawing(cfg, store):
    asset = preprocess(mug_drawing(), store, cfg, source="canvas")
    lineart = np.array(Image.open(asset.lineart_path).convert("L"))
    mask = np.array(Image.open(asset.mask_path))
    assert lineart.shape == (1024, 1024)
    assert lineart[0, 0] == 255                                # white background, padded border
    strokes = (lineart < 128).sum()
    assert 0.005 < strokes / lineart.size < 0.2
    assert (mask > 0).sum() > 3 * strokes                      # outline was filled into a silhouette
    control = np.array(Image.open(asset.control_path).convert("L"))
    assert control[0, 0] == 255                                # "bw" polarity: black lines on white


def test_preprocess_photo_with_uneven_light(cfg, store):
    shade = np.tile(np.linspace(150, 240, 800, dtype=np.uint8), (800, 1))
    photo = Image.fromarray(np.dstack([shade] * 3))
    ImageDraw.Draw(photo).ellipse([250, 200, 550, 600], outline=(30, 30, 30), width=6)
    asset = preprocess(photo, store, cfg)
    lineart = np.array(Image.open(asset.lineart_path).convert("L"))
    assert (lineart < 128).mean() < 0.15                       # the shading did not turn into strokes


def test_preprocess_rejects_empty_canvas(cfg, store):
    with pytest.raises(EmptySketchError):
        preprocess(Image.new("RGBA", (512, 512), (0, 0, 0, 0)), store, cfg)
    with pytest.raises(EmptySketchError):
        preprocess({"composite": None}, store, cfg)


def test_fill_holes_fills_a_ring():
    ring = np.zeros((100, 100), np.uint8)
    ImageDraw.Draw(img := Image.fromarray(ring)).ellipse([20, 20, 80, 80], outline=255, width=3)
    filled = fill_holes(np.array(img))
    assert filled[50, 50] == 255 and filled[2, 2] == 0


# -------------------------------------------------------------------- score

def test_stroke_fidelity_and_iou():
    a = np.zeros((64, 64), bool)
    a[10:50, 20] = True
    assert stroke_fidelity(a, a, 2.0)["f2"] == 1.0
    shifted = np.roll(a, 30, axis=1)
    assert stroke_fidelity(a, shifted, 2.0)["recall"] == 0.0
    assert iou(a, a) == 1.0 and iou(a, shifted) == 0.0


# ----------------------------------------------------------------- validate

def test_check_mesh_accepts_a_solid_and_rejects_a_plate(cfg):
    sphere = trimesh.creation.icosphere(subdivisions=3)
    assert check_mesh(sphere, cfg).ok
    plate = sphere.copy()
    plate.apply_scale([1.0, 1.0, 0.01])
    report = check_mesh(plate, cfg)
    assert not report.ok and "flat" in report.problems
    assert "too few faces" in check_mesh(trimesh.creation.box(), cfg).problems


def test_drop_fragments_removes_floaters(cfg):
    body = trimesh.creation.icosphere(subdivisions=3)
    speck = trimesh.creation.icosphere(subdivisions=1, radius=0.05)
    speck.apply_translation([3, 0, 0])
    both = trimesh.util.concatenate([body, speck])
    assert "loose fragments" in check_mesh(both, cfg).problems
    cleaned = drop_fragments(both, 0.02)
    assert len(cleaned.split(only_watertight=False)) == 1
    assert check_mesh(cleaned, cfg).ok


# ------------------------------------------------------------ texture/export

def textured_sphere():
    sphere = trimesh.creation.icosphere(subdivisions=3)
    texture = Image.new("RGB", (64, 64), (200, 40, 40))
    ImageDraw.Draw(texture).rectangle([0, 0, 31, 63], fill=(40, 40, 200))
    uv = (sphere.vertices[:, :2] + 1) / 2
    sphere.visual = trimesh.visual.TextureVisuals(uv=uv, image=texture)
    return sphere


def test_uv_seams_are_not_counted_as_fragments(cfg):
    """A textured mesh duplicates vertices along UV seams; it is still one piece (this once failed every texture)."""
    sphere = textured_sphere()
    seams = trimesh.Trimesh(vertices=sphere.vertices[sphere.faces].reshape(-1, 3),
                            faces=np.arange(len(sphere.faces) * 3).reshape(-1, 3), process=False)
    seams.visual = trimesh.visual.TextureVisuals(uv=((seams.vertices[:, :2] + 1) / 2),
                                                 image=sphere.visual.material.image)
    assert len(seams.split(only_watertight=False)) > 100        # naive splitting sees every face as a piece
    report = check_mesh(seams, cfg, expect_texture=True)
    assert report.ok, report.problems
    assert report.checks["parts"] == 1
    assert drop_fragments(seams, 0.02) is not None


def test_project_colours_follows_the_image(tmp_path):
    sphere = trimesh.creation.icosphere(subdivisions=3)
    rgba = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    d = ImageDraw.Draw(rgba)
    d.rectangle([50, 50, 99, 150], fill=(255, 0, 0, 255))     # left half red
    d.rectangle([100, 50, 150, 150], fill=(0, 0, 255, 255))   # right half blue
    path = tmp_path / "prepared.png"
    rgba.save(path)
    coloured = project_colours(sphere, path)
    colours = coloured.visual.vertex_colors
    left, right = sphere.vertices[:, 0] < -0.5, sphere.vertices[:, 0] > 0.5
    assert colours[left, 0].mean() > 200 and colours[right, 2].mean() > 200
    flipped = project_colours(sphere, path, flip_x=True).visual.vertex_colors
    assert flipped[left, 2].mean() > 200


def test_export_round_trip_keeps_geometry_and_texture(cfg, store):
    bundle = export_mesh(textured_sphere(), store, cfg)
    assert bundle.reload_ok and bundle.textured and bundle.faces == 1280
    reloaded = as_trimesh(trimesh.load(bundle.glb_path, force="scene"))
    assert abs(reloaded.bounds[0][1]) < 1e-6                   # rests on y = 0
    assert abs(reloaded.extents.max() - 1.0) < 1e-6            # scaled to 1 unit
    assert is_textured(reloaded)
    names = zipfile.ZipFile(bundle.obj_zip_path).namelist()
    assert "model.obj" in names and any(n.endswith(".mtl") for n in names)


def test_obj_material_is_not_darkened(cfg, store):
    """The MTL's Kd multiplies the texture; Hunyuan's material had 0.4 grey, so OBJs opened dark."""
    sphere = textured_sphere()
    sphere.visual.material.diffuse = [102, 102, 102, 255]       # what Hunyuan's mesh carries
    bundle = export_mesh(sphere, store, cfg)
    with zipfile.ZipFile(bundle.obj_zip_path) as z:
        mtl = next(n for n in z.namelist() if n.endswith(".mtl"))
        kd = [line.split()[1:4] for line in z.read(mtl).decode().splitlines() if line.startswith("Kd ")]
        assert any(n.endswith(".png") or n.endswith(".jpg") for n in z.namelist())   # texture travels along
    assert kd and all(abs(float(v) - 1.0) < 1e-3 for row in kd for v in row), kd


def glb_json(path):
    import json
    import struct
    data = open(path, "rb").read()
    length = struct.unpack_from("<I", data, 12)[0]
    assert struct.unpack_from("<I", data, 8)[0] == len(data)        # header length still matches the file
    return json.loads(data[20:20 + length])


@pytest.mark.parametrize("make", ["textured", "vertex colours"])
def test_exported_materials_are_matte(cfg, store, tmp_path, make):
    """glTF reads a missing metallicFactor as 1.0 (metal), which viewers show as black."""
    if make == "textured":
        mesh = textured_sphere()
    else:
        rgba = tmp_path / "p.png"
        Image.new("RGBA", (64, 64), (200, 30, 30, 255)).save(rgba)
        mesh = project_colours(trimesh.creation.icosphere(subdivisions=3), rgba)
    bundle = export_mesh(mesh, store, cfg)
    gltf = glb_json(bundle.glb_path)
    assert gltf["materials"]
    for material in gltf["materials"]:
        assert material["pbrMetallicRoughness"]["metallicFactor"] == 0.0
        assert material["pbrMetallicRoughness"]["baseColorFactor"] == [1.0, 1.0, 1.0, 1.0]
    for m in gltf["meshes"]:
        for primitive in m["primitives"]:
            assert "material" in primitive
    assert bundle.reload_ok


# ------------------------------------------------------------ state machine

def test_every_working_state_has_a_way_out():
    sources = {state for state, _ in TRANSITIONS}
    assert set(State) - TERMINAL <= sources
    assert all(isinstance(target, State) for target in TRANSITIONS.values())


def scripted(outcomes):
    """A handler that returns the given outcomes in order."""
    queue = list(outcomes)
    return lambda ctx: queue.pop(0)


def test_state_machine_retries_then_exports(cfg):
    pipe = Sketch2Shape(cfg, models=object())
    pipe.handlers = {
        State.PREPROCESS: scripted(["ok"]), State.UNDERSTAND: scripted(["ok"]),
        State.GENERATE: scripted(["ok", "ok"]), State.SCORE: scripted(["retry", "pass"]),
        State.PREPARE: scripted(["ok"]), State.SHAPE: scripted(["ok", "ok"]),
        State.CHECK_SHAPE: scripted(["retry", "pass"]), State.TEXTURE: scripted(["ok"]),
        State.CHECK_TEXTURE: scripted(["fallback"]), State.EXPORT: scripted(["ok"]),
    }
    ctx = pipe.new_context(image=None)
    assert pipe.drive(ctx) == State.DONE
    path = [t["state"] for t in ctx.store.manifest["transitions"]]
    assert path == ["preprocess", "understand", "generate", "score", "generate", "score", "prepare",
                    "shape", "check_shape", "shape", "check_shape", "texture", "check_texture", "export"]


def test_state_machine_exports_best_mesh_after_an_unexpected_error(cfg):
    pipe = Sketch2Shape(cfg, models=object())

    def boom(ctx):
        raise RuntimeError("texture model crashed")

    def shape_ok(ctx):
        ctx.mesh = trimesh.creation.icosphere()
        ctx.final_mesh = ctx.mesh
        return "ok"

    pipe.handlers.update({
        State.PREPROCESS: scripted(["ok"]), State.UNDERSTAND: scripted(["ok"]), State.GENERATE: scripted(["ok"]),
        State.SCORE: scripted(["pass"]), State.PREPARE: scripted(["ok"]), State.SHAPE: shape_ok,
        State.CHECK_SHAPE: scripted(["pass"]), State.TEXTURE: boom, State.EXPORT: scripted(["ok"]),
    })
    ctx = pipe.new_context(image=None)
    assert pipe.drive(ctx) == State.DONE
    step = ctx.store.manifest["transitions"][-2]
    assert (step["state"], step["outcome"], step["next"]) == ("texture", "error", "export")


def test_worker_process_round_trip(cfg, store):
    from sketch2shape.models import ModelManager, WorkerError
    models = ModelManager(cfg)
    result = models.run_worker("ping", "sdxl", store)
    assert result["ok"] and result["python"]
    with pytest.raises(WorkerError):
        models.run_worker("no_such_task", "sdxl", store)
    assert list(store.dir.glob("worker_no_such_task_*.log"))


@pytest.fixture
def serving_models(tmp_path, monkeypatch):
    """A ModelManager that keeps the test 'ping' worker loaded, as it keeps SDXL loaded on two GPUs."""
    from sketch2shape.models import ModelManager
    cfg = load_config(overrides={"runs_dir": str(tmp_path / "runs"), "workers.keep_loaded": ["ping"],
                                 "workers.min_free_ram_gb": 0})
    monkeypatch.setattr(ModelManager, "separate_gpus", lambda self: True)
    monkeypatch.setattr(ModelManager, "gpu_index", lambda self, role: {"hunyuan": 1}.get(role, 0))
    models = ModelManager(cfg)
    yield models
    models.keep_warm = False
    with models._lock:
        models._waiting.clear()
    models.end_run()
    for worker in list(models._kept.values()):
        if hasattr(worker, "proc"):
            worker.stop(wait_s=5)


def wait_for(condition, timeout: float = 30.0) -> None:
    import time
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.1)


def test_a_kept_worker_serves_several_jobs(serving_models, store):
    assert serving_models.run_worker("ping", "sdxl", store)["ok"]
    assert serving_models.run_worker("ping", "sdxl", store)["ok"]
    kept = serving_models._kept["ping"]
    assert kept.alive() and kept.jobs == 2
    assert [w["how"] for w in store.manifest["workers"]] == ["kept loaded", "kept loaded"]


def test_cancel_stops_the_job_but_keeps_the_model_loaded(serving_models, store):
    import threading
    import time
    from sketch2shape.models import WorkerError
    serving_models.run_worker("ping", "sdxl", store)
    threading.Timer(0.5, serving_models.cancel).start()
    started = time.monotonic()
    with pytest.raises(WorkerError, match="Cancelled"):
        serving_models.run_worker("ping", "sdxl", store, seconds=30)
    assert time.monotonic() - started < 5
    serving_models.new_run()
    assert serving_models.run_worker("ping", "sdxl", store)["ok"]     # the same loaded worker takes the next job
    assert serving_models._kept["ping"].jobs == 2


def test_a_worker_loaded_early_takes_its_one_job_then_exits(serving_models, store):
    import time
    serving_models.cfg.workers.keep_loaded = []     # a kept task needs no early worker
    serving_models.load_early("ping", "hunyuan", store)
    deadline = time.monotonic() + 30
    while "ping" not in serving_models._early and time.monotonic() < deadline:
        time.sleep(0.1)
    early = serving_models._early["ping"]
    assert serving_models.run_worker("ping", "hunyuan", store)["ok"]
    assert not early.alive() and not serving_models._early
    assert store.manifest["workers"][-1]["how"] == "loaded early"
    assert store.manifest["loaded_early"][0]["task"] == "ping"


def test_end_run_stops_a_worker_the_run_never_used(serving_models, store):
    import time
    serving_models.cfg.workers.keep_loaded = []     # a kept task needs no early worker
    serving_models.load_early("ping", "hunyuan", store)
    deadline = time.monotonic() + 30
    while "ping" not in serving_models._early and time.monotonic() < deadline:
        time.sleep(0.1)
    early = serving_models._early["ping"]
    serving_models.end_run()
    assert not early.alive() and not serving_models._early


def test_sdxl_pauses_for_the_3d_steps_and_reloads_after_the_run(serving_models, store, monkeypatch):
    serving_models.keep_warm = True                 # as in the app
    monkeypatch.setattr(type(serving_models), "kept_specs", lambda self: {"ping": ("sdxl", lambda cfg: {})})
    serving_models.run_worker("ping", "sdxl", store)
    first = serving_models._kept["ping"]
    serving_models.release_kept(("ping",))          # its GPU and RAM go to the 3D models
    assert not first.alive() and not serving_models._kept
    serving_models.end_run()                        # reloads it in the background
    wait_for(lambda: "ping" in serving_models._kept)
    assert serving_models._kept["ping"] is not first


def test_the_texture_model_waits_for_sdxl_to_free_its_gpu(serving_models, store):
    import time

    class Loaded:                                   # stands in for the kept SDXL worker on GPU 0
        gpu, stopped = 0, False
        def alive(self): return not self.stopped
        def loading(self): return False
        def stop(self, wait_s=30): self.stopped = True

    serving_models._kept["generate"] = Loaded()
    serving_models.cfg.workers.keep_loaded = []
    serving_models.load_early("ping", "paint", store)          # "paint" is GPU 0, like SDXL
    time.sleep(1.5)
    assert "ping" not in serving_models._early
    serving_models.release_kept(("generate",))
    wait_for(lambda: "ping" in serving_models._early)


def test_kept_models_make_room_when_ram_is_short(serving_models, store, monkeypatch):
    import sketch2shape.models as models_module
    serving_models.run_worker("ping", "sdxl", store)
    kept = serving_models._kept["ping"]
    monkeypatch.setattr(models_module, "free_ram_gb", lambda: 5.0)
    serving_models.cfg.workers.min_free_ram_gb = 16
    serving_models._make_room(store, "paint")
    assert not kept.alive() and not serving_models._kept
    assert store.manifest["ram_made_room"][0]["stopped"] == ["ping"]


def test_dense_hatching_becomes_an_outline_and_plain_lines_stay():
    from sketch2shape.stages.preprocess import outline_hatching
    plain = np.zeros((512, 512), np.uint8)
    cv2.circle(plain, (256, 256), 180, 255, 6)
    assert (outline_hatching(plain) == plain).all()
    hatched = plain.copy()
    for x in range(180, 330, 7):                    # a shaded patch inside the circle
        cv2.line(hatched, (x, 180), (x + 20, 330), 255, 4)
    out = outline_hatching(hatched)
    inside = out[215:295, 215:295]
    assert inside.mean() < 0.1 * hatched[215:295, 215:295].mean()     # the scribbles are gone
    assert out[256, 256 - 180 - 2: 256 - 180 + 3].any()                # the circle is still there


def test_models_loaded_ahead_match_the_jobs_sent_later(cfg):
    """SDXL is loaded at app start and the shape model early with these fields; the jobs must carry the same."""
    from sketch2shape.models import LOAD_KEYS
    from sketch2shape.stages.generate import model_job
    from sketch2shape.stages.reconstruct import shape_model_job
    assert set(model_job(cfg)) == set(LOAD_KEYS["generate"])
    assert set(shape_model_job(cfg)) == set(LOAD_KEYS["shape"])


def test_cancel_kills_a_fresh_worker(cfg, store):
    import threading
    import time
    from sketch2shape.models import ModelManager, WorkerError
    models = ModelManager(cfg)                  # ping is not kept loaded: a fresh worker per job
    threading.Timer(1.0, models.cancel).start()
    started = time.monotonic()
    with pytest.raises(WorkerError, match="Cancelled"):
        models.run_worker("ping", "sdxl", store, seconds=30)
    assert time.monotonic() - started < 10


def test_cancel_ends_the_run_after_the_current_step_without_exporting(cfg):
    pipe = Sketch2Shape(cfg, models=object())
    ctx = pipe.new_context(image=None)

    def generate(c):
        pipe.cancel(c)                          # Cancel pressed while the image is being made
        raise RuntimeError("worker stopped")

    exported = []
    pipe.handlers.update({State.PREPROCESS: scripted(["ok"]), State.UNDERSTAND: scripted(["ok"]),
                          State.GENERATE: generate, State.EXPORT: lambda c: exported.append(1) or "ok"})
    result = pipe.finish(ctx, pipe.drive(ctx))
    assert not result.ok and result.message == "Cancelled." and result.manifest["result"]["cancelled"]
    last = ctx.store.manifest["transitions"][-1]
    assert (last["state"], last["outcome"], last["next"]) == ("generate", "cancelled", "failed")
    assert not exported and "cleanup_s" in last and "setup_s" in ctx.store.manifest


def test_empty_sketch_fails_with_a_message(cfg):
    pipe = Sketch2Shape(cfg, models=object())
    result = pipe.run(Image.new("RGBA", (256, 256), (0, 0, 0, 0)), source="canvas")
    assert not result.ok and "empty" in result.message
    assert (result.run_dir / "manifest.json").exists()
