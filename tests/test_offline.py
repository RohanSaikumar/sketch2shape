"""Tests that run without a GPU: preprocessing, scoring maths, mesh checks, export and the state machine.

    python -m pytest -q tests
"""
from __future__ import annotations

import zipfile

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


def test_empty_sketch_fails_with_a_message(cfg):
    pipe = Sketch2Shape(cfg, models=object())
    result = pipe.run(Image.new("RGBA", (256, 256), (0, 0, 0, 0)), source="canvas")
    assert not result.ok and "empty" in result.message
    assert (result.run_dir / "manifest.json").exists()
