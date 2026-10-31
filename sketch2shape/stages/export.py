"""Stage 8: write the model as GLB (texture embedded) and as a zipped OBJ + MTL + PNG, then reload
the GLB to prove the file opens with the same geometry.

Normalisation follows glTF conventions: Y is up and units are metres. The model is centred, scaled so
its largest side is 1 unit, and rests on the ground plane (y = 0).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import trimesh

from ..contracts import ExportBundle
from .validate import as_trimesh, is_textured


def normalize(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    out = mesh.copy()
    low, high = out.bounds
    out.apply_translation(-(low + high) / 2.0)
    out.apply_scale(1.0 / max(float(out.extents.max()), 1e-9))
    out.apply_translation([0.0, -float(out.bounds[0][1]), 0.0])
    return out


def write_obj_zip(mesh: trimesh.Trimesh, store):
    folder = store.path("obj")
    folder.mkdir(exist_ok=True)
    try:
        from trimesh.exchange.obj import export_obj
        result = export_obj(mesh, include_texture=True, return_texture=True, mtl_name="model.mtl")
        text, extras = result if isinstance(result, tuple) else (result, {})
        (folder / "model.obj").write_text(text, encoding="utf-8")
        for name, data in (extras or {}).items():
            (folder / name).write_bytes(data)
    except TypeError:                   # older trimesh without these options
        mesh.export(folder / "model.obj")
    return Path(shutil.make_archive(str(store.path("model_obj")), "zip", folder))


def export_mesh(mesh, store, cfg) -> ExportBundle:
    mesh = as_trimesh(mesh)
    if cfg.export.normalize:
        mesh = normalize(mesh)
    glb_path = store.path("model.glb")
    mesh.export(glb_path)
    reloaded = as_trimesh(trimesh.load(glb_path, force="scene"))
    reload_ok = len(reloaded.faces) == len(mesh.faces) and len(reloaded.faces) > 0
    obj_zip = None
    if "obj" in cfg.export.formats:
        try:
            obj_zip = write_obj_zip(mesh, store)
        except Exception as exc:
            store.warn(f"OBJ export failed ({exc!r}); the GLB is still available.")
    return ExportBundle(glb_path=glb_path, obj_zip_path=obj_zip, reload_ok=reload_ok,
                        faces=len(mesh.faces), textured=is_textured(reloaded))
