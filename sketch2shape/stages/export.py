"""Stage 8: write the model as GLB (texture embedded) and as a zipped OBJ + MTL + PNG, then reload
the GLB to prove the file opens with the same geometry.

Normalisation follows glTF conventions: Y is up and units are metres. The model is centred, scaled so
its largest side is 1 unit, and rests on the ground plane (y = 0).

Materials are written as matte (metallicFactor 0) with a white base colour factor. glTF treats a missing
metallicFactor as 1.0, fully metal, which most viewers render as black or as tinted reflections instead
of the texture colours.
"""
from __future__ import annotations

import json
import shutil
import struct
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


# trimesh writes no metallicFactor (read as 1.0, metal) and a 0.4 grey baseColorFactor, which darkens the
# texture to 40 %. White factor + non-metallic shows the texture (or vertex colours) as generated.
MATTE = {"metallicFactor": 0.0, "roughnessFactor": 0.85, "baseColorFactor": [1.0, 1.0, 1.0, 1.0]}


def make_glb_matte(path: Path) -> None:
    """Rewrite a GLB's JSON chunk so every material is non-metallic and every primitive has a material."""
    data = Path(path).read_bytes()
    magic, version, _ = struct.unpack_from("<4sII", data, 0)
    json_length, json_type = struct.unpack_from("<II", data, 12)
    if magic != b"glTF" or json_type != 0x4E4F534A:          # 'JSON'
        return
    gltf = json.loads(data[20:20 + json_length].decode("utf-8"))
    materials = gltf.setdefault("materials", [])
    for material in materials:
        pbr = material.setdefault("pbrMetallicRoughness", {})
        pbr.update(MATTE)
        material["doubleSided"] = True
    default = None
    for mesh in gltf.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            if "material" not in primitive:             # e.g. vertex colours only
                if default is None:
                    materials.append({"name": "matte", "pbrMetallicRoughness": dict(MATTE), "doubleSided": True})
                    default = len(materials) - 1
                primitive["material"] = default
    chunk = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    chunk += b" " * (-len(chunk) % 4)                   # chunks are 4-byte aligned, padded with spaces
    rest = data[20 + json_length:]
    body = struct.pack("<II", len(chunk), json_type) + chunk + rest
    Path(path).write_bytes(struct.pack("<4sII", magic, version, 12 + len(body)) + body)


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
    try:
        make_glb_matte(glb_path)
    except Exception as exc:            # never lose the model over a material tweak
        store.warn(f"Could not set a matte material ({exc!r}); the model may look dark in some viewers.")
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
