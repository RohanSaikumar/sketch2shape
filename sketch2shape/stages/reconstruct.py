"""Stage 6: image to 3D with Hunyuan3D-2, as two steps, each in its own GPU worker process.

make_shape  runs the shape model (turbo + FlashVDM by default) and Hunyuan's own clean-up:
            floater removal, degenerate-face removal and face reduction.
paint_mesh  runs the texture model on the checked shape, retrying with CPU offload if the GPU
            runs out of memory.
project_colours is the fallback when texturing is off, fails or runs out of time: it projects the
prepared image straight through the mesh along the viewing axis, so the model still has colour.
"""
from __future__ import annotations

import numpy as np
import trimesh
from PIL import Image

from .validate import as_trimesh


def make_shape(models, prepared, seed: int, store, cfg) -> trimesh.Trimesh:
    r = cfg.reconstruct
    out = store.path(f"07_shape_seed{seed}.glb")
    models.run_worker("shape", "hunyuan", store, model=r.model, subfolder=r.shape_subfolder,
                      steps=int(r.shape_steps), flashvdm=bool(r.flashvdm),
                      image=str(prepared.rgba_path), seed=int(seed), out=str(out))
    return as_trimesh(trimesh.load(out, force="scene"))


def paint_mesh(models, mesh, prepared, store, cfg):
    """Returns (textured mesh, how it was made)."""
    r = cfg.reconstruct
    shape_path = store.path("08_shape_checked.glb")
    mesh.export(shape_path)                         # the shape after validation and any repair
    out = store.path("09_textured.glb")
    result = models.run_worker("paint", "hunyuan", store, model=r.model, mesh=str(shape_path),
                               image=str(prepared.rgba_path), out=str(out),
                               cpu_offload_on_oom=bool(r.paint_cpu_offload_on_oom))
    return as_trimesh(trimesh.load(out, force="scene")), result["mode"]


def project_colours(mesh: trimesh.Trimesh, rgba_path, flip_x: bool = False) -> trimesh.Trimesh:
    """Vertex colours sampled from the prepared image, mapping the mesh's x/y box onto the object's box."""
    rgba = np.array(Image.open(rgba_path).convert("RGBA"))
    alpha = rgba[..., 3] > 10
    ys, xs = np.nonzero(alpha)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    v = np.asarray(mesh.vertices)
    u = (v[:, 0] - v[:, 0].min()) / max(float(np.ptp(v[:, 0])), 1e-9)
    w = (v[:, 1] - v[:, 1].min()) / max(float(np.ptp(v[:, 1])), 1e-9)
    if flip_x:
        u = 1.0 - u
    px = np.clip(np.round(x0 + u * (x1 - x0)).astype(int), 0, rgba.shape[1] - 1)
    py = np.clip(np.round(y1 - w * (y1 - y0)).astype(int), 0, rgba.shape[0] - 1)
    colours = rgba[py, px, :3].astype(np.uint8)
    off_object = ~alpha[py, px]
    colours[off_object] = rgba[alpha][:, :3].mean(axis=0).astype(np.uint8)   # edges take the mean colour
    out = mesh.copy()
    out.visual = trimesh.visual.ColorVisuals(
        mesh=out, vertex_colors=np.column_stack([colours, np.full(len(colours), 255, np.uint8)]))
    return out
