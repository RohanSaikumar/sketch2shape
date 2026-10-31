"""Stage 6: image to 3D with Hunyuan3D-2, as two steps.

make_shape  runs the shape model (turbo + FlashVDM by default) and Hunyuan's own clean-up:
            floater removal, degenerate-face removal and face reduction.
paint_mesh  runs the texture model, retrying with CPU offload if the GPU runs out of memory.
project_colours is the fallback when texturing is off, fails or runs out of time: it projects the
prepared image straight through the mesh along the viewing axis, so the model still has colour.
"""
from __future__ import annotations

import numpy as np
import trimesh
from PIL import Image


def make_shape(models, prepared, seed: int) -> trimesh.Trimesh:
    import torch
    from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover
    pipe = models.shape()
    image = Image.open(prepared.rgba_path).convert("RGBA")
    with models.on_hunyuan_gpu():
        mesh = pipe(image=image, num_inference_steps=models.shape_steps,
                    generator=torch.manual_seed(seed))[0]
    for step in (FloaterRemover(), DegenerateFaceRemover(), FaceReducer()):
        mesh = step(mesh)
    return mesh


def paint_mesh(models, mesh, prepared, cfg):
    """Returns (textured mesh, how it was made)."""
    import torch
    pipe = models.paint()
    image = Image.open(prepared.rgba_path).convert("RGBA")
    try:
        with models.on_hunyuan_gpu():
            return pipe(mesh.copy(), image=image), "hunyuan paint"
    except torch.cuda.OutOfMemoryError:
        if not cfg.reconstruct.paint_cpu_offload_on_oom:
            raise
        models.free_memory()
        models.paint_offload()
        with models.on_hunyuan_gpu():
            return pipe(mesh.copy(), image=image), "hunyuan paint (CPU offload)"


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
