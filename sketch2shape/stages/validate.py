"""Stage 7 (quality gate 2): check that a mesh is real, usable 3D geometry.

Checks: enough faces, finite coordinates, real depth (not a flat plate, which the PS rules out),
no loose fragments, and, after texturing, a texture that is present and not a single flat colour.
Watertightness is reported but not required.
"""
from __future__ import annotations

import numpy as np
import trimesh

from ..contracts import MeshReport


def as_trimesh(obj) -> trimesh.Trimesh:
    """A single Trimesh from a Trimesh or a Scene (as loaded from a GLB)."""
    if isinstance(obj, trimesh.Scene):
        try:
            return obj.to_geometry()
        except AttributeError:          # older trimesh
            return obj.dump(concatenate=True)
    return obj


def texture_image(mesh):
    visual = getattr(mesh, "visual", None)
    if getattr(visual, "kind", None) != "texture":
        return None
    material = getattr(visual, "material", None)
    for attr in ("baseColorTexture", "image"):
        image = getattr(material, attr, None)
        if image is not None:
            return image
    return None


def is_textured(mesh) -> bool:
    image = texture_image(mesh)
    return image is not None and float(np.asarray(image.convert("RGB")).std()) > 2.0


def components(mesh: trimesh.Trimesh) -> tuple[list[np.ndarray], np.ndarray]:
    """Connected pieces as lists of face indices, plus each face's area.

    Works on bare geometry with vertices merged by position: a textured mesh duplicates vertices along
    its UV seams, so splitting it directly reports hundreds of pieces that are really one surface, and
    trimesh copies the texture image into every piece (this filled 22 GB of RAM on Kaggle).
    Face order is unchanged, so the indices also apply to the original mesh.
    """
    geometry = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=True)
    areas = geometry.area_faces
    try:
        groups = trimesh.graph.connected_components(geometry.face_adjacency, nodes=np.arange(len(geometry.faces)))
    except Exception:                   # graph library missing: treat as one piece
        groups = [np.arange(len(geometry.faces))]
    return [np.asarray(g) for g in groups], areas


def part_shares(mesh: trimesh.Trimesh) -> list[float]:
    groups, areas = components(mesh)
    total = float(areas.sum())
    return [float(areas[g].sum()) / total for g in groups] if total > 0 else [1.0]


def check_mesh(mesh, cfg, expect_texture: bool = False) -> MeshReport:
    v = cfg.validate
    mesh = as_trimesh(mesh)
    faces, vertices = len(mesh.faces), len(mesh.vertices)
    finite = bool(np.isfinite(mesh.vertices).all()) if vertices else False
    extents = mesh.extents if vertices else np.zeros(3)
    flatness = float(extents.min() / extents.max()) if vertices and extents.max() > 0 else 0.0
    shares = part_shares(mesh) if faces else []
    checks = {
        "faces": faces, "vertices": vertices, "finite": finite,
        "extents": [round(float(e), 4) for e in extents], "flatness": round(flatness, 3),
        "parts": len(shares), "watertight": bool(mesh.is_watertight) if faces else False,
        "visual": getattr(mesh.visual, "kind", None), "textured": is_textured(mesh),
    }
    problems = []
    if faces < int(v.min_faces):
        problems.append("too few faces")
    if not finite:
        problems.append("non-finite coordinates")
    if flatness < float(v.min_flatness):
        problems.append("flat")
    if any(share < float(v.max_fragment_share) for share in shares) and len(shares) > 1:
        problems.append("loose fragments")
    if expect_texture and not checks["textured"]:
        problems.append("no texture")
    return MeshReport(ok=not problems, checks=checks, problems=problems)


def drop_fragments(mesh, min_share: float) -> trimesh.Trimesh:
    """Remove pieces smaller than min_share of the surface, keeping the mesh's own visuals."""
    mesh = as_trimesh(mesh)
    groups, areas = components(mesh)
    if len(groups) < 2:
        return mesh
    total = float(areas.sum())
    kept = [g for g in groups if areas[g].sum() >= min_share * total] or [max(groups, key=lambda g: areas[g].sum())]
    if len(kept) == len(groups):
        return mesh
    return mesh.submesh([np.concatenate(kept)], append=True)
