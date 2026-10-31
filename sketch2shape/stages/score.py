"""Stage 4 (quality gate 1): rank candidate images by how faithfully they follow the sketch.

Three measurements, combined with the weights in config.yaml:
  edges       how many sketch lines reappear as edges in the image (recall-weighted F2 score)
  silhouette  overlap between the filled sketch outline and the object's cut-out
  clip        whether CLIP still sees the named object in the image (only when a name is known)
The ControlNet keeps the sketch's framing, so sketch and image line up pixel for pixel.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from PIL import Image

from ..vocab import OBJECTS
from .imageops import foreground_mask

SIZE = 512


def load_mask(path, invert: bool = False) -> np.ndarray:
    gray = np.array(Image.open(path).convert("L").resize((SIZE, SIZE), Image.NEAREST))
    return (gray < 128) if invert else (gray >= 128)


def stroke_fidelity(strokes: np.ndarray, edges: np.ndarray, tolerance_px: float) -> dict:
    """Recall: sketch pixels near an image edge. Precision: image edges near a sketch line."""
    if not strokes.any() or not edges.any():
        return {"recall": 0.0, "precision": 0.0, "f2": 0.0}
    to_edge = cv2.distanceTransform(np.where(edges, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
    to_stroke = cv2.distanceTransform(np.where(strokes, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
    recall = float((to_edge[strokes] <= tolerance_px).mean())
    precision = float((to_stroke[edges] <= tolerance_px).mean())
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    return {"recall": round(recall, 3), "precision": round(precision, 3), "f2": round(f2, 3)}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def candidate_mask(models, candidate, store) -> np.ndarray:
    """Object cut-out via background removal (saved for the prepare stage), else a colour threshold."""
    rgb = Image.open(candidate.path).convert("RGB")
    try:
        rgba = models.bg_remover()(rgb)
        candidate.cutout_path = store.save_image(rgba, candidate.path.stem + "_cutout.png")
        alpha = np.array(rgba.getchannel("A").resize((SIZE, SIZE)))
        return alpha >= 128
    except Exception:
        return foreground_mask(np.array(rgb.resize((SIZE, SIZE)))) > 0


def candidate_edges(candidate, object_mask: np.ndarray) -> np.ndarray:
    gray = np.array(Image.open(candidate.path).convert("L").resize((SIZE, SIZE)))
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150) > 0
    near_object = cv2.dilate(object_mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    return edges & near_object            # ignore background texture and stray shadows


def clip_agreement(models, intent, candidates) -> list[float]:
    from .understand import clip_zero_shot
    labels = [intent.label] + [o for o in OBJECTS if o != intent.label]
    images = [Image.open(c.path).convert("RGB") for c in candidates]
    probs = clip_zero_shot(models, images, labels, "a photo of {}")
    return [1.0 if p.argmax() == 0 else float(p[0] / p.max()) for p in probs]


def score_candidates(models, sketch, intent, candidates, cfg, store) -> None:
    """Fills candidate.scores and candidate.total in place."""
    if not candidates:
        return
    s = cfg.score
    strokes = load_mask(sketch.lineart_path, invert=True)
    sketch_shape = load_mask(sketch.mask_path)
    tolerance = float(s.edge_tolerance) * SIZE
    timings = store.manifest.setdefault("score_timings_s", [])
    clip_scores = None
    if intent.label:
        started = time.monotonic()
        try:
            clip_scores = clip_agreement(models, intent, candidates)
        except Exception as exc:
            store.warn(f"CLIP agreement skipped: {exc!r}")
        timings.append({"clip": round(time.monotonic() - started, 1)})
    weights = vars(s.weights)
    for i, candidate in enumerate(candidates):
        started = time.monotonic()
        mask = candidate_mask(models, candidate, store)
        timings.append({f"cutout_{candidate.path.stem}": round(time.monotonic() - started, 1)})
        fidelity = stroke_fidelity(strokes, candidate_edges(candidate, mask), tolerance)
        candidate.scores = {"edges": fidelity["f2"], "edge_recall": fidelity["recall"],
                            "edge_precision": fidelity["precision"],
                            "silhouette": round(iou(sketch_shape, mask), 3)}
        if clip_scores is not None:
            candidate.scores["clip"] = round(clip_scores[i], 3)
        used = {k: w for k, w in weights.items() if k in candidate.scores}
        candidate.total = round(sum(w * candidate.scores[k] for k, w in used.items()) / sum(used.values()), 3)
