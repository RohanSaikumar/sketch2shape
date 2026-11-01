"""Stage 1: turn a canvas drawing or a photo of a paper sketch into clean, centred line art.

Steps: flatten transparency onto white, invert light-on-dark sketches, even out paper lighting,
threshold, drop specks, crop to the strokes with padding, square to 1024 px, close tiny gaps,
thicken thin pencil lines, and turn dense hatching into the outline of the shaded area. Also builds
a filled silhouette of the sketch for the fidelity score.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from ..contracts import SketchAsset
from .imageops import fill_holes, remove_small_components


class EmptySketchError(ValueError):
    """The input holds no usable strokes."""


def to_rgb_on_white(image) -> np.ndarray:
    if isinstance(image, dict):                    # Gradio Sketchpad value
        image = image.get("composite")
    if image is None:
        raise EmptySketchError("No sketch was provided. Draw on the canvas or upload an image.")
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return np.array(Image.alpha_composite(white, rgba).convert("RGB"))


def extract_strokes(rgb: np.ndarray, speck_area_px: int = 20) -> np.ndarray:
    """Binary stroke mask (255 = stroke)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if gray.mean() < 127:                          # light lines on a dark background
        gray = 255 - gray
    k = max(15, (min(gray.shape) // 40) | 1)
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    flat = cv2.divide(gray, background, scale=255)   # removes uneven paper lighting in photos
    flat = cv2.GaussianBlur(flat, (3, 3), 0)
    otsu, _ = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strokes = np.where(flat < min(otsu, 200), 255, 0).astype(np.uint8)
    scale = gray.size / (1024 * 1024)
    return remove_small_components(strokes, max(4, int(speck_area_px * scale)))


def crop_square(mask: np.ndarray, pad_ratio: float, size: int) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    w, h = x1 - x0, y1 - y0
    side = int(max(w, h) * (1 + 2 * pad_ratio)) + 2
    canvas = np.zeros((side, side), np.uint8)
    ox, oy = (side - w) // 2, (side - h) // 2
    canvas[oy:oy + h, ox:ox + w] = mask[y0:y1, x0:x1]
    resized = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    return np.where(resized > 40, 255, 0).astype(np.uint8)


def tidy_strokes(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)     # joins tiny gaps between strokes
    return cv2.dilate(mask, kernel, iterations=1)              # keeps thin pencil lines visible to the ControlNet


def outline_hatching(strokes: np.ndarray, density: float = 0.5) -> np.ndarray:
    """Replace areas of dense hatching or scribbled shading with the outline of the area.

    The scribble ControlNet copies hatch strokes literally (a football came out covered in scribbles);
    the outline keeps the area's shape, and the prompt says what the object is, so the model paints it
    (black panels on a football). Tested on real sketches: only the football's panels and the landscape's
    shaded trees changed; the headphones, watering can and mug lines stay as they are."""
    size = strokes.shape[0]
    window, shrink = int(size * 0.05) | 1, int(size * 0.02) | 1     # 51 and 21 px at 1024
    on = (strokes > 0).astype(np.uint8)
    dense = (cv2.blur(on.astype(np.float32), (window, window)) > density).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dense)
    big = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 0.002 * dense.size]
    if not big:
        return strokes
    # the dense area reaches about half a window past the hatching: shrink it back to the shaded area
    region = cv2.erode(np.isin(labels, big).astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shrink, shrink)))
    region = fill_holes(region) > 0
    kept = np.where(region, 0, strokes).astype(np.uint8)
    kept = remove_small_components(kept, int(400 * (size / 1024) ** 2))     # hatch ends left at the edge
    outline = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_GRADIENT,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    kept[outline > 0] = 255
    return kept


def fill_silhouette(strokes: np.ndarray) -> np.ndarray:
    closed = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    filled = fill_holes(closed)
    if np.count_nonzero(filled) < 1.5 * np.count_nonzero(closed):   # open outline: use the convex hull
        hull = cv2.convexHull(cv2.findNonZero(strokes))
        filled = np.zeros_like(strokes)
        cv2.fillPoly(filled, [hull], 255)
    return filled


def preprocess(image, store, cfg, source: str = "upload") -> SketchAsset:
    p = cfg.preprocess
    rgb = to_rgb_on_white(image)
    store.save_image(Image.fromarray(rgb), "01_input.png")
    strokes = extract_strokes(rgb, int(p.speck_area_px))
    ratio = np.count_nonzero(strokes) / strokes.size
    if ratio < float(p.min_stroke_ratio):
        raise EmptySketchError("The sketch looks empty. Draw or upload a sketch with clearly visible lines.")
    strokes = tidy_strokes(crop_square(strokes, float(p.pad_ratio), int(p.size)))
    if p.outline_hatching:
        strokes = outline_hatching(strokes)
    lineart = 255 - strokes
    control = lineart if cfg.generate.control_polarity == "bw" else strokes
    return SketchAsset(
        lineart_path=store.save_image(Image.fromarray(lineart).convert("RGB"), "02_lineart.png"),
        control_path=store.save_image(Image.fromarray(control).convert("RGB"), "03_control.png"),
        mask_path=store.save_image(Image.fromarray(fill_silhouette(strokes)), "04_sketch_mask.png"),
        stroke_ratio=round(float(ratio), 5),
        source=source,
    )


# which edge the top of the drawing points to -> counter-clockwise quarter turns that make it upright
UPRIGHT_TURNS = {"right": 1, "bottom": 2, "left": 3}


def turn_upright(sketch: SketchAsset, up: str) -> int:
    """Rotate the saved line art, control image and mask in place when the naming call found the drawing
    sideways or upside down (e.g. a phone photo taken sideways). Returns the quarter turns applied."""
    turns = UPRIGHT_TURNS.get(up, 0)
    if turns:
        for path in {sketch.lineart_path, sketch.control_path, sketch.mask_path}:
            with Image.open(path) as image:
                turned = image.rotate(90 * turns, expand=True)   # PIL turns counter-clockwise
            turned.save(path)
    return turns
