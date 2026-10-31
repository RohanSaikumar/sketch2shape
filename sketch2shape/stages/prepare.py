"""Stage 5: isolate the chosen object and frame it the way the 3D model expects.

Uses the cut-out made while scoring when it looks sane; otherwise removes the background again,
and finally falls back to a colour threshold (the prompt asked for a plain white background).
The object is then cropped, centred with a margin and resized onto a transparent square.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from ..contracts import PreparedImage
from .imageops import foreground_mask


def alpha_ratio(rgba: Image.Image) -> float:
    return float((np.array(rgba.getchannel("A")) >= 128).mean())


def crop_and_pad(rgba: Image.Image, margin: float, size: int) -> Image.Image:
    alpha = np.array(rgba.getchannel("A"))
    ys, xs = np.nonzero(alpha > 10)
    crop = rgba.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    side = int(max(crop.size) * (1 + 2 * margin))
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2), crop)
    return canvas.resize((size, size), Image.LANCZOS)


def prepare(models, candidate, store, cfg) -> PreparedImage:
    p = cfg.prepare
    lo, hi = float(p.min_mask_ratio), float(p.max_mask_ratio)
    rgb = Image.open(candidate.path).convert("RGB")
    rgba, method = None, "rembg"
    if candidate.cutout_path is not None:
        rgba = Image.open(candidate.cutout_path).convert("RGBA")
    else:
        try:
            rgba = models.bg_remover()(rgb)
        except Exception as exc:
            store.warn(f"Background removal failed ({exc!r}); using a colour threshold.")
    if rgba is None or not lo <= alpha_ratio(rgba) <= hi:
        mask = foreground_mask(np.array(rgb))
        rgba = rgb.copy()
        rgba.putalpha(Image.fromarray(mask))
        method = "colour-threshold"
    ratio = alpha_ratio(rgba)
    rgba = crop_and_pad(rgba, float(p.margin), int(p.size))
    return PreparedImage(rgba_path=store.save_image(rgba, "06_prepared.png"),
                         mask_ratio=round(ratio, 3), method=method)
