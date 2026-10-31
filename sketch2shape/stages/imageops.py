"""Small OpenCV helpers shared by several stages. Masks are uint8 arrays with 255 = inside."""
from __future__ import annotations

import cv2
import numpy as np


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill every region the border cannot reach, so an outline becomes a solid silhouette."""
    flood = np.ascontiguousarray(np.where(mask > 0, 255, 0).astype(np.uint8))
    h, w = flood.shape
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1:-1, 1:-1] = flood
    cv2.floodFill(padded, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 128)
    return np.where(padded[1:-1, 1:-1] == 128, 0, 255).astype(np.uint8)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(np.where(mask > 0, 255, 0).astype(np.uint8), 8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def foreground_mask(rgb: np.ndarray, threshold: int = 28) -> np.ndarray:
    """Object mask for an image on a plain background: pixels that differ from the border colour."""
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]).astype(np.int16)
    background = np.median(border, axis=0)
    diff = np.abs(rgb.astype(np.int16) - background).max(axis=2)
    mask = np.where(diff > threshold, 255, 0).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return fill_holes(mask)
