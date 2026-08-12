"""Connected-component analysis: binary image → list of Blob objects."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Blob:
    label: int
    area: int
    bbox: tuple[int, int, int, int]   # (x, y, w, h)
    centroid: tuple[float, float]      # (cx, cy)


def find_blobs(binary: np.ndarray) -> list[Blob]:
    """Return one Blob per connected component in *binary* (background excluded)."""
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    blobs: list[Blob] = []
    for label in range(1, num_labels):  # 0 is background
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = float(centroids[label, 0]), float(centroids[label, 1])
        blobs.append(Blob(label=label, area=area, bbox=(x, y, w, h), centroid=(cx, cy)))
    return blobs
