"""Draw detection markers on the original image and print a report."""

from __future__ import annotations

import cv2
import numpy as np

from animal_detection.blobs import Blob

_COLOUR = (255, 0, 255)  # magenta — visible against grass/green background
_THICKNESS = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.6


def annotate(image: np.ndarray, blobs: list[Blob]) -> np.ndarray:
    """Return a copy of *image* with a bounding box, centroid dot, and index label per blob."""
    out = image.copy()
    for idx, blob in enumerate(blobs):
        x, y, w, h = blob.bbox
        cx, cy = int(blob.centroid[0]), int(blob.centroid[1])
        cv2.rectangle(out, (x, y), (x + w, y + h), _COLOUR, _THICKNESS)
        cv2.circle(out, (cx, cy), 6, _COLOUR, -1)
        label = f"#{idx + 1}"
        cv2.putText(out, label, (x, y - 8), _FONT, _FONT_SCALE, _COLOUR, _THICKNESS)
    return out


def print_report(filename: str, blobs: list[Blob]) -> None:
    """Print a summary line followed by per-detection details."""
    print(f"\n{filename}: {len(blobs)} object(s) detected")
    for idx, blob in enumerate(blobs):
        cx, cy = blob.centroid
        x, y, w, h = blob.bbox
        print(f"  #{idx + 1}  centre ({cx:.0f}, {cy:.0f})  size {w}x{h} px  area {blob.area} px")
