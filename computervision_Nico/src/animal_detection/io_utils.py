"""Image loading and debug-image saving utilities."""

from pathlib import Path

import cv2
import numpy as np


def load_image(path: Path) -> np.ndarray:
    """Load a BGR image from *path*; raise if the file cannot be read."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def save_debug(output_dir: Path, index: int, name: str, image: np.ndarray) -> None:
    """Save *image* to *output_dir* with a numbered prefix, e.g. 01_distance.jpg."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"{index:02d}_{name}.jpg"
    cv2.imwrite(str(filename), image)
