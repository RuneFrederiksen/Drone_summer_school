"""Threshold a distance image to binary and clean up with morphology."""

import cv2
import numpy as np


def tophat(distance_img: np.ndarray, ksize: int) -> np.ndarray:
    """Morphological top-hat: remove large-scale background structure.

    Subtracts the morphological opening (local minimum envelope) from the
    image, keeping only bright features smaller than *ksize* pixels. This
    suppresses the road, slow grass-colour gradients, and uneven illumination
    so that Otsu sees a clean animal-vs-background histogram.

    *ksize* should be clearly larger than the widest animal you want to keep.
    Pass 0 to skip.
    """
    if ksize <= 0:
        return distance_img
    k = ksize if ksize % 2 == 1 else ksize + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(distance_img, cv2.MORPH_TOPHAT, kernel)


def blur(distance_img: np.ndarray, ksize: int = 9) -> np.ndarray:
    """Gaussian-blur the distance image to suppress grass texture noise.

    Individual grass pixels that vary slightly in colour average back toward
    zero; animal blobs (many pixels wide) stay bright. ksize must be odd and
    >= 1; pass 0 or 1 to skip blurring entirely.
    """
    if ksize <= 1:
        return distance_img
    k = ksize if ksize % 2 == 1 else ksize + 1  # enforce odd
    return cv2.GaussianBlur(distance_img, (k, k), 0)


def threshold(
    distance_img: np.ndarray,
    method: str = "otsu",
    value: int | None = None,
) -> np.ndarray:
    """Convert *distance_img* to a binary (0/255) image.

    Args:
        distance_img: uint8 grayscale distance image (blur it first for noisy scenes).
        method:       ``"otsu"`` for automatic Otsu thresholding, or
                      ``"fixed"`` to use the explicit *value*.
        value:        Threshold level (0–255) used when *method* is ``"fixed"``.
    """
    if method == "otsu":
        _, binary = cv2.threshold(
            distance_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    elif method == "triangle":
        # Better than Otsu when foreground objects are sparse (skewed histogram).
        _, binary = cv2.threshold(
            distance_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
        )
    elif method == "fixed":
        if value is None:
            raise ValueError("method='fixed' requires a threshold value")
        _, binary = cv2.threshold(distance_img, value, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError(f"Unknown threshold method: {method!r}")
    return binary


def clean(
    binary: np.ndarray,
    open_ksize: int = 3,
    close_ksize: int = 5,
) -> np.ndarray:
    """Remove speckle noise (opening) and fill holes (closing).

    Args:
        binary:      uint8 binary image (0/255).
        open_ksize:  Kernel size for morphological opening.
        close_ksize: Kernel size for morphological closing.
    """
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)
    return closed
