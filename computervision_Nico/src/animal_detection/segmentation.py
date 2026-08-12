"""Colour-based segmentation: compute a per-pixel distance from the grass colour."""

from __future__ import annotations

import cv2
import numpy as np


def estimate_grass_colour(
    image: np.ndarray,
    method: str = "median",
    sample: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Return the representative grass colour in BGR.

    Args:
        image:  BGR image.
        method: ``"median"`` – per-channel median over the whole image (or the
                *sample* region when provided).
        sample: Optional ``(x, y, w, h)`` rectangle of known-grass pixels.
    """
    region = image
    if sample is not None:
        x, y, w, h = sample
        region = image[y : y + h, x : x + w]

    if method == "median":
        pixels = region.reshape(-1, 3).astype(np.float32)
        return np.median(pixels, axis=0)

    raise ValueError(f"Unknown grass estimation method: {method!r}")


def colour_distance(
    image: np.ndarray,
    grass_colour_bgr: np.ndarray,
    space: str = "lab",
    use_lightness: bool = False,
) -> np.ndarray:
    """Return a uint8 distance image (0 = grass, 255 = maximally different).

    Args:
        image:             BGR source image.
        grass_colour_bgr:  Reference grass colour in BGR (shape ``(3,)``).
        space:             ``"lab"``, ``"hsv"``, or ``"bgr"``.
        use_lightness:     When *space* is ``"lab"``, include the L* channel in
                           the distance (default: chroma-only, a*b* plane).
    """
    if space == "lab":
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2Lab).astype(np.float32)
        ref_pixel = np.array([[grass_colour_bgr]], dtype=np.uint8)
        ref = cv2.cvtColor(ref_pixel, cv2.COLOR_BGR2Lab).astype(np.float32)[0, 0]
        if use_lightness:
            diff = converted - ref
        else:
            diff = converted[:, :, 1:] - ref[1:]  # a*b* only

    elif space == "hsv":
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        ref_pixel = np.array([[grass_colour_bgr]], dtype=np.uint8)
        ref = cv2.cvtColor(ref_pixel, cv2.COLOR_BGR2HSV).astype(np.float32)[0, 0]
        diff = converted - ref

    elif space == "bgr":
        converted = image.astype(np.float32)
        ref = grass_colour_bgr.astype(np.float32)
        diff = converted - ref

    else:
        raise ValueError(f"Unknown colour space: {space!r}")

    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
    return dist_norm.astype(np.uint8)
