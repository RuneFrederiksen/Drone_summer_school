"""Filter blobs by size and shape to keep only plausible animals."""

from __future__ import annotations

from animal_detection.blobs import Blob


def filter_blobs(
    blobs: list[Blob],
    min_area: int = 500,
    max_area: int = 100_000,
    min_extent: float | None = None,
    aspect_range: tuple[float, float] | None = None,
    border_margin: int = 0,
    image_shape: tuple[int, int] | None = None,
) -> list[Blob]:
    """Return only blobs that satisfy all active size/shape criteria.

    Args:
        blobs:          Input blob list from ``find_blobs``.
        min_area:       Minimum pixel count (removes noise).
        max_area:       Maximum pixel count (removes background artefacts).
        min_extent:     Optional minimum fill ratio: ``area / (w * h)``.
        aspect_range:   Optional ``(min_ratio, max_ratio)`` for ``w / h``.
        border_margin:  Reject blobs whose bbox touches within this many pixels
                        of the image edge (suppresses lens-vignette corner noise).
        image_shape:    ``(height, width)`` of the source image; required when
                        *border_margin* > 0.
    """
    img_h, img_w = image_shape if image_shape else (0, 0)

    kept: list[Blob] = []
    for blob in blobs:
        if not (min_area <= blob.area <= max_area):
            continue

        x, y, w, h = blob.bbox

        if border_margin > 0 and image_shape is not None:
            if x <= border_margin or y <= border_margin:
                continue
            if (x + w) >= (img_w - border_margin) or (y + h) >= (img_h - border_margin):
                continue

        if min_extent is not None:
            extent = blob.area / (w * h) if w * h > 0 else 0.0
            if extent < min_extent:
                continue

        if aspect_range is not None:
            ratio = w / h if h > 0 else float("inf")
            lo, hi = aspect_range
            if not (lo <= ratio <= hi):
                continue

        kept.append(blob)
    return kept
