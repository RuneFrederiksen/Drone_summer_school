"""Wire all pipeline stages together into a single detect_animals call."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from animal_detection import annotate as ann
from animal_detection import blobs as bl
from animal_detection import filtering as flt
from animal_detection import io_utils
from animal_detection import segmentation as seg
from animal_detection import thresholding as thr


@dataclass
class Config:
    # Segmentation
    space: str = "lab"
    use_lightness: bool = False
    grass_method: str = "median"
    grass_sample: tuple[int, int, int, int] | None = None

    # Pre-thresholding
    tophat_ksize: int = 0   # 0 = disabled; set > animal width to suppress background
    blur_ksize: int = 9

    # Thresholding
    threshold_method: str = "otsu"
    threshold_value: int | None = None

    # Morphology
    open_ksize: int = 3
    close_ksize: int = 5

    # Blob filtering
    min_area: int = 500
    max_area: int = 100_000
    min_extent: float | None = None
    aspect_range: tuple[float, float] | None = None
    border_margin: int = 20


def detect_animals(
    image: np.ndarray,
    config: Config,
    output_dir: Path | None = None,
) -> tuple[np.ndarray, list[bl.Blob]]:
    """Run the full detection pipeline and return (annotated_image, blobs).

    Three debug images are written to *output_dir* when provided:
      01_distance   – colour distance from grass (bright = different from background)
      02_threshold  – cleaned binary mask (white = detected foreground)
      03_detections – final annotated result
    When --tophat is active a 01b_tophat image is also written.
    """

    def _save(name: str, img: np.ndarray) -> None:
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            io_utils.save_debug(output_dir, _save.counter, name, img)
            _save.counter += 1

    _save.counter = 1

    grass = seg.estimate_grass_colour(
        image, method=config.grass_method, sample=config.grass_sample
    )
    distance_img = seg.colour_distance(
        image, grass, space=config.space, use_lightness=config.use_lightness
    )
    _save("distance", distance_img)

    preproc = thr.tophat(distance_img, ksize=config.tophat_ksize)
    if config.tophat_ksize > 0:
        _save("tophat", preproc)

    blurred = thr.blur(preproc, ksize=config.blur_ksize)

    binary = thr.threshold(
        blurred, method=config.threshold_method, value=config.threshold_value
    )
    cleaned = thr.clean(binary, open_ksize=config.open_ksize, close_ksize=config.close_ksize)
    _save("threshold", cleaned)

    raw_blobs = bl.find_blobs(cleaned)
    kept = flt.filter_blobs(
        raw_blobs,
        min_area=config.min_area,
        max_area=config.max_area,
        min_extent=config.min_extent,
        aspect_range=config.aspect_range,
        border_margin=config.border_margin,
        image_shape=image.shape[:2],
    )

    annotated = ann.annotate(image, kept)
    _save("detections", annotated)

    return annotated, kept
