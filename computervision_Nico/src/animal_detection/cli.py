"""Command-line entry point for drone animal detection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from animal_detection import annotate as ann
from animal_detection import io_utils
from animal_detection.gps_utils import (
    RPI_CAM_V2_FOV_H,
    read_gps_log,
    pixel_to_gps,
)
from animal_detection.geo_filter import ExclusionZone, is_in_any_zone, load_exclusion_zones
from animal_detection.mapping import DetectionPin, generate_map
from animal_detection.pipeline import Config, detect_animals

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
_DEFAULT_GPS_LOG = "gps_locations.log"
_DEFAULT_ZONE_FILE = "ekod_runway_06_24.csv"


def _collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        images = sorted(
            p for p in input_path.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES
        )
        if not images:
            print(f"No images found in {input_path}", file=sys.stderr)
            sys.exit(1)
        return images
    print(f"Input not found: {input_path}", file=sys.stderr)
    sys.exit(1)


def _build_config(args: argparse.Namespace) -> Config:
    threshold_method = "otsu"
    threshold_value = None
    if args.threshold in ("otsu", "triangle"):
        threshold_method = args.threshold
    else:
        try:
            threshold_value = int(args.threshold)
            threshold_method = "fixed"
        except ValueError:
            print(
                f"--threshold must be 'otsu', 'triangle', or an integer 0–255, "
                f"got {args.threshold!r}",
                file=sys.stderr,
            )
            sys.exit(1)

    return Config(
        space="lab",
        use_lightness=False,
        tophat_ksize=args.tophat_ksize,
        blur_ksize=args.blur_ksize,
        threshold_method=threshold_method,
        threshold_value=threshold_value,
        open_ksize=args.open_ksize,
        close_ksize=args.close_ksize,
        min_area=args.min_area,
        max_area=args.max_area,
        border_margin=args.border_margin,
    )


def _run_dataset(
    input_path: Path,
    output_dir: Path,
    gps_log_path: Path | None,
    config: Config,
    exclusion_zones: list[ExclusionZone],
    min_altitude: float,
) -> None:
    gps_data: dict = {}
    if gps_log_path and gps_log_path.exists():
        gps_data = read_gps_log(gps_log_path)
        print(f"  GPS: {len(gps_data)} image(s) from {gps_log_path.name}")
    elif gps_log_path:
        print(f"  GPS log not found: {gps_log_path} — map will be skipped")

    images = _collect_images(input_path)
    all_pins: list[DetectionPin] = []

    for img_path in images:
        try:
            image = io_utils.load_image(img_path)
        except ValueError as e:
            print(f"\n{img_path.name}: skipped ({e})")
            continue
        img_output_dir = output_dir / img_path.stem if len(images) > 1 else output_dir
        _, blobs = detect_animals(image, config, output_dir=img_output_dir)
        ann.print_report(img_path.name, blobs)

        if not blobs or not gps_data:
            continue

        gps = gps_data.get(img_path.name)
        if gps is None:
            print(f"  (no GPS data for {img_path.name} — skipped in map)")
            continue

        if gps.altitude_m < min_altitude:
            print(f"  (altitude {gps.altitude_m:.1f} m < {min_altitude:.0f} m minimum — skipped)")
            continue

        h, w = image.shape[:2]
        for idx, blob in enumerate(blobs):
            lat, lon = pixel_to_gps(blob.centroid, (w, h), gps, fov_h_deg=RPI_CAM_V2_FOV_H)
            if exclusion_zones and is_in_any_zone(lat, lon, exclusion_zones):
                print(f"  #{idx + 1}  excluded (GPS {lat:.6f}, {lon:.6f} inside exclusion zone)")
                continue
            bx, by, bw, bh = blob.bbox
            all_pins.append(DetectionPin(
                image_name=img_path.name,
                index=idx + 1,
                lat=lat,
                lon=lon,
                width_px=bw,
                height_px=bh,
            ))

    if all_pins:
        map_path = output_dir / "detections_map.html"
        generate_map(all_pins, map_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect objects in drone images and map their GPS positions. "
                    "With no arguments, auto-discovers capture folders inside input/."
    )

    # --- Input / output ---
    parser.add_argument("--input", default=None,
                        help="Image file or folder. Defaults to auto-discovering "
                             "capture folders in input/.")
    parser.add_argument("--output", default="output",
                        help="Output folder (default: output/).")

    # --- Detection pipeline ---
    parser.add_argument("--tophat", type=int, default=301, dest="tophat_ksize",
                        help="Top-hat kernel size (default 301).")
    parser.add_argument("--blur", type=int, default=21, dest="blur_ksize",
                        help="Gaussian blur kernel size (default 21).")
    parser.add_argument("--threshold", default="triangle",
                        help="'otsu', 'triangle' (default), or a fixed integer 0–255.")
    parser.add_argument("--min-area", type=int, default=17000, dest="min_area")
    parser.add_argument("--max-area", type=int, default=100_000, dest="max_area")
    parser.add_argument("--open", type=int, default=3, dest="open_ksize")
    parser.add_argument("--close", type=int, default=5, dest="close_ksize")
    parser.add_argument("--border-margin", type=int, default=20, dest="border_margin",
                        help="Reject blobs within this many pixels of the image edge.")

    # --- GPS / mapping ---
    parser.add_argument("--gps-log", default=None, dest="gps_log",
                        help="GPS log file. Auto-discovered as gps_locations.log "
                             "inside the input folder if not specified.")
    parser.add_argument("--exclude-zone", default=None, dest="exclude_zone",
                        help="GPS exclusion polygon file. Auto-discovered as "
                             "input/ekod_runway_06_24.csv if present.")
    parser.add_argument("--min-altitude", type=float, default=0.0, dest="min_altitude",
                        help="Skip images whose GPS altitude MSL (m) is below this value "
                             "(default 0, disabled).")

    args = parser.parse_args()
    config = _build_config(args)
    output_base = Path(args.output)

    # Auto-discover exclusion zone
    exclusion_zones: list[ExclusionZone] = []
    zone_path_str = args.exclude_zone
    if zone_path_str is None:
        auto_zone = Path("input") / _DEFAULT_ZONE_FILE
        if auto_zone.exists():
            zone_path_str = str(auto_zone)
            print(f"Auto-discovered exclusion zone: {auto_zone}")
    if zone_path_str:
        zone_path = Path(zone_path_str)
        if not zone_path.exists():
            print(f"Exclusion zone file not found: {zone_path}", file=sys.stderr)
            sys.exit(1)
        exclusion_zones = load_exclusion_zones(zone_path)
        names = [z.name for z in exclusion_zones]
        print(f"Loaded {len(exclusion_zones)} exclusion zone(s): {', '.join(names)}")

    # Resolve datasets to process
    if args.input is not None:
        input_path = Path(args.input)
        gps_log_path = Path(args.gps_log) if args.gps_log else input_path / _DEFAULT_GPS_LOG
        datasets = [(input_path, gps_log_path)]
        outputs = [output_base]
    else:
        input_base = Path("input")
        if not input_base.exists():
            print("No --input specified and 'input/' folder not found.", file=sys.stderr)
            sys.exit(1)
        datasets = []
        outputs = []
        for d in sorted(input_base.iterdir()):
            if not d.is_dir():
                continue
            if any(p.suffix.lower() in _IMAGE_SUFFIXES for p in d.iterdir()):
                datasets.append((d, d / _DEFAULT_GPS_LOG))
                outputs.append(output_base / d.name)
        if not datasets:
            print("No image folders found in 'input/'. Use --input to specify a path.", file=sys.stderr)
            sys.exit(1)
        print(f"Auto-discovered {len(datasets)} dataset(s): {', '.join(d.name for d, _ in datasets)}")

    for (input_path, gps_log_path), output_dir in zip(datasets, outputs):
        print(f"\n--- {input_path.name} ---")
        _run_dataset(input_path, output_dir, gps_log_path, config, exclusion_zones, args.min_altitude)


if __name__ == "__main__":
    main()
