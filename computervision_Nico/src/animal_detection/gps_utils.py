"""GPS extraction from image EXIF and pixel-to-GPS coordinate conversion.

Typical workflow with a Pixhawk + RPi Camera v2:
  1. After the flight, open QGroundControl → Analyze → GeoTag Images.
  2. Point it at the telemetry log (.ulg / .bin) and the photo folder.
  3. QGC embeds standard GPS EXIF into each photo.
  4. This module then reads those EXIF tags automatically.

Fallback: supply a CSV file (see read_gps_csv) when images aren't geotagged yet.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GpsPoint:
    lat: float          # decimal degrees (positive = North)
    lon: float          # decimal degrees (positive = East)
    altitude_m: float   # metres above take-off or sea level
    yaw_deg: float = 0.0  # drone heading clockwise from North; 0 = North


# ---------------------------------------------------------------------------
# EXIF reading
# ---------------------------------------------------------------------------

def _rational_to_float(r) -> float:
    """Handle both IFDRational objects and (num, den) tuples."""
    if hasattr(r, "numerator"):
        return r.numerator / r.denominator if r.denominator else 0.0
    num, den = r
    return num / den if den else 0.0


def _dms_to_decimal(dms, ref: str) -> float:
    deg = _rational_to_float(dms[0])
    mn = _rational_to_float(dms[1])
    sec = _rational_to_float(dms[2])
    val = deg + mn / 60.0 + sec / 3600.0
    return -val if ref in ("S", "W") else val


def read_gps_exif(image_path: Path) -> GpsPoint | None:
    """Return GPS data embedded in *image_path*'s EXIF, or None if absent."""
    try:
        from PIL import Image
        from PIL.ExifTags import GPSTAGS

        with Image.open(image_path) as img:
            exif = img.getexif()
            if exif is None:
                return None
            gps_ifd = exif.get_ifd(0x8825)  # GPS IFD tag
            if not gps_ifd:
                return None
            gps = {GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}

            if "GPSLatitude" not in gps or "GPSLongitude" not in gps:
                return None

            lat = _dms_to_decimal(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
            lon = _dms_to_decimal(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
            alt_raw = gps.get("GPSAltitude")
            altitude = _rational_to_float(alt_raw) if alt_raw is not None else 0.0

            return GpsPoint(lat=lat, lon=lon, altitude_m=altitude)

    except Exception:
        return None


# ---------------------------------------------------------------------------
# CSV fallback
# ---------------------------------------------------------------------------

def read_gps_log(log_path: Path) -> dict[str, GpsPoint]:
    """Parse a Pixhawk-style GPS log where each line is:
    ``index,lat,lon,altitude_m,heading_deg``
    and maps to image file ``img_{index}.jpg``.
    """
    result: dict[str, GpsPoint] = {}
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                idx = int(parts[0])
                lat = float(parts[1])
                lon = float(parts[2])
                alt = float(parts[3])
                yaw = float(parts[4]) if len(parts) > 4 else 0.0
                result[f"img_{idx}.jpg"] = GpsPoint(lat=lat, lon=lon, altitude_m=alt, yaw_deg=yaw)
            except ValueError:
                continue
    return result


def read_gps_csv(csv_path: Path) -> dict[str, GpsPoint]:
    """Load GPS data from a CSV file keyed by filename.

    Expected columns (header required):
      filename, lat, lon, altitude_m, [heading_deg]

    Example::

      filename,lat,lon,altitude_m,heading_deg
      IMG_0001.jpg,59.1234,10.5678,30.0,0.0
    """
    result: dict[str, GpsPoint] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = Path(row["filename"]).name
            result[name] = GpsPoint(
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                altitude_m=float(row["altitude_m"]),
                yaw_deg=float(row.get("heading_deg", 0.0) or 0.0),
            )
    return result


# ---------------------------------------------------------------------------
# Pixel → GPS conversion
# ---------------------------------------------------------------------------

# Raspberry Pi Camera Module v2 (Sony IMX219) default FOV at full resolution.
RPI_CAM_V2_FOV_H = 62.2   # degrees horizontal
RPI_CAM_V2_FOV_V = 48.8   # degrees vertical


def pixel_to_gps(
    centroid: tuple[float, float],
    image_size: tuple[int, int],
    gps: GpsPoint,
    fov_h_deg: float = RPI_CAM_V2_FOV_H,
) -> tuple[float, float]:
    """Convert a pixel centroid to a GPS coordinate.

    Assumes the camera points straight down (nadir) and the image top is
    aligned with the drone's forward direction (yaw).

    Args:
        centroid:   (cx, cy) pixel position of the detection.
        image_size: (width, height) in pixels.
        gps:        Drone GPS + heading at the moment of capture.
        fov_h_deg:  Camera horizontal field of view in degrees.
                    Default is Raspberry Pi Camera v2 at full resolution.

    Returns:
        (latitude, longitude) in decimal degrees.
    """
    W, H = image_size
    cx, cy = centroid

    # Ground footprint width at this altitude, then GSD (metres per pixel)
    ground_width_m = 2.0 * gps.altitude_m * math.tan(math.radians(fov_h_deg / 2.0))
    gsd = ground_width_m / W  # assumes square pixels

    # Offset from image centre in the camera's local frame
    # Image x: right = positive; image y: down = positive → flip for North
    img_x_m = (cx - W / 2.0) * gsd   # rightward offset
    img_y_m = -(cy - H / 2.0) * gsd  # upward offset (positive = forward/North when yaw=0)

    # Rotate local camera frame by drone heading to get North / East offsets
    yaw = math.radians(gps.yaw_deg)
    north_m = img_y_m * math.cos(yaw) - img_x_m * math.sin(yaw)
    east_m  = img_y_m * math.sin(yaw) + img_x_m * math.cos(yaw)

    # Convert metres to decimal degrees
    delta_lat = north_m / 111_111.0
    delta_lon = east_m  / (111_111.0 * math.cos(math.radians(gps.lat)))

    return gps.lat + delta_lat, gps.lon + delta_lon
