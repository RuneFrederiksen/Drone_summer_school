"""Geofence exclusion zones for filtering GPS detections.

Usage: define one or more polygons in a plain-text file.  Any detection
whose GPS coordinate falls inside a polygon is discarded.

File format (one zone per blank-line-separated block):

    # Runway 06/24 - Odense Airport
    55.4748,10.3162
    55.4752,10.3162
    55.4752,10.3440
    55.4748,10.3440

Lines starting with '#' are zone names.  Coordinates are decimal-degree
lat,lon pairs, one per line.  The polygon does NOT need to be closed
(first == last point); the algorithm handles open polygons correctly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExclusionZone:
    name: str
    polygon: list[tuple[float, float]]  # [(lat, lon), ...]


def _point_in_polygon(lat: float, lon: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test (works for any simple polygon)."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        lat_i, lon_i = polygon[i]
        lat_j, lon_j = polygon[j]
        if ((lon_i > lon) != (lon_j > lon)) and (
            lat < (lat_j - lat_i) * (lon - lon_i) / (lon_j - lon_i) + lat_i
        ):
            inside = not inside
        j = i
    return inside


def is_in_any_zone(lat: float, lon: float, zones: list[ExclusionZone]) -> bool:
    """Return True if (lat, lon) falls inside any exclusion zone."""
    return any(_point_in_polygon(lat, lon, z.polygon) for z in zones)


def load_exclusion_zones(path: Path) -> list[ExclusionZone]:
    """Load exclusion zones from a plain-text file (see module docstring for format)."""
    zones: list[ExclusionZone] = []
    current_name = "unnamed"
    current_points: list[tuple[float, float]] = []

    def _flush() -> None:
        if len(current_points) >= 3:
            zones.append(ExclusionZone(name=current_name, polygon=current_points[:]))

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                _flush()
                current_name = "unnamed"
                current_points = []
                continue
            if line.startswith("#"):
                # Flush any previous zone if we get a new name without a blank line
                if current_points:
                    _flush()
                    current_points = []
                current_name = line[1:].strip()
                continue
            try:
                parts = line.split(",")
                lat = float(parts[0].strip())
                lon = float(parts[1].strip())
                current_points.append((lat, lon))
            except (ValueError, IndexError):
                continue

    _flush()
    return zones
