"""Generate an interactive Leaflet.js HTML map of animal detections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DetectionPin:
    image_name: str
    index: int        # 1-based detection number within the image
    lat: float
    lon: float
    width_px: int
    height_px: int


_PALETTE = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
]


def generate_map(pins: list[DetectionPin], output_path: Path) -> None:
    """Write a standalone HTML map to *output_path*.

    The map centres on the mean detection position, shows one coloured
    circle marker per detection, and groups them by image in the legend.
    Each marker opens a popup with image name, detection index, and
    coordinates.  Requires an internet connection for the map tiles.
    """
    if not pins:
        print("  (no GPS detections — map not generated)")
        return

    mean_lat = sum(p.lat for p in pins) / len(pins)
    mean_lon = sum(p.lon for p in pins) / len(pins)

    images = sorted(set(p.image_name for p in pins))
    colour_of = {img: _PALETTE[i % len(_PALETTE)] for i, img in enumerate(images)}

    # --- Leaflet marker JS ---
    marker_lines: list[str] = []
    for p in pins:
        c = colour_of[p.image_name]
        popup = (
            f"{p.image_name}<br>"
            f"Detection #{p.index}<br>"
            f"{p.lat:.7f}, {p.lon:.7f}<br>"
            f"Size: {p.width_px}&times;{p.height_px} px"
        )
        marker_lines.append(
            f'    L.circleMarker([{p.lat:.7f}, {p.lon:.7f}], '
            f'{{radius:10, color:"{c}", fillColor:"{c}", fillOpacity:0.85, weight:2}})'
            f'.addTo(map).bindPopup("{popup}");'
        )

    # --- Legend HTML ---
    legend_rows = "".join(
        f'<div style="display:flex;align-items:center;gap:7px;margin:3px 0">'
        f'<span style="background:{colour_of[img]};width:13px;height:13px;'
        f'border-radius:50%;display:inline-block;flex-shrink:0"></span>'
        f'<span style="font-size:12px">{img}</span></div>'
        for img in images
    )

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Animal Detection Map</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body {{ margin: 0; height: 100%; }}
    #map {{ height: 100vh; }}
    #legend {{
      position: absolute; bottom: 28px; right: 10px; z-index: 1000;
      background: white; padding: 10px 14px; border-radius: 7px;
      box-shadow: 0 2px 8px rgba(0,0,0,.35);
      font-family: sans-serif; line-height: 1.4; max-width: 260px;
    }}
    #legend strong {{ display:block; margin-bottom:5px; font-size:13px; }}
  </style>
</head>
<body>
<div id="map"></div>
<div id="legend">
  <strong>Detections by image</strong>
  {legend_rows}
</div>
<script>
  var map = L.map('map').setView([{mean_lat:.7f}, {mean_lon:.7f}], 19);
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    maxZoom: 22,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
  }}).addTo(map);
{chr(10).join(marker_lines)}
</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"  Map saved: {output_path}")
