"""
Step 2: Download satellite tiles using the best available source per region.

Source priority (best resolution → worst):
  France      → IGN Géoplateforme WMS       (0.2m, free, no key)
  USA         → NAIP via GEE                (0.6m, free, needs GEE project)
  EU / Turkey → ESRI World Imagery REST     (0.3–0.5m in cities, free, no key)
  Global      → Sentinel-2 via GEE          (10m, free, last resort)

Bounding boxes used for auto-detection (override via sample["satellite_source"]):
  NAIP  → continental US + Alaska + Hawaii
  IGN   → metropolitan France + Corsica
  ESRI  → everywhere else (Europe, Turkey, Middle East, Asia, …)
  S2    → automatic fallback if all others fail
"""

import io
import math
import os
import time
import urllib.request
import urllib.parse

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Region bounding boxes for auto source detection
# ---------------------------------------------------------------------------
from config import detect_sat_source as detect_source


# ---------------------------------------------------------------------------
# GEE lazy init
# ---------------------------------------------------------------------------

_ee = None


def get_ee():
    global _ee
    if _ee is None:
        import ee
        from config import GEE_PROJECT
        ee.Initialize(project=GEE_PROJECT)
        _ee = ee
    return _ee


# ---------------------------------------------------------------------------
# Source 1: NAIP — USA, 0.6 m/px, free via GEE
# ---------------------------------------------------------------------------

def fetch_naip_tile(lat, lon, sample_id, buffer_m=None, px=None):
    """Download a NAIP tile (USA only, 0.6 m resolution)."""
    from config import SAT_BUFFER_M, SAT_IMAGE_PX, NAIP_DATE_RANGE
    buffer_m = buffer_m or SAT_BUFFER_M.get("NAIP", 250)  # config-driven zoom
    px = px or SAT_IMAGE_PX
    ee = get_ee()

    point = ee.Geometry.Point([lon, lat])
    col = (ee.ImageCollection("USDA/NAIP/DOQQ")
           .filterBounds(point)
           .filterDate(*NAIP_DATE_RANGE)
           .sort("system:time_start", False))
    naip = col.mosaic()
    region = point.buffer(buffer_m).bounds()

    url = naip.getThumbURL({
        'region': region.getInfo()['coordinates'],
        'dimensions': f'{px}x{px}',
        'bands': ['R', 'G', 'B'],
        'min': 0, 'max': 255,
        'format': 'png',
    })

    save_path = _save_path(sample_id)
    urllib.request.urlretrieve(url, save_path)

    try:
        date_info = col.first().date().format("YYYY-MM").getInfo()
    except Exception:
        date_info = "unknown"

    return save_path, date_info, "NAIP"


# ---------------------------------------------------------------------------
# Source 2: IGN Géoplateforme — France only, 0.2 m/px, free, no key needed
# ---------------------------------------------------------------------------

# New IGN Géoplateforme endpoint (replaced the old wxs.ign.fr in 2023)
_IGN_WMS = "https://data.geopf.fr/wms-r/wms"

def fetch_ign_tile(lat, lon, sample_id, buffer_m=None, px=None):
    """
    Download an IGN orthophoto tile (France only, 0.2 m resolution).

    Uses the IGN Géoplateforme WMS — completely free, no API key required.
    Layer: HR.ORTHOIMAGERY.ORTHOPHOTOS (0.2 m aerial survey).
    """
    from config import SAT_BUFFER_M, SAT_IMAGE_PX
    px = px or SAT_IMAGE_PX

    # Compute bounding box in degrees from a metre buffer
    # 1 degree latitude ≈ 111 320 m; longitude shrinks with cos(lat)
    buffer_m = buffer_m or SAT_BUFFER_M.get("IGN", 200)  # config-driven zoom
    dlat = buffer_m / 111_320
    dlon = buffer_m / (111_320 * math.cos(math.radians(lat)))

    bbox = f"{lon - dlon},{lat - dlat},{lon + dlon},{lat + dlat}"

    params = urllib.parse.urlencode({
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": "HR.ORTHOIMAGERY.ORTHOPHOTOS",
        "STYLES": "",
        "CRS": "CRS:84",          # lon/lat order (WGS84)
        "BBOX": bbox,
        "WIDTH": px,
        "HEIGHT": px,
        "FORMAT": "image/png",
    })
    url = f"{_IGN_WMS}?{params}"

    save_path = _save_path(sample_id)
    _download(url, save_path)

    return save_path, "latest", "IGN"


# ---------------------------------------------------------------------------
# Source 3: ESRI World Imagery — Global, ~0.3–0.5 m in cities, free, no key
# ---------------------------------------------------------------------------

# ESRI's MapServer export endpoint — returns a georeferenced PNG for any bbox
_ESRI_EXPORT = (
    "https://server.arcgisonline.com/arcgis/rest/services/"
    "World_Imagery/MapServer/export"
)

def fetch_esri_tile(lat, lon, sample_id, buffer_m=None, px=None):
    """
    Download an ESRI World Imagery tile (global, ~0.3–0.5 m in cities).

    Uses Maxar/Airbus imagery under the hood. Free, no API key needed.
    Works well for Europe, Turkey, Middle East, and most major cities worldwide.
    """
    from config import SAT_BUFFER_M, SAT_IMAGE_PX
    px = px or SAT_IMAGE_PX

    buffer_m = buffer_m or SAT_BUFFER_M.get("ESRI", 200)  # config-driven zoom
    dlat = buffer_m / 111_320
    dlon = buffer_m / (111_320 * math.cos(math.radians(lat)))

    xmin, ymin = lon - dlon, lat - dlat
    xmax, ymax = lon + dlon, lat + dlat

    params = urllib.parse.urlencode({
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": "4326",
        "size": f"{px},{px}",
        "imageSR": "4326",
        "format": "png32",
        "pixelType": "U8",
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    })
    url = f"{_ESRI_EXPORT}?{params}"

    save_path = _save_path(sample_id)
    _download(url, save_path)

    return save_path, "latest", "ESRI"


# ---------------------------------------------------------------------------
# Source 4: Sentinel-2 via GEE — Global, 10 m/px, last resort
# ---------------------------------------------------------------------------

def fetch_s2_tile(lat, lon, sample_id, buffer_m=None, px=None):
    """
    Download a Sentinel-2 median composite (global, 10 m resolution).
    Last resort — use only when all higher-res sources fail.
    """
    from config import SAT_BUFFER_M, SAT_IMAGE_PX, S2_DATE_RANGE, S2_CLOUD_THRESHOLD
    px = px or SAT_IMAGE_PX

    # S2 is 10 m/px native — default buffer keeps ~1:1 pixel mapping
    buffer_m = buffer_m or SAT_BUFFER_M.get("S2", px // 2 * 10)  # config-driven zoom

    ee = get_ee()
    point = ee.Geometry.Point([lon, lat])

    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(point)
           .filterDate(*S2_DATE_RANGE)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", S2_CLOUD_THRESHOLD)))

    s2 = col.median()   # median composite: removes clouds/shadows statistically
    region = point.buffer(buffer_m).bounds()
    s2_rgb = s2.select(['B4', 'B3', 'B2']).multiply(0.0001)

    url = s2_rgb.getThumbURL({
        'region': region.getInfo()['coordinates'],
        'dimensions': f'{px}x{px}',
        'bands': ['B4', 'B3', 'B2'],
        'min': 0.02,
        'max': 0.20,
        'gamma': 1.4,
        'format': 'png',
    })

    save_path = _save_path(sample_id)
    urllib.request.urlretrieve(url, save_path)

    try:
        date_info = col.sort("CLOUDY_PIXEL_PERCENTAGE").first().date().format("YYYY-MM").getInfo()
    except Exception:
        date_info = "unknown"

    return save_path, date_info, "Sentinel-2"


# ---------------------------------------------------------------------------
# Fetch dispatcher with fallback chain
# ---------------------------------------------------------------------------

# Ordered fallback chain per source tag
_FALLBACK_CHAIN = {
    "NAIP":  ["NAIP",  "ESRI", "S2"],
    "IGN":   ["IGN",   "ESRI", "S2"],
    "ESRI":  ["ESRI",  "S2"],
    "S2":    ["S2"],
}

_FETCHERS = {
    "NAIP": fetch_naip_tile,
    "IGN":  fetch_ign_tile,
    "ESRI": fetch_esri_tile,
    "S2":   fetch_s2_tile,
}


def fetch_tile(sample, max_retries=3):
    """
    Fetch a satellite tile using the best available source for the location.

    Source is determined by:
      1. sample["satellite_source"] if explicitly set
      2. Auto-detection from lat/lon bounding boxes otherwise

    Falls back through the chain (e.g. IGN → ESRI → S2) on failure.
    """
    lat, lon = sample["lat"], sample["lon"]
    sid = sample["sample_id"]

    # Determine source — respect explicit override, otherwise auto-detect
    source = sample.get("satellite_source") or detect_source(lat, lon)
    chain = _FALLBACK_CHAIN.get(source, ["ESRI", "S2"])

    for src in chain:
        fetcher = _FETCHERS[src]
        for attempt in range(max_retries):
            try:
                result = fetcher(lat, lon, sid)
                path = result[0]
                # Sanity check: reject suspiciously small files (likely error tiles)
                if path and os.path.exists(path) and os.path.getsize(path) > 5_000:
                    return result
                print(f"      [{src}] returned tiny file, trying next source…")
                break
            except Exception as e:
                wait = 2 ** (attempt + 1)
                print(f"      [{src}] attempt {attempt+1}/{max_retries}: {e} (wait {wait}s)")
                time.sleep(wait)

    print(f"      ALL sources failed for {sid}")
    return None, None, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_path(sample_id):
    out_dir = os.path.join(ROOT, "output", "images", "sat")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{sample_id}.png")


def _download(url, save_path, timeout=30):
    """Download url → save_path with a browser-like User-Agent header."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; satellite-fetcher/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    with open(save_path, "wb") as f:
        f.write(data)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(samples=None, required_sources=None, excluded_sources=None):
    print("[Step 2/6] Fetching satellite imagery…")

    if samples is None:
        csv_path = os.path.join(ROOT, "output", "sample_locations.csv")
        samples = pd.read_csv(csv_path).to_dict("records")

    filtered = []
    for i, sample in enumerate(samples):
        sid = sample["sample_id"]
        lat, lon = sample["lat"], sample["lon"]

        # Fill in auto-detected source for logging if not already set
        if not sample.get("satellite_source"):
            sample["satellite_source"] = detect_source(lat, lon)

        src = sample["satellite_source"]
        print(f"  [{i+1}/{len(samples)}] {sid} ({src}): ({lat:.4f}, {lon:.4f})")

        path, date, actual_src = fetch_tile(sample)
        sample["satellite_path"] = path
        sample["satellite_date"] = date
        sample["satellite_actual_source"] = actual_src

        if path and os.path.exists(path):
            fsize = os.path.getsize(path)
            print(f"    ✓ {actual_src} | {fsize // 1024} KB | date={date}")
        else:
            print(f"    ✗ FAILED — no image saved")

        # source filter (optional)
        if required_sources and actual_src not in required_sources:
            print(f"    Skipping {sid}: {actual_src} is not one of {required_sources}")
            continue
        if excluded_sources and actual_src in excluded_sources:
            print(f"    Skipping {sid}: excluded source {actual_src}")
            continue

        filtered.append(sample)
        time.sleep(0.3)

    return filtered


if __name__ == "__main__":
    run()