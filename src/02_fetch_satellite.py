"""
Step 2: Download satellite tiles via Google Earth Engine.
Uses NAIP for US cities, Sentinel-2 for everywhere else.
"""

import os
import time
import urllib.request
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_ee = None


def get_ee():
    global _ee
    if _ee is None:
        import ee
        from config import GEE_PROJECT
        ee.Initialize(project=GEE_PROJECT)
        _ee = ee
    return _ee


def fetch_naip_tile(lat, lon, sample_id, buffer_m=250, px=512):
    """Download a NAIP tile (US only, 0.6m resolution)."""
    ee = get_ee()
    from config import NAIP_DATE_RANGE
    point = ee.Geometry.Point([lon, lat])
    naip = (ee.ImageCollection("USDA/NAIP/DOQQ")
            .filterBounds(point)
            .filterDate(*NAIP_DATE_RANGE)
            .sort("system:time_start", False)
            .mosaic())
    region = point.buffer(buffer_m).bounds()
    url = naip.getThumbURL({
        'region': region.getInfo()['coordinates'],
        'dimensions': f'{px}x{px}',
        'bands': ['R', 'G', 'B'],
        'min': 0, 'max': 255, 'format': 'png',
    })
    save_path = os.path.join(ROOT, "output", "images", "sat", f"{sample_id}.png")
    urllib.request.urlretrieve(url, save_path)

    try:
        date_info = (ee.ImageCollection("USDA/NAIP/DOQQ")
                     .filterBounds(point)
                     .filterDate(*NAIP_DATE_RANGE)
                     .sort("system:time_start", False)
                     .first().date().format("YYYY-MM").getInfo())
    except Exception:
        date_info = "unknown"
    return save_path, date_info, "NAIP"


def fetch_s2_tile(lat, lon, sample_id, buffer_m=500, px=512):
    """Download a Sentinel-2 tile (global, 10m resolution).

    Buffer is 500m so that at 10m/pixel the native footprint is ~100px,
    and GEE returns a 512x512 image with ~5x oversampling — enough detail
    to see roads, building blocks, and green areas without Minecraft pixels.
    """
    ee = get_ee()
    from config import S2_DATE_RANGE, S2_CLOUD_THRESHOLD
    point = ee.Geometry.Point([lon, lat])
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(point)
          .filterDate(*S2_DATE_RANGE)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", S2_CLOUD_THRESHOLD))
          .sort("CLOUDY_PIXEL_PERCENTAGE")
          .mosaic())
    region = point.buffer(buffer_m).bounds()
    s2_rgb = s2.select(['B4', 'B3', 'B2']).multiply(0.0001)  # to reflectance [0,1]
    # Use meters_per_pixel to let GEE serve at native resolution (10m)
    # instead of forcing 512px onto a tiny area.
    url = s2_rgb.getThumbURL({
        'region': region.getInfo()['coordinates'],
        'dimensions': f'{px}x{px}',
        'bands': ['B4', 'B3', 'B2'],
        'min': 0.02, 'max': 0.25, 'format': 'png',
    })
    save_path = os.path.join(ROOT, "output", "images", "sat", f"{sample_id}.png")
    urllib.request.urlretrieve(url, save_path)

    try:
        date_info = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                     .filterBounds(point)
                     .filterDate(*S2_DATE_RANGE)
                     .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", S2_CLOUD_THRESHOLD))
                     .sort("CLOUDY_PIXEL_PERCENTAGE")
                     .first().date().format("YYYY-MM").getInfo())
    except Exception:
        date_info = "unknown"
    return save_path, date_info, "Sentinel-2"


def fetch_tile(sample, max_retries=3):
    """Fetch satellite tile using the appropriate source for the city."""
    lat, lon = sample["lat"], sample["lon"]
    sid = sample["sample_id"]
    source = sample.get("satellite_source", "S2")

    for attempt in range(max_retries):
        try:
            if source == "NAIP":
                return fetch_naip_tile(lat, lon, sid)
            else:
                return fetch_s2_tile(lat, lon, sid)
        except Exception as e:
            wait = 2 ** (attempt + 1)
            print(f"      Retry {attempt+1}/{max_retries}: {e} (waiting {wait}s)")
            time.sleep(wait)

    # Fallback: try the other source
    print(f"      Falling back to {'S2' if source == 'NAIP' else 'NAIP'}...")
    try:
        if source == "NAIP":
            return fetch_s2_tile(lat, lon, sid)
        else:
            return fetch_naip_tile(lat, lon, sid)
    except Exception as e:
        print(f"      FAILED: {e}")
        return None, None, None


def run(samples=None):
    """Main entry point."""
    print("[Step 2/6] Fetching satellite imagery via GEE...")

    if samples is None:
        csv_path = os.path.join(ROOT, "output", "sample_locations.csv")
        samples = pd.read_csv(csv_path).to_dict('records')

    for i, sample in enumerate(samples):
        sid = sample["sample_id"]
        lat, lon = sample["lat"], sample["lon"]
        src = sample.get("satellite_source", "S2")
        print(f"  [{i+1}/{len(samples)}] {sid} ({src}): ({lat:.4f}, {lon:.4f})")

        path, date, actual_src = fetch_tile(sample)
        sample["satellite_path"] = path
        sample["satellite_date"] = date
        sample["satellite_actual_source"] = actual_src

        if path and os.path.exists(path):
            fsize = os.path.getsize(path)
            print(f"    -> {actual_src}, {fsize//1024}KB, date={date}")
            if fsize < 5000:
                print(f"    WARNING: suspiciously small ({fsize} bytes)")
        else:
            print(f"    -> FAILED")

        time.sleep(0.5)

    return samples


if __name__ == "__main__":
    run()
