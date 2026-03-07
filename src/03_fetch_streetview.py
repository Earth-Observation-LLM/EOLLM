"""
Step 3: Download Google Street View images with road-aligned headings.
City-agnostic — works globally.
"""

import os
import time
import requests
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_api_key():
    """Load Google Street View API key from api_keys.env."""
    env_path = os.path.join(ROOT, "api_keys.env")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("GOOGLE_STREETVIEW_KEY="):
                return line.split("=", 1)[1]
    raise ValueError("GOOGLE_STREETVIEW_KEY not found in api_keys.env")


def check_sv_coverage(lat, lon, api_key):
    """Check if outdoor Street View coverage exists at this location."""
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {"location": f"{lat},{lon}", "key": api_key, "source": "outdoor"}
    resp = requests.get(url, params=params, timeout=10)
    data = resp.json()
    if data.get("status") == "OK":
        return {"pano_id": data.get("pano_id"), "date": data.get("date", "unknown"),
                "status": "OK"}
    return {"status": data.get("status", "UNKNOWN")}


def fetch_streetview(lat, lon, sample_id, road_bearing, api_key):
    """Download 4 street-view images with road-aligned headings."""
    from config import SV_SIZE, SV_FOV, SV_PITCH
    headings = {
        "along_fwd":   road_bearing % 360,
        "along_bwd":   (road_bearing + 180) % 360,
        "cross_left":  (road_bearing + 90) % 360,
        "cross_right": (road_bearing - 90) % 360,
    }
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    results = {}

    for label, heading in headings.items():
        params = {
            "size": SV_SIZE, "location": f"{lat},{lon}",
            "heading": heading, "pitch": SV_PITCH, "fov": SV_FOV,
            "key": api_key, "source": "outdoor",
            "return_error_code": "true",
        }
        try:
            resp = requests.get(base_url, params=params, timeout=15)
            ct = resp.headers.get('content-type', '')
            if resp.status_code == 200 and ct.startswith('image'):
                path = os.path.join(
                    ROOT, "output", "images", "sv", f"{sample_id}_{label}.jpg"
                )
                with open(path, 'wb') as f:
                    f.write(resp.content)
                fsize = os.path.getsize(path)
                if fsize < 5000:
                    results[label] = None
                else:
                    results[label] = path
            else:
                results[label] = None
        except Exception as e:
            print(f"      ERROR {label}: {e}")
            results[label] = None
        time.sleep(0.2)

    return results


def run(samples=None):
    """Main entry point."""
    print("[Step 3/6] Fetching Street View imagery...")

    if samples is None:
        csv_path = os.path.join(ROOT, "output", "sample_locations.csv")
        samples = pd.read_csv(csv_path).to_dict('records')

    api_key = load_api_key()

    for i, sample in enumerate(samples):
        sid = sample["sample_id"]
        lat, lon = sample["lat"], sample["lon"]
        bearing = sample["road_bearing"]
        print(f"  [{i+1}/{len(samples)}] {sid}: ({lat:.4f}, {lon:.4f})")

        meta = check_sv_coverage(lat, lon, api_key)
        if meta["status"] != "OK":
            print(f"    SKIPPED: no SV coverage ({meta['status']})")
            sample["sv_status"] = meta["status"]
            sample["sv_date"] = None
            for label in ["along_fwd", "along_bwd", "cross_left", "cross_right"]:
                sample[f"sv_{label}_path"] = None
            continue

        sample["sv_status"] = "OK"
        sample["sv_date"] = meta.get("date")
        print(f"    SV OK, date={meta.get('date')}")

        images = fetch_streetview(lat, lon, sid, bearing, api_key)
        sv_count = 0
        for label, path in images.items():
            sample[f"sv_{label}_path"] = path
            if path:
                sv_count += 1
        print(f"    -> {sv_count}/4 images saved")

    return samples


if __name__ == "__main__":
    run()
