"""
Step 3: Download Google Street View images with road-aligned headings.
City-agnostic — works globally.

Includes detection for tunnel/underground imagery that passes Google's
"outdoor" source filter but is useless for urban VQA tasks.
"""

import os
import time
import math
import requests
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Max distance (meters) between requested location and the panorama that
# Google returns. If the pano is farther than this, Google likely snapped
# to a different road (e.g. a nearby highway tunnel).
SV_MAX_SNAP_DISTANCE_M = 80


def load_api_key():
    """Load Google Street View API key from api_keys.env."""
    env_path = os.path.join(ROOT, "..", "api_keys.env")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("GOOGLE_STREETVIEW_KEY="):
                return line.split("=", 1)[1]
    raise ValueError("GOOGLE_STREETVIEW_KEY not found in api_keys.env")


def _haversine_m(lat1, lon1, lat2, lon2):
    """Distance in meters between two lat/lon points."""
    R = 6_371_000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _is_tunnel_image(image_bytes):
    """Heuristic: detect tunnel / underground Street View images.

    Tunnel images under artificial fluorescent lighting have an extreme
    color channel imbalance (strong yellow/orange cast) that outdoor images
    almost never exhibit. We detect this by comparing the spread between
    the mean R, G, B values.

    Normal outdoor images: channel spread < 25
    Tunnel images:         channel spread > 40 (yellowish cast)
    """
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        r, g, b = img.split()
        n = img.size[0] * img.size[1]
        r_mean = sum(r.getdata()) / n
        g_mean = sum(g.getdata()) / n
        b_mean = sum(b.getdata()) / n
        color_spread = max(r_mean, g_mean, b_mean) - min(r_mean, g_mean, b_mean)
        return color_spread > 40
    except ImportError:
        return False
    except Exception:
        return False


def check_sv_coverage(lat, lon, api_key):
    """Check if outdoor Street View coverage exists at this location.

    Also verifies the returned panorama is close to the requested location
    (rejects cases where Google snaps to a tunnel/highway far away).
    """
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {"location": f"{lat},{lon}", "key": api_key, "source": "outdoor"}
    resp = requests.get(url, params=params, timeout=10)
    data = resp.json()
    if data.get("status") == "OK":
        # Check if the panorama location is too far from the requested point
        pano_loc = data.get("location", {})
        pano_lat = pano_loc.get("lat")
        pano_lng = pano_loc.get("lng")
        snap_dist = None
        if pano_lat is not None and pano_lng is not None:
            snap_dist = _haversine_m(lat, lon, pano_lat, pano_lng)
            if snap_dist > SV_MAX_SNAP_DISTANCE_M:
                return {"status": "SNAP_TOO_FAR",
                        "snap_distance_m": round(snap_dist, 1)}
        return {"pano_id": data.get("pano_id"), "date": data.get("date", "unknown"),
                "status": "OK", "snap_distance_m": round(snap_dist, 1) if snap_dist else None}
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
                fsize = len(resp.content)
                if fsize < 5000:
                    results[label] = None
                elif _is_tunnel_image(resp.content):
                    print(f"      {label}: rejected (tunnel/dark image)")
                    results[label] = None
                else:
                    path = os.path.join(
                        ROOT, "output", "images", "sv",
                        f"{sample_id}_{label}.jpg"
                    )
                    with open(path, 'wb') as f:
                        f.write(resp.content)
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
