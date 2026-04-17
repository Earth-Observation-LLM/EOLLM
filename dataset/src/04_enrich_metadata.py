"""
Step 4: Enrich metadata with reverse geocoding (Nominatim).
OSM context is already extracted in Step 1. Census is US-only, handled here.
"""

import os
import time
import json
import requests
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_api_keys():
    keys = {}
    env_path = os.path.join(ROOT, "..", "api_keys.env")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                keys[k] = v
    return keys


def reverse_geocode(lat, lon):
    """Reverse geocode using Nominatim (free, 1 req/s)."""
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": lat, "lon": lon, "format": "json",
            "zoom": 16, "addressdetails": 1,
        }
        headers = {"User-Agent": "EOLLM-VQA-Pipeline/1.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        data = resp.json()
        addr = data.get("address", {})
        return {
            "display_name": data.get("display_name", ""),
            "suburb": addr.get("suburb") or addr.get("neighbourhood") or addr.get("quarter", ""),
            "city_district": addr.get("city_district", ""),
            "postcode": addr.get("postcode", ""),
            "admin_area": (addr.get("city") or addr.get("town") or
                           addr.get("municipality", "")),
        }
    except Exception as e:
        print(f"    Nominatim error: {e}")
        return {}


def get_us_census(lat, lon, api_key):
    """Get Census data (US locations only)."""
    try:
        fcc_url = (f"https://geo.fcc.gov/api/census/block/find"
                   f"?latitude={lat}&longitude={lon}&format=json")
        fcc = requests.get(fcc_url, timeout=10).json()
        state_fips = fcc['State']['FIPS']
        county_fips = fcc['County']['FIPS'][2:]
        tract = fcc['Block']['FIPS'][5:11]

        census_url = (
            f"https://api.census.gov/data/2022/acs/acs5"
            f"?get=B01003_001E,B19013_001E,B25001_001E"
            f"&for=tract:{tract}"
            f"&in=state:{state_fips}%20county:{county_fips}"
            f"&key={api_key}"
        )
        resp = requests.get(census_url, timeout=10)
        data = resp.json()
        if len(data) < 2:
            return {"census_tract": f"{state_fips}{county_fips}{tract}"}

        row = data[1]

        def safe_int(v):
            if v and v != "null":
                val = int(v)
                return val if val >= 0 else None  # filter sentinel values
            return None

        return {
            "census_tract": f"{state_fips}{county_fips}{tract}",
            "census_population": safe_int(row[0]),
            "census_median_income": safe_int(row[1]),
            "census_housing_units": safe_int(row[2]),
        }
    except Exception as e:
        print(f"    Census error: {e}")
        return {}


def run(samples=None):
    """Main entry point."""
    print("[Step 4/6] Enriching metadata...")

    if samples is None:
        csv_path = os.path.join(ROOT, "output", "sample_locations.csv")
        samples = pd.read_csv(csv_path).to_dict('records')

    keys = load_api_keys()
    census_key = keys.get("CENSUS_API_KEY")

    for i, sample in enumerate(samples):
        sid = sample["sample_id"]
        lat, lon = sample["lat"], sample["lon"]
        city = sample.get("city", "")
        country = sample.get("country", "")
        print(f"  [{i+1}/{len(samples)}] {sid} ({sample.get('city_name', '')})...")

        # Reverse geocode
        geo = reverse_geocode(lat, lon)
        for k, v in geo.items():
            sample[f"geo_{k}"] = v
        print(f"    Geo: {geo.get('suburb', '?')}, {geo.get('admin_area', '?')}")
        time.sleep(1.1)  # Nominatim requires 1 req/s

        # US Census (only for US cities)
        if country == "USA" and census_key:
            census = get_us_census(lat, lon, census_key)
            for k, v in census.items():
                sample[k] = v
            print(f"    Census: pop={census.get('census_population')}, "
                  f"income={census.get('census_median_income')}")
            time.sleep(0.3)

    # Save metadata CSV
    out_path = os.path.join(ROOT, "output", "metadata_raw.csv")
    df = pd.DataFrame(samples)
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
            df[col] = df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
            )
    df.to_csv(out_path, index=False)
    print(f"  Saved enriched metadata to {out_path}")
    return samples


if __name__ == "__main__":
    run()
