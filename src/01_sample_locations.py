"""
Step 1: Generate sample locations for all configured cities.
Snaps seed points to local OSM road data and extracts OSM metadata.
Downloads OSM road data per city if not already cached.
"""

import os
import math
import json
import time
import requests
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def download_osm_roads(city_key, city_cfg):
    """Download OSM road network for a city (single Overpass query, cached)."""
    cache_path = os.path.join(ROOT, "data", f"{city_key}_roads_osm.json")
    if os.path.exists(cache_path):
        print(f"    Using cached road data: {cache_path}")
        return cache_path

    from config import OSM_ROADS_QUERY_TEMPLATE
    s, w, n, e = city_cfg["bbox"]
    query = OSM_ROADS_QUERY_TEMPLATE.format(s=s, w=w, n=n, e=e)

    print(f"    Downloading OSM roads for {city_cfg['name']}...")
    for attempt in range(3):
        resp = requests.get(
            "https://overpass-api.de/api/interpreter",
            params={"data": query},
            timeout=200,
        )
        if resp.status_code == 429:
            wait = 30 * (attempt + 1)
            print(f"    Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        break
    else:
        raise RuntimeError(f"Failed to download OSM roads for {city_key}")

    with open(cache_path, "w") as f:
        f.write(resp.text)
    data = resp.json()
    print(f"    Saved {len(data.get('elements', []))} road segments to {cache_path}")
    return cache_path


def download_osm_context(city_key, city_cfg):
    """Download OSM buildings/amenities/landuse for a city (single query, cached)."""
    cache_path = os.path.join(ROOT, "data", f"{city_key}_context_osm.json")
    if os.path.exists(cache_path):
        print(f"    Using cached context data: {cache_path}")
        return cache_path

    from config import OSM_QUERY_TEMPLATE
    s, w, n, e = city_cfg["bbox"]
    query = OSM_QUERY_TEMPLATE.format(s=s, w=w, n=n, e=e)

    print(f"    Downloading OSM context for {city_cfg['name']} (buildings/amenities/landuse)...")
    for attempt in range(3):
        resp = requests.get(
            "https://overpass-api.de/api/interpreter",
            params={"data": query},
            timeout=300,
        )
        if resp.status_code == 429:
            wait = 60 * (attempt + 1)
            print(f"    Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        break
    else:
        raise RuntimeError(f"Failed to download OSM context for {city_key}")

    with open(cache_path, "w") as f:
        f.write(resp.text)
    data = resp.json()
    print(f"    Saved {len(data.get('elements', []))} elements to {cache_path}")
    return cache_path


def load_road_nodes(roads_json_path):
    """Parse road JSON into a flat list of (lat, lon, bearing, road_name, highway_type)."""
    with open(roads_json_path) as f:
        data = json.load(f)
    nodes = []
    for way in data.get("elements", []):
        geom = way.get("geometry", [])
        tags = way.get("tags", {})
        road_name = tags.get("name", "unnamed")
        highway_type = tags.get("highway", "unknown")
        for i, node in enumerate(geom):
            nlat, nlon = node["lat"], node["lon"]
            if i + 1 < len(geom):
                nxt = geom[i + 1]
            elif i > 0:
                nxt = geom[i - 1]
            else:
                continue
            bearing = compute_bearing(nlat, nlon, nxt["lat"], nxt["lon"])
            nodes.append((nlat, nlon, bearing, road_name, highway_type))
    return nodes


def load_context_index(context_json_path):
    """Parse context JSON into categorized lists for spatial queries."""
    with open(context_json_path) as f:
        data = json.load(f)

    buildings = []  # (lat, lon, levels, building_type)
    amenities = []  # (lat, lon, amenity_type)
    landuse = []    # (lat, lon, landuse_type)
    parks = []      # (lat, lon)

    for elem in data.get("elements", []):
        tags = elem.get("tags", {})
        # Use center coords if available, otherwise skip
        center = elem.get("center", {})
        lat = center.get("lat")
        lon = center.get("lon")
        if lat is None or lon is None:
            continue

        if "building" in tags:
            levels = tags.get("building:levels")
            if levels:
                try:
                    levels = int(float(levels))
                except (ValueError, TypeError):
                    levels = None
            else:
                levels = None
            buildings.append((lat, lon, levels, tags.get("building", "yes")))

        if "amenity" in tags:
            amenities.append((lat, lon, tags["amenity"]))

        if "landuse" in tags:
            landuse.append((lat, lon, tags["landuse"]))

        if tags.get("leisure") == "park":
            parks.append((lat, lon))

    return {"buildings": buildings, "amenities": amenities, "landuse": landuse, "parks": parks}


def snap_to_road(lat, lon, road_nodes, radius_deg=0.002):
    """Find nearest road node from local index."""
    best_dist = float('inf')
    best = None
    for nlat, nlon, bearing, road_name, highway_type in road_nodes:
        if abs(nlat - lat) > radius_deg or abs(nlon - lon) > radius_deg:
            continue
        dist = (nlat - lat)**2 + (nlon - lon)**2
        if dist < best_dist:
            best_dist = dist
            best = (nlat, nlon, bearing, road_name, highway_type)
    if best is None:
        return None
    return {
        "lat": best[0], "lon": best[1],
        "road_bearing": round(best[2], 1),
        "road_name": best[3], "highway_type": best[4],
    }


def get_local_context(lat, lon, context_idx, radius_deg=0.002):
    """Extract OSM context around a point from pre-loaded index."""
    from collections import Counter

    # Buildings within radius
    nearby_buildings = []
    for blat, blon, levels, btype in context_idx["buildings"]:
        if abs(blat - lat) < radius_deg and abs(blon - lon) < radius_deg:
            nearby_buildings.append((levels, btype))

    building_count = len(nearby_buildings)
    levels_with_data = [lv for lv, _ in nearby_buildings if lv is not None]
    median_levels = None
    if levels_with_data:
        levels_with_data.sort()
        mid = len(levels_with_data) // 2
        median_levels = levels_with_data[mid]

    # Building types
    btype_counter = Counter(bt for _, bt in nearby_buildings)

    # Amenities
    nearby_amenities = []
    for alat, alon, atype in context_idx["amenities"]:
        if abs(alat - lat) < radius_deg and abs(alon - lon) < radius_deg:
            nearby_amenities.append(atype)
    amenity_count = len(nearby_amenities)
    amenity_types = list(set(nearby_amenities))[:10]

    # Land use
    nearby_landuse = []
    for llat, llon, ltype in context_idx["landuse"]:
        if abs(llat - lat) < radius_deg and abs(llon - lon) < radius_deg:
            nearby_landuse.append(ltype)
    landuse_counter = Counter(nearby_landuse)
    dominant_landuse = landuse_counter.most_common(1)[0][0] if landuse_counter else None

    # Parks
    has_park = any(
        abs(plat - lat) < radius_deg and abs(plon - lon) < radius_deg
        for plat, plon in context_idx["parks"]
    )

    # Infer land use category from OSM data
    land_use_category = infer_land_use(
        dominant_landuse, btype_counter, amenity_types, has_park, building_count,
        amenity_count
    )

    return {
        "osm_building_count": building_count,
        "osm_median_levels": median_levels,
        "osm_amenity_count": amenity_count,
        "osm_amenity_types": amenity_types,
        "osm_dominant_landuse_raw": dominant_landuse,
        "osm_has_park": has_park,
        "osm_building_types": dict(btype_counter.most_common(5)),
        "land_use_category": land_use_category,
    }


def infer_land_use(dominant_landuse, btype_counter, amenity_types, has_park,
                   building_count=0, amenity_count=0):
    """Infer a standardized land use category from OSM data.

    The logic uses a priority cascade:
      1. Explicit landuse tag from OSM (strongest signal)
      2. Building type distribution (structural signal)
      3. Amenity density ratio (functional signal)
      4. Fallback to "residential" (most common real-world default)

    "mixed" is only assigned when there is genuine evidence of multiple
    competing uses, not as a catch-all default.
    """

    # --- Open space: only when genuinely dominated by green/recreation ---
    if dominant_landuse == "recreation_ground":
        return "open_space"
    if dominant_landuse in ("grass", "forest", "meadow") and building_count < 15:
        return "open_space"
    if has_park and building_count < 10:
        return "open_space"

    # --- Explicit OSM landuse tag (strongest signal) ---
    if dominant_landuse in ("industrial", "railway"):
        return "industrial"
    if dominant_landuse in ("retail", "commercial"):
        return "commercial"
    if dominant_landuse in ("residential",):
        return "residential"

    # --- Infer from building type distribution ---
    top_btype = btype_counter.most_common(1)[0][0] if btype_counter else None
    if top_btype in ("commercial", "office", "retail", "supermarket"):
        return "commercial"
    if top_btype in ("industrial", "warehouse", "manufacture"):
        return "industrial"
    if top_btype in ("apartments", "house", "detached", "terrace", "residential",
                     "semidetached_house"):
        return "residential"
    if top_btype in ("church", "school", "hospital", "university", "public",
                     "civic", "government"):
        return "institutional"

    # --- Mixed use: require genuine evidence of BOTH residential buildings
    #     AND significant commercial amenity density ---
    commercial_amenities = {"restaurant", "cafe", "bar", "shop", "bank",
                            "fast_food", "pharmacy", "marketplace"}
    has_commercial = len(set(amenity_types) & commercial_amenities) >= 2
    residential_btypes = {"apartments", "house", "detached", "terrace",
                          "residential", "semidetached_house"}
    has_residential_buildings = bool(set(btype_counter.keys()) & residential_btypes)

    if has_commercial and has_residential_buildings:
        return "mixed"
    # High amenity density relative to buildings also suggests mixed/commercial
    if amenity_count >= 10 and building_count > 0:
        ratio = amenity_count / building_count
        if ratio > 0.3 and has_residential_buildings:
            return "mixed"
        if ratio > 0.3:
            return "commercial"

    # --- Default: residential (the most common real-world land use) ---
    # This is far more accurate than "mixed" as a fallback, since most urban
    # areas without strong commercial/industrial signals are residential.
    if building_count > 0:
        return "residential"

    return "mixed"  # truly ambiguous (no buildings, no landuse tag)


def compute_bearing(lat1, lon1, lat2, lon2):
    """Compute initial bearing from point 1 to point 2 in degrees [0, 360)."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = (math.cos(lat1) * math.sin(lat2) -
         math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    bearing = math.degrees(math.atan2(x, y))
    return bearing % 360


def run_city(city_key, city_cfg, num_samples=10):
    """Generate samples for one city. Returns list of sample dicts."""
    print(f"\n  [{city_cfg['name']}] Generating {num_samples} locations...")

    # Download/cache OSM data
    roads_path = download_osm_roads(city_key, city_cfg)
    context_path = download_osm_context(city_key, city_cfg)

    # Load indexes
    print(f"    Loading road index...")
    road_nodes = load_road_nodes(roads_path)
    print(f"    {len(road_nodes)} road nodes loaded")

    print(f"    Loading context index...")
    context_idx = load_context_index(context_path)
    print(f"    {len(context_idx['buildings'])} buildings, "
          f"{len(context_idx['amenities'])} amenities, "
          f"{len(context_idx['parks'])} parks")

    seeds = city_cfg["seeds"][:num_samples]
    samples = []
    skipped = []

    for i, (name, lat, lon, expected_char) in enumerate(seeds):
        sample_id = f"{city_key}_{i+1:04d}"
        print(f"    [{i+1}/{len(seeds)}] {name} ({lat}, {lon})...")

        road_info = snap_to_road(lat, lon, road_nodes)
        if road_info is None:
            print(f"      SKIPPED: no road found within snap radius")
            skipped.append({"sample_id": sample_id, "name": name,
                            "reason": "no_road_within_radius"})
            continue

        slat, slon = road_info["lat"], road_info["lon"]

        # Get OSM context
        context = get_local_context(slat, slon, context_idx)

        sample = {
            "sample_id": sample_id,
            "city": city_key,
            "city_name": city_cfg["name"],
            "country": city_cfg["country"],
            "neighborhood": name,
            "seed_lat": lat,
            "seed_lon": lon,
            "lat": slat,
            "lon": slon,
            "road_bearing": road_info["road_bearing"],
            "road_name": road_info["road_name"],
            "highway_type": road_info["highway_type"],
            "satellite_source": city_cfg["satellite_source"],
        }
        # Flatten context
        for k, v in context.items():
            sample[k] = v

        samples.append(sample)
        print(f"      -> ({slat:.5f}, {slon:.5f}) road={road_info['road_name']}, "
              f"landuse={context['land_use_category']}, "
              f"bldgs={context['osm_building_count']}, "
              f"levels={context['osm_median_levels']}")

    if skipped:
        print(f"    WARNING: {len(skipped)}/{len(seeds)} seeds skipped: "
              f"{[s['name'] for s in skipped]}")
    return samples


def run(cities=None, num_samples=10):
    """Main entry point. Runs across all configured cities."""
    from config import CITIES
    print("[Step 1/6] Generating sample locations...")

    if cities is None:
        cities = CITIES

    all_samples = []
    for city_key, city_cfg in cities.items():
        city_samples = run_city(city_key, city_cfg, num_samples)
        all_samples.extend(city_samples)
        # Delay between cities to avoid Overpass rate limits on first run
        if not os.path.exists(os.path.join(ROOT, "data", f"{city_key}_context_osm.json")):
            time.sleep(5)

    # Save CSV
    out_path = os.path.join(ROOT, "output", "sample_locations.csv")
    df = pd.DataFrame(all_samples)
    # Serialize dict/list columns
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
            df[col] = df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
            )
    df.to_csv(out_path, index=False)
    print(f"\n  Saved {len(all_samples)} total locations to {out_path}")
    return all_samples


if __name__ == "__main__":
    run()
