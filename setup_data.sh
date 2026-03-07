#!/usr/bin/env bash
# Downloads all OSM data required by the pipeline via Overpass API.
# Run this once before running the pipeline.

set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")" && pwd)/data"
mkdir -p "$DATA_DIR"

OVERPASS="https://overpass-api.de/api/interpreter"
TIMEOUT=300
RETRY_WAIT=60

overpass_download() {
    local label="$1"
    local out_file="$2"
    local query="$3"

    if [ -f "$out_file" ]; then
        echo "[skip] $label already exists at $out_file"
        return
    fi

    echo "[download] $label -> $out_file"
    for attempt in 1 2 3; do
        http_code=$(curl -s -o "$out_file" -w "%{http_code}" \
            --max-time "$TIMEOUT" \
            --data-urlencode "data=$query" \
            "$OVERPASS")
        if [ "$http_code" = "200" ]; then
            echo "  OK (attempt $attempt)"
            return
        elif [ "$http_code" = "429" ]; then
            echo "  Rate limited (attempt $attempt), waiting ${RETRY_WAIT}s..."
            rm -f "$out_file"
            sleep "$RETRY_WAIT"
        else
            echo "  HTTP $http_code (attempt $attempt), retrying..."
            rm -f "$out_file"
            sleep 10
        fi
    done
    echo "  ERROR: failed to download $label after 3 attempts"
    exit 1
}

# --- City bboxes: S W N E ---
# NYC
NYC="40.490,-74.260,40.920,-73.680"
# Paris
PAR="48.815,2.225,48.902,2.420"
# London
LON="51.285,-0.510,51.690,0.335"
# Singapore
SIN="1.205,103.605,1.475,104.030"
# Sao Paulo
SAO="-23.750,-46.850,-23.350,-46.350"
# Amsterdam
AMS="52.290,4.730,52.430,5.020"

roads_query() {
    local bbox="$1"
    echo "[out:json][timeout:120];way[\"highway\"~\"^(primary|secondary|tertiary|residential|trunk)$\"](${bbox});out body geom;"
}

context_query() {
    local bbox="$1"
    local s w n e
    IFS=',' read -r s w n e <<< "$bbox"
    cat <<EOF
[out:json][timeout:180];
(
  way["highway"~"^(primary|secondary|tertiary|residential|trunk|motorway)$"](${s},${w},${n},${e});
  way["building"](${s},${w},${n},${e});
  way["landuse"](${s},${w},${n},${e});
  node["amenity"](${s},${w},${n},${e});
  way["leisure"="park"](${s},${w},${n},${e});
);
out tags center;
EOF
}

echo "=== Downloading OSM road data ==="
overpass_download "NYC roads"       "$DATA_DIR/nyc_roads_osm.json"       "$(roads_query "$NYC")"
sleep 5
overpass_download "Paris roads"     "$DATA_DIR/paris_roads_osm.json"     "$(roads_query "$PAR")"
sleep 5
overpass_download "London roads"    "$DATA_DIR/london_roads_osm.json"    "$(roads_query "$LON")"
sleep 5
overpass_download "Singapore roads" "$DATA_DIR/singapore_roads_osm.json" "$(roads_query "$SIN")"
sleep 5
overpass_download "Sao Paulo roads" "$DATA_DIR/sao_paulo_roads_osm.json" "$(roads_query "$SAO")"
sleep 5
overpass_download "Amsterdam roads" "$DATA_DIR/amsterdam_roads_osm.json" "$(roads_query "$AMS")"

echo ""
echo "=== Downloading OSM context data (buildings/amenities/landuse) ==="
echo "Note: These are large queries and may take several minutes each."
sleep 10
overpass_download "NYC context"       "$DATA_DIR/nyc_context_osm.json"       "$(context_query "$NYC")"
sleep 10
overpass_download "Paris context"     "$DATA_DIR/paris_context_osm.json"     "$(context_query "$PAR")"
sleep 10
overpass_download "London context"    "$DATA_DIR/london_context_osm.json"    "$(context_query "$LON")"
sleep 10
overpass_download "Singapore context" "$DATA_DIR/singapore_context_osm.json" "$(context_query "$SIN")"
sleep 10
overpass_download "Sao Paulo context" "$DATA_DIR/sao_paulo_context_osm.json" "$(context_query "$SAO")"
sleep 10
overpass_download "Amsterdam context" "$DATA_DIR/amsterdam_context_osm.json" "$(context_query "$AMS")"

echo ""
echo "=== Done! All OSM data downloaded to $DATA_DIR ==="
