"""Example usage of the OSM client."""

import logging
from src.osm import OSMClient

# Configure logging to see debug messages
logging.basicConfig(level=logging.INFO)

# Initialize client
client = OSMClient()

# Define a small bounding box (Istanbul, Turkey - very small area)
# top_left = (lat, lon) for north-west corner
# bottom_right = (lat, lon) for south-east corner
top_left = (39.932877, 32.826850)
bottom_right = (39.919218, 32.855573)

print("Fetching OSM data for bounding box:")
print(f"  Top-left (NW): {top_left}")
print(f"  Bottom-right (SE): {bottom_right}")
print()

try:
    # Get description of the area
    result = client.describe(top_left, bottom_right)

    print("=" * 60)
    print("BOUNDING BOX")
    print("=" * 60)
    print(f"Left: {result['bbox']['left']}")
    print(f"Bottom: {result['bbox']['bottom']}")
    print(f"Right: {result['bbox']['right']}")
    print(f"Top: {result['bbox']['top']}")
    print(f"Area: {result['bbox']['area_degrees2']:.6f} square degrees")
    print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total elements: {result['summary']['total_elements']}")
    print(f"Total landmarks: {result['summary']['total_landmarks']}")
    print(f"Distinct types: {result['summary']['distinct_types']}")
    print(f"Notes: {result['summary']['notes']}")
    print()

    print("=" * 60)
    print("TOP LANDMARK TYPES")
    print("=" * 60)
    for item in result['summary']['top_types'][:5]:
        print(f"  {item['type']}: {item['count']}")
    print()

    print("=" * 60)
    print("LANDMARKS (first 10)")
    print("=" * 60)
    for i, landmark in enumerate(result['landmarks'][:10], 1):
        name = landmark['name'] or '(unnamed)'
        ltype = landmark['primary_type']
        osm_type = landmark['osm_type']
        coords = f"({landmark['lat']:.4f}, {landmark['lon']:.4f})" if landmark['lat'] else "(no coords)"
        print(f"{i}. {name}")
        print(f"   Type: {ltype} ({osm_type})")
        print(f"   Location: {coords}")
        print()

    if result['organizations']:
        print("=" * 60)
        print("ORGANIZATIONS (first 5)")
        print("=" * 60)
        for i, org in enumerate(result['organizations'][:5], 1):
            name = org['name'] or '(unnamed)'
            print(f"{i}. {name} (ID: {org['id']}, {org['osm_type']})")
            for tag, value in org['org_tags'].items():
                if value:
                    print(f"   {tag}: {value}")
            print()

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
