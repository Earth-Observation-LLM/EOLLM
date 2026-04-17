"""
Shared geometry utilities for the Urban VQA pipeline.
"""

import math


def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in meters between two lat/lon points."""
    R = 6_371_000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def sat_bbox(lat, lon, buffer_m):
    """Compute satellite tile geographic bbox: (south, west, north, east)."""
    delta_lat = buffer_m / 111320
    delta_lon = buffer_m / (111320 * math.cos(math.radians(lat)))
    return (lat - delta_lat, lon - delta_lon, lat + delta_lat, lon + delta_lon)


def quadrant_of_point(target_lat, target_lon, center_lat, center_lon):
    """Which quadrant of a north-up satellite image does the target fall in?

    Returns 'Top-Left', 'Top-Right', 'Bottom-Left', or 'Bottom-Right'.
    Top = North (higher lat), Left = West (lower lon).
    """
    ns = "Top" if target_lat >= center_lat else "Bottom"
    ew = "Left" if target_lon <= center_lon else "Right"
    return f"{ns}-{ew}"


def bearing_to_quadrant(bearing_deg):
    """Map camera bearing to satellite quadrant direction.

    Bearing 0=North, 90=East. Satellite: Top=North, Right=East.
    Returns the quadrant the camera is looking toward.
    """
    b = bearing_deg % 360
    if 0 <= b < 90:
        return "Top-Right"
    elif 90 <= b < 180:
        return "Bottom-Right"
    elif 180 <= b < 270:
        return "Bottom-Left"
    else:
        return "Top-Left"
