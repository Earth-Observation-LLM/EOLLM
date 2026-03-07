"""
City-agnostic configuration for the Urban VQA pipeline.
Defines cities, seed locations, and all shared constants.
"""

CITIES = {
    "nyc": {
        "name": "New York City",
        "country": "USA",
        "bbox": (40.490, -74.260, 40.920, -73.680),  # S, W, N, E
        "satellite_source": "NAIP",  # 0.6m resolution
        "satellite_fallback": "S2",
        "seeds": [
            ("Times Square",        40.7580, -73.9855, "commercial"),
            ("Wall Street",         40.7074, -74.0113, "commercial"),
            ("Williamsburg",        40.7150, -73.9600, "residential"),
            ("DUMBO Brooklyn",      40.7033, -73.9903, "mixed"),
            ("Long Island City",    40.7425, -73.9580, "industrial"),
            ("Upper East Side",     40.7736, -73.9566, "residential"),
            ("Harlem",              40.8116, -73.9465, "residential"),
            ("Flushing Queens",     40.7614, -73.8300, "commercial"),
            ("South Bronx",         40.8090, -73.9230, "residential"),
            ("Staten Island",       40.6433, -74.0765, "residential"),
        ],
    },
    "paris": {
        "name": "Paris",
        "country": "France",
        "bbox": (48.815, 2.225, 48.902, 2.420),
        "satellite_source": "S2",
        "seeds": [
            ("Champs-Elysees",      48.8738,  2.2950, "commercial"),
            ("Le Marais",           48.8566,  2.3622, "residential"),
            ("La Defense",          48.8920,  2.2360, "commercial"),
            ("Belleville",          48.8714,  2.3847, "mixed"),
            ("Saint-Germain",       48.8540,  2.3338, "residential"),
            ("Montmartre",          48.8867,  2.3431, "residential"),
            ("Bercy",               48.8367,  2.3750, "mixed"),
            ("Batignolles",         48.8860,  2.3190, "residential"),
            ("Republique",          48.8676,  2.3640, "mixed"),
            ("Bastille",            48.8533,  2.3694, "mixed"),
        ],
    },
    "london": {
        "name": "London",
        "country": "UK",
        "bbox": (51.285, -0.510, 51.690, 0.335),
        "satellite_source": "S2",
        "seeds": [
            ("City of London",      51.5133, -0.0886, "commercial"),
            ("Canary Wharf",        51.5054, -0.0235, "commercial"),
            ("Camden Town",         51.5392, -0.1426, "mixed"),
            ("Shoreditch",          51.5274, -0.0777, "mixed"),
            ("Brixton",             51.4613, -0.1146, "residential"),
            ("Kensington",          51.4990, -0.1940, "residential"),
            ("Stratford",           51.5430, -0.0098, "mixed"),
            ("Croydon",             51.3762, -0.0986, "commercial"),
            ("Greenwich",           51.4769, -0.0005, "residential"),
            ("Westminster",         51.4975, -0.1357, "commercial"),
        ],
    },
    "singapore": {
        "name": "Singapore",
        "country": "Singapore",
        "bbox": (1.205, 103.605, 1.475, 104.030),
        "satellite_source": "S2",
        "seeds": [
            ("Marina Bay",          1.2815, 103.8585, "commercial"),
            ("Chinatown",           1.2830, 103.8440, "mixed"),
            ("Toa Payoh",           1.3343, 103.8490, "residential"),
            ("Orchard Road",        1.3040, 103.8318, "commercial"),
            ("Jurong East",         1.3330, 103.7422, "commercial"),
            ("Bukit Timah",         1.3400, 103.7765, "residential"),
            ("Geylang",             1.3170, 103.8900, "mixed"),
            ("Punggol",             1.4050, 103.9060, "residential"),
            ("Tampines",            1.3530, 103.9450, "residential"),
            ("Sentosa",             1.2494, 103.8303, "commercial"),
        ],
    },
    "sao_paulo": {
        "name": "São Paulo",
        "country": "Brazil",
        "bbox": (-23.750, -46.850, -23.350, -46.350),
        "satellite_source": "S2",
        "seeds": [
            ("Avenida Paulista",   -23.5613, -46.6556, "commercial"),
            ("Faria Lima",         -23.5870, -46.6780, "commercial"),
            ("Liberdade",          -23.5580, -46.6330, "mixed"),
            ("Vila Madalena",      -23.5530, -46.6910, "residential"),
            ("Mooca",              -23.5620, -46.5990, "industrial"),
            ("Pinheiros",          -23.5670, -46.6930, "mixed"),
            ("Centro Historico",   -23.5505, -46.6340, "commercial"),
            ("Jardins",            -23.5680, -46.6680, "residential"),
            ("Bela Vista",         -23.5590, -46.6470, "residential"),
            ("Vila Mariana",       -23.5890, -46.6350, "residential"),
        ],
    },
    "amsterdam": {
        "name": "Amsterdam",
        "country": "Netherlands",
        "bbox": (52.290, 4.730, 52.430, 5.020),
        "satellite_source": "S2",
        "seeds": [
            ("Dam Square",          52.3730,  4.8932, "commercial"),
            ("Jordaan",             52.3748,  4.8828, "residential"),
            ("De Pijp",             52.3535,  4.8940, "mixed"),
            ("Zuidas",              52.3380,  4.8730, "commercial"),
            ("Noord",               52.3910,  4.9230, "mixed"),
            ("Oost",                52.3610,  4.9280, "residential"),
            ("Westpoort",           52.3930,  4.8200, "industrial"),
            ("Amstelveen Centrum",  52.3030,  4.8630, "residential"),
            ("Bijlmer",             52.3160,  4.9530, "residential"),
            ("Watergraafsmeer",     52.3530,  4.9370, "residential"),
        ],
    },
}

# How many samples per city
SAMPLES_PER_CITY = 10

# OSM Overpass query for full city data (buildings + roads + amenities + landuse)
OSM_QUERY_TEMPLATE = """
[out:json][timeout:180];
(
  way["highway"~"^(primary|secondary|tertiary|residential|trunk|motorway)$"]
    ({s},{w},{n},{e});
  way["building"]
    ({s},{w},{n},{e});
  way["landuse"]
    ({s},{w},{n},{e});
  node["amenity"]
    ({s},{w},{n},{e});
  way["leisure"="park"]
    ({s},{w},{n},{e});
);
out tags center;
"""

# Lightweight query: roads only (much smaller, for road snapping)
OSM_ROADS_QUERY_TEMPLATE = """
[out:json][timeout:120];
way["highway"~"^(primary|secondary|tertiary|residential|trunk)$"]
  ({s},{w},{n},{e});
out body geom;
"""

# Land use categories (universal, derived from OSM tags)
LAND_USE_CATEGORIES = {
    "residential":  "Residential",
    "commercial":   "Commercial & Office",
    "retail":       "Retail & Shopping",
    "industrial":   "Industrial & Manufacturing",
    "mixed":        "Mixed Use",
    "institutional": "Public Facilities & Institutions",
    "open_space":   "Open Space & Recreation",
    "transport":    "Transportation & Infrastructure",
}

# Building height categories (universal)
BUILDING_HEIGHT_CATEGORIES = [
    ("Low-rise (1-3 floors)",    1,  3),
    ("Mid-rise (4-7 floors)",    4,  7),
    ("High-rise (8-20 floors)",  8,  20),
    ("Skyscraper (20+ floors)", 21, 999),
]

# Road type labels (from OSM highway tag)
ROAD_LABELS = {
    "motorway":    "Motorway / Highway",
    "trunk":       "Major trunk road",
    "primary":     "Primary arterial road",
    "secondary":   "Secondary road",
    "tertiary":    "Tertiary / collector road",
    "residential": "Local residential street",
}

# Urban density categories (based on building count in 200m radius)
URBAN_DENSITY_CATEGORIES = [
    ("Low density (suburban/rural)",      0,  15),
    ("Moderate density (urban fringe)",  16,  50),
    ("High density (dense urban)",       51, 150),
    ("Very high density (urban core)",  151, 9999),
]

# GEE config
GEE_PROJECT = "bitirme-489511"
NAIP_DATE_RANGE = ("2019-01-01", "2025-12-31")
S2_DATE_RANGE = ("2023-01-01", "2025-12-31")
S2_CLOUD_THRESHOLD = 20  # max cloud %

# Street View config
SV_SIZE = "640x640"
SV_FOV = 90
SV_PITCH = -5
