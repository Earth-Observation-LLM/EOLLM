"""
City-agnostic configuration for the Urban VQA pipeline.
Defines cities, seed locations, and all shared constants.
"""

CITIES = {

    
    # ─────────────────────────────────────────────
    # EAST ASIA (Korean apartment-block morphology)
    # ─────────────────────────────────────────────

    "seoul": {
        "name": "Seoul",
        "country": "South Korea",
        "bbox": (37.480, 126.880, 37.600, 127.110),
        "seeds": [
            # Permanent iconic anchors
            ("Myeongdong",              37.5636, 126.9869, "commercial"),
            ("Gangnam Station",         37.4979, 127.0276, "commercial"),
            # Residential
            ("Mapo-gu",                 37.5537, 126.9084, "residential"),
            ("Songpa-gu Jamsil",        37.5133, 127.1001, "residential"),
            ("Seongbuk-dong",           37.5930, 127.0060, "residential"),
            # Mixed
            ("Hongdae",                 37.5563, 126.9236, "mixed"),
            ("Itaewon",                 37.5340, 126.9948, "mixed"),
            # Industrial
            ("Guro Digital Complex",    37.4846, 126.8985, "industrial"),
            ("Seongsu-dong",            37.5445, 127.0560, "industrial"),
            # Natural
            ("Bukhansan Entrance",      37.5870, 126.9650, "natural"),
            ("Yeouido Hangang Park",    37.5284, 126.9326, "natural"),
        ],
    },

    # ─────────────────────────────────────────────
    # LATIN AMERICA (Colonial + informal + modern)
    # ─────────────────────────────────────────────

    "mexico_city": {
        "name": "Mexico City",
        "country": "Mexico",
        "bbox": (19.340, -99.240, 19.500, -99.070),
        "seeds": [
            # Permanent iconic anchors
            ("Zocalo",                  19.4326, -99.1332, "commercial"),
            ("Polanco",                 19.4328, -99.1916, "commercial"),
            # Residential
            ("Coyoacan",                19.3500, -99.1620, "residential"),
            ("Roma Norte",              19.4194, -99.1637, "residential"),
            ("Condesa",                 19.4114, -99.1733, "residential"),
            # Mixed
            ("Reforma Corridor",        19.4270, -99.1677, "mixed"),
            ("Juarez",                  19.4280, -99.1560, "mixed"),
            # Industrial
            ("Azcapotzalco Industrial", 19.4869, -99.1847, "industrial"),
            ("Vallejo",                 19.4891, -99.1467, "industrial"),
            # Natural
            ("Chapultepec Park",        19.4204, -99.1894, "natural"),
            ("Viveros de Coyoacan",     19.3553, -99.1817, "natural"),
        ],
    },

    # ─────────────────────────────────────────────
    # SOUTHERN EUROPE (Iconic Eixample grid)
    # ─────────────────────────────────────────────

    "barcelona": {
        "name": "Barcelona",
        "country": "Spain",
        "bbox": (41.350, 2.080, 41.450, 2.230),
        "seeds": [
            # Permanent iconic anchors
            ("Passeig de Gracia",       41.3954,  2.1637, "commercial"),
            ("Placa Catalunya",         41.3870,  2.1700, "commercial"),
            # Residential
            ("Gracia",                  41.4025,  2.1566, "residential"),
            ("Eixample",                41.3898,  2.1614, "residential"),
            ("Sants",                   41.3747,  2.1330, "residential"),
            # Mixed
            ("El Born",                 41.3852,  2.1831, "mixed"),
            ("Poblenou",                41.3956,  2.2005, "mixed"),
            # Industrial
            ("Zona Franca",             41.3500,  2.1300, "industrial"),
            ("22@ District",            41.4035,  2.1940, "industrial"),
            # Natural
            ("Parc de la Ciutadella",   41.3882,  2.1875, "natural"),
            ("Montjuic",                41.3636,  2.1586, "natural"),
        ],
    },

    # ─────────────────────────────────────────────
    # SCANDINAVIA (Nordic island-archipelago city)
    # ─────────────────────────────────────────────

    "stockholm": {
        "name": "Stockholm",
        "country": "Sweden",
        "bbox": (59.290, 17.940, 59.370, 18.130),
        "seeds": [
            # Permanent iconic anchors
            ("Drottninggatan",          59.3326, 18.0649, "commercial"),
            ("Stureplan",               59.3370, 18.0740, "commercial"),
            # Residential
            ("Sodermalm",               59.3150, 18.0700, "residential"),
            ("Ostermalm",               59.3400, 18.0850, "residential"),
            ("Vasastan",                59.3440, 18.0500, "residential"),
            # Mixed
            ("Gamla Stan",              59.3258, 18.0716, "mixed"),
            ("Kungsholmen",             59.3340, 18.0300, "mixed"),
            # Industrial
            ("Hammarby Sjostad",        59.3040, 18.1050, "industrial"),
            ("Liljeholmen",             59.3100, 18.0230, "industrial"),
            # Natural
            ("Djurgarden",              59.3270, 18.1100, "natural"),
            ("Hagaparken",              59.3600, 18.0300, "natural"),
        ],
    },

    # ─────────────────────────────────────────────
    # NORDIC / BALTIC (Compact grid peninsula)
    # ─────────────────────────────────────────────

    "helsinki": {
        "name": "Helsinki",
        "country": "Finland",
        "bbox": (60.140, 24.870, 60.210, 25.020),
        "seeds": [
            # Permanent iconic anchors
            ("Keskusta Aleksanterinkatu", 60.1695, 24.9458, "commercial"),
            ("Kamppi",                    60.1685, 24.9320, "commercial"),
            # Residential
            ("Kallio",                    60.1840, 24.9520, "residential"),
            ("Toolo",                     60.1780, 24.9220, "residential"),
            ("Kruununhaka",               60.1720, 24.9560, "residential"),
            # Mixed
            ("Punavuori",                 60.1620, 24.9380, "mixed"),
            ("Jatkasaari",                60.1560, 24.9120, "mixed"),
            # Industrial
            ("Hernesaari",                60.1510, 24.9210, "industrial"),
            ("Suvilahti",                 60.1870, 24.9700, "industrial"),
            # Natural
            ("Kaivopuisto",               60.1580, 24.9540, "natural"),
            ("Seurasaari",                60.1820, 24.8850, "natural"),
        ],
    },

    # ─────────────────────────────────────────────
    # EASTERN MEDITERRANEAN (Ancient-modern collision)
    # ─────────────────────────────────────────────

    "athens": {
        "name": "Athens",
        "country": "Greece",
        "bbox": (37.940, 23.680, 38.010, 23.790),
        "seeds": [
            # Permanent iconic anchors
            ("Ermou Street",            37.9770, 23.7270, "commercial"),
            ("Syntagma",                37.9755, 23.7348, "commercial"),
            # Residential
            ("Pangrati",                37.9680, 23.7460, "residential"),
            ("Koukaki",                 37.9650, 23.7260, "residential"),
            ("Kypseli",                 37.9930, 23.7370, "residential"),
            # Mixed
            ("Plaka",                   37.9730, 23.7320, "mixed"),
            ("Exarchia",                37.9860, 23.7340, "mixed"),
            # Industrial
            ("Gazi",                    37.9780, 23.7130, "industrial"),
            ("Piraeus Port",            37.9470, 23.6380, "industrial"),
            # Natural
            ("National Garden",         37.9720, 23.7370, "natural"),
            ("Filopappou Hill",         37.9680, 23.7210, "natural"),
        ],
    },

}

# How many samples per city
SAMPLES_PER_CITY = 120

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
  node["highway"="traffic_signals"]
    ({s},{w},{n},{e});
  way["natural"="water"]
    ({s},{w},{n},{e});
  way["waterway"]
    ({s},{w},{n},{e});
  node["highway"="bus_stop"]
    ({s},{w},{n},{e});
  node["railway"="station"]
    ({s},{w},{n},{e});
  node["railway"="tram_stop"]
    ({s},{w},{n},{e});
  node["public_transport"="stop_position"]
    ({s},{w},{n},{e});
);
out tags center;
"""

# Lightweight query: roads only (much smaller, for road snapping).
# Motorways are excluded: they often run through tunnels/elevated sections
# where Street View imagery is useless for urban VQA, and they bias the
# road_type answer distribution toward a single category.
OSM_ROADS_QUERY_TEMPLATE = """
[out:json][timeout:120];
way["highway"~"^(primary|secondary|tertiary|residential|trunk)$"]["tunnel"!="yes"]["covered"!="yes"]
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

# Road surface classification (from OSM surface tag)
ROAD_SURFACE_BINS = {
    "asphalt": "Asphalt", "paved": "Asphalt",
    "concrete": "Concrete", "concrete:plates": "Concrete",
    "cobblestone": "Cobblestone/Sett", "sett": "Cobblestone/Sett",
    "paving_stones": "Cobblestone/Sett",
    "unpaved": "Unpaved/Gravel", "gravel": "Unpaved/Gravel",
    "dirt": "Unpaved/Gravel", "ground": "Unpaved/Gravel",
    "sand": "Unpaved/Gravel", "compacted": "Unpaved/Gravel",
}
ROAD_SURFACE_OPTIONS = ["Asphalt", "Cobblestone/Sett", "Unpaved/Gravel", "Concrete"]

# Junction type classification
JUNCTION_TYPES = {
    "roundabout": "Roundabout",
    "signalized": "Signalized intersection",
    "unsignalized": "Unsignalized intersection",
    "grade_separated": "Grade-separated (overpass/underpass)",
}

# Water proximity bins (label, min_m, max_m)
WATER_PROXIMITY_BINS = [
    ("Yes — directly adjacent to a water body", 0, 50),
    ("Yes — nearby but not adjacent", 50, 150),
    ("No water body in view", 150, 99999),
]

# Transit stop density bins (label, min_count, max_count)
TRANSIT_DENSITY_BINS = [
    ("None (no transit stops within 300m)", 0, 0),
    ("Low (1–2 stops)", 1, 2),
    ("Moderate (3–5 stops)", 3, 5),
    ("High (6+ stops)", 6, 9999),
]

# Which question types to generate (comment out to disable, code is preserved)
ENABLED_QUESTION_TYPES = {
    "land_use",
    "building_height",
    "urban_density",
    "road_type",
    "road_surface",
    "junction_type",
    "water_proximity",
    "green_space",
    "amenity_richness",
    "transit_density",
    "camera_direction",
    "mismatch_binary",
    "mismatch_mcq",
}

# Geolocation mismatch strategies: "cross_city", "same_city", or "both"
# "both" generates one variant of each per sample
MISMATCH_MCQ_STRATEGY = "both"
MISMATCH_BINARY_STRATEGY = "both"

# Satellite tile settings
# SAT_BUFFER_M: radius in metres around the sample point. Smaller = more zoomed in.
# Adjust these to control detail level per source.
SAT_BUFFER_M = {
    "NAIP": 250,   # 250m → ~500m tile (0.6 m/px native — very detailed)
    "IGN":  100,   # 200m → ~400m tile (0.2 m/px native — extremely detailed)
    "ESRI": 100,   # 200m → ~400m tile (0.3–0.5 m/px in cities)
    "S2":   1280,  # 2560m → ~5120m tile (10 m/px native — can't zoom further)
}
SAT_IMAGE_PX = 512

# ---------------------------------------------------------------------------
# Satellite source auto-detection (by coordinates)
# ---------------------------------------------------------------------------
_USA_BOXES = [
    (-125.0, 24.0, -66.0, 50.0),    # Continental US
    (-180.0, 51.0, -130.0, 72.0),   # Alaska
    (-161.0, 18.5, -154.5, 22.5),   # Hawaii
]
_FRANCE_BOX = (-5.5, 41.0, 10.0, 51.5)   # Metropolitan France + Corsica


def _in_box(lat, lon, box):
    lon_min, lat_min, lon_max, lat_max = box
    return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max


def detect_sat_source(lat: float, lon: float) -> str:
    """Return the best satellite source tag for a given coordinate."""
    if any(_in_box(lat, lon, b) for b in _USA_BOXES):
        return "NAIP"
    if _in_box(lat, lon, _FRANCE_BOX):
        return "IGN"
    return "ESRI"

# GEE config
#GEE_PROJECT = "supple-flux-481209-j1" #alperen
GEE_PROJECT = "bitirme-489511"
NAIP_DATE_RANGE = ("2019-01-01", "2025-12-31")
S2_DATE_RANGE = ("2023-01-01", "2025-12-31")
S2_CLOUD_THRESHOLD = 20  # max cloud %

# Street View config
SV_SIZE = "640x640"
SV_FOV = 90
SV_PITCH = -5
