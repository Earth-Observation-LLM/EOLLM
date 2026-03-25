"""
City-agnostic configuration for the Urban VQA pipeline.
Defines cities, seed locations, and all shared constants.
"""

CITIES = {

    # ─────────────────────────────────────────────
    # EUROPE (Ultra-dense OSM + Strong STV)
    # ─────────────────────────────────────────────

    "berlin": {
        "name": "Berlin",
        "country": "Germany",
        "bbox": (52.400, 13.200, 52.600, 13.600),
        "seeds": [
            # Permanent iconic anchors
            ("Brandenburg Gate",        52.5163,  13.3777, "commercial"),
            ("Alexanderplatz",          52.5219,  13.4132, "commercial"),
            # Residential
            ("Kreuzberg",               52.4984,  13.3913, "residential"),
            ("Charlottenburg",          52.5167,  13.3000, "residential"),
            ("Prenzlauer Berg",         52.5398,  13.4246, "residential"),
            # Commercial / Mixed
            ("Potsdamer Platz",         52.5096,  13.3759, "mixed"),
            ("Kurfurstendamm",          52.5037,  13.3115, "commercial"),
            # Industrial
            ("Siemensstadt",            52.5388,  13.2644, "industrial"),
            ("Adlershof Tech Park",     52.4334,  13.5350, "industrial"),
            # Natural
            ("Tempelhofer Feld",        52.4731,  13.4016, "natural"),
            ("Tiergarten",              52.5144,  13.3501, "natural"),
        ],
    },

    "amsterdam": {
        "name": "Amsterdam",
        "country": "Netherlands",
        "bbox": (52.270, 4.730, 52.430, 5.010),
        "seeds": [
            # Permanent iconic anchors
            ("Dam Square",              52.3729,  4.8930,  "commercial"),
            ("Rijksmuseum",             52.3600,  4.8852,  "mixed"),
            # Residential
            ("Jordaan",                 52.3740,  4.8799,  "residential"),
            ("De Pijp",                 52.3533,  4.8970,  "residential"),
            ("Oud-West",                52.3631,  4.8660,  "residential"),
            # Commercial / Business
            ("Zuidas Business District",52.3389,  4.8741,  "commercial"),
            ("Centrum",                 52.3702,  4.8951,  "commercial"),
            # Industrial
            ("Westpoort",               52.3995,  4.8016,  "industrial"),
            ("NDSM Wharf",              52.4005,  4.8941,  "industrial"),
            # Natural / Water
            ("Vondelpark",              52.3580,  4.8686,  "natural"),
            ("Amstel River",            52.3546,  4.9038,  "natural"),
        ],
    },

    # ─────────────────────────────────────────────
    # NORTH AMERICA (Grid systems + Perfect STV)
    # ─────────────────────────────────────────────

    "toronto": {
        "name": "Toronto",
        "country": "Canada",
        "bbox": (43.580, -79.640, 43.850, -79.120),
        "seeds": [
            # Permanent iconic anchors
            ("CN Tower",                43.6426, -79.3871, "commercial"),
            ("Yonge-Dundas Square",     43.6561, -79.3802, "commercial"),
            # Residential
            ("The Annex",               43.6703, -79.4071, "residential"),
            ("Leslieville",             43.6631, -79.3297, "residential"),
            ("Scarborough",             43.7731, -79.2577, "residential"),
            # Commercial
            ("Financial District",      43.6486, -79.3815, "commercial"),
            ("Liberty Village",         43.6373, -79.4181, "mixed"),
            # Industrial
            ("Port Lands",              43.6436, -79.3330, "industrial"),
            ("Etobicoke Industrial",    43.6186, -79.5249, "industrial"),
            # Natural
            ("High Park",               43.6465, -79.4637, "natural"),
            ("Toronto Islands",         43.6210, -79.3780, "natural"),
        ],
    },

    "chicago": {
        "name": "Chicago",
        "country": "USA",
        "bbox": (41.640, -87.940, 42.020, -87.520),
        "seeds": [
            # Permanent iconic anchors
            ("Willis Tower",            41.8789, -87.6359, "commercial"),
            ("Navy Pier",               41.8919, -87.6051, "mixed"),
            # Residential
            ("Lincoln Park",            41.9214, -87.6513, "residential"),
            ("Hyde Park",               41.7943, -87.5907, "residential"),
            ("Logan Square",            41.9284, -87.7073, "residential"),
            # Commercial
            ("Magnificent Mile",        41.8948, -87.6242, "commercial"),
            ("The Loop",                41.8837, -87.6289, "commercial"),
            # Industrial
            ("Calumet Industrial",      41.7160, -87.5670, "industrial"),
            ("Fulton Market",           41.8867, -87.6508, "mixed"),
            # Natural
            ("Millennium Park",         41.8826, -87.6226, "natural"),
            ("Lakefront Trail",         41.9056, -87.6163, "natural"),
        ],
    },

    # ─────────────────────────────────────────────
    # ASIA (Dense Urbanism + Flawless STV)
    # ─────────────────────────────────────────────

    "singapore": {
        "name": "Singapore",
        "country": "Singapore",
        "bbox": (1.200, 103.590, 1.480, 104.050),
        "seeds": [
            # Permanent iconic anchors
            ("Marina Bay Sands",        1.2834,  103.8607, "commercial"),
            ("Merlion Park",            1.2868,  103.8545, "commercial"),
            # Residential
            ("Tampines HDB",            1.3526,  103.9447, "residential"),
            ("Woodlands",               1.4369,  103.7865, "residential"),
            ("Bedok",                   1.3236,  103.9273, "residential"),
            # Commercial
            ("Orchard Road",            1.3048,  103.8318, "commercial"),
            ("Tiong Bahru",             1.2862,  103.8318, "mixed"),
            # Industrial
            ("Jurong Industrial",       1.2818,  103.7027, "industrial"),
            ("Tuas",                    1.3283,  103.6394, "industrial"),
            # Natural
            ("Botanic Gardens",         1.3138,  103.8159, "natural"),
            ("East Coast Park",         1.3008,  103.9122, "natural"),
        ],
    },

    "taipei": {
        "name": "Taipei",
        "country": "Taiwan",
        "bbox": (24.960, 121.450, 25.210, 121.660),
        "seeds": [
            # Permanent iconic anchors
            ("Taipei 101",              25.0339, 121.5645, "commercial"),
            ("Chiang Kai-shek Memorial",25.0347, 121.5218, "mixed"),
            # Residential
            ("Da'an District",          25.0263, 121.5434, "residential"),
            ("Shilin District",         25.0886, 121.5244, "residential"),
            ("Zhonghe",                 24.9984, 121.5032, "residential"),
            # Commercial
            ("Ximending",               25.0432, 121.5065, "commercial"),
            ("Xinyi District",          25.0326, 121.5684, "commercial"),
            # Industrial
            ("Neihu Tech Park",         25.0796, 121.5755, "industrial"),
            ("Nangang Software Park",   25.0597, 121.6141, "industrial"),
            # Natural
            ("Yangmingshan",            25.1558, 121.5484, "natural"),
            ("Dajia Riverside Park",    25.0747, 121.5358, "natural"),
        ],
    },

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
GEE_PROJECT = "supple-flux-481209-j1"
NAIP_DATE_RANGE = ("2019-01-01", "2025-12-31")
S2_DATE_RANGE = ("2023-01-01", "2025-12-31")
S2_CLOUD_THRESHOLD = 20  # max cloud %

# Street View config
SV_SIZE = "640x640"
SV_FOV = 90
SV_PITCH = -5
