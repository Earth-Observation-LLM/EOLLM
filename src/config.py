"""
City-agnostic configuration for the Urban VQA pipeline.
Defines cities, seed locations, and all shared constants.
"""

CITIES = {

    # ─────────────────────────────────────────────
    # TURKEY
    # ─────────────────────────────────────────────

    "istanbul": {
        "name": "Istanbul",
        "country": "Turkey",
        "bbox": (40.840, 28.550, 41.200, 29.450),  # S, W, N, E (S expanded for Tuzla Shipyard)
        "seeds": [
            # Permanent / iconic anchors
            ("Hagia Sophia",            41.0086,  28.9802, "commercial"),
            ("Grand Bazaar",            41.0108,  28.9680, "commercial"),
            ("Galata Tower",            41.0256,  28.9742, "mixed"),
            # Residential
            ("Kadikoy",                 40.9909,  29.0283, "residential"),
            ("Uskudar",                 41.0231,  29.0151, "residential"),
            ("Bagcilar",                41.0375,  28.8566, "residential"),
            ("Umraniye",                41.0167,  29.1167, "residential"),
            # Commercial / business
            ("Levent",                  41.0797,  29.0124, "commercial"),
            ("Sisli",                   41.0602,  28.9877, "commercial"),
            # Industrial
            ("Ikitelli OSB",            41.0630,  28.7960, "industrial"),
            ("Tuzla Shipyard",          40.8467,  29.3000, "industrial"),
            # Natural / water-adjacent (Bosphorus is permanent)
            ("Bosphorus Strait North",  41.1767,  29.0684, "mixed"),
            ("Belgrade Forest",         41.1800,  28.9600, "natural"),
        ],
    },

    "ankara": {
        "name": "Ankara",
        "country": "Turkey",
        "bbox": (39.750, 32.550, 40.100, 33.000),
        "seeds": [
            # Permanent / iconic anchors
            ("Ataturk Mausoleum Anitkabir", 39.9255, 32.8369, "commercial"),
            ("Ankara Castle",               39.9408, 32.8641, "mixed"),
            # Residential
            ("Cankaya",                     39.8900, 32.8597, "residential"),
            ("Kecioren",                    40.0000, 32.8833, "residential"),
            ("Batikent",                    39.9500, 32.7333, "residential"),
            ("Etimesgut",                   39.9500, 32.6833, "residential"),
            # Commercial
            ("Kizilay",                     39.9208, 32.8541, "commercial"),
            ("Sogutozu",                    39.9000, 32.8167, "commercial"),
            # Industrial
            ("Ostim OSB",                   39.9667, 32.7833, "industrial"),
            ("Sincan OSB",                  39.9667, 32.6167, "industrial"),
            # Natural
            ("Eymir Lake",                  39.8100, 32.7750, "natural"),
            ("Ulus Old Quarter",            39.9408, 32.8553, "mixed"),
        ],
    },

    "izmir": {
        "name": "Izmir",
        "country": "Turkey",
        "bbox": (38.280, 26.900, 38.600, 27.430),  # E expanded for Kemalpasa OSB
        "seeds": [
            # Permanent / iconic anchors
            ("Izmir Clock Tower",       38.4127,  27.1384, "commercial"),
            ("Kadifekale Castle",       38.4069,  27.1456, "mixed"),
            # Residential
            ("Karsiyaka",              38.4566,  27.1114, "residential"),
            ("Bornova",                38.4669,  27.2197, "residential"),
            ("Buca",                   38.3833,  27.1833, "residential"),
            # Commercial
            ("Konak",                  38.4127,  27.1384, "commercial"),
            ("Bayrakli",               38.4667,  27.1667, "commercial"),
            # Industrial
            ("Cigli Industrial Zone",  38.5167,  27.0167, "industrial"),
            ("Kemalpasa OSB",          38.4167,  27.4167, "industrial"),
            # Natural / water
            ("Izmir Bay Waterfront",   38.4200,  27.1000, "mixed"),
            ("Karagol Park",           38.3667,  27.1500, "natural"),
        ],
    },

    "bursa": {
        "name": "Bursa",
        "country": "Turkey",
        "bbox": (40.070, 28.900, 40.450, 29.300),  # N expanded for Gemlik Port, S expanded for Uludag
        "seeds": [
            # Permanent / iconic anchors
            ("Grand Mosque Ulu Camii",  40.1833,  29.0601, "commercial"),
            ("Bursa Castle",            40.1867,  29.0584, "mixed"),
            # Residential
            ("Nilufer",                 40.2167,  28.9667, "residential"),
            ("Osmangazi",               40.1833,  29.0500, "residential"),
            ("Yildirim",                40.1833,  29.1000, "residential"),
            # Commercial
            ("Uludag Ski Resort",       40.0833,  29.1333, "commercial"),
            ("Organize Sanayi Bursa",   40.2500,  29.0833, "commercial"),
            # Industrial
            ("DOSAB Industrial Zone",   40.2333,  28.9833, "industrial"),
            ("Gemlik Port",             40.4333,  29.1500, "industrial"),
            # Natural
            ("Uludag National Park",    40.0833,  29.1667, "natural"),
            # NOTE: Iznik Lake removed — outside bbox and no urban roads
        ],
    },

    "kayseri": {
        "name": "Kayseri",
        "country": "Turkey",
        "bbox": (38.600, 35.350, 38.850, 35.700),
        "seeds": [
            # Permanent / iconic anchors
            ("Kayseri Castle",          38.7200,  35.4878, "commercial"),
            # NOTE: Erciyes Mountain removed — outside bbox (lat 38.53 < S 38.60)
            # NOTE: Sultansazligi Wetland removed — outside bbox, no urban roads
            # Residential
            ("Melikgazi",               38.7167,  35.4833, "residential"),
            ("Kocasinan",               38.7333,  35.5333, "residential"),
            # Commercial
            ("Kayseri City Center",     38.7167,  35.4833, "commercial"),
            ("Forum Kayseri Mall Area", 38.7000,  35.5167, "commercial"),
            # Industrial
            ("Kayseri OSB 1",           38.7667,  35.5500, "industrial"),
            ("Kayseri OSB 2",           38.7833,  35.5833, "industrial"),
        ],
    },

    

    # ─────────────────────────────────────────────
    # GLOBAL CITIES
    # ─────────────────────────────────────────────

    "nyc": {
        "name": "New York City",
        "country": "USA",
        "bbox": (40.490, -74.260, 40.920, -73.680),
        "seeds": [
            ("Times Square",            40.7580, -73.9855, "commercial"),
            ("Wall Street",             40.7074, -74.0113, "commercial"),
            ("Williamsburg",            40.7150, -73.9600, "residential"),
            ("DUMBO Brooklyn",          40.7033, -73.9903, "mixed"),
            ("Long Island City",        40.7425, -73.9580, "industrial"),
            ("Upper East Side",         40.7736, -73.9566, "residential"),
            ("Harlem",                  40.8116, -73.9465, "residential"),
            ("Flushing Queens",         40.7614, -73.8300, "commercial"),
            ("South Bronx",             40.8090, -73.9230, "residential"),
            ("Central Park Reservoir",  40.7851, -73.9654, "natural"),
            ("JFK Airport",             40.6413, -73.7781, "industrial"),
        ],
    },

    "paris": {
        "name": "Paris",
        "country": "France",
        "bbox": (48.815, 2.225, 48.902, 2.420),
        "seeds": [
            # Permanent iconic anchors
            ("Eiffel Tower",            48.8584,  2.2945, "commercial"),
            ("Notre-Dame",              48.8530,  2.3499, "commercial"),
            ("Louvre Museum",           48.8606,  2.3376, "commercial"),
            # Residential
            ("Le Marais",               48.8566,  2.3622, "residential"),
            ("Saint-Germain",           48.8540,  2.3338, "residential"),
            ("Montmartre",              48.8867,  2.3431, "residential"),
            ("Batignolles",             48.8860,  2.3190, "residential"),
            # Commercial / mixed
            ("Champs-Elysees",          48.8738,  2.2950, "commercial"),
            ("La Defense",              48.8920,  2.2360, "commercial"),
            ("Republique",              48.8676,  2.3640, "mixed"),
            # Industrial / mixed
            ("Bercy",                   48.8367,  2.3750, "mixed"),
            ("Belleville",              48.8714,  2.3847, "mixed"),
            # Natural / water (Seine is permanent)
            ("Bois de Boulogne",        48.8622,  2.2479, "natural"),
        ],
    },

    "london": {
        "name": "London",
        "country": "UK",
        "bbox": (51.285, -0.510, 51.690, 0.335),
        "seeds": [
            # Permanent iconic anchors
            ("Tower of London",         51.5081, -0.0759, "commercial"),
            ("Big Ben Westminster",     51.4994, -0.1245, "commercial"),
            # Residential
            ("Brixton",                 51.4613, -0.1146, "residential"),
            ("Kensington",              51.4990, -0.1940, "residential"),
            ("Greenwich",               51.4769, -0.0005, "residential"),
            # Commercial
            ("City of London",          51.5133, -0.0886, "commercial"),
            ("Canary Wharf",            51.5054, -0.0235, "commercial"),
            ("Croydon",                 51.3762, -0.0986, "commercial"),
            # Mixed
            ("Camden Town",             51.5392, -0.1426, "mixed"),
            ("Shoreditch",              51.5274, -0.0777, "mixed"),
            ("Stratford",               51.5430, -0.0098, "mixed"),
            # Natural / water (Hyde Park lake is permanent)
            ("Hyde Park Serpentine",    51.5054, -0.1686, "natural"),
        ],
    },

    "tokyo": {
        "name": "Tokyo",
        "country": "Japan",
        "bbox": (35.530, 139.560, 35.820, 139.920),
        "seeds": [
            # Permanent iconic anchors
            ("Tokyo Imperial Palace",   35.6852, 139.7528, "mixed"),
            ("Senso-ji Asakusa",        35.7148, 139.7967, "commercial"),
            ("Tokyo Tower",             35.6586, 139.7454, "commercial"),
            # Residential
            ("Setagaya",                35.6467, 139.6531, "residential"),
            ("Adachi",                  35.7750, 139.8050, "residential"),
            ("Nerima",                  35.7356, 139.6517, "residential"),
            # Commercial
            ("Shinjuku",                35.6938, 139.7034, "commercial"),
            ("Shibuya",                 35.6598, 139.7004, "commercial"),
            ("Akihabara",               35.7022, 139.7741, "commercial"),
            # Industrial
            ("Keihin Industrial Zone",  35.5500, 139.7500, "industrial"),
            ("Ota Ward",                35.5619, 139.7161, "industrial"),
            # Natural / water (Tokyo Bay is permanent)
            ("Odaiba Tokyo Bay",        35.6245, 139.7798, "mixed"),
            ("Shinjuku Gyoen Park",     35.6852, 139.7100, "natural"),
        ],
    },

    "sydney": {
        "name": "Sydney",
        "country": "Australia",
        "bbox": (-34.170, 150.650, -33.580, 151.350),
        "seeds": [
            # Permanent iconic anchors
            ("Sydney Opera House",      -33.8568, 151.2153, "commercial"),
            ("Harbour Bridge",          -33.8523, 151.2108, "commercial"),
            # Residential
            ("Parramatta",              -33.8150, 151.0011, "residential"),
            ("Penrith",                 -33.7511, 150.6942, "residential"),
            ("Bondi",                   -33.8915, 151.2767, "residential"),
            # Commercial
            ("Sydney CBD",              -33.8688, 151.2093, "commercial"),
            ("Chatswood",               -33.7969, 151.1825, "commercial"),
            # Industrial
            ("Port Botany",             -33.9711, 151.2036, "industrial"),
            ("Wetherill Park",          -33.8500, 150.9000, "industrial"),
            # Natural / water (Sydney Harbour is permanent drowned valley)
            ("Royal National Park",     -34.0700, 151.0500, "natural"),
            ("Manly Beach",             -33.7969, 151.2869, "natural"),
        ],
    },

    "rio": {
        "name": "Rio de Janeiro",
        "country": "Brazil",
        "bbox": (-23.090, -43.800, -22.730, -43.100),
        "seeds": [
            # Permanent iconic anchors
            ("Christ the Redeemer",     -22.9519, -43.2105, "commercial"),
            ("Maracana Stadium",        -22.9122, -43.2302, "commercial"),
            # Residential
            ("Copacabana",              -22.9714, -43.1823, "residential"),
            ("Barra da Tijuca",         -23.0000, -43.3653, "residential"),
            ("Rocinha Favela",          -22.9875, -43.2481, "residential"),
            # Commercial
            ("Centro Rio",              -22.9068, -43.1729, "commercial"),
            ("Ipanema",                 -22.9839, -43.2025, "commercial"),
            # Industrial
            ("Port Zone Rio",           -22.8900, -43.1900, "industrial"),
            ("Santa Cruz Industrial",   -22.9500, -43.7000, "industrial"),
            # Natural / water (Guanabara Bay is permanent)
            ("Tijuca Forest",           -22.9333, -43.2833, "natural"),
            ("Guanabara Bay",           -22.8333, -43.1667, "natural"),
        ],
    },

    "cairo": {
        "name": "Cairo",
        "country": "Egypt",
        "bbox": (29.850, 31.050, 30.310, 31.760),  # E expanded for 10th Ramadan City, N expanded for 10th Ramadan lat
        "seeds": [
            # Permanent iconic anchors (pyramids = most stable landmark on Earth)
            ("Giza Pyramids",           29.9792,  31.1342, "commercial"),
            ("Cairo Citadel",           30.0288,  31.2599, "mixed"),
            ("Egyptian Museum",         30.0478,  31.2336, "commercial"),
            # Residential
            ("Heliopolis",              30.0875,  31.3219, "residential"),
            ("Nasr City",               30.0667,  31.3500, "residential"),
            ("Zamalek Island",          30.0619,  31.2214, "residential"),
            # Commercial
            ("Downtown Cairo",          30.0444,  31.2357, "commercial"),
            ("New Cairo",               30.0167,  31.4667, "commercial"),
            # Industrial
            ("Shubra El-Kheima",        30.1167,  31.2500, "industrial"),
            ("10th Ramadan City",       30.3000,  31.7500, "industrial"),
            # Natural / water (Nile River is permanent)
            ("Nile at Gezira",          30.0500,  31.2250, "natural"),
        ],
    },

    "rome": {
        "name": "Rome",
        "country": "Italy",
        "bbox": (41.790, 12.350, 42.000, 12.620),
        "seeds": [
            # Permanent iconic anchors (2000+ year old structures)
            ("Colosseum",               41.8902,  12.4922, "commercial"),
            ("Vatican City",            41.9022,  12.4539, "commercial"),
            ("Pantheon",                41.8986,  12.4769, "commercial"),
            # Residential
            ("Parioli",                 41.9233,  12.4908, "residential"),
            ("EUR District",            41.8281,  12.4680, "residential"),
            ("Trastevere",              41.8861,  12.4684, "residential"),
            # Commercial
            ("Termini Station Area",    41.9009,  12.5005, "commercial"),
            ("Prati",                   41.9086,  12.4586, "commercial"),
            # Industrial
            ("Ostiense",                41.8667,  12.4833, "industrial"),
            ("Tiburtina Industrial",    41.9167,  12.5500, "industrial"),
            # Natural / water (Tiber is permanent)
            ("Tiber River at Castel",   41.9022,  12.4659, "natural"),
            ("Villa Borghese Park",     41.9142,  12.4922, "natural"),
        ],
    },

    "mumbai": {
        "name": "Mumbai",
        "country": "India",
        "bbox": (18.870, 72.770, 19.270, 72.990),
        "seeds": [
            # Permanent iconic anchors
            ("Gateway of India",        18.9220,  72.8347, "commercial"),
            ("Chhatrapati Shivaji Terminal", 18.9400, 72.8356, "commercial"),
            # Residential
            ("Bandra West",             19.0596,  72.8295, "residential"),
            ("Borivali",                19.2288,  72.8561, "residential"),
            ("Dharavi",                 19.0432,  72.8520, "residential"),
            # Commercial
            ("Nariman Point",           18.9256,  72.8242, "commercial"),
            ("Bandra Kurla Complex",    19.0646,  72.8652, "commercial"),
            # Industrial
            ("Chembur Industrial",      19.0522,  72.9005, "industrial"),
            ("Thane Industrial",        19.2183,  72.9781, "industrial"),
            # Natural / water (Arabian Sea coastline is permanent)
            ("Marine Drive",            18.9433,  72.8236, "natural"),
            ("Sanjay Gandhi National Park", 19.2147, 72.9140, "natural"),
        ],
    },


    "nairobi": {
        "name": "Nairobi",
        "country": "Kenya",
        "bbox": (-1.450, 36.650, -1.130, 37.100),
        "seeds": [
            # Permanent iconic anchors
            ("Nairobi National Park",   -1.3500,  36.8167, "natural"),
            ("University of Nairobi",   -1.2800,  36.8219, "commercial"),
            # Residential
            ("Westlands",               -1.2667,  36.8083, "residential"),
            ("Eastleigh",               -1.2667,  36.8500, "residential"),
            ("Karen",                   -1.3333,  36.7167, "residential"),
            # Commercial
            ("CBD Nairobi",             -1.2833,  36.8167, "commercial"),
            ("Upper Hill",              -1.2950,  36.8150, "commercial"),
            # Industrial
            ("Industrial Area Nairobi", -1.3000,  36.8500, "industrial"),
            ("Athi River",              -1.4500,  36.9833, "industrial"),
            # Natural
            ("Karura Forest",           -1.2333,  36.8333, "natural"),
            ("Ngong Hills",             -1.3833,  36.6667, "natural"),
        ],
    },

    "cape_town": {
        "name": "Cape Town",
        "country": "South Africa",
        "bbox": (-34.200, 18.300, -33.750, 18.950),
        "seeds": [
            # Permanent iconic anchors (Table Mountain is permanent geology)
            ("Table Mountain",          -33.9626,  18.4098, "natural"),
            # NOTE: Cape Point removed — outside bbox (lat -34.36 < S -34.20)
            # Residential
            ("Camps Bay",               -33.9500,  18.3767, "residential"),
            ("Mitchells Plain",         -34.0333,  18.6167, "residential"),
            ("Khayelitsha",             -34.0333,  18.6833, "residential"),
            # Commercial
            ("Cape Town CBD",           -33.9258,  18.4232, "commercial"),
            ("Century City",            -33.8900,  18.5150, "commercial"),
            # Industrial
            ("Epping Industrial",       -33.9333,  18.5500, "industrial"),
            ("Port of Cape Town",       -33.9050,  18.4350, "industrial"),
            # Natural / water (Atlantic Ocean is permanent)
            ("Boulders Beach",          -34.1968,  18.4497, "natural"),
            ("Bloubergstrand",          -33.8000,  18.4667, "natural"),
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
