"""
City-agnostic configuration for the Urban VQA pipeline.
Defines cities, seed locations, and all shared constants.
"""

CITIES = {


    # ─────────────────────────────────────────────
    # TURKIYE (Black Sea coast + Mediterranean)
    # ─────────────────────────────────────────────

    "samsun": {
        "name": "Samsun",
        "country": "Turkiye",
        "bbox": (41.260, 36.280, 41.330, 36.380),
        "seeds": [
            # Permanent iconic anchors
            ("Cumhuriyet Meydani",      41.2867, 36.3300, "commercial"),
            ("Atakum Sahil",            41.3280, 36.2800, "commercial"),
            # Residential
            ("Ilkadim",                 41.2900, 36.3350, "residential"),
            ("Canik",                   41.2750, 36.3550, "residential"),
            ("Atakum Inland",           41.3100, 36.2900, "residential"),
            # Mixed
            ("Istiklal Caddesi",        41.2880, 36.3280, "mixed"),
            ("Liman District",          41.2920, 36.3400, "mixed"),
            # Industrial
            ("Organized Industrial",    41.2650, 36.3600, "industrial"),
            ("Gelemen",                 41.2700, 36.3700, "industrial"),
            # Natural
            ("Amisos Hill",             41.2830, 36.3200, "natural"),
            ("Kizilirmak Delta Edge",   41.3200, 36.3500, "natural"),
        ],
    },

    "antalya": {
        "name": "Antalya",
        "country": "Turkiye",
        "bbox": (36.850, 30.630, 36.920, 30.750),
        "seeds": [
            # Permanent iconic anchors
            ("Kaleici Old Town",        36.8840, 30.7060, "commercial"),
            ("MarkAntalya",             36.8920, 30.6830, "commercial"),
            # Residential
            ("Konyaalti",               36.8800, 30.6500, "residential"),
            ("Lara",                    36.8600, 30.7400, "residential"),
            ("Muratpasa",               36.8870, 30.7000, "residential"),
            # Mixed
            ("Isiklar Caddesi",         36.8890, 30.7030, "mixed"),
            ("Sirinyali",               36.8750, 30.6700, "mixed"),
            # Industrial
            ("Organized Sanayi",        36.9150, 30.6400, "industrial"),
            ("Aksu",                    36.8800, 30.7600, "industrial"),
            # Natural
            ("Duden Waterfalls",        36.8700, 30.7350, "natural"),
            ("Konyaalti Beach Park",    36.8770, 30.6450, "natural"),
        ],
    },

    # ─────────────────────────────────────────────
    # WESTERN EUROPE (Swiss precision urbanism)
    # ─────────────────────────────────────────────

    "zurich": {
        "name": "Zurich",
        "country": "Switzerland",
        "bbox": (47.340, 8.490, 47.410, 8.580),
        "seeds": [
            # Permanent iconic anchors
            ("Bahnhofstrasse",          47.3690, 8.5390, "commercial"),
            ("Paradeplatz",             47.3698, 8.5392, "commercial"),
            # Residential
            ("Wiedikon",                47.3630, 8.5200, "residential"),
            ("Oerlikon",                47.4100, 8.5450, "residential"),
            ("Seefeld",                 47.3580, 8.5550, "residential"),
            # Mixed
            ("Langstrasse",             47.3770, 8.5280, "mixed"),
            ("Niederdorf",              47.3730, 8.5440, "mixed"),
            # Industrial
            ("Zurich West",             47.3880, 8.5200, "industrial"),
            ("Altstetten",              47.3900, 8.4880, "industrial"),
            # Natural
            ("Zurichsee Promenade",     47.3540, 8.5440, "natural"),
            ("Lindenhof",               47.3730, 8.5400, "natural"),
        ],
    },

    # ─────────────────────────────────────────────
    # RUSSIA (Imperial capital, canal city)
    # ─────────────────────────────────────────────

    "st_petersburg": {
        "name": "St. Petersburg",
        "country": "Russia",
        "bbox": (59.880, 30.230, 59.980, 30.400),
        "seeds": [
            # Permanent iconic anchors
            ("Nevsky Prospekt",         59.9340, 30.3350, "commercial"),
            ("Sennaya Ploshchad",       59.9270, 30.3200, "commercial"),
            # Residential
            ("Vasilievsky Island",      59.9420, 30.2700, "residential"),
            ("Petrogradskaya",          59.9660, 30.3100, "residential"),
            ("Kupchino",                59.8300, 30.3800, "residential"),
            # Mixed
            ("Admiralteysky",           59.9300, 30.3100, "mixed"),
            ("Liteiny Prospekt",        59.9400, 30.3500, "mixed"),
            # Industrial
            ("Obvodny Canal",           59.9100, 30.3200, "industrial"),
            ("Kirov Plant Area",        59.8800, 30.2600, "industrial"),
            # Natural
            ("Summer Garden",           59.9450, 30.3350, "natural"),
            ("Krestovsky Island",       59.9700, 30.2500, "natural"),
        ],
    },

    "moscow": {
        "name": "Moscow",
        "country": "Russia",
        "bbox": (55.700, 37.500, 55.800, 37.700),
        "seeds": [
            # Permanent iconic anchors
            ("Tverskaya Street",        55.7650, 37.6050, "commercial"),
            ("Arbat",                   55.7510, 37.5940, "commercial"),
            # Residential
            ("Khamovniki",              55.7300, 37.5800, "residential"),
            ("Zamoskvorechye",          55.7350, 37.6250, "residential"),
            ("Basmanny",                55.7640, 37.6600, "residential"),
            # Mixed
            ("Kitay-Gorod",             55.7560, 37.6300, "mixed"),
            ("Patriarch Ponds",         55.7640, 37.5930, "mixed"),
            # Industrial
            ("Krasnoselsky",            55.7800, 37.6600, "industrial"),
            ("Nizhegorodsky",           55.7300, 37.6800, "industrial"),
            # Natural
            ("Gorky Park",              55.7310, 37.6010, "natural"),
            ("Sparrow Hills",           55.7100, 37.5550, "natural"),
        ],
    },

    # ─────────────────────────────────────────────
    # ATLANTIC IBERIA (Hillside + post-earthquake grid)
    # ─────────────────────────────────────────────

    "lisbon": {
        "name": "Lisbon",
        "country": "Portugal",
        "bbox": (38.700, -9.200, 38.760, -9.100),
        "seeds": [
            # Permanent iconic anchors
            ("Baixa-Chiado",            38.7100, -9.1400, "commercial"),
            ("Avenida da Liberdade",    38.7200, -9.1470, "commercial"),
            # Residential
            ("Alfama",                  38.7110, -9.1300, "residential"),
            ("Graca",                   38.7170, -9.1320, "residential"),
            ("Campo de Ourique",        38.7180, -9.1620, "residential"),
            # Mixed
            ("Bairro Alto",             38.7140, -9.1450, "mixed"),
            ("Principe Real",           38.7180, -9.1500, "mixed"),
            # Industrial
            ("Marvila",                 38.7400, -9.1050, "industrial"),
            ("Alcantara",               38.7050, -9.1750, "industrial"),
            # Natural
            ("Parque das Nacoes",       38.7630, -9.0950, "natural"),
            ("Jardim da Estrela",       38.7140, -9.1570, "natural"),
        ],
    },

    # ─────────────────────────────────────────────
    # EASTERN MEDITERRANEAN (Divided capital)
    # ─────────────────────────────────────────────

    "nicosia": {
        "name": "Nicosia",
        "country": "Cyprus",
        "bbox": (35.140, 33.330, 35.190, 33.390),
        "seeds": [
            # Permanent iconic anchors
            ("Ledra Street",            35.1700, 33.3620, "commercial"),
            ("Makariou Avenue",         35.1640, 33.3600, "commercial"),
            # Residential
            ("Strovolos",               35.1550, 33.3500, "residential"),
            ("Engomi",                  35.1650, 33.3400, "residential"),
            ("Aglantzia",               35.1600, 33.3800, "residential"),
            # Mixed
            ("Old Town Laiki Geitonia", 35.1720, 33.3640, "mixed"),
            ("Pallouriotissa",          35.1750, 33.3700, "mixed"),
            # Industrial
            ("Industrial Zone West",    35.1500, 33.3350, "industrial"),
            ("Latsia",                  35.1450, 33.3650, "industrial"),
            # Natural
            ("Pedieos River Park",      35.1580, 33.3750, "natural"),
            ("Athalassa Park",          35.1470, 33.3800, "natural"),
        ],
    },

    # ─────────────────────────────────────────────
    # NORTH ATLANTIC (Subarctic volcanic island)
    # ─────────────────────────────────────────────

    "reykjavik": {
        "name": "Reykjavik",
        "country": "Iceland",
        "bbox": (64.120, -21.980, 64.160, -21.850),
        "seeds": [
            # Permanent iconic anchors
            ("Laugavegur",              64.1437, -21.9250, "commercial"),
            ("Harpa Area",              64.1505, -21.9330, "commercial"),
            # Residential
            ("Vesturbær",               64.1480, -21.9500, "residential"),
            ("Hlíðar",                  64.1380, -21.9100, "residential"),
            ("Breiðholt",               64.1200, -21.8600, "residential"),
            # Mixed
            ("Grandi Harbour",          64.1530, -21.9500, "mixed"),
            ("Skolavordustigur",        64.1440, -21.9290, "mixed"),
            # Industrial
            ("Sundagarðar",             64.1550, -21.9600, "industrial"),
            ("Holtagarðar",             64.1300, -21.8900, "industrial"),
            # Natural
            ("Tjörnin Lake",            64.1440, -21.9370, "natural"),
            ("Laugardalur Park",        64.1440, -21.8970, "natural"),
        ],
    },

    # ─────────────────────────────────────────────
    # BALTIC (Medieval Hanseatic + Soviet layers)
    # ─────────────────────────────────────────────

    "tallinn": {
        "name": "Tallinn",
        "country": "Estonia",
        "bbox": (59.410, 24.700, 59.460, 24.800),
        "seeds": [
            # Permanent iconic anchors
            ("Viru Street",             59.4370, 24.7530, "commercial"),
            ("Rotermann Quarter",       59.4390, 24.7580, "commercial"),
            # Residential
            ("Kalamaja",                59.4440, 24.7350, "residential"),
            ("Kristiine",               59.4250, 24.7200, "residential"),
            ("Kadriorg",                59.4400, 24.7850, "residential"),
            # Mixed
            ("Old Town",                59.4370, 24.7450, "mixed"),
            ("Telliskivi",              59.4400, 24.7300, "mixed"),
            # Industrial
            ("Kopli",                   59.4520, 24.7100, "industrial"),
            ("Ülemiste",                59.4200, 24.7800, "industrial"),
            # Natural
            ("Kadriorg Park",           59.4390, 24.7900, "natural"),
            ("Pirita Beach",            59.4600, 24.8100, "natural"),
        ],
    },

    # ─────────────────────────────────────────────
    # SOUTH AMERICA (European-grid Southern Cone)
    # ─────────────────────────────────────────────

    "buenos_aires": {
        "name": "Buenos Aires",
        "country": "Argentina",
        "bbox": (-34.640, -58.450, -34.560, -58.340),
        "seeds": [
            # Permanent iconic anchors
            ("Florida Street",         -34.6037, -58.3816, "commercial"),
            ("Avenida Corrientes",     -34.6040, -58.3920, "commercial"),
            # Residential
            ("Palermo",                -34.5800, -58.4250, "residential"),
            ("Belgrano",               -34.5620, -58.4530, "residential"),
            ("Caballito",              -34.6180, -58.4300, "residential"),
            # Mixed
            ("San Telmo",              -34.6210, -58.3730, "mixed"),
            ("Recoleta",               -34.5880, -58.3930, "mixed"),
            # Industrial
            ("La Boca",                -34.6350, -58.3630, "industrial"),
            ("Barracas",               -34.6400, -58.3830, "industrial"),
            # Natural
            ("Bosques de Palermo",     -34.5720, -58.4180, "natural"),
            ("Reserva Ecologica",      -34.6150, -58.3530, "natural"),
        ],
    },

    # ─────────────────────────────────────────────
    # BENELUX (Art Nouveau + EU quarter)
    # ─────────────────────────────────────────────

    "brussels": {
        "name": "Brussels",
        "country": "Belgium",
        "bbox": (50.820, 4.320, 50.870, 4.410),
        "seeds": [
            # Permanent iconic anchors
            ("Grand Place",             50.8467, 4.3525, "commercial"),
            ("Avenue Louise",           50.8310, 4.3560, "commercial"),
            # Residential
            ("Ixelles",                 50.8280, 4.3680, "residential"),
            ("Schaerbeek",              50.8620, 4.3750, "residential"),
            ("Uccle",                   50.8100, 4.3400, "residential"),
            # Mixed
            ("Saint-Gilles",            50.8300, 4.3460, "mixed"),
            ("Sainte-Catherine",        50.8510, 4.3460, "mixed"),
            # Industrial
            ("Canal Zone Molenbeek",    50.8550, 4.3300, "industrial"),
            ("Anderlecht Abattoir",     50.8430, 4.3300, "industrial"),
            # Natural
            ("Bois de la Cambre",       50.8130, 4.3700, "natural"),
            ("Parc du Cinquantenaire",  50.8400, 4.3930, "natural"),
        ],
    },

    # ─────────────────────────────────────────────
    # CENTRAL EUROPE (Imperial Habsburg capital)
    # ─────────────────────────────────────────────

    "vienna": {
        "name": "Vienna",
        "country": "Austria",
        "bbox": (48.170, 16.330, 48.240, 16.420),
        "seeds": [
            # Permanent iconic anchors
            ("Stephansplatz",           48.2085, 16.3735, "commercial"),
            ("Mariahilfer Strasse",     48.1960, 16.3500, "commercial"),
            # Residential
            ("Leopoldstadt",            48.2200, 16.3900, "residential"),
            ("Ottakring",               48.2150, 16.3200, "residential"),
            ("Favoriten",               48.1750, 16.3750, "residential"),
            # Mixed
            ("Neubau",                  48.2010, 16.3500, "mixed"),
            ("Josefstadt",              48.2100, 16.3470, "mixed"),
            # Industrial
            ("Simmering",               48.1800, 16.4200, "industrial"),
            ("Erdberg",                 48.1900, 16.4000, "industrial"),
            # Natural
            ("Prater",                  48.2100, 16.4050, "natural"),
            ("Donauinsel",              48.2250, 16.3950, "natural"),
        ],
    },

    # ─────────────────────────────────────────────
    # PANNONIAN (River-split dual city)
    # ─────────────────────────────────────────────

    "budapest": {
        "name": "Budapest",
        "country": "Hungary",
        "bbox": (47.460, 19.020, 47.530, 19.120),
        "seeds": [
            # Permanent iconic anchors
            ("Vaci Utca",               47.4930, 19.0540, "commercial"),
            ("Andrassy Ut",             47.5020, 19.0650, "commercial"),
            # Residential
            ("Buda Hills",              47.5100, 19.0200, "residential"),
            ("Ujlipotvaros",            47.5150, 19.0600, "residential"),
            ("Ferencvaros",             47.4800, 19.0700, "residential"),
            # Mixed
            ("Erzsebetvaros",           47.4990, 19.0680, "mixed"),
            ("Belvaros",                47.4920, 19.0570, "mixed"),
            # Industrial
            ("Csepel",                  47.4300, 19.0800, "industrial"),
            ("Kobanya",                 47.4850, 19.1100, "industrial"),
            # Natural
            ("Margaret Island",         47.5200, 19.0500, "natural"),
            ("Gellert Hill",            47.4860, 19.0400, "natural"),
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
