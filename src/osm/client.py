"""OpenStreetMap API v0.6 client for geographic area description."""

from typing import Tuple, Dict, Any, List, Optional
import os
import logging
from collections import Counter

import requests
import xml.etree.ElementTree as ET


Coordinates = Tuple[float, float]


class OSMClientError(Exception):
    """High-level error for OSM client failures."""
    pass


class OSMClient:
    """Client for OpenStreetMap API v0.6 that describes geographic areas."""

    LANDMARK_TAGS = {
        'amenity', 'tourism', 'shop', 'historic', 'building',
        'leisure', 'landuse', 'office', 'public_transport',
        'railway', 'aeroway', 'man_made'
    }

    ORG_TAGS = {'operator', 'owner', 'brand', 'network'}

    def __init__(
        self,
        base_url: Optional[str] = None,
        session: Optional[requests.Session] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize OSM client.

        Args:
            base_url: OSM API base URL. Defaults to OSM_API_BASE_URL env var,
                     or "https://api.openstreetmap.org" if not set.
            session: Optional requests.Session instance for HTTP calls.
            logger: Optional logger instance.
        """
        if base_url is None:
            base_url = os.getenv('OSM_API_BASE_URL', 'https://api.openstreetmap.org')

        self.base_url = base_url
        self._session = session
        self.logger = logger or logging.getLogger(__name__)

    @property
    def session(self) -> requests.Session:
        """Lazy-initialize session if not provided."""
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def _normalize_bbox(
        self, top_left: Coordinates, bottom_right: Coordinates
    ) -> Dict[str, float]:
        """
        Normalize coordinates to OSM bounding box format.

        Args:
            top_left: (lat, lon) for north-west corner
            bottom_right: (lat, lon) for south-east corner

        Returns:
            Dict with keys: left, bottom, right, top (all floats)

        Raises:
            ValueError: If bbox is invalid (zero area or negative)
        """
        lat_top, lon_left = top_left
        lat_bottom, lon_right = bottom_right

        left = min(lon_left, lon_right)
        right = max(lon_left, lon_right)
        top = max(lat_top, lat_bottom)
        bottom = min(lat_top, lat_bottom)

        if left == right:
            raise ValueError("Bounding box has zero width")
        if top == bottom:
            raise ValueError("Bounding box has zero height")

        area = (right - left) * (top - bottom)
        if area <= 0:
            raise ValueError(f"Bounding box has invalid area: {area}")

        if area > 0.25:
            self.logger.warning(
                f"Large bounding box area: {area:.4f} square degrees. "
                "OSM API may reject this request."
            )

        return {
            'left': left,
            'bottom': bottom,
            'right': right,
            'top': top
        }

    def _fetch_map_xml(self, bbox: Dict[str, float]) -> str:
        """
        Fetch OSM map data via API.

        Args:
            bbox: Normalized bounding box dict

        Returns:
            Raw XML text from OSM API

        Raises:
            OSMClientError: On HTTP errors or empty response
        """
        url = f"{self.base_url.rstrip('/')}/api/0.6/map"
        bbox_param = f"{bbox['left']},{bbox['bottom']},{bbox['right']},{bbox['top']}"
        params = {'bbox': bbox_param}

        headers = {'User-Agent': 'my-osm-client/0.1'}
        api_key = os.getenv('OSM_API_KEY')
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        self.logger.debug(f"Fetching OSM data: {url}?bbox={bbox_param}")

        try:
            response = self.session.get(url, params=params, headers=headers, timeout=30)
        except requests.RequestException as e:
            raise OSMClientError(f"HTTP request failed: {e}")

        if response.status_code != 200:
            body_snippet = response.text[:200] if response.text else '<empty>'
            self.logger.error(
                f"OSM API returned status {response.status_code}. "
                f"Body: {body_snippet}"
            )
            raise OSMClientError(
                f"OSM API error: HTTP {response.status_code}"
            )

        if not response.text or not response.text.strip():
            raise OSMClientError("OSM API returned empty response")

        return response.text

    def _extract_tags(self, element: ET.Element) -> Dict[str, str]:
        """Extract all tags from an OSM element."""
        tags = {}
        for tag in element.findall('tag'):
            k = tag.get('k')
            v = tag.get('v')
            if k:
                tags[k] = v or ''
        return tags

    def _is_landmark(self, tags: Dict[str, str]) -> bool:
        """Determine if an element is a landmark based on tags."""
        if 'name' in tags:
            return True
        return bool(self.LANDMARK_TAGS & set(tags.keys()))

    def _determine_primary_type(self, tags: Dict[str, str]) -> str:
        """Determine primary type classification for a landmark."""
        if 'amenity' in tags:
            return f"amenity:{tags['amenity']}"
        if 'tourism' in tags:
            return f"tourism:{tags['tourism']}"
        if 'shop' in tags:
            return f"shop:{tags['shop']}"
        if 'landuse' in tags:
            return f"landuse:{tags['landuse']}"
        if 'building' in tags:
            val = tags['building']
            return f"building:{val}" if val else "building:yes"

        for key in ['leisure', 'historic', 'office', 'public_transport',
                    'railway', 'aeroway', 'man_made']:
            if key in tags:
                return f"{key}:{tags[key]}"

        return "other"

    def _extract_org_tags(self, tags: Dict[str, str]) -> Optional[Dict[str, Optional[str]]]:
        """Extract organization-related tags if present."""
        org_tags = {
            'operator': tags.get('operator'),
            'owner': tags.get('owner'),
            'brand': tags.get('brand'),
            'network': tags.get('network')
        }
        if any(v is not None for v in org_tags.values()):
            return org_tags
        return None

    def _parse_osm_xml(self, xml_text: str) -> Dict[str, Any]:
        """
        Parse OSM XML and extract structured data.

        Args:
            xml_text: Raw OSM XML response

        Returns:
            Dict containing parsed nodes, ways, relations
        """
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            raise OSMClientError(f"Failed to parse OSM XML: {e}")

        nodes_data = {}
        ways_data = []
        relations_data = []

        for node in root.findall('node'):
            node_id = node.get('id')
            lat = node.get('lat')
            lon = node.get('lon')

            if node_id and lat and lon:
                tags = self._extract_tags(node)
                nodes_data[int(node_id)] = {
                    'lat': float(lat),
                    'lon': float(lon),
                    'tags': tags
                }

        for way in root.findall('way'):
            way_id = way.get('id')
            if way_id:
                tags = self._extract_tags(way)
                node_refs = [int(nd.get('ref')) for nd in way.findall('nd')
                            if nd.get('ref')]
                ways_data.append({
                    'id': int(way_id),
                    'tags': tags,
                    'node_refs': node_refs
                })

        for relation in root.findall('relation'):
            rel_id = relation.get('id')
            if rel_id:
                tags = self._extract_tags(relation)
                relations_data.append({
                    'id': int(rel_id),
                    'tags': tags
                })

        return {
            'nodes': nodes_data,
            'ways': ways_data,
            'relations': relations_data
        }

    def _build_landmarks(
        self,
        nodes_data: Dict[int, Dict],
        ways_data: List[Dict],
        relations_data: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Build landmarks list from parsed OSM data."""
        landmarks = []

        for node_id, node_info in nodes_data.items():
            tags = node_info['tags']
            if self._is_landmark(tags):
                landmarks.append({
                    'id': node_id,
                    'osm_type': 'node',
                    'name': tags.get('name'),
                    'primary_type': self._determine_primary_type(tags),
                    'raw_tags': tags,
                    'lat': node_info['lat'],
                    'lon': node_info['lon']
                })

        for way in ways_data:
            tags = way['tags']
            if self._is_landmark(tags):
                lat, lon = None, None
                if way['node_refs']:
                    valid_coords = [
                        (nodes_data[ref]['lat'], nodes_data[ref]['lon'])
                        for ref in way['node_refs']
                        if ref in nodes_data
                    ]
                    if valid_coords:
                        lat = sum(c[0] for c in valid_coords) / len(valid_coords)
                        lon = sum(c[1] for c in valid_coords) / len(valid_coords)

                landmarks.append({
                    'id': way['id'],
                    'osm_type': 'way',
                    'name': tags.get('name'),
                    'primary_type': self._determine_primary_type(tags),
                    'raw_tags': tags,
                    'lat': lat,
                    'lon': lon
                })

        for relation in relations_data:
            tags = relation['tags']
            if self._is_landmark(tags):
                landmarks.append({
                    'id': relation['id'],
                    'osm_type': 'relation',
                    'name': tags.get('name'),
                    'primary_type': self._determine_primary_type(tags),
                    'raw_tags': tags,
                    'lat': None,
                    'lon': None
                })

        return landmarks

    def _build_organizations(self, landmarks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract organization information from landmarks."""
        organizations = []

        for landmark in landmarks:
            org_tags = self._extract_org_tags(landmark['raw_tags'])
            if org_tags:
                organizations.append({
                    'id': landmark['id'],
                    'osm_type': landmark['osm_type'],
                    'name': landmark['name'],
                    'org_tags': org_tags
                })

        return organizations

    def _analyze_buildings(self, landmarks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze building statistics and types."""
        buildings = [lm for lm in landmarks if lm['primary_type'].startswith('building:')]

        if not buildings:
            return {
                'total_count': 0,
                'types': {},
                'residential_count': 0,
                'commercial_count': 0,
                'industrial_count': 0,
                'other_count': 0
            }

        # Categorize building types
        residential_types = {'house', 'apartments', 'residential', 'detached', 'terrace',
                           'dormitory', 'semidetached_house', 'bungalow'}
        commercial_types = {'commercial', 'retail', 'office', 'supermarket', 'shop',
                          'mall', 'kiosk', 'hotel'}
        industrial_types = {'industrial', 'warehouse', 'factory', 'manufacture'}

        residential_count = 0
        commercial_count = 0
        industrial_count = 0
        other_count = 0
        building_types = Counter()

        for building in buildings:
            building_type = building['primary_type'].replace('building:', '')
            building_types[building_type] += 1

            if building_type in residential_types:
                residential_count += 1
            elif building_type in commercial_types:
                commercial_count += 1
            elif building_type in industrial_types:
                industrial_count += 1
            else:
                other_count += 1

        return {
            'total_count': len(buildings),
            'types': dict(building_types.most_common(10)),
            'residential_count': residential_count,
            'commercial_count': commercial_count,
            'industrial_count': industrial_count,
            'other_count': other_count
        }

    def _analyze_amenities(self, landmarks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze amenity distribution."""
        amenities = [lm for lm in landmarks if lm['primary_type'].startswith('amenity:')]

        if not amenities:
            return {'total_count': 0, 'types': {}}

        amenity_types = Counter(
            lm['primary_type'].replace('amenity:', '')
            for lm in amenities
        )

        return {
            'total_count': len(amenities),
            'types': dict(amenity_types.most_common(10))
        }

    def _classify_area_type(self, building_stats: Dict[str, Any],
                           amenity_stats: Dict[str, Any],
                           landmarks: List[Dict[str, Any]]) -> str:
        """Classify the overall area type based on landmark composition."""
        total_buildings = building_stats['total_count']

        if total_buildings == 0:
            return 'undeveloped or natural'

        residential = building_stats['residential_count']
        commercial = building_stats['commercial_count']
        industrial = building_stats['industrial_count']

        # Calculate ratios
        total_categorized = residential + commercial + industrial
        if total_categorized == 0:
            return 'mixed urban'

        res_ratio = residential / total_categorized
        com_ratio = commercial / total_categorized
        ind_ratio = industrial / total_categorized

        # Classification logic
        if ind_ratio > 0.4:
            return 'industrial'
        elif res_ratio > 0.6:
            if amenity_stats['total_count'] > total_buildings * 0.1:
                return 'residential with amenities'
            return 'predominantly residential'
        elif com_ratio > 0.4:
            return 'commercial district'
        elif res_ratio > 0.3 and com_ratio > 0.2:
            return 'mixed-use urban'
        else:
            return 'mixed urban'

    def _build_summary(
        self,
        total_elements: int,
        landmarks: List[Dict[str, Any]],
        type_dist: Dict[str, int],
        bbox_area: float
    ) -> Dict[str, Any]:
        """Build comprehensive summary statistics."""
        top_types = [
            {'type': t, 'count': c}
            for t, c in sorted(type_dist.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

        # Analyze buildings
        building_stats = self._analyze_buildings(landmarks)

        # Analyze amenities
        amenity_stats = self._analyze_amenities(landmarks)

        # Calculate building density (buildings per km²)
        # Convert square degrees to approximate km² (at mid-latitudes, 1° ≈ 111km)
        area_km2 = bbox_area * (111.0 ** 2)
        building_density = building_stats['total_count'] / area_km2 if area_km2 > 0 else 0

        # Classify area type
        area_type = self._classify_area_type(building_stats, amenity_stats, landmarks)

        # Generate notes
        if top_types:
            top_two = top_types[:2]
            if len(top_two) == 1:
                notes = f"Area dominated by {top_two[0]['type']} landmarks."
            else:
                notes = f"Area dominated by {top_two[0]['type']} and {top_two[1]['type']} landmarks."
        else:
            notes = "No significant landmarks detected in this area."

        # Commercial vs residential ratio
        total_categorized = (building_stats['residential_count'] +
                           building_stats['commercial_count'])
        if total_categorized > 0:
            commercial_ratio = building_stats['commercial_count'] / total_categorized
            residential_ratio = building_stats['residential_count'] / total_categorized
        else:
            commercial_ratio = 0.0
            residential_ratio = 0.0

        return {
            'total_elements': total_elements,
            'total_landmarks': len(landmarks),
            'distinct_types': len(type_dist),
            'top_types': top_types,
            'buildings': {
                'total': building_stats['total_count'],
                'density_per_km2': round(building_density, 2),
                'types': building_stats['types'],
                'residential': building_stats['residential_count'],
                'commercial': building_stats['commercial_count'],
                'industrial': building_stats['industrial_count'],
                'other': building_stats['other_count'],
                'commercial_ratio': round(commercial_ratio, 3),
                'residential_ratio': round(residential_ratio, 3)
            },
            'amenities': {
                'total': amenity_stats['total_count'],
                'types': amenity_stats['types']
            },
            'area_classification': area_type,
            'notes': notes
        }

    def describe(self, top_left: Coordinates, bottom_right: Coordinates) -> Dict[str, Any]:
        """
        Describe a geographic area using OpenStreetMap data.

        Args:
            top_left: (lat, lon) for north-west corner
            bottom_right: (lat, lon) for south-east corner

        Returns:
            Dict containing:
                - bbox: Normalized bounding box with area
                - landmarks: List of identified landmarks
                - type_distribution: Count of landmarks by type
                - organizations: List of organization-related features
                - summary: Aggregate statistics and notes

        Raises:
            ValueError: If coordinates are invalid
            OSMClientError: If API call or parsing fails
        """
        bbox = self._normalize_bbox(top_left, bottom_right)
        xml_text = self._fetch_map_xml(bbox)
        parsed = self._parse_osm_xml(xml_text)

        nodes_data = parsed['nodes']
        ways_data = parsed['ways']
        relations_data = parsed['relations']

        total_elements = len(nodes_data) + len(ways_data) + len(relations_data)

        landmarks = self._build_landmarks(nodes_data, ways_data, relations_data)

        type_counter = Counter(lm['primary_type'] for lm in landmarks)
        type_distribution = dict(type_counter)

        organizations = self._build_organizations(landmarks)

        area = (bbox['right'] - bbox['left']) * (bbox['top'] - bbox['bottom'])

        summary = self._build_summary(total_elements, landmarks, type_distribution, area)

        return {
            'bbox': {
                'left': bbox['left'],
                'bottom': bbox['bottom'],
                'right': bbox['right'],
                'top': bbox['top'],
                'area_degrees2': area
            },
            'landmarks': landmarks,
            'type_distribution': type_distribution,
            'organizations': organizations,
            'summary': summary
        }
