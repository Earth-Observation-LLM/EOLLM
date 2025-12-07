"""Tests for OSM client module."""

import pytest
import requests_mock
from src.osm import OSMClient, OSMClientError, Coordinates


SAMPLE_OSM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="Test">
  <node id="1" lat="40.7580" lon="29.9510" version="1">
    <tag k="name" v="Test Restaurant"/>
    <tag k="amenity" v="restaurant"/>
    <tag k="operator" v="Test Corp"/>
  </node>
  <node id="2" lat="40.7585" lon="29.9515" version="1">
    <tag k="name" v="Test Supermarket"/>
    <tag k="shop" v="supermarket"/>
    <tag k="brand" v="TestMart"/>
  </node>
  <node id="3" lat="40.7590" lon="29.9520" version="1">
    <tag k="highway" v="traffic_signals"/>
  </node>
  <way id="100" version="1">
    <nd ref="1"/>
    <nd ref="2"/>
    <tag k="name" v="Main Street"/>
    <tag k="highway" v="residential"/>
  </way>
  <way id="101" version="1">
    <nd ref="1"/>
    <nd ref="3"/>
    <tag k="building" v="commercial"/>
    <tag k="name" v="Office Building"/>
    <tag k="owner" v="Test Owner LLC"/>
  </way>
</osm>
"""


class TestOSMClient:
    """Unit tests for OSMClient with mocked HTTP."""

    def test_init_default_base_url(self, monkeypatch):
        """Test initialization with default base URL."""
        monkeypatch.delenv('OSM_API_BASE_URL', raising=False)
        client = OSMClient()
        assert client.base_url == 'https://api.openstreetmap.org'

    def test_init_custom_base_url(self):
        """Test initialization with custom base URL."""
        custom_url = 'https://custom.osm.example.com'
        client = OSMClient(base_url=custom_url)
        assert client.base_url == custom_url

    def test_init_env_base_url(self, monkeypatch):
        """Test initialization from environment variable."""
        env_url = 'https://env.osm.example.com'
        monkeypatch.setenv('OSM_API_BASE_URL', env_url)
        client = OSMClient()
        assert client.base_url == env_url

    def test_normalize_bbox_valid(self):
        """Test bounding box normalization with valid coordinates."""
        client = OSMClient()
        top_left = (40.76, 29.95)
        bottom_right = (40.75, 29.96)

        bbox = client._normalize_bbox(top_left, bottom_right)

        assert bbox['left'] == 29.95
        assert bbox['right'] == 29.96
        assert bbox['top'] == 40.76
        assert bbox['bottom'] == 40.75
        assert bbox['right'] > bbox['left']
        assert bbox['top'] > bbox['bottom']

    def test_normalize_bbox_swapped_coords(self):
        """Test bbox normalization handles swapped coordinates correctly."""
        client = OSMClient()
        top_left = (40.75, 29.96)
        bottom_right = (40.76, 29.95)

        bbox = client._normalize_bbox(top_left, bottom_right)

        assert bbox['left'] == 29.95
        assert bbox['right'] == 29.96
        assert bbox['top'] == 40.76
        assert bbox['bottom'] == 40.75

    def test_normalize_bbox_zero_width(self):
        """Test that zero-width bbox raises ValueError."""
        client = OSMClient()
        top_left = (40.76, 29.95)
        bottom_right = (40.75, 29.95)

        with pytest.raises(ValueError, match="zero width"):
            client._normalize_bbox(top_left, bottom_right)

    def test_normalize_bbox_zero_height(self):
        """Test that zero-height bbox raises ValueError."""
        client = OSMClient()
        top_left = (40.75, 29.95)
        bottom_right = (40.75, 29.96)

        with pytest.raises(ValueError, match="zero height"):
            client._normalize_bbox(top_left, bottom_right)

    def test_fetch_map_xml_success(self):
        """Test successful XML fetch."""
        client = OSMClient()

        with requests_mock.Mocker() as m:
            m.get(
                'https://api.openstreetmap.org/api/0.6/map',
                text=SAMPLE_OSM_XML
            )

            bbox = {'left': 29.95, 'bottom': 40.75, 'right': 29.96, 'top': 40.76}
            xml = client._fetch_map_xml(bbox)

            assert '<?xml version' in xml
            assert '<osm' in xml
            assert 'Test Restaurant' in xml

    def test_fetch_map_xml_http_error(self):
        """Test that HTTP errors raise OSMClientError."""
        client = OSMClient()

        with requests_mock.Mocker() as m:
            m.get(
                'https://api.openstreetmap.org/api/0.6/map',
                status_code=400,
                text='Bad Request'
            )

            bbox = {'left': 29.95, 'bottom': 40.75, 'right': 29.96, 'top': 40.76}

            with pytest.raises(OSMClientError, match='HTTP 400'):
                client._fetch_map_xml(bbox)

    def test_fetch_map_xml_empty_response(self):
        """Test that empty response raises OSMClientError."""
        client = OSMClient()

        with requests_mock.Mocker() as m:
            m.get(
                'https://api.openstreetmap.org/api/0.6/map',
                text=''
            )

            bbox = {'left': 29.95, 'bottom': 40.75, 'right': 29.96, 'top': 40.76}

            with pytest.raises(OSMClientError, match='empty response'):
                client._fetch_map_xml(bbox)

    def test_describe_basic_structure(self):
        """Test describe returns correct structure with mocked data."""
        client = OSMClient()

        with requests_mock.Mocker() as m:
            m.get(
                'https://api.openstreetmap.org/api/0.6/map',
                text=SAMPLE_OSM_XML
            )

            top_left = (40.76, 29.95)
            bottom_right = (40.75, 29.96)

            result = client.describe(top_left, bottom_right)

            assert 'bbox' in result
            assert 'landmarks' in result
            assert 'type_distribution' in result
            assert 'organizations' in result
            assert 'summary' in result

    def test_describe_bbox_values(self):
        """Test describe bbox contains correct values."""
        client = OSMClient()

        with requests_mock.Mocker() as m:
            m.get(
                'https://api.openstreetmap.org/api/0.6/map',
                text=SAMPLE_OSM_XML
            )

            top_left = (40.76, 29.95)
            bottom_right = (40.75, 29.96)

            result = client.describe(top_left, bottom_right)

            bbox = result['bbox']
            assert bbox['left'] == 29.95
            assert bbox['right'] == 29.96
            assert bbox['top'] == 40.76
            assert bbox['bottom'] == 40.75
            assert bbox['area_degrees2'] == pytest.approx(0.01 * 0.01)

    def test_describe_landmarks_count(self):
        """Test correct number of landmarks identified."""
        client = OSMClient()

        with requests_mock.Mocker() as m:
            m.get(
                'https://api.openstreetmap.org/api/0.6/map',
                text=SAMPLE_OSM_XML
            )

            top_left = (40.76, 29.95)
            bottom_right = (40.75, 29.96)

            result = client.describe(top_left, bottom_right)

            landmarks = result['landmarks']
            assert len(landmarks) == 4

            names = [lm['name'] for lm in landmarks if lm['name']]
            assert 'Test Restaurant' in names
            assert 'Test Supermarket' in names
            assert 'Main Street' in names
            assert 'Office Building' in names

    def test_describe_landmark_types(self):
        """Test primary type classification."""
        client = OSMClient()

        with requests_mock.Mocker() as m:
            m.get(
                'https://api.openstreetmap.org/api/0.6/map',
                text=SAMPLE_OSM_XML
            )

            top_left = (40.76, 29.95)
            bottom_right = (40.75, 29.96)

            result = client.describe(top_left, bottom_right)

            landmarks_by_name = {lm['name']: lm for lm in result['landmarks'] if lm['name']}

            assert landmarks_by_name['Test Restaurant']['primary_type'] == 'amenity:restaurant'
            assert landmarks_by_name['Test Supermarket']['primary_type'] == 'shop:supermarket'
            assert landmarks_by_name['Office Building']['primary_type'] == 'building:commercial'

    def test_describe_type_distribution(self):
        """Test type distribution counting."""
        client = OSMClient()

        with requests_mock.Mocker() as m:
            m.get(
                'https://api.openstreetmap.org/api/0.6/map',
                text=SAMPLE_OSM_XML
            )

            top_left = (40.76, 29.95)
            bottom_right = (40.75, 29.96)

            result = client.describe(top_left, bottom_right)

            type_dist = result['type_distribution']
            assert type_dist['amenity:restaurant'] == 1
            assert type_dist['shop:supermarket'] == 1
            assert type_dist['building:commercial'] == 1
            assert type_dist['other'] == 1

    def test_describe_organizations(self):
        """Test organization extraction."""
        client = OSMClient()

        with requests_mock.Mocker() as m:
            m.get(
                'https://api.openstreetmap.org/api/0.6/map',
                text=SAMPLE_OSM_XML
            )

            top_left = (40.76, 29.95)
            bottom_right = (40.75, 29.96)

            result = client.describe(top_left, bottom_right)

            orgs = result['organizations']
            assert len(orgs) == 3

            org_ids = {org['id'] for org in orgs}
            assert 1 in org_ids
            assert 2 in org_ids
            assert 101 in org_ids

            org_by_id = {org['id']: org for org in orgs}
            assert org_by_id[1]['org_tags']['operator'] == 'Test Corp'
            assert org_by_id[2]['org_tags']['brand'] == 'TestMart'
            assert org_by_id[101]['org_tags']['owner'] == 'Test Owner LLC'

    def test_describe_summary(self):
        """Test summary statistics."""
        client = OSMClient()

        with requests_mock.Mocker() as m:
            m.get(
                'https://api.openstreetmap.org/api/0.6/map',
                text=SAMPLE_OSM_XML
            )

            top_left = (40.76, 29.95)
            bottom_right = (40.75, 29.96)

            result = client.describe(top_left, bottom_right)

            summary = result['summary']
            assert summary['total_elements'] == 5
            assert summary['total_landmarks'] == 4
            assert summary['distinct_types'] == 4
            assert len(summary['top_types']) > 0
            assert 'notes' in summary
            assert isinstance(summary['notes'], str)
            assert len(summary['notes']) > 0

            # Check new building statistics
            assert 'buildings' in summary
            assert 'amenities' in summary
            assert 'area_classification' in summary

            assert isinstance(summary['buildings'], dict)
            assert 'total' in summary['buildings']
            assert 'density_per_km2' in summary['buildings']
            assert 'residential' in summary['buildings']
            assert 'commercial' in summary['buildings']
            assert 'commercial_ratio' in summary['buildings']
            assert 'residential_ratio' in summary['buildings']

    def test_describe_landmark_coordinates(self):
        """Test that node landmarks have coordinates."""
        client = OSMClient()

        with requests_mock.Mocker() as m:
            m.get(
                'https://api.openstreetmap.org/api/0.6/map',
                text=SAMPLE_OSM_XML
            )

            top_left = (40.76, 29.95)
            bottom_right = (40.75, 29.96)

            result = client.describe(top_left, bottom_right)

            node_landmarks = [lm for lm in result['landmarks'] if lm['osm_type'] == 'node']

            for lm in node_landmarks:
                assert lm['lat'] is not None
                assert lm['lon'] is not None
                assert isinstance(lm['lat'], float)
                assert isinstance(lm['lon'], float)

    def test_parse_invalid_xml(self):
        """Test that invalid XML raises OSMClientError."""
        client = OSMClient()

        with requests_mock.Mocker() as m:
            m.get(
                'https://api.openstreetmap.org/api/0.6/map',
                text='<invalid><xml'
            )

            top_left = (40.76, 29.95)
            bottom_right = (40.75, 29.96)

            with pytest.raises(OSMClientError, match='parse'):
                client.describe(top_left, bottom_right)


@pytest.mark.integration
class TestOSMClientIntegration:
    """Integration tests using real OSM API (no mocking)."""

    def test_describe_real_api_small_bbox(self):
        """Test describe with actual OSM API call in Ankara, Turkey."""
        client = OSMClient()

        # Very small bbox in Ankara (approximately 2km x 3km area)
        top_left = (39.933765, 32.827074)
        bottom_right = (39.912350, 32.855643)

        # Make actual API call (no mocking)
        result = client.describe(top_left, bottom_right)

        # Verify structure
        assert 'bbox' in result
        assert 'landmarks' in result
        assert 'type_distribution' in result
        assert 'organizations' in result
        assert 'summary' in result

        # Verify bbox values
        assert result['bbox']['left'] == 32.827074
        assert result['bbox']['right'] == 32.855643
        assert result['bbox']['top'] == 39.933765
        assert result['bbox']['bottom'] == 39.912350

        # Verify types
        assert isinstance(result['landmarks'], list)
        assert isinstance(result['type_distribution'], dict)
        assert isinstance(result['organizations'], list)
        assert isinstance(result['summary'], dict)

        # Verify summary structure
        assert 'total_elements' in result['summary']
        assert 'total_landmarks' in result['summary']
        assert 'distinct_types' in result['summary']
        assert 'top_types' in result['summary']
        assert 'notes' in result['summary']

        # Verify summary types
        assert isinstance(result['summary']['total_elements'], int)
        assert isinstance(result['summary']['total_landmarks'], int)
        assert isinstance(result['summary']['distinct_types'], int)
        assert isinstance(result['summary']['top_types'], list)
        assert isinstance(result['summary']['notes'], str)

        # Verify we got actual data from OSM
        assert result['summary']['total_elements'] > 0, "Should have received actual OSM data"

        # Print summary for manual verification
        print(f"\n--- Real API Test Results ---")
        print(f"Total elements: {result['summary']['total_elements']}")
        print(f"Total landmarks: {result['summary']['total_landmarks']}")
        print(f"Distinct types: {result['summary']['distinct_types']}")
        print(f"Notes: {result['summary']['notes']}")
        if result['summary']['top_types']:
            print(f"Top landmark type: {result['summary']['top_types'][0]}")
