"""
OmniGeoFusion — API Tests
============================
Integration tests for FastAPI endpoints.
Uses TestClient (no real model loading needed).
"""

import pytest
import torch
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client with mocked models."""
    # Mock backbone
    mock_backbone = MagicMock()
    mock_backbone.return_value = torch.randn(1, 512)
    mock_backbone.parameters.return_value = iter([
        torch.randn(10, 10)
    ])

    # Mock task heads
    mock_urban_head = MagicMock()
    mock_urban_head.return_value = {
        'change_score': torch.tensor([0.35]),
        'change_type':  torch.randn(1, 6),
    }

    mock_flood_head = MagicMock()
    mock_flood_head.return_value = {
        'flood_probability': torch.tensor([0.45]),
        'flood_depth':       torch.tensor([0.8]),
        'damage_level':      torch.randn(1, 5),
        'road_blocked':      torch.tensor([0.2]),
    }

    mock_agri_head = MagicMock()
    mock_agri_head.return_value = {
        'ndvi':          torch.tensor([0.65]),
        'lai':           torch.tensor([2.5]),
        'crop_height_m': torch.tensor([0.8]),
        'soil_moisture': torch.tensor([0.35]),
        'cwsi':          torch.tensor([0.3]),
        'et_mm_day':     torch.tensor([4.2]),
        'stress_type':   torch.randn(1, 4),
    }

    with patch.dict('src.api.router.MODELS', {
        'backbone':           mock_backbone,
        'urban_change_head':  mock_urban_head,
        'flood_damage_head':  mock_flood_head,
        'agriculture_head':   mock_agri_head,
    }):
        from src.api.main import app
        yield TestClient(app)


# ── Health Check ──────────────────────────────────────────────
class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        resp = client.get('/health')
        assert resp.status_code == 200

    def test_health_response_schema(self, client):
        resp = client.get('/health')
        data = resp.json()
        assert 'status'  in data
        assert 'version' in data
        assert 'device'  in data


# ── Model Info ────────────────────────────────────────────────
class TestModelInfoEndpoint:

    def test_model_info_returns_200(self, client):
        resp = client.get('/api/v1/model/info')
        assert resp.status_code == 200

    def test_model_info_schema(self, client):
        resp = client.get('/api/v1/model/info')
        data = resp.json()
        assert 'name'       in data
        assert 'version'    in data
        assert 'modalities' in data
        assert 'tasks'      in data
        assert len(data['modalities']) == 6
        assert len(data['tasks'])      == 3


# ── Urban Change Endpoint ─────────────────────────────────────
class TestUrbanChangeEndpoint:

    VALID_PAYLOAD = {
        'bbox': {
            'min_lon': 4.85, 'min_lat': 52.35,
            'max_lon': 4.95, 'max_lat': 52.40,
        },
        'date_t1':    '2020-07-07',
        'date_t2':    '2023-07-07',
        'modalities': ['sentinel2', 'sentinel1', 'lidar'],
    }

    def test_returns_200(self, client):
        resp = client.post(
            '/api/v1/fusion/urban-change',
            json=self.VALID_PAYLOAD
        )
        assert resp.status_code == 200

    def test_response_schema(self, client):
        resp = client.post(
            '/api/v1/fusion/urban-change',
            json=self.VALID_PAYLOAD
        )
        data = resp.json()
        assert 'change_score'    in data
        assert 'change_type'     in data
        assert 'change_detected' in data
        assert 'day_gap'         in data
        assert 'geometry'        in data

    def test_change_score_range(self, client):
        resp = client.post(
            '/api/v1/fusion/urban-change',
            json=self.VALID_PAYLOAD
        )
        score = resp.json()['change_score']
        assert 0.0 <= score <= 1.0

    def test_bbox_too_large_returns_400(self, client):
        payload = self.VALID_PAYLOAD.copy()
        payload['bbox'] = {
            'min_lon': 0.0, 'min_lat': 0.0,
            'max_lon': 10.0, 'max_lat': 10.0,  # > 100km²
        }
        resp = client.post(
            '/api/v1/fusion/urban-change', json=payload
        )
        assert resp.status_code == 400

    def test_invalid_bbox_returns_422(self, client):
        payload = self.VALID_PAYLOAD.copy()
        payload['bbox'] = {
            'min_lon': 5.0, 'min_lat': 52.0,
            'max_lon': 4.0,  # max < min → invalid
            'max_lat': 53.0,
        }
        resp = client.post(
            '/api/v1/fusion/urban-change', json=payload
        )
        assert resp.status_code == 422


# ── Flood Damage Endpoint ─────────────────────────────────────
class TestFloodDamageEndpoint:

    VALID_PAYLOAD = {
        'bbox': {
            'min_lon': 4.40, 'min_lat': 51.88,
            'max_lon': 4.55, 'max_lat': 51.95,
        },
        'disaster_type': 'flood',
        'date_pre':      '2023-06-01',
        'date_post':     '2023-06-15',
        'modalities':    ['sentinel2', 'sentinel1', 'iot'],
    }

    def test_returns_200(self, client):
        resp = client.post(
            '/api/v1/fusion/flood-damage',
            json=self.VALID_PAYLOAD
        )
        assert resp.status_code == 200

    def test_response_schema(self, client):
        resp = client.post(
            '/api/v1/fusion/flood-damage',
            json=self.VALID_PAYLOAD
        )
        data = resp.json()
        assert 'flood_detected'        in data
        assert 'flood_fraction'        in data
        assert 'mean_flood_depth_m'    in data
        assert 'dominant_damage_level' in data
        assert 'road_accessible_pct'   in data

    def test_flood_fraction_range(self, client):
        resp = client.post(
            '/api/v1/fusion/flood-damage',
            json=self.VALID_PAYLOAD
        )
        frac = resp.json()['flood_fraction']
        assert 0.0 <= frac <= 1.0


# ── Agriculture Endpoint ──────────────────────────────────────
class TestAgricultureEndpoint:

    VALID_PAYLOAD = {
        'bbox': {
            'min_lon': 5.30, 'min_lat': 52.40,
            'max_lon': 5.36, 'max_lat': 52.46,
        },
        'date':      '2023-07-15',
        'crop_type': 'wheat',
        'modalities': [
            'sentinel2', 'sentinel1',
            'lidar', 'thermal', 'iot'
        ],
    }

    def test_returns_200(self, client):
        resp = client.post(
            '/api/v1/fusion/agriculture',
            json=self.VALID_PAYLOAD
        )
        assert resp.status_code == 200

    def test_response_schema(self, client):
        resp = client.post(
            '/api/v1/fusion/agriculture',
            json=self.VALID_PAYLOAD
        )
        data = resp.json()
        assert 'ndvi'                   in data
        assert 'lai'                    in data
        assert 'crop_height_m'          in data
        assert 'soil_moisture'          in data
        assert 'cwsi'                   in data
        assert 'et_mm_day'              in data
        assert 'stress_type'            in data
        assert 'irrigation_recommended' in data

    def test_ndvi_range(self, client):
        resp = client.post(
            '/api/v1/fusion/agriculture',
            json=self.VALID_PAYLOAD
        )
        ndvi = resp.json()['ndvi']
        assert -1.0 <= ndvi <= 1.0

    def test_irrigation_flag_type(self, client):
        resp = client.post(
            '/api/v1/fusion/agriculture',
            json=self.VALID_PAYLOAD
        )
        flag = resp.json()['irrigation_recommended']
        assert isinstance(flag, bool)
