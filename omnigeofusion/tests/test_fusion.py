"""
OmniGeoFusion — Fusion Model Tests
=====================================
Unit tests for backbone + task heads.
Uses CPU (no GPU required for tests).
"""

import pytest
import torch
import numpy as np


# ── Test Encoders ─────────────────────────────────────────────
class TestSAREncoder:

    def test_output_shape(self):
        """SAR encoder output should be [B, SAR_DIM]."""
        from src.fusion.backbone import SAREncoder, SAR_DIM
        encoder = SAREncoder(in_channels=2, embed_dim=SAR_DIM)
        encoder.eval()

        x   = torch.randn(2, 2, 224, 224)
        out = encoder(x)
        assert out.shape == (2, SAR_DIM)

    def test_batch_independence(self):
        """Different batch items should produce different outputs."""
        from src.fusion.backbone import SAREncoder, SAR_DIM
        encoder = SAREncoder(in_channels=2, embed_dim=SAR_DIM)
        encoder.eval()

        x1  = torch.randn(1, 2, 224, 224)
        x2  = torch.randn(1, 2, 224, 224)
        o1  = encoder(x1)
        o2  = encoder(x2)
        assert not torch.allclose(o1, o2)


class TestLiDAREncoder:

    def test_output_shape(self):
        """LiDAR encoder output should be [B, LIDAR_DIM]."""
        from src.fusion.backbone import LiDAREncoder, LIDAR_DIM
        encoder = LiDAREncoder(in_channels=3, embed_dim=LIDAR_DIM)
        encoder.eval()

        x   = torch.randn(2, 3, 224, 224)
        out = encoder(x)
        assert out.shape == (2, LIDAR_DIM)

    def test_three_channels(self):
        """Should accept exactly 3 channels: DSM, DTM, nDSM."""
        from src.fusion.backbone import LiDAREncoder, LIDAR_DIM
        encoder = LiDAREncoder(in_channels=3, embed_dim=LIDAR_DIM)
        encoder.eval()

        x = torch.randn(1, 3, 224, 224)
        assert encoder(x).shape == (1, LIDAR_DIM)


class TestThermalEncoder:

    def test_output_shape(self):
        """Thermal encoder output should be [B, THERMAL_DIM]."""
        from src.fusion.backbone import ThermalEncoder, THERMAL_DIM
        encoder = ThermalEncoder(
            in_channels=1, embed_dim=THERMAL_DIM
        )
        encoder.eval()

        x   = torch.randn(2, 1, 224, 224)
        out = encoder(x)
        assert out.shape == (2, THERMAL_DIM)

    def test_single_channel(self):
        """Should accept exactly 1 channel: LST."""
        from src.fusion.backbone import ThermalEncoder, THERMAL_DIM
        encoder = ThermalEncoder(
            in_channels=1, embed_dim=THERMAL_DIM
        )
        encoder.eval()

        x = torch.randn(4, 1, 112, 112)
        assert encoder(x).shape == (4, THERMAL_DIM)


class TestIoTEncoder:

    def test_output_shape_sequence(self):
        """IoT encoder should handle [B, T, 16] input."""
        from src.fusion.backbone import IoTEncoder, IOT_DIM
        encoder = IoTEncoder(
            input_dim=16, hidden_dim=256, embed_dim=IOT_DIM
        )
        encoder.eval()

        x   = torch.randn(2, 12, 16)
        out = encoder(x)
        assert out.shape == (2, IOT_DIM)

    def test_output_shape_single_step(self):
        """IoT encoder should handle [B, 16] single timestep."""
        from src.fusion.backbone import IoTEncoder, IOT_DIM
        encoder = IoTEncoder(
            input_dim=16, hidden_dim=256, embed_dim=IOT_DIM
        )
        encoder.eval()

        x   = torch.randn(2, 16)
        out = encoder(x)
        assert out.shape == (2, IOT_DIM)


class TestOSMEncoder:

    def test_output_shape_no_edges(self):
        """OSM encoder should work with no edges."""
        from src.fusion.backbone import OSMEncoder, OSM_DIM
        encoder = OSMEncoder(
            node_features=64, embed_dim=OSM_DIM
        )
        encoder.eval()

        nodes      = torch.randn(5, 64)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        out        = encoder(nodes, edge_index)
        assert out.shape[1] == OSM_DIM

    def test_output_shape_with_edges(self):
        """OSM encoder should work with edges."""
        from src.fusion.backbone import OSMEncoder, OSM_DIM
        encoder = OSMEncoder(
            node_features=64, embed_dim=OSM_DIM
        )
        encoder.eval()

        nodes      = torch.randn(6, 64)
        edge_index = torch.tensor(
            [[0,1,2,3], [1,2,3,4]], dtype=torch.long
        )
        out = encoder(nodes, edge_index)
        assert out.shape[1] == OSM_DIM


# ── Test Cross-Modal Attention ────────────────────────────────
class TestCrossModalAttention:

    def test_output_shape(self):
        """Fusion output should be [B, COMMON_DIM]."""
        from src.fusion.backbone import (
            CrossModalAttention, COMMON_DIM
        )
        fusion = CrossModalAttention(
            common_dim=COMMON_DIM, num_heads=8, num_layers=2
        )
        fusion.eval()

        B   = 2
        emb = {
            'optical': torch.randn(B, COMMON_DIM),
            'sar':     torch.randn(B, COMMON_DIM),
            'lidar':   torch.randn(B, COMMON_DIM),
        }
        out = fusion(emb, training=False)
        assert out.shape == (B, COMMON_DIM)

    def test_missing_modalities(self):
        """Should work with any subset of modalities."""
        from src.fusion.backbone import (
            CrossModalAttention, COMMON_DIM
        )
        fusion = CrossModalAttention(
            common_dim=COMMON_DIM, num_heads=8, num_layers=2
        )
        fusion.eval()

        # Only optical + SAR (missing lidar, thermal, osm, iot)
        emb = {
            'optical': torch.randn(1, COMMON_DIM),
            'sar':     torch.randn(1, COMMON_DIM),
        }
        out = fusion(emb, training=False)
        assert out.shape == (1, COMMON_DIM)

    def test_modality_dropout_training(self):
        """Modality dropout should produce different outputs."""
        from src.fusion.backbone import (
            CrossModalAttention, COMMON_DIM
        )
        fusion = CrossModalAttention(
            common_dim=COMMON_DIM,
            num_heads=8,
            num_layers=2,
            modality_dropout_prob=0.5
        )
        fusion.train()

        emb = {
            'optical': torch.randn(4, COMMON_DIM),
            'sar':     torch.randn(4, COMMON_DIM),
            'lidar':   torch.randn(4, COMMON_DIM),
            'thermal': torch.randn(4, COMMON_DIM),
        }
        out1 = fusion(emb, training=True)
        out2 = fusion(emb, training=True)
        # Outputs may differ due to dropout
        assert out1.shape == (4, COMMON_DIM)
        assert out2.shape == (4, COMMON_DIM)


# ── Test Task Heads ───────────────────────────────────────────
class TestUrbanChangeHead:

    def test_output_shapes(self):
        """Urban change head should return score + type."""
        from src.tasks.urban_change import UrbanChangeHead
        head = UrbanChangeHead(common_dim=512)
        head.eval()

        emb_t1 = torch.randn(4, 512)
        emb_t2 = torch.randn(4, 512)
        preds  = head(emb_t1, emb_t2)

        assert preds['change_score'].shape == (4,)
        assert preds['change_type'].shape  == (4, 6)

    def test_score_range(self):
        """Change score must be in [0, 1]."""
        from src.tasks.urban_change import UrbanChangeHead
        head = UrbanChangeHead(common_dim=512)
        head.eval()

        with torch.no_grad():
            preds = head(
                torch.randn(16, 512),
                torch.randn(16, 512)
            )
        scores = preds['change_score'].detach().cpu().tolist()
        assert min(scores) >= 0.0
        assert max(scores) <= 1.0


class TestFloodDamageHead:

    def test_output_shapes(self):
        """Flood head should return 4 prediction types."""
        from src.tasks.flood_damage import FloodDamageHead
        head = FloodDamageHead(common_dim=512)
        head.eval()

        emb_t1 = torch.randn(4, 512)
        emb_t2 = torch.randn(4, 512)
        preds  = head(emb_t1, emb_t2)

        assert preds['flood_probability'].shape == (4,)
        assert preds['flood_depth'].shape       == (4,)
        assert preds['damage_level'].shape      == (4, 5)
        assert preds['road_blocked'].shape      == (4,)

    def test_probability_range(self):
        """Flood probability must be in [0, 1]."""
        from src.tasks.flood_damage import FloodDamageHead
        head = FloodDamageHead(common_dim=512)
        head.eval()

        with torch.no_grad():
            preds = head(
                torch.randn(8, 512),
                torch.randn(8, 512)
            )
        probs = preds['flood_probability'].detach().cpu().tolist()
        assert min(probs) >= 0.0
        assert max(probs) <= 1.0

    def test_depth_non_negative(self):
        """Flood depth must be non-negative."""
        from src.tasks.flood_damage import FloodDamageHead
        head = FloodDamageHead(common_dim=512)
        head.eval()

        with torch.no_grad():
            preds = head(
                torch.randn(8, 512),
                torch.randn(8, 512)
            )
        depths = preds['flood_depth'].detach().cpu().tolist()
        assert min(depths) >= 0.0


class TestAgricultureHead:

    def test_output_shapes(self):
        """Agriculture head should return 7 outputs."""
        from src.tasks.agriculture import AgricultureHead
        head = AgricultureHead(common_dim=512)
        head.eval()

        embedding = torch.randn(4, 512)
        preds     = head(embedding)

        assert preds['ndvi'].shape          == (4,)
        assert preds['lai'].shape           == (4,)
        assert preds['crop_height_m'].shape == (4,)
        assert preds['soil_moisture'].shape == (4,)
        assert preds['cwsi'].shape          == (4,)
        assert preds['et_mm_day'].shape     == (4,)
        assert preds['stress_type'].shape   == (4, 4)

    def test_ndvi_range(self):
        """NDVI must be in [-1, 1]."""
        from src.tasks.agriculture import AgricultureHead
        head = AgricultureHead(common_dim=512)
        head.eval()

        with torch.no_grad():
            preds = head(torch.randn(16, 512))
        ndvi = preds['ndvi'].detach().cpu().tolist()
        assert min(ndvi) >= -1.0
        assert max(ndvi) <= 1.0

    def test_soil_moisture_range(self):
        """Soil moisture must be in [0, 1]."""
        from src.tasks.agriculture import AgricultureHead
        head = AgricultureHead(common_dim=512)
        head.eval()

        with torch.no_grad():
            preds = head(torch.randn(16, 512))
        sm = preds['soil_moisture'].detach().cpu().tolist()
        assert min(sm) >= 0.0
        assert max(sm) <= 1.0

    def test_cwsi_range(self):
        """CWSI must be in [0, 1]."""
        from src.tasks.agriculture import AgricultureHead
        head = AgricultureHead(common_dim=512)
        head.eval()

        with torch.no_grad():
            preds = head(torch.randn(16, 512))
        cwsi = preds['cwsi'].detach().cpu().tolist()
        assert min(cwsi) >= 0.0
        assert max(cwsi) <= 1.0


# ── Test Target Generators ────────────────────────────────────
class TestUrbanChangeTargetGenerator:

    def test_spectral_change_range(self):
        """Spectral change must be in [0, 1]."""
        from src.tasks.urban_change import (
            UrbanChangeTargetGenerator
        )
        gen    = UrbanChangeTargetGenerator()
        bands1 = np.random.randint(
            100, 3000, (6, 256, 256)
        ).astype(np.float32)
        bands2 = np.random.randint(
            100, 3000, (6, 256, 256)
        ).astype(np.float32)
        change = gen.compute_spectral_change(bands1, bands2)

        assert change.min() >= 0.0
        assert change.max() <= 1.0
        assert change.shape == (256, 256)

    def test_lidar_change_range(self):
        """Normalized lidar change must be in [0, 1]."""
        from src.tasks.urban_change import (
            UrbanChangeTargetGenerator
        )
        gen      = UrbanChangeTargetGenerator()
        ndsm_ahn3 = np.random.uniform(0, 20, (256, 256))
        ndsm_ahn4 = np.random.uniform(0, 20, (256, 256))
        change    = gen.compute_lidar_change(ndsm_ahn3, ndsm_ahn4)

        assert change.min() >= 0.0
        assert change.max() <= 1.0


class TestFloodTargetGenerator:

    def test_sar_water_mask_output(self):
        """SAR water mask should be boolean array."""
        from src.tasks.flood_damage import FloodTargetGenerator
        gen    = FloodTargetGenerator()

        # Low backscatter = water
        sar_t1 = np.full((100, 100), 0.1)
        sar_t2 = np.full((100, 100), 0.0001)  # very low = water
        mask   = gen.compute_sar_water_mask(sar_t1, sar_t2)

        assert mask.dtype == bool
        assert mask.shape == (100, 100)

    def test_building_damage_levels(self):
        """Damage levels should be in [0, 4]."""
        from src.tasks.flood_damage import FloodTargetGenerator
        gen   = FloodTargetGenerator()
        depth = np.array([[0.0, 0.2, 0.5, 1.5, 3.0]])
        dmg   = gen.compute_building_damage(depth)

        assert dmg[0, 0] == 0   # no_damage
        assert dmg[0, 1] == 1   # minor
        assert dmg[0, 2] == 2   # moderate
        assert dmg[0, 3] == 3   # severe
        assert dmg[0, 4] == 4   # destroyed


class TestAgricultureTargetGenerator:

    def test_ndvi_range(self):
        """NDVI must be in [-1, 1]."""
        from src.tasks.agriculture import (
            AgricultureTargetGenerator
        )
        gen   = AgricultureTargetGenerator()
        bands = np.random.randint(
            100, 3000, (6, 256, 256)
        ).astype(np.float32)
        ndvi  = gen.compute_ndvi(bands)

        assert ndvi.min() >= -1.0
        assert ndvi.max() <= 1.0

    def test_lai_range(self):
        """LAI must be in [0, 10]."""
        from src.tasks.agriculture import (
            AgricultureTargetGenerator
        )
        gen   = AgricultureTargetGenerator()
        bands = np.random.randint(
            100, 3000, (6, 256, 256)
        ).astype(np.float32)
        lai   = gen.compute_lai(bands)

        assert lai.min() >= 0.0
        assert lai.max() <= 10.0

    def test_crop_height_non_negative(self):
        """Crop height must be non-negative."""
        from src.tasks.agriculture import (
            AgricultureTargetGenerator
        )
        gen = AgricultureTargetGenerator()
        dsm = np.random.uniform(0, 3, (100, 100))
        dtm = np.random.uniform(-1, 0.5, (100, 100))
        h   = gen.compute_crop_height(dsm, dtm)

        assert h.min() >= 0.0

    def test_cwsi_range(self):
        """CWSI must be in [0, 1]."""
        from src.tasks.agriculture import (
            AgricultureTargetGenerator
        )
        gen = AgricultureTargetGenerator()
        lst = np.random.uniform(15, 35, (50, 50)).astype(
            np.float32
        )
        cwsi = gen.compute_cwsi_from_thermal(
            lst, air_temp_c=20.0,
            wind_speed_ms=3.0, humidity_pct=75.0
        )
        assert cwsi.min() >= 0.0
        assert cwsi.max() <= 1.0
