"""
OmniGeoFusion — Data Pipeline Tests
=====================================
Unit tests for all data pipeline modules.
"""

import pytest
import numpy as np
import torch


# ── Test Sentinel Pipeline ─────────────────────────────────────
class TestSentinel2Pipeline:

    def test_compute_cloud_cover_clear(self):
        """Clear sky patch should have low cloud cover."""
        from src.data.sentinel.sentinel_pipeline import (
            Sentinel2Pipeline
        )
        cfg      = {'aws': {'bucket': 'test', 'prefix': 'test'}}
        pipeline = Sentinel2Pipeline(cfg)

        # SCL class 4 = vegetation (valid, not cloud)
        scl = np.full((256, 256), 4, dtype=np.uint8)
        cc  = pipeline.compute_cloud_cover(scl)
        assert cc == 0.0

    def test_compute_cloud_cover_cloudy(self):
        """Cloudy patch should have high cloud cover."""
        from src.data.sentinel.sentinel_pipeline import (
            Sentinel2Pipeline
        )
        cfg      = {'aws': {'bucket': 'test', 'prefix': 'test'}}
        pipeline = Sentinel2Pipeline(cfg)

        # SCL class 9 = cloud high probability
        scl = np.full((256, 256), 9, dtype=np.uint8)
        cc  = pipeline.compute_cloud_cover(scl)
        assert cc == 1.0

    def test_chip_scene_nodata_filter(self):
        """Nodata patches should be filtered out."""
        from src.data.sentinel.sentinel_pipeline import (
            Sentinel2Pipeline
        )
        import rasterio
        cfg      = {'aws': {'bucket': 'test', 'prefix': 'test'}}
        pipeline = Sentinel2Pipeline(cfg)

        # All zeros = nodata
        bands     = np.zeros((6, 512, 512), dtype=np.float32)
        scl       = np.full((512, 512), 4, dtype=np.uint8)
        transform = rasterio.transform.from_bounds(
            4.0, 52.0, 5.0, 53.0, 512, 512
        )
        patches = pipeline.chip_scene(
            bands, scl, transform, 'test_scene'
        )
        assert len(patches) == 0

    def test_chip_scene_valid_patches(self):
        """Valid patches should be returned."""
        from src.data.sentinel.sentinel_pipeline import (
            Sentinel2Pipeline
        )
        import rasterio
        cfg      = {'aws': {'bucket': 'test', 'prefix': 'test'}}
        pipeline = Sentinel2Pipeline(cfg)

        # Valid data
        bands = np.random.randint(
            100, 3000, (6, 512, 512)
        ).astype(np.float32)
        scl   = np.full((512, 512), 4, dtype=np.uint8)
        transform = rasterio.transform.from_bounds(
            4.0, 52.0, 5.0, 53.0, 512, 512
        )
        patches = pipeline.chip_scene(
            bands, scl, transform, 'test_scene'
        )
        assert len(patches) > 0
        assert patches[0]['bands'].shape == (6, 256, 256)


# ── Test LiDAR Pipeline ───────────────────────────────────────
class TestAHNPipeline:

    def test_compute_ndsm(self):
        """nDSM should be DSM - DTM, clipped to ≥ 0."""
        from src.data.lidar.lidar_pipeline import AHNPipeline
        cfg      = {'aws': {'bucket': 'test', 'prefix': 'test'}}
        pipeline = AHNPipeline(cfg)

        dsm  = np.array([[10.0, 5.0], [3.0, -9999.0]])
        dtm  = np.array([[2.0,  2.0], [2.0, -9999.0]])
        ndsm = pipeline.compute_ndsm(dsm, dtm)

        assert ndsm[0, 0] == pytest.approx(8.0)
        assert ndsm[0, 1] == pytest.approx(3.0)
        assert ndsm[1, 1] == pytest.approx(-9999.0)

    def test_compute_height_change(self):
        """Height change should detect construction + demolition."""
        from src.data.lidar.lidar_pipeline import AHNPipeline
        cfg      = {'aws': {'bucket': 'test', 'prefix': 'test'}}
        pipeline = AHNPipeline(cfg)

        ndsm_ahn3 = np.array([[2.0, 10.0], [5.0, -9999.0]])
        ndsm_ahn4 = np.array([[8.0, 3.0],  [5.0, -9999.0]])
        change    = pipeline.compute_height_change(
            ndsm_ahn3, ndsm_ahn4
        )

        # Construction: +6m
        assert change[0, 0] == pytest.approx(6.0)
        # Demolition: -7m
        assert change[0, 1] == pytest.approx(-7.0)
        # No change
        assert change[1, 0] == pytest.approx(0.0)
        # Nodata
        assert change[1, 1] == pytest.approx(-9999.0)

    def test_compute_flood_depth(self):
        """Flood depth should be max(0, water_level - terrain)."""
        from src.data.lidar.lidar_pipeline import AHNPipeline
        cfg      = {'aws': {'bucket': 'test', 'prefix': 'test'}}
        pipeline = AHNPipeline(cfg)

        dtm          = np.array([[-1.0, 0.5], [2.0, -9999.0]])
        water_level  = 1.0
        flood_mask   = np.array([[True, True], [False, False]])
        depth        = pipeline.compute_flood_depth(
            dtm, water_level, flood_mask
        )

        assert depth[0, 0] == pytest.approx(2.0)  # -1 → 1m = 2m deep
        assert depth[0, 1] == pytest.approx(0.5)  # 0.5 → 1m = 0.5m deep
        assert depth[1, 0] == pytest.approx(0.0)  # not flooded

    def test_compute_terrain_slope(self):
        """Slope should be computed from DTM gradient."""
        from src.data.lidar.lidar_pipeline import AHNPipeline
        cfg      = {'aws': {'bucket': 'test', 'prefix': 'test'}}
        pipeline = AHNPipeline(cfg)

        # Flat terrain
        dtm_flat  = np.zeros((10, 10), dtype=np.float32)
        slope     = pipeline.compute_terrain_slope(dtm_flat)
        assert slope.max() == pytest.approx(0.0, abs=1e-5)


# ── Test IoT Pipeline ─────────────────────────────────────────
class TestIoTFeatureExtractor:

    def test_feature_vector_shape(self):
        """IoT feature vector must be exactly 16 dimensions."""
        from src.data.iot.iot_pipeline import IoTFeatureExtractor
        cfg       = {}
        extractor = IoTFeatureExtractor(cfg)
        assert len(extractor.feature_names) == 16

    def test_feature_names_order(self):
        """Feature names must match expected order."""
        from src.data.iot.iot_pipeline import IoTFeatureExtractor
        cfg       = {}
        extractor = IoTFeatureExtractor(cfg)

        expected = [
            'no2', 'pm25', 'pm10', 'o3',
            'water_level_m', 'water_temp_c',
            'air_temp_c', 'temp_min_c', 'temp_max_c',
            'rainfall_mm', 'wind_speed_ms', 'humidity_pct',
            'soil_temp_5cm_c', 'soil_temp_10cm_c',
            'soil_temp_20cm_c', 'soil_temp_50cm_c',
        ]
        assert extractor.feature_names == expected

    def test_soil_moisture_proxy(self):
        """Soil moisture proxy should be in [0, 1]."""
        from src.data.iot.iot_pipeline import IoTFeatureExtractor
        cfg       = {}
        extractor = IoTFeatureExtractor(cfg)

        sm = extractor.compute_soil_moisture_proxy(
            rainfall_mm=10.0,
            temp_c=15.0,
            wind_ms=3.0,
            humidity_pct=80.0
        )
        assert 0.0 <= sm <= 1.0

    def test_flood_risk_zero(self):
        """Flood risk should be 0 when water below terrain."""
        from src.data.iot.iot_pipeline import IoTFeatureExtractor
        cfg       = {}
        extractor = IoTFeatureExtractor(cfg)

        risk = extractor.compute_flood_risk(
            water_level_m=-1.0,
            rainfall_mm=0.0,
            terrain_elevation_m=2.0
        )
        assert risk == pytest.approx(0.0, abs=0.1)

    def test_normalize_shape(self):
        """Normalized features should keep same shape."""
        from src.data.iot.iot_pipeline import IoTFeatureExtractor
        cfg       = {}
        extractor = IoTFeatureExtractor(cfg)

        features = np.random.randn(16).astype(np.float32)
        stats    = {
            name: {'mean': 0.0, 'std': 1.0}
            for name in extractor.feature_names
        }
        normalized = extractor.normalize(features, stats)
        assert normalized.shape == (16,)


# ── Test Thermal Pipeline ─────────────────────────────────────
class TestLandsatThermalPipeline:

    def test_dn_to_celsius(self):
        """DN to Celsius conversion should match formula."""
        from src.data.thermal.thermal_pipeline import (
            LandsatThermalPipeline,
            LANDSAT_SCALE, LANDSAT_OFFSET, KELVIN_OFFSET
        )
        cfg      = {'aws': {'bucket': 'test', 'prefix': 'test'}}
        pipeline = LandsatThermalPipeline(cfg)

        dn  = np.array([[10000, 20000]], dtype=np.uint16)
        lst = pipeline.dn_to_celsius(dn)

        expected_0 = (
            10000 * LANDSAT_SCALE + LANDSAT_OFFSET - KELVIN_OFFSET
        )
        assert lst[0, 0] == pytest.approx(expected_0, rel=1e-4)

    def test_dn_to_celsius_nodata(self):
        """Nodata pixels should be NaN in output."""
        from src.data.thermal.thermal_pipeline import (
            LandsatThermalPipeline
        )
        cfg      = {'aws': {'bucket': 'test', 'prefix': 'test'}}
        pipeline = LandsatThermalPipeline(cfg)

        dn  = np.array([[0, 10000]], dtype=np.uint16)
        lst = pipeline.dn_to_celsius(dn, nodata=0)
        assert np.isnan(lst[0, 0])
        assert not np.isnan(lst[0, 1])

    def test_compute_uhi_range(self):
        """UHI should center around zero."""
        from src.data.thermal.thermal_pipeline import (
            LandsatThermalPipeline
        )
        cfg      = {'aws': {'bucket': 'test', 'prefix': 'test'}}
        pipeline = LandsatThermalPipeline(cfg)

        lst = np.random.uniform(20, 40, (100, 100)).astype(
            np.float32
        )
        uhi = pipeline.compute_uhi(lst)
        assert not np.isnan(uhi).all()
        # UHI mean should be approximately zero
        assert abs(np.nanmean(uhi)) < 1.0

    def test_compute_cwsi_range(self):
        """CWSI should be in [0, 1]."""
        from src.data.thermal.thermal_pipeline import (
            LandsatThermalPipeline
        )
        cfg      = {'aws': {'bucket': 'test', 'prefix': 'test'}}
        pipeline = LandsatThermalPipeline(cfg)

        lst  = np.random.uniform(15, 35, (50, 50)).astype(
            np.float32
        )
        cwsi = pipeline.compute_cwsi(
            lst, air_temp_c=20.0,
            wind_speed_ms=3.0, humidity_pct=70.0
        )
        valid = ~np.isnan(cwsi)
        assert cwsi[valid].min() >= 0.0
        assert cwsi[valid].max() <= 1.0


# ── Test OSM Pipeline ─────────────────────────────────────────
class TestOSMFeatureExtractor:

    def test_encode_building_type(self):
        """Building type encoding should return float."""
        from src.data.osm.osm_pipeline import OSMFeatureExtractor
        cfg       = {}
        extractor = OSMFeatureExtractor(cfg)
        val       = extractor._encode_building_type('residential')
        assert isinstance(val, float)
        assert val > 0

    def test_encode_road_class(self):
        """Road class encoding should return float."""
        from src.data.osm.osm_pipeline import OSMFeatureExtractor
        cfg       = {}
        extractor = OSMFeatureExtractor(cfg)
        val       = extractor._encode_road_class('motorway')
        assert isinstance(val, float)
        assert val > 0

    def test_building_stats_empty(self):
        """Empty buildings should return zero stats."""
        import geopandas as gpd
        from src.data.osm.osm_pipeline import OSMFeatureExtractor
        cfg       = {}
        extractor = OSMFeatureExtractor(cfg)

        empty_gdf = gpd.GeoDataFrame(
            columns=[
                'geometry', 'osm_id', 'building_type',
                'height_m', 'area_m2', 'name'
            ],
            crs='EPSG:4326'
        )
        stats = extractor._building_stats(
            empty_gdf, [4.8, 52.3, 4.9, 52.4]
        )
        assert stats['building_count'] == 0
        assert stats['building_density'] == 0.0
