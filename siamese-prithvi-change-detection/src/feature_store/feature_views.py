"""
Feast Feature View Definitions — GeoAI MLOps Project 1
======================================================
Two feature views:

1. sentinel2_patch_features:
   The 34 tabular features fed INTO the model as input.
   Excludes ndvi_mean, ndwi_mean, ndbi_mean — these are used ONLY
   to compute the training target, not as model inputs (prevents
   the model from shortcutting to the answer via tabular inputs).

2. sentinel2_change_label_features:
   The 3 excluded features (ndvi_mean, ndwi_mean, ndbi_mean) +
   metadata needed to compute the weak supervision target at
   pairing time. Used ONLY for label computation, not model input.
"""
from datetime import timedelta
from feast import FeatureView, Field
from feast.types import Float64, Int64, String
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from entities import patch

# ── Source: existing PostGIS sentinel2_patches table ─────────────
sentinel2_source = PostgreSQLSource(
    name="sentinel2_patches_source",
    query="""
        SELECT
            patch_id,
            tile_id,
            aoi,
            acquisition_date,
            cloud_cover,
            valid_pixel_ratio,
            total_pixels,
            valid_pixels,
            ndvi_mean,
            ndvi_std,
            ndwi_mean,
            ndwi_std,
            mndwi_mean,
            ndbi_mean,
            ndbi_std,
            arvi_mean,
            evi_mean,
            bsi_mean,
            nbi_mean,
            savi_mean,
            bright_mean,
            blue_mean, blue_std, blue_p25, blue_p75,
            green_mean, green_std, green_p25, green_p75,
            red_mean, red_std, red_p25, red_p75,
            nir_mean, nir_std, nir_p25, nir_p75,
            swir_mean, swir_std, swir_p25, swir_p75,
            ingested_at AS event_timestamp
        FROM sentinel2_patches
    """,
    timestamp_field="event_timestamp",
)

# ── Feature View 1: the 34 model INPUT features ──────────────────
# ndvi_mean, ndwi_mean, ndbi_mean deliberately EXCLUDED here
# (they are in sentinel2_change_label_features instead)
sentinel2_patch_features = FeatureView(
    name="sentinel2_patch_features",
    entities=[patch],
    ttl=timedelta(days=730),  # 2 years — covers our full backfill range
    schema=[
        # Spectral index stats (excluding the 3 used in target formula)
        Field(name="ndvi_std",    dtype=Float64),
        Field(name="ndwi_std",    dtype=Float64),
        Field(name="mndwi_mean",  dtype=Float64),
        Field(name="ndbi_std",    dtype=Float64),
        Field(name="arvi_mean",   dtype=Float64),
        Field(name="evi_mean",    dtype=Float64),
        Field(name="bsi_mean",    dtype=Float64),
        Field(name="nbi_mean",    dtype=Float64),
        Field(name="savi_mean",   dtype=Float64),
        Field(name="bright_mean", dtype=Float64),
        # Blue band stats
        Field(name="blue_mean",   dtype=Float64),
        Field(name="blue_std",    dtype=Float64),
        Field(name="blue_p25",    dtype=Float64),
        Field(name="blue_p75",    dtype=Float64),
        # Green band stats
        Field(name="green_mean",  dtype=Float64),
        Field(name="green_std",   dtype=Float64),
        Field(name="green_p25",   dtype=Float64),
        Field(name="green_p75",   dtype=Float64),
        # Red band stats
        Field(name="red_mean",    dtype=Float64),
        Field(name="red_std",     dtype=Float64),
        Field(name="red_p25",     dtype=Float64),
        Field(name="red_p75",     dtype=Float64),
        # NIR band stats
        Field(name="nir_mean",    dtype=Float64),
        Field(name="nir_std",     dtype=Float64),
        Field(name="nir_p25",     dtype=Float64),
        Field(name="nir_p75",     dtype=Float64),
        # SWIR band stats
        Field(name="swir_mean",   dtype=Float64),
        Field(name="swir_std",    dtype=Float64),
        Field(name="swir_p25",    dtype=Float64),
        Field(name="swir_p75",    dtype=Float64),
        # Quality/context features
        Field(name="valid_pixel_ratio", dtype=Float64),
        Field(name="cloud_cover",       dtype=Float64),
        Field(name="total_pixels",      dtype=Int64),
        Field(name="valid_pixels",      dtype=Int64),
    ],
    source=sentinel2_source,
    description=(
        "34 tabular features per Sentinel-2 patch fed as model INPUT. "
        "Excludes ndvi_mean/ndwi_mean/ndbi_mean (used only for target). "
        "Standardized using TRAIN split stats before feeding to model."
    ),
)

# ── Feature View 2: excluded features for target computation ─────
# These 3 features compute the weak supervision target:
# target = sqrt(d_ndvi² + d_ndwi² + d_ndbi²)
# NEVER fed to model as input — used only at pairing time for label
sentinel2_change_label_features = FeatureView(
    name="sentinel2_change_label_features",
    entities=[patch],
    ttl=timedelta(days=730),
    schema=[
        Field(name="ndvi_mean", dtype=Float64),
        Field(name="ndwi_mean", dtype=Float64),
        Field(name="ndbi_mean", dtype=Float64),
        # Metadata for pairing and stratification
        Field(name="aoi",              dtype=String),
        Field(name="cloud_cover",      dtype=Float64),
        Field(name="valid_pixel_ratio",dtype=Float64),
    ],
    source=sentinel2_source,
    description=(
        "Features used ONLY to compute the weak supervision target. "
        "ndvi_mean/ndwi_mean/ndbi_mean are excluded from model input "
        "to prevent shortcutting to the target formula."
    ),
)
