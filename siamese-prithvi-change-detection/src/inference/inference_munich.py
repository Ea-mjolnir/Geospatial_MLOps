"""
GeoAI MLOps Project 1 — Munich Inference Pipeline
===================================================
Runs change detection inference on Munich Sentinel-2 imagery
using the trained SiamesePrithviModel.

Pipeline:
    1. Download T1 (July 2020) + T2 (July 2023) scenes from AWS S3
    2. Preprocess: chip into 256x256 patches, cloud filter, normalize
    3. Compute 34 tabular features per patch
    4. Run inference with best trained model
    5. Full geospatial analytics + output

Usage:
    python src/inference/inference_munich.py \
        --model /gdrive/MyDrive/geoai_mlops/checkpoints/best/best_model_chunk6_r2_0.9919.pt \
        --output /gdrive/MyDrive/geoai_mlops/inference/munich_2020_2023
"""

import os
import sys
import json
import logging
import argparse
import warnings
import numpy as np
import boto3
import rasterio
import torch
import pandas as pd
import geopandas as gpd

from pathlib import Path
from datetime import datetime
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from shapely.geometry import box

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────
TILE        = '32UPU'
TILE_PATH   = '32/U/PU'
S3_BUCKET   = 'sentinel-cogs'
S3_PREFIX   = 'sentinel-s2-l2a-cogs'

# Best low-cloud summer scenes for Munich
T1_SCENE    = 'S2A_32UPU_20200707_0_L2A'   # July 2020
T2_SCENE    = 'S2B_32UPU_20230707_0_L2A'   # July 2023
T1_DATE     = '2020-07-07'
T2_DATE     = '2023-07-07'
DAY_GAP     = (datetime(2023, 7, 7) - datetime(2020, 7, 7)).days  # 1096

BANDS       = ['B02', 'B03', 'B04', 'B08', 'B11']  # blue,green,red,nir,swir
PATCH_SIZE  = 256
PATCH_STEP  = 256   # no overlap — full coverage
CLOUD_MAX   = 0.3   # skip patches with >30% cloud cover
AOI         = 'munich'

# Stats paths — check EC2 first, then GDrive (Colab)
def _find_stats_dir():
    candidates = [
        os.path.expanduser('~/geoai-mlops-p1/data/stats'),
        '/gdrive/MyDrive/geoai_mlops/stats',
        '/content/geoai-mlops-p1/data/stats',
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, 'tabular_stats.json')):
            return path
    raise FileNotFoundError(
        f'Stats directory not found. Tried: {candidates}')

STATS_DIR           = _find_stats_dir()
TABULAR_STATS_PATH  = os.path.join(STATS_DIR, 'tabular_stats.json')
BAND_STATS_PATH     = os.path.join(STATS_DIR, 'band_stats.json')
DAY_GAP_STATS_PATH  = os.path.join(STATS_DIR, 'day_gap_stats.json')

# All 34 feature names in exact training order
FEATURE_NAMES = [
    'ndvi_std', 'ndwi_std', 'mndwi_mean', 'ndbi_std',
    'arvi_mean', 'evi_mean', 'bsi_mean', 'nbi_mean',
    'savi_mean', 'bright_mean',
    'blue_mean', 'blue_std', 'blue_p25', 'blue_p75',
    'green_mean', 'green_std', 'green_p25', 'green_p75',
    'red_mean', 'red_std', 'red_p25', 'red_p75',
    'nir_mean', 'nir_std', 'nir_p25', 'nir_p75',
    'swir_mean', 'swir_std', 'swir_p25', 'swir_p75',
    'valid_pixel_ratio', 'cloud_cover', 'total_pixels', 'valid_pixels',
]

# Change score thresholds for classification
CHANGE_THRESHOLDS = {
    'no_change':     (0.00, 0.20),
    'low_change':    (0.20, 0.40),
    'medium_change': (0.40, 0.60),
    'high_change':   (0.60, 1.00),
}


# ── Step 1: Download scenes ────────────────────────────────────
def download_scene(scene_name, year, month, output_dir):
    """Download all required bands for one scene from AWS S3."""
    # sentinel-cogs is a public bucket — use unsigned requests
    from botocore import UNSIGNED
    from botocore.config import Config
    s3 = boto3.client(
        's3', region_name='us-east-1',
        config=Config(signature_version=UNSIGNED)
    )
    os.makedirs(output_dir, exist_ok=True)

    scene_dir = os.path.join(output_dir, scene_name)
    os.makedirs(scene_dir, exist_ok=True)

    bands_to_download = BANDS + ['SCL']
    downloaded = []

    for band in bands_to_download:
        local_path = os.path.join(scene_dir, f'{band}.tif')
        if os.path.exists(local_path):
            log.info(f'  ✅ {band}.tif already exists — skipping')
            downloaded.append(local_path)
            continue

        s3_key = f'{S3_PREFIX}/{TILE_PATH}/{year}/{month}/{scene_name}/{band}.tif'
        log.info(f'  Downloading {band}.tif from s3://{S3_BUCKET}/{s3_key}')
        try:
            s3.download_file(S3_BUCKET, s3_key, local_path)
            downloaded.append(local_path)
            log.info(f'  ✅ {band}.tif downloaded')
        except Exception as e:
            log.error(f'  ❌ Failed to download {band}.tif: {e}')

    return scene_dir


# ── Step 2: Read + resample scene bands ───────────────────────
def read_scene_bands(scene_dir):
    """
    Read all 5 bands + SCL from scene directory.
    Resamples 20m bands (B11, SCL) to match 10m bands.
    Returns stacked array [5, H, W] and SCL [H, W].
    """
    # Read 10m reference band first (B02)
    with rasterio.open(os.path.join(scene_dir, 'B02.tif')) as src:
        profile  = src.profile.copy()
        height   = src.height
        width    = src.width
        transform = src.transform
        crs      = src.crs

    band_arrays = []
    band_names  = ['B02', 'B03', 'B04', 'B08', 'B11']

    for band in band_names:
        path = os.path.join(scene_dir, f'{band}.tif')
        with rasterio.open(path) as src:
            if src.height != height or src.width != width:
                # Resample to 10m resolution
                from rasterio.enums import Resampling
                data = src.read(
                    1,
                    out_shape=(height, width),
                    resampling=Resampling.bilinear
                ).astype(np.float32)
            else:
                data = src.read(1).astype(np.float32)
        band_arrays.append(data)

    # Read SCL (cloud mask) — resample to 10m
    with rasterio.open(os.path.join(scene_dir, 'SCL.tif')) as src:
        scl = src.read(
            1,
            out_shape=(height, width),
            resampling=rasterio.enums.Resampling.nearest
        )

    bands = np.stack(band_arrays, axis=0)  # [5, H, W]

    return bands, scl, profile, transform, crs


# ── Step 3: Compute tabular features ──────────────────────────
def compute_tabular_features(bands, scl):
    """
    Compute all 34 tabular features from a patch.

    Args:
        bands: [5, H, W] float32 array (B02,B03,B04,B08,B11)
        scl:   [H, W] uint8 SCL cloud mask

    Returns:
        dict of 34 features
    """
    blue  = bands[0].astype(np.float32)
    green = bands[1].astype(np.float32)
    red   = bands[2].astype(np.float32)
    nir   = bands[3].astype(np.float32)
    swir  = bands[4].astype(np.float32)

    eps = 1e-8

    # Valid pixel mask (SCL: 4=vegetation, 5=bare soil, 6=water, 7=unclassified)
    valid_mask = np.isin(scl, [4, 5, 6, 7])
    total_pixels = bands.shape[1] * bands.shape[2]
    valid_pixels = valid_mask.sum()
    valid_pixel_ratio = float(valid_pixels) / total_pixels

    # Cloud cover (SCL: 8=cloud medium, 9=cloud high, 10=thin cirrus)
    cloud_mask  = np.isin(scl, [8, 9, 10])
    cloud_cover = float(cloud_mask.sum()) / total_pixels

    # Use valid pixels only for index computation
    if valid_pixels < 100:
        # Not enough valid pixels — return zeros
        return {k: 0.0 for k in FEATURE_NAMES}

    b  = blue[valid_mask]
    g  = green[valid_mask]
    r  = red[valid_mask]
    n  = nir[valid_mask]
    s  = swir[valid_mask]

    # Spectral indices
    ndvi  = (n - r) / (n + r + eps)
    ndwi  = (g - n) / (g + n + eps)
    mndwi = (g - s) / (g + s + eps)
    ndbi  = (s - n) / (s + n + eps)
    arvi  = (n - (2*r - b)) / (n + (2*r - b) + eps)
    evi   = 2.5 * (n - r) / (n + 6*r - 7.5*b + 1 + eps)
    bsi   = ((s + r) - (n + b)) / ((s + r) + (n + b) + eps)
    nbi   = (r * s) / (n + eps)
    savi  = 1.5 * (n - r) / (n + r + 0.5 + eps)
    bright = (b + g + r + n + s) / 5.0

    def safe_stats(arr):
        return {
            'mean': float(np.mean(arr)),
            'std':  float(np.std(arr)),
            'p25':  float(np.percentile(arr, 25)),
            'p75':  float(np.percentile(arr, 75)),
        }

    b_stats = safe_stats(b)
    g_stats = safe_stats(g)
    r_stats = safe_stats(r)
    n_stats = safe_stats(n)
    s_stats = safe_stats(s)

    features = {
        'ndvi_std':          float(np.std(ndvi)),
        'ndwi_std':          float(np.std(ndwi)),
        'mndwi_mean':        float(np.mean(mndwi)),
        'ndbi_std':          float(np.std(ndbi)),
        'arvi_mean':         float(np.mean(arvi)),
        'evi_mean':          float(np.mean(evi)),
        'bsi_mean':          float(np.mean(bsi)),
        'nbi_mean':          float(np.mean(nbi)),
        'savi_mean':         float(np.mean(savi)),
        'bright_mean':       float(np.mean(bright)),
        'blue_mean':         b_stats['mean'],
        'blue_std':          b_stats['std'],
        'blue_p25':          b_stats['p25'],
        'blue_p75':          b_stats['p75'],
        'green_mean':        g_stats['mean'],
        'green_std':         g_stats['std'],
        'green_p25':         g_stats['p25'],
        'green_p75':         g_stats['p75'],
        'red_mean':          r_stats['mean'],
        'red_std':           r_stats['std'],
        'red_p25':           r_stats['p25'],
        'red_p75':           r_stats['p75'],
        'nir_mean':          n_stats['mean'],
        'nir_std':           n_stats['std'],
        'nir_p25':           n_stats['p25'],
        'nir_p75':           n_stats['p75'],
        'swir_mean':         s_stats['mean'],
        'swir_std':          s_stats['std'],
        'swir_p25':          s_stats['p25'],
        'swir_p75':          s_stats['p75'],
        'valid_pixel_ratio': valid_pixel_ratio,
        'cloud_cover':       cloud_cover,
        'total_pixels':      float(total_pixels),
        'valid_pixels':      float(valid_pixels),
    }

    return features


# ── Normalization helpers ──────────────────────────────────────
def normalize_bands(bands, band_stats):
    """Normalize 5-band patch using training band statistics."""
    band_keys = ['blue', 'green', 'red', 'nir', 'swir']
    normalized = np.zeros_like(bands, dtype=np.float32)
    for i, key in enumerate(band_keys):
        mean = band_stats[key]['mean']
        std  = band_stats[key]['std']
        normalized[i] = (bands[i] - mean) / (std + 1e-8)
    return normalized


def normalize_tabular(features, tabular_stats):
    """Normalize 34 tabular features using training statistics."""
    normalized = []
    for key in FEATURE_NAMES:
        val  = features.get(key, 0.0)
        mean = tabular_stats[key]['mean']
        std  = tabular_stats[key]['std']
        normalized.append((val - mean) / (std + 1e-8))
    return np.array(normalized, dtype=np.float32)


def normalize_day_gap(day_gap, day_gap_stats):
    """Normalize day gap using training statistics."""
    mean = day_gap_stats['mean']
    std  = day_gap_stats['std']
    return np.array([(day_gap - mean) / (std + 1e-8)], dtype=np.float32)


# ── Step 4: Chip scene into patches ───────────────────────────
def chip_scene(bands, scl, transform):
    """
    Chip full scene into 256x256 patches.
    Returns list of (patch_bands, patch_scl, patch_row, patch_col, geo_bounds)
    """
    _, H, W = bands.shape
    patches  = []

    rows = range(0, H - PATCH_SIZE + 1, PATCH_STEP)
    cols = range(0, W - PATCH_SIZE + 1, PATCH_STEP)

    log.info(f'Chipping scene: {H}x{W} → {len(rows)}x{len(cols)} patches')

    for row in rows:
        for col in cols:
            patch_bands = bands[:, row:row+PATCH_SIZE, col:col+PATCH_SIZE]
            patch_scl   = scl[row:row+PATCH_SIZE, col:col+PATCH_SIZE]

            # Skip patches dominated by nodata (zeros)
            if (patch_bands == 0).mean() > 0.5:
                continue

            # Compute geographic bounds of this patch
            x_min, y_max = rasterio.transform.xy(
                transform, row, col, offset='ul'
            )
            x_max, y_min = rasterio.transform.xy(
                transform, row+PATCH_SIZE, col+PATCH_SIZE, offset='ul'
            )

            patches.append({
                'bands':    patch_bands,
                'scl':      patch_scl,
                'row':      row,
                'col':      col,
                'x_min':    x_min,
                'y_min':    y_min,
                'x_max':    x_max,
                'y_max':    y_max,
            })

    log.info(f'Valid patches: {len(patches):,}')
    return patches


# ── Step 5: Run inference ──────────────────────────────────────
def run_inference(t1_patches, t2_patches, model, device,
                  band_stats, tabular_stats, day_gap_stats,
                  batch_size=16):
    """
    Run model inference on all T1/T2 patch pairs.
    Matches patches by (row, col) position.

    Returns list of results with change scores.
    """
    from tqdm import tqdm

    # Build lookup by (row, col)
    t1_lookup = {(p['row'], p['col']): p for p in t1_patches}
    t2_lookup = {(p['row'], p['col']): p for p in t2_patches}

    # Find common locations
    common = set(t1_lookup.keys()) & set(t2_lookup.keys())
    log.info(f'Common patch locations: {len(common):,}')

    # Normalize day gap once
    day_gap_norm = normalize_day_gap(DAY_GAP, day_gap_stats)

    # Build batch lists
    results = []
    batch_patches = []

    def process_batch(batch):
        """Run one batch through the model."""
        patch_t1   = torch.tensor(
            np.stack([b['t1_bands'] for b in batch])
        ).to(device)
        patch_t2   = torch.tensor(
            np.stack([b['t2_bands'] for b in batch])
        ).to(device)
        tabular_t1 = torch.tensor(
            np.stack([b['t1_tab'] for b in batch])
        ).to(device)
        tabular_t2 = torch.tensor(
            np.stack([b['t2_tab'] for b in batch])
        ).to(device)
        day_gap_t  = torch.tensor(
            np.stack([b['day_gap'] for b in batch])
        ).to(device)

        with torch.no_grad():
            scores = model(patch_t1, patch_t2, tabular_t1,
                          tabular_t2, day_gap_t)

        return scores.cpu().numpy().flatten()

    model.eval()
    pbar = tqdm(sorted(common), desc='Running inference', ncols=80)

    for loc in pbar:
        p1 = t1_lookup[loc]
        p2 = t2_lookup[loc]

        # Skip high cloud patches
        t1_feat = compute_tabular_features(p1['bands'], p1['scl'])
        t2_feat = compute_tabular_features(p2['bands'], p2['scl'])

        if (t1_feat['cloud_cover'] > CLOUD_MAX or
                t2_feat['cloud_cover'] > CLOUD_MAX):
            continue

        # Normalize
        t1_bands_norm = normalize_bands(p1['bands'], band_stats)
        t2_bands_norm = normalize_bands(p2['bands'], band_stats)
        t1_tab_norm   = normalize_tabular(t1_feat, tabular_stats)
        t2_tab_norm   = normalize_tabular(t2_feat, tabular_stats)

        batch_patches.append({
            't1_bands': t1_bands_norm,
            't2_bands': t2_bands_norm,
            't1_tab':   t1_tab_norm,
            't2_tab':   t2_tab_norm,
            'day_gap':  day_gap_norm,
            'row':      loc[0],
            'col':      loc[1],
            'x_min':    p1['x_min'],
            'y_min':    p1['y_min'],
            'x_max':    p1['x_max'],
            'y_max':    p1['y_max'],
            't1_cloud': t1_feat['cloud_cover'],
            't2_cloud': t2_feat['cloud_cover'],
        })

        # Process when batch is full
        if len(batch_patches) >= batch_size:
            scores = process_batch(batch_patches)
            for i, b in enumerate(batch_patches):
                results.append({
                    'patch_row':    b['row'],
                    'patch_col':    b['col'],
                    'x_min':        b['x_min'],
                    'y_min':        b['y_min'],
                    'x_max':        b['x_max'],
                    'y_max':        b['y_max'],
                    'change_score': float(scores[i]),
                    't1_cloud':     b['t1_cloud'],
                    't2_cloud':     b['t2_cloud'],
                    'date_t1':      T1_DATE,
                    'date_t2':      T2_DATE,
                    'day_gap':      DAY_GAP,
                    'aoi':          AOI,
                    'tile':         TILE,
                })
            batch_patches = []

    # Process remaining batch
    if batch_patches:
        scores = process_batch(batch_patches)
        for i, b in enumerate(batch_patches):
            results.append({
                'patch_row':    b['row'],
                'patch_col':    b['col'],
                'x_min':        b['x_min'],
                'y_min':        b['y_min'],
                'x_max':        b['x_max'],
                'y_max':        b['y_max'],
                'change_score': float(scores[i]),
                't1_cloud':     b['t1_cloud'],
                't2_cloud':     b['t2_cloud'],
                'date_t1':      T1_DATE,
                'date_t2':      T2_DATE,
                'day_gap':      DAY_GAP,
                'aoi':          AOI,
                'tile':         TILE,
            })

    log.info(f'Inference complete: {len(results):,} patch pairs scored')
    return results


# ── Step 6: Geospatial analytics + outputs ────────────────────
def run_geospatial_analytics(results, output_dir, crs):
    """
    Full geospatial analytics pipeline:
    A) CSV of all predictions
    B) GeoJSON vector output
    C) Change classification
    D) Statistical summary
    E) Change score GeoTIFF
    F) Hotspot detection
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)

    log.info(f'\n{"="*60}')
    log.info('GEOSPATIAL ANALYTICS')
    log.info(f'{"="*60}')

    # ── A) CSV output ──────────────────────────────────────────
    csv_path = os.path.join(output_dir, 'munich_change_scores.csv')
    df.to_csv(csv_path, index=False)
    log.info(f'✅ A) CSV saved: {csv_path}')

    # ── B) Change classification ───────────────────────────────
    def classify_change(score):
        if score < 0.20:
            return 'no_change'
        elif score < 0.40:
            return 'low_change'
        elif score < 0.60:
            return 'medium_change'
        else:
            return 'high_change'

    df['change_class'] = df['change_score'].apply(classify_change)

    # ── C) Statistical summary ─────────────────────────────────
    summary = {
        'total_patches':     len(df),
        'date_t1':           T1_DATE,
        'date_t2':           T2_DATE,
        'day_gap':           DAY_GAP,
        'tile':              TILE,
        'aoi':               AOI,
        'change_score': {
            'mean':   float(df['change_score'].mean()),
            'std':    float(df['change_score'].std()),
            'min':    float(df['change_score'].min()),
            'max':    float(df['change_score'].max()),
            'p25':    float(df['change_score'].quantile(0.25)),
            'median': float(df['change_score'].median()),
            'p75':    float(df['change_score'].quantile(0.75)),
            'p90':    float(df['change_score'].quantile(0.90)),
            'p95':    float(df['change_score'].quantile(0.95)),
        },
        'change_classes': df['change_class'].value_counts().to_dict(),
        'change_class_pct': (
            df['change_class'].value_counts(normalize=True) * 100
        ).round(2).to_dict(),
    }

    summary_path = os.path.join(output_dir, 'munich_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log.info(f'✅ C) Summary saved: {summary_path}')

    # Print summary
    print(f'\n{"═"*60}')
    print(f'  MUNICH CHANGE DETECTION SUMMARY')
    print(f'  T1: {T1_DATE} → T2: {T2_DATE} ({DAY_GAP} days)')
    print(f'  Total patches: {len(df):,}')
    print(f'{"═"*60}')
    print(f'  Change score statistics:')
    print(f'    Mean:   {summary["change_score"]["mean"]:.4f}')
    print(f'    Std:    {summary["change_score"]["std"]:.4f}')
    print(f'    Median: {summary["change_score"]["median"]:.4f}')
    print(f'    P90:    {summary["change_score"]["p90"]:.4f}')
    print(f'    Max:    {summary["change_score"]["max"]:.4f}')
    print(f'\n  Change classes:')
    for cls, count in summary['change_classes'].items():
        pct = summary['change_class_pct'][cls]
        print(f'    {cls:<20}: {count:>6,} patches ({pct:.1f}%)')
    print(f'{"═"*60}')

    # ── D) GeoJSON vector output ───────────────────────────────
    geometries = [
        box(row['x_min'], row['y_min'], row['x_max'], row['y_max'])
        for _, row in df.iterrows()
    ]
    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs=crs)

    # Save full GeoJSON
    geojson_path = os.path.join(output_dir, 'munich_change_map.geojson')
    gdf.to_file(geojson_path, driver='GeoJSON')
    log.info(f'✅ D) GeoJSON saved: {geojson_path}')

    # Save hotspots only (high change)
    hotspots = gdf[gdf['change_class'] == 'high_change'].copy()
    if len(hotspots) > 0:
        hotspot_path = os.path.join(output_dir, 'munich_hotspots.geojson')
        hotspots.to_file(hotspot_path, driver='GeoJSON')
        log.info(f'✅ F) Hotspots saved: {len(hotspots):,} patches → {hotspot_path}')
    else:
        log.info('ℹ️  No high-change hotspots detected')

    # Save by change class
    for cls in ['no_change', 'low_change', 'medium_change', 'high_change']:
        cls_gdf = gdf[gdf['change_class'] == cls]
        if len(cls_gdf) > 0:
            cls_path = os.path.join(output_dir, f'munich_{cls}.geojson')
            cls_gdf.to_file(cls_path, driver='GeoJSON')

    # ── E) Change score GeoTIFF at full 10m resolution ──────────────
    # Each patch fills its full 256x256 pixel footprint in the raster
    if len(df) > 0:
        # Get raster extent from all patch bounds
        x_min_all = df['x_min'].min()
        y_min_all = df['y_min'].min()
        x_max_all = df['x_max'].max()
        y_max_all = df['y_max'].max()

        # 10m pixel size — matches Sentinel-2 native resolution
        pixel_size = 10.0

        # Raster dimensions at 10m resolution
        raster_w = max(1, int(round((x_max_all - x_min_all) / pixel_size)))
        raster_h = max(1, int(round((y_max_all - y_min_all) / pixel_size)))

        log.info(f'Raster size: {raster_h} x {raster_w} pixels at 10m resolution')

        change_raster = np.full((raster_h, raster_w), np.nan, dtype=np.float32)
        class_raster  = np.full((raster_h, raster_w), 0,      dtype=np.uint8)

        class_map = {
            'no_change': 1, 'low_change': 2,
            'medium_change': 3, 'high_change': 4
        }

        raster_transform = from_bounds(
            x_min_all, y_min_all, x_max_all, y_max_all,
            raster_w, raster_h
        )

        for _, row in df.iterrows():
            # Each patch fills a 256x256 pixel block at 10m resolution
            col_start = int(round((row['x_min'] - x_min_all) / pixel_size))
            row_start = int(round((y_max_all - row['y_max']) / pixel_size))
            col_end   = col_start + PATCH_SIZE
            row_end   = row_start + PATCH_SIZE

            # Clamp to raster bounds
            col_start = max(0, col_start)
            row_start = max(0, row_start)
            col_end   = min(raster_w, col_end)
            row_end   = min(raster_h, row_end)

            if col_end > col_start and row_end > row_start:
                change_raster[row_start:row_end, col_start:col_end] = row['change_score']
                class_raster[row_start:row_end, col_start:col_end]  = class_map.get(row['change_class'], 0)

        # Save change score GeoTIFF
        tiff_path = os.path.join(output_dir, 'munich_change_score.tif')
        with rasterio.open(
            tiff_path, 'w',
            driver='GTiff',
            height=raster_h, width=raster_w,
            count=1,
            dtype='float32',
            crs=crs,
            transform=raster_transform,
            nodata=np.nan,
        ) as dst:
            dst.write(change_raster, 1)
            dst.update_tags(
                description='Change detection score (0=no change, 1=high change)',
                date_t1=T1_DATE, date_t2=T2_DATE,
                day_gap=str(DAY_GAP), model='SiamesePrithviModel',
                tile=TILE, aoi=AOI,
            )
        log.info(f'✅ E) Change score GeoTIFF saved: {tiff_path}')

        # Save change class GeoTIFF
        class_tiff_path = os.path.join(
            output_dir, 'munich_change_class.tif')
        with rasterio.open(
            class_tiff_path, 'w',
            driver='GTiff',
            height=raster_h, width=raster_w,
            count=1,
            dtype='uint8',
            crs=crs,
            transform=raster_transform,
            nodata=0,
        ) as dst:
            dst.write(class_raster, 1)
            dst.update_tags(
                classes='1=no_change,2=low_change,3=medium_change,4=high_change',
                date_t1=T1_DATE, date_t2=T2_DATE,
            )
        log.info(f'✅ E) Change class GeoTIFF saved: {class_tiff_path}')

    log.info(f'\n✅ All outputs saved to: {output_dir}')
    return summary


# ── Main ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Munich change detection inference'
    )
    parser.add_argument(
        '--model', required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--output', required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--tmp', default='/tmp/munich_scenes',
        help='Temporary directory for downloaded scenes'
    )
    parser.add_argument(
        '--batch-size', type=int, default=16,
        help='Inference batch size'
    )
    args = parser.parse_args()

    log.info('='*60)
    log.info('MUNICH CHANGE DETECTION INFERENCE')
    log.info(f'  T1: {T1_SCENE} ({T1_DATE})')
    log.info(f'  T2: {T2_SCENE} ({T2_DATE})')
    log.info(f'  Day gap: {DAY_GAP} days')
    log.info(f'  Model: {args.model}')
    log.info(f'  Output: {args.output}')
    log.info('='*60)

    # Load normalization stats
    log.info('Loading normalization stats...')
    tabular_stats  = json.load(open(TABULAR_STATS_PATH))
    band_stats     = json.load(open(BAND_STATS_PATH))
    day_gap_stats  = json.load(open(DAY_GAP_STATS_PATH))

    # Load model
    log.info(f'Loading model from {args.model}...')
    # Add repo root to path for both EC2 and Colab
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from src.training.model import SiamesePrithviModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Device: {device}')

    model = SiamesePrithviModel(
        prithvi_model_name='prithvi_eo_v1_100',
        n_tabular_features=34,
        mlp_hidden_1=256,
        mlp_hidden_2=64,
        dropout_1=0.3,
        dropout_2=0.2,
    ).to(device)

    ckpt = torch.load(args.model, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        log.info(f'✅ Model loaded — val_r2: {ckpt.get("val_r2", "?")}')
    else:
        model.load_state_dict(ckpt)
        log.info('✅ Model loaded (direct state dict)')

    model.eval()

    # Step 1: Download scenes
    log.info('\nStep 1: Downloading scenes...')
    t1_dir = download_scene(
        T1_SCENE, year=2020, month=7,
        output_dir=os.path.join(args.tmp, 'T1')
    )
    t2_dir = download_scene(
        T2_SCENE, year=2023, month=7,
        output_dir=os.path.join(args.tmp, 'T2')
    )

    # Step 2: Read bands
    log.info('\nStep 2: Reading scene bands...')
    t1_bands, t1_scl, t1_profile, t1_transform, crs = read_scene_bands(t1_dir)
    t2_bands, t2_scl, t2_profile, t2_transform, crs = read_scene_bands(t2_dir)
    log.info(f'T1 scene shape: {t1_bands.shape}')
    log.info(f'T2 scene shape: {t2_bands.shape}')

    # Step 3: Chip into patches
    log.info('\nStep 3: Chipping scenes into patches...')
    t1_patches = chip_scene(t1_bands, t1_scl, t1_transform)
    t2_patches = chip_scene(t2_bands, t2_scl, t2_transform)

    # Step 4: Run inference
    log.info('\nStep 4: Running inference...')
    results = run_inference(
        t1_patches, t2_patches, model, device,
        band_stats, tabular_stats, day_gap_stats,
        batch_size=args.batch_size
    )

    if not results:
        log.error('No results — check cloud cover or patch matching')
        return

    # Step 5: Geospatial analytics
    log.info('\nStep 5: Running geospatial analytics...')
    run_geospatial_analytics(results, args.output, crs)

    log.info('\n🎉 Munich inference complete!')
    log.info(f'   Results: {args.output}')


if __name__ == '__main__':
    main()
