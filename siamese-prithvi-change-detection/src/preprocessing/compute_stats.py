"""
GeoAI MLOps Project 1 — Standardization Statistics Script
==========================================================
Computes mean and standard deviation for all features used in training,
using TRAINING DATA ONLY (berlin + hamburg patches from train_pairs.csv).

CRITICAL: stats are computed from TRAIN only — never from validation or
test data. Using test/val data here would constitute data leakage,
making test performance optimistically biased.

These saved stats serve two purposes:
  1. Training:   standardize features on the fly in the Dataset class
                 formula: (value - mean) / std
  2. Inference:  standardize new patches using the SAME ruler the model
                 was trained with — without this, model output is garbage

Usage:
    python src/preprocessing/compute_stats.py

Prerequisites:
    data/pairs/train_pairs.csv must exist (run generate_pairs.py first)

Output:
    data/stats/tabular_stats.json   mean/std for all 34 tabular features
    data/stats/band_stats.json      mean/std for all 5 imagery bands
    data/stats/day_gap_stats.json   mean/std for the day_gap scalar
    data/stats/stats_summary.json   human-readable summary for inspection
"""

import os
import json
import logging
import csv
import boto3
import psycopg2
import numpy as np
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────
PROJECT_DIR  = os.path.expanduser('~/geoai-mlops-p1')
PAIRS_DIR    = os.path.join(PROJECT_DIR, 'data', 'pairs')
OUTPUT_DIR   = os.path.join(PROJECT_DIR, 'data', 'stats')
TRAIN_CSV    = os.path.join(PAIRS_DIR, 'train_pairs.csv')

# The 34 tabular features fed into the model as input
# (excludes ndvi_mean, ndwi_mean, ndbi_mean — used only for target)
TABULAR_FEATURES = [
    'ndvi_std', 'ndwi_std', 'mndwi_mean', 'ndbi_std',
    'arvi_mean', 'evi_mean', 'bsi_mean', 'nbi_mean',
    'savi_mean', 'bright_mean',
    'blue_mean',  'blue_std',  'blue_p25',  'blue_p75',
    'green_mean', 'green_std', 'green_p25', 'green_p75',
    'red_mean',   'red_std',   'red_p25',   'red_p75',
    'nir_mean',   'nir_std',   'nir_p25',   'nir_p75',
    'swir_mean',  'swir_std',  'swir_p25',  'swir_p75',
    'valid_pixel_ratio', 'cloud_cover',
    'total_pixels', 'valid_pixels',
]

# The 5 imagery bands in our patches (order matters — must match
# how the Dataset class loads the GeoTIFF bands)
IMAGERY_BANDS = ['blue', 'green', 'red', 'nir', 'swir']

# Band name → PostGIS column for representative stats
# We use the per-band mean columns as a proxy for imagery pixel stats
# (actual per-pixel stats would require loading all S3 GeoTIFFs —
#  PostGIS band means are a good, fast approximation for standardization)
BAND_PROXY_COLS = {
    'blue':  'blue_mean',
    'green': 'green_mean',
    'red':   'red_mean',
    'nir':   'nir_mean',
    'swir':  'swir_mean',
}


def get_db_connection():
    """Connect to PostGIS using SSM parameters."""
    ssm     = boto3.client('ssm', region_name='us-east-1')
    db_host = ssm.get_parameter(
        Name='/geoai-mlops-p1/db/host'
    )['Parameter']['Value']
    db_user = ssm.get_parameter(
        Name='/geoai-mlops-p1/db/user'
    )['Parameter']['Value']
    db_pass = ssm.get_parameter(
        Name='/geoai-mlops-p1/db/password', WithDecryption=True
    )['Parameter']['Value']
    return psycopg2.connect(
        host=db_host, port=5432,
        database='geoai_features',
        user=db_user, password=db_pass,
        connect_timeout=30
    )


def load_train_patch_ids():
    """
    Load all unique patch_ids from train_pairs.csv.
    Both patch_id_1 and patch_id_2 are included since both
    are training patches that need to be standardized.
    """
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(
            f"Train pairs CSV not found: {TRAIN_CSV}\n"
            "Run generate_pairs.py first."
        )

    patch_ids = set()
    with open(TRAIN_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            patch_ids.add(row['patch_id_1'])
            patch_ids.add(row['patch_id_2'])

    log.info(f"Unique training patch_ids loaded: {len(patch_ids):,}")
    return list(patch_ids)


def load_day_gaps():
    """Load day_gap values from train_pairs.csv for day_gap stats."""
    day_gaps = []
    with open(TRAIN_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            day_gaps.append(float(row['day_gap']))
    log.info(f"Day gap values loaded: {len(day_gaps):,}")
    return day_gaps


def fetch_tabular_features(cur, patch_ids):
    """
    Fetch all 34 tabular feature values for training patch_ids.
    Uses batched IN queries to avoid hitting PostgreSQL parameter limits.
    """
    log.info(f"Fetching tabular features for {len(patch_ids):,} patches...")

    cols    = ', '.join(TABULAR_FEATURES)
    values  = defaultdict(list)
    batch   = 5000  # PostgreSQL IN clause limit safety margin

    for i in range(0, len(patch_ids), batch):
        batch_ids = patch_ids[i:i + batch]
        placeholders = ','.join(['%s'] * len(batch_ids))
        cur.execute(
            f"SELECT {cols} FROM sentinel2_patches "
            f"WHERE patch_id IN ({placeholders})",
            batch_ids
        )
        rows = cur.fetchall()
        for row in rows:
            for j, col in enumerate(TABULAR_FEATURES):
                val = row[j]
                if val is not None:
                    values[col].append(float(val))

        log.info(
            f"  Batch {i//batch + 1}: fetched {len(rows):,} rows "
            f"(total so far: {len(list(values.values())[0]) if values else 0:,})"
        )

    return values


def compute_stats(values_dict):
    """
    Compute mean and std for each feature.
    Uses np.std with ddof=0 (population std) — standard for
    feature standardization (not sample std with ddof=1).
    """
    stats = {}
    for feature, values in values_dict.items():
        arr  = np.array(values, dtype=np.float64)
        mean = float(np.mean(arr))
        std  = float(np.std(arr, ddof=0))

        # Guard against zero std (constant feature — rare but possible)
        # Replace with 1.0 so standardization doesn't divide by zero
        if std < 1e-8:
            log.warning(
                f"Feature '{feature}' has near-zero std ({std:.2e}) — "
                f"setting std=1.0 to prevent division by zero"
            )
            std = 1.0

        stats[feature] = {
            'mean': mean,
            'std':  std,
            'min':  float(np.min(arr)),
            'max':  float(np.max(arr)),
            'n':    len(values),
        }
    return stats


def compute_band_stats(tabular_stats):
    """
    Derive imagery band stats from the PostGIS band mean columns.
    These are used to standardize the 256x256 patch pixel values
    before feeding them through the Prithvi ViT encoder.

    We use band_mean columns as a proxy for the actual pixel distribution.
    This is a fast approximation — computing true per-pixel stats would
    require loading all ~52GB of S3 GeoTIFFs, which is impractical here.
    The approximation is good enough for standardization purposes since
    the band_mean values reflect the same underlying pixel distributions.
    """
    band_stats = {}
    for band, proxy_col in BAND_PROXY_COLS.items():
        if proxy_col in tabular_stats:
            band_stats[band] = {
                'mean': tabular_stats[proxy_col]['mean'],
                'std':  tabular_stats[proxy_col]['std'],
                'note': f'derived from PostGIS {proxy_col} column',
            }
            log.info(
                f"  Band {band}: mean={band_stats[band]['mean']:.4f} "
                f"std={band_stats[band]['std']:.4f}"
            )
        else:
            log.warning(f"Band {band}: proxy column {proxy_col} not found")

    return band_stats


def save_stats(tabular_stats, band_stats, day_gap_stats):
    """Save all stats to JSON files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Tabular stats (34 features)
    tabular_path = os.path.join(OUTPUT_DIR, 'tabular_stats.json')
    with open(tabular_path, 'w') as f:
        json.dump(tabular_stats, f, indent=2)
    log.info(f"✅ tabular_stats.json saved: {len(tabular_stats)} features")

    # Band stats (5 imagery bands)
    band_path = os.path.join(OUTPUT_DIR, 'band_stats.json')
    with open(band_path, 'w') as f:
        json.dump(band_stats, f, indent=2)
    log.info(f"✅ band_stats.json saved: {len(band_stats)} bands")

    # Day gap stats (1 scalar)
    day_gap_path = os.path.join(OUTPUT_DIR, 'day_gap_stats.json')
    with open(day_gap_path, 'w') as f:
        json.dump(day_gap_stats, f, indent=2)
    log.info(f"✅ day_gap_stats.json saved")

    # Human-readable summary
    summary = {
        'computed_from':    'TRAIN split only (berlin + hamburg)',
        'n_training_pairs': day_gap_stats['n'],
        'tabular_features': {
            k: {'mean': round(v['mean'], 6), 'std': round(v['std'], 6)}
            for k, v in tabular_stats.items()
        },
        'imagery_bands': {
            k: {'mean': round(v['mean'], 6), 'std': round(v['std'], 6)}
            for k, v in band_stats.items()
        },
        'day_gap': {
            'mean': round(day_gap_stats['mean'], 2),
            'std':  round(day_gap_stats['std'], 2),
            'min':  round(day_gap_stats['min'], 0),
            'max':  round(day_gap_stats['max'], 0),
        },
        'usage': (
            'Apply at both training AND inference time: '
            '(value - mean) / std for each feature. '
            'Never recompute from val/test data.'
        ),
    }
    summary_path = os.path.join(OUTPUT_DIR, 'stats_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log.info(f"✅ stats_summary.json saved")

    return tabular_path, band_path, day_gap_path, summary_path


def main():
    log.info("=" * 60)
    log.info("COMPUTING STANDARDIZATION STATISTICS")
    log.info("Source: TRAIN split only (berlin + hamburg)")
    log.info("=" * 60)

    # Load training patch IDs from the generated CSV
    patch_ids = load_train_patch_ids()

    # Load day gap values
    day_gaps = load_day_gaps()

    # Connect to PostGIS
    conn = get_db_connection()
    cur  = conn.cursor()

    # Fetch all 34 tabular feature values for training patches
    tabular_values = fetch_tabular_features(cur, patch_ids)
    cur.close()
    conn.close()

    log.info(
        f"Features fetched: {len(tabular_values)} columns, "
        f"~{len(list(tabular_values.values())[0]):,} values each"
    )

    # Compute tabular stats (mean + std per feature)
    log.info("Computing tabular feature stats...")
    tabular_stats = compute_stats(tabular_values)

    # Derive band stats from tabular proxies
    log.info("Deriving imagery band stats...")
    band_stats = compute_band_stats(tabular_stats)

    # Compute day gap stats
    log.info("Computing day_gap stats...")
    day_gap_arr  = np.array(day_gaps, dtype=np.float64)
    day_gap_stats = {
        'mean': float(np.mean(day_gap_arr)),
        'std':  float(np.std(day_gap_arr, ddof=0)),
        'min':  float(np.min(day_gap_arr)),
        'max':  float(np.max(day_gap_arr)),
        'n':    len(day_gaps),
    }
    log.info(
        f"  day_gap: mean={day_gap_stats['mean']:.1f}d "
        f"std={day_gap_stats['std']:.1f}d "
        f"min={day_gap_stats['min']:.0f}d "
        f"max={day_gap_stats['max']:.0f}d"
    )

    # Save all stats to JSON files
    paths = save_stats(tabular_stats, band_stats, day_gap_stats)

    log.info("=" * 60)
    log.info("STANDARDIZATION STATS COMPLETE")
    log.info(f"  Tabular:  {len(tabular_stats)} features")
    log.info(f"  Imagery:  {len(band_stats)} bands")
    log.info(f"  day_gap:  1 scalar")
    log.info(f"  Output:   {OUTPUT_DIR}/")
    log.info("  These stats MUST be used at inference time too.")
    log.info("=" * 60)


if __name__ == '__main__':
    main()
