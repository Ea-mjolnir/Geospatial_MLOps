"""
GeoAI MLOps Project 1 — Pair Generation Script
================================================
Generates balanced train/validation/test pair CSV files from PostGIS.

Usage:
    python src/preprocessing/generate_pairs.py

Output:
    data/pairs/train_pairs.csv
    data/pairs/val_pairs.csv
    data/pairs/test_pairs.csv

Each row in the CSV represents one (T1, T2) patch pair:
    patch_id_1  — earlier patch (T1)
    patch_id_2  — later patch (T2)
    aoi         — berlin / hamburg / brandenburg
    mgrs_grid_cell — Sentinel-2 MGRS tile identifier (spatial anchor)
    patch_row   — pixel row offset within the MGRS tile
    patch_col   — pixel column offset within the MGRS tile
    date_1      — acquisition date of T1
    date_2      — acquisition date of T2
    day_gap     — days between T1 and T2
    bucket      — A_short (7-30d) / B_medium (31-90d) / C_long (91-365d)

Pairing logic:
    Two patches are a valid pair if they share the same:
      mgrs_grid_cell + patch_row + patch_col (= same physical location)
    And have different acquisition_date values (= different timestamps)
    And the gap falls within a valid bucket (7-365 days)

Train/validation/test split:
    TRAIN:       berlin + hamburg locations (85% of their unique locations)
    VALIDATION:  berlin + hamburg locations (15% of their unique locations)
    TEST:        all brandenburg locations (held out entirely)
    Split is by LOCATION (mgrs_grid_cell + row + col), NOT by pair,
    to prevent the same physical spot appearing in both train and val.

Bucket balancing:
    Each split is balanced across the 3 time-gap buckets by capping
    each bucket to the size of the smallest one. This prevents
    long-range pairs from dominating by combinatorics.

Run this script AFTER the Airflow backfill completes (all 77 runs)
to ensure training uses the full dataset.
"""

import os
import csv
import random
import logging
import boto3
import psycopg2
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────
RANDOM_SEED    = 42
TRAIN_SPLIT    = 0.85      # 85% of berlin+hamburg locations → train
VAL_SPLIT      = 0.15      # 15% of berlin+hamburg locations → val
TRAIN_AOIS     = {'berlin', 'hamburg'}
TEST_AOI       = 'brandenburg'
BUCKETS        = {'A_short', 'B_medium', 'C_long'}
PROJECT_DIR    = os.path.expanduser('~/geoai-mlops-p1')
OUTPUT_DIR     = os.path.join(PROJECT_DIR, 'data', 'pairs')

CSV_HEADER = [
    'patch_id_1', 'patch_id_2', 'aoi', 'mgrs_grid_cell',
    'patch_row', 'patch_col', 'date_1', 'date_2', 'day_gap', 'bucket'
]

PAIRING_QUERY = """
    WITH patch_locations AS (
        SELECT
            patch_id,
            tile_id,
            split_part(tile_id, chr(95), 2) AS mgrs_grid_cell,
            patch_row,
            patch_col,
            acquisition_date,
            aoi
        FROM sentinel2_patches
        WHERE acquisition_date IS NOT NULL
    ),
    pairs AS (
        SELECT
            a.patch_id        AS patch_id_1,
            b.patch_id        AS patch_id_2,
            a.aoi,
            a.mgrs_grid_cell,
            a.patch_row,
            a.patch_col,
            a.acquisition_date AS date_1,
            b.acquisition_date AS date_2,
            (b.acquisition_date - a.acquisition_date) AS day_gap,
            CASE
                WHEN (b.acquisition_date - a.acquisition_date)
                     BETWEEN 7  AND 30  THEN 'A_short'
                WHEN (b.acquisition_date - a.acquisition_date)
                     BETWEEN 31 AND 90  THEN 'B_medium'
                WHEN (b.acquisition_date - a.acquisition_date)
                     BETWEEN 91 AND 365 THEN 'C_long'
                ELSE NULL
            END AS bucket
        FROM patch_locations a
        JOIN patch_locations b
          ON  a.mgrs_grid_cell   = b.mgrs_grid_cell
         AND  a.patch_row        = b.patch_row
         AND  a.patch_col        = b.patch_col
         AND  a.acquisition_date < b.acquisition_date
    )
    SELECT
        patch_id_1, patch_id_2, aoi, mgrs_grid_cell,
        patch_row, patch_col, date_1, date_2, day_gap, bucket
    FROM pairs
    WHERE bucket IS NOT NULL
    ORDER BY aoi, mgrs_grid_cell, patch_row, patch_col, date_1, date_2;
"""


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


def fetch_all_pairs(cur):
    """Run the pairing query and return all valid pairs."""
    log.info("Running pairing query on PostGIS...")
    cur.execute(PAIRING_QUERY)
    pairs = cur.fetchall()
    log.info(f"Total valid pairs fetched: {len(pairs):,}")
    return pairs


def split_by_aoi(pairs):
    """Separate pairs into train+val (berlin/hamburg) and test (brandenburg)."""
    train_val = [p for p in pairs if p[2] in TRAIN_AOIS]
    test      = [p for p in pairs if p[2] == TEST_AOI]
    log.info(f"Train+val pairs (berlin+hamburg): {len(train_val):,}")
    log.info(f"Test pairs (brandenburg):          {len(test):,}")
    return train_val, test


def split_by_location(train_val_pairs, seed=RANDOM_SEED):
    """
    Split train_val pairs into train and validation by UNIQUE LOCATION.
    Location = (mgrs_grid_cell, patch_row, patch_col)
    Splitting by location (not pair) prevents the same physical spot
    appearing in both train and validation sets (spatial leakage).
    """
    random.seed(seed)
    locations  = list(set((p[3], p[4], p[5]) for p in train_val_pairs))
    random.shuffle(locations)
    split_idx  = int(len(locations) * TRAIN_SPLIT)
    train_locs = set(locations[:split_idx])
    val_locs   = set(locations[split_idx:])

    train = [p for p in train_val_pairs if (p[3], p[4], p[5]) in train_locs]
    val   = [p for p in train_val_pairs if (p[3], p[4], p[5]) in val_locs]

    log.info(f"Train: {len(train):,} pairs across {len(train_locs):,} locations")
    log.info(f"Val:   {len(val):,} pairs across {len(val_locs):,} locations")
    return train, val


def balance_buckets(pairs, split_name, seed=RANDOM_SEED):
    """
    Balance pairs across the 3 time-gap buckets by capping each
    bucket to the size of the smallest one.
    Prevents long-range pairs dominating by combinatorics.
    """
    random.seed(seed)
    buckets = {b: [] for b in BUCKETS}
    for p in pairs:
        if p[9] in buckets:
            buckets[p[9]].append(p)

    for b, bpairs in buckets.items():
        log.info(f"  {split_name} {b}: {len(bpairs):,} pairs")

    cap = min(len(v) for v in buckets.values())
    log.info(f"  {split_name} cap per bucket: {cap:,} → total: {cap * 3:,}")

    balanced = []
    for b, bpairs in buckets.items():
        random.shuffle(bpairs)
        balanced.extend(bpairs[:cap])

    random.shuffle(balanced)
    return balanced


def save_csv(pairs, split_name):
    """Save pairs to a CSV file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f'{split_name}_pairs.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(pairs)
    log.info(f"✅ Saved {split_name}_pairs.csv: {len(pairs):,} pairs → {path}")
    return path


def print_summary(train, val, test):
    """Print a final summary of the generated splits."""
    log.info("=" * 60)
    log.info("PAIR GENERATION COMPLETE")
    log.info("=" * 60)
    for split, pairs in [('TRAIN', train), ('VAL', val), ('TEST', test)]:
        buckets = {}
        for p in pairs:
            buckets[p[9]] = buckets.get(p[9], 0) + 1
        log.info(
            f"  {split:<6}: {len(pairs):>7,} pairs | "
            f"A={buckets.get('A_short',0):,} "
            f"B={buckets.get('B_medium',0):,} "
            f"C={buckets.get('C_long',0):,}"
        )
    log.info(f"  Output: {OUTPUT_DIR}/")
    log.info("=" * 60)


def main():
    log.info("Starting pair generation...")
    log.info(f"Random seed: {RANDOM_SEED}")
    log.info(f"Train AOIs:  {TRAIN_AOIS}")
    log.info(f"Test AOI:    {TEST_AOI}")
    log.info(f"Train split: {TRAIN_SPLIT*100:.0f}% of locations")

    # Connect to PostGIS
    conn = get_db_connection()
    cur  = conn.cursor()

    # Fetch all valid pairs
    all_pairs = fetch_all_pairs(cur)
    cur.close()
    conn.close()

    if not all_pairs:
        log.error("No pairs found — is the backfill complete?")
        return

    # Split by AOI
    train_val_pairs, test_pairs = split_by_aoi(all_pairs)

    # Split train_val by location
    train_pairs, val_pairs = split_by_location(train_val_pairs)

    # Balance buckets within each split
    log.info("Balancing TRAIN buckets...")
    train_balanced = balance_buckets(train_pairs, 'TRAIN')
    log.info("Balancing VAL buckets...")
    val_balanced   = balance_buckets(val_pairs,   'VAL')
    log.info("Balancing TEST buckets...")
    test_balanced  = balance_buckets(test_pairs,  'TEST')

    # Save CSVs
    save_csv(train_balanced, 'train')
    save_csv(val_balanced,   'val')
    save_csv(test_balanced,  'test')

    # Summary
    print_summary(train_balanced, val_balanced, test_balanced)


if __name__ == '__main__':
    main()
