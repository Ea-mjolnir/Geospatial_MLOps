"""
GeoAI MLOps Project 1 — PyTorch Dataset Class
===============================================
SentinelPairDataset: loads (T1, T2) patch pairs for Siamese Prithvi training.

For each pair the Dataset returns:
    patch_t1:   torch.Tensor [5, 256, 256]  standardized imagery for T1
    patch_t2:   torch.Tensor [5, 256, 256]  standardized imagery for T2
    tabular_t1: torch.Tensor [34]           standardized tabular features for T1
    tabular_t2: torch.Tensor [34]           standardized tabular features for T2
    day_gap:    torch.Tensor [1]            standardized days between T1 and T2
    target:     torch.Tensor [1]            weak supervision change magnitude
                                            = sqrt(d_ndvi² + d_ndwi² + d_ndbi²)

Usage:
    from src.training.dataset import SentinelPairDataset

    # With features cache (no PostGIS needed — fast)
    train_ds = SentinelPairDataset(
        pairs_csv='data/pairs/train_pairs.csv',
        tabular_stats_path='data/stats/tabular_stats.json',
        band_stats_path='data/stats/band_stats.json',
        day_gap_stats_path='data/stats/day_gap_stats.json',
        patches_s3_bucket='geoai-mlops-p1-data-288528696055',
        patches_s3_prefix='processed/patches',
        features_cache='data/features/features_cache.parquet',
        patches_local_dir='/content/patches',  # optional local SSD
        max_pairs=15000,                        # optional subsample
    )

    # Without features cache (uses PostGIS — slower)
    train_ds = SentinelPairDataset(
        pairs_csv='data/pairs/train_pairs.csv',
        ...
        db_host=..., db_user=..., db_pass=...,
    )

Design decisions:
    - Features cache (parquet): preferred path — loads all features in <1s
      No PostGIS queries during training — fastest approach
    - PostGIS fallback: used when no features_cache provided
      Fetches all features in one batch query at init, cached in memory
    - Imagery: loaded from local SSD if patches_local_dir provided,
      otherwise fetched from S3 on the fly per sample
    - Standardization: applied on the fly using pre-computed stats
    - Target: computed fresh per pair from ndvi/ndwi/ndbi deltas
    - Band order: blue=0, green=1, red=2, nir=3, swir=4 (consistent)
"""
import os
import json
import csv
import io
import logging
import math
import numpy as np
import torch
from torch.utils.data import Dataset
import boto3
import psycopg2
import rasterio

log = logging.getLogger(__name__)

# Band order — must match how generate_pairs and ingestion pipeline saved bands
BAND_ORDER = ['blue', 'green', 'red', 'nir', 'swir']

# The 34 tabular features in the exact order fed to the model
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

# Features needed ONLY for target computation — not fed to model
TARGET_FEATURES = ['ndvi_mean', 'ndwi_mean', 'ndbi_mean']

# All features to fetch from PostGIS per patch (tabular + target)
ALL_DB_FEATURES = TABULAR_FEATURES + TARGET_FEATURES


class SentinelPairDataset(Dataset):
    """
    PyTorch Dataset for Siamese Prithvi change detection training.

    Initialization (fast path — features cache):
        Loads pair list from CSV.
        Loads 15MB parquet cache into memory (<1 second).
        Builds tabular normalization arrays from tabular_stats.json.
        No PostGIS queries during training.

    Initialization (slow path — PostGIS):
        Loads pair list from CSV.
        Fetches ALL tabular features for ALL patches from PostGIS
        in one batch query and caches them in memory.

    __getitem__:
        Loads 2 patch GeoTIFFs from local SSD or S3 (5 bands each).
        Applies standardization to imagery and tabular features.
        Computes the weak supervision target fresh from the pair.
        Returns tensors ready for the model.
    """

    def __init__(
        self,
        pairs_csv,
        tabular_stats_path,
        band_stats_path,
        day_gap_stats_path,
        patches_s3_bucket,
        patches_s3_prefix,
        db_host=None,
        db_user=None,
        db_pass=None,
        db_name='geoai_features',
        patch_size=256,
        augment=False,
        max_pairs=None,
        pairs_offset=0,
        patches_local_dir=None,
        features_cache=None,
    ):
        """
        Args:
            pairs_csv:           path to train/val/test_pairs.csv
            tabular_stats_path:  path to tabular_stats.json
            band_stats_path:     path to band_stats.json
            day_gap_stats_path:  path to day_gap_stats.json
            patches_s3_bucket:   S3 bucket name
            patches_s3_prefix:   S3 prefix for patches
            db_host/user/pass:   PostGIS credentials (optional if features_cache)
            db_name:             PostGIS database name
            patch_size:          expected patch size in pixels (default 256)
            augment:             apply random augmentation (train only)
            max_pairs:           subsample to this many pairs (optional)
            patches_local_dir:   local SSD path for patches (faster than S3)
            features_cache:      path to parquet features cache (fastest path)
        """
        self.pairs_csv         = pairs_csv
        self.patches_s3_bucket = patches_s3_bucket
        self.patches_s3_prefix = patches_s3_prefix.rstrip('/')
        self.patch_size        = patch_size
        self.augment           = augment
        self.patches_local_dir = patches_local_dir
        self.features_cache    = features_cache

        # Load features from parquet cache if available (fast path)
        if features_cache and os.path.exists(features_cache):
            import pandas as pd
            log.info(f"Loading features cache: {features_cache}")
            self._features_df = pd.read_parquet(features_cache)
            self._features_df = self._features_df.set_index('patch_id')
            log.info(f"✅ Features cache: {len(self._features_df):,} patches")
        else:
            self._features_df = None
            log.info("No features cache — will query PostGIS")

        # Load standardization stats
        log.info("Loading standardization stats...")
        self.tabular_stats  = self._load_json(tabular_stats_path)
        self.band_stats     = self._load_json(band_stats_path)
        self.day_gap_stats  = self._load_json(day_gap_stats_path)

        # Build normalization arrays from tabular_stats
        # Shape: [34] — one mean/std per tabular feature
        self._tabular_mean = np.array(
            [self.tabular_stats[f]['mean'] for f in TABULAR_FEATURES],
            dtype=np.float32
        )
        self._tabular_std = np.array(
            [self.tabular_stats[f]['std']  for f in TABULAR_FEATURES],
            dtype=np.float32
        )
        log.info(f"✅ Tabular normalization built: {len(self._tabular_mean)} features")

        # Load pair list from CSV
        log.info(f"Loading pairs from {pairs_csv}...")
        self.pairs = self._load_pairs(pairs_csv)

        # Apply deterministic slice for incremental training
        if pairs_offset > 0 or max_pairs:
            start = pairs_offset or 0
            end   = (start + max_pairs) if max_pairs else len(self.pairs)
            self.pairs = self.pairs[start:end]
        log.info(f"Loaded {len(self.pairs):,} pairs (offset={pairs_offset}, max={max_pairs})")

        # Fetch tabular features — from parquet cache or PostGIS
        if self._features_df is not None:
            log.info("Using parquet features cache — skipping PostGIS")
            self.patch_features = {}  # not used when cache available
        else:
            log.info("Fetching tabular features from PostGIS (batch, cached)...")
            self.patch_features = self._fetch_all_tabular_features(
                db_host, db_user, db_pass, db_name
            )
            log.info(
                f"Cached features for {len(self.patch_features):,} unique patches"
            )

        # S3 client (shared, thread-safe for reads)
        session  = boto3.Session(region_name='us-east-1')
        self.s3  = session.client('s3')

    # ── Private helpers ────────────────────────────────────────────

    @staticmethod
    def _load_json(path):
        with open(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def _load_pairs(csv_path):
        pairs = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append({
                    'patch_id_1': row['patch_id_1'],
                    'patch_id_2': row['patch_id_2'],
                    'day_gap':    float(row['day_gap']),
                    'bucket':     row['bucket'],
                    'aoi':        row['aoi'],
                })
        return pairs

    def _fetch_all_tabular_features(self, db_host, db_user, db_pass, db_name):
        """
        Fetch all tabular + target features for every patch_id
        referenced in the pairs CSV, in one batched query.
        Cached in self.patch_features dict for O(1) access per sample.
        """
        patch_ids = set()
        for pair in self.pairs:
            patch_ids.add(pair['patch_id_1'])
            patch_ids.add(pair['patch_id_2'])
        patch_ids = list(patch_ids)

        conn = psycopg2.connect(
            host=db_host, port=5432, database=db_name,
            user=db_user, password=db_pass,
            connect_timeout=30
        )
        cur   = conn.cursor()
        cols  = ', '.join(ALL_DB_FEATURES)
        cache = {}
        batch = 5000

        for i in range(0, len(patch_ids), batch):
            batch_ids    = patch_ids[i:i + batch]
            placeholders = ', '.join(['%s'] * len(batch_ids))
            cur.execute(
                f"SELECT patch_id, {cols} FROM sentinel2_patches "
                f"WHERE patch_id IN ({placeholders})",
                batch_ids
            )
            for row in cur.fetchall():
                pid  = row[0]
                vals = row[1:]
                cache[pid] = {
                    col: float(vals[j]) if vals[j] is not None else 0.0
                    for j, col in enumerate(ALL_DB_FEATURES)
                }

        cur.close()
        conn.close()
        return cache

    def _get_features_from_cache(self, patch_id):
        """
        Get standardized tabular features from parquet cache.
        Returns torch.Tensor [34] — no PostGIS query needed.
        """
        try:
            row      = self._features_df.loc[patch_id]
            features = row[TABULAR_FEATURES].values.astype(np.float32)
            features = (features - self._tabular_mean) / self._tabular_std
            return torch.tensor(features, dtype=torch.float32)
        except KeyError:
            log.warning(f"patch_id {patch_id} not in features cache — zeros")
            return torch.zeros(len(TABULAR_FEATURES), dtype=torch.float32)

    def _standardize_tabular(self, patch_id):
        """
        Return standardized tabular features for one patch (PostGIS path).
        Shape: [34]
        """
        feat = self.patch_features.get(patch_id, {})
        arr  = np.zeros(len(TABULAR_FEATURES), dtype=np.float32)
        for i, col in enumerate(TABULAR_FEATURES):
            raw    = feat.get(col, 0.0)
            arr[i] = (raw - self._tabular_mean[i]) / self._tabular_std[i]
        return arr

    def _compute_target(self, patch_id_1, patch_id_2):
        """
        Compute weak supervision target fresh from the true pair.
        target = sqrt(d_ndvi² + d_ndwi² + d_ndbi²)

        Uses ndvi_mean, ndwi_mean, ndbi_mean from parquet cache or PostGIS.
        These are EXCLUDED from tabular features — not fed to model.
        """
        if self._features_df is not None:
            try:
                f1     = self._features_df.loc[patch_id_1]
                f2     = self._features_df.loc[patch_id_2]
                d_ndvi = float(f2['ndvi_mean']) - float(f1['ndvi_mean'])
                d_ndwi = float(f2['ndwi_mean']) - float(f1['ndwi_mean'])
                d_ndbi = float(f2['ndbi_mean']) - float(f1['ndbi_mean'])
            except KeyError:
                d_ndvi, d_ndwi, d_ndbi = 0.0, 0.0, 0.0
        else:
            f1     = self.patch_features.get(patch_id_1, {})
            f2     = self.patch_features.get(patch_id_2, {})
            d_ndvi = f2.get('ndvi_mean', 0.0) - f1.get('ndvi_mean', 0.0)
            d_ndwi = f2.get('ndwi_mean', 0.0) - f1.get('ndwi_mean', 0.0)
            d_ndbi = f2.get('ndbi_mean', 0.0) - f1.get('ndbi_mean', 0.0)

        return math.sqrt(d_ndvi**2 + d_ndwi**2 + d_ndbi**2)

    def _standardize_day_gap(self, day_gap):
        """Standardize the day_gap scalar."""
        mean = self.day_gap_stats['mean']
        std  = self.day_gap_stats['std']
        return (day_gap - mean) / std

    def _load_patch_from_local(self, patch_id):
        """
        Load a 5-band patch from local SSD or Google Drive.
        Shape: [5, 256, 256]
        Much faster than S3 — use when patches_local_dir is set.
        """
        parts      = patch_id.split('__')
        tile_id    = parts[0]
        patch_name = parts[1] if len(parts) > 1 else patch_id
        arrays = []
        for band in BAND_ORDER:
            local_path = os.path.join(
                self.patches_local_dir, tile_id, patch_name, f'{band}.tif'
            )
            try:
                with rasterio.open(local_path) as src:
                    arr = src.read(1).astype(np.float32)
            except Exception as e:
                log.warning(f"Failed to load {local_path}: {e} — zeros")
                arr = np.zeros(
                    (self.patch_size, self.patch_size), dtype=np.float32
                )
            arrays.append(arr)
        return np.stack(arrays, axis=0)

    def _load_patch_from_s3(self, patch_id):
        """
        Load a 5-band patch from S3.
        Shape: [5, 256, 256]
        S3 path: s3://bucket/processed/patches/<tile_id>/<patch_name>/<band>.tif
        """
        parts      = patch_id.split('__')
        tile_id    = parts[0]
        patch_name = parts[1] if len(parts) > 1 else patch_id
        arrays = []
        for band in BAND_ORDER:
            s3_key = (
                f"{self.patches_s3_prefix}/{tile_id}"
                f"/{patch_name}/{band}.tif"
            )
            try:
                obj = self.s3.get_object(
                    Bucket=self.patches_s3_bucket,
                    Key=s3_key
                )
                buf = io.BytesIO(obj['Body'].read())
                with rasterio.open(buf) as src:
                    arr = src.read(1).astype(np.float32)
            except Exception as e:
                log.warning(f"Failed to load {s3_key}: {e} — zeros")
                arr = np.zeros(
                    (self.patch_size, self.patch_size), dtype=np.float32
                )
            arrays.append(arr)
        return np.stack(arrays, axis=0)

    def _standardize_patch(self, patch):
        """
        Standardize imagery patch using per-band stats.
        Input:  numpy array [5, H, W]
        Output: numpy array [5, H, W] standardized
        """
        for i, band in enumerate(BAND_ORDER):
            mean     = self.band_stats[band]['mean']
            std      = self.band_stats[band]['std']
            patch[i] = (patch[i] - mean) / std
        return patch

    def _augment_pair(self, patch_t1, patch_t2):
        """
        Apply consistent random augmentation to both patches of a pair.
        SAME transformation applied to T1 and T2 — preserves change signal.
        Only applied during training (self.augment=True).
        """
        if np.random.random() > 0.5:
            patch_t1 = np.flip(patch_t1, axis=2).copy()
            patch_t2 = np.flip(patch_t2, axis=2).copy()
        if np.random.random() > 0.5:
            patch_t1 = np.flip(patch_t1, axis=1).copy()
            patch_t2 = np.flip(patch_t2, axis=1).copy()
        return patch_t1, patch_t2

    # ── Public interface ───────────────────────────────────────────

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Returns one training sample as a dict of tensors.

        Returns:
            patch_t1:   FloatTensor [5, 256, 256]
            patch_t2:   FloatTensor [5, 256, 256]
            tabular_t1: FloatTensor [34]
            tabular_t2: FloatTensor [34]
            day_gap:    FloatTensor [1]
            target:     FloatTensor [1]
            meta:       dict with patch_ids, aoi, bucket
        """
        pair = self.pairs[idx]
        pid1 = pair['patch_id_1']
        pid2 = pair['patch_id_2']

        # Load imagery from local SSD or S3
        if self.patches_local_dir:
            patch_t1 = self._load_patch_from_local(pid1)
            patch_t2 = self._load_patch_from_local(pid2)
        else:
            patch_t1 = self._load_patch_from_s3(pid1)
            patch_t2 = self._load_patch_from_s3(pid2)

        # Standardize imagery
        patch_t1 = self._standardize_patch(patch_t1)
        patch_t2 = self._standardize_patch(patch_t2)

        # Apply augmentation (train only)
        if self.augment:
            patch_t1, patch_t2 = self._augment_pair(patch_t1, patch_t2)

        # Load tabular features from cache or PostGIS
        if self._features_df is not None:
            tabular_t1 = self._get_features_from_cache(pid1).numpy()
            tabular_t2 = self._get_features_from_cache(pid2).numpy()
        else:
            tabular_t1 = self._standardize_tabular(pid1)
            tabular_t2 = self._standardize_tabular(pid2)

        # Standardize day_gap
        day_gap_std = self._standardize_day_gap(pair['day_gap'])

        # Compute weak supervision target
        target = self._compute_target(pid1, pid2)

        return {
            'patch_t1':   torch.from_numpy(patch_t1),
            'patch_t2':   torch.from_numpy(patch_t2),
            'tabular_t1': torch.from_numpy(tabular_t1),
            'tabular_t2': torch.from_numpy(tabular_t2),
            'day_gap':    torch.tensor([day_gap_std], dtype=torch.float32),
            'target':     torch.tensor([target],      dtype=torch.float32),
            'meta': {
                'patch_id_1':  pid1,
                'patch_id_2':  pid2,
                'aoi':         pair['aoi'],
                'bucket':      pair['bucket'],
                'day_gap_raw': pair['day_gap'],
            }
        }

    def get_split_info(self):
        """Return summary statistics about this dataset split."""
        buckets = {}
        aois    = {}
        for pair in self.pairs:
            buckets[pair['bucket']] = buckets.get(pair['bucket'], 0) + 1
            aois[pair['aoi']]       = aois.get(pair['aoi'], 0) + 1
        return {
            'n_pairs': len(self.pairs),
            'buckets': buckets,
            'aois':    aois,
            'augment': self.augment,
        }
