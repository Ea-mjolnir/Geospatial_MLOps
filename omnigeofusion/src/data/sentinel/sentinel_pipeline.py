"""
OmniGeoFusion — Sentinel Data Pipeline
=======================================
Downloads and preprocesses Sentinel-2 and Sentinel-1
data for the Netherlands study area.

Sources:
  Sentinel-2: s3://sentinel-cogs/ (public, no-sign-request)
  Sentinel-1: Copernicus Data Space Ecosystem

Outputs:
  Preprocessed patches saved to S3:
  s3://omnigeofusion-data/netherlands/sentinel2/
  s3://omnigeofusion-data/netherlands/sentinel1/
"""

import os
import json
import logging
import numpy as np
import boto3
import rasterio
import yaml

from pathlib import Path
from datetime import datetime, timedelta
from botocore import UNSIGNED
from botocore.config import Config
from rasterio.enums import Resampling
from typing import List, Dict, Tuple, Optional

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────
SENTINEL2_BUCKET = 'sentinel-cogs'
SENTINEL2_PREFIX = 'sentinel-s2-l2a-cogs'

# Netherlands MGRS tiles
NL_TILES = {
    'amsterdam':  ['31UFT', '31UFU'],
    'rotterdam':  ['31UFT', '31UFS'],
    'flevoland':  ['31UFU', '31UGU'],
    'nationwide': ['31UFT', '31UFU', '31UFS', '31UGU',
                   '31UGT', '32ULC', '32UMC', '32ULD']
}

# Sentinel-2 bands
S2_BANDS    = ['B02', 'B03', 'B04', 'B08', 'B11', 'B8A']
S2_BAND_RES = {
    'B02': 10, 'B03': 10, 'B04': 10,
    'B08': 10, 'B11': 20, 'B8A': 20
}

# Sentinel-1 polarizations
S1_POLS = ['VV', 'VH']

# Patch settings
PATCH_SIZE = 256
PATCH_STEP = 256
CLOUD_MAX  = 0.20


# ── Sentinel-2 Pipeline ────────────────────────────────────────
class Sentinel2Pipeline:
    """
    Downloads and preprocesses Sentinel-2 scenes
    for Netherlands study area.
    """

    def __init__(self, cfg: dict):
        self.cfg    = cfg
        self.s3_pub = boto3.client(
            's3', region_name='us-east-1',
            config=Config(signature_version=UNSIGNED)
        )
        self.s3_prv = boto3.client(
            's3', region_name='us-east-1',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
        )
        self.output_bucket = cfg['aws']['bucket']
        self.output_prefix = cfg['aws']['prefix'] + '/sentinel2'

    def list_scenes(
        self,
        tile: str,
        start_date: str,
        end_date: str
    ) -> List[Dict]:
        """
        List available Sentinel-2 scenes for a tile
        within a date range.

        Args:
            tile:       MGRS tile ID (e.g. '31UFT')
            start_date: 'YYYY-MM-DD'
            end_date:   'YYYY-MM-DD'

        Returns:
            List of scene metadata dicts
        """
        # Parse tile into S3 path components
        # e.g. 31UFT → 31/U/FT
        zone    = tile[:2]
        lat_band = tile[2]
        square  = tile[3:]
        prefix  = f'{SENTINEL2_PREFIX}/{zone}/{lat_band}/{square}/'

        scenes = []
        start  = datetime.strptime(start_date, '%Y-%m-%d')
        end    = datetime.strptime(end_date, '%Y-%m-%d')

        # List by year/month
        current = start.replace(day=1)
        while current <= end:
            month_prefix = (
                f'{prefix}{current.year}/{current.month}/'
            )
            try:
                resp = self.s3_pub.list_objects_v2(
                    Bucket=SENTINEL2_BUCKET,
                    Prefix=month_prefix,
                    Delimiter='/'
                )
                for obj in resp.get('CommonPrefixes', []):
                    scene_prefix = obj['Prefix']
                    scene_name   = scene_prefix.split('/')[-2]

                    # Extract date from scene name
                    # e.g. S2A_31UFT_20200707_0_L2A
                    parts = scene_name.split('_')
                    if len(parts) >= 3:
                        scene_date = datetime.strptime(
                            parts[2], '%Y%m%d'
                        )
                        if start <= scene_date <= end:
                            scenes.append({
                                'name':   scene_name,
                                'prefix': scene_prefix,
                                'date':   scene_date.strftime('%Y-%m-%d'),
                                'tile':   tile,
                            })
            except Exception as e:
                log.warning(f'Error listing {month_prefix}: {e}')

            # Next month
            if current.month == 12:
                current = current.replace(
                    year=current.year+1, month=1
                )
            else:
                current = current.replace(month=current.month+1)

        log.info(f'Found {len(scenes)} scenes for tile {tile}')
        return scenes

    def download_scene(
        self,
        scene: Dict,
        output_dir: str
    ) -> Optional[str]:
        """
        Download all bands for one scene.

        Returns:
            Path to scene directory or None if failed
        """
        scene_dir = os.path.join(output_dir, scene['name'])
        os.makedirs(scene_dir, exist_ok=True)

        bands_to_download = S2_BANDS + ['SCL']
        success = True

        for band in bands_to_download:
            local_path = os.path.join(scene_dir, f'{band}.tif')
            if os.path.exists(local_path):
                continue

            s3_key = f"{scene['prefix']}{band}.tif"
            try:
                self.s3_pub.download_file(
                    SENTINEL2_BUCKET, s3_key, local_path
                )
                log.debug(f'Downloaded {band}.tif')
            except Exception as e:
                log.error(f'Failed to download {band}: {e}')
                success = False

        return scene_dir if success else None

    def read_scene(
        self,
        scene_dir: str
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Read all bands from scene directory.
        Resamples 20m bands to 10m.

        Returns:
            bands:     [6, H, W] float32 array
            scl:       [H, W] uint8 cloud mask
            profile:   rasterio profile dict
        """
        # Read reference band (B02 at 10m)
        with rasterio.open(
            os.path.join(scene_dir, 'B02.tif')
        ) as src:
            profile   = src.profile.copy()
            height    = src.height
            width     = src.width
            transform = src.transform
            crs       = src.crs

        band_arrays = []
        for band in S2_BANDS:
            path = os.path.join(scene_dir, f'{band}.tif')
            with rasterio.open(path) as src:
                if src.height != height or src.width != width:
                    data = src.read(
                        1,
                        out_shape=(height, width),
                        resampling=Resampling.bilinear
                    ).astype(np.float32)
                else:
                    data = src.read(1).astype(np.float32)
            band_arrays.append(data)

        # Read cloud mask
        with rasterio.open(
            os.path.join(scene_dir, 'SCL.tif')
        ) as src:
            scl = src.read(
                1,
                out_shape=(height, width),
                resampling=Resampling.nearest
            )

        bands = np.stack(band_arrays, axis=0)
        return bands, scl, profile

    def compute_cloud_cover(self, scl: np.ndarray) -> float:
        """Compute cloud cover fraction from SCL mask."""
        cloud_classes = [8, 9, 10]  # cloud medium, high, cirrus
        cloud_mask    = np.isin(scl, cloud_classes)
        return float(cloud_mask.sum()) / scl.size

    def chip_scene(
        self,
        bands: np.ndarray,
        scl: np.ndarray,
        transform: rasterio.transform.Affine,
        scene_name: str
    ) -> List[Dict]:
        """
        Chip scene into 256x256 patches.
        Filters cloudy and nodata patches.

        Returns:
            List of patch dicts with bands + metadata
        """
        _, H, W = bands.shape
        patches  = []

        rows = range(0, H - PATCH_SIZE + 1, PATCH_STEP)
        cols = range(0, W - PATCH_SIZE + 1, PATCH_STEP)

        for row in rows:
            for col in cols:
                patch_bands = bands[
                    :, row:row+PATCH_SIZE, col:col+PATCH_SIZE
                ]
                patch_scl = scl[
                    row:row+PATCH_SIZE, col:col+PATCH_SIZE
                ]

                # Skip nodata patches
                if (patch_bands == 0).mean() > 0.5:
                    continue

                # Skip cloudy patches
                cloud_cover = self.compute_cloud_cover(patch_scl)
                if cloud_cover > CLOUD_MAX:
                    continue

                # Compute geographic bounds
                x_min, y_max = rasterio.transform.xy(
                    transform, row, col, offset='ul'
                )
                x_max, y_min = rasterio.transform.xy(
                    transform,
                    row + PATCH_SIZE,
                    col + PATCH_SIZE,
                    offset='ul'
                )

                patches.append({
                    'bands':       patch_bands,
                    'scl':         patch_scl,
                    'row':         row,
                    'col':         col,
                    'x_min':       x_min,
                    'y_min':       y_min,
                    'x_max':       x_max,
                    'y_max':       y_max,
                    'cloud_cover': cloud_cover,
                    'scene_name':  scene_name,
                })

        log.info(f'Chipped {len(patches)} valid patches')
        return patches

    def save_patch_to_s3(
        self,
        patch: Dict,
        scene_date: str,
        tile: str,
        crs: rasterio.crs.CRS,
        transform: rasterio.transform.Affine
    ) -> str:
        """Save patch as COG GeoTIFF to S3."""
        import tempfile
        from rasterio.transform import from_bounds

        patch_id  = (
            f"{tile}_{scene_date}_"
            f"r{patch['row']:05d}_c{patch['col']:05d}"
        )
        s3_key    = (
            f"{self.output_prefix}/{tile}/{scene_date}/"
            f"{patch_id}.tif"
        )

        patch_transform = from_bounds(
            patch['x_min'], patch['y_min'],
            patch['x_max'], patch['y_max'],
            PATCH_SIZE, PATCH_SIZE
        )

        with tempfile.NamedTemporaryFile(
            suffix='.tif', delete=False
        ) as tmp:
            with rasterio.open(
                tmp.name, 'w',
                driver='GTiff',
                height=PATCH_SIZE,
                width=PATCH_SIZE,
                count=len(S2_BANDS),
                dtype='float32',
                crs=crs,
                transform=patch_transform,
            ) as dst:
                dst.write(patch['bands'])
                dst.update_tags(
                    patch_id=patch_id,
                    scene=patch['scene_name'],
                    date=scene_date,
                    tile=tile,
                    cloud_cover=str(patch['cloud_cover']),
                    bands=','.join(S2_BANDS),
                )

            self.s3_prv.upload_file(tmp.name, self.output_bucket, s3_key)
            os.unlink(tmp.name)

        return patch_id

    def process_tile(
        self,
        tile: str,
        start_date: str,
        end_date: str,
        tmp_dir: str = '/tmp/sentinel2'
    ) -> List[str]:
        """
        Full pipeline for one tile + date range:
        list → download → chip → save to S3

        Returns:
            List of patch IDs processed
        """
        os.makedirs(tmp_dir, exist_ok=True)
        patch_ids = []

        scenes = self.list_scenes(tile, start_date, end_date)
        log.info(f'Processing {len(scenes)} scenes for tile {tile}')

        for scene in scenes:
            log.info(f'Processing scene: {scene["name"]}')

            scene_dir = self.download_scene(scene, tmp_dir)
            if not scene_dir:
                continue

            try:
                bands, scl, profile = self.read_scene(scene_dir)
                patches = self.chip_scene(
                    bands, scl,
                    profile['transform'],
                    scene['name']
                )

                for patch in patches:
                    patch_id = self.save_patch_to_s3(
                        patch,
                        scene['date'],
                        tile,
                        profile['crs'],
                        profile['transform']
                    )
                    patch_ids.append(patch_id)

                log.info(
                    f'Scene {scene["name"]}: '
                    f'{len(patches)} patches saved to S3'
                )

            except Exception as e:
                log.error(f'Failed to process scene {scene["name"]}: {e}')

        return patch_ids


# ── Sentinel-1 Pipeline ────────────────────────────────────────
class Sentinel1Pipeline:
    """
    Downloads and preprocesses Sentinel-1 SAR scenes
    for Netherlands study area.

    Outputs:
      - VV/VH backscatter (dB)
      - SAR coherence between T1/T2 pairs
    """

    def __init__(self, cfg: dict):
        self.cfg    = cfg
        self.s3_pub = boto3.client(
            's3', region_name='us-east-1',
            config=Config(signature_version=UNSIGNED)
        )
        self.s3_prv = boto3.client(
            's3', region_name='us-east-1',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
        )
        self.output_bucket = cfg['aws']['bucket']
        self.output_prefix = cfg['aws']['prefix'] + '/sentinel1'

    def compute_sar_coherence(
        self,
        sar_t1: np.ndarray,
        sar_t2: np.ndarray,
        window_size: int = 5
    ) -> np.ndarray:
        """
        Compute SAR coherence between two SAR images.
        High coherence = stable surface = no change
        Low coherence = change occurred

        Args:
            sar_t1:      [2, H, W] VV/VH at T1
            sar_t2:      [2, H, W] VV/VH at T2
            window_size: coherence window size

        Returns:
            coherence: [H, W] float32 [0, 1]
        """
        from scipy.ndimage import uniform_filter

        # Use VV polarization for coherence
        vv_t1 = sar_t1[0].astype(np.complex64)
        vv_t2 = sar_t2[0].astype(np.complex64)

        # Cross-correlation
        cross = vv_t1 * np.conj(vv_t2)

        # Spatial averaging
        cross_avg = uniform_filter(
            np.real(cross), window_size
        ) + 1j * uniform_filter(
            np.imag(cross), window_size
        )
        power_t1 = uniform_filter(
            np.abs(vv_t1)**2, window_size
        )
        power_t2 = uniform_filter(
            np.abs(vv_t2)**2, window_size
        )

        # Coherence magnitude
        coherence = np.abs(cross_avg) / (
            np.sqrt(power_t1 * power_t2) + 1e-10
        )
        coherence = np.clip(coherence, 0, 1)

        return coherence.astype(np.float32)

    def compute_coherence_loss(
        self,
        coherence: np.ndarray
    ) -> np.ndarray:
        """
        Convert coherence to change signal.
        Low coherence → high change signal.

        Returns:
            change_signal: [H, W] float32 [0, 1]
        """
        return 1.0 - coherence


# ── Pair Generator ─────────────────────────────────────────────
class SentinelPairGenerator:
    """
    Generates T1/T2 patch pairs for training.
    Ensures spatial intersection of all modalities.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def generate_pairs(
        self,
        patches_t1: List[Dict],
        patches_t2: List[Dict],
        max_day_gap: int = 365,
        min_day_gap: int = 30
    ) -> List[Dict]:
        """
        Match T1/T2 patches by location.
        Filter by day gap constraints.

        Returns:
            List of pair dicts
        """
        # Build T1 lookup by (row, col)
        t1_lookup = {
            (p['row'], p['col']): p
            for p in patches_t1
        }

        pairs = []
        for p2 in patches_t2:
            loc = (p2['row'], p2['col'])
            if loc not in t1_lookup:
                continue

            p1 = t1_lookup[loc]

            # Compute day gap
            d1 = datetime.strptime(p1['date'], '%Y-%m-%d')
            d2 = datetime.strptime(p2['date'], '%Y-%m-%d')
            day_gap = abs((d2 - d1).days)

            if day_gap < min_day_gap or day_gap > max_day_gap:
                continue

            pairs.append({
                'patch_t1':    p1,
                'patch_t2':    p2,
                'day_gap':     day_gap,
                'tile':        p1.get('tile', ''),
                'row':         p1['row'],
                'col':         p1['col'],
            })

        log.info(f'Generated {len(pairs):,} patch pairs')
        return pairs


# ── Main ───────────────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='OmniGeoFusion Sentinel data pipeline'
    )
    parser.add_argument(
        '--config', default='configs/data_config.yaml'
    )
    parser.add_argument(
        '--tile', default='31UFT',
        help='MGRS tile ID'
    )
    parser.add_argument(
        '--start-date', default='2020-06-01'
    )
    parser.add_argument(
        '--end-date', default='2020-09-30'
    )
    parser.add_argument(
        '--modality', choices=['s2', 's1', 'both'],
        default='both'
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )

    if args.modality in ['s2', 'both']:
        log.info('Starting Sentinel-2 pipeline...')
        s2 = Sentinel2Pipeline(cfg)
        patch_ids = s2.process_tile(
            args.tile,
            args.start_date,
            args.end_date
        )
        log.info(f'✅ Sentinel-2 complete: {len(patch_ids)} patches')

    if args.modality in ['s1', 'both']:
        log.info('Starting Sentinel-1 pipeline...')
        s1 = Sentinel1Pipeline(cfg)
        log.info('✅ Sentinel-1 pipeline initialized')


if __name__ == '__main__':
    main()
