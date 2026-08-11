"""
OmniGeoFusion — Target Generation Pipeline
============================================
Generates training targets for all 3 tasks
from existing downloaded data (no manual labels).

Task A (Urban Change — Amsterdam 31UFT):
  Input:  S2 patch (6 bands) + LiDAR nDSM
  Target: change_score = f(NDBI, NDVI, nDSM)
  Range:  [0, 1] continuous

Task B (Flood Risk — Rotterdam 31UFS):
  Input:  S2 patch + LiDAR DTM + SAR VV
  Target: flood_risk = f(DTM elevation, SAR water)
  Range:  [0, 1] continuous

Task C (Crop Health — Flevoland 31UGU):
  Input:  S2 patch (6 bands)
  Target: crop_health = NDVI (normalized)
  Range:  [0, 1] continuous

Output:
  s3://omnigeofusion-data-288528696055/
    netherlands/targets/{task}/{tile}/{date}/{patch_id}.npy
    netherlands/targets/index/{area}_targets.json
"""

import os
import json
import logging
import tempfile
import numpy as np
import boto3
import rasterio
import yaml

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple

log = logging.getLogger(__name__)
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# ── Constants ──────────────────────────────────────────────────
STUDY_TILES = {
    'amsterdam': '31UFT',
    'rotterdam': '31UFS',
    'flevoland': '31UGU',
}

TASK_MAP = {
    'amsterdam': 'urban_change',
    'rotterdam': 'flood_risk',
    'flevoland': 'crop_health',
}

# S2 band indices (0-based)
# Bands: B02, B03, B04, B08, B11, B8A
B02 = 0  # Blue
B03 = 1  # Green
B04 = 2  # Red
B08 = 3  # NIR
B11 = 4  # SWIR
B8A = 5  # Red Edge

WORKERS = 8


# ── Spectral Indices ──────────────────────────────────────────
def compute_ndvi(s2: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red)"""
    nir = s2[B08].astype(np.float32)
    red = s2[B04].astype(np.float32)
    denom = nir + red + 1e-10
    return np.clip((nir - red) / denom, -1, 1)


def compute_ndbi(s2: np.ndarray) -> np.ndarray:
    """NDBI = (SWIR - NIR) / (SWIR + NIR) — built-up index"""
    swir = s2[B11].astype(np.float32)
    nir  = s2[B08].astype(np.float32)
    denom = swir + nir + 1e-10
    return np.clip((swir - nir) / denom, -1, 1)


def compute_ndwi(s2: np.ndarray) -> np.ndarray:
    """NDWI = (Green - NIR) / (Green + NIR) — water index"""
    green = s2[B03].astype(np.float32)
    nir   = s2[B08].astype(np.float32)
    denom = green + nir + 1e-10
    return np.clip((green - nir) / denom, -1, 1)


def compute_evi(s2: np.ndarray) -> np.ndarray:
    """EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)"""
    nir  = s2[B08].astype(np.float32) / 10000
    red  = s2[B04].astype(np.float32) / 10000
    blue = s2[B02].astype(np.float32) / 10000
    denom = nir + 6 * red - 7.5 * blue + 1 + 1e-10
    return np.clip(2.5 * (nir - red) / denom, -1, 1)


# ── Target Generators ─────────────────────────────────────────
def generate_urban_change_target(
    s2: np.ndarray,
    lidar: Optional[np.ndarray] = None
) -> float:
    """
    Urban change score for Task A.

    High score = urban/built-up area
    Uses NDBI (built-up), NDVI (vegetation inverse),
    and nDSM height if LiDAR available.

    Returns scalar [0, 1]
    """
    ndbi = compute_ndbi(s2)
    ndvi = compute_ndvi(s2)

    # Urban score = high NDBI + low NDVI
    urban = (ndbi + 1) / 2           # normalize to [0,1]
    veg   = (1 - (ndvi + 1) / 2)     # inverse NDVI

    score = 0.6 * urban.mean() + 0.4 * veg.mean()

    # Boost score if LiDAR shows tall structures
    if lidar is not None:
        ndsm = lidar[2]  # band 2 = nDSM
        valid = ndsm != -9999
        if valid.sum() > 0:
            mean_height = float(ndsm[valid].mean())
            height_score = np.clip(mean_height / 20.0, 0, 1)
            score = 0.5 * score + 0.5 * height_score

    return float(np.clip(score, 0, 1))


def generate_flood_risk_target(
    s2: np.ndarray,
    lidar: Optional[np.ndarray] = None,
    sar: Optional[np.ndarray] = None
) -> float:
    """
    Flood risk score for Task B.

    High score = high flood risk
    Uses DTM elevation (low = risk),
    NDWI (water presence),
    SAR VV (water surface if available).

    Returns scalar [0, 1]
    """
    ndwi = compute_ndwi(s2)

    # Water score from S2
    water_score = float(np.clip((ndwi.mean() + 1) / 2, 0, 1))

    # DTM elevation score (lower = more flood risk)
    elev_score = 0.5  # default
    if lidar is not None:
        dtm   = lidar[1]  # band 1 = DTM
        valid = dtm != -9999
        if valid.sum() > 0:
            mean_elev = float(dtm[valid].mean())
            # Netherlands: below 0m = high risk, above 5m = low risk
            elev_score = float(
                np.clip(1 - (mean_elev + 5) / 10, 0, 1)
            )

    # SAR water detection
    sar_score = 0.5  # default
    if sar is not None:
        vv    = sar[0]  # VV_dB
        valid = vv != -9999
        if valid.sum() > 0:
            # Low VV backscatter = water surface
            mean_vv   = float(vv[valid].mean())
            sar_score = float(
                np.clip(1 - (mean_vv + 25) / 20, 0, 1)
            )

    # Combine
    if lidar is not None and sar is not None:
        score = 0.3 * water_score + 0.4 * elev_score + 0.3 * sar_score
    elif lidar is not None:
        score = 0.4 * water_score + 0.6 * elev_score
    else:
        score = water_score

    return float(np.clip(score, 0, 1))


def generate_crop_health_target(
    s2: np.ndarray,
) -> float:
    """
    Crop health score for Task C.

    High score = healthy vegetation
    Uses NDVI as primary indicator,
    EVI as secondary.

    Returns scalar [0, 1]
    """
    ndvi = compute_ndvi(s2)
    evi  = compute_evi(s2)

    # Normalize to [0, 1]
    ndvi_score = float((ndvi.mean() + 1) / 2)
    evi_score  = float((evi.mean() + 1) / 2)

    score = 0.6 * ndvi_score + 0.4 * evi_score
    return float(np.clip(score, 0, 1))


# ── Target Pipeline ───────────────────────────────────────────
class TargetPipeline:
    """
    Generates targets for all patches in each area.
    Reads S2 + LiDAR + SAR from S3.
    Saves targets as .npy files to S3.
    """

    def __init__(self, cfg: dict):
        self.cfg           = cfg
        self.output_bucket = cfg['aws']['bucket']
        self.prefix        = cfg['aws']['prefix']
        self.s3            = boto3.client(
            's3', region_name='us-east-1',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
        )

    def load_index(self, modality: str, area: str) -> List[Dict]:
        """Load index file from S3."""
        key = f"{self.prefix}/{modality}/index/{area}_index.json"
        try:
            obj = self.s3.get_object(
                Bucket=self.output_bucket, Key=key
            )
            return json.loads(obj['Body'].read())
        except Exception as e:
            log.warning(f'No {modality} index for {area}: {e}')
            return []

    def download_patch(self, s3_key: str) -> Optional[np.ndarray]:
        """Download one patch from S3."""
        with tempfile.NamedTemporaryFile(
            suffix='.tif', delete=False
        ) as tmp:
            tmp_path = tmp.name
        try:
            self.s3.download_file(
                self.output_bucket, s3_key, tmp_path
            )
            with rasterio.open(tmp_path) as src:
                return src.read().astype(np.float32)
        except Exception as e:
            log.debug(f'Download error {s3_key}: {e}')
            return None
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def save_target(
        self,
        target: float,
        patch_id: str,
        task: str,
        tile: str,
        date: str,
    ) -> str:
        """Save target scalar as .npy to S3."""
        s3_key = (
            f"{self.prefix}/targets/{task}/"
            f"{tile}/{date}/{patch_id}.npy"
        )
        with tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False
        ) as tmp:
            tmp_path = tmp.name
        try:
            np.save(tmp_path, np.array([target], dtype=np.float32))
            self.s3.upload_file(
                tmp_path, self.output_bucket, s3_key
            )
            return s3_key
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def process_patch(self, args: Dict) -> Optional[Dict]:
        """Generate target for one patch."""
        area     = args['area']
        task     = TASK_MAP[area]
        s2_entry = args['s2_entry']
        patch_id = s2_entry['patch_id']
        tile     = s2_entry['tile']
        date     = s2_entry['date']

        # Download S2
        s2 = self.download_patch(s2_entry['s3_key'])
        if s2 is None:
            return None

        # Download LiDAR if available
        lidar = None
        if args.get('lidar_key'):
            lidar = self.download_patch(args['lidar_key'])

        # Download SAR if available
        sar = None
        if args.get('sar_key'):
            sar = self.download_patch(args['sar_key'])

        # Generate target
        if task == 'urban_change':
            target = generate_urban_change_target(s2, lidar)
        elif task == 'flood_risk':
            target = generate_flood_risk_target(s2, lidar, sar)
        elif task == 'crop_health':
            target = generate_crop_health_target(s2)
        else:
            return None

        # Save target
        s3_key = self.save_target(
            target, patch_id, task, tile, date
        )

        return {
            'patch_id': patch_id,
            'tile':     tile,
            'area':     area,
            'task':     task,
            'date':     date,
            'target':   target,
            's3_key':   s3_key,
            's2_key':   s2_entry['s3_key'],
        }

    def build_task_list(self, area: str) -> List[Dict]:
        """
        Build list of patch processing tasks.
        Matches S2 patches with LiDAR + SAR by location.
        """
        s2_index    = self.load_index('sentinel2', area)
        lidar_index = self.load_index('lidar', area)
        sar_index   = self.load_index('sentinel1', area)

        # Build lookup by (row, col)
        lidar_lookup = {
            (e['row'], e['col']): e['s3_key']
            for e in lidar_index
        }
        sar_lookup = {
            (e['row'], e['col']): e['s3_key']
            for e in sar_index
        }

        tasks = []
        for entry in s2_index:
            loc       = (entry['row'], entry['col'])
            lidar_key = lidar_lookup.get(loc)
            sar_key   = sar_lookup.get(loc)

            tasks.append({
                'area':      area,
                's2_entry':  entry,
                'lidar_key': lidar_key,
                'sar_key':   sar_key,
            })

        log.info(
            f'{area}: {len(tasks)} S2 patches, '
            f'{sum(1 for t in tasks if t["lidar_key"])} with LiDAR, '
            f'{sum(1 for t in tasks if t["sar_key"])} with SAR'
        )
        return tasks

    def process_area(self, area: str) -> Dict:
        """Generate all targets for one area."""
        tile = STUDY_TILES[area]
        task = TASK_MAP[area]

        log.info(f'\n{"="*60}')
        log.info(f'Target Generation: {area} → {task}')
        log.info(f'{"="*60}')

        tasks         = self.build_task_list(area)
        index_entries = []

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {
                ex.submit(self.process_patch, t): t
                for t in tasks
            }
            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result()
                    if result:
                        index_entries.append(result)
                except Exception as e:
                    log.error(f'Error: {e}')

                if (i + 1) % 200 == 0:
                    log.info(
                        f'  {i+1}/{len(tasks)} done '
                        f'({len(index_entries)} targets)'
                    )

        # Save target index
        index_key = (
            f"{self.prefix}/targets/index/{area}_targets.json"
        )
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as tmp:
            json.dump(index_entries, tmp, indent=2)
            tmp_path = tmp.name
        self.s3.upload_file(
            tmp_path, self.output_bucket, index_key
        )
        os.unlink(tmp_path)

        # Log target distribution
        targets = [e['target'] for e in index_entries]
        if targets:
            log.info(
                f'Target stats: '
                f'min={min(targets):.3f} '
                f'max={max(targets):.3f} '
                f'mean={np.mean(targets):.3f}'
            )

        log.info(f'\n{"="*60}')
        log.info(
            f'✅ {area}: {len(index_entries)} targets generated'
        )
        log.info(f'{"="*60}')

        return {
            'area':      area,
            'task':      task,
            'n_targets': len(index_entries),
        }


# ── Main ───────────────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='OmniGeoFusion target generation'
    )
    parser.add_argument(
        '--config', default='configs/data_config.yaml'
    )
    parser.add_argument(
        '--area',
        choices=['amsterdam', 'rotterdam', 'flevoland', 'all'],
        default='all'
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )

    cfg      = yaml.safe_load(open(args.config))
    pipeline = TargetPipeline(cfg)
    areas    = (
        list(STUDY_TILES.keys()) if args.area == 'all'
        else [args.area]
    )

    total = 0
    for area in areas:
        r      = pipeline.process_area(area)
        total += r['n_targets']
        log.info(f'  {area}: {r["n_targets"]} {r["task"]} targets')

    log.info(f'\nTARGET GENERATION COMPLETE: {total} targets')


if __name__ == '__main__':
    main()
