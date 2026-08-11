"""
OmniGeoFusion — LiDAR AHN4 Pipeline
======================================
Downloads DSM + DTM patches from PDOK WCS API.
Computes nDSM (normalized DSM = DSM - DTM).

Source:
  PDOK WCS: https://service.pdok.nl/rws/ahn/wcs/v1_0
  Layers: dsm_05m + dtm_05m (0.5m resolution)
  CRS: EPSG:28992 (RD New)
  Coverage: 100% Netherlands
  License: CC-0 (public domain)

Strategy:
  Match S2 patch locations from index
  Request 256x256 patches via WCS GetCoverage
  Compute nDSM = DSM - DTM
  Stack [DSM, DTM, nDSM] → [3, 256, 256]
  Upload to S3

Output:
  s3://omnigeofusion-data-288528696055/
    netherlands/lidar/{tile}/{date}/{patch_id}.tif
"""

import os
import json
import logging
import tempfile
import numpy as np
import boto3
import rasterio
import requests
import yaml

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pyproj import Transformer
from rasterio.transform import from_bounds
from typing import List, Dict, Tuple, Optional

log = logging.getLogger(__name__)
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('boto3').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# ── Constants ──────────────────────────────────────────────────
WCS_URL    = "https://service.pdok.nl/rws/ahn/wcs/v1_0"
PATCH_SIZE = 256
PIXEL_SIZE = 10
NODATA_AHN = 3.4028234663852886e+38
NODATA_OUT = -9999.0
WORKERS    = 8

TILE_ORIGINS = {
    '31UFT': (600000.0, 5800020.0),
    '31UFS': (600000.0, 5700000.0),
    '31UGU': (699960.0, 5900040.0),
}

STUDY_TILES = {
    'amsterdam': '31UFT',
    'rotterdam': '31UFS',
    'flevoland': '31UGU',
}

T_32631_TO_28992 = Transformer.from_crs(
    'EPSG:32631', 'EPSG:28992', always_xy=True
)


# ── Helpers ───────────────────────────────────────────────────
def compute_patch_bounds(
    tile: str, row: int, col: int
) -> Tuple[float, float, float, float]:
    """Compute patch bounds in EPSG:32631."""
    origin_x, origin_y = TILE_ORIGINS[tile]
    patch_m = PATCH_SIZE * PIXEL_SIZE
    x_min   = origin_x + col * PIXEL_SIZE
    y_max   = origin_y - row * PIXEL_SIZE
    x_max   = x_min + patch_m
    y_min   = y_max - patch_m
    return x_min, y_min, x_max, y_max


def to_rd_new(
    x_min: float, y_min: float,
    x_max: float, y_max: float,
) -> Tuple[float, float, float, float]:
    """Convert bounds EPSG:32631 → EPSG:28992."""
    x0, y0 = T_32631_TO_28992.transform(x_min, y_min)
    x1, y1 = T_32631_TO_28992.transform(x_max, y_max)
    return x0, y0, x1, y1


def fetch_wcs(
    layer: str,
    x_min_rd: float, y_min_rd: float,
    x_max_rd: float, y_max_rd: float,
) -> Optional[np.ndarray]:
    """Fetch 256x256 patch from PDOK WCS."""
    params = {
        "SERVICE":  "WCS", "VERSION": "1.0.0",
        "REQUEST":  "GetCoverage",
        "COVERAGE": layer, "CRS": "EPSG:28992",
        "BBOX":     f"{x_min_rd:.0f},{y_min_rd:.0f},"
                    f"{x_max_rd:.0f},{y_max_rd:.0f}",
        "WIDTH":    str(PATCH_SIZE),
        "HEIGHT":   str(PATCH_SIZE),
        "FORMAT":   "GEOTIFF",
    }
    try:
        resp = requests.get(WCS_URL, params=params, timeout=60)
        if resp.status_code != 200 or len(resp.content) < 1000:
            return None

        with tempfile.NamedTemporaryFile(
            suffix='.tif', delete=False
        ) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name

        with rasterio.open(tmp_path) as src:
            data = src.read(1).astype(np.float32)
        os.unlink(tmp_path)

        data[data >= NODATA_AHN / 2] = NODATA_OUT
        data[data <= -1e10]           = NODATA_OUT
        return data

    except Exception as e:
        log.debug(f'WCS error {layer}: {e}')
        return None


# ── LiDAR Pipeline ────────────────────────────────────────────
class LiDARPipeline:

    def __init__(self, cfg: dict):
        self.cfg           = cfg
        self.output_bucket = cfg['aws']['bucket']
        self.output_prefix = cfg['aws']['prefix'] + '/lidar'
        self.s3            = boto3.client(
            's3', region_name='us-east-1',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
        )

    def load_s2_index(self, area: str) -> List[Dict]:
        key = (
            f"{self.cfg['aws']['prefix']}/sentinel2/"
            f"index/{area}_index.json"
        )
        obj   = self.s3.get_object(
            Bucket=self.output_bucket, Key=key
        )
        index = json.loads(obj['Body'].read())
        log.info(f'S2 index: {len(index)} patches')
        return index

    def fetch_patch(
        self, tile: str, row: int, col: int
    ) -> Optional[np.ndarray]:
        """Fetch DSM+DTM+nDSM → [3,256,256]."""
        x_min, y_min, x_max, y_max = compute_patch_bounds(
            tile, row, col
        )
        x0, y0, x1, y1 = to_rd_new(x_min, y_min, x_max, y_max)

        dsm = fetch_wcs('dsm_05m', x0, y0, x1, y1)
        if dsm is None:
            return None
        dtm = fetch_wcs('dtm_05m', x0, y0, x1, y1)
        if dtm is None:
            return None

        valid = (dsm != NODATA_OUT) & (dtm != NODATA_OUT)
        if float(valid.sum()) / valid.size < 0.5:
            return None

        ndsm = np.where(
            valid,
            np.clip(dsm - dtm, 0, None),
            NODATA_OUT
        ).astype(np.float32)

        return np.stack([dsm, dtm, ndsm], axis=0)

    def upload_patch(
        self,
        patch_data: np.ndarray,
        tile: str, date: str,
        row: int, col: int,
    ) -> Optional[str]:
        patch_id = f"{tile}_{date}_r{row:05d}_c{col:05d}"
        s3_key   = (
            f"{self.output_prefix}/{tile}/{date}/{patch_id}.tif"
        )
        x_min, y_min, x_max, y_max = compute_patch_bounds(
            tile, row, col
        )
        transform = from_bounds(
            x_min, y_min, x_max, y_max,
            PATCH_SIZE, PATCH_SIZE
        )
        with tempfile.NamedTemporaryFile(
            suffix='.tif', delete=False
        ) as tmp:
            tmp_path = tmp.name
        try:
            with rasterio.open(
                tmp_path, 'w',
                driver='GTiff',
                height=PATCH_SIZE, width=PATCH_SIZE,
                count=3, dtype='float32',
                crs='EPSG:32631',
                transform=transform,
                nodata=NODATA_OUT,
            ) as dst:
                dst.write(patch_data)
                dst.update_tags(
                    patch_id=patch_id, date=date, tile=tile,
                    bands='DSM,DTM,nDSM', source='AHN4_PDOK'
                )
            self.s3.upload_file(
                tmp_path, self.output_bucket, s3_key
            )
            return patch_id
        except Exception as e:
            log.error(f'Upload failed: {e}')
            return None
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def process_one(self, args: Dict) -> Optional[Dict]:
        tile = args['tile']
        row  = args['row']
        col  = args['col']
        date = args['date']
        area = args['area']

        patch = self.fetch_patch(tile, row, col)
        if patch is None:
            return None

        patch_id = self.upload_patch(patch, tile, date, row, col)
        if not patch_id:
            return None

        return {
            'patch_id': patch_id,
            'tile':     tile,
            'area':     area,
            'date':     date,
            'row':      row,
            'col':      col,
            's3_key': (
                f"{self.output_prefix}/{tile}/{date}/{patch_id}.tif"
            ),
        }

    def process_area(self, area: str) -> Dict:
        tile = STUDY_TILES[area]

        log.info(f'\n{"="*60}')
        log.info(f'LiDAR: {area} ({tile}) — PDOK WCS AHN4')
        log.info(f'{"="*60}')

        s2_index   = self.load_s2_index(area)
        date_counts = Counter(e['date'] for e in s2_index)
        lidar_date  = date_counts.most_common(1)[0][0]
        log.info(
            f'{len(s2_index)} patches | date label: {lidar_date}'
        )

        tasks = [
            {
                'tile': tile, 'area': area,
                'row':  e['row'], 'col': e['col'],
                'date': lidar_date,
            }
            for e in s2_index
        ]

        index_entries = []
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {
                ex.submit(self.process_one, t): t for t in tasks
            }
            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result()
                    if result:
                        index_entries.append(result)
                except Exception as e:
                    log.error(f'Error: {e}')

                if (i + 1) % 100 == 0:
                    log.info(
                        f'  {i+1}/{len(tasks)} done '
                        f'({len(index_entries)} uploaded)'
                    )

        # Save index
        index_key = (
            f"{self.output_prefix}/index/{area}_index.json"
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

        log.info(f'\n{"="*60}')
        log.info(f'✅ {area}: {len(index_entries)} LiDAR patches')
        log.info(f'{"="*60}')

        return {'area': area, 'n_patches': len(index_entries)}


# ── Main ───────────────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='OmniGeoFusion LiDAR AHN4 pipeline'
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
    pipeline = LiDARPipeline(cfg)
    areas    = (
        list(STUDY_TILES.keys()) if args.area == 'all'
        else [args.area]
    )

    total = 0
    for area in areas:
        r      = pipeline.process_area(area)
        total += r['n_patches']
        log.info(f'  {area}: {r["n_patches"]}')

    log.info(f'\nLiDAR COMPLETE: {total} patches')


if __name__ == '__main__':
    main()
