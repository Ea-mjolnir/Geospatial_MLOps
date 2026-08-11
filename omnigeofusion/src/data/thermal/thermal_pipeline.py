"""
OmniGeoFusion — Thermal Pipeline
==================================
Downloads Landsat 8/9 LWIR11 thermal patches
from Microsoft Planetary Computer.

Source:
  Planetary Computer landsat-c2-l2 collection
  Band: lwir11 (Land Surface Temperature)
  CRS: EPSG:32631 (same as S2 — no reprojection needed)
  Resolution: 30m (resampled to 256x256 per patch)

Processing:
  DN → LST (°C): DN * 0.00341802 + 149.0 - 273.15
  Output: [1, 256, 256] float32 LST in Celsius

Strategy:
  Match S2 patch locations from index
  Find best Landsat scene per S2 date (closest + low cloud)
  Read patches via windowed COG reading
  Upload to S3

Output:
  s3://omnigeofusion-data-288528696055/
    netherlands/thermal/{tile}/{date}/{patch_id}.tif
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
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from typing import List, Dict, Tuple, Optional

log = logging.getLogger(__name__)
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('boto3').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = '.tiff,.tif,.TIF'

# ── Constants ──────────────────────────────────────────────────
STAC_URL   = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
TOKEN_URL  = "https://planetarycomputer.microsoft.com/api/sas/v1/token/landsateuwest/landsat-c2"
COLLECTION = "landsat-c2-l2"

PATCH_SIZE  = 256
PIXEL_SIZE  = 10
NODATA_OUT  = -9999.0
WORKERS     = 4

# DN to LST conversion
LST_SCALE  = 0.00341802
LST_OFFSET = 149.0
KELVIN     = 273.15

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

STUDY_BBOXES = {
    'amsterdam': [4.7, 52.2, 5.1, 52.5],
    'rotterdam': [4.2, 51.8, 4.7, 52.0],
    'flevoland': [5.2, 52.3, 5.8, 52.7],
}


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


def dn_to_lst(data: np.ndarray) -> np.ndarray:
    """Convert Landsat DN to Land Surface Temperature (°C)."""
    valid = data > 0
    lst   = np.where(
        valid,
        data * LST_SCALE + LST_OFFSET - KELVIN,
        NODATA_OUT
    )
    return lst.astype(np.float32)


# ── Token Manager ─────────────────────────────────────────────
class TokenManager:
    """Auto-refreshing Planetary Computer SAS token."""

    def __init__(self):
        self._token  = None
        self._expiry = None

    def get_token(self) -> str:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if self._token and self._expiry and now < self._expiry:
            return self._token
        resp         = requests.get(TOKEN_URL, timeout=30)
        resp.raise_for_status()
        data         = resp.json()
        self._token  = data['token']
        self._expiry = datetime.strptime(
            data['msft:expiry'], '%Y-%m-%dT%H:%M:%SZ'
        ) - timedelta(minutes=5)
        log.info(f'Token refreshed → expires {self._expiry}')
        return self._token

    def sign(self, url: str) -> str:
        return f"{url}?{self.get_token()}"


# ── Scene Finder ──────────────────────────────────────────────
class LandsatSceneFinder:
    """Finds best Landsat scene for a date + bbox."""

    def __init__(self, token_mgr: TokenManager):
        self.token_mgr = token_mgr

    def find_scenes(
        self,
        bbox: List[float],
        target_date: str,
        max_day_gap: int = 16,
        max_cloud: float = 50.0,
    ) -> List[Dict]:
        """Find Landsat scenes near target_date."""
        target = datetime.strptime(target_date, '%Y-%m-%d')
        start  = (target - timedelta(days=max_day_gap)).strftime('%Y-%m-%d')
        end    = (target + timedelta(days=max_day_gap)).strftime('%Y-%m-%d')

        payload = {
            "collections": [COLLECTION],
            "bbox":        bbox,
            "datetime":    f"{start}/{end}",
            "limit":       20,
            "query":       {"eo:cloud_cover": {"lt": max_cloud}},
        }
        resp  = requests.post(STAC_URL, json=payload, timeout=30)
        resp.raise_for_status()
        items = resp.json().get('features', [])

        def day_diff(item):
            dt = datetime.strptime(
                item['properties']['datetime'][:10], '%Y-%m-%d'
            )
            return abs((dt - target).days)

        return sorted(items, key=day_diff)

    def get_signed_url(self, item: Dict) -> Optional[str]:
        """Get signed thermal URL (lwir11 for L8/L9, lwir for L7)."""
        assets = item.get('assets', {})
        # Try lwir11 first (Landsat 8/9), then lwir (Landsat 7)
        for key in ['lwir11', 'lwir', 'ST_B10', 'B10']:
            url = assets.get(key, {}).get('href', '')
            if url:
                return self.token_mgr.sign(url)
        return None


# ── Thermal Patch Reader ──────────────────────────────────────
class ThermalPatchReader:
    """Reads thermal patches from Landsat COG."""

    def __init__(self, lwir_url: str, scene_id: str):
        self.lwir_url  = lwir_url
        self.scene_id  = scene_id
        self._src      = None
        self._crs      = None

    def __enter__(self):
        self._src = rasterio.open(self.lwir_url)
        self._crs = self._src.crs
        log.info(
            f'Opened Landsat: {self.scene_id[:50]}'
            f' | CRS: {self._crs}'
            f' | Shape: {self._src.height}x{self._src.width}'
        )
        return self

    def __exit__(self, *args):
        if self._src:
            self._src.close()

    def read_patch(
        self,
        x_min: float, y_min: float,
        x_max: float, y_max: float,
    ) -> Optional[np.ndarray]:
        """Read one thermal patch at bbox.
        Reprojects bounds from EPSG:32631 to scene CRS if needed.
        """
        from rasterio.warp import transform_bounds
        try:
            # Reproject bounds if scene CRS differs from S2 CRS
            if self._crs and str(self._crs) != 'EPSG:32631':
                x_min, y_min, x_max, y_max = transform_bounds(
                    'EPSG:32631', self._crs,
                    x_min, y_min, x_max, y_max
                )

            # Get pixel indices
            row_min, col_min = self._src.index(x_min, y_max)
            row_max, col_max = self._src.index(x_max, y_min)

            row_min = max(0, row_min)
            col_min = max(0, col_min)
            row_max = min(self._src.height, row_max)
            col_max = min(self._src.width,  col_max)

            if row_max <= row_min or col_max <= col_min:
                return None

            win  = Window(
                col_min, row_min,
                col_max - col_min,
                row_max - row_min
            )
            data = self._src.read(
                1, window=win,
                out_shape=(PATCH_SIZE, PATCH_SIZE),
                resampling=Resampling.bilinear
            )

            # Check valid pixels
            valid_frac = float((data > 0).sum()) / data.size
            if valid_frac < 0.5:
                return None

            return dn_to_lst(data)

        except Exception as e:
            log.debug(f'Read error: {e}')
            return None


# ── Thermal Pipeline ──────────────────────────────────────────
class ThermalPipeline:
    """
    Downloads Landsat thermal patches for all study areas.
    Matches S2 patch locations.
    """

    def __init__(self, cfg: dict):
        self.cfg           = cfg
        self.output_bucket = cfg['aws']['bucket']
        self.output_prefix = cfg['aws']['prefix'] + '/thermal'
        self.s3            = boto3.client(
            's3', region_name='us-east-1',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
        )
        self.token_mgr    = TokenManager()
        self.scene_finder = LandsatSceneFinder(self.token_mgr)

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

    def upload_patch(
        self,
        lst_data: np.ndarray,
        tile: str,
        landsat_date: str,
        s2_date: str,
        row: int,
        col: int,
        scene_id: str,
    ) -> Optional[str]:
        """Upload thermal patch to S3."""
        patch_id = f"{tile}_{landsat_date}_r{row:05d}_c{col:05d}"
        s3_key   = (
            f"{self.output_prefix}/{tile}/"
            f"{landsat_date}/{patch_id}.tif"
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
                height=PATCH_SIZE,
                width=PATCH_SIZE,
                count=1,
                dtype='float32',
                crs='EPSG:32631',
                transform=transform,
                nodata=NODATA_OUT,
            ) as dst:
                dst.write(lst_data[np.newaxis, :, :])
                dst.update_tags(
                    patch_id=patch_id,
                    landsat_date=landsat_date,
                    s2_date=s2_date,
                    tile=tile,
                    bands='LST_Celsius',
                    scene=scene_id,
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

    def process_date_batch(
        self,
        reader: ThermalPatchReader,
        entries: List[Dict],
        tile: str,
        landsat_date: str,
        s2_date: str,
        scene_id: str,
    ) -> List[Dict]:
        """Process all patches for one date — serial read + upload."""
        index_entries = []

        for i, entry in enumerate(entries):
            row = entry['row']
            col = entry['col']

            x_min, y_min, x_max, y_max = compute_patch_bounds(
                tile, row, col
            )

            lst_data = reader.read_patch(
                x_min, y_min, x_max, y_max
            )
            if lst_data is None:
                continue

            patch_id = self.upload_patch(
                lst_data, tile,
                landsat_date, s2_date,
                row, col, scene_id
            )
            if patch_id:
                index_entries.append({
                    'patch_id':      patch_id,
                    'tile':          tile,
                    'landsat_date':  landsat_date,
                    's2_date':       s2_date,
                    'row':           row,
                    'col':           col,
                    'scene_id':      scene_id,
                    's3_key': (
                        f"{self.output_prefix}/{tile}/"
                        f"{landsat_date}/{patch_id}.tif"
                    ),
                })

            del lst_data

            if (i + 1) % 100 == 0:
                log.info(
                    f'  {i+1}/{len(entries)} done '
                    f'({len(index_entries)} uploaded)'
                )

        return index_entries

    def process_area(self, area: str) -> Dict:
        """Full thermal pipeline for one study area."""
        tile = STUDY_TILES[area]
        bbox = STUDY_BBOXES[area]

        log.info(f'\n{"="*60}')
        log.info(f'Thermal Pipeline: {area} ({tile})')
        log.info(f'Source: Planetary Computer Landsat-C2-L2')
        log.info(f'{"="*60}')

        s2_index    = self.load_s2_index(area)
        date_groups = {}
        for entry in s2_index:
            d = entry['date']
            date_groups.setdefault(d, []).append(entry)

        log.info(f'Unique S2 dates: {len(date_groups)}')

        all_index = []

        for s2_date, entries in sorted(date_groups.items()):
            log.info(
                f'\nS2 date: {s2_date} ({len(entries)} patches)'
            )

            scenes = self.scene_finder.find_scenes(bbox, s2_date)
            if not scenes:
                log.warning(f'No Landsat scene for {s2_date}')
                continue

            scene        = scenes[0]
            landsat_date = scene['properties']['datetime'][:10]
            cloud        = scene['properties'].get(
                'eo:cloud_cover', 'N/A'
            )
            day_gap      = abs((
                datetime.strptime(landsat_date, '%Y-%m-%d') -
                datetime.strptime(s2_date, '%Y-%m-%d')
            ).days)

            log.info(
                f'Best Landsat: {scene["id"][:50]}'
                f' (cloud: {cloud}%, gap: {day_gap}d)'
            )

            lwir_url = self.scene_finder.get_signed_url(scene)
            if not lwir_url:
                log.warning('No LWIR11 asset found')
                continue

            with ThermalPatchReader(
                lwir_url, scene['id']
            ) as reader:
                entries_done = self.process_date_batch(
                    reader, entries,
                    tile, landsat_date, s2_date,
                    scene['id'],
                )
                all_index.extend(entries_done)

            log.info(
                f'Date {s2_date}: '
                f'{len(entries_done)}/{len(entries)} uploaded'
            )
            log.info(f'Running total: {len(all_index)} patches')

        # Save index
        index_key = (
            f"{self.output_prefix}/index/{area}_index.json"
        )
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as tmp:
            json.dump(all_index, tmp, indent=2)
            tmp_path = tmp.name
        self.s3.upload_file(
            tmp_path, self.output_bucket, index_key
        )
        os.unlink(tmp_path)

        log.info(f'\n{"="*60}')
        log.info(
            f'✅ {area}: {len(all_index)} thermal patches'
        )
        log.info(f'{"="*60}')

        return {'area': area, 'n_patches': len(all_index)}


# ── Main ───────────────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='OmniGeoFusion Thermal pipeline'
    )
    parser.add_argument(
        '--config', default='configs/data_config.yaml'
    )
    parser.add_argument(
        '--area',
        choices=[
            'amsterdam', 'rotterdam', 'flevoland', 'all'
        ],
        default='all'
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )

    cfg      = yaml.safe_load(open(args.config))
    pipeline = ThermalPipeline(cfg)
    areas    = (
        list(STUDY_TILES.keys()) if args.area == 'all'
        else [args.area]
    )

    total = 0
    for area in areas:
        r      = pipeline.process_area(area)
        total += r['n_patches']
        log.info(f'  {area}: {r["n_patches"]}')

    log.info(f'\nTHERMAL COMPLETE: {total} patches')


if __name__ == '__main__':
    main()
