"""
OmniGeoFusion — Sentinel-1 SAR Pipeline
=========================================
Downloads Sentinel-1 RTC patches from
Microsoft Planetary Computer via windowed
reading — no full scene download needed.

Key features:
  1. Compute patch bounds from row/col + tile origin
     → No S2 patch downloads needed
  2. Reproject S2 bounds (EPSG:32631) → S1 CRS (EPSG:32632)
     → S1 RTC scenes are in UTM Zone 32N
  3. Open COG files ONCE per scene → read many patches
  4. Serial reading (rasterio not thread-safe for remote)
  5. Parallel S3 uploads (ThreadPool)

Source:
  Planetary Computer sentinel-1-rtc
  Pre-processed: calibrated + terrain corrected
  Polarizations: VV + VH → converted to dB
  CRS: EPSG:32632 (UTM Zone 32N)

Output:
  s3://omnigeofusion-data-288528696055/
    netherlands/sentinel1/{tile}/{s1_date}/{patch_id}.tif
    netherlands/sentinel1/index/{area}_index.json
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

from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from rasterio.windows import Window
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from typing import List, Dict, Tuple, Optional

log = logging.getLogger(__name__)
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('boto3').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = '.tiff,.tif'

# ── Constants ──────────────────────────────────────────────────
STAC_URL   = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
TOKEN_URL  = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel1euwestrtc/sentinel1-grd-rtc"
COLLECTION = "sentinel-1-rtc"

PATCH_SIZE = 256
PIXEL_SIZE = 10       # S2 pixel size metres
NODATA_VAL = -32768.0
NODATA_MAX = 0.30
UPLOAD_WORKERS = 4

# Tile origins in EPSG:32631 (computed from S2 patches)
TILE_ORIGINS = {
    '31UFT': (600000.0, 5800020.0),  # Amsterdam
    '31UFS': (600000.0, 5700000.0),  # Rotterdam
    '31UGU': (699960.0, 5900040.0),  # Flevoland
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
    """
    Compute patch bounds in EPSG:32631 from row/col.
    Uses known tile origins — no S2 download needed.
    Returns (x_min, y_min, x_max, y_max).
    """
    origin_x, origin_y = TILE_ORIGINS[tile]
    patch_m = PATCH_SIZE * PIXEL_SIZE  # 2560m
    x_min   = origin_x + col * PIXEL_SIZE
    y_max   = origin_y - row * PIXEL_SIZE
    x_max   = x_min + patch_m
    y_min   = y_max - patch_m
    return x_min, y_min, x_max, y_max


def reproject_bounds(
    x_min: float, y_min: float,
    x_max: float, y_max: float,
    src_crs: str, dst_crs
) -> Tuple[float, float, float, float]:
    """Reproject bbox from src_crs to dst_crs."""
    return transform_bounds(
        src_crs, dst_crs,
        x_min, y_min, x_max, y_max
    )


def dn_to_db(data: np.ndarray) -> np.ndarray:
    """Convert sigma0 to dB: 10 * log10(sigma0)."""
    valid = (data != NODATA_VAL) & (data > 0)
    db    = np.where(
        valid,
        10 * np.log10(np.abs(data) + 1e-10),
        -9999.0
    )
    return db.astype(np.float32)


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
class S1SceneFinder:
    """Finds best S1 RTC scene for a date + bbox."""

    def __init__(self, token_mgr: TokenManager):
        self.token_mgr = token_mgr

    def find_scenes_for_date(
        self,
        bbox: List[float],
        target_date: str,
        max_day_gap: int = 12
    ) -> List[Dict]:
        """Find S1 scenes within max_day_gap of target_date."""
        target = datetime.strptime(target_date, '%Y-%m-%d')
        start  = (
            target - timedelta(days=max_day_gap)
        ).strftime('%Y-%m-%d')
        end    = (
            target + timedelta(days=max_day_gap)
        ).strftime('%Y-%m-%d')

        payload = {
            "collections": [COLLECTION],
            "bbox":        bbox,
            "datetime":    f"{start}/{end}",
            "limit":       20,
        }
        resp  = requests.post(
            STAC_URL, json=payload, timeout=30
        )
        resp.raise_for_status()
        items = resp.json().get('features', [])

        def day_diff(item):
            dt = datetime.strptime(
                item['properties']['datetime'][:10],
                '%Y-%m-%d'
            )
            return abs((dt - target).days)

        return sorted(items, key=day_diff)

    def get_signed_urls(
        self, item: Dict
    ) -> Tuple[str, str]:
        """Get signed VV + VH URLs."""
        assets = item.get('assets', {})
        vv_url = assets.get('vv', {}).get('href', '')
        vh_url = assets.get('vh', {}).get('href', '')
        return (
            self.token_mgr.sign(vv_url),
            self.token_mgr.sign(vh_url)
        )


# ── SAR Patch Reader ──────────────────────────────────────────
class SARPatchReader:
    """
    Reads SAR patches from open COG files.
    Opens VV + VH files ONCE → reads many patches.
    Serial reading (rasterio remote not thread-safe).
    Reprojects S2 bounds (32631) → S1 CRS (32632).
    """

    def __init__(self, vv_url: str, vh_url: str, scene_id: str):
        self.vv_url   = vv_url
        self.vh_url   = vh_url
        self.scene_id = scene_id
        self._vv_src  = None
        self._vh_src  = None
        self._s1_crs  = None

    def __enter__(self):
        self._vv_src = rasterio.open(self.vv_url)
        self._vh_src = rasterio.open(self.vh_url)
        self._s1_crs = self._vv_src.crs
        log.info(
            f'Opened S1 COG: {self.scene_id[:50]}'
            f' | CRS: {self._s1_crs}'
            f' | Shape: {self._vv_src.height}x{self._vv_src.width}'
        )
        return self

    def __exit__(self, *args):
        if self._vv_src:
            self._vv_src.close()
        if self._vh_src:
            self._vh_src.close()

    def read_patch(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
    ) -> Optional[np.ndarray]:
        """
        Read one [2, 256, 256] SAR patch.

        Steps:
          1. Reproject bounds from EPSG:32631 → S1 CRS
          2. Compute pixel window in S1 scene
          3. Read + resample to 256x256
          4. Convert to dB
          5. Return [VV_dB, VH_dB] stack

        Returns None if too much nodata or out of bounds.
        """
        # Step 1: Reproject S2 bounds → S1 CRS
        try:
            x_min_s, y_min_s, x_max_s, y_max_s = \
                reproject_bounds(
                    x_min, y_min, x_max, y_max,
                    'EPSG:32631', self._s1_crs
                )
        except Exception as e:
            log.debug(f'Reproject error: {e}')
            return None

        bands = []
        for src, pol in [
            (self._vv_src, 'VV'),
            (self._vh_src, 'VH')
        ]:
            try:
                # Step 2: Compute pixel window
                row_min, col_min = src.index(
                    x_min_s, y_max_s
                )
                row_max, col_max = src.index(
                    x_max_s, y_min_s
                )

                row_min = max(0, row_min)
                col_min = max(0, col_min)
                row_max = min(src.height, row_max)
                col_max = min(src.width,  col_max)

                if row_max <= row_min or col_max <= col_min:
                    return None

                win = Window(
                    col_min, row_min,
                    col_max - col_min,
                    row_max - row_min
                )

                # Step 3: Read + resample
                data = src.read(
                    1,
                    window=win,
                    out_shape=(PATCH_SIZE, PATCH_SIZE),
                    resampling=Resampling.bilinear
                )

                # Check nodata fraction
                nodata_frac = float(
                    (data == NODATA_VAL).sum()
                ) / data.size
                if nodata_frac > NODATA_MAX:
                    return None

                # Step 4: Convert to dB
                bands.append(dn_to_db(data))

            except Exception as e:
                log.debug(f'{pol} read error: {e}')
                return None

        if len(bands) != 2:
            return None

        # Step 5: Stack [2, 256, 256]
        return np.stack(bands, axis=0)


# ── SAR Pipeline ──────────────────────────────────────────────
class SARPipeline:
    """
    Full SAR pipeline for all study areas.

    For each area:
      1. Load S2 patch index (patch locations)
      2. Group patches by S2 date
      3. Find best S1 scene per date
      4. Open COG files ONCE per scene
      5. Read patches serially (COG thread safety)
      6. Upload patches to S3 in parallel
    """

    def __init__(self, cfg: dict):
        self.cfg           = cfg
        self.output_bucket = cfg['aws']['bucket']
        self.output_prefix = cfg['aws']['prefix'] + '/sentinel1'
        self.s3            = boto3.client(
            's3', region_name='us-east-1',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
        )
        self.token_mgr    = TokenManager()
        self.scene_finder = S1SceneFinder(self.token_mgr)

    def load_s2_index(self, area: str) -> List[Dict]:
        """Load S2 patch index from S3."""
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
        patch_data: np.ndarray,
        tile: str,
        s1_date: str,
        s2_date: str,
        row: int,
        col: int,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        scene_id: str,
    ) -> Optional[str]:
        """Upload one SAR patch GeoTIFF to S3."""
        patch_id = (
            f"{tile}_{s1_date}_"
            f"r{row:05d}_c{col:05d}"
        )
        s3_key = (
            f"{self.output_prefix}/{tile}/"
            f"{s1_date}/{patch_id}.tif"
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
                count=2,
                dtype='float32',
                crs='EPSG:32631',
                transform=transform,
            ) as dst:
                dst.write(patch_data)
                dst.update_tags(
                    patch_id=patch_id,
                    s1_date=s1_date,
                    s2_date=s2_date,
                    tile=tile,
                    bands='VV_dB,VH_dB',
                    scene=scene_id,
                )
            self.s3.upload_file(
                tmp_path, self.output_bucket, s3_key
            )
            return patch_id
        except Exception as e:
            log.error(f'Upload failed {patch_id}: {e}')
            return None
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def process_date_batch(
        self,
        reader: SARPatchReader,
        entries: List[Dict],
        tile: str,
        s1_date: str,
        s2_date: str,
        scene_id: str,
    ) -> List[Dict]:
        """
        Process all patches for one date.
        Read one patch → upload immediately → free memory.
        Peak RAM: ~10MB per patch (safe for all machines).
        """
        index_entries = []

        for i, entry in enumerate(entries):
            row = entry['row']
            col = entry['col']

            x_min, y_min, x_max, y_max = compute_patch_bounds(
                tile, row, col
            )

            # Read one patch
            patch_data = reader.read_patch(
                x_min, y_min, x_max, y_max
            )
            if patch_data is None:
                continue

            # Upload immediately
            try:
                patch_id = self.upload_patch(
                    patch_data, tile, s1_date, s2_date,
                    row, col,
                    x_min, y_min, x_max, y_max,
                    scene_id
                )
                if patch_id:
                    index_entries.append({
                        'patch_id': patch_id,
                        'tile':     tile,
                        's1_date':  s1_date,
                        's2_date':  s2_date,
                        'row':      row,
                        'col':      col,
                        'scene_id': scene_id,
                        's3_key': (
                            f"{self.output_prefix}/{tile}/"
                            f"{s1_date}/{patch_id}.tif"
                        ),
                    })
            except Exception as e:
                log.error(f'Upload error: {e}')

            # Free memory immediately
            del patch_data

            if (i + 1) % 100 == 0:
                log.info(
                    f'  {i+1}/{len(entries)} processed, '
                    f'{len(index_entries)} uploaded'
                )

        return index_entries

    def process_area(self, area: str) -> Dict:
        """Full SAR pipeline for one study area."""
        tile = STUDY_TILES[area]
        bbox = STUDY_BBOXES[area]

        log.info(f'\n{"="*60}')
        log.info(f'SAR Pipeline: {area} ({tile})')
        log.info(f'{"="*60}')

        # Load S2 index
        s2_index = self.load_s2_index(area)

        # Group by S2 date
        date_groups: Dict[str, List[Dict]] = {}
        for entry in s2_index:
            d = entry['date']
            date_groups.setdefault(d, []).append(entry)

        log.info(f'Unique S2 dates: {len(date_groups)}')

        all_index = []

        for s2_date, entries in sorted(date_groups.items()):
            log.info(
                f'\nS2 date: {s2_date} '
                f'({len(entries)} patches)'
            )

            # Find best S1 scene
            scenes = self.scene_finder.find_scenes_for_date(
                bbox, s2_date
            )
            if not scenes:
                log.warning(f'No S1 scene for {s2_date}')
                continue

            scene   = scenes[0]
            s1_date = scene['properties']['datetime'][:10]
            day_gap = abs((
                datetime.strptime(s1_date, '%Y-%m-%d') -
                datetime.strptime(s2_date, '%Y-%m-%d')
            ).days)

            log.info(
                f'Best S1: {scene["id"][:55]}'
                f' (day gap: {day_gap})'
            )

            # Get signed URLs
            vv_url, vh_url = \
                self.scene_finder.get_signed_urls(scene)

            # Open COG files once + process all patches
            with SARPatchReader(
                vv_url, vh_url, scene['id']
            ) as reader:
                entries_done = self.process_date_batch(
                    reader, entries,
                    tile, s1_date, s2_date,
                    scene['id'],
                )
                all_index.extend(entries_done)

            log.info(
                f'Date {s2_date}: '
                f'{len(entries_done)}/{len(entries)} uploaded'
            )
            log.info(
                f'Running total: {len(all_index)} patches'
            )

        # Save index to S3
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
            f'✅ {area}: {len(all_index)} SAR patches'
        )
        log.info(
            f'   Index: s3://{self.output_bucket}/{index_key}'
        )
        log.info(f'{"="*60}')

        return {'area': area, 'n_patches': len(all_index)}


# ── Main ───────────────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='OmniGeoFusion SAR pipeline'
    )
    parser.add_argument(
        '--config', default='configs/data_config.yaml'
    )
    parser.add_argument(
        '--area',
        choices=[
            'amsterdam', 'rotterdam',
            'flevoland', 'all'
        ],
        default='all'
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )

    cfg      = yaml.safe_load(open(args.config))
    pipeline = SARPipeline(cfg)
    areas    = (
        list(STUDY_TILES.keys()) if args.area == 'all'
        else [args.area]
    )

    total   = 0
    results = {}
    for area in areas:
        result        = pipeline.process_area(area)
        results[area] = result
        total        += result['n_patches']

    log.info(f'\n{"="*60}')
    log.info(f'SAR PIPELINE COMPLETE')
    log.info(f'Total SAR patches: {total}')
    for area, r in results.items():
        log.info(f'  {area}: {r["n_patches"]}')
    log.info(f'{"="*60}')


if __name__ == '__main__':
    main()
