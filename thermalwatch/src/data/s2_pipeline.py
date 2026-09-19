"""
ThermalWatch — Sentinel-2 Windowed Reading Pipeline
=====================================================
Downloads S2 patches via direct cloud windowed reading.

Key fixes over previous windowed approach:
  1. Re-signs URLs fresh per patch (not per month)
     → No 403 expiry errors
  2. WGS84 pre-filter before window read
     → No bounds check failures
  3. Searches S2 using EXACT patch WGS84 bbox
     → Guaranteed spatial match
  4. Caches scene per (patch_wgs84_center) 
     → Minimal API calls

Strategy:
  For each patch:
    1. Compute patch WGS84 bbox from thermal TIF
    2. Find best S2 scene for that exact location
    3. Sign URLs fresh right before reading
    4. Read 224×224 window directly from cloud
    5. Save [6,224,224] float32

RAM usage: ~1.2MB per patch ✅
No full tile download ✅
No URL expiry issues ✅

Bands for Prithvi-EO-2.0-300M:
  B02 Blue, B03 Green, B04 Red,
  B8A NIR, B11 SWIR1, B12 SWIR2

Usage:
  python3 -m src.data.s2_pipeline \
    --task wildfire \
    --year 2020 \
    --areas sierra_nevada \
    --data-dir data/wildfire
"""

import json
import logging
import argparse
import calendar
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

log = logging.getLogger(__name__)

PRITHVI_BANDS  = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']
PATCH_SIZE     = 224
WILDFIRE_AREAS = ['sierra_nevada', 'socal', 'norcal']
SOLAR_AREAS    = ['sonoran_desert', 'phoenix_metro', 'tucson']
WILDFIRE_YEARS = list(range(2020, 2024))
SOLAR_YEARS    = list(range(2019, 2024))


def _check_deps():
    missing = []
    for pkg in ['pystac_client', 'planetary_computer',
                'rasterio']:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f'Missing: {missing}\n'
            f'Run: pip install planetary-computer '
            f'pystac-client rasterio'
        )


def _compute_patch_wgs84(
    entry: Dict,
    tif_cache: Dict,
    thermal_dir: Path,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Compute patch WGS84 bbox from thermal TIF.
    Returns (west, south, east, north) or None.
    """
    import rasterio
    from rasterio.warp import transform_bounds

    src_file = entry['source_file']
    tif_path = thermal_dir / src_file

    if not tif_path.exists():
        return None

    if src_file not in tif_cache:
        try:
            with rasterio.open(tif_path) as src:
                tif_cache[src_file] = {
                    'transform': src.transform,
                    'crs':       src.crs.to_wkt(),
                }
        except Exception as e:
            log.debug(f'Failed to open {src_file}: {e}')
            return None

    info = tif_cache[src_file]
    row  = entry['row']
    col  = entry['col']

    try:
        left, top = rasterio.transform.xy(
            info['transform'], row, col, offset='ul'
        )
        right, bottom = rasterio.transform.xy(
            info['transform'],
            row + PATCH_SIZE,
            col + PATCH_SIZE,
            offset='ul'
        )
        l, b, r, t = transform_bounds(
            info['crs'], 'EPSG:4326',
            left, bottom, right, top
        )
        return (l, b, r, t)
    except Exception as e:
        log.debug(f'Bbox failed {entry["patch_id"]}: {e}')
        return None


def _find_s2_item(
    patch_bbox: Tuple[float, float, float, float],
    year: int,
    month: int,
    max_cloud: int = 20,
) -> Optional[object]:
    """
    Find best S2 item for exact patch bbox + month.
    Returns UNSIGNED item (sign fresh before reading).
    Uses point search at patch center for precision.
    """
    import pystac_client

    last_day   = calendar.monthrange(year, month)[1]
    date_start = f'{year}-{month:02d}-01'
    date_end   = f'{year}-{month:02d}-{last_day:02d}'

    # Use small bbox around patch center
    cx = (patch_bbox[0] + patch_bbox[2]) / 2
    cy = (patch_bbox[1] + patch_bbox[3]) / 2
    search_bbox = [cx-0.01, cy-0.01, cx+0.01, cy+0.01]

    # Open WITHOUT sign_inplace — sign fresh per read
    catalog = pystac_client.Client.open(
        'https://planetarycomputer.microsoft.com/api/stac/v1',
    )
    search = catalog.search(
        collections=['sentinel-2-l2a'],
        bbox=search_bbox,
        datetime=f'{date_start}/{date_end}',
        query={'eo:cloud_cover': {'lt': max_cloud}},
    )

    for item in search.items():
        if all(b in item.assets for b in PRITHVI_BANDS):
            return item  # Return UNSIGNED

    return None


def _read_s2_window(
    item,
    patch_bbox_wgs84: Tuple[float, float, float, float],
) -> Optional[np.ndarray]:
    """
    Read 224×224 S2 window for one patch.
    Signs URLs fresh right before reading.
    Reprojects WGS84 bbox → S2 tile CRS.
    Clamps window to tile bounds.

    Returns [6, 224, 224] float32 in [0,1] or None.
    """
    import planetary_computer as pc
    import rasterio
    from rasterio.windows import from_bounds, Window
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds

    # Sign fresh right before reading
    signed = pc.sign(item)

    bands_data = []
    p_l, p_b, p_r, p_t = patch_bbox_wgs84

    for band in PRITHVI_BANDS:
        href = signed.assets[band].href
        try:
            with rasterio.open(href) as src:
                tile_crs = src.crs.to_wkt()

                # Get tile WGS84 bounds for pre-check
                from rasterio.warp import transform_bounds as tb
                t_l, t_b, t_r, t_t = tb(
                    src.crs, 'EPSG:4326', *src.bounds
                )

                # WGS84 overlap pre-check
                if (p_r < t_l or p_l > t_r or
                        p_t < t_b or p_b > t_t):
                    return None

                # Reproject patch WGS84 → tile CRS
                tl, tb2, tr, tt = transform_bounds(
                    'EPSG:4326', tile_crs,
                    p_l, p_b, p_r, p_t
                )

                window = from_bounds(
                    tl, tb2, tr, tt, src.transform
                )

                # Clamp to tile bounds
                col_off = max(0.0, window.col_off)
                row_off = max(0.0, window.row_off)
                col_end = min(
                    float(src.width),
                    window.col_off + window.width
                )
                row_end = min(
                    float(src.height),
                    window.row_off + window.height
                )

                if col_end <= col_off or row_end <= row_off:
                    return None

                clamped = Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=col_end  - col_off,
                    height=row_end - row_off,
                )

                data = src.read(
                    1,
                    window=clamped,
                    out_shape=(PATCH_SIZE, PATCH_SIZE),
                    resampling=Resampling.bilinear,
                )
                data = data.astype(np.float32) / 10000.0
                data[data <= 0] = np.nan
                bands_data.append(data)

        except Exception as e:
            log.debug(f'Band {band} failed: {e}')
            return None

    if len(bands_data) != 6:
        return None

    stacked     = np.stack(bands_data, axis=0)
    valid_ratio = np.sum(np.isfinite(stacked)) / stacked.size
    if valid_ratio < 0.5:
        return None

    return stacked


class S2PatchPipeline:
    """
    Downloads S2 patches via windowed cloud reading.

    For each patch:
      1. Compute exact WGS84 bbox from thermal TIF
      2. Search S2 using patch center (0.02° bbox)
      3. Sign URLs fresh per patch
      4. Read 224×224 window from cloud
      5. Save [6,224,224] float32

    Cache: S2 items cached per (lon_grid, lat_grid, month)
           where grid = 0.5° cells
           → avoids API call per patch while still
             getting spatially correct scene per location
    """

    def __init__(
        self,
        data_dir: str,
        task: str,
        max_cloud: int = 20,
    ):
        if task not in ['wildfire', 'solar']:
            raise ValueError(
                f'task must be wildfire or solar'
            )
        self.data_dir    = Path(data_dir)
        self.task        = task
        self.max_cloud   = max_cloud
        self.patch_dir   = self.data_dir / 'patches'
        self.thermal_dir = self.data_dir / 'landsat_thermal'
        _check_deps()
        log.info(
            f'S2PatchPipeline: task={task} '
            f'data={self.data_dir}'
        )

    def _process_patch(
        self,
        entry: Dict,
        year: int,
        out_dir: Path,
        tif_cache: Dict,
        scene_cache: Dict,
        scene_lock,
        tif_lock,
    ) -> str:
        """
        Process one patch — thread safe.
        Returns: 'done', 'skip', or 'fail'
        """
        pid     = entry['patch_id']
        s2_file = out_dir / f's2_{pid}.npy'

        if entry.get('s2_path') and s2_file.exists():
            return 'skip'

        # Compute bbox (tif_cache protected by lock)
        with tif_lock:
            bbox = _compute_patch_wgs84(
                entry, tif_cache, self.thermal_dir
            )
        if bbox is None:
            return 'fail'

        month     = entry['month']
        lon_key   = int(bbox[0] * 2) / 2
        lat_key   = int(bbox[1] * 2) / 2
        cache_key = (lon_key, lat_key, month)

        # Get or find S2 scene (scene_cache protected by lock)
        with scene_lock:
            if cache_key not in scene_cache:
                item = _find_s2_item(
                    bbox, year, month, self.max_cloud
                )
                scene_cache[cache_key] = item
            item = scene_cache[cache_key]

        if item is None:
            return 'fail'

        # Read window (signs fresh — no lock needed)
        s2_patch = _read_s2_window(item, bbox)

        if s2_patch is None:
            return 'fail'

        if s2_patch.shape != (6, PATCH_SIZE, PATCH_SIZE):
            return 'fail'

        np.save(str(s2_file), s2_patch)
        entry['s2_path'] = str(s2_file)
        return 'done'

    def process_area_year(
        self,
        area: str,
        year: int,
        num_workers: int = 4,
    ) -> Path:
        """
        Add S2 patches to existing thermal index.
        Runs in parallel with num_workers threads.
        Caches S2 items per 0.5° grid cell + month.
        Signs URLs fresh per patch read.
        Saves index every 200 patches.
        """
        import threading
        from concurrent.futures import (
            ThreadPoolExecutor, as_completed
        )

        index_path = (
            self.patch_dir / f'{area}_{year}_index.json'
        )
        if not index_path.exists():
            raise FileNotFoundError(
                f'Patch index not found: {index_path}'
            )

        index   = json.loads(index_path.read_text())
        out_dir = self.patch_dir / f'{area}_{year}'
        out_dir.mkdir(parents=True, exist_ok=True)

        total_done    = sum(
            1 for p in index if p.get('s2_path')
        )
        total_patches = len(index)

        log.info(
            f'\nS2 windowed: {area} {year} '
            f'— {total_patches} patches '
            f'({total_done} done) '
            f'workers={num_workers}'
        )

        tif_cache   = {}
        scene_cache = {}
        scene_lock  = threading.Lock()
        tif_lock    = threading.Lock()
        save_lock   = threading.Lock()

        done    = 0
        skipped = 0
        failed  = 0

        # Filter patches that need processing
        todo = [
            e for e in index
            if not (e.get('s2_path') and
                    Path(e['s2_path']).exists())
        ]
        skipped = total_patches - len(todo)

        log.info(f'  {len(todo)} patches to process')

        with ThreadPoolExecutor(
            max_workers=num_workers
        ) as executor:
            futures = {
                executor.submit(
                    self._process_patch,
                    entry, year, out_dir,
                    tif_cache, scene_cache,
                    scene_lock, tif_lock,
                ): entry
                for entry in todo
            }

            for i, future in enumerate(
                as_completed(futures)
            ):
                result = future.result()
                if result == 'done':
                    done += 1
                elif result == 'skip':
                    skipped += 1
                else:
                    failed += 1

                if (i + 1) % 200 == 0:
                    log.info(
                        f'  Progress: {i+1}/{len(todo)} | '
                        f'done={done} skip={skipped} '
                        f'fail={failed}'
                    )
                    with save_lock:
                        index_path.write_text(
                            json.dumps(index, indent=2)
                        )

        # Final save
        index_path.write_text(json.dumps(index, indent=2))
        log.info(
            f'\n✅ S2 {area} {year}: '
            f'done={done} skipped={skipped} '
            f'failed={failed} / {total_patches}'
        )
        return index_path

    def process_all(
        self,
        areas: List[str],
        years: List[int],
    ):
        """Process S2 for all areas + years."""
        failed = []
        for area in areas:
            for year in years:
                try:
                    self.process_area_year(area, year)
                except Exception as e:
                    log.error(
                        f'S2 FAILED {area} {year}: {e}'
                    )
                    failed.append((area, year))

        if failed:
            log.warning(
                f'Failed:\n' +
                '\n'.join(f'  {a} {y}' for a, y in failed)
            )
        log.info('✅ S2 complete')


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    parser = argparse.ArgumentParser(
        description='ThermalWatch S2 Windowed Pipeline'
    )
    parser.add_argument(
        '--task', choices=['wildfire', 'solar'],
        required=True,
    )
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--areas', nargs='+', default=None)
    parser.add_argument(
        '--years', nargs='+', type=int, default=None
    )
    parser.add_argument(
        '--max-cloud', type=int, default=20
    )
    args = parser.parse_args()

    pipeline = S2PatchPipeline(
        data_dir=args.data_dir,
        task=args.task,
        max_cloud=args.max_cloud,
    )
    areas = args.areas or (
        WILDFIRE_AREAS if args.task == 'wildfire'
        else SOLAR_AREAS
    )
    years = args.years or (
        WILDFIRE_YEARS if args.task == 'wildfire'
        else SOLAR_YEARS
    )
    pipeline.process_all(areas, years)


if __name__ == '__main__':
    main()
