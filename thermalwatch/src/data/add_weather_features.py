"""
ThermalWatch — Add Weather Features to Patch Indexes
======================================================
Re-downloads ERA5 (wildfire) and NSRDB (solar) weather
data and stores weather features directly in patch indexes.

Weather vector [5] per patch:
  Wildfire (ERA5):
    [t2m, u10, v10, humidity, precip]
    monthly mean values for patch location

  Solar (NSRDB):
    [ghi, dni, dhi, wind_speed, air_temperature]
    monthly mean values for patch location

Usage:
  python3 -m src.data.add_weather_features \
    --task wildfire \
    --data-dir data/wildfire

  python3 -m src.data.add_weather_features \
    --task solar \
    --data-dir data/solar
"""

import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

log = logging.getLogger(__name__)

WILDFIRE_AREAS = ['sierra_nevada', 'socal', 'norcal']
SOLAR_AREAS    = ['sonoran_desert', 'phoenix_metro', 'tucson']
WILDFIRE_YEARS = list(range(2020, 2024))
SOLAR_YEARS    = list(range(2019, 2024))

# ERA5 variables
ERA5_VARS = [
    '2m_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'relative_humidity',
    'total_precipitation',
]

# NSRDB variables
NSRDB_VARS = [
    'GHI', 'DNI', 'DHI',
    'Wind Speed', 'Temperature'
]


def _get_era5_features(
    era5_path: Path,
    lat: float,
    lon: float,
) -> Optional[List[float]]:
    """
    Extract monthly mean ERA5 features for a location.
    Returns [t2m, u10, v10, humidity, precip] or None.
    Handles both .nc and .zip (CDS new format) files.
    """
    try:
        import netCDF4 as nc
        import numpy as np
        import zipfile, tempfile, os

        # CDS now returns zip archives containing NetCDF
        actual_path = era5_path
        tmp_dir     = None

        if zipfile.is_zipfile(str(era5_path)):
            tmp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(str(era5_path)) as zf:
                zf.extractall(tmp_dir)
                # Find the .nc file inside
                nc_files = [
                    f for f in os.listdir(tmp_dir)
                    if f.endswith('.nc')
                ]
                if not nc_files:
                    return None
                actual_path = Path(tmp_dir) / nc_files[0]

        ds = nc.Dataset(str(actual_path))

        # Find nearest grid point
        lats = ds.variables['latitude'][:]
        lons = ds.variables['longitude'][:]
        lat_idx = int(np.argmin(np.abs(lats - lat)))
        lon_idx = int(np.argmin(np.abs(lons - lon)))

        features = []
        # Extract 4 ERA5 variables
        era5_vars = ['t2m', 'u10', 'v10', 'd2m']
        raw = {}
        for var in era5_vars:
            if var in ds.variables:
                data = ds.variables[var][
                    :, lat_idx, lon_idx
                ]
                raw[var] = float(np.nanmean(data))
            else:
                raw[var] = 0.0

        ds.close()

        # Compute wind speed from u10 + v10
        wind_speed = float(np.sqrt(
            raw['u10']**2 + raw['v10']**2
        ))

        # Return 5-value weather vector
        features = [
            raw['t2m'],    # 2m temperature (K)
            raw['u10'],    # 10m wind U (m/s)
            raw['v10'],    # 10m wind V (m/s)
            raw['d2m'],    # 2m dewpoint (K)
            wind_speed,    # wind speed magnitude (m/s)
        ]

        # Cleanup temp dir if we unzipped
        if tmp_dir:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return features

    except Exception as e:
        log.debug(f'ERA5 feature extraction failed: {e}')
        return None


def _get_nsrdb_features(
    nsrdb_dir: Path,
    lat: float,
    lon: float,
    area: str,
    year: int,
) -> Optional[List[float]]:
    """
    Extract monthly mean NSRDB features for a location.
    Finds nearest grid point in NSRDB CSV files.
    Returns [ghi, dni, dhi, wind, temp] or None.
    """
    try:
        import pandas as pd

        # Find nearest NSRDB grid point
        nsrdb_files = list(nsrdb_dir.glob(
            f'{area}_*_{year}.csv'
        ))
        if not nsrdb_files:
            return None

        best_file = None
        best_dist = float('inf')

        for f in nsrdb_files:
            parts = f.stem.split('_')
            # Format: {area}_{lat}_{lon}_{year}
            # area can be multi-part e.g. sonoran_desert
            # so parse from end
            try:
                file_year = int(parts[-1])
                file_lon  = float(parts[-2])
                file_lat  = float(parts[-3])
                dist = (
                    (file_lat - lat)**2 +
                    (file_lon - lon)**2
                ) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_file = f
            except Exception:
                continue

        if best_file is None:
            return None

        # Read CSV — skip first 2 metadata rows
        df = pd.read_csv(best_file, skiprows=2)

        features = []
        for var in NSRDB_VARS:
            if var in df.columns:
                features.append(float(df[var].mean()))
            else:
                features.append(0.0)

        return features

    except Exception as e:
        log.debug(f'NSRDB feature extraction failed: {e}')
        return None


class WeatherFeatureAdder:
    """
    Downloads weather data and adds weather_features [5]
    to each patch index entry.

    For wildfire: uses ERA5 NetCDF files
    For solar:    uses NSRDB CSV files

    Skips patches that already have weather_features.
    Saves index after each area/year.
    """

    def __init__(
        self,
        data_dir: str,
        task: str,
    ):
        if task not in ['wildfire', 'solar']:
            raise ValueError(
                f'task must be wildfire or solar'
            )
        self.data_dir  = Path(data_dir)
        self.task      = task
        self.patch_dir = self.data_dir / 'patches'

    def _download_era5(
        self,
        area: str,
        year: int,
        month: int,
    ) -> Optional[Path]:
        """Download ERA5 for one area/month if not exists."""
        import sys
        sys.path.insert(0, str(
            Path(__file__).parent.parent.parent
        ))
        from src.data.wildfire.wildfire_pipeline import (
            ERA5Pipeline
        )
        pipeline = ERA5Pipeline(self.data_dir)
        try:
            return pipeline.download(area, year, month)
        except Exception as e:
            log.error(f'ERA5 download failed: {e}')
            return None

    def _download_nsrdb(
        self,
        area: str,
        lat: float,
        lon: float,
        year: int,
    ) -> Optional[Path]:
        """Download NSRDB for one grid point if not exists."""
        import sys
        sys.path.insert(0, str(
            Path(__file__).parent.parent.parent
        ))
        from src.config import get_config
        from src.data.solar.solar_pipeline import NSRDBPipeline
        cfg      = get_config()
        pipeline = NSRDBPipeline(
            self.data_dir,
            cfg['NREL_API_KEY'],
            cfg['THERMALWATCH_EMAIL'],
        )
        try:
            return pipeline.download(area, lat, lon, year)
        except Exception as e:
            log.error(f'NSRDB download failed: {e}')
            return None

    def _get_patch_wgs84(
        self,
        entry: Dict,
        tif_cache: Dict,
    ) -> Optional[Tuple[float, float]]:
        """Get patch center lat/lon in WGS84."""
        import rasterio
        from rasterio.warp import transform_bounds

        src_file = entry['source_file']
        tif_path = (
            self.data_dir / 'landsat_thermal' / src_file
        )

        if not tif_path.exists():
            return None

        if src_file not in tif_cache:
            try:
                with rasterio.open(tif_path) as src:
                    tif_cache[src_file] = {
                        'transform': src.transform,
                        'crs':       src.crs.to_wkt(),
                    }
            except Exception:
                return None

        info = tif_cache[src_file]
        row  = entry['row']
        col  = entry['col']

        try:
            cx, cy = rasterio.transform.xy(
                info['transform'],
                row + 112, col + 112,
            )
            l, b, r, t = transform_bounds(
                info['crs'], 'EPSG:4326',
                cx - 1, cy - 1, cx + 1, cy + 1
            )
            return ((b + t) / 2, (l + r) / 2)  # lat, lon
        except Exception:
            return None

    def process_area_year(
        self,
        area: str,
        year: int,
        months: List[int],
    ) -> Path:
        """Add weather features to one area/year index."""
        index_path = (
            self.patch_dir / f'{area}_{year}_index.json'
        )
        if not index_path.exists():
            raise FileNotFoundError(
                f'Index not found: {index_path}'
            )

        index = json.loads(index_path.read_text())
        era5_dir  = self.data_dir / 'era5'
        nsrdb_dir = self.data_dir / 'nsrdb'
        tif_cache: Dict = {}

        # Download weather data for all months first
        if self.task == 'wildfire':
            log.info(
                f'Downloading ERA5: {area} {year}...'
            )
            for month in months:
                self._download_era5(area, year, month)
        else:
            log.info(
                f'Downloading NSRDB: {area} {year}...'
            )
            # NSRDB grid points already downloaded
            # just need to verify files exist

        done    = 0
        skipped = 0
        failed  = 0
        total   = len(index)

        log.info(
            f'Adding weather features: '
            f'{area} {year} — {total} patches'
        )

        for i, entry in enumerate(index):
            # Skip if already has weather features
            if entry.get('weather_features'):
                skipped += 1
                continue

            month = entry['month']

            if self.task == 'wildfire':
                era5_path = (
                    era5_dir /
                    f'{area}_{year}_{month:02d}.nc'
                )
                if not era5_path.exists():
                    failed += 1
                    continue

                # Get patch location
                loc = self._get_patch_wgs84(
                    entry, tif_cache
                )
                if loc is None:
                    failed += 1
                    continue

                lat, lon = loc
                features = _get_era5_features(
                    era5_path, lat, lon
                )

            else:  # solar
                loc = self._get_patch_wgs84(
                    entry, tif_cache
                )
                if loc is None:
                    failed += 1
                    continue

                lat, lon = loc
                features = _get_nsrdb_features(
                    nsrdb_dir, lat, lon, area, year
                )

            if features is None or len(features) != 5:
                failed += 1
                continue

            entry['weather_features'] = features
            done += 1

            if (i + 1) % 1000 == 0:
                log.info(
                    f'  Progress: {i+1}/{total} | '
                    f'done={done} skip={skipped} '
                    f'fail={failed}'
                )
                index_path.write_text(
                    json.dumps(index, indent=2)
                )

        index_path.write_text(json.dumps(index, indent=2))
        log.info(
            f'✅ {area} {year}: '
            f'done={done} skip={skipped} fail={failed}'
        )
        return index_path

    def process_all(
        self,
        areas: List[str],
        years: List[int],
        months: List[int],
    ):
        """Process all areas + years."""
        failed = []
        for area in areas:
            for year in years:
                try:
                    self.process_area_year(
                        area, year, months
                    )
                except Exception as e:
                    log.error(
                        f'FAILED {area} {year}: {e}'
                    )
                    failed.append((area, year))

        if failed:
            log.warning(
                f'Failed:\n' +
                '\n'.join(f'  {a} {y}' for a, y in failed)
            )
        log.info('✅ Weather features complete')


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    parser = argparse.ArgumentParser(
        description='Add weather features to patch indexes'
    )
    parser.add_argument(
        '--task', choices=['wildfire', 'solar'],
        required=True,
    )
    parser.add_argument('--data-dir', required=True)
    args = parser.parse_args()

    if args.task == 'wildfire':
        areas  = WILDFIRE_AREAS
        years  = WILDFIRE_YEARS
        months = [6, 7, 8, 9, 10, 11]
    else:
        areas  = SOLAR_AREAS
        years  = SOLAR_YEARS
        months = [3, 4, 5, 6, 7, 8, 9, 10]

    adder = WeatherFeatureAdder(
        data_dir=args.data_dir,
        task=args.task,
    )
    adder.process_all(areas, years, months)


if __name__ == '__main__':
    main()
