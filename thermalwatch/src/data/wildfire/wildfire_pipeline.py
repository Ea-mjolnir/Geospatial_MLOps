"""
ThermalWatch — Wildfire Data Pipeline
=======================================
Downloads wildfire data for California study areas.
All data stored locally under data/wildfire/

Data Sources:
  1. VIIRS Active Fire (NASA FIRMS) — thermal hotspots
     Historical endpoint: VIIRS_SNPP_SP (not NRT)
  2. Landsat 8/9 Band 10 — thermal imagery (Celsius)
     Source: Microsoft Planetary Computer
  3. NIFC — historical fire perimeters (labels)
     Source: NIFC ArcGIS REST API
  4. ERA5 — weather (temperature, wind, humidity)
     Source: Copernicus CDS API

Study Areas: sierra_nevada, socal, norcal (California)
Historical Range: 2018-2023 (fire seasons Jun-Nov)

API Keys Required:
  FIRMS: https://firms.modaps.eosdis.nasa.gov/api/
  ERA5:  https://cds.climate.copernicus.eu (free account)

Usage:
  python3 -m src.data.wildfire.wildfire_pipeline \
    --data-dir data/wildfire \
    --firms-key YOUR_FIRMS_KEY \
    --areas sierra_nevada socal norcal \
    --years 2020 2021 2022
"""

import json
import logging
import argparse
import calendar
import requests
import numpy as np
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── Study Areas ───────────────────────────────────────────────
STUDY_AREAS = {
    'sierra_nevada': {
        'bbox':  [-122.0, 36.5, -118.0, 40.0],  # W,S,E,N
        'state': 'CA',
    },
    'socal': {
        'bbox':  [-119.0, 33.5, -116.0, 35.5],  # W,S,E,N
        'state': 'CA',
    },
    'norcal': {
        'bbox':  [-123.0, 38.0, -120.0, 40.5],  # W,S,E,N
        'state': 'CA',
    },
}

TRAIN_YEARS        = list(range(2020, 2024))  # 2020-2023 — bulk of NIFC data
FIRE_SEASON_MONTHS = [6, 7, 8, 9, 10, 11]


def month_date_range(year: int, month: int) -> Tuple[str, str]:
    """Return (start, end) date strings for a given month."""
    last_day = calendar.monthrange(year, month)[1]
    return (
        f'{year}-{month:02d}-01',
        f'{year}-{month:02d}-{last_day:02d}',
    )


# ── 1. VIIRS Pipeline ─────────────────────────────────────────
class VIIRSPipeline:
    """
    Downloads VIIRS active fire hotspots from NASA FIRMS.

    Uses VIIRS_SNPP_SP (standard product) for historical data
    — NOT NRT which only covers last 7 days.

    Output: CSV with columns:
      latitude, longitude, bright_ti4, bright_ti5,
      scan, track, acq_date, acq_time, satellite,
      confidence, version, bright_t31, frp, daynight

    Stored: data/wildfire/viirs/{area}_{year}_{month:02d}.csv
    API key: https://firms.modaps.eosdis.nasa.gov/api/
    """
    BASE_URL = 'https://firms.modaps.eosdis.nasa.gov/api/area/csv'
    PRODUCT  = 'VIIRS_SNPP_SP'  # Standard Product for historical

    def __init__(self, data_dir: Path, api_key: str):
        if not api_key:
            raise ValueError(
                'FIRMS API key required. '
                'Get one at: '
                'https://firms.modaps.eosdis.nasa.gov/api/'
            )
        self.out_dir = data_dir / 'viirs'
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key

    def download(
        self,
        area: str,
        year: int,
        month: int,
    ) -> Path:
        """
        Download VIIRS hotspots for one area/month.
        Raises on failure — no silent fallback.
        """
        out = self.out_dir / f'{area}_{year}_{month:02d}.csv'
        if out.exists():
            log.info(f'Skip (exists): {out.name}')
            return out

        if area not in STUDY_AREAS:
            raise ValueError(
                f'Unknown area: {area}. '
                f'Choose from: {list(STUDY_AREAS.keys())}'
            )

        cfg        = STUDY_AREAS[area]
        bbox       = cfg['bbox']
        date_start, _ = month_date_range(year, month)

        # FIRMS area API: bbox as W,S,E,N
        url = (
            f'{self.BASE_URL}/{self.api_key}/'
            f'{self.PRODUCT}/'
            f'{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}/'
            f'5/{date_start}'
        )

        log.info(
            f'VIIRS ({self.PRODUCT}): '
            f'{area} {year}-{month:02d}'
        )

        # Retry up to 3 times with increasing timeout
        import time as _time
        last_error = None
        for attempt in range(3):
            try:
                timeout = 60 + attempt * 60  # 60, 120, 180s
                r = requests.get(url, timeout=timeout)

                if r.status_code == 401:
                    raise PermissionError(
                        f'Invalid FIRMS API key: '
                        f'{self.api_key[:8]}...'
                    )
                if r.status_code == 429:
                    log.warning(
                        'FIRMS rate limit — waiting 30s'
                    )
                    _time.sleep(30)
                    continue

                r.raise_for_status()

                if len(r.content) < 10:
                    raise ValueError(
                        f'FIRMS empty response: '
                        f'{area} {year}-{month:02d}'
                    )

                out.write_bytes(r.content)
                n = len(
                    r.content.decode().strip().split('\n')
                ) - 1
                log.info(
                    f'✅ VIIRS: {n} hotspots → {out.name}'
                )
                return out

            except PermissionError:
                raise
            except Exception as e:
                last_error = e
                log.warning(
                    f'VIIRS attempt {attempt+1}/3 failed: {e}'
                )
                if attempt < 2:
                    _time.sleep(10)

        raise RuntimeError(
            f'VIIRS failed after 3 attempts: {last_error}'
        )

    def download_all(
        self,
        area: str,
        years: List[int] = TRAIN_YEARS,
        months: List[int] = FIRE_SEASON_MONTHS,
    ) -> List[Path]:
        """Download VIIRS for all years + fire season months."""
        paths  = []
        total  = len(years) * len(months)
        done   = 0
        failed = []

        log.info(
            f'VIIRS: {area} {years[0]}-{years[-1]} '
            f'({total} months)'
        )

        for year in years:
            for month in months:
                try:
                    path = self.download(area, year, month)
                    paths.append(path)
                except Exception as e:
                    log.error(
                        f'VIIRS FAILED {area} '
                        f'{year}-{month:02d}: {e}'
                    )
                    failed.append((area, year, month))
                done += 1
                log.info(f'  Progress: {done}/{total}')

        if failed:
            log.warning(
                f'VIIRS: {len(failed)} failed downloads:\n'
                + '\n'.join(
                    f'  {a} {y}-{m:02d}'
                    for a, y, m in failed
                )
            )

        log.info(
            f'✅ VIIRS complete: '
            f'{len(paths)}/{total} downloaded'
        )
        return paths


# ── 2. Landsat Thermal Pipeline ───────────────────────────────
class LandsatThermalPipeline:
    """
    Downloads Landsat 8/9 Band 10 thermal imagery.
    Converts raw DN → degrees Celsius using:
      T(K) = 0.00341802 × DN + 149.0
      T(C) = T(K) - 273.15

    Output: single-band GeoTIFF in Celsius.
    Stored: data/wildfire/landsat_thermal/
    Source: Microsoft Planetary Computer (no key needed)

    Requires:
      pip install planetary-computer pystac-client rasterio
    """

    def __init__(self, data_dir: Path):
        self.out_dir = data_dir / 'landsat_thermal'
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._check_deps()

    @staticmethod
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
                f'Missing packages: {missing}\n'
                f'Run: pip install planetary-computer '
                f'pystac-client rasterio'
            )

    def download(
        self,
        area: str,
        year: int,
        month: int,
        max_cloud: int = 30,
        max_scenes: int = 2,
    ) -> List[Path]:
        """
        Download Landsat Band 10 for one area/month.
        Raises on API failure.
        """
        import pystac_client
        import planetary_computer as pc
        import rasterio

        if area not in STUDY_AREAS:
            raise ValueError(f'Unknown area: {area}')

        cfg             = STUDY_AREAS[area]
        bbox            = cfg['bbox']
        date_start, date_end = month_date_range(year, month)
        paths           = []

        catalog = pystac_client.Client.open(
            'https://planetarycomputer.microsoft.com'
            '/api/stac/v1',
            modifier=pc.sign_inplace,
        )
        search = catalog.search(
            collections=['landsat-c2-l2'],
            bbox=bbox,
            datetime=f'{date_start}/{date_end}',
            query={'eo:cloud_cover': {'lt': max_cloud}},
        )
        items = list(search.items())
        log.info(
            f'Landsat: {len(items)} scenes found '
            f'{area} {year}-{month:02d}'
        )

        if not items:
            log.warning(
                f'No Landsat scenes for {area} '
                f'{year}-{month:02d} '
                f'(cloud<{max_cloud}%)'
            )
            return paths

        for item in items[:max_scenes]:
            out = (
                self.out_dir /
                f'{area}_{year}_{month:02d}'
                f'_{item.id}_B10.tif'
            )
            if out.exists():
                paths.append(out)
                continue

            # Handle both Landsat 8/9 (lwir11) and Landsat 7 (lwir)
            if 'lwir11' in item.assets:
                band_key = 'lwir11'
            elif 'lwir' in item.assets:
                band_key = 'lwir'
            else:
                log.warning(
                    f'No thermal band in scene {item.id} '
                    f'— skipping. '
                    f'Available: {list(item.assets.keys())}'
                )
                continue

            signed = pc.sign(item)
            href   = signed.assets[band_key].href

            with rasterio.open(href) as src:
                data    = src.read(1)
                profile = src.profile

            if data.max() == 0:
                raise ValueError(
                    f'Scene {item.id} Band 10 is all zeros '
                    f'— likely corrupt or no data'
                )

            # DN → Kelvin → Celsius
            temp_k = (
                0.00341802 * data.astype(np.float32) + 149.0
            )
            temp_c = temp_k - 273.15

            profile.update(dtype=rasterio.float32, count=1)
            with rasterio.open(out, 'w', **profile) as dst:
                dst.write(temp_c, 1)

            log.info(
                f'✅ Thermal: {out.name} '
                f'(range: {temp_c.min():.1f}°C - '
                f'{temp_c.max():.1f}°C)'
            )
            paths.append(out)

        return paths

    def download_all(
        self,
        area: str,
        years: List[int] = TRAIN_YEARS,
        months: List[int] = FIRE_SEASON_MONTHS,
    ) -> List[Path]:
        """Download Landsat thermal for all years/months."""
        paths  = []
        total  = len(years) * len(months)
        done   = 0
        failed = []

        log.info(
            f'Landsat thermal: {area} '
            f'{years[0]}-{years[-1]} ({total} months)'
        )

        for year in years:
            for month in months:
                try:
                    p = self.download(area, year, month)
                    paths.extend(p)
                except Exception as e:
                    log.error(
                        f'Landsat FAILED {area} '
                        f'{year}-{month:02d}: {e}'
                    )
                    failed.append((area, year, month))
                done += 1
                log.info(f'  Progress: {done}/{total}')

        if failed:
            log.warning(
                f'Landsat: {len(failed)} failed:\n'
                + '\n'.join(
                    f'  {a} {y}-{m:02d}'
                    for a, y, m in failed
                )
            )

        log.info(
            f'✅ Landsat complete: {len(paths)} scenes'
        )
        return paths


# ── 3. NIFC Fire Perimeters Pipeline ──────────────────────────
class NIFCPipeline:
    """
    Downloads historical fire perimeters from NIFC.
    Uses OpenData GeoJSON API — downloads all, filters locally.
    State format: US-CA not CA.

    Output: GeoJSON per year.
    Stored: data/wildfire/nifc/fires_{state}_{year}.geojson
    """
    BASE_URL = (
        'https://opendata.arcgis.com/api/v3/datasets/'
        '5e72b1699bf74eefb3f3aff6f4ba5511_0/downloads/data'
    )

    def __init__(self, data_dir: Path):
        self.out_dir = data_dir / 'nifc'
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._cache  = None

    def _load_all(self) -> list:
        """Download full NIFC dataset (cached in memory)."""
        if self._cache is not None:
            return self._cache
        log.info('Downloading full NIFC dataset...')
        r = requests.get(
            self.BASE_URL,
            params={
                'format':       'geojson',
                'spatialRefId': '4326',
            },
            timeout=120,
        )
        r.raise_for_status()
        self._cache = r.json().get('features', [])
        log.info(
            f'NIFC: {len(self._cache)} total perimeters'
        )
        return self._cache

    def download(
        self,
        year: int,
        state: str = 'CA',
        min_acres: int = 100,
    ) -> Path:
        """
        Filter + save fire perimeters for one year.
        Raises if no fires found.
        """
        out = self.out_dir / f'fires_{state}_{year}.geojson'
        if out.exists():
            log.info(f'Skip (exists): {out.name}')
            return out

        all_features = self._load_all()
        state_code   = f'US-{state}'

        filtered = []
        for feat in all_features:
            props = feat['properties']
            s     = props.get('attr_POOState', '') or ''
            d     = props.get(
                'attr_FireDiscoveryDateTime', ''
            ) or ''
            size  = props.get('attr_IncidentSize', 0) or 0
            if (s == state_code and
                    str(year) in d and
                    size >= min_acres):
                filtered.append(feat)

        if not filtered:
            raise ValueError(
                f'NIFC: 0 fires for {state} {year} '
                f'(min_acres={min_acres}).'
            )

        geojson = {
            'type':     'FeatureCollection',
            'features': filtered,
        }
        out.write_text(json.dumps(geojson, indent=2))
        log.info(
            f'✅ NIFC: {len(filtered)} fires → {out.name}'
        )
        return out

    def download_all(
        self,
        years: List[int] = TRAIN_YEARS,
        state: str = 'CA',
    ) -> List[Path]:
        """Download NIFC for all years. Raises on any failure."""
        paths  = []
        failed = []

        for year in years:
            try:
                path = self.download(year, state)
                paths.append(path)
            except Exception as e:
                log.error(f'NIFC FAILED {state} {year}: {e}')
                failed.append(year)

        if failed:
            raise RuntimeError(
                f'NIFC download failed for years: {failed}\n'
                f'Labels required for training.'
            )

        log.info(f'✅ NIFC complete: {len(paths)} years')
        return paths

    def load(self, path: Path) -> gpd.GeoDataFrame:
        """Load fire perimeters. Raises if file invalid."""
        if not path.exists():
            raise FileNotFoundError(
                f'NIFC file not found: {path}'
            )
        gdf = gpd.read_file(path)
        if len(gdf) == 0:
            raise ValueError(f'NIFC file is empty: {path}')
        return gdf


class ERA5Pipeline:
    """
    Downloads ERA5 weather reanalysis from Copernicus CDS.

    Variables per grid cell per 6 hours:
      t2m:  2m temperature (Kelvin)
      u10:  10m wind U component (m/s)
      v10:  10m wind V component (m/s)
      r:    relative humidity (%)
      tp:   total precipitation (m)

    Output: NetCDF per area/month.
    Stored: data/wildfire/era5/{area}_{year}_{month:02d}.nc

    Setup required:
      1. pip install cdsapi
      2. Register: https://cds.climate.copernicus.eu
      3. Create ~/.cdsapirc with your key
    """

    def __init__(self, data_dir: Path):
        self.out_dir = data_dir / 'era5'
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._check_deps()

    @staticmethod
    def _check_deps():
        try:
            import cdsapi
        except ImportError:
            raise ImportError(
                'cdsapi not installed.\n'
                'Run: pip install cdsapi\n'
                'Then setup ~/.cdsapirc: '
                'https://cds.climate.copernicus.eu/api-how-to'
            )

    def download(
        self,
        area: str,
        year: int,
        month: int,
    ) -> Path:
        """
        Download ERA5 monthly data for one area.
        Raises on failure or empty file.
        """
        import cdsapi

        out = self.out_dir / f'{area}_{year}_{month:02d}.nc'
        if out.exists():
            if out.stat().st_size < 1000:
                log.warning(
                    f'ERA5 file too small (possibly corrupt): '
                    f'{out.name} — re-downloading'
                )
                out.unlink()
            else:
                log.info(f'Skip (exists): {out.name}')
                return out

        if area not in STUDY_AREAS:
            raise ValueError(f'Unknown area: {area}')

        cfg  = STUDY_AREAS[area]
        bbox = cfg['bbox']
        _, last_day_str = month_date_range(year, month)
        last_day = int(last_day_str.split('-')[2])

        log.info(f'ERA5: {area} {year}-{month:02d}')
        c = cdsapi.Client()
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
                    '2m_temperature',
                    '10m_u_component_of_wind',
                    '10m_v_component_of_wind',
                    '2m_dewpoint_temperature',
                ],
                'year':  str(year),
                'month': f'{month:02d}',
                'day':   [
                    f'{d:02d}' for d in range(1, last_day + 1)
                ],
                'time':  ['00:00', '06:00', '12:00', '18:00'],
                'area':  [
                    bbox[3], bbox[0],
                    bbox[1], bbox[2],
                ],
                'format': 'netcdf',
            },
            str(out)
        )

        if not out.exists() or out.stat().st_size < 1000:
            raise RuntimeError(
                f'ERA5 download produced empty/missing file: '
                f'{out}'
            )

        log.info(
            f'✅ ERA5: {out.name} '
            f'({out.stat().st_size / 1e6:.1f}MB)'
        )
        return out

    def download_all(
        self,
        area: str,
        years: List[int] = TRAIN_YEARS,
        months: List[int] = FIRE_SEASON_MONTHS,
    ) -> List[Path]:
        """Download ERA5 for all fire season months."""
        paths  = []
        total  = len(years) * len(months)
        done   = 0
        failed = []

        log.info(
            f'ERA5: {area} {years[0]}-{years[-1]} '
            f'({total} months)'
        )

        for year in years:
            for month in months:
                try:
                    path = self.download(area, year, month)
                    paths.append(path)
                except Exception as e:
                    log.error(
                        f'ERA5 FAILED {area} '
                        f'{year}-{month:02d}: {e}'
                    )
                    failed.append((area, year, month))
                done += 1
                log.info(f'  Progress: {done}/{total}')

        if failed:
            log.warning(
                f'ERA5: {len(failed)} failed:\n'
                + '\n'.join(
                    f'  {a} {y}-{m:02d}'
                    for a, y, m in failed
                )
            )

        log.info(
            f'✅ ERA5 complete: {len(paths)}/{total} files'
        )
        return paths


# ── Master Pipeline ───────────────────────────────────────────
class WildfirePipeline:
    """
    Master pipeline — coordinates all 4 data sources.
    Stores everything locally under data/wildfire/

    Note: NIFC failures raise immediately (labels required).
    VIIRS/Landsat/ERA5 failures are logged but non-fatal
    since partial data is still usable for training.
    """

    def __init__(
        self,
        data_dir: str = 'data/wildfire',
        firms_api_key: str = '',
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.viirs   = VIIRSPipeline(
            self.data_dir, firms_api_key
        )
        self.landsat = LandsatThermalPipeline(self.data_dir)
        self.nifc    = NIFCPipeline(self.data_dir)
        self.era5    = ERA5Pipeline(self.data_dir)

        log.info('WildfirePipeline initialized')
        log.info(
            f'  Data dir:    {self.data_dir.resolve()}'
        )
        log.info(f'  Train years: {TRAIN_YEARS}')
        log.info(f'  Fire months: {FIRE_SEASON_MONTHS}')

    def run(
        self,
        areas: List[str] = list(STUDY_AREAS.keys()),
        years: List[int] = TRAIN_YEARS,
    ) -> Dict:
        """
        Run full historical download for all areas+years.
        NIFC raises on failure.
        Other sources log failures and continue.
        """
        log.info(
            f'\n{"="*60}\n'
            f'WildfirePipeline — Historical Download\n'
            f'Areas: {areas}\n'
            f'Years: {years[0]}-{years[-1]}\n'
            f'{"="*60}'
        )

        results = {}

        # Step 1: NIFC labels — raises on failure
        log.info('\n[1/4] NIFC fire perimeters (labels)...')
        results['nifc'] = self.nifc.download_all(years)

        # Steps 2-4: per area
        for area in areas:
            log.info(f'\n--- Area: {area} ---')
            results[area] = {}

            log.info(f'[2/4] VIIRS hotspots: {area}')
            results[area]['viirs'] = (
                self.viirs.download_all(area, years)
            )

            log.info(f'[3/4] Landsat thermal: {area}')
            results[area]['thermal'] = (
                self.landsat.download_all(area, years)
            )

            log.info(f'[4/4] ERA5 weather: {area}')
            results[area]['era5'] = (
                self.era5.download_all(area, years)
            )

            log.info(
                f'\n✅ {area} summary:\n'
                f'   VIIRS:   '
                f'{len(results[area]["viirs"])} files\n'
                f'   Thermal: '
                f'{len(results[area]["thermal"])} scenes\n'
                f'   ERA5:    '
                f'{len(results[area]["era5"])} files'
            )

        # Save manifest
        manifest = {
            'nifc': [str(p) for p in results['nifc']],
        }
        for area in areas:
            if area in results:
                manifest[area] = {
                    k: [str(p) for p in v]
                    for k, v in results[area].items()
                    if isinstance(v, list)
                }

        manifest_path = self.data_dir / 'manifest.json'
        manifest_path.write_text(
            json.dumps(manifest, indent=2)
        )
        log.info(f'\n✅ Manifest saved: {manifest_path}')
        log.info(
            f'\n{"="*60}\n'
            f'✅ WildfirePipeline complete!\n'
            f'   Data stored: {self.data_dir.resolve()}\n'
            f'{"="*60}'
        )
        return results


# ── Entry Point ───────────────────────────────────────────────
def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    parser = argparse.ArgumentParser(
        description='ThermalWatch Wildfire Data Pipeline'
    )
    parser.add_argument(
        '--data-dir',
        default='data/wildfire',
        help='Local directory to store downloaded data'
    )
    parser.add_argument(
        '--areas',
        nargs='+',
        default=list(STUDY_AREAS.keys()),
        choices=list(STUDY_AREAS.keys()),
        help='Study areas to download'
    )
    parser.add_argument(
        '--years',
        nargs='+',
        type=int,
        default=TRAIN_YEARS,
        help='Years to download'
    )
    parser.add_argument(
        '--firms-key',
        required=True,
        help='NASA FIRMS API key (required)'
    )
    args = parser.parse_args()

    pipeline = WildfirePipeline(
        data_dir=args.data_dir,
        firms_api_key=args.firms_key,
    )
    pipeline.run(areas=args.areas, years=args.years)


if __name__ == '__main__':
    main()
