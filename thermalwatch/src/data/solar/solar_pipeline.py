"""
ThermalWatch — Solar Farm Data Pipeline
========================================
Downloads solar farm data for Arizona study areas.
All data stored locally under data/solar/

Data Sources:
  1. OSM — solar farm boundaries (Overpass API)
  2. Landsat 8/9 Band 10 — panel thermal hotspots
     Source: Microsoft Planetary Computer
  3. NSRDB — solar irradiance grid (NREL)
     Grid download, NOT just centroid point
  4. EIA — monthly power output per facility
     Source: EIA Open Data API

Study Areas: sonoran_desert, phoenix_metro, tucson (Arizona)
Historical Range: 2019-2023 (all months, peak: Mar-Oct)

API Keys Required:
  NREL/NSRDB: https://developer.nrel.gov/signup/
  EIA:        https://www.eia.gov/opendata/register.php

Usage:
  python3 -m src.data.solar.solar_pipeline \
    --data-dir data/solar \
    --nrel-key YOUR_NREL_KEY \
    --eia-key YOUR_EIA_KEY \
    --areas sonoran_desert phoenix_metro \
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
    'sonoran_desert': {
        'bbox':  [-114.0, 31.5, -110.0, 34.0],  # W,S,E,N
        'state': 'AZ',
        'lat':   32.75,
        'lon':   -112.0,
    },
    'phoenix_metro': {
        'bbox':  [-113.0, 32.5, -111.0, 34.0],  # W,S,E,N
        'state': 'AZ',
        'lat':   33.45,
        'lon':   -112.07,
    },
    'tucson': {
        'bbox':  [-112.0, 31.5, -110.0, 33.0],  # W,S,E,N
        'state': 'AZ',
        'lat':   32.22,
        'lon':   -110.93,
    },
}

TRAIN_YEARS = list(range(2019, 2024))
PEAK_MONTHS = [3, 4, 5, 6, 7, 8, 9, 10]
ALL_MONTHS  = list(range(1, 13))


def month_date_range(year: int, month: int) -> Tuple[str, str]:
    """Return (start, end) date strings for a given month."""
    last_day = calendar.monthrange(year, month)[1]
    return (
        f'{year}-{month:02d}-01',
        f'{year}-{month:02d}-{last_day:02d}',
    )


# ── 1. OSM Solar Farm Boundaries ─────────────────────────────
class OSMSolarPipeline:
    """
    Downloads solar farm boundaries from OpenStreetMap
    via Geofabrik Arizona state extract + osmium filtering.

    Uses Geofabrik Arizona OSM extract (free, no API key).
    Filters for solar farms by tags:
      power=plant + plant:source=solar
      generator:source=solar
      plant:source=solar

    Output: GeoJSON per area.
    Stored: data/solar/osm/{area}_farms.geojson

    Raises if:
      - Geofabrik download fails
      - osmium not installed
      - No farms found after filtering
    """
    GEOFABRIK_URL = (
        'https://download.geofabrik.de/'
        'north-america/us/arizona-latest.osm.pbf'
    )

    def __init__(self, data_dir: Path):
        self.out_dir  = data_dir / 'osm'
        self.pbf_path = data_dir / 'arizona-latest.osm.pbf'
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._check_deps()

    @staticmethod
    def _check_deps():
        try:
            import osmium
        except ImportError:
            raise ImportError(
                'osmium not installed. '
                'Run: pip install osmium'
            )

    def _download_pbf(self) -> Path:
        """
        Download Arizona OSM extract from Geofabrik.
        Raises on failure.
        """
        if self.pbf_path.exists():
            size = self.pbf_path.stat().st_size / 1e6
            log.info(
                f'Skip (exists): arizona-latest.osm.pbf '
                f'({size:.0f}MB)'
            )
            return self.pbf_path

        log.info('Downloading Arizona OSM from Geofabrik...')
        r = requests.get(
            self.GEOFABRIK_URL,
            stream=True,
            timeout=300,
            allow_redirects=True,
        )
        r.raise_for_status()

        total = int(r.headers.get('content-length', 0))
        done  = 0

        with open(self.pbf_path, 'wb') as f:
            for chunk in r.iter_content(
                chunk_size=1024 * 1024
            ):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done / total * 100
                    if done % (10*1024*1024) < 1024*1024:
                        log.info(
                            f'  Downloading: {pct:.0f}% '
                            f'({done/1e6:.0f}/'
                            f'{total/1e6:.0f}MB)'
                        )

        size = self.pbf_path.stat().st_size / 1e6
        log.info(f'✅ Arizona OSM: {size:.0f}MB downloaded')
        return self.pbf_path

    def _extract_solar_farms(
        self,
        bbox: List[float],
    ) -> List[Dict]:
        """
        Extract solar farm ways from PBF within bbox.
        bbox: [W, S, E, N]
        Raises if osmium fails.
        """
        import osmium
        from shapely.geometry import Polygon

        west, south, east, north = (
            bbox[0], bbox[1], bbox[2], bbox[3]
        )

        class SolarFarmHandler(osmium.SimpleHandler):
            def __init__(self):
                super().__init__()
                self.farms = []

            def way(self, w):
                tags = {t.k: t.v for t in w.tags}

                is_solar = (
                    (tags.get('power') == 'plant' and
                     tags.get('plant:source') == 'solar') or
                    tags.get('generator:source') == 'solar' or
                    tags.get('plant:source') == 'solar'
                )
                if not is_solar:
                    return

                try:
                    coords = [
                        (n.lon, n.lat)
                        for n in w.nodes
                        if n.location.valid()
                    ]
                except osmium.InvalidLocationError:
                    return

                if len(coords) < 3:
                    return

                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                if (max(lons) < west or min(lons) > east or
                        max(lats) < south or
                        min(lats) > north):
                    return

                # Filter by minimum size — solar farms
                # must be at least 0.001 degrees wide
                # (~100m) to exclude individual panels
                lon_span = max(lons) - min(lons)
                lat_span = max(lats) - min(lats)
                if lon_span < 0.001 and lat_span < 0.001:
                    return

                self.farms.append({
                    'id':       w.id,
                    'name':     tags.get('name', 'Unknown'),
                    'operator': tags.get('operator', ''),
                    'capacity': tags.get(
                        'plant:output:electricity', ''
                    ),
                    'geometry': Polygon(coords),
                })

        handler = SolarFarmHandler()
        handler.apply_file(
            str(self.pbf_path),
            locations=True,
        )
        return handler.farms

    def download(self, area: str) -> Path:
        """
        Extract solar farm polygons for one area from PBF.
        Raises if no farms found.
        """
        if area not in STUDY_AREAS:
            raise ValueError(
                f'Unknown area: {area}. '
                f'Choose from: {list(STUDY_AREAS.keys())}'
            )

        out = self.out_dir / f'{area}_farms.geojson'
        if out.exists():
            log.info(f'Skip (exists): {out.name}')
            return out

        self._download_pbf()

        cfg  = STUDY_AREAS[area]
        bbox = cfg['bbox']

        log.info(
            f'Extracting solar farms: {area} bbox={bbox}'
        )
        farms = self._extract_solar_farms(bbox)

        if not farms:
            raise ValueError(
                f'No solar farms found for {area}. '
                f'bbox={bbox}.'
            )

        gdf = gpd.GeoDataFrame(farms, crs='EPSG:4326')
        gdf.to_file(out, driver='GeoJSON')
        log.info(
            f'✅ OSM: {len(gdf)} solar farms → {out.name}'
        )
        return out

    def download_all(
        self,
        areas: List[str] = list(STUDY_AREAS.keys()),
    ) -> Dict[str, Path]:
        """
        Download PBF once then extract for all areas.
        Raises if any area fails.
        """
        self._download_pbf()

        results = {}
        failed  = []

        for area in areas:
            try:
                results[area] = self.download(area)
            except Exception as e:
                log.error(f'OSM FAILED {area}: {e}')
                failed.append(area)

        if failed:
            raise RuntimeError(
                f'OSM failed for areas: {failed}'
                f'Farm boundaries required for training.'
            )

        log.info(f'✅ OSM complete: {len(results)} areas')
        return results


# ── 2. NSRDB Solar Irradiance Pipeline ───────────────────────
class NSRDBPipeline:
    """
    Downloads solar irradiance from NREL NSRDB PSM3.

    Downloads a GRID of points across the study area,
    NOT just a single centroid point.
    Grid spacing: 0.5 degrees (~55km)

    Variables per hourly timestep:
      GHI:  Global Horizontal Irradiance (W/m²)
      DNI:  Direct Normal Irradiance (W/m²)
      DHI:  Diffuse Horizontal Irradiance (W/m²)
      Wind: wind speed (m/s)
      Temp: air temperature (°C)
      Cloud: cloud type code

    Output: CSV per grid point per year.
    Stored: data/solar/nsrdb/{area}_{lat}_{lon}_{year}.csv
    API key: https://developer.nrel.gov/signup/ (free)

    Raises if:
      - No API key provided
      - API request fails
      - Response is not valid CSV
    """
    BASE_URL = (
        'https://developer.nlr.gov/api/nsrdb/v2/solar/'
        'nsrdb-GOES-aggregated-v4-0-0-download.csv'
    )
    GRID_SPACING = 0.5  # degrees (~55km)

    def __init__(self, data_dir: Path, api_key: str, email: str):
        if not api_key or api_key == 'DEMO_KEY':
            raise ValueError(
                'Valid NREL API key required for NSRDB. '
                'DEMO_KEY is heavily rate limited.\n'
                'Get free key: '
                'https://developer.nrel.gov/signup/'
            )
        self.out_dir = data_dir / 'nsrdb'
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key
        self.email   = email

    def _grid_points(
        self, area: str
    ) -> List[Tuple[float, float]]:
        """Generate grid of lat/lon points for study area."""
        cfg  = STUDY_AREAS[area]
        bbox = cfg['bbox']
        west, south, east, north = (
            bbox[0], bbox[1], bbox[2], bbox[3]
        )

        points = []
        lat = south
        while lat <= north:
            lon = west
            while lon <= east:
                points.append(
                    (round(lat, 2), round(lon, 2))
                )
                lon += self.GRID_SPACING
            lat += self.GRID_SPACING

        log.info(
            f'NSRDB grid: {area} → {len(points)} points '
            f'({self.GRID_SPACING}° spacing)'
        )
        return points

    def download(
        self,
        area: str,
        lat: float,
        lon: float,
        year: int,
    ) -> Path:
        """
        Download NSRDB for one grid point + year.
        Raises on failure or invalid response.
        """
        out = (
            self.out_dir /
            f'{area}_{lat:.2f}_{lon:.2f}_{year}.csv'
        )
        if out.exists():
            if out.stat().st_size < 100:
                log.warning(
                    f'NSRDB file too small — re-downloading: '
                    f'{out.name}'
                )
                out.unlink()
            else:
                log.info(f'Skip (exists): {out.name}')
                return out

        params = {
            'api_key':      self.api_key,
            'wkt':          f'POINT({lon} {lat})',
            'names':        year,
            'interval':     60,
            'utc':          'false',
            'full_name':    'ThermalWatch',
            'email':        self.email,
            'affiliation':  'ThermalWatch Project',
            'mailing_list': 'false',
            'reason':       'research',
            'attributes': (
                'ghi,dhi,dni,wind_speed,'
                'air_temperature,cloud_type'
            ),
        }

        r = requests.get(
            self.BASE_URL, params=params, timeout=120
        )

        if r.status_code == 401:
            raise PermissionError(
                f'Invalid NREL API key: '
                f'{self.api_key[:8]}...'
            )
        if r.status_code == 429:
            raise RuntimeError(
                'NREL API rate limit exceeded. '
                'Wait and retry.'
            )
        r.raise_for_status()

        if len(r.content) < 100:
            raise ValueError(
                f'NSRDB returned empty response for '
                f'lat={lat} lon={lon} year={year}'
            )

        # Validate it's CSV not error JSON
        first_line = r.content.decode('utf-8', errors='replace')\
            .split('\n')[0]
        if first_line.startswith('{'):
            err = json.loads(r.content)
            raise ValueError(
                f'NSRDB API error: '
                f'{err.get("errors", r.text[:200])}'
            )

        out.write_bytes(r.content)
        log.info(
            f'✅ NSRDB: {out.name} '
            f'({out.stat().st_size / 1e3:.1f}KB)'
        )
        return out

    def download_all(
        self,
        areas: List[str] = list(STUDY_AREAS.keys()),
        years: List[int] = TRAIN_YEARS,
    ) -> List[Path]:
        """
        Download NSRDB grid for all areas + years.
        Logs failures but continues — partial data usable.
        """
        paths  = []
        failed = []

        for area in areas:
            grid   = self._grid_points(area)
            total  = len(grid) * len(years)
            done   = 0

            log.info(
                f'NSRDB: {area} — {len(grid)} grid points × '
                f'{len(years)} years = {total} downloads'
            )

            for lat, lon in grid:
                for year in years:
                    try:
                        path = self.download(
                            area, lat, lon, year
                        )
                        paths.append(path)
                    except Exception as e:
                        log.error(
                            f'NSRDB FAILED '
                            f'{area} {lat},{lon} {year}: {e}'
                        )
                        failed.append((area, lat, lon, year))
                    done += 1
                    if done % 10 == 0:
                        log.info(
                            f'  Progress {area}: {done}/{total}'
                        )

        if failed:
            log.warning(
                f'NSRDB: {len(failed)} failed downloads. '
                f'Re-run to retry.'
            )

        log.info(
            f'✅ NSRDB complete: {len(paths)} files'
        )
        return paths


# ── 3. EIA Power Output Pipeline ─────────────────────────────
class EIAPipeline:
    """
    Downloads monthly solar generation from EIA.

    Each record contains:
      plant_id:   facility identifier
      plant_name: facility name
      state:      state code
      fuel_type:  SUN (solar)
      generation: MWh generated that month

    Output: JSON per state/year.
    Stored: data/solar/eia/solar_{state}_{year}.json
    API key: https://www.eia.gov/opendata/register.php (free)

    Raises if:
      - No API key provided
      - API request fails
      - Response contains no data
    """
    BASE_URL = (
        'https://api.eia.gov/v2/electricity/facility-fuel'
    )

    def __init__(self, data_dir: Path, api_key: str):
        if not api_key:
            raise ValueError(
                'EIA API key required.\n'
                'Get free key: '
                'https://www.eia.gov/opendata/register.php'
            )
        self.out_dir = data_dir / 'eia'
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key

    def download(
        self,
        year: int,
        state: str = 'AZ',
    ) -> Path:
        """
        Download monthly solar generation for one year.
        Raises on failure or empty response.
        """
        out = self.out_dir / f'solar_{state}_{year}.json'
        if out.exists():
            log.info(f'Skip (exists): {out.name}')
            return out

        params = {
            'api_key':             self.api_key,
            'frequency':           'monthly',
            'data[0]':             'generation',
            'facets[state][]':     state,
            'facets[fuel-type][]': 'SUN',
            'start':               f'{year}-01',
            'end':                 f'{year}-12',
            'offset':              0,
            'length':              5000,
        }

        log.info(f'EIA: {state} {year}')
        r = requests.get(
            self.BASE_URL, params=params, timeout=60
        )

        if r.status_code == 403:
            raise PermissionError(
                f'Invalid EIA API key: '
                f'{self.api_key[:8]}...'
            )
        r.raise_for_status()

        data = r.json()
        rows = data.get('response', {}).get('data', [])

        if not rows:
            raise ValueError(
                f'EIA returned 0 records for '
                f'{state} {year}. '
                f'Check state code and year range.'
            )

        out.write_text(json.dumps(data, indent=2))
        log.info(
            f'✅ EIA: {len(rows)} facility-months '
            f'→ {out.name}'
        )
        return out

    def download_all(
        self,
        years: List[int] = TRAIN_YEARS,
        state: str = 'AZ',
    ) -> List[Path]:
        """Download EIA for all years. Raises on any failure."""
        paths  = []
        failed = []

        for year in years:
            try:
                path = self.download(year, state)
                paths.append(path)
            except Exception as e:
                log.error(f'EIA FAILED {state} {year}: {e}')
                failed.append(year)

        if failed:
            raise RuntimeError(
                f'EIA download failed for years: {failed}\n'
                f'Power output labels required for training.'
            )

        log.info(f'✅ EIA complete: {len(paths)} years')
        return paths


# ── 4. Landsat Thermal Pipeline (Solar) ──────────────────────
class SolarLandsatPipeline:
    """
    Downloads Landsat 8/9 Band 10 thermal for solar farms.
    Identical conversion to wildfire pipeline:
      T(K) = 0.00341802 × DN + 149.0
      T(C) = T(K) - 273.15

    Lower cloud threshold (20%) vs wildfire (30%)
    because solar panels need clear sky context.

    Output: single-band GeoTIFF in Celsius.
    Stored: data/solar/landsat_thermal/
    Source: Microsoft Planetary Computer (no key needed)

    Raises if:
      - Dependencies not installed
      - Scene Band 10 is all zeros (corrupt)
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
        max_cloud: int = 20,
        max_scenes: int = 3,
    ) -> List[Path]:
        """
        Download Landsat Band 10 for one area/month.
        Raises on corrupt data or API failure.
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
            f'Solar Landsat: {len(items)} scenes '
            f'{area} {year}-{month:02d}'
        )

        if not items:
            log.warning(
                f'No Landsat scenes for {area} '
                f'{year}-{month:02d} '
                f'(cloud<{max_cloud}%). '
                f'Try increasing max_cloud.'
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

            if 'lwir11' in item.assets:
                band_key = 'lwir11'
            elif 'lwir' in item.assets:
                band_key = 'lwir'
            else:
                log.warning(
                    f'No thermal band in scene '
                    f'{item.id} — skipping.'
                )
                continue
            signed = pc.sign(item)
            href   = signed.assets[band_key].href

            with rasterio.open(href) as src:
                data    = src.read(1)
                profile = src.profile

            if data.max() == 0:
                raise ValueError(
                    f'Scene {item.id} Band 10 all zeros '
                    f'— corrupt or no data'
                )

            temp_k = (
                0.00341802 * data.astype(np.float32) + 149.0
            )
            temp_c = temp_k - 273.15

            # Solar panels in Arizona: valid range
            # Panel temp: 20°C - 90°C
            # Ambient: -5°C to 50°C
            if temp_c.max() < 0 or temp_c.min() > 100:
                raise ValueError(
                    f'Scene {item.id} has unrealistic '
                    f'temperatures: '
                    f'min={temp_c.min():.1f}°C '
                    f'max={temp_c.max():.1f}°C'
                )

            profile.update(dtype=rasterio.float32, count=1)
            with rasterio.open(out, 'w', **profile) as dst:
                dst.write(temp_c, 1)

            log.info(
                f'✅ Solar thermal: {out.name} '
                f'(range: {temp_c.min():.1f}°C - '
                f'{temp_c.max():.1f}°C)'
            )
            paths.append(out)

        return paths

    def download_all(
        self,
        area: str,
        years: List[int] = TRAIN_YEARS,
        months: List[int] = PEAK_MONTHS,
    ) -> List[Path]:
        """Download Landsat thermal for all years/months."""
        paths  = []
        total  = len(years) * len(months)
        done   = 0
        failed = []

        log.info(
            f'Solar Landsat: {area} '
            f'{years[0]}-{years[-1]} ({total} months)'
        )

        for year in years:
            for month in months:
                try:
                    p = self.download(area, year, month)
                    paths.extend(p)
                except Exception as e:
                    log.error(
                        f'Solar Landsat FAILED {area} '
                        f'{year}-{month:02d}: {e}'
                    )
                    failed.append((area, year, month))
                done += 1
                log.info(f'  Progress: {done}/{total}')

        if failed:
            log.warning(
                f'Solar Landsat: {len(failed)} failed:\n'
                + '\n'.join(
                    f'  {a} {y}-{m:02d}'
                    for a, y, m in failed
                )
            )

        log.info(
            f'✅ Solar Landsat complete: {len(paths)} scenes'
        )
        return paths


# ── Master Pipeline ───────────────────────────────────────────
class SolarPipeline:
    """
    Master pipeline — coordinates all solar data sources.
    Stores everything locally under data/solar/

    Note:
      OSM + EIA raise on failure (required for training).
      NSRDB + Landsat log failures and continue
      (partial coverage still usable).
    """

    def __init__(
        self,
        data_dir: str   = 'data/solar',
        nrel_api_key: str = '',
        eia_api_key: str  = '',
        email: str        = '',
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.osm     = OSMSolarPipeline(self.data_dir)
        self.nsrdb   = NSRDBPipeline(
            self.data_dir, nrel_api_key, email
        )
        self.eia     = EIAPipeline(
            self.data_dir, eia_api_key
        )
        self.landsat = SolarLandsatPipeline(self.data_dir)

        log.info('SolarPipeline initialized')
        log.info(
            f'  Data dir:    {self.data_dir.resolve()}'
        )
        log.info(f'  Train years: {TRAIN_YEARS}')
        log.info(f'  Peak months: {PEAK_MONTHS}')

    def run(
        self,
        areas: List[str] = list(STUDY_AREAS.keys()),
        years: List[int] = TRAIN_YEARS,
    ) -> Dict:
        """
        Run full historical download for all areas+years.
        OSM + EIA raise on failure.
        NSRDB + Landsat log failures and continue.
        """
        log.info(
            f'\n{"="*60}\n'
            f'SolarPipeline — Historical Download\n'
            f'Areas: {areas}\n'
            f'Years: {years[0]}-{years[-1]}\n'
            f'{"="*60}'
        )

        results = {}

        # Step 1: OSM farm boundaries — raises on failure
        log.info('\n[1/4] OSM solar farm boundaries...')
        results['osm'] = self.osm.download_all(areas)

        # Step 2: EIA power output — raises on failure
        log.info('\n[2/4] EIA power output (labels)...')
        results['eia'] = self.eia.download_all(years)

        # Steps 3-4: per area
        for area in areas:
            log.info(f'\n--- Area: {area} ---')
            results[area] = {}

            # Step 3: NSRDB irradiance grid
            log.info(f'[3/4] NSRDB irradiance: {area}')
            results[area]['nsrdb'] = (
                self.nsrdb.download_all([area], years)
            )

            # Step 4: Landsat thermal
            log.info(f'[4/4] Landsat thermal: {area}')
            results[area]['thermal'] = (
                self.landsat.download_all(area, years)
            )

            log.info(
                f'\n✅ {area} summary:\n'
                f'   NSRDB:   '
                f'{len(results[area]["nsrdb"])} files\n'
                f'   Thermal: '
                f'{len(results[area]["thermal"])} scenes'
            )

        # Save manifest
        manifest = {
            'osm': {
                area: str(path)
                for area, path in results['osm'].items()
            },
            'eia': [str(p) for p in results['eia']],
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
            f'✅ SolarPipeline complete!\n'
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
        description='ThermalWatch Solar Data Pipeline'
    )
    parser.add_argument(
        '--data-dir',
        default='data/solar',
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
        '--nrel-key',
        required=True,
        help='NREL API key (required for NSRDB)'
    )
    parser.add_argument(
        '--eia-key',
        required=True,
        help='EIA API key (required for power output)'
    )
    args = parser.parse_args()

    pipeline = SolarPipeline(
        data_dir=args.data_dir,
        nrel_api_key=args.nrel_key,
        eia_api_key=args.eia_key,
    )
    pipeline.run(areas=args.areas, years=args.years)


if __name__ == '__main__':
    main()
