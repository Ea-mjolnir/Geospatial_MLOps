"""
ThermalWatch — Patch Extractor
================================
Extracts 224×224 patches from raw downloaded data
for both wildfire and solar tasks.

Takes raw GeoTIFF files (Landsat thermal)
and extracts fixed-size patches aligned to a common grid.

For each patch saves:
  thermal_{patch_id}.npy → [1, 224, 224] surface temp (°C)

Also builds patch index JSON with:
  - patch_id, area, year, month
  - local file paths
  - OSM feature vector
  - bbox coordinates
  - thermal stats

Local storage:
  data/wildfire/patches/{area}_{year}/thermal_{patch_id}.npy
  data/wildfire/patches/{area}_{year}_index.json
  data/solar/patches/{area}_{year}/thermal_{patch_id}.npy
  data/solar/patches/{area}_{year}_index.json

GDrive mirrors same structure under:
  My Drive/thermalwatch/data/wildfire/patches/
  My Drive/thermalwatch/data/solar/patches/

Usage:
  python3 -m src.data.patch_extractor \
    --task wildfire \
    --data-dir data/wildfire \
    --areas sierra_nevada socal norcal \
    --years 2020 2021 2022 2023 \
    --patch-size 224
"""

import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

PATCH_SIZE   = 224
STRIDE       = 224   # no overlap
NODATA_VALUE = -124.1  # Landsat nodata in Celsius

WILDFIRE_AREAS = ['sierra_nevada', 'socal', 'norcal']
SOLAR_AREAS    = ['sonoran_desert', 'phoenix_metro', 'tucson']
TRAIN_YEARS    = list(range(2020, 2024))
SOLAR_YEARS    = list(range(2019, 2024))


def check_rasterio():
    try:
        import rasterio
        return rasterio
    except ImportError:
        raise ImportError(
            'rasterio not installed.\n'
            'Run: pip install rasterio'
        )


def extract_patches(
    tif_path: Path,
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Extract patches from one GeoTIFF file.

    Returns list of (patch_array, row, col) tuples.
    patch_array shape: [bands, patch_size, patch_size]

    Nodata pixels (≈ -124.1°C) replaced with NaN.

    Raises if:
      - File does not exist
      - File is corrupt or unreadable
      - No valid patches found
    """
    rasterio = check_rasterio()

    if not tif_path.exists():
        raise FileNotFoundError(
            f'TIF file not found: {tif_path}'
        )

    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)

    if data.size == 0:
        raise ValueError(
            f'TIF file has 0 pixels: {tif_path}'
        )

    # Replace nodata value with NaN
    data[np.abs(data - NODATA_VALUE) < 0.5] = np.nan

    bands, height, width = data.shape
    patches = []

    for row in range(0, height - patch_size + 1, stride):
        for col in range(0, width - patch_size + 1, stride):
            patch = data[
                :,
                row:row + patch_size,
                col:col + patch_size,
            ]

            # Skip patches with <50% valid pixels
            valid_ratio = np.sum(
                np.isfinite(patch)
            ) / patch.size

            if valid_ratio < 0.5:
                log.debug(
                    f'Skip patch row={row} col={col}: '
                    f'only {valid_ratio:.0%} valid pixels'
                )
                continue

            patches.append((patch, row, col))

    if not patches:
        raise ValueError(
            f'No valid patches from {tif_path.name}. '
            f'All patches had <50% valid pixels.'
        )

    log.info(
        f'  {len(patches)} patches from {tif_path.name}'
    )
    return patches


# ── Wildfire Patch Extractor ──────────────────────────────────
class WildfirePatchExtractor:
    """
    Extracts patches from wildfire Landsat thermal files.

    OSM feature matching:
      OSM index key: {area}_r{row:04d}_c{col:04d}
      These are area-level grid cells (0.02° patches)
      Patch pixel (row, col) from TIF maps to OSM grid
      via geographic coordinates.

    Output per patch:
      thermal_{patch_id}.npy: [1, 224, 224] temp in Celsius
                               NaN for nodata pixels

    Index JSON: {area}_{year}_index.json
    """

    def __init__(
        self,
        data_dir: str,
        patch_size: int = PATCH_SIZE,
    ):
        self.data_dir   = Path(data_dir)
        self.patch_size = patch_size
        self.patch_dir  = self.data_dir / 'patches'
        self.patch_dir.mkdir(parents=True, exist_ok=True)

        # Load OSM index for all areas
        self.osm_index = {}
        for area in WILDFIRE_AREAS:
            osm_path = (
                self.data_dir / 'osm' / f'{area}_osm.json'
            )
            if osm_path.exists():
                osm_data = json.loads(osm_path.read_text())
                # Key by patch_id: {area}_r{row:04d}_c{col:04d}
                for entry in osm_data:
                    self.osm_index[entry['patch_id']] = entry
                log.info(
                    f'Loaded OSM: {area} '
                    f'({len(osm_data)} patches)'
                )
            else:
                log.warning(
                    f'OSM not found for {area} — '
                    f'will use zero features'
                )

    def _get_osm_patch_id(
        self,
        area: str,
        tif_path: Path,
        pixel_row: int,
        pixel_col: int,
    ) -> str:
        """
        Convert TIF pixel (row, col) to OSM patch grid key.

        TIF is in UTM (EPSG:32610) — must reproject to
        WGS84 (EPSG:4326) before computing OSM grid cell.
        """
        rasterio = check_rasterio()
        from pyproj import Transformer

        with rasterio.open(tif_path) as src:
            transform = src.transform
            crs       = src.crs
            # Patch center in TIF pixel coords
            px = pixel_col + self.patch_size / 2
            py = pixel_row + self.patch_size / 2
            x, y = rasterio.transform.xy(transform, py, px)

        # Reproject to WGS84 if not already
        if crs.to_epsg() != 4326:
            transformer = Transformer.from_crs(
                crs.to_epsg(), 4326, always_xy=True
            )
            lon, lat = transformer.transform(x, y)
        else:
            lon, lat = x, y

        from src.data.wildfire.osm_pipeline import STUDY_AREAS
        bbox = STUDY_AREAS[area]['bbox']
        west, south = bbox[0], bbox[1]

        OSM_PATCH = 0.02
        osm_row = int((lat - south) / OSM_PATCH)
        osm_col = int((lon - west)  / OSM_PATCH)

        return f'{area}_r{osm_row:04d}_c{osm_col:04d}'

    def _get_osm_features(
        self,
        area: str,
        tif_path: Path,
        pixel_row: int,
        pixel_col: int,
    ) -> List[float]:
        """
        Get OSM feature vector for a patch.
        Returns zeros if OSM data not available.
        """
        try:
            osm_patch_id = self._get_osm_patch_id(
                area, tif_path, pixel_row, pixel_col
            )
            if osm_patch_id in self.osm_index:
                osm = self.osm_index[osm_patch_id]
                return [
                    float(osm.get('road_count',        0)),
                    float(osm.get('building_count',     0)),
                    float(osm.get('forest_count',       0)),
                    float(osm.get('water_count',        0)),
                    float(osm.get('powerline_count',    0)),
                    float(osm.get('residential_count',  0)),
                    float(osm.get('farmland_count',     0)),
                ]
            else:
                return [0.0] * 7
        except Exception as e:
            log.debug(f'OSM lookup failed: {e}')
            return [0.0] * 7

    def extract_area_year(
        self,
        area: str,
        year: int,
    ) -> Path:
        """
        Extract all patches for one area + year.
        Saves patches as .npy files + builds index JSON.
        Raises if no thermal files found.
        """
        thermal_dir = self.data_dir / 'landsat_thermal'
        tif_files   = sorted(
            thermal_dir.glob(f'{area}_{year}_*.tif')
        )

        if not tif_files:
            raise FileNotFoundError(
                f'No Landsat thermal files for '
                f'{area} {year} in {thermal_dir}.\n'
                f'Run wildfire_pipeline.py first.'
            )

        out_dir = self.patch_dir / f'{area}_{year}'
        out_dir.mkdir(parents=True, exist_ok=True)

        index = []

        for tif_path in tif_files:
            # TIF name: {area}_{year}_{month}_{scene_id}_B10.tif
            # area = sierra_nevada (2 parts) → month at index 3
            # area = socal/norcal (1 part)   → month at index 2
            area_parts = len(area.split('_'))
            parts = tif_path.stem.split('_')
            month = int(parts[area_parts + 1])

            log.info(
                f'Extracting {area} {year}-{month:02d} '
                f'from {tif_path.name}'
            )

            try:
                patches = extract_patches(
                    tif_path, self.patch_size
                )
            except Exception as e:
                log.error(
                    f'Extraction failed {tif_path.name}: {e}'
                )
                continue

            for patch_arr, row, col in patches:
                patch_id = (
                    f'{area}_{year}_{month:02d}'
                    f'_r{row:05d}_c{col:05d}'
                )
                patch_file = (
                    out_dir / f'thermal_{patch_id}.npy'
                )

                np.save(str(patch_file), patch_arr)

                # Get OSM features via geo-coordinate lookup
                osm_features = self._get_osm_features(
                    area, tif_path, row, col
                )

                # Thermal stats (ignoring NaN)
                valid = patch_arr[np.isfinite(patch_arr)]
                temp_mean = float(np.mean(valid)) if len(valid) else 0.0
                temp_max  = float(np.max(valid))  if len(valid) else 0.0
                temp_std  = float(np.std(valid))  if len(valid) else 0.0

                index.append({
                    'patch_id':     patch_id,
                    'area':         area,
                    'year':         year,
                    'month':        month,
                    'row':          row,
                    'col':          col,
                    'thermal_path': str(patch_file),
                    'osm_features': osm_features,
                    'temp_mean_c':  temp_mean,
                    'temp_max_c':   temp_max,
                    'temp_std_c':   temp_std,
                    'source_file':  tif_path.name,
                })

        if not index:
            raise RuntimeError(
                f'No patches extracted for {area} {year}.'
            )

        index_path = (
            self.patch_dir / f'{area}_{year}_index.json'
        )
        index_path.write_text(json.dumps(index, indent=2))
        log.info(
            f'✅ {area} {year}: {len(index)} patches '
            f'→ {index_path.name}'
        )
        return index_path

    def extract_all(
        self,
        areas: List[str] = WILDFIRE_AREAS,
        years: List[int] = TRAIN_YEARS,
    ) -> List[Path]:
        """Extract patches for all areas + years."""
        index_paths = []
        failed      = []

        for area in areas:
            for year in years:
                try:
                    path = self.extract_area_year(area, year)
                    index_paths.append(path)
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

        log.info(
            f'✅ Wildfire patch extraction complete: '
            f'{len(index_paths)} area-years'
        )
        return index_paths


# ── Solar Patch Extractor ─────────────────────────────────────
class SolarPatchExtractor:
    """
    Extracts patches from solar Landsat thermal files.

    Output per patch:
      thermal_{patch_id}.npy: [1, 224, 224] temp in Celsius
                               NaN for nodata pixels

    Index JSON: {area}_{year}_index.json
    """

    def __init__(
        self,
        data_dir: str,
        patch_size: int = PATCH_SIZE,
    ):
        self.data_dir   = Path(data_dir)
        self.patch_size = patch_size
        self.patch_dir  = self.data_dir / 'patches'
        self.patch_dir.mkdir(parents=True, exist_ok=True)

        # Load OSM farm locations
        self.farm_index = {}
        for area in SOLAR_AREAS:
            farm_path = (
                self.data_dir / 'osm' /
                f'{area}_farms.geojson'
            )
            if farm_path.exists():
                import geopandas as gpd
                gdf = gpd.read_file(farm_path)
                self.farm_index[area] = gdf
                log.info(
                    f'Loaded farms: {area} ({len(gdf)} farms)'
                )
            else:
                log.warning(
                    f'Farm boundaries not found for {area}'
                )

    def extract_area_year(
        self,
        area: str,
        year: int,
    ) -> Path:
        """
        Extract patches for one area + year.
        Raises if no thermal files found.
        """
        thermal_dir = self.data_dir / 'landsat_thermal'
        tif_files   = sorted(
            thermal_dir.glob(f'{area}_{year}_*.tif')
        )

        if not tif_files:
            raise FileNotFoundError(
                f'No Landsat thermal files for '
                f'{area} {year} in {thermal_dir}.\n'
                f'Run solar_pipeline.py first.'
            )

        out_dir = self.patch_dir / f'{area}_{year}'
        out_dir.mkdir(parents=True, exist_ok=True)

        index = []

        for tif_path in tif_files:
            # TIF: {area}_{year}_{month}_{scene_id}_B10.tif
            area_parts = len(area.split('_'))
            parts = tif_path.stem.split('_')
            month = int(parts[area_parts + 1])

            log.info(
                f'Extracting {area} {year}-{month:02d} '
                f'from {tif_path.name}'
            )

            try:
                patches = extract_patches(
                    tif_path, self.patch_size
                )
            except Exception as e:
                log.error(
                    f'Extraction failed {tif_path.name}: {e}'
                )
                continue

            for patch_arr, row, col in patches:
                patch_id = (
                    f'{area}_{year}_{month:02d}'
                    f'_r{row:05d}_c{col:05d}'
                )
                patch_file = (
                    out_dir / f'thermal_{patch_id}.npy'
                )
                np.save(str(patch_file), patch_arr)

                # Thermal stats (ignoring NaN)
                valid = patch_arr[np.isfinite(patch_arr)]
                temp_mean = float(np.mean(valid)) if len(valid) else 0.0
                temp_max  = float(np.max(valid))  if len(valid) else 0.0
                temp_std  = float(np.std(valid))  if len(valid) else 0.0

                index.append({
                    'patch_id':     patch_id,
                    'area':         area,
                    'year':         year,
                    'month':        month,
                    'row':          row,
                    'col':          col,
                    'thermal_path': str(patch_file),
                    'temp_mean_c':  temp_mean,
                    'temp_max_c':   temp_max,
                    'temp_std_c':   temp_std,
                    'source_file':  tif_path.name,
                })

        if not index:
            raise RuntimeError(
                f'No patches extracted for {area} {year}.'
            )

        index_path = (
            self.patch_dir / f'{area}_{year}_index.json'
        )
        index_path.write_text(json.dumps(index, indent=2))
        log.info(
            f'✅ {area} {year}: {len(index)} patches '
            f'→ {index_path.name}'
        )
        return index_path

    def extract_all(
        self,
        areas: List[str] = SOLAR_AREAS,
        years: List[int] = SOLAR_YEARS,
    ) -> List[Path]:
        """Extract patches for all areas + years."""
        index_paths = []
        failed      = []

        for area in areas:
            for year in years:
                try:
                    path = self.extract_area_year(area, year)
                    index_paths.append(path)
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

        log.info(
            f'✅ Solar patch extraction complete: '
            f'{len(index_paths)} area-years'
        )
        return index_paths


# ── Entry Point ───────────────────────────────────────────────
def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    parser = argparse.ArgumentParser(
        description='ThermalWatch Patch Extractor'
    )
    parser.add_argument(
        '--task',
        choices=['wildfire', 'solar'],
        required=True,
    )
    parser.add_argument(
        '--data-dir',
        required=True,
        help='Data directory e.g. data/wildfire'
    )
    parser.add_argument(
        '--areas',
        nargs='+',
        default=None,
    )
    parser.add_argument(
        '--years',
        nargs='+',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--patch-size',
        type=int,
        default=224,
    )
    args = parser.parse_args()

    if args.task == 'wildfire':
        extractor = WildfirePatchExtractor(
            data_dir=args.data_dir,
            patch_size=args.patch_size,
        )
        areas = args.areas or WILDFIRE_AREAS
        years = args.years or TRAIN_YEARS
    else:
        extractor = SolarPatchExtractor(
            data_dir=args.data_dir,
            patch_size=args.patch_size,
        )
        areas = args.areas or SOLAR_AREAS
        years = args.years or SOLAR_YEARS

    extractor.extract_all(areas, years)


if __name__ == '__main__':
    main()
