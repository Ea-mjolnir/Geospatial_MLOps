"""
ThermalWatch — Generate Training Labels
=========================================
Generates task-specific prediction targets for
each patch in the index files.

Wildfire labels (per patch):
  risk_score:     [0,1] fire risk based on proximity
                  to NIFC fire perimeters + VIIRS hotspots
  alert_level:    {0,1,2,3} no/low/med/high risk
  spread_prob:    [0,1] fire spread probability
                  based on VIIRS hotspot density
  structure_risk: [0,1] risk to structures
                  based on OSM building count + risk_score

Solar labels (per patch):
  efficiency_score:  [0,1] actual vs expected output
                     from EIA / NSRDB irradiance
  hotspot_score:     [0,1] thermal anomaly score
                     from temp_mean vs expected
  degradation_rate:  [0,1] estimated degradation
  maintenance_flag:  {0,1} needs maintenance

Usage:
  python3 -m src.data.generate_labels --task wildfire
  python3 -m src.data.generate_labels --task solar
"""

import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

WILDFIRE_AREAS = ['sierra_nevada', 'socal', 'norcal']
SOLAR_AREAS    = ['sonoran_desert', 'phoenix_metro', 'tucson']
WILDFIRE_YEARS = list(range(2020, 2024))
SOLAR_YEARS    = list(range(2019, 2024))


def _load_fire_perimeters(
    nifc_dir: Path,
    year: int,
) -> Optional[object]:
    """Load NIFC fire perimeters for one year."""
    try:
        import geopandas as gpd
        path = nifc_dir / f'fires_CA_{year}.geojson'
        if not path.exists():
            return None
        return gpd.read_file(path)
    except Exception as e:
        log.error(f'Failed to load NIFC {year}: {e}')
        return None


def _load_viirs(
    viirs_dir: Path,
    area: str,
    year: int,
    month: int,
) -> Optional[object]:
    """Load VIIRS hotspots for one area/month."""
    try:
        import pandas as pd
        path = viirs_dir / f'{area}_{year}_{month:02d}.csv'
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception as e:
        log.debug(f'VIIRS load failed: {e}')
        return None


def _patch_wgs84_center(
    entry: Dict,
    tif_cache: Dict,
    thermal_dir: Path,
) -> Optional[Tuple[float, float]]:
    """Get patch center in WGS84. Returns (lat, lon)."""
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
        except Exception:
            return None

    info = tif_cache[src_file]
    row  = entry['row'] + 112
    col  = entry['col'] + 112

    try:
        cx, cy = rasterio.transform.xy(
            info['transform'], row, col
        )
        l, b, r, t = transform_bounds(
            info['crs'], 'EPSG:4326',
            cx - 1, cy - 1, cx + 1, cy + 1
        )
        return ((b + t) / 2, (l + r) / 2)
    except Exception:
        return None


def _compute_wildfire_labels(
    entry: Dict,
    fire_gdf,
    viirs_df,
    tif_cache: Dict,
    thermal_dir: Path,
) -> Dict:
    """
    Compute wildfire labels for one patch.

    risk_score: distance-based proximity to fire perimeters
      0.0 = far from any fire
      1.0 = inside fire perimeter

    alert_level: 0=none 1=low 2=medium 3=high

    spread_prob: VIIRS hotspot density within 0.1° radius
      normalized to [0,1]

    structure_risk: risk_score × normalized building count
    """
    try:
        from shapely.geometry import Point
        import geopandas as gpd

        loc = _patch_wgs84_center(
            entry, tif_cache, thermal_dir
        )
        if loc is None:
            return _zero_wildfire_labels()

        lat, lon  = loc
        patch_pt  = Point(lon, lat)

        # Risk score from NIFC perimeters
        risk_score = 0.0
        if fire_gdf is not None and len(fire_gdf) > 0:
            # Check if patch inside any fire perimeter
            patch_gs = gpd.GeoSeries(
                [patch_pt], crs='EPSG:4326'
            )
            fire_crs = fire_gdf.crs
            if fire_crs != patch_gs.crs:
                patch_gs = patch_gs.to_crs(fire_crs)

            pt = patch_gs.iloc[0]

            # Inside fire → risk=1.0
            inside = fire_gdf.geometry.contains(pt).any()
            if inside:
                risk_score = 1.0
            else:
                # Distance-based decay
                try:
                    fire_utm = fire_gdf.to_crs('EPSG:3857')
                    pt_utm   = patch_gs.to_crs(
                        'EPSG:3857'
                    ).iloc[0]
                    dists = fire_utm.geometry.distance(pt_utm)
                    min_dist = float(dists.min())
                    # decay: 1.0 at 0km, 0.0 at 50km
                    risk_score = max(
                        0.0,
                        1.0 - min_dist / 50000.0
                    )
                except Exception:
                    risk_score = 0.0

        # Alert level from risk score
        if risk_score >= 0.8:
            alert_level = 3
        elif risk_score >= 0.5:
            alert_level = 2
        elif risk_score >= 0.2:
            alert_level = 1
        else:
            alert_level = 0

        # Spread probability from VIIRS hotspots
        spread_prob = 0.0
        if viirs_df is not None and len(viirs_df) > 0:
            try:
                # Count hotspots within 0.1° radius
                lats = viirs_df['latitude'].values
                lons = viirs_df['longitude'].values
                dists = np.sqrt(
                    (lats - lat)**2 + (lons - lon)**2
                )
                nearby = int(np.sum(dists < 0.1))
                # Normalize: 10+ hotspots → prob=1.0
                spread_prob = min(1.0, nearby / 10.0)
            except Exception:
                spread_prob = 0.0

        # Structure risk
        osm       = entry.get('osm_features', [0]*7)
        buildings = float(osm[1]) if len(osm) > 1 else 0.0
        bldg_norm = min(1.0, buildings / 500.0)
        structure_risk = float(risk_score * bldg_norm)

        return {
            'risk_score':     float(risk_score),
            'alert_level':    int(alert_level),
            'spread_prob':    float(spread_prob),
            'structure_risk': float(structure_risk),
        }

    except Exception as e:
        log.debug(f'Label computation failed: {e}')
        return _zero_wildfire_labels()


def _zero_wildfire_labels() -> Dict:
    return {
        'risk_score':     0.0,
        'alert_level':    0,
        'spread_prob':    0.0,
        'structure_risk': 0.0,
    }


def _load_eia(
    eia_dir: Path,
    year: int,
) -> Optional[Dict]:
    """Load EIA power output for one year."""
    try:
        path = eia_dir / f'solar_AZ_{year}.json'
        if not path.exists():
            return None
        return json.loads(path.read_text())
    except Exception:
        return None


def _compute_solar_labels(
    entry: Dict,
    eia_data: Optional[Dict],
    nsrdb_dir: Path,
    tif_cache: Dict,
    thermal_dir: Path,
) -> Dict:
    """
    Compute solar labels for one patch.

    efficiency_score:
      actual GHI utilization vs expected
      derived from temp_mean vs NSRDB GHI
      High temp + low GHI → low efficiency
      Low temp + high GHI → high efficiency

    hotspot_score:
      thermal anomaly relative to area mean
      Patches significantly hotter than average
      → possible hotspot/fault

    degradation_rate:
      estimated from temp deviation pattern
      (simplified without time series)

    maintenance_flag:
      1 if hotspot_score > 0.7
    """
    try:
        temp_mean = float(entry.get('temp_mean_c', 30.0))
        temp_max  = float(entry.get('temp_max_c',  50.0))
        temp_std  = float(entry.get('temp_std_c',   5.0))

        # Get NSRDB weather features from patch
        weather = entry.get('weather_features', [])
        if len(weather) >= 3:
            ghi = float(weather[0])  # GHI W/m²
            dni = float(weather[1])  # DNI W/m²
        else:
            ghi = 300.0
            dni = 400.0

        # Efficiency score
        # Expected temp: base 25°C + irradiance heating
        # Rule: +1°C per 100 W/m² irradiance above STC
        expected_temp = 25.0 + (ghi / 100.0) * 1.0
        temp_deviation = temp_mean - expected_temp

        # High deviation → lower efficiency
        # Normal range: ±10°C → efficiency 0.5-1.0
        efficiency_score = float(np.clip(
            1.0 - (temp_deviation / 40.0), 0.0, 1.0
        ))

        # Hotspot score from thermal std + max temp
        # High std → uneven heating → possible fault
        hotspot_score = float(np.clip(
            (temp_std / 20.0) * (temp_max / 100.0),
            0.0, 1.0
        ))

        # Degradation rate estimate
        # Higher mean temp → faster degradation
        degradation_rate = float(np.clip(
            (temp_mean - 25.0) / 100.0, 0.0, 1.0
        ))

        # Maintenance flag
        maintenance_flag = float(hotspot_score > 0.7)

        return {
            'efficiency_score':  efficiency_score,
            'hotspot_score':     hotspot_score,
            'degradation_rate':  degradation_rate,
            'maintenance_flag':  maintenance_flag,
        }

    except Exception as e:
        log.debug(f'Solar label failed: {e}')
        return {
            'efficiency_score':  0.5,
            'hotspot_score':     0.0,
            'degradation_rate':  0.0,
            'maintenance_flag':  0.0,
        }


def generate_wildfire_labels(
    data_dir: Path,
    areas: List[str] = WILDFIRE_AREAS,
    years: List[int] = WILDFIRE_YEARS,
):
    """Add wildfire labels to all patch indexes."""
    nifc_dir    = data_dir / 'nifc'
    viirs_dir   = data_dir / 'viirs'
    thermal_dir = data_dir / 'landsat_thermal'
    patch_dir   = data_dir / 'patches'

    # Cache fire perimeters per year
    fire_cache: Dict = {}
    tif_cache:  Dict = {}

    for area in areas:
        for year in years:
            idx_path = (
                patch_dir / f'{area}_{year}_index.json'
            )
            if not idx_path.exists():
                log.warning(f'Index not found: {idx_path}')
                continue

            index = json.loads(idx_path.read_text())

            # Load fire perimeters for this year
            if year not in fire_cache:
                fire_cache[year] = _load_fire_perimeters(
                    nifc_dir, year
                )

            fire_gdf = fire_cache[year]
            done = skipped = failed = 0

            log.info(
                f'Generating wildfire labels: '
                f'{area} {year} — {len(index)} patches'
            )

            for i, entry in enumerate(index):
                # Skip if already labelled
                if entry.get('risk_score') is not None:
                    skipped += 1
                    continue

                month    = entry['month']
                viirs_df = _load_viirs(
                    viirs_dir, area, year, month
                )

                labels = _compute_wildfire_labels(
                    entry, fire_gdf, viirs_df,
                    tif_cache, thermal_dir
                )
                entry.update(labels)
                done += 1

                if (i + 1) % 1000 == 0:
                    log.info(
                        f'  {i+1}/{len(index)} | '
                        f'done={done} skip={skipped}'
                    )
                    idx_path.write_text(
                        json.dumps(index, indent=2)
                    )

            idx_path.write_text(
                json.dumps(index, indent=2)
            )
            log.info(
                f'✅ {area} {year}: '
                f'done={done} skip={skipped}'
            )


def generate_solar_labels(
    data_dir: Path,
    areas: List[str] = SOLAR_AREAS,
    years: List[int] = SOLAR_YEARS,
):
    """Add solar labels to all patch indexes."""
    eia_dir     = data_dir / 'eia'
    nsrdb_dir   = data_dir / 'nsrdb'
    thermal_dir = data_dir / 'landsat_thermal'
    patch_dir   = data_dir / 'patches'
    tif_cache:  Dict = {}

    for area in areas:
        for year in years:
            idx_path = (
                patch_dir / f'{area}_{year}_index.json'
            )
            if not idx_path.exists():
                log.warning(f'Index not found: {idx_path}')
                continue

            index    = json.loads(idx_path.read_text())
            eia_data = _load_eia(eia_dir, year)
            done = skipped = 0

            log.info(
                f'Generating solar labels: '
                f'{area} {year} — {len(index)} patches'
            )

            for i, entry in enumerate(index):
                if entry.get('efficiency_score') is not None:
                    skipped += 1
                    continue

                labels = _compute_solar_labels(
                    entry, eia_data, nsrdb_dir,
                    tif_cache, thermal_dir
                )
                entry.update(labels)
                done += 1

                if (i + 1) % 1000 == 0:
                    log.info(
                        f'  {i+1}/{len(index)} | '
                        f'done={done} skip={skipped}'
                    )
                    idx_path.write_text(
                        json.dumps(index, indent=2)
                    )

            idx_path.write_text(
                json.dumps(index, indent=2)
            )
            log.info(
                f'✅ {area} {year}: '
                f'done={done} skip={skipped}'
            )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    parser = argparse.ArgumentParser(
        description='ThermalWatch Label Generation'
    )
    parser.add_argument(
        '--task',
        choices=['wildfire', 'solar', 'both'],
        default='both',
    )
    parser.add_argument(
        '--wildfire-dir', default='data/wildfire'
    )
    parser.add_argument(
        '--solar-dir', default='data/solar'
    )
    args = parser.parse_args()

    if args.task in ['wildfire', 'both']:
        generate_wildfire_labels(Path(args.wildfire_dir))

    if args.task in ['solar', 'both']:
        generate_solar_labels(Path(args.solar_dir))

    log.info('✅ Label generation complete')


if __name__ == '__main__':
    main()
