"""
OmniGeoFusion — Build Multimodal Training Index
=================================================
Matches patches across all modalities by (row, col)
and builds train/val split index files.

Matching strategy:
  S2 is anchor (all patches have S2)
  SAR, LiDAR, Thermal matched by (row, col) only
  OSM features loaded from JSON index per patch_id
  IoT features loaded from area-level aggregate

Output:
  /content/omnigeofusion_data/train_index.json
  /content/omnigeofusion_data/val_index.json
  (also saved to GDrive)

Usage:
  python3 -m src.data.build_train_index \
    --data-dir /content/omnigeofusion_data \
    --gdrive-dir /gdrive/MyDrive/omnigeofusion/data \
    --train-ratio 0.85
"""

import os
import json
import random
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List

log = logging.getLogger(__name__)

AREAS = ['amsterdam', 'rotterdam', 'flevoland']


def load_index(data_dir: Path, area: str, modality: str) -> List[Dict]:
    """Load index JSON for one area + modality."""
    path = data_dir / 'indexes' / f'{area}_{modality}.json'
    if not path.exists():
        log.warning(f'Index not found: {path}')
        return []
    return json.loads(path.read_text())


def build_index(
    data_dir: str,
    gdrive_dir: str = None,
    train_ratio: float = 0.85,
    seed: int = 42,
):
    """Build multimodal training index."""
    random.seed(seed)
    data_dir   = Path(data_dir)
    gdrive_dir = Path(gdrive_dir) if gdrive_dir else None

    log.info('Loading indexes...')

    # Load all modality indexes
    s2_idx  = {a: load_index(data_dir, a, 'sentinel2') for a in AREAS}
    sar_idx = {a: load_index(data_dir, a, 'sentinel1') for a in AREAS}
    lid_idx = {a: load_index(data_dir, a, 'lidar')     for a in AREAS}
    thm_idx = {a: load_index(data_dir, a, 'thermal')   for a in AREAS}
    osm_idx = {
        a: {d['patch_id']: d for d in load_index(data_dir, a, 'osm')}
        for a in AREAS
    }
    iot_idx = {a: load_index(data_dir, a, 'iot') for a in AREAS}

    for area in AREAS:
        log.info(
            f'{area}: S2={len(s2_idx[area])} '
            f'SAR={len(sar_idx[area])} '
            f'LiDAR={len(lid_idx[area])} '
            f'Thermal={len(thm_idx[area])}'
        )

    # Build lookup by (row, col) — S2 as anchor
    lookup = {area: {} for area in AREAS}

    for area in AREAS:
        for entry in s2_idx[area]:
            local = data_dir / entry['s3_key']
            if local.exists():
                key = (entry['row'], entry['col'])
                lookup[area][key] = {
                    'patch_id':   entry['patch_id'],
                    'tile':       entry['tile'],
                    'area':       area,
                    'date':       entry['date'],
                    'row':        entry['row'],
                    'col':        entry['col'],
                    'day_gap':    90,
                    's2_t1_path': str(local),
                }

    # Match SAR, LiDAR, Thermal by (row, col) only
    for mod, mod_idx, path_key in [
        ('sentinel1', sar_idx, 'sar_path'),
        ('lidar',     lid_idx, 'lidar_path'),
        ('thermal',   thm_idx, 'thermal_path'),
    ]:
        for area in AREAS:
            mod_lookup = {}
            for e in mod_idx[area]:
                k     = (e['row'], e['col'])
                local = data_dir / e['s3_key']
                if local.exists():
                    mod_lookup[k] = str(local)

            matched = 0
            for key in lookup[area]:
                if key in mod_lookup:
                    lookup[area][key][path_key] = mod_lookup[key]
                    matched += 1

            log.info(
                f'{mod}/{area}: {matched}/{len(lookup[area])} matched'
            )

    # Add OSM + IoT features
    for area in AREAS:
        iot_data   = iot_idx[area]
        iot_vector = [0.0] * 16
        if iot_data:
            e = iot_data[0]
            iot_vector = [
                e.get('NO2_mean',  0), e.get('NO2_min',  0),
                e.get('NO2_max',   0), e.get('PM10_mean', 0),
                e.get('PM10_min',  0), e.get('PM10_max',  0),
                e.get('O3_mean',   0), e.get('O3_min',   0),
                e.get('O3_max',    0), e.get('PM25_mean', 0),
                e.get('PM25_min',  0), e.get('PM25_max',  0),
                0, 0, 0, 0,
            ]

        for key, sample in lookup[area].items():
            pid = sample['patch_id']
            # OSM
            if pid in osm_idx.get(area, {}):
                osm = osm_idx[area][pid]
                sample['osm_features'] = [
                    osm.get('building_count',       0),
                    osm.get('building_density',     0),
                    osm.get('mean_building_height', 0),
                    osm.get('building_coverage',    0),
                    osm.get('road_count',           0),
                    osm.get('road_density',         0),
                    osm.get('waterway_count',       0),
                    osm.get('lu_residential',       0),
                    osm.get('lu_farmland',          0),
                    osm.get('lu_forest',            0),
                    osm.get('lu_water',             0),
                    osm.get('lu_commercial',        0),
                    osm.get('lu_industrial',        0),
                    osm.get('urban_score',          0),
                    osm.get('population_proxy',     0),
                    osm.get('total_building_area',  0),
                ]
            else:
                sample['osm_features'] = [0.0] * 16
            # IoT
            sample['iot_features'] = iot_vector

    # Flatten + split
    all_samples = [
        sample
        for area in AREAS
        for sample in lookup[area].values()
        if 's2_t1_path' in sample
    ]

    random.shuffle(all_samples)
    split       = int(len(all_samples) * train_ratio)
    train_index = all_samples[:split]
    val_index   = all_samples[split:]

    # Coverage stats
    n = len(train_index)
    log.info(f'\n=== Coverage ({n} train samples) ===')
    for key, label in [
        ('s2_t1_path',   'S2'),
        ('sar_path',     'SAR'),
        ('lidar_path',   'LiDAR'),
        ('thermal_path', 'Thermal'),
        ('osm_features', 'OSM'),
        ('iot_features', 'IoT'),
    ]:
        if key in ('osm_features', 'iot_features'):
            count = sum(
                1 for s in train_index
                if any(v != 0 for v in s.get(key, [0]))
            )
        else:
            count = sum(1 for s in train_index if key in s)
        log.info(f'  {label:<8}: {count}/{n} ({count/n*100:.0f}%)')

    # Save indexes
    for idx_data, name in [
        (train_index, 'train'),
        (val_index,   'val')
    ]:
        # /content
        path = data_dir / f'{name}_index.json'
        path.write_text(json.dumps(idx_data, indent=2))
        log.info(f'✅ Saved {path}')

        # GDrive
        if gdrive_dir:
            gdrive_path = gdrive_dir / f'{name}_index.json'
            gdrive_path.write_text(json.dumps(idx_data, indent=2))
            log.info(f'✅ Saved {gdrive_path}')

    log.info(
        f'\n✅ Index built: '
        f'{len(train_index)} train, {len(val_index)} val'
    )
    return train_index, val_index


def main():
    parser = argparse.ArgumentParser(
        description='Build multimodal training index'
    )
    parser.add_argument(
        '--data-dir',
        default='/content/omnigeofusion_data'
    )
    parser.add_argument(
        '--gdrive-dir',
        default='/gdrive/MyDrive/omnigeofusion/data'
    )
    parser.add_argument(
        '--train-ratio', type=float, default=0.85
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )

    build_index(
        data_dir=args.data_dir,
        gdrive_dir=args.gdrive_dir,
        train_ratio=args.train_ratio,
    )


if __name__ == '__main__':
    main()
