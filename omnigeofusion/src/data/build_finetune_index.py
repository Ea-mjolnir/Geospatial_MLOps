"""
OmniGeoFusion — Build Fine-tuning Index
=========================================
Builds task-specific train/val indexes for fine-tuning.
Matches patches with targets per task/area.

Tasks:
  urban_change → amsterdam (31UFT)
  flood_damage → rotterdam (31UFS)
  agriculture  → flevoland (31UGU)

Output:
  /content/omnigeofusion_data/urban_change/train_index.json
  /content/omnigeofusion_data/urban_change/val_index.json
  (same for flood_damage, agriculture)
"""

import os
import json
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List

log = logging.getLogger(__name__)

TASK_CONFIG = {
    'urban_change': {'area': 'amsterdam', 'tile': '31UFT'},
    'flood_damage': {'area': 'rotterdam',  'tile': '31UFS'},
    'agriculture':  {'area': 'flevoland',  'tile': '31UGU'},
}

TASK_TARGET_KEY = {
    'urban_change': 'urban_change',
    'flood_damage': 'flood_risk',
    'agriculture':  'crop_health',
}

MODALITIES = ['sentinel2', 'sentinel1', 'lidar', 'thermal']

MOD_KEYS = {
    'sentinel2': 's2_t1_path',
    'sentinel1': 'sar_path',
    'lidar':     'lidar_path',
    'thermal':   'thermal_path',
}


def load_json(path: Path) -> list:
    if not path.exists():
        log.warning(f'Not found: {path}')
        return []
    return json.loads(path.read_text())


def build_finetune_index(
    data_dir: str,
    gdrive_dir: str = None,
    train_ratio: float = 0.85,
    seed: int = 42,
):
    random.seed(seed)
    data_dir   = Path(data_dir)
    gdrive_dir = Path(gdrive_dir) if gdrive_dir else None

    for task, cfg in TASK_CONFIG.items():
        area       = cfg['area']
        tile       = cfg['tile']
        target_key = TASK_TARGET_KEY[task]

        log.info(f'\n=== {task} ({area}) ===')

        # Load modality indexes
        mod_lookups = {}
        for mod in MODALITIES:
            idx = load_json(
                data_dir / 'indexes' / f'{area}_{mod}.json'
            )
            lookup = {}
            for e in idx:
                local = data_dir / e['s3_key']
                if local.exists():
                    lookup[(e['row'], e['col'])] = str(local)
            mod_lookups[mod] = lookup
            log.info(f'  {mod}: {len(lookup)} patches')

        # Load targets
        targets_path = data_dir / 'indexes' / f'{area}_targets.json'
        if not targets_path.exists():
            # Try alternative path
            targets_path = data_dir / 'targets' / f'{area}_targets.json'
        targets = load_json(targets_path)
        tgt_lookup = {t['patch_id']: t for t in targets}
        log.info(f'  targets: {len(tgt_lookup)}')

        # Load OSM + IoT
        osm_data = load_json(
            data_dir / 'indexes' / f'{area}_osm.json'
        )
        osm_lookup = {d['patch_id']: d for d in osm_data}

        iot_data   = load_json(
            data_dir / 'indexes' / f'{area}_iot.json'
        )
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

        # Build S2 anchor index
        s2_idx = load_json(
            data_dir / 'indexes' / f'{area}_sentinel2.json'
        )

        samples = []
        for entry in s2_idx:
            local_s2 = data_dir / entry['s3_key']
            if not local_s2.exists():
                continue

            pid = entry['patch_id']
            key = (entry['row'], entry['col'])

            # Skip if no target
            if pid not in tgt_lookup:
                continue

            target = tgt_lookup[pid]

            sample = {
                'patch_id':   pid,
                'tile':       tile,
                'area':       area,
                'date':       entry['date'],
                'row':        entry['row'],
                'col':        entry['col'],
                'day_gap':    90,
                's2_t1_path': str(local_s2),
            }

            # Add other modalities
            for mod in ['sentinel1', 'lidar', 'thermal']:
                path_key = MOD_KEYS[mod]
                if key in mod_lookups[mod]:
                    sample[path_key] = mod_lookups[mod][key]

            # Add OSM features
            if pid in osm_lookup:
                osm = osm_lookup[pid]
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

            # Add IoT features
            sample['iot_features'] = iot_vector

            # Add target values
            sample.update({
                k: v for k, v in target.items()
                if k not in sample
            })

            samples.append(sample)

        log.info(f'  Total matched: {len(samples)}')

        # Train/val split
        random.shuffle(samples)
        split      = int(len(samples) * train_ratio)
        train_idx  = samples[:split]
        val_idx    = samples[split:]

        # Save to data_dir/task/
        task_dir = data_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / 'train_index.json').write_text(
            json.dumps(train_idx, indent=2)
        )
        (task_dir / 'val_index.json').write_text(
            json.dumps(val_idx, indent=2)
        )

        # Save to GDrive
        if gdrive_dir:
            gdrive_task = gdrive_dir / task
            gdrive_task.mkdir(parents=True, exist_ok=True)
            (gdrive_task / 'train_index.json').write_text(
                json.dumps(train_idx, indent=2)
            )
            (gdrive_task / 'val_index.json').write_text(
                json.dumps(val_idx, indent=2)
            )

        log.info(
            f'  ✅ {task}: train={len(train_idx)} val={len(val_idx)}'
        )


def main():
    parser = argparse.ArgumentParser(
        description='Build fine-tuning index'
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

    build_finetune_index(
        data_dir=args.data_dir,
        gdrive_dir=args.gdrive_dir,
        train_ratio=args.train_ratio,
    )


if __name__ == '__main__':
    main()
