"""
ThermalWatch — Build Training Index
=====================================
Builds train/val split index files for both
wildfire and solar tasks.

Reads patch indexes from:
  data/wildfire/patches/{area}_{year}_index.json
  data/solar/patches/{area}_{year}_index.json

Outputs:
  data/indexes/wildfire_train_index.json
  data/indexes/wildfire_val_index.json
  data/indexes/solar_train_index.json
  data/indexes/solar_val_index.json

Usage:
  python3 -m src.data.build_index \
    --wildfire-dir data/wildfire \
    --solar-dir data/solar \
    --index-dir data/indexes \
    --train-ratio 0.85
"""

import json
import logging
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

log = logging.getLogger(__name__)

WILDFIRE_AREAS = ['sierra_nevada', 'socal', 'norcal']
SOLAR_AREAS    = ['sonoran_desert', 'phoenix_metro', 'tucson']
TRAIN_YEARS    = list(range(2020, 2024))  # 2020-2023
SOLAR_YEARS    = list(range(2019, 2024))  # 2019-2023


def load_patch_indexes(
    patch_dir: Path,
    areas: List[str],
    years: List[int],
) -> List[Dict]:
    """
    Load all patch index JSON files for given areas+years.
    Raises if no index files found at all.
    """
    all_patches = []
    missing     = []

    for area in areas:
        for year in years:
            index_path = (
                patch_dir / f'{area}_{year}_index.json'
            )
            if not index_path.exists():
                missing.append(f'{area}_{year}')
                continue

            patches = json.loads(index_path.read_text())
            if not patches:
                log.warning(
                    f'Empty index: {index_path.name}'
                )
                continue

            all_patches.extend(patches)
            log.info(
                f'Loaded {len(patches)} patches: '
                f'{area} {year}'
            )

    if missing:
        log.warning(
            f'Missing patch indexes (run patch_extractor): '
            + ', '.join(missing)
        )

    if not all_patches:
        raise RuntimeError(
            f'No patch indexes found in {patch_dir}.\n'
            f'Run patch_extractor.py first.'
        )

    log.info(f'Total patches loaded: {len(all_patches)}')
    return all_patches


def split_train_val(
    patches: List[Dict],
    train_ratio: float = 0.85,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split patches into train/val sets.
    Splits by area to avoid spatial leakage:
      - Each area goes entirely to train or val
      - NOT random patch-level split

    Raises if patches list is empty.
    """
    if not patches:
        raise ValueError(
            'Cannot split empty patches list.'
        )

    random.seed(seed)

    # Group by area
    by_area: Dict[str, List[Dict]] = {}
    for p in patches:
        area = p.get('area', 'unknown')
        if area not in by_area:
            by_area[area] = []
        by_area[area].append(p)

    areas  = list(by_area.keys())
    n_train = max(1, int(len(areas) * train_ratio))

    random.shuffle(areas)
    train_areas = areas[:n_train]
    val_areas   = areas[n_train:]

    log.info(f'Train areas: {train_areas}')
    log.info(f'Val areas:   {val_areas}')

    train = [
        p for p in patches
        if p.get('area') in train_areas
    ]
    val = [
        p for p in patches
        if p.get('area') in val_areas
    ]

    if not val:
        # Fallback: random patch split if only 1 area
        log.warning(
            'Only 1 area — using random patch split '
            'instead of area split'
        )
        random.shuffle(patches)
        split = int(len(patches) * train_ratio)
        train = patches[:split]
        val   = patches[split:]

    log.info(
        f'Split: {len(train)} train / {len(val)} val'
    )
    return train, val


def build_wildfire_index(
    wildfire_dir: Path,
    index_dir: Path,
    train_ratio: float = 0.85,
) -> Tuple[Path, Path]:
    """
    Build wildfire train/val index.
    Raises if no patches available.
    """
    patch_dir = wildfire_dir / 'patches'

    if not patch_dir.exists():
        raise FileNotFoundError(
            f'Wildfire patch dir not found: {patch_dir}\n'
            f'Run patch_extractor.py --task wildfire first.'
        )

    patches = load_patch_indexes(
        patch_dir, WILDFIRE_AREAS, TRAIN_YEARS
    )
    train, val = split_train_val(patches, train_ratio)

    train_path = index_dir / 'wildfire_train_index.json'
    val_path   = index_dir / 'wildfire_val_index.json'

    train_path.write_text(json.dumps(train, indent=2))
    val_path.write_text(json.dumps(val, indent=2))

    log.info(
        f'✅ Wildfire index:\n'
        f'   Train: {len(train)} → {train_path.name}\n'
        f'   Val:   {len(val)} → {val_path.name}'
    )
    return train_path, val_path


def build_solar_index(
    solar_dir: Path,
    index_dir: Path,
    train_ratio: float = 0.85,
) -> Tuple[Path, Path]:
    """
    Build solar train/val index.
    Raises if no patches available.
    """
    patch_dir = solar_dir / 'patches'

    if not patch_dir.exists():
        raise FileNotFoundError(
            f'Solar patch dir not found: {patch_dir}\n'
            f'Run patch_extractor.py --task solar first.'
        )

    patches = load_patch_indexes(
        patch_dir, SOLAR_AREAS, SOLAR_YEARS
    )
    train, val = split_train_val(patches, train_ratio)

    train_path = index_dir / 'solar_train_index.json'
    val_path   = index_dir / 'solar_val_index.json'

    train_path.write_text(json.dumps(train, indent=2))
    val_path.write_text(json.dumps(val, indent=2))

    log.info(
        f'✅ Solar index:\n'
        f'   Train: {len(train)} → {train_path.name}\n'
        f'   Val:   {len(val)} → {val_path.name}'
    )
    return train_path, val_path


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    parser = argparse.ArgumentParser(
        description='ThermalWatch Build Training Index'
    )
    parser.add_argument(
        '--wildfire-dir',
        default='/gdrive/MyDrive/thermalwatch/data/wildfire',
    )
    parser.add_argument(
        '--solar-dir',
        default='/gdrive/MyDrive/thermalwatch/data/solar',
    )
    parser.add_argument(
        '--index-dir',
        default='/gdrive/MyDrive/thermalwatch/data/indexes',
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.85,
    )
    parser.add_argument(
        '--task',
        choices=['wildfire', 'solar', 'both'],
        default='both',
    )
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    if args.task in ['wildfire', 'both']:
        build_wildfire_index(
            Path(args.wildfire_dir),
            index_dir,
            args.train_ratio,
        )

    if args.task in ['solar', 'both']:
        build_solar_index(
            Path(args.solar_dir),
            index_dir,
            args.train_ratio,
        )


if __name__ == '__main__':
    main()
