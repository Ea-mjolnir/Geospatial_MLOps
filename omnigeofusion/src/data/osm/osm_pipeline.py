"""
OmniGeoFusion — OSM Data Pipeline (Sparse Sampling)
=====================================================
Extracts OSM features for 50 representative patches
per study area via Overpass API.

Strategy:
  - Sample 50 spatially distributed patches from S2 index
  - One combined Overpass query per patch (not 4)
  - Sequential with 2s delay (respect rate limits)
  - Retry with backoff on errors
  - Save JSON features to S3

Output:
  s3://omnigeofusion-data-288528696055/
    netherlands/osm/index/{area}_osm.json
"""

import os
import json
import time
import logging
import tempfile
import numpy as np
import boto3
import requests
import yaml

from typing import List, Dict, Optional
from pyproj import Transformer
from shapely.geometry import Polygon, LineString

log = logging.getLogger(__name__)
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# ── Constants ──────────────────────────────────────────────────
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
PATCHES_PER_AREA = 1666
REQUEST_DELAY    = 1.0   # seconds between requests
MAX_RETRIES      = 3

TILE_ORIGINS = {
    '31UFT': (600000.0, 5800020.0),
    '31UFS': (600000.0, 5700000.0),
    '31UGU': (699960.0, 5900040.0),
}

STUDY_TILES = {
    'amsterdam': '31UFT',
    'rotterdam': '31UFS',
    'flevoland': '31UGU',
}

T_32631_TO_4326 = Transformer.from_crs(
    'EPSG:32631', 'EPSG:4326', always_xy=True
)


# ── Helpers ───────────────────────────────────────────────────
def patch_bbox_lonlat(
    tile: str, row: int, col: int
) -> List[float]:
    """Compute patch bbox in WGS84."""
    origin_x, origin_y = TILE_ORIGINS[tile]
    x_min = origin_x + col * 10
    y_max = origin_y - row * 10
    x_max = x_min + 2560
    y_min = y_max - 2560
    lon_min, lat_min = T_32631_TO_4326.transform(x_min, y_min)
    lon_max, lat_max = T_32631_TO_4326.transform(x_max, y_max)
    return [lon_min, lat_min, lon_max, lat_max]


def query_overpass(
    bbox: List[float],
    timeout: int = 60
) -> dict:
    """
    Single combined Overpass query for all OSM layers.
    Returns raw JSON response.
    """
    s, w, n, e = bbox[1], bbox[0], bbox[3], bbox[2]
    bbox_str   = f'{s},{w},{n},{e}'

    query = f"""
[out:json][timeout:{timeout}];
(
  way["building"]({bbox_str});
  way["highway"]({bbox_str});
  way["landuse"]({bbox_str});
  way["waterway"]({bbox_str});
  way["natural"]({bbox_str});
);
out body geom;
"""
    session = requests.Session()
    session.headers['User-Agent'] = 'OmniGeoFusion/1.0'

    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(
                OVERPASS_URL,
                params={"data": query},
                timeout=timeout + 10
            )
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code in [429, 504]:
                wait = (attempt + 1) * 10
                log.warning(
                    f'Rate limited ({resp.status_code}), '
                    f'waiting {wait}s...'
                )
                time.sleep(wait)
            else:
                log.warning(f'HTTP {resp.status_code}')
                break
        except Exception as e:
            log.debug(f'Attempt {attempt+1} failed: {e}')
            time.sleep(5)

    return {'elements': []}


def extract_features(
    elements: List[dict],
    bbox: List[float]
) -> Dict:
    """Extract statistical features from Overpass elements."""
    patch_area = 2560 ** 2  # m²

    buildings  = []
    roads      = []
    landuse    = {}
    waterways  = 0

    for elem in elements:
        tags = elem.get('tags', {})
        geom = elem.get('geometry', [])

        if 'building' in tags:
            coords = [
                (n['lon'], n['lat'])
                for n in geom if 'lon' in n
            ]
            if len(coords) >= 3:
                try:
                    poly = Polygon(coords)
                    buildings.append({
                        'area':   poly.area * 1e10,
                        'type':   tags.get('building', 'yes'),
                        'height': float(
                            str(tags.get('height', '3'))
                            .replace('m', '').strip()
                        ) if tags.get('height') else 3.0,
                    })
                except Exception:
                    pass

        elif 'highway' in tags:
            coords = [
                (n['lon'], n['lat'])
                for n in geom if 'lon' in n
            ]
            if len(coords) >= 2:
                roads.append({
                    'class':    tags.get('highway', 'unknown'),
                    'maxspeed': tags.get('maxspeed', '50'),
                })

        elif 'landuse' in tags or 'natural' in tags:
            lu = tags.get('landuse', tags.get('natural', 'unknown'))
            landuse[lu] = landuse.get(lu, 0) + 1

        elif 'waterway' in tags:
            waterways += 1

    # Building stats
    n_bld  = len(buildings)
    b_area = sum(b['area'] for b in buildings)
    b_ht   = np.mean([b['height'] for b in buildings]) if buildings else 0.0

    # Road stats
    n_rd   = len(roads)

    # Landuse fractions
    lu_residential = landuse.get('residential', 0)
    lu_farmland    = landuse.get('farmland', 0) + landuse.get('farmyard', 0)
    lu_forest      = landuse.get('forest', 0) + landuse.get('wood', 0)
    lu_water       = landuse.get('water', 0)
    lu_commercial  = landuse.get('commercial', 0) + landuse.get('retail', 0)
    lu_industrial  = landuse.get('industrial', 0)

    return {
        'building_count':        n_bld,
        'building_density':      n_bld / patch_area * 1e6,
        'mean_building_height':  float(b_ht),
        'total_building_area':   float(b_area),
        'building_coverage':     float(b_area / patch_area),
        'population_proxy':      float(b_area * b_ht / 3 / 30),
        'road_count':            n_rd,
        'road_density':          n_rd / patch_area * 1e6,
        'waterway_count':        waterways,
        'lu_residential':        lu_residential,
        'lu_farmland':           lu_farmland,
        'lu_forest':             lu_forest,
        'lu_water':              lu_water,
        'lu_commercial':         lu_commercial,
        'lu_industrial':         lu_industrial,
        'urban_score':           float(
            n_bld * 0.4 + n_rd * 0.3 +
            lu_commercial * 0.3
        ),
    }


# ── OSM Pipeline ──────────────────────────────────────────────
class OSMSparsePipeline:
    """
    Sparse OSM pipeline — 50 patches per area.
    Uses single combined Overpass query per patch.
    """

    def __init__(self, cfg: dict):
        self.cfg           = cfg
        self.output_bucket = cfg['aws']['bucket']
        self.output_prefix = cfg['aws']['prefix'] + '/osm'
        self.s3            = boto3.client(
            's3', region_name='us-east-1',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
        )

    def load_s2_index(self, area: str) -> List[Dict]:
        key = (
            f"{self.cfg['aws']['prefix']}/sentinel2/"
            f"index/{area}_index.json"
        )
        obj   = self.s3.get_object(
            Bucket=self.output_bucket, Key=key
        )
        return json.loads(obj['Body'].read())

    def sample_patches(
        self, index: List[Dict], n: int = PATCHES_PER_AREA
    ) -> List[Dict]:
        """
        Spatially uniform sample of n patches.
        Uses linspace to ensure even distribution.
        """
        if len(index) <= n:
            return index
        indices = np.linspace(0, len(index)-1, n, dtype=int)
        return [index[i] for i in indices]

    def process_patch(
        self, tile: str, area: str, entry: Dict
    ) -> Optional[Dict]:
        """Process one patch — query + extract features."""
        bbox     = patch_bbox_lonlat(tile, entry['row'], entry['col'])
        data     = query_overpass(bbox)
        elements = data.get('elements', [])

        if not elements:
            log.warning(f'No elements for {entry["patch_id"]}')
            features = extract_features([], bbox)
        else:
            features = extract_features(elements, bbox)

        features['patch_id'] = entry['patch_id']
        features['tile']     = tile
        features['area']     = area
        features['row']      = entry['row']
        features['col']      = entry['col']
        features['date']     = entry['date']
        return features

    def process_area(self, area: str) -> Dict:
        tile = STUDY_TILES[area]

        log.info(f'\n{"="*60}')
        log.info(f'OSM Sparse Pipeline: {area} ({tile})')
        log.info(f'Target: {PATCHES_PER_AREA} patches')
        log.info(f'{"="*60}')

        # Load + sample S2 index
        s2_index = self.load_s2_index(area)
        samples  = self.sample_patches(s2_index)
        log.info(
            f'Sampled {len(samples)} from '
            f'{len(s2_index)} patches'
        )

        results = []
        for i, entry in enumerate(samples):
            log.info(
                f'  [{i+1}/{len(samples)}] '
                f'{entry["patch_id"]}'
            )
            result = self.process_patch(tile, area, entry)
            if result:
                results.append(result)
                log.info(
                    f'    buildings={result["building_count"]} '
                    f'roads={result["road_count"]}'
                )

            # Rate limit delay
            time.sleep(REQUEST_DELAY)

        # Save to S3
        index_key = (
            f"{self.output_prefix}/index/{area}_osm.json"
        )
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as tmp:
            json.dump(results, tmp, indent=2)
            tmp_path = tmp.name
        self.s3.upload_file(
            tmp_path, self.output_bucket, index_key
        )
        os.unlink(tmp_path)

        log.info(f'\n{"="*60}')
        log.info(f'✅ {area}: {len(results)} OSM patches')
        log.info(
            f'   s3://{self.output_bucket}/{index_key}'
        )
        log.info(f'{"="*60}')

        return {'area': area, 'n_patches': len(results)}


# ── Main ───────────────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='OmniGeoFusion OSM sparse pipeline'
    )
    parser.add_argument(
        '--config', default='configs/data_config.yaml'
    )
    parser.add_argument(
        '--area',
        choices=['amsterdam', 'rotterdam', 'flevoland', 'all'],
        default='all'
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )

    cfg      = yaml.safe_load(open(args.config))
    pipeline = OSMSparsePipeline(cfg)
    areas    = (
        list(STUDY_TILES.keys()) if args.area == 'all'
        else [args.area]
    )

    total = 0
    for area in areas:
        r      = pipeline.process_area(area)
        total += r['n_patches']

    log.info(f'\nOSM COMPLETE: {total} patches')


if __name__ == '__main__':
    main()
