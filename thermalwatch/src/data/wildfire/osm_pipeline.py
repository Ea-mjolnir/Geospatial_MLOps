"""
ThermalWatch — Wildfire OSM Pipeline
======================================
Downloads OSM contextual features for California
wildfire study areas using Geofabrik state extract
+ osmium-tool bbox extraction + osmium python processing.

Strategy:
  1. Download California PBF once (~1.3GB)
  2. Use osmium-tool to extract small bbox PBF per area
  3. Run osmium python single-pass on small bbox PBF
  4. Assign features to 0.02° patches via spatial binning

Features per patch:
  road_count, building_count, forest_count,
  water_count, powerline_count,
  residential_count, farmland_count

Output: JSON index per area.
Stored: data/wildfire/osm/{area}_osm.json

Requires:
  pip install osmium
  sudo apt-get install osmium-tool

Usage:
  python3 -m src.data.wildfire.osm_pipeline \
    --data-dir data/wildfire \
    --areas sierra_nevada socal norcal
"""

import json
import logging
import argparse
import subprocess
import requests
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

log = logging.getLogger(__name__)

STUDY_AREAS = {
    'sierra_nevada': {
        'bbox': [-122.0, 36.5, -118.0, 40.0],  # W,S,E,N
    },
    'socal': {
        'bbox': [-119.0, 33.5, -116.0, 35.5],
    },
    'norcal': {
        'bbox': [-123.0, 38.0, -120.0, 40.5],
    },
}

PATCH_SIZE_DEG = 0.02  # ~2km patch
GEOFABRIK_URL  = (
    'https://download.geofabrik.de/'
    'north-america/us/california-latest.osm.pbf'
)


class WildfireOSMPipeline:
    """
    Downloads OSM features for wildfire context.

    Steps:
      1. Download California PBF (~1.3GB) once
      2. Extract study area bbox PBF via osmium-tool
         (small file, fits in RAM)
      3. Single-pass osmium python extraction
      4. Assign features to patches by spatial binning

    Raises if:
      - Geofabrik download fails
      - osmium or osmium-tool not installed
      - bbox extraction fails
      - 0 patches processed
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.out_dir  = data_dir / 'osm'
        self.pbf_path = data_dir / 'california-latest.osm.pbf'
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._check_deps()

    @staticmethod
    def _check_deps():
        # Check osmium python
        try:
            import osmium
        except ImportError:
            raise ImportError(
                'osmium not installed.\n'
                'Run: pip install osmium'
            )
        # Check osmium-tool
        result = subprocess.run(
            ['osmium', '--version'],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                'osmium-tool not installed.\n'
                'Run: sudo apt-get install osmium-tool'
            )

    def _download_pbf(self) -> Path:
        """Download California OSM PBF from Geofabrik."""
        if self.pbf_path.exists():
            size = self.pbf_path.stat().st_size / 1e6
            log.info(
                f'Skip (exists): california-latest.osm.pbf '
                f'({size:.0f}MB)'
            )
            return self.pbf_path

        log.info(
            'Downloading California OSM from Geofabrik '
            '(~1.3GB)...'
        )
        r = requests.get(
            GEOFABRIK_URL,
            stream=True,
            timeout=600,
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
                    if done % (50*1024*1024) < 1024*1024:
                        log.info(
                            f'  {pct:.0f}% '
                            f'({done/1e6:.0f}/'
                            f'{total/1e6:.0f}MB)'
                        )

        size = self.pbf_path.stat().st_size / 1e6
        log.info(f'✅ California OSM: {size:.0f}MB')
        return self.pbf_path

    def _extract_bbox_pbf(
        self,
        area: str,
        bbox: List[float],
    ) -> Path:
        """
        Use osmium-tool to extract study area bbox
        from full California PBF.
        Output is a small PBF that fits in RAM.
        Raises if osmium-tool fails.
        """
        out = self.out_dir / f'{area}.osm.pbf'
        if out.exists():
            size = out.stat().st_size / 1e6
            log.info(
                f'Skip (exists): {out.name} ({size:.0f}MB)'
            )
            return out

        west, south, east, north = (
            bbox[0], bbox[1], bbox[2], bbox[3]
        )
        bbox_str = f'{west},{south},{east},{north}'

        log.info(
            f'Extracting {area} bbox from California PBF...'
        )
        result = subprocess.run(
            [
                'osmium', 'extract',
                '--bbox', bbox_str,
                '--output', str(out),
                '--overwrite',
                str(self.pbf_path),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f'osmium extract failed for {area}:\n'
                f'{result.stderr}'
            )

        size = out.stat().st_size / 1e6
        log.info(f'✅ Extracted {area}: {size:.1f}MB')
        return out

    def _extract_area_features(
        self,
        area_pbf: Path,
        bbox: List[float],
    ) -> Dict:
        """
        Single-pass osmium extraction on small bbox PBF.
        Returns dict keyed by (row, col) with feature counts.
        """
        import osmium

        west, south, east, north = (
            bbox[0], bbox[1], bbox[2], bbox[3]
        )
        ps = PATCH_SIZE_DEG

        patch_features = defaultdict(lambda: {
            'road_count':        0,
            'building_count':    0,
            'forest_count':      0,
            'water_count':       0,
            'powerline_count':   0,
            'residential_count': 0,
            'farmland_count':    0,
        })

        class AreaHandler(osmium.SimpleHandler):
            def way(self, w):
                tags = {t.k: t.v for t in w.tags}

                feat = None
                if 'highway' in tags:
                    feat = 'road_count'
                elif 'building' in tags:
                    feat = 'building_count'
                elif (tags.get('landuse') == 'forest' or
                      tags.get('natural') == 'wood'):
                    feat = 'forest_count'
                elif (tags.get('natural') == 'water' or
                      'waterway' in tags):
                    feat = 'water_count'
                elif tags.get('power') == 'line':
                    feat = 'powerline_count'
                elif tags.get('landuse') == 'residential':
                    feat = 'residential_count'
                elif tags.get('landuse') == 'farmland':
                    feat = 'farmland_count'
                else:
                    return

                lons, lats = [], []
                try:
                    for n in w.nodes:
                        if not n.location.valid():
                            continue
                        lon = n.location.lon
                        lat = n.location.lat
                        if (west <= lon <= east and
                                south <= lat <= north):
                            lons.append(lon)
                            lats.append(lat)
                except osmium.InvalidLocationError:
                    return

                if not lons:
                    return

                clon = sum(lons) / len(lons)
                clat = sum(lats) / len(lats)
                row  = int((clat - south) / ps)
                col  = int((clon - west)  / ps)
                patch_features[(row, col)][feat] += 1

        handler = AreaHandler()
        handler.apply_file(str(area_pbf), locations=True)

        log.info(
            f'OSM extracted: {len(patch_features)} '
            f'patches with features'
        )
        return dict(patch_features)

    def download(self, area: str) -> Path:
        """
        Full pipeline for one area:
          1. Download California PBF
          2. Extract area bbox PBF
          3. Extract features
          4. Build patch index
        Raises if any step fails.
        """
        if area not in STUDY_AREAS:
            raise ValueError(
                f'Unknown area: {area}. '
                f'Choose from: {list(STUDY_AREAS.keys())}'
            )

        out = self.out_dir / f'{area}_osm.json'
        if out.exists():
            log.info(f'Skip (exists): {out.name}')
            return out

        cfg  = STUDY_AREAS[area]
        bbox = cfg['bbox']
        west, south, east, north = (
            bbox[0], bbox[1], bbox[2], bbox[3]
        )
        ps = PATCH_SIZE_DEG

        # Step 1: Download full California PBF
        self._download_pbf()

        # Step 2: Extract small area bbox PBF
        area_pbf = self._extract_bbox_pbf(area, bbox)

        # Step 3: Extract features from small PBF
        log.info(f'OSM single-pass extraction: {area}')
        patch_features = self._extract_area_features(
            area_pbf, bbox
        )

        # Step 4: Build patch index
        results = []
        lat = south
        while lat < north:
            lon = west
            while lon < east:
                row = int((lat  - south) / ps)
                col = int((lon  - west)  / ps)
                patch_id = (
                    f'{area}_r{row:04d}_c{col:04d}'
                )
                feats = patch_features.get(
                    (row, col),
                    {
                        'road_count':        0,
                        'building_count':    0,
                        'forest_count':      0,
                        'water_count':       0,
                        'powerline_count':   0,
                        'residential_count': 0,
                        'farmland_count':    0,
                    }
                )
                results.append({
                    'patch_id': patch_id,
                    'bbox': [
                        round(lon, 4),
                        round(lat, 4),
                        round(lon + ps, 4),
                        round(lat + ps, 4),
                    ],
                    **feats,
                })
                lon += ps
            lat += ps

        if not results:
            raise RuntimeError(
                f'OSM: 0 patches built for {area}.'
            )

        out.write_text(json.dumps(results, indent=2))
        log.info(
            f'✅ OSM: {len(results)} patches → {out.name}'
        )
        return out

    def download_all(
        self,
        areas: List[str] = list(STUDY_AREAS.keys()),
    ) -> Dict[str, Path]:
        """Download PBF once, extract for all areas."""
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
            )

        log.info(
            f'✅ OSM wildfire complete: {len(results)} areas'
        )
        return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    parser = argparse.ArgumentParser(
        description='ThermalWatch Wildfire OSM Pipeline'
    )
    parser.add_argument(
        '--data-dir', default='data/wildfire'
    )
    parser.add_argument(
        '--areas',
        nargs='+',
        default=list(STUDY_AREAS.keys()),
        choices=list(STUDY_AREAS.keys()),
    )
    args = parser.parse_args()

    pipeline = WildfireOSMPipeline(Path(args.data_dir))
    pipeline.download_all(args.areas)


if __name__ == '__main__':
    main()
