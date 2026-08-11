"""
OmniGeoFusion — IoT Sensor Data Pipeline
==========================================
Fetches air quality data from Luchtmeetnet API
for Netherlands study areas.

Source:
  Luchtmeetnet: https://api.luchtmeetnet.nl/open_api
  Parameters: NO2, PM10, PM25, O3
  Coverage: 25 stations across Netherlands
  Historical: 2020-2022 available

Strategy:
  1. Fetch all station locations
  2. For each S2 acquisition date per area:
     → Fetch daily mean NO2, PM10 from nearest stations
     → Spatial interpolation to study area centroid
  3. Save as JSON to S3

Output:
  s3://omnigeofusion-data-288528696055/
    netherlands/iot/index/{area}_iot.json
"""

import os
import json
import logging
import tempfile
import numpy as np
import boto3
import requests
import yaml

from datetime import datetime, timedelta
from typing import List, Dict, Optional
from scipy.spatial import KDTree

log = logging.getLogger(__name__)
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# ── Constants ──────────────────────────────────────────────────
BASE_URL   = "https://api.luchtmeetnet.nl/open_api"
PARAMETERS = ["NO2", "PM10", "O3", "PM25"]
REQUEST_DELAY = 1.0

STUDY_TILES = {
    'amsterdam': '31UFT',
    'rotterdam': '31UFS',
    'flevoland': '31UGU',
}

# Study area centroids (lon, lat)
AREA_CENTROIDS = {
    'amsterdam': (4.9, 52.37),
    'rotterdam': (4.48, 51.92),
    'flevoland': (5.50, 52.52),
}


# ── Luchtmeetnet Client ───────────────────────────────────────
class LuchtmeetnetClient:
    """Fetches air quality data from Luchtmeetnet API."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'OmniGeoFusion/1.0'

    def get_stations(self) -> List[Dict]:
        """Get all stations with coordinates."""
        try:
            resp     = self.session.get(
                f"{BASE_URL}/stations", timeout=10
            )
            stations = resp.json().get('data', [])
            result   = []
            import time
            for s in stations:
                try:
                    r    = self.session.get(
                        f"{BASE_URL}/stations/{s['number']}",
                        timeout=10
                    )
                    info   = r.json().get('data', {})
                    coords = info.get('geometry', {}).get('coordinates', [])
                    if coords and len(coords) >= 2:
                        result.append({
                            'number':   s['number'],
                            'location': s['location'],
                            'lon':      float(coords[0]),
                            'lat':      float(coords[1]),
                        })
                except Exception as e:
                    log.debug(f'Station {s["number"]} error: {e}')
                time.sleep(0.5)
            log.info(f'Found {len(result)} stations with coords')
            return result
        except Exception as e:
            log.error(f'Failed to get stations: {e}')
            return []

    def get_daily_mean(
        self,
        station: str,
        date: str,
        formula: str
    ) -> Optional[float]:
        """Get daily mean for one station/date/parameter."""
        try:
            resp = self.session.get(
                f"{BASE_URL}/measurements",
                params={
                    "station_number": station,
                    "formula":        formula,
                    "start":          f"{date}T00:00:00",
                    "end":            f"{date}T23:59:59",
                    "page":           1,
                },
                timeout=15
            )
            data   = resp.json().get('data', [])
            values = [
                d['value'] for d in data
                if d.get('value') is not None
            ]
            return float(np.mean(values)) if values else None
        except Exception as e:
            log.debug(f'Measurement error: {e}')
            return None


# ── IoT Pipeline ──────────────────────────────────────────────
class IoTPipeline:
    """
    Fetches IoT air quality data for study areas.
    Uses spatial interpolation from nearest stations.
    """

    def __init__(self, cfg: dict):
        self.cfg           = cfg
        self.output_bucket = cfg['aws']['bucket']
        self.output_prefix = cfg['aws']['prefix'] + '/iot'
        self.s3            = boto3.client(
            's3', region_name='us-east-1',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
        )
        self.client   = LuchtmeetnetClient()
        self._stations = None  # cache stations across areas

    def load_s2_index(self, area: str) -> List[Dict]:
        """Load S2 patch index to get unique dates."""
        key = (
            f"{self.cfg['aws']['prefix']}/sentinel2/"
            f"index/{area}_index.json"
        )
        obj   = self.s3.get_object(
            Bucket=self.output_bucket, Key=key
        )
        return json.loads(obj['Body'].read())

    def interpolate_to_point(
        self,
        stations: List[Dict],
        values: Dict[str, float],
        target_lon: float,
        target_lat: float,
        k: int = 3
    ) -> Optional[float]:
        """
        IDW interpolation from k nearest stations
        to target point.
        """
        coords = np.array([
            [s['lon'], s['lat']] for s in stations
            if s['number'] in values
        ])
        vals = np.array([
            values[s['number']] for s in stations
            if s['number'] in values
        ])

        if len(vals) == 0:
            return None

        target = np.array([[target_lon, target_lat]])
        dists  = np.sqrt(
            np.sum((coords - target) ** 2, axis=1)
        )

        # IDW weights
        dists  = np.maximum(dists, 1e-6)
        k      = min(k, len(vals))
        idx    = np.argsort(dists)[:k]
        w      = 1.0 / dists[idx] ** 2
        return float(np.average(vals[idx], weights=w))

    def process_area(self, area: str) -> Dict:
        """
        Fetch IoT data for all unique S2 dates in area.
        Interpolates air quality to area centroid.
        """
        log.info(f'\n{"="*60}')
        log.info(f'IoT Pipeline: {area}')
        log.info(f'{"="*60}')

        # Get stations (cached across areas)
        if self._stations is None:
            self._stations = self.client.get_stations()
        stations = self._stations

        if not stations:
            log.error('No stations found')
            return {'area': area, 'n_records': 0}

        log.info(f'Stations: {len(stations)}')

        # Get unique dates from S2 index
        s2_index  = self.load_s2_index(area)
        dates     = sorted(set(e['date'] for e in s2_index))
        log.info(f'Unique S2 dates: {len(dates)}')

        target_lon, target_lat = AREA_CENTROIDS[area]
        records = []

        for date in dates:
            log.info(f'  Fetching {date}...')
            record = {
                'date': date,
                'area': area,
            }

            for param in PARAMETERS:
                # Fetch from all stations
                values = {}
                for station in stations:
                    val = self.client.get_daily_mean(
                        station['number'], date, param
                    )
                    if val is not None:
                        values[station['number']] = val

                log.info(
                    f'    {param}: {len(values)} stations'
                )

                # Interpolate to area centroid
                interpolated = self.interpolate_to_point(
                    stations, values,
                    target_lon, target_lat
                )
                record[f'{param}_mean']    = interpolated
                record[f'{param}_n_stns']  = len(values)

                if values:
                    record[f'{param}_min'] = float(
                        min(values.values())
                    )
                    record[f'{param}_max'] = float(
                        max(values.values())
                    )
                else:
                    record[f'{param}_min'] = None
                    record[f'{param}_max'] = None

            records.append(record)
            log.info(
                f'    ✅ {date}: '
                f'NO2={record.get("NO2_mean", "N/A")}'
            )

        # Save to S3
        index_key = (
            f"{self.output_prefix}/index/{area}_iot.json"
        )
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as tmp:
            json.dump(records, tmp, indent=2)
            tmp_path = tmp.name
        self.s3.upload_file(
            tmp_path, self.output_bucket, index_key
        )
        os.unlink(tmp_path)

        log.info(f'\n{"="*60}')
        log.info(f'✅ {area}: {len(records)} IoT records')
        log.info(
            f'   s3://{self.output_bucket}/{index_key}'
        )
        log.info(f'{"="*60}')

        return {'area': area, 'n_records': len(records)}


# ── Main ───────────────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='OmniGeoFusion IoT pipeline'
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
    pipeline = IoTPipeline(cfg)
    areas    = (
        list(STUDY_TILES.keys()) if args.area == 'all'
        else [args.area]
    )

    total = 0
    for area in areas:
        r      = pipeline.process_area(area)
        total += r['n_records']

    log.info(f'\nIoT COMPLETE: {total} records')


if __name__ == '__main__':
    main()
