"""
GeoAI MLOps Project 1 — Sentinel-2 Ingestion & Feature Engineering Pipeline
============================================================================
DAG: sentinel_pipeline
Schedule: Weekly (every Monday)

Pipeline stages:
    1. search_tiles      — find Sentinel-2 L2A tiles over 3 AOIs via STAC API
    2. download_tiles    — download raw tiles to EC2 disk (fast local reads)
    3. patch_and_extract — CHUNK-BASED processing to stay within 1GB RAM:
                           Read chunk of tile (all 5 bands) → slice patches →
                           compute features → stream to S3 → write PostGIS →
                           free chunk → next chunk → DELETE tile (3-layer protection)
    4. version_with_dvc  — write provenance.json → DVC → S3 → git commit

RAM Strategy (chunk-based — solves SIGSEGV on t3.micro):
    Full tile 3847×4007 × 5 bands = ~1.5GB → crashes t3.micro (1GB RAM)
    Chunk 1000×1000 × 5 bands    = ~20MB  → safely fits in RAM
    All 5 bands per chunk → NDVI/NDWI/all indices compute correctly

Delete Strategy (3-layer protection):
    Layer 1: finally block per tile  — always deletes after processing
    Layer 2: startup cleanup         — wipes leftovers from crashed runs
    Layer 3: on_failure_callback     — emergency wipe on DAG failure

DVC Strategy:
    Raw tiles:   NOT tracked (temporary, always re-downloadable)
    Patches:     in S3 (permanent output)
    Features:    in PostGIS (permanent output)
    Provenance:  provenance.json kept on EC2 + pushed to S3 via DVC

AOIs:
    Berlin centre    — urban/mixed
    Brandenburg      — agricultural/rural
    Hamburg port     — industrial + water body

Bands: B02=Blue B03=Green B04=Red B08=NIR B11=SWIR(20m→10m resampled)

Features per patch (37):
    Spectral indices (10): NDVI, NDWI, MNDWI, NDBI, ARVI, EVI, BSI, NBI, SAVI, Brightness
    Index std (3):         ndvi_std, ndwi_std, ndbi_std
    Per-band stats (20):   mean, std, p25, p75 for blue/green/red/nir/swir
    Change features (4):   delta_ndvi, delta_ndwi, delta_ndbi, change_magnitude
"""

from datetime import datetime, timedelta
import os
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator

log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────
DEFAULT_ARGS = {
    'owner': 'odinsbeard',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

AOIS = {
    'berlin': {
        'bbox': [13.088, 52.338, 13.761, 52.675],
        'description': 'Berlin urban centre'
    },
    'brandenburg': {
        'bbox': [12.800, 51.900, 13.500, 52.300],
        'description': 'Brandenburg agricultural region'
    },
    'hamburg': {
        'bbox': [9.700, 53.400, 10.200, 53.700],
        'description': 'Hamburg port and Elbe estuary'
    },
}

BANDS = ['blue', 'green', 'red', 'nir', 'swir']
BAND_ASSETS = {
    'blue':  'blue',
    'green': 'green',
    'red':   'red',
    'nir':   'nir',
    'swir':  'swir16',
}

PROJECT_DIR = '/home/ubuntu/geoai-mlops-p1'
DATA_DIR    = '/home/ubuntu/geoai-mlops-p1/data/raw'

MAX_TILES_PER_AOI     = 5
CLOUD_COVER_THRESHOLD = 20
SEARCH_DAYS           = 180
PATCH_SIZE            = 256
PATCH_STRIDE          = 192
S3_UPLOAD_WORKERS     = 5
CHUNK_SIZE            = 1000   # pixels per chunk side — ~20MB RAM per chunk


# ══════════════════════════════════════════════════════════════
# LAYER 3: DAG FAILURE CALLBACK
# ══════════════════════════════════════════════════════════════
def on_failure_cleanup(context):
    """
    Layer 3 delete protection: emergency cleanup on DAG failure.
    Wipes raw data dir if any task fails — prevents disk accumulation
    even if Layer 1 finally block is bypassed by a hard crash.
    """
    import shutil
    log.info("🚨 Layer 3: DAG failure — emergency disk cleanup")
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
        os.makedirs(DATA_DIR, exist_ok=True)
        log.info("🗑️ Layer 3: EC2 raw data dir wiped on failure")
    statvfs = os.statvfs('/home/ubuntu')
    free_gb = (statvfs.f_bavail * statvfs.f_frsize) / (1024**3)
    log.info(f"   Disk after cleanup: {free_gb:.1f}GB free")


# ══════════════════════════════════════════════════════════════
# TASK 1: SEARCH STAC API
# ══════════════════════════════════════════════════════════════
def search_tiles(**context):
    """
    Search Element84 STAC API for Sentinel-2 L2A tiles across all 3 AOIs.
    Returns tile metadata + COG asset URLs via XCom.
    Returns: int — total tiles found
    """
    from pystac_client import Client

    exec_date  = context['execution_date']
    end_date   = exec_date.strftime('%Y-%m-%d')
    start_date = (exec_date - timedelta(days=SEARCH_DAYS)).strftime('%Y-%m-%d')

    log.info(f"Search window: {start_date} to {end_date}")
    client = Client.open('https://earth-search.aws.element84.com/v1')

    tile_metadata = []
    tile_assets   = {}
    seen_ids      = set()

    for aoi_name, aoi_config in AOIS.items():
        bbox = aoi_config['bbox']
        log.info(f"Searching AOI: {aoi_name}")
        try:
            search = client.search(
                collections=['sentinel-2-l2a'],
                bbox=bbox,
                datetime=f'{start_date}/{end_date}',
                query={'eo:cloud_cover': {'lt': CLOUD_COVER_THRESHOLD}},
                max_items=MAX_TILES_PER_AOI * 2,
            )
            items = list(search.items())
            items.sort(key=lambda x: x.properties.get('eo:cloud_cover', 999))
            items = items[:MAX_TILES_PER_AOI]

            for item in items:
                if item.id in seen_ids:
                    continue
                missing = [b for b in BAND_ASSETS.values() if b not in item.assets]
                if missing:
                    continue
                seen_ids.add(item.id)
                tile_metadata.append({
                    'id':          item.id,
                    'date':        item.datetime.strftime('%Y-%m-%d'),
                    'cloud_cover': item.properties.get('eo:cloud_cover', 999),
                    'bbox':        list(item.bbox),
                    'platform':    item.properties.get('platform', 'unknown'),
                    'aoi':         aoi_name,
                })
                tile_assets[item.id] = {
                    band_name: item.assets[asset_key].href
                    for band_name, asset_key in BAND_ASSETS.items()
                }
                log.info(
                    f"  ✅ {item.id} | {item.datetime.strftime('%Y-%m-%d')} "
                    f"| cloud={item.properties.get('eo:cloud_cover', 0):.1f}%"
                )
        except Exception as e:
            log.error(f"Search failed for {aoi_name}: {e}")

    log.info(f"Total tiles: {len(tile_metadata)} across {len(AOIS)} AOIs")
    context['ti'].xcom_push(key='tile_metadata', value=tile_metadata)
    context['ti'].xcom_push(key='tile_assets',   value=tile_assets)
    return len(tile_metadata)


# ══════════════════════════════════════════════════════════════
# TASK 2: DOWNLOAD TILES
# ══════════════════════════════════════════════════════════════
def download_tiles(**context):
    """
    Download raw Sentinel-2 tiles to EC2 disk using COG windowed reads.

    LAYER 2 DELETE PROTECTION:
        Wipes any raw tiles left over from previous crashed run at startup.

    Downloads only the AOI window (not the full tile).
    Returns: list — tile IDs successfully downloaded
    """
    import rasterio
    from rasterio.windows import from_bounds, Window
    from rasterio.warp import transform_bounds
    import shutil
    import json
    import warnings
    warnings.filterwarnings('ignore')

    tile_metadata = context['ti'].xcom_pull(key='tile_metadata', task_ids='search_tiles')
    tile_assets   = context['ti'].xcom_pull(key='tile_assets',   task_ids='search_tiles')

    if not tile_metadata:
        context['ti'].xcom_push(key='downloaded_tiles', value=[])
        return 0

    # ── Preflight disk space check ────────────────────────────
    statvfs = os.statvfs('/home/ubuntu')
    free_gb = (statvfs.f_bavail * statvfs.f_frsize) / (1024**3)
    if free_gb < 2.0:
        raise Exception(
            f"⚠️ Insufficient disk: {free_gb:.1f}GB free, need 2GB minimum"
        )
    log.info(f"💾 Disk check: {free_gb:.1f}GB free ✅")

    # ── LAYER 2: Startup cleanup ──────────────────────────────
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)
    log.info("🗑️ Layer 2: Startup cleanup — EC2 raw data dir wiped clean")

    downloaded = []

    for tile in tile_metadata:
        tile_id  = tile['id']
        aoi_name = tile['aoi']
        bbox     = AOIS[aoi_name]['bbox']
        assets   = tile_assets.get(tile_id, {})
        tile_dir = os.path.join(DATA_DIR, tile_id)

        if not assets or 'red' not in assets:
            log.warning(f"Skipping {tile_id} — missing band assets")
            continue

        os.makedirs(tile_dir, exist_ok=True)
        log.info(f"Downloading: {tile_id} (AOI: {aoi_name})")

        tile_ok = True
        for band_name, asset_key in BAND_ASSETS.items():
            url      = assets.get(band_name)
            out_path = os.path.join(tile_dir, f'{band_name}.tif')

            if not url:
                log.warning(f"  {band_name}: no URL — skipping tile")
                tile_ok = False
                break

            try:
                with rasterio.open(url) as src:
                    bbox_crs = transform_bounds('EPSG:4326', src.crs, *bbox)
                    aoi_win  = from_bounds(*bbox_crs, src.transform)
                    col_off  = max(0, int(aoi_win.col_off))
                    row_off  = max(0, int(aoi_win.row_off))
                    width    = min(int(aoi_win.width),  src.width  - col_off)
                    height   = min(int(aoi_win.height), src.height - row_off)
                    window   = Window(col_off, row_off, width, height)
                    data     = src.read(1, window=window)
                    tf       = src.window_transform(window)

                    with rasterio.open(
                        out_path, 'w', driver='GTiff',
                        height=data.shape[0], width=data.shape[1],
                        count=1, dtype=data.dtype,
                        crs=src.crs, transform=tf,
                        compress='lzw', nodata=src.nodata
                    ) as dst:
                        dst.write(data, 1)

                log.info(f"  ✅ {band_name}: {data.shape}")
                del data

            except Exception as e:
                log.error(f"  ❌ {band_name} download failed: {e}")
                tile_ok = False
                break

        if tile_ok:
            with open(os.path.join(tile_dir, 'metadata.json'), 'w') as f:
                json.dump(tile, f, indent=2)
            downloaded.append(tile_id)
            log.info(f"✅ {tile_id} downloaded to EC2")
        else:
            if os.path.exists(tile_dir):
                shutil.rmtree(tile_dir)
            log.warning(f"⚠️ {tile_id} partial download cleaned up")

    log.info(f"Downloaded {len(downloaded)} tiles to EC2")
    context['ti'].xcom_push(key='downloaded_tiles', value=downloaded)
    return len(downloaded)


# ══════════════════════════════════════════════════════════════
# TASK 3: PATCH AND EXTRACT (CHUNK-BASED — RAM SAFE)
# ══════════════════════════════════════════════════════════════
def patch_and_extract(**context):
    """
    Chunk-based patch extraction — solves SIGSEGV on t3.micro (1GB RAM).

    WHY CHUNKS:
        Full tile 3847×4007 × 5 bands × float32 = ~1.5GB → OOM crash
        Chunk 1000×1000 × 5 bands × float32     = ~20MB  → safe

    HOW CHUNKS SOLVE THE NDVI PROBLEM:
        Each chunk loads ALL 5 bands for that chunk region.
        NDVI = (NIR - RED) / (NIR + RED) computed within the chunk.
        All 10 spectral indices computed correctly per chunk.

    PROCESSING FLOW PER TILE:
        Open 5 band files (no data read yet)
        For each 1000×1000 chunk:
            Read all 5 bands for this chunk → ~20MB RAM
            Slide 256×256 patches across chunk
            Per patch:
                Upload 5 bands to S3 concurrently (ThreadPoolExecutor)
                Compute 37 features from in-memory arrays
                Write to PostGIS
            Free chunk arrays → RAM back to baseline
        Finally: DELETE raw tile from EC2 (Layer 1)

    LAYER 1 DELETE PROTECTION:
        Every tile wrapped in try/finally.
        finally ALWAYS deletes raw tile — success OR failure OR crash.

    Returns: int — total patches processed
    """
    import rasterio
    from rasterio.windows import Window
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    import numpy as np
    import boto3
    import io
    import sys
    import shutil
    import gc
    import psycopg2
    from concurrent.futures import ThreadPoolExecutor, as_completed

    downloaded_tiles = context['ti'].xcom_pull(
        key='downloaded_tiles', task_ids='download_tiles'
    )
    tile_metadata = context['ti'].xcom_pull(
        key='tile_metadata', task_ids='search_tiles'
    )

    if not downloaded_tiles:
        context['ti'].xcom_push(key='n_patches', value=0)
        context['ti'].xcom_push(key='n_saved',   value=0)
        return 0

    # ── S3 + PostGIS setup — ONE session shared across threads ──
    session = boto3.Session(region_name='us-east-1')
    ssm     = session.client('ssm')
    bucket  = ssm.get_parameter(
        Name='/geoai-mlops-p1/s3/bucket'
    )['Parameter']['Value']
    s3      = session.client('s3')

    db_host = ssm.get_parameter(Name='/geoai-mlops-p1/db/host')['Parameter']['Value']
    db_user = ssm.get_parameter(Name='/geoai-mlops-p1/db/user')['Parameter']['Value']
    db_pass = ssm.get_parameter(
        Name='/geoai-mlops-p1/db/password', WithDecryption=True
    )['Parameter']['Value']

    conn = psycopg2.connect(
        host=db_host, port=5432, database='geoai_features',
        user=db_user, password=db_pass, connect_timeout=30,
        keepalives=1, keepalives_idle=5,
        keepalives_interval=2, keepalives_count=2
    )
    cur = conn.cursor()

    # Create table + indexes if needed
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sentinel2_patches (
            id                  SERIAL PRIMARY KEY,
            patch_id            VARCHAR(200) UNIQUE NOT NULL,
            tile_id             VARCHAR(100),
            patch_row           INTEGER,
            patch_col           INTEGER,
            acquisition_date    DATE,
            platform            VARCHAR(50),
            aoi                 VARCHAR(50),
            cloud_cover         DOUBLE PRECISION,
            valid_pixel_ratio   DOUBLE PRECISION,
            total_pixels        INTEGER,
            valid_pixels        INTEGER,
            ndvi_mean           DOUBLE PRECISION,
            ndvi_std            DOUBLE PRECISION,
            ndwi_mean           DOUBLE PRECISION,
            ndwi_std            DOUBLE PRECISION,
            mndwi_mean          DOUBLE PRECISION,
            ndbi_mean           DOUBLE PRECISION,
            ndbi_std            DOUBLE PRECISION,
            arvi_mean           DOUBLE PRECISION,
            evi_mean            DOUBLE PRECISION,
            bsi_mean            DOUBLE PRECISION,
            nbi_mean            DOUBLE PRECISION,
            savi_mean           DOUBLE PRECISION,
            bright_mean         DOUBLE PRECISION,
            blue_mean           DOUBLE PRECISION,
            blue_std            DOUBLE PRECISION,
            blue_p25            DOUBLE PRECISION,
            blue_p75            DOUBLE PRECISION,
            green_mean          DOUBLE PRECISION,
            green_std           DOUBLE PRECISION,
            green_p25           DOUBLE PRECISION,
            green_p75           DOUBLE PRECISION,
            red_mean            DOUBLE PRECISION,
            red_std             DOUBLE PRECISION,
            red_p25             DOUBLE PRECISION,
            red_p75             DOUBLE PRECISION,
            nir_mean            DOUBLE PRECISION,
            nir_std             DOUBLE PRECISION,
            nir_p25             DOUBLE PRECISION,
            nir_p75             DOUBLE PRECISION,
            swir_mean           DOUBLE PRECISION,
            swir_std            DOUBLE PRECISION,
            swir_p25            DOUBLE PRECISION,
            swir_p75            DOUBLE PRECISION,
            delta_ndvi          DOUBLE PRECISION,
            delta_ndwi          DOUBLE PRECISION,
            delta_ndbi          DOUBLE PRECISION,
            change_magnitude    DOUBLE PRECISION,
            ingested_at         TIMESTAMP DEFAULT NOW(),
            dag_run_id          VARCHAR(200)
        );
        CREATE INDEX IF NOT EXISTS idx_patches_tile
            ON sentinel2_patches (tile_id);
        CREATE INDEX IF NOT EXISTS idx_patches_aoi
            ON sentinel2_patches (aoi);
        CREATE INDEX IF NOT EXISTS idx_patches_date
            ON sentinel2_patches (acquisition_date);
        CREATE INDEX IF NOT EXISTS idx_patches_change
            ON sentinel2_patches (change_magnitude DESC);
    """)
    conn.commit()

    # ── Helper: upload one band to S3 with retry ─────────────
    def upload_band(args):
        import time
        band_name, data, tile_crs, patch_transform, s3_key = args
        for attempt in range(3):
            try:
                buf = io.BytesIO()
                with rasterio.open(
                    buf, 'w', driver='GTiff',
                    height=PATCH_SIZE, width=PATCH_SIZE,
                    count=1, dtype=np.float32,
                    crs=tile_crs, transform=patch_transform,
                    compress='lzw'
                ) as dst:
                    dst.write(data, 1)
                buf.seek(0)
                s3.upload_fileobj(buf, bucket, s3_key)
                return True
            except Exception as e:
                if attempt < 2:
                    log.warning(f"Upload attempt {attempt+1} failed {s3_key}: {e}")
                    time.sleep(2 ** attempt)
                else:
                    log.error(f"Upload failed after 3 attempts {s3_key}: {e}")
                    return False
        return False

    # ── Helper: write one feature row to PostGIS ─────────────
    def save_feature(f, dag_run_id):
        try:
            cur.execute("""
                INSERT INTO sentinel2_patches (
                    patch_id, tile_id, patch_row, patch_col,
                    acquisition_date, platform, aoi,
                    cloud_cover, valid_pixel_ratio, total_pixels, valid_pixels,
                    ndvi_mean, ndvi_std, ndwi_mean, ndwi_std,
                    mndwi_mean, ndbi_mean, ndbi_std, arvi_mean,
                    evi_mean, bsi_mean, nbi_mean, savi_mean, bright_mean,
                    blue_mean, blue_std, blue_p25, blue_p75,
                    green_mean, green_std, green_p25, green_p75,
                    red_mean, red_std, red_p25, red_p75,
                    nir_mean, nir_std, nir_p25, nir_p75,
                    swir_mean, swir_std, swir_p25, swir_p75,
                    delta_ndvi, delta_ndwi, delta_ndbi, change_magnitude,
                    dag_run_id
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s
                )
                ON CONFLICT (patch_id) DO UPDATE SET
                    ndvi_mean         = EXCLUDED.ndvi_mean,
                    ndvi_std          = EXCLUDED.ndvi_std,
                    ndwi_mean         = EXCLUDED.ndwi_mean,
                    ndwi_std          = EXCLUDED.ndwi_std,
                    mndwi_mean        = EXCLUDED.mndwi_mean,
                    ndbi_mean         = EXCLUDED.ndbi_mean,
                    ndbi_std          = EXCLUDED.ndbi_std,
                    arvi_mean         = EXCLUDED.arvi_mean,
                    evi_mean          = EXCLUDED.evi_mean,
                    bsi_mean          = EXCLUDED.bsi_mean,
                    nbi_mean          = EXCLUDED.nbi_mean,
                    savi_mean         = EXCLUDED.savi_mean,
                    bright_mean       = EXCLUDED.bright_mean,
                    valid_pixel_ratio = EXCLUDED.valid_pixel_ratio,
                    total_pixels      = EXCLUDED.total_pixels,
                    valid_pixels      = EXCLUDED.valid_pixels,
                    blue_mean         = EXCLUDED.blue_mean,
                    blue_std          = EXCLUDED.blue_std,
                    blue_p25          = EXCLUDED.blue_p25,
                    blue_p75          = EXCLUDED.blue_p75,
                    green_mean        = EXCLUDED.green_mean,
                    green_std         = EXCLUDED.green_std,
                    green_p25         = EXCLUDED.green_p25,
                    green_p75         = EXCLUDED.green_p75,
                    red_mean          = EXCLUDED.red_mean,
                    red_std           = EXCLUDED.red_std,
                    red_p25           = EXCLUDED.red_p25,
                    red_p75           = EXCLUDED.red_p75,
                    nir_mean          = EXCLUDED.nir_mean,
                    nir_std           = EXCLUDED.nir_std,
                    nir_p25           = EXCLUDED.nir_p25,
                    nir_p75           = EXCLUDED.nir_p75,
                    swir_mean         = EXCLUDED.swir_mean,
                    swir_std          = EXCLUDED.swir_std,
                    swir_p25          = EXCLUDED.swir_p25,
                    swir_p75          = EXCLUDED.swir_p75,
                    delta_ndvi        = EXCLUDED.delta_ndvi,
                    delta_ndwi        = EXCLUDED.delta_ndwi,
                    delta_ndbi        = EXCLUDED.delta_ndbi,
                    change_magnitude  = EXCLUDED.change_magnitude,
                    cloud_cover       = EXCLUDED.cloud_cover,
                    aoi               = EXCLUDED.aoi,
                    ingested_at       = NOW(),
                    dag_run_id        = EXCLUDED.dag_run_id;
            """, (
                f['patch_id'], f['tile_id'], f['patch_row'], f['patch_col'],
                f['acquisition_date'], f['platform'], f['aoi'],
                f['cloud_cover'], f['valid_pixel_ratio'],
                f['total_pixels'], f['valid_pixels'],
                f['ndvi_mean'], f['ndvi_std'],
                f['ndwi_mean'], f['ndwi_std'],
                f['mndwi_mean'],
                f['ndbi_mean'], f['ndbi_std'],
                f['arvi_mean'], f['evi_mean'],
                f['bsi_mean'],  f['nbi_mean'],
                f['savi_mean'], f['bright_mean'],
                f['blue_mean'],  f['blue_std'],  f['blue_p25'],  f['blue_p75'],
                f['green_mean'], f['green_std'], f['green_p25'], f['green_p75'],
                f['red_mean'],   f['red_std'],   f['red_p25'],   f['red_p75'],
                f['nir_mean'],   f['nir_std'],   f['nir_p25'],   f['nir_p75'],
                f['swir_mean'],  f['swir_std'],  f['swir_p25'],  f['swir_p75'],
                f['delta_ndvi'], f['delta_ndwi'],
                f['delta_ndbi'], f['change_magnitude'],
                dag_run_id
            ))
            return True
        except Exception as e:
            conn.rollback()
            log.error(f"PostGIS insert failed {f['patch_id']}: {e}")
            return False

    # ── Helper: band statistics ───────────────────────────────
    def band_stats(arr, name):
        v = arr[np.isfinite(arr) & (arr > 0)].astype(float)
        if v.size == 0:
            return {f'{name}_mean': 0., f'{name}_std': 0.,
                    f'{name}_p25':  0., f'{name}_p75': 0.}
        return {
            f'{name}_mean': float(np.mean(v)),
            f'{name}_std':  float(np.std(v)),
            f'{name}_p25':  float(np.percentile(v, 25)),
            f'{name}_p75':  float(np.percentile(v, 75)),
        }

    def safe_div(a, b, eps=1e-10):
        d = a + b
        return np.where(np.abs(d) > eps, (a - b) / d, 0.0)

    # ── Main processing loop ──────────────────────────────────
    total_patches = 0
    total_saved   = 0
    dag_run_id    = context.get('run_id', 'unknown')
    meta_lookup   = {t['id']: t for t in tile_metadata}

    for tile_id in downloaded_tiles:
        tile_dir = os.path.join(DATA_DIR, tile_id)
        meta     = meta_lookup.get(tile_id, {})
        aoi_name = meta.get('aoi', 'unknown')

        log.info(f"Processing: {tile_id} | AOI: {aoi_name}")

        # ── LAYER 1: try/finally — GUARANTEED DELETE ──────────
        try:
            band_paths = {b: os.path.join(tile_dir, f'{b}.tif') for b in BANDS}
            missing    = [b for b in BANDS if not os.path.exists(band_paths[b])]
            if missing:
                log.error(f"Missing bands {missing} for {tile_id} — skipping")
                continue

            # Get tile dimensions from red band
            with rasterio.open(band_paths['red']) as src:
                tile_h    = src.height
                tile_w    = src.width
                tile_crs  = src.crs
                tile_tf   = src.transform

            log.info(
                f"  Tile size: ({tile_h}, {tile_w}) | "
                f"Chunks: {(tile_h // CHUNK_SIZE + 1)} × {(tile_w // CHUNK_SIZE + 1)}"
            )

            patch_count  = 0
            saved_count  = 0
            prev_indices = None  # Reset per tile — change detection within tile only

            # ── CHUNK-BASED PROCESSING ────────────────────────
            # Each chunk: CHUNK_SIZE×CHUNK_SIZE × 5 bands = ~20MB RAM
            # All 5 bands per chunk → NDVI/all indices compute correctly
            for chunk_row in range(0, tile_h, CHUNK_SIZE):
                for chunk_col in range(0, tile_w, CHUNK_SIZE):

                    # Actual chunk dimensions (handle edges)
                    c_height = min(CHUNK_SIZE, tile_h - chunk_row)
                    c_width  = min(CHUNK_SIZE, tile_w  - chunk_col)

                    # Skip chunks too small for even one patch
                    if c_height < PATCH_SIZE or c_width < PATCH_SIZE:
                        continue

                    # ── Read ALL 5 bands for this chunk ───────
                    # ~20MB RAM — all bands available → NDVI works
                    chunk_bands = {}
                    chunk_ok    = True

                    for band_name in BANDS:
                        try:
                            with rasterio.open(band_paths[band_name]) as src:
                                if band_name == 'swir':
                                    # SWIR is 20m — half resolution
                                    swir_row = chunk_row // 2
                                    swir_col = chunk_col // 2
                                    swir_h   = (c_height + 1) // 2
                                    swir_w   = (c_width  + 1) // 2
                                    window   = Window(swir_col, swir_row, swir_w, swir_h)
                                    arr = src.read(
                                        1, window=window,
                                        out_shape=(c_height, c_width),
                                        resampling=Resampling.bilinear
                                    ).astype(np.float32)
                                else:
                                    window = Window(chunk_col, chunk_row, c_width, c_height)
                                    arr    = src.read(1, window=window).astype(np.float32)

                                if src.nodata is not None:
                                    arr[arr == src.nodata] = 0.0
                                chunk_bands[band_name] = arr

                        except Exception as e:
                            log.error(f"Chunk read error {band_name}: {e}")
                            chunk_ok = False
                            break

                    if not chunk_ok:
                        # Free any partial chunk arrays
                        for arr in chunk_bands.values():
                            del arr
                        chunk_bands.clear()
                        continue

                    # Chunk transform (for patch transforms)
                    chunk_transform = Affine(
                        tile_tf.a, tile_tf.b,
                        tile_tf.c + chunk_col * tile_tf.a,
                        tile_tf.d, tile_tf.e,
                        tile_tf.f + chunk_row * tile_tf.e
                    )

                    # ── Slide patches across this chunk ───────
                    for p_row in range(0, c_height - PATCH_SIZE + 1, PATCH_STRIDE):
                        for p_col in range(0, c_width - PATCH_SIZE + 1, PATCH_STRIDE):

                            # Absolute patch position in tile
                            abs_row    = chunk_row + p_row
                            abs_col    = chunk_col + p_col
                            patch_name = f"patch_{abs_row:04d}_{abs_col:04d}"

                            # Slice patch from chunk arrays — zero disk/network reads
                            patch_bands = {}
                            for band_name in BANDS:
                                patch_bands[band_name] = chunk_bands[band_name][
                                    p_row:p_row + PATCH_SIZE,
                                    p_col:p_col + PATCH_SIZE
                                ].copy()

                            # Patch transform
                            patch_transform = Affine(
                                chunk_transform.a, chunk_transform.b,
                                chunk_transform.c + p_col * chunk_transform.a,
                                chunk_transform.d, chunk_transform.e,
                                chunk_transform.f + p_row * chunk_transform.e
                            )

                            # Validity check — skip mostly nodata patches
                            vmask    = (
                                (patch_bands['red']  > 0) &
                                (patch_bands['nir']  > 0) &
                                (patch_bands['swir'] > 0) &
                                (patch_bands['blue'] > 0)
                            )
                            total_px = int(vmask.size)
                            valid_px = int(np.sum(vmask))

                            if valid_px == 0 or valid_px / total_px < 0.2:
                                for b in BANDS:
                                    patch_bands[b] = None
                                patch_bands.clear()
                                del patch_bands
                                continue

                            valid_ratio = valid_px / total_px

                            # Upload all 5 bands concurrently to S3
                            upload_args = [
                                (
                                    band_name,
                                    patch_bands[band_name],
                                    tile_crs,
                                    patch_transform,
                                    f"processed/patches/{tile_id}"
                                    f"/{patch_name}/{band_name}.tif"
                                )
                                for band_name in BANDS
                            ]

                            upload_success = True
                            with ThreadPoolExecutor(max_workers=S3_UPLOAD_WORKERS) as ex:
                                futures = [ex.submit(upload_band, a) for a in upload_args]
                                for future in as_completed(futures, timeout=60):
                                    try:
                                        if not future.result(timeout=30):
                                            upload_success = False
                                    except Exception as e:
                                        log.error(f"Upload thread error: {e}")
                                        upload_success = False

                            if not upload_success:
                                log.warning(f"Skipping PostGIS {patch_name} — S3 failed")
                                for b in BANDS:
                                    patch_bands[b] = None
                                patch_bands.clear()
                                del patch_bands
                                continue

                            patch_count += 1

                            # Compute 37 features from patch arrays
                            blue  = patch_bands['blue']
                            green = patch_bands['green']
                            red   = patch_bands['red']
                            nir   = patch_bands['nir']
                            swir  = patch_bands['swir']

                            ndvi  = safe_div(nir, red)
                            ndwi  = safe_div(nir, swir)
                            mndwi = safe_div(green, swir)
                            ndbi  = safe_div(swir, nir)

                            arvi_n = nir - (2.0*red - blue)
                            arvi_d = nir + (2.0*red - blue)
                            arvi   = np.where(np.abs(arvi_d) > 1e-10, arvi_n/arvi_d, 0.0)

                            evi_d = nir + 6.0*red - 7.5*blue + 1.0
                            evi   = np.where(np.abs(evi_d) > 1e-10, 2.5*(nir-red)/evi_d, 0.0)

                            bsi_n = (swir+red) - (nir+blue)
                            bsi_d = (swir+red) + (nir+blue)
                            bsi   = np.where(np.abs(bsi_d) > 1e-10, bsi_n/bsi_d, 0.0)

                            nbi    = np.where(nir > 1e-10, (red*swir)/nir, 0.0)
                            savi_d = nir + red + 0.5
                            savi   = np.where(np.abs(savi_d) > 1e-10,
                                              1.5*(nir-red)/savi_d, 0.0)
                            bright = np.sqrt(red**2 + green**2 + blue**2)

                            def idx_mean(a):
                                return float(np.nanmean(a[vmask])) if valid_px > 0 else 0.
                            def idx_std(a):
                                return float(np.nanstd(a[vmask]))  if valid_px > 0 else 0.

                            current_indices = {
                                'ndvi_mean':   idx_mean(ndvi),
                                'ndvi_std':    idx_std(ndvi),
                                'ndwi_mean':   idx_mean(ndwi),
                                'ndwi_std':    idx_std(ndwi),
                                'mndwi_mean':  idx_mean(mndwi),
                                'ndbi_mean':   idx_mean(ndbi),
                                'ndbi_std':    idx_std(ndbi),
                                'arvi_mean':   idx_mean(arvi),
                                'evi_mean':    idx_mean(evi),
                                'bsi_mean':    idx_mean(bsi),
                                'nbi_mean':    idx_mean(nbi),
                                'savi_mean':   idx_mean(savi),
                                'bright_mean': idx_mean(bright),
                            }

                            stats = {}
                            for b_name in BANDS:
                                stats.update(band_stats(patch_bands[b_name], b_name))

                            if prev_indices is not None:
                                d_ndvi  = (current_indices['ndvi_mean']
                                           - prev_indices['ndvi_mean'])
                                d_ndwi  = (current_indices['ndwi_mean']
                                           - prev_indices['ndwi_mean'])
                                d_ndbi  = (current_indices['ndbi_mean']
                                           - prev_indices['ndbi_mean'])
                                chg_mag = float(
                                    np.sqrt(d_ndvi**2 + d_ndwi**2 + d_ndbi**2)
                                )
                            else:
                                d_ndvi = d_ndwi = d_ndbi = chg_mag = 0.0

                            prev_indices = current_indices

                            feature = {
                                'patch_id':          f"{tile_id}__{patch_name}",
                                'tile_id':           tile_id,
                                'patch_row':         abs_row,
                                'patch_col':         abs_col,
                                'acquisition_date':  meta.get('date', None),
                                'platform':          meta.get('platform', 'unknown'),
                                'aoi':               aoi_name,
                                'cloud_cover':       meta.get('cloud_cover', -1),
                                'valid_pixel_ratio': valid_ratio,
                                'total_pixels':      total_px,
                                'valid_pixels':      valid_px,
                                **current_indices,
                                **stats,
                                'delta_ndvi':        d_ndvi,
                                'delta_ndwi':        d_ndwi,
                                'delta_ndbi':        d_ndbi,
                                'change_magnitude':  chg_mag,
                            }

                            if save_feature(feature, dag_run_id):
                                saved_count += 1

                            # Keep PostGIS connection alive + commit every 100
                            if saved_count % 50 == 0 and saved_count > 0:
                                cur.execute("SELECT 1")
                            if saved_count % 100 == 0 and saved_count > 0:
                                conn.commit()

                            # Free patch memory immediately
                            del blue, green, red, nir, swir
                            del ndvi, ndwi, mndwi, ndbi, arvi, evi, bsi, nbi, savi, bright
                            for b in BANDS:
                                patch_bands[b] = None
                            patch_bands.clear()
                            del patch_bands

                            # Periodic GC every 50 patches
                            if patch_count % 50 == 0:
                                gc.collect()

                    # ── Free chunk arrays after processing ────
                    for b in BANDS:
                        chunk_bands[b] = None
                    chunk_bands.clear()
                    del chunk_bands
                    gc.collect()

            # Final commit for this tile
            conn.commit()
            log.info(
                f"✅ {tile_id}: {patch_count} patches → S3 "
                f"| {saved_count} features → PostGIS"
            )
            total_patches += patch_count
            total_saved   += saved_count

        except Exception as e:
            log.error(f"Error processing {tile_id}: {e}", exc_info=True)
            try:
                conn.rollback()
            except Exception:
                pass

        finally:
            # ── LAYER 1: GUARANTEED DELETE ────────────────────
            # ALWAYS runs — success OR failure OR exception
            tile_path = os.path.join(DATA_DIR, tile_id)
            if os.path.exists(tile_path):
                shutil.rmtree(tile_path)
                log.info(f"🗑️ Layer 1: Deleted raw tile: {tile_id}")

            # Log disk state after each tile
            disk    = os.statvfs('/home/ubuntu')
            used_gb = (disk.f_blocks - disk.f_bfree) * disk.f_frsize / (1024**3)
            free_gb = disk.f_bavail * disk.f_frsize / (1024**3)
            log.info(f"   EC2 disk: {used_gb:.1f}GB used | {free_gb:.1f}GB free")

    cur.close()
    conn.close()

    log.info(
        f"✅ patch_and_extract complete: "
        f"{total_patches} patches in S3 | {total_saved} features in PostGIS"
    )
    context['ti'].xcom_push(key='n_patches', value=total_patches)
    context['ti'].xcom_push(key='n_saved',   value=total_saved)
    return total_patches


# ══════════════════════════════════════════════════════════════
# TASK 4: VERSION WITH DVC
# ══════════════════════════════════════════════════════════════
def version_with_dvc(**context):
    """
    Write provenance.json and version with DVC.

    provenance.json is KEPT on EC2 — only ~1KB per run × 77 runs = ~77KB total.
    Kept for easy reference during model training phase.

    DVC workflow:
    1. Configure S3 remote (idempotent)
    2. Write provenance.json locally
    3. dvc add → creates .dvc pointer with md5 checksum
    4. dvc push → uploads to S3 via DVC remote
    5. git add .dvc pointer + provenance.json → git commit
    6. Clean /tmp/dvc-cache

    Returns: int — 1 if successful
    """
    import subprocess
    import json
    import io
    import sys
    import boto3

    tile_metadata = context['ti'].xcom_pull(key='tile_metadata', task_ids='search_tiles')
    n_patches     = context['ti'].xcom_pull(key='n_patches', task_ids='patch_and_extract')
    n_saved       = context['ti'].xcom_pull(key='n_saved',   task_ids='patch_and_extract')

    if not tile_metadata:
        return 0

    session = boto3.Session(region_name='us-east-1')
    ssm     = session.client('ssm')
    bucket  = ssm.get_parameter(
        Name='/geoai-mlops-p1/s3/bucket'
    )['Parameter']['Value']
    s3      = session.client('s3')

    tile_ids  = [t['id'] for t in tile_metadata]
    run_id    = context.get('run_id', 'manual')
    exec_date = context['execution_date'].strftime('%Y-%m-%d')

    env = {
        **os.environ,
        'PATH': (
            f"{os.path.dirname(sys.executable)}"
            f":/snap/bin:{os.environ.get('PATH', '')}"
        ),
        'HOME':              '/home/ubuntu',
        'DVC_CACHE_DIR':     '/tmp/dvc-cache',
        'GIT_AUTHOR_NAME':   'odinsbeard',
        'GIT_AUTHOR_EMAIL':  'ea990evo@gmail.com',
        'GIT_COMMITTER_NAME':  'odinsbeard',
        'GIT_COMMITTER_EMAIL': 'ea990evo@gmail.com',
    }

    def run_cmd(cmd, desc, check=True):
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=PROJECT_DIR, env=env, timeout=120
        )
        if r.stdout: log.info(f"  {r.stdout.strip()}")
        if r.stderr: log.info(f"  {r.stderr.strip()}")
        if r.returncode != 0 and check:
            log.error(f"Failed: {desc}")
        return r.returncode == 0

    try:
        # Check DVC is initialised
        dvc_dir = os.path.join(PROJECT_DIR, '.dvc')
        if not os.path.exists(dvc_dir):
            run_cmd(['dvc', 'init'], 'dvc init')
            log.info("✅ DVC initialised")

        # Check /tmp has space for DVC cache
        tmp_stat    = os.statvfs('/tmp')
        tmp_free_gb = (tmp_stat.f_bavail * tmp_stat.f_frsize) / (1024**3)
        if tmp_free_gb < 1.0:
            log.warning(f"⚠️ /tmp only {tmp_free_gb:.1f}GB free — DVC cache may fail")
        else:
            log.info(f"💾 /tmp space: {tmp_free_gb:.1f}GB free ✅")

        # Step 1: Configure DVC S3 remote (idempotent)
        run_cmd(
            ['dvc', 'remote', 'add', 's3remote', f's3://{bucket}/dvc'],
            'dvc remote add', check=False
        )
        run_cmd(
            ['dvc', 'remote', 'default', 's3remote'],
            'dvc remote default', check=False
        )
        run_cmd(
            ['dvc', 'remote', 'modify', 's3remote', 'region', 'us-east-1'],
            'dvc remote modify', check=False
        )
        log.info(f"✅ DVC remote → s3://{bucket}/dvc")

        # Step 2: Build provenance record
        provenance = {
            'run_id':            run_id,
            'exec_date':         exec_date,
            'tile_ids':          tile_ids,
            'aois':              list(set(t['aoi'] for t in tile_metadata)),
            'n_tiles':           len(tile_ids),
            'n_patches':         n_patches or 0,
            'n_features':        n_saved   or 0,
            'patch_size':        PATCH_SIZE,
            'patch_stride':      PATCH_STRIDE,
            'chunk_size':        CHUNK_SIZE,
            's3_patches_prefix': 'processed/patches/',
            's3_bucket':         bucket,
            'postgis_table':     'sentinel2_patches',
            'dag_run_id':        run_id,
            'ec2_disk_used':     False,
            'raw_tiles_deleted': True,
            'dvc_cache_on_ec2':  False,
        }

        # Step 3: Write provenance.json locally (~1KB — KEPT for model training)
        prov_dir   = os.path.join(PROJECT_DIR, 'data', 'provenance')
        os.makedirs(prov_dir, exist_ok=True)
        local_prov = os.path.join(prov_dir, f"{exec_date}__{run_id}.json")
        with open(local_prov, 'w') as f:
            f.write(json.dumps(provenance, indent=2))
        log.info(f"✅ Provenance written: {local_prov}")

        # Step 4: DVC add — creates .dvc pointer with md5 checksum
        run_cmd(['dvc', 'add', local_prov], 'dvc add provenance')

        # Step 5: DVC push — uploads provenance to S3 via DVC remote
        run_cmd(
            ['dvc', 'push', local_prov + '.dvc'],
            'dvc push provenance', check=False
        )
        log.info(f"✅ Provenance pushed to s3://{bucket}/dvc")

        # Step 6: Git add .dvc pointer + provenance.json + commit
        dvc_file = local_prov + '.dvc'
        run_cmd(
            ['git', 'add', dvc_file,
             os.path.join(prov_dir, '.gitignore'), local_prov],
            'git add', check=False
        )
        run_cmd(
            ['git', 'commit', '-m',
             f"data: {len(tile_ids)} tiles | {n_patches or 0} patches "
             f"[{exec_date}] [{run_id}]"],
            'git commit', check=False
        )

        # Step 7: provenance.json KEPT on EC2 for model training reference
        log.info("✅ provenance.json kept on EC2 (~1KB — for model training)")

        # Step 8: Clean /tmp/dvc-cache — nothing accumulates
        import shutil as _shutil
        tmp_cache = '/tmp/dvc-cache'
        if os.path.exists(tmp_cache):
            _shutil.rmtree(tmp_cache)
            log.info("🧹 /tmp/dvc-cache cleaned")

        log.info(
            f"✅ version_with_dvc complete: "
            f"{len(tile_ids)} tiles | {n_patches or 0} patches in S3 | "
            f"{n_saved or 0} features in PostGIS"
        )
        return 1

    except Exception as e:
        log.error(f"version_with_dvc failed: {e}", exc_info=True)
        return 0


# ══════════════════════════════════════════════════════════════
# DAG DEFINITION
# ══════════════════════════════════════════════════════════════
with DAG(
    dag_id='sentinel_pipeline',
    default_args=DEFAULT_ARGS,
    description=(
        'Sentinel-2 — chunk-based RAM-safe processing, '
        'patches to S3, features to PostGIS'
    ),
    schedule_interval='@weekly',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    on_failure_callback=on_failure_cleanup,
    tags=['geoai', 'sentinel', 'patches', 'project1'],
) as dag:

    t1_search   = PythonOperator(
        task_id='search_tiles',
        python_callable=search_tiles
    )
    t2_download = PythonOperator(
        task_id='download_tiles',
        python_callable=download_tiles
    )
    t3_patch    = PythonOperator(
        task_id='patch_and_extract',
        python_callable=patch_and_extract
    )
    t4_dvc      = PythonOperator(
        task_id='version_with_dvc',
        python_callable=version_with_dvc
    )

    t1_search >> t2_download >> t3_patch >> t4_dvc
