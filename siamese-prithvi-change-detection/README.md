# GeoAI MLOps Project 1 — Siamese Prithvi Change Detection

## Overview
A production-grade geospatial AI pipeline for detecting land cover change
using Sentinel-2 satellite imagery. Built on the Prithvi-EO-1.0-100M
foundation model with a weakly supervised training approach.

## Architecture
Sentinel-2 patches (T1, T2)
↓
Prithvi-EO-1.0-100M encoder (shared weights — Siamese)
↓
embedding_diff = embedding_t2 - embedding_t1
↓
MLP fusion head (embedding_diff + tabular features + day_gap)
↓
change score [0, 1]
## Model
- **Backbone**: Prithvi-EO-1.0-100M (86M parameters) via TerraTorch
- **Architecture**: SiamesePrithviModel
- **Input**: 5-band Sentinel-2 patches (B02, B03, B04, B08, B11) at 256×256px
- **Tabular features**: 34 spectral indices + band statistics per patch
- **Output**: scalar change score per patch pair
- **Target**: sqrt(d_ndvi² + d_ndwi² + d_ndbi²) — weakly supervised

## Training
### Dataset
- **Train**: Berlin + Hamburg patch pairs (138,948 pairs)
- **Validation**: Berlin + Hamburg patch pairs (25,092 pairs)
- **Test**: Brandenburg patch pairs (55,881 pairs) — held out entirely
- **Spatial split**: by location to prevent leakage

### Incremental Training
Training was done in 6 chunks due to Colab memory constraints:

| Chunk | Pairs | Val R2 | Test R2 | Spearman |
|-------|-------|--------|---------|----------|
| 2     | 26,512 | 0.9764 | 0.9673 | 0.9878  |
| 3     | 26,512 | 0.9851 | 0.9764 | 0.9910  |
| 4     | 26,512 | 0.9888 | 0.9805 | 0.9934  |
| 5     | 26,512 | 0.9907 | 0.9852 | 0.9944  |
| 6     | 6,388  | 0.9919 | 0.9849 | 0.9942  |

**Best model**: `best_model_chunk6_r2_0.9919.pt`
**Final Test R2**: 0.9849 | **Spearman**: 0.9942

### Two-phase training
- **Phase 1** (3 epochs): Freeze Prithvi encoder, train MLP head only
- **Phase 2** (5 epochs): Unfreeze encoder, full fine-tuning at low LR

## Inference — Munich Change Detection
Inference on completely unseen Munich data (MGRS tile 32UPU):
- **T1**: 2020-07-07 (Sentinel-2A)
- **T2**: 2023-07-07 (Sentinel-2B)
- **Day gap**: 1,095 days (~3 years)
- **Total patches**: 833

| Change Class  | Patches | Percentage |
|---------------|---------|------------|
| No change     | 268     | 32.2%      |
| Low change    | 556     | 66.8%      |
| Medium change | 9       | 1.1%       |
| High change   | 0       | 0.0%       |

### Outputs
- `munich_change_score.tif` — 374MB GeoTIFF at 10m resolution
- `munich_change_class.tif` — 94MB classified raster
- `munich_change_map.geojson` — vector polygons with change scores
- `munich_medium_change.geojson` — 9 medium change hotspot patches

## Infrastructure
- **EC2**: t3.micro (i-017e056f0ebb8d382) — MLflow + data processing
- **S3**: geoai-mlops-p1-data-288528696055 — patches + repo bundles
- **RDS**: PostgreSQL (PostGIS) — patch metadata + features
- **MLflow**: http://3.91.55.220:5000 — experiment tracking
- **Training**: Google Colab (T4 GPU)

## Project Structure
geoai-mlops-p1/
├── src/
│ ├── training/
│ │ ├── model.py # SiamesePrithviModel architecture
│ │ ├── dataset.py # SentinelPairDataset
│ │ ├── train.py # Training pipeline
│ │ └── train_config.json # Default config
│ ├── inference/
│ │ └── inference_munich.py # Munich inference pipeline
│ ├── preprocessing/
│ │ └── generate_pairs.py # Pair generation from PostGIS
│ └── features/
├── notebooks/
│ ├── Geoai-Training---v1.0.ipynb # Colab training notebook
│ └── Geoai-inference-V1.0.ipynb # Colab inference notebook
├── dags/
│ └── sentinel_pipeline.py # Airflow DAG for data ingestion
└── README.md
## Setup
### Requirements
```bash
pip install terratorch torch==2.2.0 transformers timm einops
pip install mlflow psycopg2-binary boto3 rasterio scipy
pip install pandas tqdm geopandas shapely
```

### Training (Google Colab)
1. Open `notebooks/Geoai-Training---v1.0.ipynb` in Colab
2. Set `CHUNK_NUMBER` in Cell 5
3. Run Cells 1 → 2 → 3 → 4 → 5 → 5b → 6 → 7 → 8

### Inference (Google Colab)
1. Open `notebooks/Geoai-inference-V1.0.ipynb` in Colab
2. Run Cells 1 → 2 → 3 → 6 → 9

## Key Design Decisions
1. **Weakly supervised target**: sqrt(d_ndvi² + d_ndwi² + d_ndbi²) — no manual labels needed
2. **Prithvi via TerraTorch**: BACKBONE_REGISTRY for clean model loading
3. **Zero-pad bands**: 5→6 bands (SWIR2=zeros) preserves pretrained weights
4. **Features cache**: 15MB parquet replaces all PostGIS queries at training time
5. **Incremental training**: deterministic pair slicing + versioned checkpoints + crash recovery
6. **Spatial train/val split**: by location to prevent data leakage
7. **Test on Brandenburg**: completely held-out region for unbiased evaluation

## MLflow Tracking
View experiment results at: http://3.91.55.220:5000
Model registry: `SiamesePrithviChangeDetection` (v5)
