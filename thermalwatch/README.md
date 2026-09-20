# ThermalWatch — Wildfire Risk & Solar Farm Health Monitoring

> **GeoAI MLOps Project 3** | Geospatial foundation model fine-tuning for environmental monitoring

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-orange)](https://pytorch.org)
[![Prithvi](https://img.shields.io/badge/Prithvi--EO-2.0--300M-green)](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)
[![CI/CD](https://github.com/Ea-mjolnir/Geospatial_MLOps/actions/workflows/thermalwatch-ci.yml/badge.svg)](https://github.com/Ea-mjolnir/Geospatial_MLOps/actions/workflows/thermalwatch-ci.yml)

---

## Overview

ThermalWatch is a multi-modal deep learning platform for:
- **Wildfire risk prediction** — California 2020–2022 (Landsat thermal + Sentinel-2 + ERA5 + OSM)
- **Solar farm health monitoring** — Arizona 2019–2022 (Landsat thermal + NSRDB weather)

---

## Architecture
ThermalWatchBackbone
├── ThermalEncoder ResNet-18 (1-channel) → 512d [11.4M params]
├── OpticalEncoder Prithvi-EO-2.0-300M frozen → 1024d [303.9M params]
├── WeatherEncoder MLP [5] → 128d [~9K params]
├── OSMEncoder MLP [7] → 64d [~3K params]
└── CrossModalAttention (4 modalities × 512d) → 512d [4.2M params]

Total: 320.4M | Trainable (Phase 1): 16.5M

---

## Dataset

| Task | Areas | Years | Patches | Labels |
|------|-------|-------|---------|--------|
| Wildfire | Sierra Nevada, SoCal, NorCal (CA) | 2020–2022 (train) / 2023 (val) | 109,828 | risk_score, alert_level, spread_prob, structure_risk |
| Solar | Phoenix Metro, Sonoran Desert, Tucson (AZ) | 2019–2021 (train) / 2022 (val) | 276,768 | efficiency_score, hotspot_score, degradation_rate, maintenance_flag |

**Data sources:** Landsat 8/9 (USGS), Sentinel-2 (Planetary Computer), ERA5 (Copernicus CDS), NSRDB (NREL), NIFC/VIIRS fire perimeters, OSM Overpass API, EIA solar farm locations

> ⚠️ Raw data not included (48GB). See `scripts/prepare_gdrive.sh` for data pipeline.

---

## Training Results

### Wildfire (val set — NorCal 2023)

| Metric | Value |
|--------|-------|
| Alert Level Accuracy | **75.6%** |
| Alert Level F1 | 0.231 |
| Risk Score RMSE | 0.314 |
| Risk Score R² | -0.974 |
| Spread Prob RMSE | 0.068 |
| Structure Risk RMSE | 0.030 |

### Solar (val set — Arizona 2022)

| Metric | Value |
|--------|-------|
| Efficiency Score R² | **0.264** |
| Efficiency Score RMSE | 0.212 |
| Degradation Rate R² | **0.181** |
| Hotspot Score RMSE | 0.117 |
| Maintenance Accuracy | 95.7% |
| Maintenance F1 | 0.068 |

**Notes:**
- Alert classification works well (75.6% vs 25% random baseline) ✅
- Solar efficiency and degradation show positive R² ✅
- Regression tasks (risk_score, hotspot) are harder due to label sparsity
- Temporal domain shift (train 2020-2022 → val 2023) impacts val metrics

---

## Training Pipeline
Phase 1 (30 epochs): Frozen backbone → train heads only
batch_size=256 | lr_head=1e-3 | ~1hr/epoch on T4 GPU

Phase 2 (25 epochs): Unfreeze ThermalEncoder + fusion
batch_size=128 | lr_backbone=1e-5 | lr_head=1e-4
Early stopping patience=5

Resume: automatic — detects latest checkpoint

---

## Project Structure
hermalwatch/
├── src/
│ ├── data/
│ │ ├── wildfire/wildfire_pipeline.py
│ │ ├── solar/solar_pipeline.py
│ │ ├── patch_extractor.py
│ │ ├── build_index.py
│ │ ├── s2_pipeline.py
│ │ ├── add_weather_features.py
│ │ ├── generate_labels.py
│ │ └── extract_prithvi_embeddings.py
│ └── models/
│ ├── backbone/thermal_backbone.py
│ └── training/
│ ├── ssl_pretrain.py
│ └── finetune.py
├── notebooks/
│ └── thermalwatch_colab.ipynb
├── scripts/
│ └── prepare_gdrive.sh
└── requirements.txt

---

## Quick Start (Google Colab)

1. Upload data to Google Drive following `scripts/prepare_gdrive.sh`
2. Open `notebooks/thermalwatch_colab.ipynb` in Colab
3. Run cells 1–10 in order
4. Checkpoints saved to GDrive automatically

**Requirements:**
- Google Colab with T4 GPU (15GB)
- Google Drive with ~50GB free
- Python 3.11 venv (auto-installed in Cell 2)

---

## Key Dependencies
torch==2.2.0
terratorch>=0.1.0 # Prithvi-EO-2.0-300M via TerraTorch
numpy==1.26.4
rasterio>=1.3.9
scikit-learn>=1.4.0

---

## Lessons Learned

- **SSL pre-training**: Investigated InfoNCE and BYOL — both collapsed on thermally similar patches. Pivoted to direct supervised fine-tuning with ImageNet initialization.
- **Class imbalance**: 71% no-fire patches. WeightedRandomSampler + focal loss caused overfitting; standard MSE + CE with natural distribution generalized better.
- **Temporal split**: Geographic split (area-based) caused domain shift. Time-based split (2020–2022 train / 2023 val) improved val distribution match.
- **OOM prevention**: Unfreezing Prithvi (303M params) caused GPU OOM on T4. Solution: unfreeze ThermalEncoder + fusion layers only (~16.5M trainable).

---

## Related Projects

| Project | Description | Repo |
|---------|-------------|------|
| P1 — Siamese Change Detection | Prithvi-EO Siamese network for land-use change detection | [github.com/Ea-mjolnir/Geospatial_MLOps](https://github.com/Ea-mjolnir/Geospatial_MLOps) |
| P2 — OmniGeoFusion | Multimodal geospatial fusion platform | [github.com/Ea-mjolnir/Geospatial_MLOps](https://github.com/Ea-mjolnir/Geospatial_MLOps) |

---

## Author

**Enoch Asare** — Geospatial Analyst / Cloud Engineer  
M.Sc. Geodesy & Geoinformation, TU Berlin  
[GitHub: Ea-mjolnir](https://github.com/Ea-mjolnir)

---

## Roadmap

| Feature | Status |
|---------|--------|
| Data pipeline | ✅ Complete |
| Model architecture | ✅ Complete |
| SSL pre-training | ✅ Investigated (InfoNCE + BYOL) |
| Supervised fine-tuning | ✅ Complete |
| Evaluation | ✅ Complete |
| Colab training notebook | ✅ Complete |
| FastAPI inference endpoints | ✅ Complete |
| Docker containerization | ✅ Complete |
| GitHub Actions CI/CD | ✅ Complete |
| MLflow experiment tracking | 🔄 In Progress |
| Airflow data pipeline DAGs | 🔄 In Progress |
