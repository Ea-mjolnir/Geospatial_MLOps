# MLOps Portfolio

A collection of end-to-end MLOps projects spanning geospatial AI,
remote sensing, and production ML pipelines.

## Projects

### 1. [Siamese Prithvi Change Detection](./siamese-prithvi-change-detection/)
A production-grade geospatial AI pipeline for detecting land cover change
using Sentinel-2 satellite imagery and the Prithvi-EO-1.0-100M foundation model.

**Key highlights:**
- Weakly supervised training — no manual labels needed
- 86M parameter Siamese network with Prithvi backbone
- Incremental training across 6 chunks (138,948 pairs)
- Final Test R2: 0.9849 | Spearman: 0.9942
- Munich inference on completely unseen data

**Stack:** PyTorch · TerraTorch · Prithvi-EO · Sentinel-2 · MLflow · AWS · Colab

---
*More projects coming soon.*
