# GeoFoundationHub

**Universal Geospatial Foundation Model Registry & Serving Platform**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

GeoFoundationHub is a production-grade serving platform for geospatial foundation models. It provides a unified REST API for running inference across multiple foundation models with advanced serving patterns including A/B testing, shadow deployment, canary rollouts, and intelligent multi-tenant routing.

---

## Models

| Model | Params | Embedding | Sensor | Source |
|-------|--------|-----------|--------|--------|
| [Prithvi-EO-2.0-300M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M) | 300M | 1024d | Multispectral/Thermal | IBM/NASA |
| [SatMAE](https://arxiv.org/abs/2207.08051) | 307M | 1024d | Multispectral/SAR | Stanford |
| [RemoteCLIP](https://arxiv.org/abs/2306.11029) | 151M | 768d | Optical RGB | CLIP-RS |

---

## Architecture
                ┌─────────────────────┐
                │    Client / Tenant   │
                │   (API Key Auth)     │
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────┐
                │    FastAPI Gateway   │
                │   /infer /registry  │
                └──────────┬──────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
 ┌────────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
 │ Intelligent   │ │ A/B + Shadow│ │   Canary    │
 │   Router      │ │ Deployment  │ │   Rollout   │
 └────────┬──────┘ └──────┬──────┘ └──────┬──────┘
          │                │                │
 ┌────────▼────────────────▼────────────────▼──────┐
 │                  Model Registry                  │
 │   Prithvi-EO-2.0-300M │ SatMAE │ RemoteCLIP    │
 └──────────────────────────────────────────────────┘

---

## Features

### Intelligent Routing
Routes requests to the best model based on input data characteristics — no model selection needed from the user:

| Input | Best Model |
|-------|-----------|
| 6-band multispectral | Prithvi-EO-2.0-300M |
| Thermal imagery | Prithvi-EO-2.0-300M |
| RGB optical | RemoteCLIP |
| 3-band multispectral | SatMAE |
| SAR data | SatMAE |

### Advanced Serving Patterns
- **A/B Testing** — Split traffic between model versions to compare performance
- **Shadow Deployment** — Run challenger model silently without affecting users
- **Canary Rollout** — Gradually shift traffic to new model (e.g. 5% → 20% → 100%)
- **Multi-tenant Routing** — Different organizations get intelligent routing per their data

### Multi-tenant API
| Tenant | API Key | Limit |
|--------|---------|-------|
| California Fire Department | `fire-agency-key-2026` | 5,000/day |
| NASA Earth Science Division | `nasa-earth-key-2026` | 10,000/day |
| AgroCrop Analytics | `agri-co-key-2026` | 2,000/day |
| Public API | `public-key-2026` | 100/day |

---

## API Endpoints
GET /health → Health check + registry stats
GET /registry/ → List all registered models
GET /registry/summary → Registry summary
GET /registry/{model_id} → Model details
POST /registry/ → Register new model
PATCH /registry/{model_id} → Update model status
POST /infer/ → Intelligent inference
POST /infer/compare → Compare all models on same input
POST /routing/ab → Configure A/B test
POST /routing/shadow → Configure shadow deployment
POST /routing/canary → Configure canary rollout
POST /routing/tenant → Add tenant route
GET /routing/stats → Routing statistics
GET /tenants/ → List tenants
POST /tenants/ → Create tenant
POST /tenants/{id}/model → Assign model to tenant

---

## Quick Start

### Inference Request
```bash
curl -X POST http://localhost:8001/infer/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: fire-agency-key-2026" \
  -d '{
    "image": [...],
    "channels": 6,
    "height": 224,
    "width": 224,
    "sensor_type": "multispectral"
  }'
```

### Response
```json
{
  "output": {
    "model_name": "Prithvi-EO-2.0-300M",
    "model_version": "2.0.0",
    "embedding": [...],
    "latency_ms": 86.5
  },
  "routing": {
    "pattern": "intelligent_tenant",
    "model_id": "abc123",
    "tenant_id": "fire-agency"
  }
}
```

---

## Project Structure
geofoundationhub/
├── src/
│ ├── registry/ → Model registration + metadata
│ ├── serving/ → Model loader (startup)
│ ├── routing/ → Router + intelligent routing
│ ├── api/ → FastAPI gateway + routers
│ ├── monitoring/ → Metrics stub
│ └── auth/ → Multi-tenant API key management
├── models/
│ ├── base.py → BaseModelAdapter (abstract)
│ ├── prithvi/ → Prithvi-EO-2.0-300M adapter
│ ├── satmae/ → SatMAE adapter
│ └── remoteclip/ → RemoteCLIP adapter
├── notebooks/
│ └── geofoundationhub_colab.ipynb → Full Colab demo
├── requirements.txt
├── requirements_colab.txt
├── Dockerfile
└── docker-compose.yml

---

## Test Results (Google Colab T4 GPU)
Real Sentinel-2 Inference (California Sierra Nevada, Aug 2021):
Prithvi-EO-2.0-300M: 1024d embeddings | ~86ms latency
SatMAE: 1024d embeddings | ~80ms latency
RemoteCLIP: 768d embeddings | ~20ms latency

Cross-patch similarity (3 Sierra Nevada scenes):
Prithvi: 0.93-0.97 (consistent regional representation)
SatMAE: 0.73-0.84 (sensitive to spectral differences)
RemoteCLIP: 0.84-0.94 (balanced)

Serving Patterns:
✅ Intelligent routing: 5/5 correct
✅ Multi-tenant routing: 7/7 correct
✅ A/B testing: 50/50 split verified
✅ Shadow deployment: Champion served, challenger silent
✅ Canary rollout: 80/20 traffic split verified
✅ Model comparison: All 3 models on real patches

---

## Part of GeoAI MLOps Portfolio

| Project | Description |
|---------|-------------|
| P1 — Siamese Change Detection | Prithvi-based change detection |
| P2 — OmniGeoFusion | Multimodal fusion platform |
| P3 — ThermalWatch | Wildfire + solar monitoring |
| **P4 — GeoFoundationHub** | **Foundation model serving platform** |

---

## References

- [Prithvi-EO-2.0](https://arxiv.org/abs/2310.18660) — IBM/NASA
- [SatMAE](https://arxiv.org/abs/2207.08051) — Stanford
- [RemoteCLIP](https://arxiv.org/abs/2306.11029) — CLIP-RS
- [TerraTorch](https://github.com/IBM/terratorch) — IBM
