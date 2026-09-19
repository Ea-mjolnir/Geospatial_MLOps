# ThermalWatch — FastAPI Inference Server
# =========================================
# Checkpoints mounted at runtime:
#   docker run -v /path/to/checkpoints:/app/checkpoints \
#              -p 8000:8000 thermalwatch

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    libgeos-dev \
    libgdal-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch CPU first
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir \
    fastapi==0.110.0 \
    uvicorn[standard]==0.27.0 \
    pydantic>=2.6.0 \
    numpy==1.26.4 \
    rasterio>=1.3.9 \
    scikit-learn>=1.4.0 \
    terratorch>=0.1.0 \
    transformers==4.40.0 \
    timm \
    einops

# Copy source code only (no checkpoints)
COPY src/ ./src/

# Checkpoint directory — mount at runtime
# docker run -v /path/to/checkpoints:/app/checkpoints
RUN mkdir -p /app/checkpoints

# Environment
ENV PYTHONPATH=/app
ENV MPLBACKEND=agg
ENV PYTHONUNBUFFERED=1
ENV WILDFIRE_CKPT=/app/checkpoints/wildfire_best_phase1.pt
ENV SOLAR_CKPT=/app/checkpoints/solar_best_phase1.pt

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
