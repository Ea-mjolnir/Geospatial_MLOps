FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install torch CPU only (~200MB instead of 2GB CUDA)
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.0+cpu torchvision==0.17.0+cpu

# Install remaining requirements
RUN pip install --no-cache-dir \
    --use-deprecated=legacy-resolver \
    -r requirements.txt

# Pin numpy + scipy + huggingface_hub
RUN pip install --no-cache-dir \
    numpy==1.26.4 scipy==1.11.4 \
    albumentations==1.3.1 \
    huggingface_hub==0.26.0 \
    transformers==4.40.0 \
    --force-reinstall

# Copy source code
COPY src/ src/
COPY models/ models/

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
