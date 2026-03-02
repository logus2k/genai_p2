FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# --- System dependencies ---
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3-pip \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN pip install --no-cache-dir --upgrade pip

# --- Python dependencies ---
# Layer order: deps change rarely, code changes often
COPY webv2/requirements.txt /tmp/requirements.txt

# PyTorch must come from the PyTorch cu118 wheel index for CUDA support
RUN pip install --no-cache-dir \
        torch==2.1.0 \
        --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir \
        fastapi==0.104.1 \
        uvicorn==0.24.0 \
        python-socketio==5.8.0 \
        pandas==2.1.3 \
        transformers==4.35.0 \
        scikit-learn==1.3.2 \
        pyarrow==14.0.1 \
        numpy==1.26.4 \
        minisom==2.3.5

# --- HuggingFace: tokenizer + config only (no base weights) ---
# AutoModel.from_config() is used at runtime, so only tokenizer vocab
# and model config are needed here (~2MB, not the 422MB base weights).
ENV HF_HOME=/app/hf_cache
RUN python -c "\
from transformers import AutoTokenizer, AutoConfig; \
print('Downloading SciBERT tokenizer...'); \
AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased'); \
print('Downloading SciBERT config...'); \
AutoConfig.from_pretrained('allenai/scibert_scivocab_uncased'); \
print('Done.')"

WORKDIR /app

# --- Model files ---
COPY scibert_finetuned_model.pt  /app/scibert_finetuned_model.pt
COPY scibert_label_encoder.pkl   /app/scibert_label_encoder.pkl
COPY som_3d_model.pkl            /app/som_3d_model.pkl

# --- Dataset ---
COPY data/ /app/data/

# --- Application code ---
# Copied last so code changes don't invalidate the expensive layers above
COPY webv2/ /app/webv2/

# --- Runtime ---
WORKDIR /app/webv2

# Use only the baked-in HF cache; fail fast if cache is incomplete
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV PYTHONUNBUFFERED=1

EXPOSE 6543
CMD ["python", "main.py"]
