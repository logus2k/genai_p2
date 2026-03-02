# SciPredictor — arXiv Paper Classifier

Fine-tune a transformer model to classify arXiv papers by `primary_subject`, with an interactive web application for exploring predictions and embeddings in 3D.

## Overview

This project trains a SciBERT-based sequence classifier on arXiv metadata (title + abstract) and provides a web interface to:

- Predict the subject category of any paper in the dataset
- Visualise the embedding space in 3D (t-SNE or Self-Organizing Map)
- Browse the original PDF side-by-side with the predictions

The classifier supports 172 arXiv subject categories and achieves predictions with per-class confidence scores.

## Dataset

arXiv papers dataset (title, abstract, primary subject):

- HuggingFace: https://huggingface.co/datasets/nick007x/arxiv-papers
- Direct parquet download: https://huggingface.co/datasets/nick007x/arxiv-papers/resolve/main/train.parquet

## Model

Base model: [allenai/scibert_scivocab_uncased](https://huggingface.co/allenai/scibert_scivocab_uncased)

Architecture (`EnhancedClassifier`):
- SciBERT backbone (768-D hidden size)
- LayerNorm → Linear pre-classifier → GELU → Dropout (0.3)
- Linear classification head (172 classes)

Pre-trained weights are available on Docker Hub (see below).

## Running with Docker

The easiest way to run SciPredictor is via the pre-built Docker image, which includes the fine-tuned model weights, dataset, and all dependencies.

**Prerequisites:** Docker and Docker Compose installed.

```bash
# Pull and run (CPU)
docker compose up -d

# Or pull the image directly
docker pull <your-username>/scipredictor
docker run -p 6543:6543 <your-username>/scipredictor
```

Then open http://localhost:6543 in your browser.

### GPU support

If you have an NVIDIA GPU and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed, keep `docker-compose.override.yml` alongside `docker-compose.yml` and run:

```bash
docker compose up -d
```

On CPU-only hosts, remove or rename `docker-compose.override.yml` first.

### Building the image locally

```bash
# Place the following files in the project root before building:
#   scibert_finetuned_model.pt   (fine-tuned weights)
#   scibert_label_encoder.pkl
#   som_3d_model.pkl
#   data/arxiv_papers/train.parquet

docker compose build
docker compose up -d
docker compose logs -f scipredictor
```

## Running without Docker

**Requirements:** Python 3.10+, CUDA recommended.

```bash
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r webv2/requirements.txt
```

Download the dataset parquet file to `data/arxiv_papers/train.parquet`, then:

```bash
cd webv2
python main.py
```

Server starts on http://localhost:6543.

## Training

To reproduce the fine-tuned model from scratch, refer to the training notebook. The project also includes experiments with Longformer and LoRA (PEFT).

To (re)train the SOM visualisation model after fine-tuning:

```bash
cd webv2
python train_som.py
```

This produces `som_3d_model.pkl` in the project root.

## Project Structure

```
├── webv2/                      # SciPredictor web application
│   ├── main.py                 # FastAPI + Socket.IO server (port 6543)
│   ├── model_utils.py          # SciBERT inference pipeline
│   ├── train_som.py            # One-time SOM training script
│   ├── requirements.txt
│   └── frontend/               # Vanilla JS + Three.js client
├── data/
│   └── arxiv_papers/
│       └── train.parquet       # arXiv dataset (1.6 GB)
├── scibert_finetuned_model.pt  # Fine-tuned model weights (423 MB)
├── scibert_label_encoder.pkl   # Label encoder (172 classes)
├── som_3d_model.pkl            # Pre-trained SOM model
├── Dockerfile
├── docker-compose.yml
└── docker-compose.override.yml # GPU support (optional)
```

## License

Apache 2.0 — see [LICENSE.md](LICENSE.md).
