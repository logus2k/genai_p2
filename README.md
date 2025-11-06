# Generative AI — Project 2

Fine-tune Longformer to classify arXiv papers by `primary_subject`.

## Overview

This project trains a sequence-classification model on arXiv metadata (title + abstract) and evaluates accuracy, weighted/macro F1, and top-k accuracy.

It supports:

- Full fine-tuning (Longformer / LED)
- Parameter-Efficient Fine-Tuning with LoRA (PEFT)
- Balanced sampling per subject (cap **K** per class)
- Robust padding collator (handles `global_attention_mask`)

## Requirements

- Python 3.10+
- PyTorch (CUDA recommended)
- `transformers`, `datasets`, `accelerate`, `scikit-learn`, `pandas`, `numpy`, `peft`

```bash
pip install torch transformers datasets accelerate scikit-learn pandas numpy peft
```
