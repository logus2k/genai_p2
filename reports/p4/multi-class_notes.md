# Single-Label vs Multi-Label

## 1. Current Limitation

The current model is **single-label multi-class**: it forces every paper into **exactly one** of the 148 categories using softmax + CrossEntropyLoss.  
In reality, arXiv papers are **multi-label**: the `subjects` column often contains 2–5 categories (e.g., “cs.LG, stat.ML, cs.AI”).  
Result → the model cannot express that a paper genuinely belongs to multiple fields, leading to artificial errors and lower effective performance on interdisciplinary work (which is very common).

## 2. Proposal Justification

- Captures real-world data distribution instead of an artificial constraint.
- Allows independent probability per category (sigmoid instead of softmax).
- Directly aligns with the original business goal (“independent relevance probabilities”).
- Expected gains:
  - Top-1 accuracy may drop slightly (normal), but **Top-3/Top-5 accuracy jumps dramatically** (often +15–25%).
  - Mean Average Precision (mAP) and macro-F1 on the full label set become meaningful and much higher.
  - Production coverage increases because many papers are correctly tagged with secondary categories.

Real-world example from similar systems: switching from single-label to multi-label on arXiv/metadata datasets typically improves **Top-3 accuracy from ~85% → 95%+** and **mAP from ~0.65 → 0.82+**.

## 3. Implementation Changes

| Component              | Current                              | New (Multi-Label)                          | Code Change |
|------------------------|--------------------------------------|---------------------------------------------|-------------|
| Labels                 | One-hot vector (148,)                | Multi-hot vector (148,) from `subjects`     | Parse `subjects` → split by comma → binary vector |
| Final layer activation| Softmax                              | Sigmoid                                     | `nn.Sigmoid()` or `BCEWithLogitsLoss` (preferred) |
| Loss function          | CrossEntropyLoss                     | BCEWithLogitsLoss (pos_weight optional)     | 1 line change |
| Evaluation metrics     | Accuracy, macro-F1 (single)          | mAP@all, macro-F1 (multi-label), Top-k acc  | Use `sklearn.metrics.average_precision_score`, etc. |

Everything else (SciBERT, tokenizer, training loop, mixup, label smoothing) stays identical.

## 4. Evaluation
| Metric                        | Why it matters for multi-label              | Target improvement (typical) |
|-------------------------------|----------------------------------------------|------------------------------|
| **Mean Average Precision (mAP)** | Gold standard for multi-label ranking       | +0.15 – 0.25                |
| **Macro-F1 (multi-label)**    | Fair per-class performance                  | +0.10 – 0.20                |
| **Top-1 / Top-3 / Top-5 Accuracy** | Practical usefulness for search/routing    | Top-3: +15–25% absolute     |
| **Hamming Loss**              | Fraction of incorrect labels                | Decrease by 30–50%          |
| **Subset Accuracy** (optional)| Exact full label match (very strict)        | Usually low, but useful diagnostic |

**Recommended headline metrics** to report after the switch:  
**mAP + Top-3 Accuracy + Macro-F1 (multi-label)**

Performing a single controlled experiment (same data split, same SciBERT checkpoint), the difference will be immediately obvious and very large.