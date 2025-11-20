# SciPredictor Guide

This document explains the three visualization modes in SciPredictor and what each one represents.

---

## Overview

SciPredictor uses three complementary 3D visualizations to explore how arXiv paper categories are organized in the model's learned embedding space and how new samples are classified:

1. **t-SNE** — semantic similarity (force-directed graph)
2. **SOM Grid** — self-organizing map (topologically organized grid)
3. **SOM Heat** — category density heatmap (concentration visualization)

---

## View 1: t-SNE

### What It Shows

A force-directed 3D embedding space where categories and the current sample are positioned based on **semantic similarity**. Categories that are close together are semantically similar according to the model's SciBERT embeddings.

### How It Works

1. The model embeds all 148 arXiv categories and your sample into 768-dimensional feature vectors (extracted from SciBERT's classifier weights)
2. t-SNE (t-Distributed Stochastic Neighbor Embedding) projects these high-dimensional vectors into 3D space
3. The projection preserves local neighborhood structure: nearby points in high-D space stay nearby in 3D
4. Each render recomputes t-SNE from scratch, so positions may vary slightly between predictions

### Visual Elements

- **Red sphere** = your current sample
- **Colored spheres** = all 148 arXiv categories, colored by domain prefix (cs, math, physics, etc.)
- **Blue spheres** = top 5 predicted categories (your model's best guesses)
- **Black lines** = connections from sample to each of the top 5 predictions

### What to Look For

- **Sample surrounded by blue spheres** = model is confident; similar categories nearby
- **Sample far from all blues** = model struggled; no clear similar category
- **Clustered colored regions** = related domains group naturally (e.g., all cs.* categories together)
- **Sparse regions** = rare or unique category spaces (e.g., specialized physics subcategories)

### Strengths

- Visually intuitive: proximity = semantic similarity
- Good for exploring relationships between categories
- Shows why the model made its predictions (nearest neighbors)

### Weaknesses

- Non-deterministic: re-rendering gives different layouts
- Computationally expensive (recomputed per prediction)
- Doesn't preserve global structure—far-apart clusters might actually be very dissimilar

---

## View 2: SOM Grid

### What It Shows

A **fixed 6×6×6 Self-Organizing Map** where categories are pinned to stable grid positions. Unlike t-SNE, the grid never changes—same sample always lands in the same neuron.

### How It Works

1. A SOM is a neural network with 216 neurons (6 × 6 × 6 grid positions)
2. During training, the SOM learns to organize categories in a topologically-preserving way
3. Categories that are semantically similar end up in nearby neurons
4. When you predict on a new sample, its embedding is matched to the closest neuron (Best Matching Unit or BMU)
5. The sample position is determined by this BMU lookup—always the same for identical inputs

### Visual Elements

- **Red sphere** = your sample's best matching neuron
- **Colored spheres** = all categories pinned to their learned grid positions
- **Blue spheres** = top 5 predicted categories
- **Black lines** = connections from sample to top 5 predictions
- **Grid structure** = fixed 6×6×6 arrangement with visible lattice pattern

### What to Look For

- **Sample in cluster of blue spheres** = sample landed in the right region
- **Sample isolated (no blues nearby)** = model uncertain; far from any strong category match
- **Dense regions** = popular categories that attracted neurons during training
- **Empty neurons** = unused positions in the SOM grid

### Strengths

- **Deterministic**: same sample always lands in same position
- **Fast**: BMU lookup is O(n) vs t-SNE's iterative fitting
- **Consistent reference**: grid positions are stable landmarks
- **Reveals organization**: shows how SOM decided to arrange categories

### Weaknesses

- Less intuitive than t-SNE (grid structure less obviously meaningful)
- Fixed grid may be suboptimal for some category distributions
- Requires pre-training (train_som.py) to generate the model

---

## View 3: SOM Heat

### What It Shows

A **3D heatmap of the SOM grid** where color and box size indicate how many categories occupy each neuron. Shows concentration and density patterns without individual category labels.

### How It Works

1. For each of the 216 neurons in the 6×6×6 grid:
   - Count how many categories are mapped to that position
   - Color code by density: blue (0-1 categories) → yellow (medium) → red (many)
   - Scale box size with occupancy
2. Categories are organized in 3D layers (Z-axis shows depth)
3. Numbers display exact count for populated neurons
4. Sample position shown as a larger red sphere

### Visual Elements

- **Blue boxes** = empty or sparse neurons (0-1 categories)
- **Yellow boxes** = medium density (2-5 categories)
- **Red boxes** = high density (6+ categories, color hotspots)
- **Box size** = scales with occupancy (larger = more categories)
- **Numbers** = exact count displayed on each non-empty neuron
- **Red sphere** = current sample's BMU position
- **45° rotation** = default isometric view showing all three grid dimensions

### What to Look For

- **Red hotspots** = "attractor" regions where many categories cluster
  - Predictions here tend to have high confidence (many similar categories)
  - Model found natural groupings
- **Blue wastelands** = unused regions
  - Sparse feature space
  - Good for discriminating between very different categories
- **Sample in red region** = confident prediction zone
- **Sample in blue region** = uncertain prediction zone
- **3D patterns** = overall organization strategy (e.g., dense on one side, sparse on another)

### Strengths

- **Shows global structure** at a glance (density patterns across the entire map)
- **Fast to render** (boxes instead of individual category spheres)
- **Reveals SOM organization quality** (are categories well-distributed?)
- **Best for understanding category concentration**

### Weaknesses

- Loses individual category identity (only shows counts)
- Less useful for finding specific categories
- Requires mouse rotation to see full 3D structure

---

## How the Three Views Complement Each Other

| Aspect | t-SNE | SOM Grid | SOM Heat |
|--------|-------|----------|----------|
| **Semantic clarity** | High | Medium | Low |
| **Consistency** | Low (re-renders) | High (fixed) | High (fixed) |
| **Speed** | Slow | Fast | Fast |
| **Best for** | Exploring relationships | Navigation & reference | Understanding density |
| **Individual categories** | Visible, labeled | Visible, labeled | Counted, not identified |
| **Global structure** | Approximate | Topological | Density-based |

### Typical Workflow

1. **Start with t-SNE** — understand why specific categories were predicted (visual intuition)
2. **Switch to SOM Grid** — verify the predictions are stable (same sample = same position)
3. **Check SOM Heat** — understand if sample landed in confident or uncertain region

---

## Interpreting Predictions

### When the Actual Label Matches

If your sample's actual category appears in the top 5 predictions:
- ✓ **MATCH** indicator shows in green
- Green-bordered prediction item highlights the match
- The model correctly identified the category

### When the Actual Label is Not in Top 5

A red warning box appears: `Actual label "[category]" not in top 5 predictions`
- Model either misclassified or was uncertain
- In t-SNE/SOM Grid, the sample may be far from the actual category
- In SOM Heat, sample may have landed in sparse (low-confidence) region

---

## Technical Details

### Embeddings

- **Dimension**: 768-D (SciBERT hidden layer size)
- **Source**: Classifier weights from fine-tuned SciBERT model
- **Categories**: 148 unique arXiv subject classifications

### t-SNE Parameters

- **Method**: t-Distributed Stochastic Neighbor Embedding
- **Components**: 3D (recomputed per prediction)
- **Perplexity**: 10
- **Learning rate**: auto (adaptive)
- **Initialization**: PCA

### SOM Parameters

- **Grid**: 6 × 6 × 6 = 216 neurons
- **Input dimension**: 768-D
- **Training**: 1000 iterations on all 148 categories
- **Neighborhood**: Gaussian function with σ=1
- **Learning rate**: 0.5 (annealing during training)

### Data Flow

```
Input Text (Title + Abstract)
         ↓
    SciBERT Tokenizer
         ↓
    SciBERT Model
         ↓
768-D Embedding Vector
         ↓
    ├─→ t-SNE Projection → t-SNE View
    ├─→ BMU Lookup (SOM) → SOM Grid View
    └─→ BMU Lookup (SOM) → SOM Heat View
         ↓
Classifier Linear Layer
         ↓
Softmax Probabilities
         ↓
Top 5 Predictions
```

---

## Tips for Exploration

### Navigate the 3D Views

- **Mouse drag** = rotate view
- **Mouse wheel** = zoom in/out
- **Click and drag** = pan (in some browsers)

### Comparing Predictions

- Use the index field to jump to specific samples (0-indexed)
- Use "Random" to explore the dataset
- Each new prediction re-renders all three views with current data

### Understanding Errors

1. Load a sample with an incorrect prediction
2. Switch between views to see:
   - In t-SNE: where did the sample land relative to the actual category?
   - In SOM Grid: is the sample far from the correct category's neuron?
   - In SOM Heat: did the sample land in a sparse (uncertain) region?

### Finding Category Regions

- t-SNE is best for discovering where categories group naturally
- SOM Grid provides stable reference points for repeated navigation
- SOM Heat shows which regions of the grid are "active"

---

## Troubleshooting

### "SOM visualization not available"

- The SOM model hasn't been trained yet
- Run `train_som.py` to generate `som_3d_model.pkl`
- Make sure the file is in the correct path (`../som_3d_model.pkl`)

### t-SNE layout looks very different each prediction

- This is normal—t-SNE is non-deterministic
- Different random initializations produce different layouts
- The semantic relationships are preserved, just arranged differently

### Heatmap numbers hard to read

- Zoom in with mouse wheel
- Rotate the view (mouse drag) to get a better angle
- Numbers are only shown for non-empty neurons

### Sample position seems wrong

- If using t-SNE: this is expected—layout changes per render
- If using SOM Grid: position should be consistent for same sample
- Check that you're using the same model weights (model_path in config)

---

## Further Reading

- **t-SNE**: [Visualizing Data using t-SNE](https://jmlr.org/papers/v9/vandermaaten08a.html)
- **SOM**: [Self-Organizing Maps](http://www.scholarpedia.org/article/Self-organizing_maps)
- **SciBERT**: [SciBERT: A Pretrained Language Model for Scientific Text](https://arxiv.org/abs/1903.10676)

---
