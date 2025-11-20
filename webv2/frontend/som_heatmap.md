# SOM Heatmap Visualization

---

## Overview

The SOM heatmap is a third visualization mode that complements t-SNE and the SOM point cloud. It displays category density across the 6×6×6 Self-Organizing Map grid, providing insights into how categories are distributed in the learned topology.

## What the Heatmap Shows

- **Grid Structure**: A 3D representation of all 216 neurons (6×6×6) in your trained SOM
- **Color Intensity**: Blue (low density) → Yellow (medium) → Red (high density)
- **Box Size**: Larger boxes = more categories at that neuron position
- **Numbers**: For neurons with 2-3 categories, the count is displayed
- **Red Sphere**: Current sample position (its best matching unit)

## Three Visualization Modes

Click the button in the top bar to cycle through:

1. **t-SNE**: Force-directed graph showing semantic similarity
   - Best for: Exploring intuitive relationships
   - Recomputed per prediction (accurate but slower)

2. **SOM Grid**: 3D point cloud of categories on fixed grid
   - Best for: Comparing to heatmap structure
   - Fixed positions (consistent reference)

3. **SOM Heat** (NEW): Category density heatmap
   - Best for: Understanding SOM organization and concentration areas
   - Shows where categories cluster naturally

## How to Use

### Navigation
- **Mouse Drag**: Rotate the 3D grid
- **Mouse Wheel**: Zoom in/out (on some browsers)

### Interpreting the Heatmap

**High-density regions (red):**
- Many categories share similar embeddings
- These are "attractor" regions of the feature space
- Samples predicting categories here will have lower confidence spread

**Low-density regions (blue):**
- Few or no categories
- Sparse feature space
- Good regions for discriminating between dissimilar categories

**Sample position (red sphere):**
- Where your current sample maps in the SOM
- Its nearest categories are in the surrounding neurons
- Useful for understanding which category region "won"

## Value Over Other Visualizations

### vs. t-SNE
- **Deterministic**: Same position every time (t-SNE varies with random seed)
- **Shows global structure**: Density patterns reveal true topology
- **Faster inference**: BMU lookup vs. iterative fitting
- **Interpretable**: Grid coordinates are stable references

### vs. SOM Point Cloud
- **Density context**: See concentration, not just positions
- **Global view**: All neurons visible simultaneously
- **Pattern discovery**: Identify empty regions and hotspots
- **Clustering insight**: Visually clear which regions dominate

## Technical Details

### Grid Organization
- **6×6×6 neurons** = 216 total positions
- Trained on 768-dimensional SciBERT embeddings
- 172 arXiv subject categories mapped to neurons
- Some neurons are empty (sparse coverage)

### Color Mapping
```
Density 0%   : Blue    (#4a90e2)
Density 50%  : Yellow  (#f5d547)
Density 100% : Red     (#ff4444)
```

### Box Scaling
Base size: 0.12 units
Scales with occupancy: 0.6x to 1.2x depending on category count

## Example Insights

**Physics papers** (many categories, physics.* prefix)
→ Would likely form a dense red region on the SOM

**Rare domain papers** (few categories, niche subjects)
→ Would be scattered blue neurons

**Sample prediction**:
- If sample lands in red region → confident in category choice
- If sample in blue region → model less certain, may spread confidence across domains

## Implementation Notes

The heatmap uses Three.js with:
- Phong lighting for 3D depth perception
- Box geometries for neuron representation
- Sprite labels for text rendering
- Mouse drag controls for 3D rotation

All rendering happens in `som_heatmap.js` and integrates with your existing FastAPI socket system.

## Future Enhancements

1. **Time slider**: Show density changes during SOM training
2. **Domain coloring**: Color neurons by dominant domain instead of density
3. **Hover tooltips**: Show category names when hovering over neurons
4. **Heatmap legend**: Visual colormap reference on screen
5. **Export**: Save heatmap as 2D image or full 3D model

---
