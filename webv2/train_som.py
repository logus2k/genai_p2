# train_som.py
import torch
import pickle
import numpy as np
from minisom import MiniSom
from model_utils import ModelPredictor, CONFIG


def train_som_for_categories(model_path: str, output_path: str = "../som_3d_model.pkl"):
    """
    Train a 3D Self-Organizing Map on category embeddings.
    This only needs to be run once - the trained SOM is saved and reused.
    """
    
    print("Loading model and extracting category embeddings...")
    predictor = ModelPredictor(model_path)
    
    # Get category embeddings from classifier weights
    # Shape: (num_categories, hidden_size) = (172, 768)
    with torch.no_grad():
        category_weights = predictor.model.classifier.weight.detach().cpu().numpy()
    
    num_categories = category_weights.shape[0]
    embedding_dim = category_weights.shape[1]
    
    print(f"Training SOM on {num_categories} categories with {embedding_dim}-dimensional embeddings")
    
    # 3D SOM grid dimensions - adjust these for desired granularity
    # Using 6x6x6 = 216 neurons (slightly more than 172 categories)
    grid_x, grid_y, grid_z = 6, 6, 6
    
    print(f"Creating {grid_x}x{grid_y}x{grid_z} = {grid_x*grid_y*grid_z} neuron 3D SOM grid")
    
    # Initialize SOM
    # Note: minisom supports 2D grids, so we'll flatten to 2D then reconstruct 3D positions
    som = MiniSom(
        x=grid_x * grid_y,  # Flatten Z dimension into rows
        y=grid_z,
        input_len=embedding_dim,
        sigma=1,  # Neighborhood radius
        learning_rate=0.5,
        neighborhood_function='gaussian',
        random_seed=42
    )
    
    print("Initializing SOM weights...")
    som.pca_weights_init(category_weights)
    
    print("Training SOM (this may take a minute)...")
    som.train_batch(category_weights, num_iteration=1000, verbose=True)
    
    print("Training complete!")
    
    # Map each category to its BMU (Best Matching Unit) position
    category_positions_2d = {}
    for i, cat_name in enumerate(predictor.categories):
        bmu = som.winner(category_weights[i])
        category_positions_2d[cat_name] = bmu
    
    # Convert 2D grid positions to 3D coordinates
    category_positions_3d = {}
    for cat_name, (row, col) in category_positions_2d.items():
        # Unflatten: row contains both x and y, col is z
        x = row // grid_y
        y = row % grid_y
        z = col
        category_positions_3d[cat_name] = [float(x), float(y), float(z)]
    
    # Package everything needed for visualization
    som_data = {
        'som': som,
        'grid_dims': (grid_x, grid_y, grid_z),
        'category_positions_3d': category_positions_3d,
        'categories': predictor.categories,
        'embedding_dim': embedding_dim
    }
    
    print(f"Saving SOM model to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(som_data, f)
    
    print("Done! SOM model saved successfully.")
    print(f"\nSample category positions:")
    for i, (cat, pos) in enumerate(list(category_positions_3d.items())[:5]):
        print(f"  {cat}: {pos}")
    
    return som_data


if __name__ == "__main__":
    model_path = "../scibert_finetuned_model.pt"
    train_som_for_categories(model_path)
