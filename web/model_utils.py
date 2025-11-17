# model_utils.py

import torch
import pandas as pd
from transformers import AutoTokenizer
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import json


class ModelPredictor:
    def __init__(self, model_path, categories_file=None):
        # Load your model, tokenizer, and label encoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model (adjust based on your model architecture)
        self.model = torch.load(model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer (adjust based on your model)
        self.tokenizer = AutoTokenizer.from_pretrained("your_tokenizer_path")  # Replace with your tokenizer
        
        # Load label encoder (you may need to save this separately)
        # self.label_encoder = joblib.load("label_encoder.pkl")  # If saved separately
        
        # Load categories
        if categories_file:
            with open(categories_file, 'r') as f:
                self.categories = [line.strip() for line in f if line.strip()]
        else:
            # Define your categories list here
            self.categories = [
                "Computer Vision and Pattern Recognition (cs.CV)",
                "Quantum Physics (quant-ph)",
                # ... include all your categories
            ]
        
        # Create label encoder from categories
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.categories)
    
    def predict_with_confidence(self, text, top_k=3):
        """Predict category with confidence scores for new text"""
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,  # Adjust based on your model
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)  # Adjust based on your model's forward method
            probs = F.softmax(logits, dim=-1)
            
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs[0], top_k)
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            category = self.label_encoder.inverse_transform([idx.cpu().item()])[0]
            # Extract domain (first part before parenthesis)
            domain = category.split('(')[0].strip().split()[0] if '(' in category else 'other'
            results.append({
                'category': category,
                'domain': domain,
                'confidence': prob.cpu().item()
            })
        
        return results
    
    def get_sample_by_index(self, df, index):
        """Get a specific sample from the dataframe"""
        if index < 0 or index >= len(df):
            return None
        
        row = df.iloc[index]
        # Adjust column names based on your dataset structure
        text = f"\n{row.get('title', '')}\n{row.get('abstract', '')}"
        actual_label = row.get('primary_subject', 'Unknown')
        
        return {
            'index': index,
            'text': text,
            'actual_label': actual_label,
            'metadata': {col: row[col] for col in df.columns if col not in ['title', 'abstract', 'primary_subject']}
        }
