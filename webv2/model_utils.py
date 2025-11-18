import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle

CONFIG = {
    "MAX_LENGTH": 512,
    "DROPOUT_RATE": 0.3
}


class EnhancedClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout_rate=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.pre_classifier = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.activation = nn.GELU()

    def forward(self, input_ids, attention_mask, return_embeddings=False):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            hidden = outputs.pooler_output
        else:
            hidden = outputs.last_hidden_state[:, 0]

        hidden = self.layer_norm(hidden)
        hidden = self.pre_classifier(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)

        if return_embeddings:
            return hidden

        logits = self.classifier(hidden)
        return logits


def subject_to_domain(category: str) -> str:
    if "(" in category:
        prefix = category.split("(")[0].strip()
        return prefix.split()[0]
    return "other"


def arxiv_pdf_link_from_doi(doi: str):
    arxiv_id = doi.split("arXiv.")[-1]
    return arxiv_id, f"https://arxiv.org/pdf/{arxiv_id}.pdf"


class ModelPredictor:

    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_encoder, self.categories = self._load_label_encoder()

        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = EnhancedClassifier(
            model_name="allenai/scibert_scivocab_uncased",
            num_classes=len(self.categories),
            dropout_rate=CONFIG["DROPOUT_RATE"]
        ).to(self.device)

        state = torch.load(model_path, map_location=self.device)
        if list(state.keys())[0].startswith("module."):
            state = {k.replace("module.", ""): v for k, v in state.items()}

        self.model.load_state_dict(state)
        self.model.eval()

        self.encoder = LabelEncoder()
        self.encoder.fit(self.categories)

        self.category_embeddings = self._compute_category_embeddings()

    def _load_label_encoder(self):
        with open("../scibert_label_encoder.pkl", "rb") as f:
            enc = pickle.load(f)
        return enc, enc.classes_

    def _embed_text(self, text: str):
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=CONFIG["MAX_LENGTH"],
            return_tensors="pt",
        )
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            emb = self.model(ids, mask, return_embeddings=True)
        return emb[0].cpu().numpy()

    def _compute_category_embeddings(self):
        emb_map = {}
        for cat in self.categories:
            emb_map[cat] = self._embed_text(cat).tolist()
        return emb_map

    def predict_with_embeddings(self, text: str, top_k: int = 5):
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=CONFIG["MAX_LENGTH"],
            return_tensors="pt",
        )
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(ids, mask)
            sample_emb = self.model(ids, mask, return_embeddings=True)
            probs = F.softmax(logits, dim=-1)

        top_probs, top_idx = torch.topk(probs[0], top_k)
        preds = []

        sample_vec = sample_emb[0].cpu().numpy().tolist()

        for prob, idx in zip(top_probs, top_idx):
            label = self.encoder.inverse_transform([idx.item()])[0]
            preds.append({
                "category": label,
                "domain": subject_to_domain(label),
                "confidence": float(prob.item())
            })

        return {
            "predictions": preds,
            "sample_embedding": sample_vec,
            "category_embeddings": self.category_embeddings
        }

    def get_sample_by_index(self, df, index):
        row = df.iloc[index]
        title = str(row.get("title", ""))
        abstract = str(row.get("abstract", ""))
        text = f"{title} {abstract}".strip()

        actual_label = str(row.get("primary_subject", None))

        doi = row.get("doi", None)
        pdf_url = None
        if isinstance(doi, str) and "arXiv" in doi:
            _, pdf_url = arxiv_pdf_link_from_doi(doi)

        return {
            "index": int(index),
            "text": text,
            "actual_label": actual_label,
            "pdf_url": pdf_url
        }
