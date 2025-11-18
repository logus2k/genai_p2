import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from sklearn.manifold import TSNE
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


class ModelPredictor:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load label encoder
        self.label_encoder, self.categories = self._load_label_encoder()

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # model
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

        # domain color lookup
        self.domain_colors = self._build_domain_color_map()

        # precompute static category embeddings and their t-SNE positions
        (
            self.category_embeddings,
            self.tsne_positions
        ) = self._precompute_category_tsne()

    def _load_label_encoder(self):
        with open("../scibert_label_encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
        return encoder, encoder.classes_

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

    def _build_domain_color_map(self):
        unique_domains = sorted({subject_to_domain(c) for c in self.categories})
        base_colors = [
            0xff5733, 0xff8d1a, 0xf5b041, 0x52be80, 0x2e86c1,
            0x8e44ad, 0x566573, 0x27ae60, 0x1abc9c, 0xcd6155
        ]
        domain_colors = {}
        for i, d in enumerate(unique_domains):
            domain_colors[d] = base_colors[i % len(base_colors)]
        return domain_colors

    def _precompute_category_tsne(self):
        mat = []
        for c in self.categories:
            emb = torch.zeros(self.model.hidden_size)
            mat.append(emb.numpy())

        # Proper embeddings: call classifier for each class
        with torch.no_grad():
            W = self.model.classifier.weight.detach().cpu().numpy()
        mat = W

        tsne = TSNE(n_components=3, perplexity=10, learning_rate='auto', init='random')
        coords = tsne.fit_transform(mat)

        cat_emb = {}
        tsne_pos = {}

        for i, c in enumerate(self.categories):
            cat_emb[c] = mat[i]
            tsne_pos[c] = coords[i].tolist()

        return cat_emb, tsne_pos

    def predict_with_embeddings(self, text: str, top_k: int = 5):
        sample_vec = self._embed_text(text)

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=CONFIG["MAX_LENGTH"],
            return_tensors="pt",
        )
        ids = encoding["input_ids"].to(self.device)
        mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(ids, mask)
            probs = F.softmax(logits, dim=-1)

        top_probs, top_idx = torch.topk(probs[0], top_k)

        preds = []
        for prob, idx in zip(top_probs, top_idx):
            label = self.label_encoder.inverse_transform([idx.item()])[0]
            preds.append({
                "category": label,
                "domain": subject_to_domain(label),
                "confidence": float(prob.item())
            })

        from sklearn.manifold import TSNE
        all_cat_mat = np.vstack(list(self.category_embeddings.values()))
        combined = np.vstack([sample_vec, all_cat_mat])

        tsne = TSNE(n_components=3, perplexity=10, learning_rate='auto', init='pca')
        combined_pos = tsne.fit_transform(combined)

        sample_pos = combined_pos[0].tolist()

        all_cat_pos = {}
        offset = 1
        for i, c in enumerate(self.categories):
            all_cat_pos[c] = combined_pos[i + offset].tolist()

        return {
            "predictions": preds,
            "sample_embedding": sample_vec.tolist(),
            "sample_tsne_pos": sample_pos,
            "all_categories_tsne": all_cat_pos,
            "domain_colors": self.domain_colors
        }
    
    def arxiv_pdf_link_from_doi(self, doi: str):
        arxiv_id = doi.split("arXiv.")[-1]
        pdf_link = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        return arxiv_id, pdf_link

    def get_sample_by_index(self, df, index):
        row = df.iloc[index]
        title = str(row.get("title", ""))
        abstract = str(row.get("abstract", ""))
        text = f"{title} {abstract}".strip()

        actual_label = str(row.get("primary_subject", None))

        doi = row.get("doi", None)
        pdf_url = None
        if isinstance(doi, str) and "arXiv" in doi:
            _, pdf_url = self.arxiv_pdf_link_from_doi(doi)

        return {
            "index": int(index),
            "text": text,
            "actual_label": actual_label,
            "pdf_url": pdf_url
        }
