import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle


CONFIG = {
	'MAX_LENGTH': 512,
	'DROPOUT_RATE': 0.3
}


# ------------------------------------------------------------
#  EnhancedClassifier (full architecture, exactly as trained)
# ------------------------------------------------------------
class EnhancedClassifier(nn.Module):
	def __init__(self, model_name, num_classes, dropout_rate=0.3, use_local=False):
		super().__init__()

		# Load SciBERT base model
		if use_local:
			self.bert = AutoModel.from_pretrained(model_name, local_files_only=True)
		else:
			self.bert = AutoModel.from_pretrained(model_name)

		self.config = self.bert.config
		self.hidden_size = self.config.hidden_size

		# Enhanced head
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

		# Prefer pooler output when available
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


# ------------------------------------------------------------
# Utility — Extract domain from label
# ------------------------------------------------------------
def subject_to_domain(category: str) -> str:
	"""
	Extracts a generic domain (first token before space or parentheses).
	"""
	if "(" in category:
		prefix = category.split("(")[0].strip()
		return prefix.split()[0]
	return "other"


# ------------------------------------------------------------
# Utility — DOI → arXiv PDF link
# ------------------------------------------------------------
def arxiv_pdf_link_from_doi(doi: str):
	arxiv_id = doi.split("arXiv.")[-1]
	pdf_link = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
	return arxiv_id, pdf_link


# ------------------------------------------------------------
# Main predictor wrapper
# ------------------------------------------------------------
class ModelPredictor:
	def __init__(self, model_path: str):

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# Load label encoder + classes
		self.label_encoder, self.categories = self._load_label_encoder()

		# Label encoder
		self.label_encoder = LabelEncoder()
		self.label_encoder.fit(self.categories)

		# Tokenizer (SciBERT)
		self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
		if self.tokenizer.pad_token is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token

		# Build model
		self.model = EnhancedClassifier(
			model_name="allenai/scibert_scivocab_uncased",
			num_classes=len(self.categories),
			dropout_rate=CONFIG["DROPOUT_RATE"],
			use_local=False,
		).to(self.device)

		# Load checkpoint
		state = torch.load(model_path, map_location=self.device)

		# Strip "module." if saved with DataParallel
		if list(state.keys())[0].startswith("module."):
			state = {k.replace("module.", ""): v for k, v in state.items()}

		self.model.load_state_dict(state)
		self.model.eval()

		print(f"✓ Model loaded on {self.device}")

	# ------------------------------------------------------------
	def predict_with_confidence(self, text: str, top_k: int = 5):
		encoding = self.tokenizer(
			text,
			truncation=True,
			padding="max_length",
			max_length=CONFIG["MAX_LENGTH"],
			return_tensors="pt",
		)

		input_ids = encoding["input_ids"].to(self.device)
		attention_mask = encoding["attention_mask"].to(self.device)

		with torch.no_grad():
			logits = self.model(input_ids, attention_mask)
			probs = F.softmax(logits, dim=-1)

		top_probs, top_idx = torch.topk(probs[0], top_k)

		results = []
		for prob, idx in zip(top_probs, top_idx):
			label = self.label_encoder.inverse_transform([idx.item()])[0]
			results.append(
				{
					"category": label,
					"domain": subject_to_domain(label),
					"confidence": float(prob.item()),
				}
			)

		return results

	# ------------------------------------------------------------
	def get_sample_by_index(self, df, index):
		row = df.iloc[index]

		# Build text
		title = str(row.get("title", ""))
		abstract = str(row.get("abstract", ""))
		text = f"{title} {abstract}".strip()

		# Label
		actual_label = str(row.get("primary_subject", None))

		# PDF link (if DOI exists)
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

	
	def _load_label_encoder(self):
		# Load the encoder from training
		with open("../scibert_label_encoder.pkl", "rb") as f:
			label_encoder = pickle.load(f)

		classes = label_encoder.classes_

		print(f"[INFO] Loaded {len(classes)} classes from label_encoder.pkl")

		return label_encoder, classes

