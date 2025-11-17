# label_cluster_visualizer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


@dataclass
class LabelClusterVisualizer:
	"""
	Utility for visualizing clustering of label texts using a transformer encoder.

	Typical usage:

		viz = LabelClusterVisualizer(model_name="models/longformer-base-4096")
		result = viz.visualize(categories)  # categories: List[str] of primary_subjects

	You can then inspect:
		- result["coords"]       -> np.ndarray [num_labels, 2]
		- result["cluster_ids"]  -> np.ndarray [num_labels]
		- result["codes"]        -> List[str] (short label codes like 'cs.CV')
		- result["labels"]       -> original full label strings

	And print a textual summary:
		viz.print_cluster_summary()
	"""

	model_name: str
	max_length: int = 64
	n_clusters: int = 12
	random_state: int = 42
	device: Optional[str] = None  # "cuda", "cpu", or None for auto

	# Internal state (populated after visualize())
	_tokenizer: Any = None
	_model: Any = None
	_last_embeddings: Optional[np.ndarray] = None
	_last_coords: Optional[np.ndarray] = None
	_last_cluster_ids: Optional[np.ndarray] = None
	_last_codes: Optional[List[str]] = None
	_last_labels: Optional[List[str]] = None

	def __post_init__(self) -> None:
		if self.device is None:
			self.device = "cuda" if torch.cuda.is_available() else "cpu"

		self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
		self._model = AutoModel.from_pretrained(self.model_name)
		self._model.to(self.device)
		self._model.eval()

	def _extract_code(self, subject: str) -> str:
		"""
		Extract a compact code from a full subject string.

		Example:
			"Computer Vision and Pattern Recognition (cs.CV)" -> "cs.CV"
		If no parentheses are found, returns the original string.
		"""
		if "(" in subject and ")" in subject:
			code = subject.split("(")[-1].split(")")[0].strip()
			return code
		return subject

	def embed_labels(self, labels: List[str]) -> np.ndarray:
		"""
		Encode label texts into embeddings using the underlying model.

		Returns:
			embeddings: np.ndarray of shape [num_labels, hidden_dim]
		"""
		if len(labels) == 0:
			raise ValueError("embed_labels: labels list is empty")

		enc = self._tokenizer(
			labels,
			padding=True,
			truncation=True,
			max_length=self.max_length,
			return_tensors="pt",
		)
		enc = {k: v.to(self.device) for k, v in enc.items()}

		with torch.no_grad():
			out = self._model(**enc)

		hidden = out.last_hidden_state  # [B, L, H]
		# CLS pooling (token 0):
		# For RoBERTa/Longformer-style models, this is common
		embeddings = hidden[:, 0, :].detach().cpu().numpy()
		return embeddings

	def cluster(self, embeddings: np.ndarray) -> np.ndarray:
		"""
		Run k-means clustering on embeddings.

		Returns:
			cluster_ids: np.ndarray of shape [num_labels] with integer cluster id.
		"""
		if embeddings.ndim != 2:
			raise ValueError(f"cluster: expected 2D embeddings, got shape {embeddings.shape}")

		kmeans = KMeans(
			n_clusters=self.n_clusters,
			random_state=self.random_state,
			n_init=10,
		)
		cluster_ids = kmeans.fit_predict(embeddings)
		return cluster_ids

	def project_to_2d(self, embeddings: np.ndarray) -> np.ndarray:
		"""
		Project embeddings to 2D with t-SNE for visualization.

		Returns:
			coords: np.ndarray [num_labels, 2]

		NOTES:
			Perplexity represents the effective number of neighbors each point considers 
			when calculating pairwise similarities in the high-dimensional space.
		
			In the context of t-SNE (t-distributed Stochastic Neighbor Embedding),
			perplexity is a crucial hyperparameter that essentially controls the balance
			between preserving local vs. global structure in the data during dimensionality reduction.
		"""
		if embeddings.ndim != 2:
			raise ValueError(f"project_to_2d: expected 2D embeddings, got shape {embeddings.shape}")

		num_points = embeddings.shape[0]
		# Perplexity must be < num_points
		# A reasonable value must be selected
		perplexity = min(30, max(5, num_points // 5))

		tsne = TSNE(
			n_components=2,
			random_state=self.random_state,
			perplexity=perplexity,
			init="random",
		)
		coords = tsne.fit_transform(embeddings)
		return coords

	def visualize(
		self,
		label_texts: List[str],
		show_codes: bool = True,
		figsize: Tuple[int, int] = (12, 9),
		alpha: float = 0.85,
		fontsize: int = 7,
	) -> Dict[str, Any]:
		"""
		Full pipeline:
			- Compute embeddings for label_texts
			- Cluster them
			- Project to 2D
			- Plot scatter with codes as text labels

		Stores results on the instance and returns them as a dict.
		"""
		if len(label_texts) == 0:
			raise ValueError("visualize: label_texts list is empty")

		# Prepare codes (compact labels) for annotation
		label_codes = [self._extract_code(t) for t in label_texts]

		# 1) Embeddings
		embeddings = self.embed_labels(label_texts)

		# 2) Clustering
		cluster_ids = self.cluster(embeddings)

		# 3) 2D projection
		coords = self.project_to_2d(embeddings)

		# 4) Plot
		plt.figure(figsize=figsize)
		scatter = plt.scatter(
			coords[:, 0],
			coords[:, 1],
			c=cluster_ids,
			cmap="tab20",
			alpha=alpha,
		)

		if show_codes:
			for i, code in enumerate(label_codes):
				plt.text(
					coords[i, 0],
					coords[i, 1],
					code,
					fontsize=fontsize,
					ha="center",
					va="center",
				)

		plt.title("Label clustering (transformer embeddings)")
		plt.xlabel("t-SNE dim 1")
		plt.ylabel("t-SNE dim 2")
		plt.tight_layout()
		plt.show()

		# Save state
		self._last_embeddings = embeddings
		self._last_coords = coords
		self._last_cluster_ids = cluster_ids
		self._last_codes = label_codes
		self._last_labels = list(label_texts)

		return {
			"embeddings": embeddings,
			"coords": coords,
			"cluster_ids": cluster_ids,
			"codes": label_codes,
			"labels": list(label_texts),
		}

	def get_cluster_summary(self) -> Dict[int, List[Tuple[str, str]]]:
		"""
		Return a mapping: cluster_id -> list of (code, full_label).

		Requires that visualize() has been called at least once.
		"""
		if (
			self._last_cluster_ids is None
			or self._last_codes is None
			or self._last_labels is None
		):
			raise RuntimeError("get_cluster_summary: call visualize() first")

		clusters: Dict[int, List[Tuple[str, str]]] = {}
		for cid, code, label in zip(
			self._last_cluster_ids, self._last_codes, self._last_labels
		):
			clusters.setdefault(int(cid), []).append((code, label))
		return clusters

	def print_cluster_summary(self) -> None:
		"""
		Print a human-readable summary of clusters:
			=== Cluster 0 ===
			  cs.CV      Computer Vision and Pattern Recognition (cs.CV)
			  cs.LG      Machine Learning (cs.LG)
			...
		"""
		clusters = self.get_cluster_summary()
		for cid in sorted(clusters.keys()):
			print(f"\n=== Cluster {cid} ===")
			for code, full in clusters[cid]:
				print(f"  {code:10s}  {full}")
