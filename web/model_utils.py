import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import numpy as np


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

		# Load categories (same order as training)
		self.categories = self._build_category_list()

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
	def predict_with_confidence(self, text: str, top_k: int = 3):
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
	def get_sample_by_index(self, df, index: int):
		if index < 0 or index >= len(df):
			return None

		row = df.iloc[index]

		text = ""
		if "title" in row:
			text += row["title"] + "\n"
		if "abstract" in row:
			text += row["abstract"]

		actual_label = row.get("primary_subject", "Unknown")

		metadata = {
			col: row[col]
			for col in df.columns
			if col not in ["title", "abstract", "primary_subject"]
		}

		return {
			"index": index,
			"text": text,
			"actual_label": actual_label,
			"metadata": metadata,
		}

	# ------------------------------------------------------------
	def _build_category_list(self):
		"""
		Load the same category list used during training (shortened here for readability).
		Replace this with your full list from training.
		"""
		from sklearn.preprocessing import LabelEncoder

		# Full exact list should be here.
		# I will load the same list you provided earlier.
		return [
			"Computer Vision and Pattern Recognition (cs.CV)",
			"Quantum Physics (quant-ph)",
			"High Energy Physics - Phenomenology (hep-ph)",
			"Machine Learning (cs.LG)",
			"High Energy Physics - Theory (hep-th)",
			"Mesoscale and Nanoscale Physics (cond-mat.mes-hall)",
			"Materials Science (cond-mat.mtrl-sci)",
			"Computation and Language (cs.CL)",
			"General Relativity and Quantum Cosmology (gr-qc)",
			"Analysis of PDEs (math.AP)",
			"Astrophysics of Galaxies (astro-ph.GA)",
			"Strongly Correlated Electrons (cond-mat.str-el)",
			"Combinatorics (math.CO)",
			"Solar and Stellar Astrophysics (astro-ph.SR)",
			"High Energy Astrophysical Phenomena (astro-ph.HE)",
			"Cosmology and Nongalactic Astrophysics (astro-ph.CO)",
			"Statistical Mechanics (cond-mat.stat-mech)",
			"Probability (math.PR)",
			"Algebraic Geometry (math.AG)",
			"Information Theory (cs.IT)",
			"Optimization and Control (math.OC)",
			"Nuclear Theory (nucl-th)",
			"Number Theory (math.NT)",
			"Superconductivity (cond-mat.supr-con)",
			"Numerical Analysis (math.NA)",
			"Robotics (cs.RO)",
			"Differential Geometry (math.DG)",
			"Soft Condensed Matter (cond-mat.soft)",
			"Artificial Intelligence (cs.AI)",
			"Optics (physics.optics)",
			"Cryptography and Security (cs.CR)",
			"Earth and Planetary Astrophysics (astro-ph.EP)",
			"High Energy Physics - Experiment (hep-ex)",
			"Dynamical Systems (math.DS)",
			"Systems and Control (eess.SY)",
			"Methodology (stat.ME)",
			"Functional Analysis (math.FA)",
			"Signal Processing (eess.SP)",
			"Instrumentation and Methods for Astrophysics (astro-ph.IM)",
			"Machine Learning (stat.ML)",
			"Image and Video Processing (eess.IV)",
			"Networking and Internet Architecture (cs.NI)",
			"High Energy Physics - Lattice (hep-lat)",
			"Fluid Dynamics (physics.flu-dyn)",
			"Software Engineering (cs.SE)",
			"Data Structures and Algorithms (cs.DS)",
			"Geometric Topology (math.GT)",
			"Representation Theory (math.RT)",
			"Classical Analysis and ODEs (math.CA)",
			"Statistics Theory (math.ST)",
			"Human-Computer Interaction (cs.HC)",
			"Distributed, Parallel, and Cluster Computing (cs.DC)",
			"Group Theory (math.GR)",
			"Quantum Gases (cond-mat.quant-gas)",
			"Computers and Society (cs.CY)",
			"Instrumentation and Detectors (physics.ins-det)",
			"Information Retrieval (cs.IR)",
			"Nuclear Experiment (nucl-ex)",
			"Disordered Systems and Neural Networks (cond-mat.dis-nn)",
			"Physics and Society (physics.soc-ph)",
			"Atomic Physics (physics.atom-ph)",
			"Rings and Algebras (math.RA)",
			"Complex Variables (math.CV)",
			"Social and Information Networks (cs.SI)",
			"Chemical Physics (physics.chem-ph)",
			"Logic in Computer Science (cs.LO)",
			"Quantum Algebra (math.QA)",
			"Plasma Physics (physics.plasm-ph)",
			"Logic (math.LO)",
			"Applications (stat.AP)",
			"Algebraic Topology (math.AT)",
			"Applied Physics (physics.app-ph)",
			"Commutative Algebra (math.AC)",
			"Sound (cs.SD)",
			"Chaotic Dynamics (nlin.CD)",
			"Operator Algebras (math.OA)",
			"Computer Science and Game Theory (cs.GT)",
			"General Physics (physics.gen-ph)",
			"Populations and Evolution (q-bio.PE)",
			"Audio and Speech Processing (eess.AS)",
			"Neural and Evolutionary Computing (cs.NE)",
			"Computational Physics (physics.comp-ph)",
			"Neurons and Cognition (q-bio.NC)",
			"Other Condensed Matter (cond-mat.other)",
			"Databases (cs.DB)",
			"Metric Geometry (math.MG)",
			"Exactly Solvable and Integrable Systems (nlin.SI)",
			"Biological Physics (physics.bio-ph)",
			"Accelerator Physics (physics.acc-ph)",
			"Quantitative Methods (q-bio.QM)",
			"Computational Complexity (cs.CC)",
			"Programming Languages (cs.PL)",
			"Pattern Formation and Solitons (nlin.PS)",
			"Symplectic Geometry (math.SG)",
			"Spectral Theory (math.SP)",
			"Discrete Mathematics (cs.DM)",
			"Medical Physics (physics.med-ph)",
			"Computational Engineering, Finance, and Science (cs.CE)",
			"General Economics (econ.GN)",
			"Computational Geometry (cs.CG)",
			"Classical Physics (physics.class-ph)",
			"Category Theory (math.CT)",
			"Hardware Architecture (cs.AR)",
			"General Mathematics (math.GM)",
			"Computation (stat.CO)",
			"Geophysics (physics.geo-ph)",
			"Digital Libraries (cs.DL)",
			"Atmospheric and Oceanic Physics (physics.ao-ph)",
			"General Topology (math.GN)",
			"Biomolecules (q-bio.BM)",
			"Multiagent Systems (cs.MA)",
			"Graphics (cs.GR)",
			"Adaptation and Self-Organizing Systems (nlin.AO)",
			"Econometrics (econ.EM)",
			"History and Overview (math.HO)",
			"Data Analysis, Statistics and Probability (physics.data-an)",
			"Formal Languages and Automata Theory (cs.FL)",
			"History and Philosophy of Physics (physics.hist-ph)",
			"Physics Education (physics.ed-ph)",
			"K-Theory and Homology (math.KT)",
			"Multimedia (cs.MM)",
			"Emerging Technologies (cs.ET)",
			"Molecular Networks (q-bio.MN)",
			"Theoretical Economics (econ.TH)",
			"Statistical Finance (q-fin.ST)",
			"Space Physics (physics.space-ph)",
			"Other Computer Science (cs.OH)",
			"Genomics (q-bio.GN)",
			"Mathematical Finance (q-fin.MF)",
			"General Finance (q-fin.GN)",
			"Risk Management (q-fin.RM)",
			"Computational Finance (q-fin.CP)",
			"Symbolic Computation (cs.SC)",
			"Portfolio Management (q-fin.PM)",
			"Performance (cs.PF)",
			"Popular Physics (physics.pop-ph)",
			"Tissues and Organs (q-bio.TO)",
			"Pricing of Securities (q-fin.PR)",
			"Trading and Market Microstructure (q-fin.TR)",
			"Atomic and Molecular Clusters (physics.atm-clus)",
			"Mathematical Software (cs.MS)",
			"Cell Behavior (q-bio.CB)",
			"Other Quantitative Biology (q-bio.OT)",
			"Other Statistics (stat.OT)",
			"Cellular Automata and Lattice Gases (nlin.CG)",
			"Subcellular Processes (q-bio.SC)",
			"Operating Systems (cs.OS)",
			"General Literature (cs.GL)",
		]
