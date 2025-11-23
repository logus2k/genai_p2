# Improving the Performance of a LLM Finetuned for arXiv Paper Classification

In this report we proposes three targeted enhancements to the Large Language Model (LLM) system for classifying arXiv scientific papers, as proposed in project 2, and implemented in project 3. We used SciBERT fine-tuned for single-label multi-class classification on a balanced subset of titles and abstracts, achieving ~60% test accuracy. Drawing from that implementation, supplementary notes on boundaries (e.g., single-label focus, lack of temporal modeling), and recent advancements in LLM techniques (e.g., retrieval-augmented methods, reasoning augmentation, and self-improving paradigms), can effectively address the current model key limitations creatively, while balancing implementation effort.

Each proposal follows a structured approach: first, we identify the current limitation. Then, we propose a solution to address it, and justify our approach, by explaining the integration rationale. Finally, we define our chosen evaluation metrics. Proposals are prioritized by impact and effort: high-impact/low-effort first. Overall, when implemented, these proposals could boost the model accuracy to 70-80% on interdisciplinary papers, while also improving its deployability.

## Proposal 1: Retrieval-Augmented Multi-Label Classification

### 1. Current Limitation

The current system is restricted to single-label classification on the `primary_subject`, ignoring the multi-label `subjects` field, which leads to poor handling of interdisciplinary papers (e.g., a physics paper with math applications). This results in forced mutually exclusive predictions via softmax, contradicting the business goal of independent probabilities for multiple categories. Additionally, the balanced subset (max 1000 samples/category) underrepresents rare classes and real-world imbalances, contributing to lower F1-scores in bottom-performing categories (~0.2-0.4 as per evaluation heatmaps). Temporal drift from decades-spanning data exacerbates this, as terminology evolves (e.g., "neural networks" shifting contexts), but is unaddressed.

### 2. Solution Proposal

Integrate a Retrieval-Augmented Generation (RAG) module with a multi-label classifier head. Specifically:

- Use the `subjects` field as multi-label targets (binary vectors for 148 categories).
- Replace softmax with sigmoid activation and Binary Cross-Entropy (BCE) loss for independent probabilities.
- Add a RAG component: Embed abstracts using a dense retriever (e.g., Sentence-BERT or the existing SciBERT embeddings), query a vector store (e.g., FAISS index built from the full 2.55M dataset) to retrieve top-k similar papers (k=5-10), and concatenate their subjects/categories as context prompts to the LLM input.
- For rare classes, apply focal loss to downweight well-classified examples, enhancing focus on underrepresented ones.

This builds on the notebook's tokenizer and dataset class, adding a retrieval step in data preparation.

### 3. Approach Justification

RAG addresses information gaps by injecting relevant external knowledge at inference time, improving handling of rare/interdisciplinary cases without retraining the entire model. It integrates seamlessly: During fine-tuning, augment inputs with retrieved contexts (e.g., "Similar papers: [subjects list]"); at deployment, use the same for new papers. Multi-label sigmoid enables non-normalized probabilities, aligning with the notes' emphasis on full `subjects` prediction. Focal loss creatively handles imbalance without oversampling, reducing effort compared to few-shot methods. Expected improvements: Better capture of overlaps (e.g., +15-20% mAP on multi-label tasks per recent benchmarks), and robustness to drift via dynamic retrieval. Effort: Medium (2-4 weeks for a data scientist): Modify loss/head (~1 day), build FAISS index (~2 days), integrate into loop (~1 week testing). Leverages notebook's modularity (e.g., add to CustomDataset).

### 4. Evaluation Metrics

- **Primary**: Mean Average Precision (mAP) across labels, as it rewards ranking and handles imbalance better than macro-F1.
- **Secondary**: Hamming Loss (fraction of wrong labels) for multi-label accuracy; Partial Accuracy@K (correct top-k labels match).
- **Theoretical Rigor**: Compare against baseline on held-out test set with interdisciplinary subsets (e.g., papers with >1 subject). Use statistical tests (e.g., paired t-test on mAP) for significance; monitor calibration via Brier Score for probability reliability.

## Proposal 2: LLM-Generated Reasoning for Enhanced Calibration and Selective Prediction

### 1. Current Limitation
Predictions lack explainability, making it hard for stakeholders (e.g., librarians) to trust or refine outputs, especially in low-confidence cases (~40% of test set below 0.7 threshold per deployment notes). Overconfidence from label smoothing/mixup is mitigated but not eliminated, leading to errors in edgy categories. The single-stage classification ignores reasoning chains, reducing performance on nuanced scientific abstracts where context (e.g., methodology vs. theory) matters. No selective prediction exists beyond ad-hoc thresholds, wasting resources on uncertain cases.

### 2. Solution Proposal

Adopt a two-stage approach with LLM-generated reasoning: 

- Stage 1: Prompt the fine-tuned LLM to generate a reasoning trace (e.g., "Step 1: Identify key terms [list]. Step 2: Map to domains [reason]. Step 3: Predict labels [output]") before classification.
- Stage 2: Feed the reasoning as additional input to the classifier head for final prediction.
- Add temperature scaling post-training to calibrate probabilities (e.g., optimize scaling factor on validation logits).
- For selective prediction: If max probability <0.6, flag for manual review; else, auto-classify.

Use Chain-of-Thought (CoT) prompting during fine-tuning, drawing from raw abstracts to simulate reasoning without extra data.

### 3. Approach Justification

LLM-generated reasoning enhances classification by making implicit knowledge explicit, improving accuracy on complex texts. It integrates via prompt engineering in the notebook's tokenizer (e.g., prepend "Reason step-by-step:" to inputs), requiring minimal code changes. Calibration via temperature scaling reduces overconfidence, aligning with notes' extension for selective prediction and boosting high-confidence accuracy (~72% baseline to potentially 85%). Creativity: Repurpose the LLM for self-reasoning, enabling explainable outputs (e.g., traces for users). For temporal drift, reasoning can adapt to evolving terms via dynamic prompts. Effort: Low (1-2 weeks): Add CoT to training (~2 days), implement scaling (~1 day), test selective logic (~3 days). High relevance, as it builds on existing evaluation (e.g., confusion matrices) for quick wins.

### 4. Evaluation Metrics

- **Primary**: Expected Calibration Error (ECE), binning probabilities and measuring deviation from true accuracy.
- **Secondary**: Accuracy@Coverage (accuracy on high-confidence subset vs. coverage ratio); Reasoning Quality Score (human-rated coherence of traces on 1-5 scale, or automated via perplexity).
- **Theoretical Rigor**: Use reliability diagrams for visual ECE assessment; A/B testing on test set subsets (e.g., pre- vs. post-2020 papers for drift). Compute confidence intervals via bootstrapping to ensure statistical validity.

## Proposal 3: Reinforcement Pre-Training for Handling Temporal Drift and Rare Classes

### 1. Current Limitation

No modeling of temporal drift means the system underperforms on recent papers with new terminology (e.g., "transformer" pre-2017 vs. now). Rare classes suffer from sampling caps, with few-shot-like issues unaddressed. Static fine-tuning lacks adaptability, risking obsolescence as arXiv grows (~200k papers/year). The notes highlight this as a boundary, with no incremental retraining.

### 2. Solution Proposal

Incorporate Reinforcement Pre-Training (RPT): 

- Pre-train the LLM on raw arXiv text with RL rewards for next-token reasoning (e.g., reward accurate domain predictions in sequences).
- For drift: Stratify data by submission date, using time-aware rewards (e.g., higher for recent papers).
- Add an agentic loop: The LLM generates synthetic rare-class examples via self-evolution (e.g., mutate abstracts), then fine-tunes incrementally.
- Use active knowledge distillation: Query the model on uncertain samples, distill from a teacher (e.g., larger LLM like Llama 3.1).

### 3. Approach Justication

RPT unlocks richer reasoning from raw text using intrinsic RL signals, improving adaptability without labeled data. Integration: Add RL phase before notebook's fine-tuning (e.g., modify loss to include rewards via PPO). For drift, date-stratified batches ensure temporal awareness; agentic self-evolution creatively addresses rare classes via notes' few-shot suggestion. Expected impact: +10-15% on recent/rare subsets, per 2025 surveys on specialized LLMs. Effort: High (4-6 weeks): Implement RL (e.g., Hugging Face RLHF tools, ~2 weeks), synthetic generation (~1 week), incremental pipeline (~2 weeks). Prioritize if full-scale deployment is targeted, as it enables monitoring per notes.

### 4. Evaluation Metrics

- **Primary**: Temporal Accuracy Decay (accuracy drop from oldest to newest quintiles).
- **Secondary**: Few-Shot F1 (on held-out rare classes with 1-10 samples); Distillation Efficiency (KL-divergence between teacher/student).
- **Theoretical Rigor**: Longitudinal analysis via time-series regression on accuracy; use McNemar's test for pre/post-improvement significance. Monitor via drift detection (e.g., Kolmogorov-Smirnov on embedding distributions).

## Conclusions

Prioritize Proposal 1 (multi-label RAG) for immediate impact on core limitations with medium effort, followed by Proposal 2 (reasoning/calibration) for quick trust-building. Proposal 3 suits long-term scalability but higher effort. These proposals creatively extend the notebook (e.g., leveraging CONFIG for experiments), potentially increasing production coverage from 12% to 50%. Future iterations could benchmark via A/B tests, ensuring alignment with evolving LLM trends. Total estimated effort: 7-12 weeks phased, yielding a more robust, explainable system for arXiv classification.

---
