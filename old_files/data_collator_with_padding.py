"""
Custom collator that also pads global_attention_mask
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch

@dataclass
class DataCollatorWithGlobalPadding:
	tokenizer: Any
	pad_to_multiple_of: Optional[int] = 8
	create_default_gam: bool = True  # auto-create CLS-global mask if missing

	def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
		# Collect or create per-sample global_attention_mask BEFORE padding
		gams = []
		for f in features:
			m = f.pop("global_attention_mask", None)
			if m is None and self.create_default_gam:
				seq_len = len(f["input_ids"])
				m = [1] + [0] * (seq_len - 1)  # CLS (pos 0) global
			gams.append(m)

		# Dynamic padding for input_ids/attention_mask/etc.  (NO truncation kw)
		batch = self.tokenizer.pad(
			features,
			padding=True,
			pad_to_multiple_of=self.pad_to_multiple_of,
			return_tensors="pt",
		)

		# Pad GAM to the same length as input_ids (if any present)
		if any(m is not None for m in gams):
			max_len = batch["input_ids"].shape[1]
			gams_padded = []
			for m in gams:
				if m is None:
					m = [1] + [0] * (max_len - 1)
				elif len(m) < max_len:
					m = m + [0] * (max_len - len(m))
				else:
					m = m[:max_len]
				gams_padded.append(m)
			batch["global_attention_mask"] = torch.tensor(gams_padded, dtype=torch.long)

		return batch
