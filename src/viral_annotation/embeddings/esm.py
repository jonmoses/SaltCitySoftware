"""Per-protein ESM-2 embeddings (docs/01 Step 1).

Runs a protein sequence through ESM-2 to get [L x d] per-residue representations,
then pools to a single [d] vector. Mean pooling is the default; CLS pooling is
available to benchmark.

torch / transformers are imported lazily inside the class so that importing this
module (and the rest of the package) does not require the heavy `[ml]` extra.

KNOWN vs. TO-SWEEP:
  * Model ids and dims (config.ESM2_MODELS) are known.
  * `repr_layer` — which transformer layer to pool from — is a hyperparameter;
    the last layer is the default here but is NOT necessarily optimal (docs/01).
"""

from __future__ import annotations

from typing import Sequence

from viral_annotation.config import (
    DEFAULT_ESM_MODEL,
    DEFAULT_POOLING,
    ESM2_MODELS,
)


class ESMEmbedder:
    """Wraps an ESM-2 model to produce pooled per-protein embeddings."""

    def __init__(
        self,
        model_key: str = DEFAULT_ESM_MODEL,
        pooling: str = DEFAULT_POOLING,
        repr_layer: int | None = None,   # None -> last hidden layer
        device: str | None = None,
    ):
        if model_key not in ESM2_MODELS:
            raise KeyError(
                f"unknown ESM model {model_key!r}; choose from {list(ESM2_MODELS)}"
            )
        if pooling not in ("mean", "cls"):
            raise ValueError(f"pooling must be 'mean' or 'cls', got {pooling!r}")

        self.spec = ESM2_MODELS[model_key]
        self.pooling = pooling
        self.repr_layer = repr_layer
        self._device = device
        self._model = None
        self._tokenizer = None

    @property
    def dim(self) -> int:
        return self.spec.dim

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Lazy heavy imports.
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.spec.hf_name)
        self._model = AutoModel.from_pretrained(self.spec.hf_name).to(self._device).eval()

    def embed(self, sequences: Sequence[str], batch_size: int = 8):
        """Embed sequences -> numpy array [N x d] of pooled vectors.

        Batch by similar length upstream to minimize padding waste (docs/04).
        """
        self._ensure_loaded()
        import numpy as np
        import torch

        out: list = []
        for start in range(0, len(sequences), batch_size):
            batch = list(sequences[start:start + batch_size])
            enc = self._tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            ).to(self._device)
            with torch.no_grad():
                result = self._model(**enc, output_hidden_states=True)
            hidden = (
                result.last_hidden_state
                if self.repr_layer is None
                else result.hidden_states[self.repr_layer]
            )  # [B, L, d]
            pooled = self._pool(hidden, enc["attention_mask"], torch)
            out.append(pooled.cpu().numpy())
        return np.concatenate(out, axis=0)

    def _pool(self, hidden, attention_mask, torch):
        if self.pooling == "cls":
            # Token 0 is the start-of-sequence token for ESM-2.
            return hidden[:, 0, :]
        # Mean over real (non-pad) residues. ESM adds BOS/EOS tokens; the
        # attention mask covers them — acceptable for a baseline, but a stricter
        # variant would also exclude BOS/EOS. Left as a documented choice.
        mask = attention_mask.unsqueeze(-1).type_as(hidden)  # [B, L, 1]
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts
