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


def _auto_device(torch) -> str:
    """Pick the best available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ESMEmbedder:
    """Wraps an ESM-2 model to produce pooled per-protein embeddings."""

    def __init__(
        self,
        model_key: str = DEFAULT_ESM_MODEL,
        pooling: str = DEFAULT_POOLING,
        repr_layer: int | None = None,   # None -> last hidden layer
        device: str | None = None,
        max_length: int = 1022,          # cap residues; long polyproteins are truncated
        max_tokens: int = 4096,          # token budget per batch (bounds O(L^2) memory)
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
        self.max_length = max_length
        self.max_tokens = max_tokens
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
            self._device = _auto_device(torch)
        self._tokenizer = AutoTokenizer.from_pretrained(self.spec.hf_name)
        self._model = AutoModel.from_pretrained(self.spec.hf_name).to(self._device).eval()

    def embed(self, sequences: Sequence[str], batch_size: int | None = None):
        """Embed sequences -> numpy array [N x d] of pooled vectors.

        Memory-safe by construction: sequences are truncated to `max_length`,
        sorted by length so padding is minimal, and packed into batches under a
        token budget (`max_tokens`) so a few long polyproteins can't OOM the GPU.
        `batch_size`, if given, caps the per-batch count too. Output order matches
        the input order. Only the needed layer is materialized.
        """
        self._ensure_loaded()
        import numpy as np
        import torch

        seqs = [s[: self.max_length] for s in sequences]
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        results: list = [None] * len(seqs)

        i = 0
        while i < len(order):
            # Greedily grow a batch until the token budget (count * longest) is hit.
            j = i
            longest = 0
            while j < len(order):
                longest = max(longest, len(seqs[order[j]]) + 2)  # +2 for BOS/EOS
                count = j - i + 1
                if count * longest > self.max_tokens and j > i:
                    break
                if batch_size and count >= batch_size:
                    j += 1
                    break
                j += 1

            idx = order[i:j]
            batch = [seqs[k] for k in idx]
            enc = self._tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_length + 2,
            ).to(self._device)
            want_layers = self.repr_layer is not None
            with torch.no_grad():
                result = self._model(**enc, output_hidden_states=want_layers)
            hidden = (
                result.hidden_states[self.repr_layer]
                if want_layers
                else result.last_hidden_state
            )  # [B, L, d]
            pooled = self._pool(hidden, enc["attention_mask"], torch).cpu().numpy()
            for slot, k in enumerate(idx):
                results[k] = pooled[slot]
            # Release this batch's GPU buffers. Without this, the MPS allocator
            # caches each batch's peak and the high-water mark climbs across
            # thousands of sequences until it hits the watermark and OOMs.
            del result, hidden, enc
            if self._device == "mps":
                torch.mps.empty_cache()
            elif self._device == "cuda":
                torch.cuda.empty_cache()
            i = j

        return np.stack(results, axis=0)

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
