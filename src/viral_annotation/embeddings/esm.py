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


def _windows(seq: str, size: int) -> list[str]:
    """Split a sequence into consecutive non-overlapping windows of `size`.

    The last window is the remainder (length <= size). An empty sequence yields
    a single empty window so indexing stays simple.
    """
    if not seq:
        return [""]
    return [seq[k:k + size] for k in range(0, len(seq), size)]


def _finalize_stats(owner, n_proteins, w_n, w_sum, w_sq, w_max, w_min):
    """Combine per-window sufficient stats into per-protein [mean|max|min|std] (4d).

    max/min are taken element-wise ACROSS a protein's windows; mean and std come
    from summed (count, sum, sumsq) over ALL residues — so multi-window long
    proteins get exact statistics, not per-window averages. `owner[j]` is the
    protein index that window `j` belongs to.
    """
    import numpy as np

    d = w_sum.shape[1]
    P_n = np.zeros(n_proteins, dtype="float64")
    P_sum = np.zeros((n_proteins, d), dtype="float64")
    P_sq = np.zeros((n_proteins, d), dtype="float64")
    P_max = np.full((n_proteins, d), -np.inf, dtype="float64")
    P_min = np.full((n_proteins, d), np.inf, dtype="float64")
    for j, i in enumerate(owner):
        P_n[i] += w_n[j]
        P_sum[i] += w_sum[j]
        P_sq[i] += w_sq[j]
        np.maximum(P_max[i], w_max[j], out=P_max[i])
        np.minimum(P_min[i], w_min[j], out=P_min[i])

    nz = np.clip(P_n, 1.0, None)[:, None]
    mean = P_sum / nz
    std = np.sqrt(np.clip(P_sq / nz - mean ** 2, 0.0, None))
    P_max[~np.isfinite(P_max)] = 0.0   # proteins with no residues (shouldn't happen)
    P_min[~np.isfinite(P_min)] = 0.0
    return np.concatenate([mean, P_max, P_min, std], axis=1).astype("float32")


class ESMEmbedder:
    """Wraps an ESM-2 model to produce pooled per-protein embeddings."""

    def __init__(
        self,
        model_key: str = DEFAULT_ESM_MODEL,
        pooling: str = DEFAULT_POOLING,
        repr_layer: int | None = None,   # None -> last hidden layer
        device: str | None = None,
        max_length: int = 1022,          # window size / residue cap
        max_tokens: int = 4096,          # token budget per batch (bounds O(L^2) memory)
        window: bool = False,            # True: non-overlapping windows + length-weighted pool
    ):
        if model_key not in ESM2_MODELS:
            raise KeyError(
                f"unknown ESM model {model_key!r}; choose from {list(ESM2_MODELS)}"
            )
        if pooling not in ("mean", "cls", "stats"):
            raise ValueError(f"pooling must be 'mean', 'cls', or 'stats', got {pooling!r}")

        self.spec = ESM2_MODELS[model_key]
        self.pooling = pooling
        self.repr_layer = repr_layer
        self._device = device
        self.max_length = max_length
        self.max_tokens = max_tokens
        self.window = window
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
        # On CUDA, load the weights in fp16 (not just autocast the math): halves
        # resident weight memory (~11GB -> ~5.6GB for 3B) so big backbones fit a
        # 16GB card without spilling to shared system RAM, and it's faster. The
        # forward already ran fp16 under autocast, so pooled features are
        # unchanged. CPU/MPS keep fp32 (no fp16 op coverage guarantee).
        dtype = torch.float16 if self._device == "cuda" else None
        self._model = (
            AutoModel.from_pretrained(self.spec.hf_name, torch_dtype=dtype)
            .to(self._device).eval()
        )

    def embed(self, sequences: Sequence[str], batch_size: int | None = None):
        """Embed sequences -> numpy array [N x D], one vector per protein.

        Pooling: "mean"/"cls" give D=d; "stats" gives D=4d = concat of per-
        dimension mean|max|min|std over residues (max catches active-site-like
        signals the mean dilutes; std catches heterogeneity).

        Long proteins: window=False truncates to `max_length`; window=True splits
        into non-overlapping windows and recombines to ONE per-protein vector. For
        "stats" the recombination uses sufficient statistics, so max/min are taken
        ACROSS windows and std is over ALL residues (exact). Output order matches
        input; the protein is never split at the prediction level.
        """
        if self.pooling == "stats":
            return self._embed_stats(sequences, batch_size)

        if not self.window:
            return self._embed_flat([s[: self.max_length] for s in sequences], batch_size)

        import numpy as np

        flat, owner, wlen = [], [], []
        for i, s in enumerate(sequences):
            for chunk in _windows(s, self.max_length):
                flat.append(chunk)
                owner.append(i)
                wlen.append(len(chunk))
        vecs = self._embed_flat(flat, batch_size)  # [num_windows x d], input order

        d = vecs.shape[1]
        acc = np.zeros((len(sequences), d), dtype="float64")
        wsum = np.zeros(len(sequences), dtype="float64")
        for j, i in enumerate(owner):
            acc[i] += vecs[j] * wlen[j]
            wsum[i] += wlen[j]
        return (acc / np.clip(wsum, 1.0, None)[:, None]).astype("float32")

    def _forward_batches(self, sequences: Sequence[str], batch_size: int | None = None):
        """Yield (idx, hidden[B,L,d], attention_mask[B,L]) per length-sorted batch.

        Shared by the pooled and stats embedders. Batches are packed under the
        token budget so a few long sequences can't OOM the GPU; only the needed
        layer is materialized; GPU buffers are freed after each yield resumes.
        """
        self._ensure_loaded()
        import contextlib

        import torch

        # On CUDA, run the (heavy) transformer forward under fp16 autocast so T4-class
        # tensor cores are used — typically 5-10x faster than fp32 with no meaningful
        # effect on pooled features. No-op on CPU/MPS (keeps those paths bit-identical).
        amp = (torch.autocast(device_type="cuda", dtype=torch.float16)
               if self._device == "cuda" else contextlib.nullcontext())

        seqs = [s[: self.max_length] for s in sequences]
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
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
            enc = self._tokenizer(
                [seqs[k] for k in idx], return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length + 2,
            ).to(self._device)
            want_layers = self.repr_layer is not None
            with torch.no_grad(), amp:
                result = self._model(**enc, output_hidden_states=want_layers)
            hidden = (
                result.hidden_states[self.repr_layer]
                if want_layers
                else result.last_hidden_state
            ).float()  # [B, L, d] -> fp32 for stable pooling (heavy forward ran in fp16)
            yield idx, hidden, enc["attention_mask"]
            del result, hidden, enc
            if self._device == "mps":
                torch.mps.empty_cache()
            elif self._device == "cuda":
                torch.cuda.empty_cache()
            i = j

    def _embed_flat(self, sequences: Sequence[str], batch_size: int | None = None):
        """Embed a flat list (each <= max_length) -> [N x d] via mean/cls pooling."""
        import numpy as np
        import torch

        results: list = [None] * len(sequences)
        for idx, hidden, mask in self._forward_batches(sequences, batch_size):
            pooled = self._pool(hidden, mask, torch).cpu().numpy()
            for slot, k in enumerate(idx):
                results[k] = pooled[slot]
        return np.stack(results, axis=0)

    def _embed_stats(self, sequences: Sequence[str], batch_size: int | None = None):
        """mean|max|min|std pooling -> [N x 4d], exact across windows."""
        import numpy as np
        import torch

        flat, owner = [], []
        for i, s in enumerate(sequences):
            chunks = _windows(s, self.max_length) if self.window else [s[: self.max_length]]
            for c in chunks:
                flat.append(c)
                owner.append(i)

        d = self.spec.dim
        nW = len(flat)
        w_n = np.zeros(nW, dtype="float64")
        w_sum = np.zeros((nW, d), dtype="float64")
        w_sq = np.zeros((nW, d), dtype="float64")
        w_max = np.full((nW, d), -np.inf, dtype="float64")
        w_min = np.full((nW, d), np.inf, dtype="float64")
        for idx, hidden, mask in self._forward_batches(flat, batch_size):
            n, ssum, ssq, smax, smin = self._residue_stats(hidden, mask, torch)
            for slot, k in enumerate(idx):
                w_n[k], w_sum[k], w_sq[k] = n[slot], ssum[slot], ssq[slot]
                w_max[k], w_min[k] = smax[slot], smin[slot]
        return _finalize_stats(owner, len(sequences), w_n, w_sum, w_sq, w_max, w_min)

    def embed_residues(self, sequences: Sequence[str], batch_size: int | None = None):
        """Return per-protein per-residue embeddings for learned (attention) pooling.

        One float16 array [L_i x d] per input protein (input order), holding the
        REAL residue vectors only (BOS/EOS stripped), with a protein's windows
        concatenated in order so attention can range over the whole sequence.
        """
        import numpy as np

        flat, owner, slen = [], [], []
        for i, s in enumerate(sequences):
            chunks = _windows(s, self.max_length) if self.window else [s[: self.max_length]]
            for c in chunks:
                flat.append(c)
                owner.append(i)
                slen.append(len(c))

        win_res: list = [None] * len(flat)
        for idx, hidden, _mask in self._forward_batches(flat, batch_size):
            h = hidden.cpu().numpy()  # [B, Lpad, d]
            for slot, k in enumerate(idx):
                L = slen[k]
                win_res[k] = h[slot, 1:1 + L, :]  # drop BOS at 0; EOS sits at 1+L

        grouped: list = [[] for _ in range(len(sequences))]
        for j, i in enumerate(owner):
            grouped[i].append(win_res[j])
        return [
            np.concatenate(ws, axis=0).astype("float16") if ws else
            np.zeros((0, self.spec.dim), dtype="float16")
            for ws in grouped
        ]

    def _residue_stats(self, hidden, attention_mask, torch):
        """Per-sequence sufficient stats over real residues -> numpy (n, sum, sumsq, max, min)."""
        m = attention_mask.unsqueeze(-1).type_as(hidden)          # [B, L, 1]
        n = m.sum(dim=1)                                           # [B, 1]
        ssum = (hidden * m).sum(dim=1)                            # [B, d]
        ssq = (hidden * hidden * m).sum(dim=1)                    # [B, d]
        pad = attention_mask.unsqueeze(-1) == 0                   # [B, L, 1] bool
        smax = hidden.masked_fill(pad, float("-inf")).max(dim=1).values
        smin = hidden.masked_fill(pad, float("inf")).min(dim=1).values
        return (n.squeeze(-1).cpu().numpy(), ssum.cpu().numpy(), ssq.cpu().numpy(),
                smax.cpu().numpy(), smin.cpu().numpy())

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
