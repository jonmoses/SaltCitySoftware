"""End-to-end LoRA fine-tuning of the ESM-2 backbone (the `--finetune lora` mode).

The frozen-embedding trainer (`heads.fit_namespace`) caps out at a linear/attention
head on *fixed* pooled vectors. Here the backbone is **unfrozen** with low-rank
adapters and trained jointly with the per-namespace pooling + heads, so the
representation itself adapts to the pathogen's sequences — the biggest lever over a
frozen pLM + linear map.

Design (see docs/08 + the plan):
  * **One shared LoRA backbone, three per-namespace heads** (multi-task). A single
    backbone forward yields hidden states `[B,L,d]` that feed an attention pooler for
    molecular_function and a mean pool for biological_process / cellular_component —
    far cheaper than three separate fine-tunes and it fits a Kaggle T4.
  * **Per-namespace evidence policy is preserved by masking**, not by splitting the
    backbone: every protein contributes loss to a namespace only if it is in that
    namespace's `train_pool` (so MF can stay manual-only while BP/CC train on
    manual+IEA). Val/test always score manual-only, exactly as the frozen path.
  * **No per-residue disk cache.** Residues are computed live and back-propagated
    through, so the ~hundreds-of-GB cache the frozen attention path needs is gone.

Sequences are truncated to `max_length` for training (bacterial proteins average
~300 aa, so few are clipped) — distinct from the frozen path's windowing of >1022aa
proteins. torch / transformers / peft are imported lazily, per the repo convention.
"""

from __future__ import annotations

from dataclasses import dataclass

from viral_annotation.config import (
    ESM2_MODELS,
    FT_ASL_CLIP,
    FT_ASL_GAMMA_NEG,
    FT_ASL_GAMMA_POS,
    FT_LORA_ALPHA,
    FT_LORA_DROPOUT,
    FT_LORA_R,
    FT_LORA_TARGETS,
    GO_NAMESPACES,
    POS_WEIGHT_CLAMP,
    TRAIN_EARLY_STOP_PATIENCE,
    TRAIN_WEIGHT_DECAY,
)
from viral_annotation.data.dataset import build_labels, select_vocab
from viral_annotation.evaluation.metrics import fmax_matrix
from viral_annotation.training.heads import Head


@dataclass
class FinetuneArtifacts:
    """Everything needed to score with and persist a trained fine-tuned model."""

    model: object              # MultiTaskAnnotator (eval-ready)
    tokenizer: object
    vocabs: dict               # ns -> TermVocab
    pooling: dict              # ns -> "attention" | "mean"
    model_key: str
    n_heads: int
    hidden_dims: list
    dropout: float
    max_length: int


# --- model construction -----------------------------------------------------
def build_lora_backbone(model_key, *, r=FT_LORA_R, alpha=FT_LORA_ALPHA,
                        dropout=FT_LORA_DROPOUT, targets=FT_LORA_TARGETS,
                        grad_checkpointing=True):
    """Load the ESM-2 backbone and wrap it with LoRA adapters (only adapters train)."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    spec = ESM2_MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
    base = AutoModel.from_pretrained(spec.hf_name)
    if grad_checkpointing:
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()   # required for checkpointing + frozen base
    lconf = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                       target_modules=list(targets), task_type=TaskType.FEATURE_EXTRACTION)
    return get_peft_model(base, lconf), tokenizer, spec.dim


def build_multitask_model(backbone, dim, ns_specs, *, n_heads, hidden_dims, dropout):
    """Assemble the shared-backbone, per-namespace multi-task annotator.

    `ns_specs` is a list of dicts: {"ns", "num_terms", "pooling"}.
    """
    import torch.nn as nn

    from viral_annotation.classifier.model import build_classifier
    from viral_annotation.classifier.pooling import build_attn_pool

    class MultiTaskAnnotator(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.namespaces = [s["ns"] for s in ns_specs]
            self.poolings = {s["ns"]: s["pooling"] for s in ns_specs}
            self.attn = nn.ModuleDict()
            self.heads = nn.ModuleDict()
            for s in ns_specs:
                if s["pooling"] == "attention":
                    pool = build_attn_pool(dim, n_heads=n_heads)
                    self.attn[s["ns"]] = pool
                    in_dim = pool.out_dim
                else:
                    in_dim = dim
                self.heads[s["ns"]] = build_classifier(
                    in_dim, s["num_terms"], hidden_dims=hidden_dims, dropout=dropout,
                    layernorm=True, gelu=True)

        def forward(self, input_ids, attention_mask):
            hidden = self.backbone(input_ids=input_ids,
                                   attention_mask=attention_mask).last_hidden_state
            m = attention_mask.unsqueeze(-1).type_as(hidden)               # [B,L,1]
            mean = (hidden * m).sum(1) / m.sum(1).clamp(min=1)             # masked mean
            logits = {}
            for ns in self.namespaces:
                pooled = (self.attn[ns](hidden, attention_mask)
                          if self.poolings[ns] == "attention" else mean)
                logits[ns] = self.heads[ns](pooled)
            return logits

    return MultiTaskAnnotator()


# --- helpers ----------------------------------------------------------------
def _pool_mask(proteins, pool):
    """Per-protein {0,1} mask: which proteins train a namespace (its train_pool)."""
    import numpy as np

    if pool == "all":
        return np.ones(len(proteins), dtype="float32")
    return np.array([1.0 if p.has_manual else 0.0 for p in proteins], dtype="float32")


def _masked_pos_weight(Y, mask):
    """neg/pos per term over the proteins that actually train this namespace."""
    import numpy as np

    rows = mask.astype(bool)
    Ym = Y[rows] if rows.any() else Y
    pos = Ym.sum(axis=0)
    neg = Ym.shape[0] - pos
    return np.clip(neg / np.clip(pos, 1, None), 0, POS_WEIGHT_CLAMP).astype("float32")


def _autocast(device, torch):
    """fp16 autocast on CUDA (T4 tensor cores); a no-op elsewhere."""
    import contextlib

    if str(device) == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _predict_all(model, proteins, tokenizer, device, max_length, batch_size):
    """One eval pass over `proteins` -> {ns: [P x N_ns] probabilities}."""
    import numpy as np
    import torch

    seqs = [p.sequence for p in proteins]
    parts = {ns: [] for ns in model.namespaces}
    model.eval()
    with torch.no_grad():
        for s in range(0, len(seqs), batch_size):
            enc = tokenizer(seqs[s:s + batch_size], return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length + 2).to(device)
            with _autocast(device, torch):
                logits = model(enc["input_ids"], enc["attention_mask"])
            for ns in model.namespaces:
                parts[ns].append(torch.sigmoid(logits[ns]).float().cpu().numpy())
    if not seqs:
        return {ns: np.zeros((0, model.heads[ns][-1].out_features), "float32")
                for ns in model.namespaces}
    return {ns: np.concatenate(parts[ns], axis=0).astype("float32") for ns in model.namespaces}


# --- training ---------------------------------------------------------------
def fit_finetune(model, tokenizer, train_prots, val_prots, vocabs, policy, device, hp):
    """Train the multi-task model end-to-end; early-stop on summed val Fmax.

    Returns (val_fmax_by_ns, epochs_run). `model` is left holding the best weights.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset

    from viral_annotation.classifier.losses import make_loss

    namespaces = model.namespaces
    Ytr = {ns: build_labels(train_prots, vocabs[ns], policy[ns]["train_field"]) for ns in namespaces}
    Yval = {ns: build_labels(val_prots, vocabs[ns], "terms_manual") for ns in namespaces}
    tr_mask = {ns: _pool_mask(train_prots, policy[ns]["train_pool"]) for ns in namespaces}
    crit = {ns: make_loss(hp.loss, pos_weight=_masked_pos_weight(Ytr[ns], tr_mask[ns]),
                          device=device, gamma_neg=FT_ASL_GAMMA_NEG,
                          gamma_pos=FT_ASL_GAMMA_POS, clip=FT_ASL_CLIP)
            for ns in namespaces}
    seqs = [p.sequence for p in train_prots]

    class _Idx(Dataset):
        def __len__(self): return len(seqs)
        def __getitem__(self, i): return i

    def collate(idxs):
        enc = tokenizer([seqs[i] for i in idxs], return_tensors="pt", padding=True,
                        truncation=True, max_length=hp.max_length + 2)
        ys = {ns: torch.from_numpy(Ytr[ns][idxs]) for ns in namespaces}
        ms = {ns: torch.from_numpy(tr_mask[ns][idxs]) for ns in namespaces}
        return enc, ys, ms

    loader = DataLoader(_Idx(), batch_size=hp.ft_batch, shuffle=True, collate_fn=collate)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.heads.parameters()) + list(model.attn.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": hp.ft_backbone_lr, "weight_decay": 0.0},
        {"params": head_params, "lr": hp.ft_head_lr, "weight_decay": TRAIN_WEIGHT_DECAY},
    ])
    scaler = torch.amp.GradScaler("cuda", enabled=str(device) == "cuda")
    accum = max(1, hp.ft_grad_accum)

    best, best_state, best_fmax, wait, epoch = -1.0, None, {}, 0, 0
    for epoch in range(1, hp.ft_epochs + 1):
        model.train()
        optimizer.zero_grad()
        pending = False
        for step, (enc, ys, ms) in enumerate(loader):
            enc = {k: v.to(device) for k, v in enc.items()}
            with _autocast(device, torch):
                logits = model(enc["input_ids"], enc["attention_mask"])
                loss = 0.0
                for ns in namespaces:
                    y, mask = ys[ns].to(device), ms[ns].to(device)        # [B,N], [B]
                    elem = crit[ns](logits[ns], y)                        # [B,N]
                    denom = (mask.sum() * y.shape[1]).clamp(min=1)
                    loss = loss + (elem * mask.unsqueeze(1)).sum() / denom
            scaler.scale(loss / accum).backward()
            pending = True
            if (step + 1) % accum == 0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
                pending = False
        if pending:
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

        probs = _predict_all(model, val_prots, tokenizer, device, hp.max_length, hp.ft_batch)
        fmax = {ns: fmax_matrix(probs[ns], Yval[ns]).fmax for ns in namespaces}
        score = float(np.mean(list(fmax.values()))) if fmax else 0.0
        if score > best + 1e-4:
            best, best_fmax, wait = score, fmax, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= TRAIN_EARLY_STOP_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_fmax, epoch


# --- orchestration: build vocab, train, wrap as per-namespace Heads ---------
def run_finetune(split, pools, policy, pooling_by_ns, dag, device, hp) -> tuple[dict, FinetuneArtifacts]:
    """Train one shared-backbone multi-task model and return per-namespace `Head`s
    (so the trainer's eval/report path is reused unchanged) + saveable artifacts."""
    import numpy as np

    # Per-namespace vocab from each namespace's own train pool (same as fit_namespace).
    vocabs, ns_specs = {}, []
    for ns in GO_NAMESPACES:
        vocab = select_vocab(pools[policy[ns]["train_pool"]], dag, hp.min_count,
                             field=policy[ns]["vocab_field"], namespaces=[ns])
        if len(vocab) == 0:
            continue
        vocabs[ns] = vocab
        ns_specs.append({"ns": ns, "num_terms": len(vocab), "pooling": pooling_by_ns[ns]})
    if not ns_specs:
        return {}, None

    train_prots = pools["all"]   # possibly capped by --train-pool-cap (bounds T4 time)
    backbone, tokenizer, dim = build_lora_backbone(hp.model_key)
    model = build_multitask_model(backbone, dim, ns_specs, n_heads=hp.heads,
                                  hidden_dims=hp.ft_hidden, dropout=hp.ft_dropout).to(device)

    val_fmax, epochs = fit_finetune(model, tokenizer, train_prots, split.val,
                                    vocabs, policy, device, hp)

    artifacts = FinetuneArtifacts(
        model=model, tokenizer=tokenizer, vocabs=vocabs,
        pooling={s["ns"]: s["pooling"] for s in ns_specs}, model_key=hp.model_key,
        n_heads=hp.heads, hidden_dims=list(hp.ft_hidden), dropout=hp.ft_dropout,
        max_length=hp.max_length)

    # One backbone pass per distinct protein set, shared across the three Heads.
    cache: dict = {}

    def make_predict(ns):
        def predict(prots):
            key = id(prots)
            if key not in cache:
                cache[key] = _predict_all(model, prots, tokenizer, device,
                                          hp.max_length, hp.ft_batch)
            return cache[key][ns]
        return predict

    heads = {}
    for ns in vocabs:
        mask = _pool_mask(train_prots, policy[ns]["train_pool"]).astype(bool)
        Ytr = build_labels(train_prots, vocabs[ns], policy[ns]["train_field"])
        prior = Ytr[mask].mean(axis=0) if mask.any() else Ytr.mean(axis=0)
        heads[ns] = Head(vocab=vocabs[ns], predict=make_predict(ns), prior=prior,
                         val_fmax=val_fmax.get(ns, 0.0), epochs=epochs,
                         pooling=pooling_by_ns[ns], state=None)
    return heads, artifacts


# --- persistence ------------------------------------------------------------
def save_finetuned(artifacts: FinetuneArtifacts, out_dir, test_rows):
    """Persist adapter + heads/poolers + meta under `out_dir` (e.g. models/bacterial/finetuned)."""
    import json

    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts.model.backbone.save_pretrained(str(out_dir / "adapter"))
    torch.save({"heads": artifacts.model.heads.state_dict(),
                "attn": artifacts.model.attn.state_dict()}, out_dir / "heads.pt")

    res_by_ns = {ns: res for ns, res, _nv in test_rows}
    naive_by_ns = {ns: nv for ns, _res, nv in test_rows}
    meta = {
        "esm_model": artifacts.model_key,
        "finetune": "lora",
        "n_heads": artifacts.n_heads,
        "hidden_dims": artifacts.hidden_dims,
        "dropout": artifacts.dropout,
        "max_length": artifacts.max_length,
        "namespaces": {
            ns: {
                "pooling": artifacts.pooling[ns],
                "fmax": res_by_ns[ns].fmax if ns in res_by_ns else None,
                "naive_fmax": naive_by_ns[ns].fmax if ns in naive_by_ns else None,
                "terms": artifacts.vocabs[ns].terms,
            }
            for ns in artifacts.vocabs
        },
    }
    (out_dir / "finetuned.meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def load_finetuned(model_dir, device="cpu"):
    """Rebuild a trained fine-tuned model from `save_finetuned` output.

    Returns (model, tokenizer, meta). The backbone loads the base ESM-2 + LoRA
    adapter; heads/poolers load from heads.pt.
    """
    import json

    import torch
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer

    meta = json.loads((model_dir / "finetuned.meta.json").read_text())
    spec = ESM2_MODELS[meta["esm_model"]]
    base = AutoModel.from_pretrained(spec.hf_name)
    backbone = PeftModel.from_pretrained(base, str(model_dir / "adapter"))
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)

    ns_specs = [{"ns": ns, "num_terms": len(info["terms"]), "pooling": info["pooling"]}
                for ns, info in meta["namespaces"].items()]
    model = build_multitask_model(backbone, spec.dim, ns_specs, n_heads=meta["n_heads"],
                                  hidden_dims=meta["hidden_dims"], dropout=meta["dropout"])
    state = torch.load(model_dir / "heads.pt", map_location="cpu")
    model.heads.load_state_dict(state["heads"])
    model.attn.load_state_dict(state["attn"])
    return model.to(device).eval(), tokenizer, meta
