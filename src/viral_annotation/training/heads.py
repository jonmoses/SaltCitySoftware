"""Per-namespace classifier heads — the two ways we turn ESM features into GO scores.

  * fit_pooled_head     — fixed pooled vectors (mean/stats) -> linear/MLP head.
  * fit_attention_head  — per-residue embeddings -> learned attention pooler + head.

`fit_namespace` picks one per the requested pooling and returns a uniform `Head`
(vocab + a predict() closure + the training prior), so the trainer treats both the
same. This is the single home for what used to be split across train.py (pooled),
train_attn.py (attention), and train_combined.py (the dispatch).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from viral_annotation.classifier.model import build_classifier, predict_proba
from viral_annotation.classifier.pooling import build_attn_classifier
from viral_annotation.config import (
    POS_WEIGHT_CLAMP,
    TRAIN_EARLY_STOP_PATIENCE,
    TRAIN_SEED,
    TRAIN_WEIGHT_DECAY,
)
from viral_annotation.data.dataset import build_labels, select_vocab
from viral_annotation.embeddings.cache import embed_records
from viral_annotation.embeddings.esm import ESMEmbedder
from viral_annotation.embeddings.residue_cache import load_residues
from viral_annotation.evaluation.metrics import fmax_matrix


@dataclass
class Head:
    """A trained namespace head: its vocab, a predict() closure, and the prior.

    `predict(proteins) -> [P x N]` probabilities over `vocab`. `state` is the
    head's torch state_dict for pooled heads (serializable); None for attention
    heads, which need the per-residue cache at inference and aren't served.
    """

    vocab: object
    predict: Callable
    prior: object
    val_fmax: float
    epochs: int
    pooling: str
    state: dict | None = None


def compute_pos_weight(Ytr):
    """Per-term BCE positive weight = neg/pos, clamped — counters class imbalance."""
    import numpy as np

    pos = Ytr.sum(axis=0)
    neg = Ytr.shape[0] - pos
    return np.clip(neg / np.clip(pos, 1, None), 0, POS_WEIGHT_CLAMP).astype("float32")


# --- pooled (fixed-feature) head -------------------------------------------
def fit_pooled_head(Xtr, Ytr, Xva, Yva, *, hidden_dims, epochs, lr, batch_size,
                    device, patience, pos_weight=None):
    """Train a linear/MLP head on fixed pooled features; early-stop on val Fmax.

    Returns (best_model, best_val_fmax, epochs_run).
    """
    import torch
    from torch import nn

    if pos_weight is None:
        pos_weight = compute_pos_weight(Ytr)

    model = build_classifier(Xtr.shape[1], Ytr.shape[1], hidden_dims=hidden_dims).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=TRAIN_WEIGHT_DECAY)
    Xt = torch.tensor(Xtr, device=device)
    Yt = torch.tensor(Ytr, device=device)
    n = Xt.shape[0]

    best, best_state, wait, epoch = -1.0, None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            optimizer.zero_grad()
            criterion(model(Xt[idx]), Yt[idx]).backward()
            optimizer.step()
        val_fmax = fmax_matrix(predict_proba(model, Xva), Yva).fmax
        if val_fmax > best + 1e-4:
            best, wait = val_fmax, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best, epoch


# --- attention (per-residue) head ------------------------------------------
def make_residue_dataset(proteins, Y, cache_dir, max_residues):
    """A torch Dataset yielding (per-residue embedding [<=max_residues x d], label)."""
    from torch.utils.data import Dataset

    class ResidueDataset(Dataset):
        def __len__(self):
            return len(proteins)

        def __getitem__(self, i):
            X = load_residues(proteins[i].accession, cache_dir)[:max_residues]
            return X, Y[i]

    return ResidueDataset()


def collate_residues(batch):
    """Pad a batch of variable-length residue tensors and build the validity mask."""
    import numpy as np
    import torch

    Xs, ys = zip(*batch)
    d = Xs[0].shape[1]
    L = max(x.shape[0] for x in Xs)
    Xp = np.zeros((len(Xs), L, d), dtype="float32")
    mask = np.zeros((len(Xs), L), dtype="float32")
    for k, x in enumerate(Xs):
        Xp[k, : x.shape[0]] = x
        mask[k, : x.shape[0]] = 1.0
    return torch.from_numpy(Xp), torch.from_numpy(mask), torch.from_numpy(np.stack(ys).astype("float32"))


def predict_residues(model, dataset, device, batch_size):
    """Run an attention head over a residue dataset -> [P x N] probabilities."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_residues)
    model.eval()
    out = []
    with torch.no_grad():
        for X, mask, _y in dl:
            p = torch.sigmoid(model(X.to(device), mask.to(device)))
            out.append(p.cpu().numpy())
    return np.concatenate(out, axis=0).astype("float32")


def fit_attention_head(train_ds, val_ds, Yval, input_dim, num_terms, pos_weight, *,
                       n_heads, hidden_dims, epochs, lr, batch_size, device, patience):
    """Train an attention pooler + head jointly; early-stop on val Fmax."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    model = build_attn_classifier(input_dim, num_terms, n_heads=n_heads,
                                  hidden_dims=hidden_dims).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=TRAIN_WEIGHT_DECAY)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_residues)

    best, best_state, wait, epoch = -1.0, None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for X, mask, Y in loader:
            optimizer.zero_grad()
            criterion(model(X.to(device), mask.to(device)), Y.to(device)).backward()
            optimizer.step()
        val_fmax = fmax_matrix(predict_residues(model, val_ds, device, batch_size), Yval).fmax
        if val_fmax > best + 1e-4:
            best, wait = val_fmax, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best, epoch


def cap_pool(proteins, cap, seed):
    """Subsample a train pool to `cap`, keeping ALL manual-having proteins plus a
    seeded sample of the IEA-only remainder. Bounds attention's per-epoch disk reads."""
    import random

    if cap is None or len(proteins) <= cap:
        return proteins
    manual = [p for p in proteins if p.has_manual]
    iea = [p for p in proteins if not p.has_manual]
    random.Random(seed).shuffle(iea)
    return manual + iea[: max(0, cap - len(manual))]


# --- the dispatch -----------------------------------------------------------
def fit_namespace(ns, policy, pooling, split, pools, dag, device, cache_dir, hp) -> Head | None:
    """Train one namespace's head with the requested pooling; return a Head (or None
    if the namespace has no terms above the frequency floor)."""
    import numpy as np

    train_prots = pools[policy["train_pool"]]
    if pooling == "attention":
        train_prots = cap_pool(train_prots, getattr(hp, "train_pool_cap", None), TRAIN_SEED)

    vocab = select_vocab(train_prots, dag, hp.min_count,
                         field=policy["vocab_field"], namespaces=[ns])
    if len(vocab) == 0:
        return None
    Ytr = build_labels(train_prots, vocab, policy["train_field"])
    Yval = build_labels(split.val, vocab, "terms_manual")
    pos_weight = compute_pos_weight(Ytr)

    if pooling == "attention":
        train_ds = make_residue_dataset(train_prots, Ytr, cache_dir, hp.max_residues)
        val_ds = make_residue_dataset(split.val, Yval, cache_dir, hp.max_residues)
        model, val_fmax, epochs = fit_attention_head(
            train_ds, val_ds, Yval, hp.input_dim, len(vocab), pos_weight,
            n_heads=hp.heads, hidden_dims=hp.hidden, epochs=hp.attn_epochs, lr=hp.lr,
            batch_size=hp.attn_batch, device=device, patience=TRAIN_EARLY_STOP_PATIENCE)

        def predict(prots):
            ds = make_residue_dataset(prots, np.zeros((len(prots), len(vocab)), "float32"),
                                      cache_dir, hp.max_residues)
            return predict_residues(model, ds, device, hp.attn_batch)

        state = None
    else:
        embedder = ESMEmbedder(model_key=hp.model_key, pooling=pooling, window=True)
        Xtr = embed_records(train_prots, hp.model_key, pooling, None, embedder=embedder)[1]
        _, Xva = embed_records(split.val, hp.model_key, pooling, None, embedder=embedder)
        model, val_fmax, epochs = fit_pooled_head(
            Xtr, Ytr, Xva, Yval, hidden_dims=hp.hidden, epochs=hp.epochs, lr=hp.lr,
            batch_size=hp.batch, device=device, patience=TRAIN_EARLY_STOP_PATIENCE,
            pos_weight=pos_weight)

        def predict(prots):
            _, X = embed_records(prots, hp.model_key, pooling, None, embedder=embedder)
            return predict_proba(model, X)

        state = model.state_dict()

    return Head(vocab=vocab, predict=predict, prior=Ytr.mean(axis=0),
                val_fmax=val_fmax, epochs=epochs, pooling=pooling, state=state)
