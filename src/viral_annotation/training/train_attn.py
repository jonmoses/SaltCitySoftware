"""Per-namespace training with multi-head ATTENTION POOLING (learned per-residue).

Parallel to training.train (which uses fixed pooled features). Here the pooling is
learned: per-residue ESM embeddings are cached once, then an attention pooler +
classifier head are trained jointly per GO namespace. Same data discipline as
train.py — per-namespace evidence policy, cluster split, Coronaviridae holdout,
manual-only eval, Naive baseline, seeded.

Run:  python -m viral_annotation.training.train_attn [--limit N] [--heads H] ...
Use --limit for the subset validation before paying the full per-residue embed.
"""

from __future__ import annotations

import argparse
import time

from viral_annotation.config import (
    DEFAULT_ESM_MODEL,
    ESM2_MODELS,
    GO_NAMESPACES,
    GO_OBO_PATH,
    HOLDOUT_FAMILY,
    MIN_TERM_COUNT,
    NAMESPACE_POLICY,
    POS_WEIGHT_CLAMP,
    TRAIN_EARLY_STOP_PATIENCE,
    TRAIN_LR,
    TRAIN_SEED,
    TRAIN_WEIGHT_DECAY,
)
from viral_annotation.classifier.pooling import build_attn_classifier
from viral_annotation.data import labels as labels_mod
from viral_annotation.data.cluster import cluster_sequences
from viral_annotation.data.dataset import build_labels, select_vocab
from viral_annotation.data.split import cluster_split, split_proteins
from viral_annotation.embeddings.residue_cache import cache_residues, load_residues, residue_cache_dir
from viral_annotation.evaluation.metrics import apply_hierarchical_correction, fmax_matrix
from viral_annotation.ontology import GoDag
from viral_annotation.training.train import NS_SHORT, _annotation_stats, _auto_device

MAX_RESIDUES = 2048   # cap residues per protein for attention (bounds batch memory)
TRAIN_POOL_CAP = 6000  # cap train proteins per head — attention reads every residue
                       # file each epoch, so the full 15k all-pool is I/O-bound


def _cap_pool(proteins, cap, seed):
    """Subsample a train pool to `cap`, keeping ALL manual-having proteins and a
    seeded sample of the IEA-only remainder. Bounds per-epoch disk reads."""
    import random

    if len(proteins) <= cap:
        return proteins
    manual = [p for p in proteins if p.has_manual]
    iea = [p for p in proteins if not p.has_manual]
    random.Random(seed).shuffle(iea)
    return manual + iea[: max(0, cap - len(manual))]


def _make_dataset(proteins, Y, cache_dir, max_residues):
    import numpy as np
    from torch.utils.data import Dataset

    class ResidueDataset(Dataset):
        def __len__(self):
            return len(proteins)

        def __getitem__(self, i):
            X = load_residues(proteins[i].accession, cache_dir)[:max_residues]
            return X, Y[i]

    return ResidueDataset()


def _collate(batch):
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


def _predict(model, dataset, device, batch_size):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    model.eval()
    out = []
    with torch.no_grad():
        for X, mask, _y in dl:
            p = torch.sigmoid(model(X.to(device), mask.to(device)))
            out.append(p.cpu().numpy())
    return np.concatenate(out, axis=0).astype("float32")


def _train_head(train_ds, val_ds, Yval, input_dim, num_terms, pos_weight, *,
                n_heads, hidden_dims, epochs, lr, batch_size, device, patience):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    model = build_attn_classifier(input_dim, num_terms, n_heads=n_heads,
                                  hidden_dims=hidden_dims).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=TRAIN_WEIGHT_DECAY)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate)

    best, best_state, wait, ep = -1.0, None, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        for X, mask, Y in loader:
            optimizer.zero_grad()
            loss = criterion(model(X.to(device), mask.to(device)), Y.to(device))
            loss.backward()
            optimizer.step()
        vf = fmax_matrix(_predict(model, val_ds, device, batch_size), Yval).fmax
        if vf > best + 1e-4:
            best, wait = vf, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best, ep


def run(limit=None, model_key=DEFAULT_ESM_MODEL, repr_layer=None, min_count=MIN_TERM_COUNT,
        n_heads=8, hidden_dims=None, epochs=100, lr=TRAIN_LR, batch_size=16,
        max_residues=MAX_RESIDUES, train_pool_cap=TRAIN_POOL_CAP,
        holdout_family=HOLDOUT_FAMILY, use_cluster=True):
    import numpy as np
    import torch

    t0 = time.time()
    torch.manual_seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    device = _auto_device(torch)
    input_dim = ESM2_MODELS[model_key].dim
    print(f"[1/5] device={device} | attention pooling (heads={n_heads}) | loading GO DAG …")
    dag = GoDag.from_obo(GO_OBO_PATH)

    print(f"[2/5] fetching viral reviewed proteins (limit={limit}) …")
    proteins = [p for p in labels_mod.label_proteins(list(labels_mod.fetch_raw(limit=limit)), dag)
                if p.sequence]
    print(f"       {_annotation_stats(proteins)}")

    if use_cluster:
        print(f"[3/5] cluster split (holdout {holdout_family}) …")
        clusters = cluster_sequences(proteins)
        split = cluster_split(proteins, clusters, holdout_family=holdout_family)
    else:
        split = split_proteins(proteins)
    manual_train = [p for p in split.train if p.has_manual]
    print(f"       {split.summary()} | manual-having train {len(manual_train)}")
    if not split.val or not split.test:
        raise SystemExit("val/test empty — increase --limit.")

    print("[4/5] caching per-residue embeddings (fp16) …")
    pool_prots = {"all": split.train, "manual_having": manual_train}
    eval_sets = [split.val, split.test] + ([split.holdout] if split.holdout else [])
    for group in list(pool_prots.values()) + eval_sets:
        cache_residues(group, model_key, repr_layer)
    cache_dir = residue_cache_dir(model_key, repr_layer)

    print(f"[5/5] training per-namespace attention heads …")
    heads = {}
    combo = {"test": ([], [], []), "zero": ([], [], [])}  # (prob, true, naive) per split
    for ns in GO_NAMESPACES:
        pol = NAMESPACE_POLICY[ns]
        train_prots = _cap_pool(pool_prots[pol["train_pool"]], train_pool_cap, TRAIN_SEED)
        vocab = select_vocab(train_prots, dag, min_count, field=pol["vocab_field"], namespaces=[ns])
        if len(vocab) == 0:
            print(f"       {NS_SHORT[ns]}: empty vocab — skipped")
            continue
        Ytr = build_labels(train_prots, vocab, pol["train_field"])
        Yval = build_labels(split.val, vocab, "terms_manual")
        Yte = build_labels(split.test, vocab, "terms_manual")
        pos = Ytr.sum(axis=0)
        pw = np.clip((Ytr.shape[0] - pos) / np.clip(pos, 1, None), 0, POS_WEIGHT_CLAMP).astype("float32")

        train_ds = _make_dataset(train_prots, Ytr, cache_dir, max_residues)
        val_ds = _make_dataset(split.val, Yval, cache_dir, max_residues)
        test_ds = _make_dataset(split.test, Yte, cache_dir, max_residues)
        model, vbest, ep = _train_head(
            train_ds, val_ds, Yval, input_dim, len(vocab), pw, n_heads=n_heads,
            hidden_dims=hidden_dims, epochs=epochs, lr=lr, batch_size=batch_size,
            device=device, patience=TRAIN_EARLY_STOP_PATIENCE)

        prob_te = apply_hierarchical_correction(_predict(model, test_ds, device, batch_size), vocab, dag)
        res = fmax_matrix(prob_te, Yte)
        naive_te = np.tile(Ytr.mean(axis=0), (Yte.shape[0], 1))
        naive_res = fmax_matrix(naive_te, Yte)
        combo["test"][0].append(prob_te); combo["test"][1].append(Yte); combo["test"][2].append(naive_te)
        heads[ns] = {"vocab": vocab, "model": model, "prior": Ytr.mean(axis=0),
                     "result": res, "naive": naive_res}

        zline = ""
        if split.holdout:
            Yho = build_labels(split.holdout, vocab, "terms_manual")
            ho_ds = _make_dataset(split.holdout, Yho, cache_dir, max_residues)
            p = apply_hierarchical_correction(_predict(model, ho_ds, device, batch_size), vocab, dag)
            nv = np.tile(heads[ns]["prior"], (Yho.shape[0], 1))
            heads[ns]["zeroshot"] = fmax_matrix(p, Yho)
            heads[ns]["zeroshot_naive"] = fmax_matrix(nv, Yho)
            combo["zero"][0].append(p); combo["zero"][1].append(Yho); combo["zero"][2].append(nv)
            zline = (f" | zero-shot {heads[ns]['zeroshot'].fmax:.4f} "
                     f"(naive {heads[ns]['zeroshot_naive'].fmax:.4f})")

        # Report each namespace immediately (test + zero-shot), so a slow later
        # head doesn't hide the result we care about.
        print(f"       {NS_SHORT[ns]}: N={len(vocab):4d} train={len(train_prots)} "
              f"test Fmax={res.fmax:.4f} (naive {naive_res.fmax:.4f}){zline} "
              f"val={vbest:.3f} ep={ep}", flush=True)

    def _report(title, key):
        if not combo[key][0]:
            return
        ov = fmax_matrix(np.concatenate(combo[key][0], 1), np.concatenate(combo[key][1], 1))
        nv = fmax_matrix(np.concatenate(combo[key][2], 1), np.concatenate(combo[key][1], 1))
        print(f"\n=== {title} ===")
        print(f"  {'namespace':20s} {'Fmax':>7s} {'naive':>7s} {'lift':>7s}")
        for ns in GO_NAMESPACES:
            if ns in heads:
                r = heads[ns]["result"] if key == "test" else heads[ns].get("zeroshot")
                n = heads[ns]["naive"] if key == "test" else heads[ns].get("zeroshot_naive")
                if r:
                    print(f"  {ns:20s} {r.fmax:7.4f} {n.fmax:7.4f} {r.fmax - n.fmax:+7.4f}")
        print(f"  {'overall':20s} {ov.fmax:7.4f} {nv.fmax:7.4f} {ov.fmax - nv.fmax:+7.4f}")

    _report("TEST (manual-only, hierarchically corrected)", "test")
    if split.holdout:
        _report(f"ZERO-SHOT — held-out {holdout_family} ({len(split.holdout)} proteins)", "zero")
    print(f"\n[done] elapsed {time.time() - t0:.1f}s")
    return heads


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train viral GO classifier with attention pooling.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=TRAIN_LR)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-residues", type=int, default=MAX_RESIDUES)
    ap.add_argument("--train-pool-cap", type=int, default=TRAIN_POOL_CAP)
    ap.add_argument("--min-count", type=int, default=MIN_TERM_COUNT)
    ap.add_argument("--hidden", type=int, nargs="*", default=None)
    ap.add_argument("--random-split", action="store_true")
    ap.add_argument("--holdout-family", default=HOLDOUT_FAMILY)
    args = ap.parse_args(argv)
    run(limit=args.limit, n_heads=args.heads, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, max_residues=args.max_residues,
        train_pool_cap=args.train_pool_cap, min_count=args.min_count,
        hidden_dims=args.hidden, use_cluster=not args.random_split,
        holdout_family=args.holdout_family or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
