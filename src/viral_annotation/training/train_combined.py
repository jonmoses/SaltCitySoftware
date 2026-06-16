"""Per-namespace pooling: the synthesized production trainer.

Each GO namespace trains with the pooling its policy specifies (config.NAMESPACE_POLICY):
  * MF    -> learned multi-head ATTENTION pooling (per-residue) — wins zero-shot MF
  * BP/CC -> MEAN pooling (fixed, cached) — attention doesn't help, and is costly
Everything else matches the other trainers: per-namespace evidence policy, cluster
split, Coronaviridae holdout, manual-only eval, Naive baseline, seeded. Reuses the
linear trainer (train._train_head) and the attention trainer (train_attn._train_head).

Run:  python -m viral_annotation.training.train_combined [--limit N] ...
With the mean + per-residue caches already built, this is training-only (~minutes).
"""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

from viral_annotation.config import (
    DEFAULT_ESM_MODEL,
    ESM2_MODELS,
    GO_NAMESPACES,
    GO_OBO_PATH,
    HOLDOUT_FAMILY,
    MIN_TERM_COUNT,
    NAMESPACE_POLICY,
    POS_WEIGHT_CLAMP,
    TRAIN_BATCH_SIZE,
    TRAIN_EARLY_STOP_PATIENCE,
    TRAIN_EPOCHS,
    TRAIN_LR,
    TRAIN_SEED,
)
from viral_annotation.classifier.model import predict_proba
from viral_annotation.data import labels as labels_mod
from viral_annotation.data.cluster import cluster_sequences
from viral_annotation.data.dataset import build_labels, select_vocab
from viral_annotation.data.split import cluster_split, split_proteins
from viral_annotation.embeddings.cache import embed_records
from viral_annotation.embeddings.esm import ESMEmbedder
from viral_annotation.embeddings.residue_cache import cache_residues, residue_cache_dir
from viral_annotation.evaluation.metrics import apply_hierarchical_correction, fmax_matrix
from viral_annotation.ontology import GoDag
from viral_annotation.training.train import NS_SHORT, _annotation_stats, _auto_device
from viral_annotation.training.train import _train_head as _train_linear
from viral_annotation.training.train_attn import (
    MAX_RESIDUES,
    _make_dataset,
    _predict,
)
from viral_annotation.training.train_attn import _train_head as _train_attn


def _fit_namespace(ns, pol, split, pool_prots, dag, device, cache_dir, hp):
    """Train one namespace head per its pooling policy; return a predict() closure."""
    import numpy as np

    pooling = pol["pooling"]
    train_prots = pool_prots[pol["train_pool"]]
    vocab = select_vocab(train_prots, dag, hp.min_count,
                         field=pol["vocab_field"], namespaces=[ns])
    if len(vocab) == 0:
        return None
    Ytr = build_labels(train_prots, vocab, pol["train_field"])
    Yval = build_labels(split.val, vocab, "terms_manual")

    if pooling == "attention":
        pos = Ytr.sum(axis=0)
        pw = np.clip((Ytr.shape[0] - pos) / np.clip(pos, 1, None), 0, POS_WEIGHT_CLAMP).astype("float32")
        train_ds = _make_dataset(train_prots, Ytr, cache_dir, hp.max_residues)
        val_ds = _make_dataset(split.val, Yval, cache_dir, hp.max_residues)
        model, vbest, ep = _train_attn(
            train_ds, val_ds, Yval, hp.input_dim, len(vocab), pw, n_heads=hp.heads,
            hidden_dims=hp.hidden, epochs=hp.attn_epochs, lr=hp.lr,
            batch_size=hp.attn_batch, device=device, patience=TRAIN_EARLY_STOP_PATIENCE)

        def predict(prots):
            ds = _make_dataset(prots, np.zeros((len(prots), len(vocab)), "float32"),
                               cache_dir, hp.max_residues)
            return _predict(model, ds, device, hp.attn_batch)
    else:  # mean / stats — fixed pooling on cached feature vectors
        embedder = ESMEmbedder(model_key=hp.model_key, pooling=pooling, window=True)
        Xtr = embed_records(train_prots, hp.model_key, pooling, None, embedder=embedder)[1]
        _, Xva = embed_records(split.val, hp.model_key, pooling, None, embedder=embedder)
        model, vbest, ep = _train_linear(
            Xtr, Ytr, Xva, Yval, hidden_dims=hp.hidden, epochs=hp.epochs, lr=hp.lr,
            batch_size=hp.batch, device=device, patience=TRAIN_EARLY_STOP_PATIENCE)

        def predict(prots):
            _, X = embed_records(prots, hp.model_key, pooling, None, embedder=embedder)
            return predict_proba(model, X)

    return {"vocab": vocab, "predict": predict, "prior": Ytr.mean(axis=0),
            "vbest": vbest, "ep": ep, "pooling": pooling}


def run(limit=None, model_key=DEFAULT_ESM_MODEL, min_count=MIN_TERM_COUNT, hidden_dims=None,
        epochs=TRAIN_EPOCHS, attn_epochs=100, lr=TRAIN_LR, batch_size=TRAIN_BATCH_SIZE,
        attn_batch=16, heads=8, max_residues=MAX_RESIDUES, holdout_family=HOLDOUT_FAMILY,
        use_cluster=True):
    import numpy as np
    import torch

    t0 = time.time()
    torch.manual_seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    device = _auto_device(torch)
    hp = SimpleNamespace(model_key=model_key, min_count=min_count, hidden=hidden_dims,
                         epochs=epochs, attn_epochs=attn_epochs, lr=lr, batch=batch_size,
                         attn_batch=attn_batch, heads=heads, max_residues=max_residues,
                         input_dim=ESM2_MODELS[model_key].dim)
    pooled = {ns: NAMESPACE_POLICY[ns]["pooling"] for ns in GO_NAMESPACES}
    print(f"[1/5] device={device} | per-namespace pooling {pooled} | loading GO DAG …")
    dag = GoDag.from_obo(GO_OBO_PATH)

    print(f"[2/5] fetching viral reviewed proteins (limit={limit}) …")
    proteins = [p for p in labels_mod.label_proteins(list(labels_mod.fetch_raw(limit=limit)), dag)
                if p.sequence]
    print(f"       {_annotation_stats(proteins)}")

    if use_cluster:
        clusters = cluster_sequences(proteins)
        split = cluster_split(proteins, clusters, holdout_family=holdout_family)
    else:
        split = split_proteins(proteins)
    pool_prots = {"all": split.train, "manual_having": [p for p in split.train if p.has_manual]}
    print(f"[3/5] {split.summary()} | manual-having train {len(pool_prots['manual_having'])}")
    if not split.val or not split.test:
        raise SystemExit("val/test empty — increase --limit.")

    # Per-residue cache only for the pools attention namespaces actually use.
    cache_dir = None
    attn_pools = {NAMESPACE_POLICY[ns]["train_pool"] for ns in GO_NAMESPACES
                  if NAMESPACE_POLICY[ns]["pooling"] == "attention"}
    if attn_pools:
        print("[4/5] caching per-residue embeddings for attention namespaces …")
        groups = [pool_prots[p] for p in attn_pools] + [split.val, split.test]
        groups += [split.holdout] if split.holdout else []
        for g in groups:
            cache_residues(g, model_key, None)
        cache_dir = residue_cache_dir(model_key, None)

    print("[5/5] training per-namespace heads …")
    combo = {"test": ([], [], []), "zero": ([], [], [])}
    heads_out = {}
    for ns in GO_NAMESPACES:
        fit = _fit_namespace(ns, NAMESPACE_POLICY[ns], split, pool_prots, dag, device, cache_dir, hp)
        if fit is None:
            print(f"       {NS_SHORT[ns]}: empty vocab — skipped")
            continue
        vocab = fit["vocab"]
        Yte = build_labels(split.test, vocab, "terms_manual")
        prob_te = apply_hierarchical_correction(fit["predict"](split.test), vocab, dag)
        res = fmax_matrix(prob_te, Yte)
        naive_te = np.tile(fit["prior"], (Yte.shape[0], 1))
        nres = fmax_matrix(naive_te, Yte)
        combo["test"][0].append(prob_te); combo["test"][1].append(Yte); combo["test"][2].append(naive_te)
        heads_out[ns] = {"result": res, "naive": nres}

        zline = ""
        if split.holdout:
            Yho = build_labels(split.holdout, vocab, "terms_manual")
            prob_ho = apply_hierarchical_correction(fit["predict"](split.holdout), vocab, dag)
            naive_ho = np.tile(fit["prior"], (Yho.shape[0], 1))
            zr = fmax_matrix(prob_ho, Yho); zn = fmax_matrix(naive_ho, Yho)
            combo["zero"][0].append(prob_ho); combo["zero"][1].append(Yho); combo["zero"][2].append(naive_ho)
            heads_out[ns]["zeroshot"] = zr
            zline = f" | zero-shot {zr.fmax:.4f} (naive {zn.fmax:.4f})"
        print(f"       {NS_SHORT[ns]} [{fit['pooling']}]: N={len(vocab):4d} "
              f"test {res.fmax:.4f} (naive {nres.fmax:.4f}){zline} val={fit['vbest']:.3f} ep={fit['ep']}",
              flush=True)

    def _report(title, key):
        if not combo[key][0]:
            return
        true_cat = np.concatenate(combo[key][1], axis=1)
        ov = fmax_matrix(np.concatenate(combo[key][0], axis=1), true_cat)
        nv = fmax_matrix(np.concatenate(combo[key][2], axis=1), true_cat)
        print(f"\n=== {title} ===")
        print(f"  {'overall':20s} Fmax={ov.fmax:.4f}  naive={nv.fmax:.4f}  lift={ov.fmax - nv.fmax:+.4f}")

    _report("TEST (manual-only, hierarchically corrected)", "test")
    if split.holdout:
        _report(f"ZERO-SHOT — held-out {holdout_family}", "zero")
    print(f"\n[done] elapsed {time.time() - t0:.1f}s")
    return heads_out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train viral GO classifier, pooling per namespace.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    ap.add_argument("--attn-epochs", type=int, default=100)
    ap.add_argument("--min-count", type=int, default=MIN_TERM_COUNT)
    ap.add_argument("--random-split", action="store_true")
    ap.add_argument("--holdout-family", default=HOLDOUT_FAMILY)
    args = ap.parse_args(argv)
    run(limit=args.limit, heads=args.heads, epochs=args.epochs, attn_epochs=args.attn_epochs,
        min_count=args.min_count, use_cluster=not args.random_split,
        holdout_family=args.holdout_family or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
