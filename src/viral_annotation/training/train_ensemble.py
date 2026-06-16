"""Ensemble eval: fuse pLM heads with homology (BLAST-KNN) + InterPro (IEA proxy).

Targets the zero-shot BP/CC gap. For each namespace it trains the pLM head
(reusing train_combined._fit_namespace), builds three [P x N] score matrices
(pLM / homology / InterPro), grid-searches fusion weights on validation, and
reports pLM-only vs ensemble on test and zero-shot — against the Naive baseline.

Run:  python -m viral_annotation.training.train_ensemble [--limit N] ...
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
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_LR,
    TRAIN_SEED,
)
from viral_annotation.classifier.ensemble import fuse, search_weights
from viral_annotation.data import labels as labels_mod
from viral_annotation.data.cluster import cluster_sequences
from viral_annotation.data.dataset import build_labels
from viral_annotation.data.homology import homology_scores
from viral_annotation.data.split import cluster_split, split_proteins
from viral_annotation.embeddings.residue_cache import cache_residues, residue_cache_dir
from viral_annotation.evaluation.metrics import apply_hierarchical_correction, fmax_matrix
from viral_annotation.ontology import GoDag
from viral_annotation.training.train import NS_SHORT, _annotation_stats, _auto_device
from viral_annotation.training.train_combined import _fit_namespace

NS_FULL = {"molecular_function", "biological_process", "cellular_component"}


def run(limit=None, model_key=DEFAULT_ESM_MODEL, min_count=MIN_TERM_COUNT, heads=8,
        epochs=TRAIN_EPOCHS, attn_epochs=100, holdout_family=HOLDOUT_FAMILY, use_cluster=True):
    import numpy as np
    import torch

    t0 = time.time()
    torch.manual_seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    device = _auto_device(torch)
    hp = SimpleNamespace(model_key=model_key, min_count=min_count, hidden=None,
                         epochs=epochs, attn_epochs=attn_epochs, lr=TRAIN_LR,
                         batch=TRAIN_BATCH_SIZE, attn_batch=16, heads=heads,
                         max_residues=2048, input_dim=ESM2_MODELS[model_key].dim)
    print(f"[1/4] device={device} | loading GO DAG …")
    dag = GoDag.from_obo(GO_OBO_PATH)

    print(f"[2/4] fetching viral reviewed proteins (limit={limit}) …")
    proteins = [p for p in labels_mod.label_proteins(list(labels_mod.fetch_raw(limit=limit)), dag)
                if p.sequence]
    print(f"       {_annotation_stats(proteins)}")
    if use_cluster:
        split = cluster_split(proteins, cluster_sequences(proteins), holdout_family=holdout_family)
    else:
        split = split_proteins(proteins)
    pool_prots = {"all": split.train, "manual_having": [p for p in split.train if p.has_manual]}
    db = pool_prots["manual_having"]   # homology transfers these proteins' manual labels
    print(f"[3/4] {split.summary()} | homology db (manual-having train) {len(db)}")
    if not split.val or not split.test:
        raise SystemExit("val/test empty — increase --limit.")

    cache_dir = None
    if any(NAMESPACE_POLICY[ns]["pooling"] == "attention" for ns in GO_NAMESPACES):
        for g in [pool_prots["manual_having"], split.val, split.test] + \
                 ([split.holdout] if split.holdout else []):
            cache_residues(g, model_key, None)
        cache_dir = residue_cache_dir(model_key, None)

    print("[4/4] per-namespace: train pLM, build components, fuse …")
    acc = {"test": {"plm": [], "ens": [], "true": [], "naive": []},
           "zero": {"plm": [], "ens": [], "true": [], "naive": []}}
    for ns in GO_NAMESPACES:
        fit = _fit_namespace(ns, NAMESPACE_POLICY[ns], split, pool_prots, dag, device, cache_dir, hp)
        if fit is None:
            print(f"       {NS_SHORT[ns]}: empty vocab — skipped")
            continue
        vocab = fit["vocab"]

        def comps(prots):
            return {
                "plm": apply_hierarchical_correction(fit["predict"](prots), vocab, dag),
                "homology": homology_scores(prots, db, dag, vocab),
                "interpro": build_labels(prots, vocab, "terms_iea"),
            }

        Yval = build_labels(split.val, vocab, "terms_manual")
        weights, _ = search_weights(comps(split.val), Yval)

        for key, prots in (("test", split.test), ("zero", split.holdout)):
            if not prots:
                continue
            c = comps(prots)
            Y = build_labels(prots, vocab, "terms_manual")
            plm = c["plm"]
            ens = apply_hierarchical_correction(fuse(c, weights), vocab, dag)
            acc[key]["plm"].append(plm); acc[key]["ens"].append(ens)
            acc[key]["true"].append(Y)
            acc[key]["naive"].append(np.tile(fit["prior"], (Y.shape[0], 1)))
            if key == "test":
                rp, re = fmax_matrix(plm, Y).fmax, fmax_matrix(ens, Y).fmax
                w = {k: round(v, 2) for k, v in weights.items() if k != "plm"}
                print(f"       {NS_SHORT[ns]}: N={len(vocab):4d} test pLM {rp:.4f} -> ens {re:.4f} "
                      f"({re - rp:+.4f}) weights={w}", flush=True)

    def _report(title, key):
        if not acc[key]["true"]:
            return
        true = np.concatenate(acc[key]["true"], axis=1)
        plm = fmax_matrix(np.concatenate(acc[key]["plm"], axis=1), true).fmax
        ens = fmax_matrix(np.concatenate(acc[key]["ens"], axis=1), true).fmax
        nv = fmax_matrix(np.concatenate(acc[key]["naive"], axis=1), true).fmax
        print(f"\n=== {title} ===")
        print(f"  pLM-only  {plm:.4f}  (naive {nv:.4f}, lift {plm - nv:+.4f})")
        print(f"  ENSEMBLE  {ens:.4f}  (naive {nv:.4f}, lift {ens - nv:+.4f})  vs pLM {ens - plm:+.4f}")

    _report("TEST (manual-only)", "test")
    if split.holdout:
        _report(f"ZERO-SHOT — held-out {holdout_family}", "zero")
    print(f"\n[done] elapsed {time.time() - t0:.1f}s")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ensemble eval: pLM + homology + InterPro.")
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
