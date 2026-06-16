"""Virus-only NetGO-3.0-style benchmark runner.

Temporal (CAFA) split from QuickGO dated experimental annotations, then evaluate
Naive / BLAST-KNN / LR-ESM / Ensemble with Fmax + M-AUPR + Smin per ontology —
the structure of NetGO 3.0's Table 1, adapted for viruses.

Run via the CLI:  va-benchmark [--cutoff YYYYMMDD] [--min-count N]
"""

from __future__ import annotations

import time

from viral_annotation.config import (
    DEFAULT_ESM_MODEL,
    GO_NAMESPACES,
    GO_OBO_PATH,
    TRAIN_BATCH_SIZE,
    TRAIN_EARLY_STOP_PATIENCE,
    TRAIN_LR,
    TRAIN_SEED,
)
from viral_annotation.classifier.ensemble import fuse, search_weights
from viral_annotation.classifier.model import predict_proba
from viral_annotation.data import labels as labels_mod
from viral_annotation.data import quickgo
from viral_annotation.data.dataset import build_labels, select_vocab
from viral_annotation.data.homology import homology_scores
from viral_annotation.embeddings.esm import ESMEmbedder
from viral_annotation.embeddings.cache import embed_records
from viral_annotation.evaluation.metrics import (
    apply_hierarchical_correction,
    fmax_matrix,
    information_accretion,
    m_aupr,
    smin,
)
from viral_annotation.ontology import GoDag
from viral_annotation.benchmark.temporal import build_temporal_split
from viral_annotation.training.heads import fit_pooled_head
from viral_annotation.training.pipeline import auto_device

METHODS = ["Naive", "BLAST-KNN", "LR-ESM", "Ensemble"]
NS_ABBR = {"molecular_function": "MFO", "biological_process": "BPO", "cellular_component": "CCO"}


def _embed(prots, model_key, embedder):
    return embed_records(prots, model_key, "mean", None, window=True, embedder=embedder)[1]


def run(cutoff=20240101, model_key=DEFAULT_ESM_MODEL, min_count=3, epochs=300):
    import numpy as np
    import torch

    t0 = time.time()
    torch.manual_seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    device = auto_device(torch)
    print(f"[1/4] device={device} | loading GO DAG …")
    dag = GoDag.from_obo(GO_OBO_PATH)

    print("[2/4] QuickGO dated experimental annotations + UniProt sequences …")
    ann = quickgo.fetch_or_load()
    proteins = [p for p in labels_mod.label_proteins(list(labels_mod.fetch_raw()), dag) if p.sequence]
    seq_by_acc = {p.accession: p.sequence for p in proteins}
    covered = len({a.accession for a in ann} & set(seq_by_acc))
    print(f"       {len(ann)} annotations; {covered} of "
          f"{len({a.accession for a in ann})} annotated proteins have sequences")

    split = build_temporal_split(ann, seq_by_acc, dag, cutoff)
    print(f"[3/4] temporal split @ cutoff {cutoff}: {split.summary()}")

    embedder = ESMEmbedder(model_key=model_key, pooling="mean", window=True)
    results = {m: {} for m in METHODS}
    print("[4/4] evaluating per ontology (Naive / BLAST-KNN / LR-ESM / Ensemble) …")
    for ns in GO_NAMESPACES:
        train_ns, test_ns = split.train[ns], split.test[ns]
        if not train_ns or not test_ns:
            print(f"       {NS_ABBR[ns]}: insufficient data (train {len(train_ns)} test {len(test_ns)}) — skip")
            continue
        vocab = select_vocab(train_ns, dag, min_count, field="terms_manual", namespaces=[ns])
        if len(vocab) == 0:
            print(f"       {NS_ABBR[ns]}: empty vocab — skip")
            continue
        ia = information_accretion([p.terms_manual for p in train_ns], dag)

        # carve a validation set from train (LR-ESM early stop + ensemble weights)
        idx = np.random.RandomState(TRAIN_SEED).permutation(len(train_ns))
        n_val = max(1, int(0.15 * len(train_ns)))
        val_ns = [train_ns[i] for i in idx[:n_val]]
        tr_ns = [train_ns[i] for i in idx[n_val:]]

        Xtr, Xva, Xte = _embed(tr_ns, model_key, embedder), _embed(val_ns, model_key, embedder), _embed(test_ns, model_key, embedder)
        Ytr = build_labels(tr_ns, vocab, "terms_manual")
        Yva = build_labels(val_ns, vocab, "terms_manual")
        Yte = build_labels(test_ns, vocab, "terms_manual")

        naive = np.tile(Ytr.mean(axis=0), (len(test_ns), 1))
        lr_model, _, _ = fit_pooled_head(Xtr, Ytr, Xva, Yva, hidden_dims=None, epochs=epochs,
                                         lr=TRAIN_LR, batch_size=TRAIN_BATCH_SIZE, device=device,
                                         patience=TRAIN_EARLY_STOP_PATIENCE)
        lr_te = predict_proba(lr_model, Xte)
        hom_te = homology_scores(test_ns, tr_ns, dag, vocab)

        w, _ = search_weights({"plm": predict_proba(lr_model, Xva),
                               "homology": homology_scores(val_ns, tr_ns, dag, vocab)}, Yva)
        ens_te = fuse({"plm": lr_te, "homology": hom_te}, w)

        for name, prob in (("Naive", naive), ("BLAST-KNN", hom_te), ("LR-ESM", lr_te), ("Ensemble", ens_te)):
            pc = apply_hierarchical_correction(prob, vocab, dag)
            results[name][ns] = (fmax_matrix(pc, Yte).fmax, m_aupr(pc, Yte), smin(pc, Yte, ia, vocab.terms))
        print(f"       {NS_ABBR[ns]}: N={len(vocab)} test={len(test_ns)} done", flush=True)

    _print_table(results)
    print(f"\n[done] elapsed {time.time() - t0:.1f}s")
    return results


def _print_table(results):
    order = [ns for ns in GO_NAMESPACES]
    for metric, idx, better in (("Fmax", 0, "higher"), ("M-AUPR", 1, "higher"), ("Smin", 2, "lower")):
        print(f"\n=== {metric} ({better} better) ===")
        print(f"  {'method':12s} " + " ".join(f"{NS_ABBR[ns]:>7s}" for ns in order))
        for m in METHODS:
            cells = []
            for ns in order:
                v = results[m].get(ns)
                cells.append(f"{v[idx]:7.3f}" if v else f"{'—':>7s}")
            print(f"  {m:12s} " + " ".join(cells))
