"""End-to-end training for the viral GO classifier (docs/01).

Per-namespace evidence policy (config.NAMESPACE_POLICY): a full-set experiment
showed IEA training poisons Molecular Function in viruses — manual-MF is curated
protein-binding/adaptor terms, IEA-MF is domain-rule ligand binding, and the two
are nearly disjoint (MF Fmax 0.09 under IEA training, ~0.20 manual-only). So each
GO namespace trains its OWN linear head under its own policy:
  * MF    -> manual-having proteins, manual labels, manual-selected vocab
  * BP/CC -> all proteins, manual+IEA labels (asymmetric), terms_all vocab
Validation/test always score on manual-only labels. Heads combine for the overall
metric. See memory: iea-manual-mf-distribution-shift.

Pipeline: fetch labels -> propagate (tier-split) -> split -> cache ESM embeddings
-> per namespace {select vocab, train head with pos-weighted BCE + early stop on
val Fmax, hierarchically correct} -> per-namespace + overall Fmax on manual test.

Run:  python -m viral_annotation.training.train [--limit N] [--epochs E] ...
"""

from __future__ import annotations

import argparse
import json
import time

from viral_annotation.config import (
    DEFAULT_ESM_MODEL,
    DEFAULT_POOLING,
    GO_NAMESPACES,
    GO_OBO_PATH,
    HOLDOUT_FAMILY,
    MIN_TERM_COUNT,
    MODELS_DIR,
    NAMESPACE_POLICY,
    POS_WEIGHT_CLAMP,
    TRAIN_BATCH_SIZE,
    TRAIN_EARLY_STOP_PATIENCE,
    TRAIN_EPOCHS,
    TRAIN_LR,
    TRAIN_WEIGHT_DECAY,
)
from viral_annotation.classifier.model import build_classifier, predict_proba
from viral_annotation.data import labels as labels_mod
from viral_annotation.data.cluster import cluster_sequences
from viral_annotation.data.dataset import build_labels, select_vocab
from viral_annotation.data.split import cluster_split, split_proteins
from viral_annotation.embeddings.cache import embed_records
from viral_annotation.evaluation.metrics import apply_hierarchical_correction, fmax_matrix
from viral_annotation.ontology import GoDag

NS_SHORT = {
    "molecular_function": "MF",
    "biological_process": "BP",
    "cellular_component": "CC",
}


def _auto_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _annotation_stats(proteins) -> str:
    n_manual = sum(p.n_manual for p in proteins)
    n_iea = sum(p.n_iea for p in proteins)
    have_manual = sum(1 for p in proteins if p.has_manual)
    return (
        f"{len(proteins)} proteins | manual-having {have_manual} "
        f"({100 * have_manual / max(len(proteins), 1):.1f}%) | "
        f"raw annotations manual={n_manual} iea={n_iea}"
    )


def _train_head(Xtr, Ytr, Xva, Yva, *, hidden_dims, epochs, lr, batch_size, device, patience):
    """Train one multi-label linear/MLP head; early-stop on validation Fmax.

    Returns (best_model, best_val_fmax, epochs_run).
    """
    import numpy as np
    import torch
    from torch import nn

    pos = Ytr.sum(axis=0)
    neg = Ytr.shape[0] - pos
    pw = np.clip(neg / np.clip(pos, 1, None), 0, POS_WEIGHT_CLAMP).astype("float32")

    model = build_classifier(Xtr.shape[1], Ytr.shape[1], hidden_dims=hidden_dims).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=TRAIN_WEIGHT_DECAY)
    Xt = torch.tensor(Xtr, device=device)
    Yt = torch.tensor(Ytr, device=device)
    n = Xt.shape[0]

    best, best_state, wait, ep = -1.0, None, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(Xt[idx]), Yt[idx])
            loss.backward()
            optimizer.step()
        vf = fmax_matrix(predict_proba(model, Xva), Yva).fmax
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


def run(
    limit: int | None = None,
    model_key: str = DEFAULT_ESM_MODEL,
    pooling: str = DEFAULT_POOLING,
    repr_layer: int | None = None,
    min_count: int = MIN_TERM_COUNT,
    hidden_dims: list[int] | None = None,
    epochs: int = TRAIN_EPOCHS,
    lr: float = TRAIN_LR,
    batch_size: int = TRAIN_BATCH_SIZE,
    use_cluster: bool = True,
    holdout_family: str | None = HOLDOUT_FAMILY,
    save: bool = True,
):
    import numpy as np
    import torch

    t0 = time.time()
    device = _auto_device(torch)
    print(f"[1/5] device={device} | loading GO DAG …")
    dag = GoDag.from_obo(GO_OBO_PATH)

    print(f"[2/5] fetching viral reviewed proteins (limit={limit}) …")
    raw = list(labels_mod.fetch_raw(limit=limit))
    proteins = [p for p in labels_mod.label_proteins(raw, dag) if p.sequence]
    print(f"       {_annotation_stats(proteins)}")

    if use_cluster:
        print(f"[3/5] clustering at 30% identity (MMseqs2) + split"
              f"{f', holding out {holdout_family}' if holdout_family else ''} …")
        clusters = cluster_sequences(proteins)
        split = cluster_split(proteins, clusters, holdout_family=holdout_family)
    else:
        print("[3/5] random split (NOT leakage-safe) …")
        split = split_proteins(proteins)
    manual_train = [p for p in split.train if p.has_manual]
    print(f"       split={'cluster' if use_cluster else 'random'}: {split.summary()} "
          f"| manual-having train {len(manual_train)}")
    if not split.val or not split.test:
        raise SystemExit("val/test empty — too few manual-having proteins. Increase --limit.")

    # Embed every pool we'll need, once. All cached after the first full run, so
    # this is instant on re-runs (no ESM forward passes).
    from viral_annotation.embeddings.esm import ESMEmbedder

    embedder = ESMEmbedder(model_key=model_key, pooling=pooling, repr_layer=repr_layer)
    print(f"[4/5] embedding ({model_key}, {pooling}) — cached where possible …")
    pool_prots = {"all": split.train, "manual_having": manual_train}
    pool_X = {
        name: embed_records(prots, model_key, pooling, repr_layer, embedder=embedder)[1]
        for name, prots in pool_prots.items()
    }
    _, Xva = embed_records(split.val, model_key, pooling, repr_layer, embedder=embedder)
    _, Xte = embed_records(split.test, model_key, pooling, repr_layer, embedder=embedder)
    Xho = (embed_records(split.holdout, model_key, pooling, repr_layer, embedder=embedder)[1]
           if split.holdout else None)

    arch = "linear" if not hidden_dims else hidden_dims
    print(f"[5/5] training per-namespace heads (arch={arch}) …")
    heads: dict[str, dict] = {}
    combo_prob, combo_true = [], []
    for ns in GO_NAMESPACES:
        pol = NAMESPACE_POLICY[ns]
        train_prots = pool_prots[pol["train_pool"]]
        Xtr = pool_X[pol["train_pool"]]
        vocab = select_vocab(train_prots, dag, min_count,
                             field=pol["vocab_field"], namespaces=[ns])
        if len(vocab) == 0:
            print(f"       {NS_SHORT[ns]}: empty vocab — skipped")
            continue
        Ytr = build_labels(train_prots, vocab, pol["train_field"])
        Yva = build_labels(split.val, vocab, "terms_manual")
        Yte = build_labels(split.test, vocab, "terms_manual")

        model, vbest, ep = _train_head(
            Xtr, Ytr, Xva, Yva, hidden_dims=hidden_dims, epochs=epochs, lr=lr,
            batch_size=batch_size, device=device, patience=TRAIN_EARLY_STOP_PATIENCE,
        )
        prob_te = apply_hierarchical_correction(predict_proba(model, Xte), vocab, dag)
        res = fmax_matrix(prob_te, Yte)
        heads[ns] = {"vocab": vocab, "model": model, "state": model.state_dict(),
                     "policy": pol, "result": res}
        combo_prob.append(prob_te)
        combo_true.append(Yte)
        print(f"       {NS_SHORT[ns]}: N={len(vocab):4d} pool={pol['train_pool']:13s} "
              f"train={pol['train_field']:12s} -> Fmax={res.fmax:.4f} "
              f"(P={res.precision:.3f} R={res.recall:.3f} tau={res.threshold:.2f}) "
              f"val={vbest:.3f} epochs={ep}")

    # Overall: namespaces own disjoint term sets, so concatenate columns.
    overall = fmax_matrix(np.concatenate(combo_prob, axis=1), np.concatenate(combo_true, axis=1))

    print("\n=== TEST (manual-only labels, hierarchically corrected) ===")
    for ns in GO_NAMESPACES:
        if ns in heads:
            r = heads[ns]["result"]
            print(f"  {ns:20s} Fmax={r.fmax:.4f}  P={r.precision:.3f}  R={r.recall:.3f}  (N={r.n_terms})")
    print(f"  {'overall':20s} Fmax={overall.fmax:.4f}  P={overall.precision:.3f}  "
          f"R={overall.recall:.3f}  (N={overall.n_terms})")

    # Zero-shot: recover the held-out family's known functions with the SAME heads.
    zeroshot = None
    if split.holdout and Xho is not None:
        zprob, ztrue = [], []
        for ns in GO_NAMESPACES:
            if ns not in heads:
                continue
            h = heads[ns]
            p = apply_hierarchical_correction(predict_proba(h["model"], Xho), h["vocab"], dag)
            y = build_labels(split.holdout, h["vocab"], "terms_manual")
            heads[ns]["zeroshot"] = fmax_matrix(p, y)
            zprob.append(p)
            ztrue.append(y)
        zeroshot = fmax_matrix(np.concatenate(zprob, axis=1), np.concatenate(ztrue, axis=1))
        print(f"\n=== ZERO-SHOT — held-out {holdout_family} "
              f"({len(split.holdout)} manual-having proteins, never trained on) ===")
        for ns in GO_NAMESPACES:
            if ns in heads and "zeroshot" in heads[ns]:
                z = heads[ns]["zeroshot"]
                print(f"  {ns:20s} Fmax={z.fmax:.4f}  P={z.precision:.3f}  R={z.recall:.3f}  (N={z.n_terms})")
        print(f"  {'overall':20s} Fmax={zeroshot.fmax:.4f}  P={zeroshot.precision:.3f}  "
              f"R={zeroshot.recall:.3f}  (N={zeroshot.n_terms})")

    print(f"\n[done] elapsed {time.time() - t0:.1f}s")

    if save:
        _save_models(heads, overall, model_key, pooling, repr_layer, hidden_dims)
    out = {**{ns: heads[ns]["result"] for ns in heads}, "overall": overall}
    if zeroshot is not None:
        out["zeroshot_overall"] = zeroshot
    return out


def _save_models(heads, overall, model_key, pooling, repr_layer, hidden_dims):
    import torch

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({ns: heads[ns]["state"] for ns in heads}, MODELS_DIR / "go_classifier.pt")
    meta = {
        "esm_model": model_key,
        "pooling": pooling,
        "repr_layer": repr_layer,
        "hidden_dims": hidden_dims or [],
        "overall_fmax": overall.fmax,
        "namespaces": {
            ns: {
                "policy": heads[ns]["policy"],
                "fmax": heads[ns]["result"].fmax,
                "terms": heads[ns]["vocab"].terms,
            }
            for ns in heads
        },
    }
    (MODELS_DIR / "go_classifier.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[saved] {MODELS_DIR / 'go_classifier.pt'} (+ meta.json)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the viral GO classifier (per-namespace).")
    ap.add_argument("--limit", type=int, default=None, help="cap proteins fetched (dry run)")
    ap.add_argument("--model", dest="model_key", default=DEFAULT_ESM_MODEL)
    ap.add_argument("--min-count", type=int, default=MIN_TERM_COUNT)
    ap.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    ap.add_argument("--lr", type=float, default=TRAIN_LR)
    ap.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    ap.add_argument("--hidden", type=int, nargs="*", default=None,
                    help="hidden layer widths for an MLP head (default: linear)")
    ap.add_argument("--random-split", action="store_true",
                    help="use the old random split instead of the 30%% identity cluster split")
    ap.add_argument("--holdout-family", default=HOLDOUT_FAMILY,
                    help="viral family held out for zero-shot eval (empty string to disable)")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)

    run(
        limit=args.limit, model_key=args.model_key, min_count=args.min_count,
        hidden_dims=args.hidden, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, use_cluster=not args.random_split,
        holdout_family=args.holdout_family or None, save=not args.no_save,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
