"""End-to-end training for the viral GO classifier (docs/01; plan slice).

Pipeline: fetch viral labels -> propagate (tier-split) -> asymmetric split ->
precompute/cache ESM embeddings -> select term vocabulary -> train a linear
multi-label head with positive-weighted BCE, early-stopping on validation Fmax ->
hierarchically correct -> report per-namespace Fmax on the manual-only test set.

Run:  python -m viral_annotation.training.train [--limit N] [--epochs E] ...
"""

from __future__ import annotations

import argparse
import json
import time

from viral_annotation.config import (
    DEFAULT_ESM_MODEL,
    DEFAULT_POOLING,
    GO_OBO_PATH,
    MIN_TERM_COUNT,
    MODELS_DIR,
    POS_WEIGHT_CLAMP,
    TRAIN_BATCH_SIZE,
    TRAIN_EARLY_STOP_PATIENCE,
    TRAIN_EPOCHS,
    TRAIN_LR,
    TRAIN_WEIGHT_DECAY,
)
from viral_annotation.classifier.model import build_classifier, predict_proba
from viral_annotation.data import labels as labels_mod
from viral_annotation.data.dataset import build_labels, select_vocab
from viral_annotation.data.split import split_proteins
from viral_annotation.embeddings.cache import embed_records
from viral_annotation.evaluation.metrics import (
    apply_hierarchical_correction,
    fmax_by_namespace,
    fmax_matrix,
)
from viral_annotation.ontology import GoDag


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
    save: bool = True,
):
    import numpy as np
    import torch
    from torch import nn

    t0 = time.time()
    device = _auto_device(torch)
    print(f"[1/6] device={device} | loading GO DAG …")
    dag = GoDag.from_obo(GO_OBO_PATH)

    print(f"[2/6] fetching viral reviewed proteins (limit={limit}) …")
    raw = list(labels_mod.fetch_raw(limit=limit))
    proteins = labels_mod.label_proteins(raw, dag)
    proteins = [p for p in proteins if p.sequence]  # drop any empty sequences
    print(f"       {_annotation_stats(proteins)}")

    split = split_proteins(proteins)
    print(f"[3/6] split: {split.summary()}")
    if not split.val or not split.test:
        raise SystemExit(
            "val/test empty — too few manual-having proteins. Increase --limit."
        )

    vocab = select_vocab(split.train, dag, min_count=min_count)
    ns_sizes = {ns: len(c) for ns, c in vocab.columns_by_namespace().items()}
    print(f"[4/6] term vocabulary: N={len(vocab)} (min_count={min_count}) {ns_sizes}")
    if len(vocab) == 0:
        raise SystemExit("empty term vocabulary — lower --min-count or raise --limit.")

    print(f"[5/6] embedding ({model_key}, {pooling}) — cached where possible …")
    from viral_annotation.embeddings.esm import ESMEmbedder

    embedder = ESMEmbedder(model_key=model_key, pooling=pooling, repr_layer=repr_layer)
    _, Xtr = embed_records(split.train, model_key, pooling, repr_layer, embedder=embedder)
    _, Xva = embed_records(split.val, model_key, pooling, repr_layer, embedder=embedder)
    _, Xte = embed_records(split.test, model_key, pooling, repr_layer, embedder=embedder)
    Ytr = build_labels(split.train, vocab, "terms_all")        # train: manual+iea
    Yva = build_labels(split.val, vocab, "terms_manual")       # eval: manual-only
    Yte = build_labels(split.test, vocab, "terms_manual")

    # Positive-weighted BCE to fight class imbalance: w_t = n_neg / n_pos, clamped.
    pos = Ytr.sum(axis=0)
    neg = Ytr.shape[0] - pos
    pos_weight = np.clip(neg / np.clip(pos, 1, None), 0, POS_WEIGHT_CLAMP).astype("float32")

    model = build_classifier(Xtr.shape[1], len(vocab), hidden_dims=hidden_dims).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=TRAIN_WEIGHT_DECAY)

    Xtr_t = torch.tensor(Xtr, device=device)
    Ytr_t = torch.tensor(Ytr, device=device)

    print(f"[6/6] training: {epochs} epochs max, batch={batch_size}, "
          f"arch={'linear' if not hidden_dims else hidden_dims} …")
    best_val = -1.0
    best_state = None
    patience = 0
    n = Xtr_t.shape[0]
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(Xtr_t[idx]), Ytr_t[idx])
            loss.backward()
            optimizer.step()

        # Validation Fmax (overall) on raw probs — cheap signal for early stopping.
        val_prob = predict_proba(model, Xva)
        val_fmax = fmax_matrix(val_prob, Yva).fmax
        improved = val_fmax > best_val + 1e-4
        if improved:
            best_val, patience = val_fmax, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if epoch == 1 or epoch % 10 == 0 or improved:
            print(f"       epoch {epoch:3d} | loss {loss.item():.4f} | val Fmax {val_fmax:.4f}"
                  f"{'  *' if improved else ''}")
        if patience >= TRAIN_EARLY_STOP_PATIENCE:
            print(f"       early stop at epoch {epoch} (best val Fmax {best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test eval WITH hierarchical correction (consistent, interpretable).
    test_prob = predict_proba(model, Xte)
    test_prob = apply_hierarchical_correction(test_prob, vocab, dag)
    results = fmax_by_namespace(test_prob, Yte, vocab)

    print("\n=== TEST (manual-only labels, hierarchically corrected) ===")
    for key in ("molecular_function", "biological_process", "cellular_component", "overall"):
        if key in results:
            r = results[key]
            print(f"  {key:20s} Fmax={r.fmax:.4f}  P={r.precision:.3f}  R={r.recall:.3f}  "
                  f"tau={r.threshold:.2f}  (N={r.n_terms})")
    print(f"\n[done] elapsed {time.time() - t0:.1f}s")

    if save:
        _save_model(model, vocab, model_key, pooling, repr_layer, hidden_dims, results)
    return results


def _save_model(model, vocab, model_key, pooling, repr_layer, hidden_dims, results):
    import torch

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "go_classifier.pt")
    meta = {
        "terms": vocab.terms,
        "namespaces": vocab.namespaces,
        "esm_model": model_key,
        "pooling": pooling,
        "repr_layer": repr_layer,
        "hidden_dims": hidden_dims or [],
        "test_fmax": {k: v.fmax for k, v in results.items()},
    }
    (MODELS_DIR / "go_classifier.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[saved] {MODELS_DIR / 'go_classifier.pt'} (+ meta.json)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the viral GO classifier.")
    ap.add_argument("--limit", type=int, default=None, help="cap proteins fetched (dry run)")
    ap.add_argument("--model", dest="model_key", default=DEFAULT_ESM_MODEL)
    ap.add_argument("--min-count", type=int, default=MIN_TERM_COUNT)
    ap.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    ap.add_argument("--lr", type=float, default=TRAIN_LR)
    ap.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    ap.add_argument("--hidden", type=int, nargs="*", default=None,
                    help="hidden layer widths for an MLP head (default: linear)")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)

    run(
        limit=args.limit,
        model_key=args.model_key,
        min_count=args.min_count,
        hidden_dims=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        save=not args.no_save,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
