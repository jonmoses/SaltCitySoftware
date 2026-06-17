"""CLI for the viral/bacterial GO classifier trainer (`training.train`).

    va-train                          # viral, mean pooling, saves the servable model
    va-train --domain bacterial       # bacterial profile (taxon 2, aceae holdout, mean)
    va-train --pooling attention      # learned per-residue pooling (not servable)
    va-train --pooling per-namespace  # domain policy: attention MF, mean BP/CC
    va-train --ensemble homology      # late-fuse a BLAST-KNN component
    va-train --limit 400              # quick subset run

Unspecified knobs (--model/--pooling/--min-count/--holdout-family) resolve from the
selected `--domain` profile (config.PathogenDomain), so the viral path is unchanged.
"""

from __future__ import annotations

import argparse

from viral_annotation.config import (
    DEFAULT_DOMAIN,
    DOMAINS,
    FT_GRAD_ACCUM,
    FT_MAX_LENGTH,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_LR,
)
from viral_annotation.training.train import POOLING_CHOICES, _USE_DOMAIN, run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the GO classifier (per-namespace heads).")
    ap.add_argument("--domain", default=DEFAULT_DOMAIN, choices=list(DOMAINS),
                    help="pathogen domain profile (default viral)")
    ap.add_argument("--limit", type=int, default=None, help="cap proteins fetched (dry run)")
    ap.add_argument("--model", dest="model_key", default=None,
                    help="ESM model key (default: domain profile)")
    ap.add_argument("--pooling", default=None, choices=POOLING_CHOICES,
                    help="residue->protein pooling (default: domain profile)")
    ap.add_argument("--ensemble", choices=["homology"], default=None,
                    help="late-fuse an extra component (homology = BLAST-KNN)")
    ap.add_argument("--min-count", type=int, default=None,
                    help="term-frequency floor (default: domain profile)")
    ap.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    ap.add_argument("--lr", type=float, default=TRAIN_LR)
    ap.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    ap.add_argument("--hidden", type=int, nargs="*", default=None,
                    help="hidden layer widths for an MLP head (default: linear)")
    ap.add_argument("--random-split", action="store_true",
                    help="use a plain random split instead of the 30%% identity cluster split")
    ap.add_argument("--holdout-family", default=None,
                    help="viral/bacterial family held out for zero-shot eval "
                         "(default: domain profile; empty string to disable)")
    ap.add_argument("--records", default=None,
                    help="path to a cached RawProtein JSONL (from labels.save_raw) to load "
                         "instead of fetching UniProt — must match --domain; skips the fetch")
    ap.add_argument("--finetune", choices=["none", "lora"], default="none",
                    help="'lora' end-to-end fine-tunes the ESM backbone (adapters) + heads; "
                         "default 'none' trains heads on frozen embeddings")
    ap.add_argument("--loss", choices=["bce", "asl"], default="bce",
                    help="multi-label loss: 'bce' (pos-weighted, default) or 'asl' (asymmetric)")
    ap.add_argument("--max-length", type=int, default=FT_MAX_LENGTH,
                    help="sequence truncation length for the LoRA fine-tune path")
    ap.add_argument("--grad-accum", type=int, default=FT_GRAD_ACCUM,
                    help="gradient-accumulation steps for the LoRA fine-tune path")
    ap.add_argument("--train-pool-cap", type=int, default=None,
                    help="cap the fine-tune train pool (keeps all manual-having + sampled "
                         "IEA) to bound GPU time on large corpora")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)

    # None -> use the domain's holdout; "" -> explicitly disable; else the given name.
    if args.holdout_family is None:
        holdout = _USE_DOMAIN
    elif args.holdout_family == "":
        holdout = None
    else:
        holdout = args.holdout_family

    run(limit=args.limit, domain=args.domain, model_key=args.model_key, pooling=args.pooling,
        ensemble=args.ensemble, min_count=args.min_count, hidden_dims=args.hidden,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        use_cluster=not args.random_split, holdout_family=holdout, save=not args.no_save,
        records_path=args.records, finetune=args.finetune, loss=args.loss,
        max_length=args.max_length, grad_accum=args.grad_accum,
        train_pool_cap=args.train_pool_cap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
