"""CLI for the virus-only NetGO-style temporal benchmark (`benchmark.run`).

    va-benchmark [--cutoff YYYYMMDD] [--min-count N] [--epochs E]
"""

from __future__ import annotations

import argparse

from viral_annotation.benchmark.run import run
from viral_annotation.config import DEFAULT_ESM_MODEL


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Virus-only NetGO-style temporal benchmark.")
    ap.add_argument("--cutoff", type=int, default=20240101, help="YYYYMMDD train/test boundary")
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--model", dest="model_key", default=DEFAULT_ESM_MODEL,
                    help="ESM model key (e.g. 650M, 3B)")
    args = ap.parse_args(argv)
    run(cutoff=args.cutoff, min_count=args.min_count, epochs=args.epochs,
        model_key=args.model_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
