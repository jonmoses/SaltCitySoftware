"""CLI for the virus-only NetGO-style temporal benchmark (`benchmark.run`).

    va-benchmark [--cutoff YYYYMMDD] [--min-count N] [--epochs E]
"""

from __future__ import annotations

import argparse

from viral_annotation.benchmark.run import run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Virus-only NetGO-style temporal benchmark.")
    ap.add_argument("--cutoff", type=int, default=20240101, help="YYYYMMDD train/test boundary")
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args(argv)
    run(cutoff=args.cutoff, min_count=args.min_count, epochs=args.epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
