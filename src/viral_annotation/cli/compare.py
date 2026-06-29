"""CLI for the ESM-2 backbone head-to-head benchmark (`benchmark.compare`).

    va-compare-embeddings                               # 650M vs 3B, both domains+protocols
    va-compare-embeddings --domains viral               # viral only
    va-compare-embeddings --protocols cluster           # skip the temporal CAFA run
    va-compare-embeddings --models 650M 3B --seeds 5 --bootstrap 1000

Runs entirely off the cached embeddings + cached records (no GPU, no UniProt fetch).
Bootstrap is the slow part on large test sets — lower --bootstrap for a quick look.
"""

from __future__ import annotations

import argparse

from viral_annotation.benchmark.compare import run_comparison


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Head-to-head benchmark of two ESM-2 backbones.")
    ap.add_argument("--domains", nargs="+", default=["viral", "bacterial"],
                    choices=["viral", "bacterial"], help="pathogen domains to compare")
    ap.add_argument("--protocols", nargs="+", default=["cluster", "temporal"],
                    choices=["cluster", "temporal"],
                    help="cluster (viral+bacterial) and/or temporal CAFA (viral only)")
    ap.add_argument("--models", nargs="+", default=["650M", "3B"],
                    help="ESM model keys to compare (delta/CI assume exactly two)")
    ap.add_argument("--seeds", type=int, default=None,
                    help="number of seeds per fit (default: 5 viral / 3 bacterial)")
    ap.add_argument("--bootstrap", type=int, default=1000,
                    help="paired bootstrap resamples for the delta CI (0 disables)")
    ap.add_argument("--cutoff", type=int, default=20240101,
                    help="YYYYMMDD train/test boundary for the temporal protocol")
    args = ap.parse_args(argv)

    run_comparison(domains=tuple(args.domains), protocols=tuple(args.protocols),
                   models=tuple(args.models), seeds=args.seeds, n_boot=args.bootstrap,
                   cutoff=args.cutoff)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
