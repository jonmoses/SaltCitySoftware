"""propagate — annotations.tsv + go-basic.obo -> labels.tsv (true-path closed).

Reads raw per-protein GO annotations, closes each protein's terms under ancestors
(per evidence tier), and writes the propagated labels. One job, no network/GPU.

Usage:
    python -m tools.propagate --obo data/go-basic.obo \\
        --in work/annotations.tsv --out work/labels.tsv
"""

from __future__ import annotations

import argparse
from typing import Sequence

from valib.artifacts import group_by_first, read_tsv3, write_tsv3
from valib.godag import GoDag
from valib.labels import propagate_annotations


# Pre:  argv is None (use sys.argv) or a list of CLI tokens.
# Post: returns parsed args with .obo, .infile, .outfile string paths.
# Inputs:  argv (Sequence[str] | None)
# Outputs: argparse.Namespace
def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="True-path propagate annotation rows.")
    p.add_argument("--obo", required=True, help="path to go-basic.obo")
    p.add_argument("--in", dest="infile", required=True, help="annotations.tsv in")
    p.add_argument("--out", dest="outfile", required=True, help="labels.tsv out")
    return p.parse_args(argv)


# Pre:  the OBO and input TSV named in argv exist and are readable; the output
#       directory exists.
# Post: writes labels.tsv (accession, go_id, tier) propagated under the DAG and
#       prints the row count. Returns process exit code 0 on success.
# Inputs:  argv (Sequence[str] | None)
# Outputs: int — exit code
def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    dag = GoDag.from_obo(args.obo)
    grouped = group_by_first(read_tsv3(args.infile))
    rows = propagate_annotations(grouped, dag)
    n = write_tsv3(args.outfile, rows)
    print(f"[propagate] {len(grouped)} proteins -> {n} label rows -> {args.outfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
