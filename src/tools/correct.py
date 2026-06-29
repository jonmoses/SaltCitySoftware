"""correct — pred.tsv + go-basic.obo -> pred.corrected.tsv (DAG-consistent).

Reads per-protein prediction scores and raises each parent term's score to at
least its best descendant's, so the output never violates the GO hierarchy. One
job, no network/GPU.

Usage:
    python -m tools.correct --obo data/go-basic.obo \\
        --in work/pred.tsv --out work/pred.corrected.tsv
"""

from __future__ import annotations

import argparse
from typing import Sequence

from valib.artifacts import group_by_first, read_tsv3, write_tsv3
from valib.godag import GoDag
from valib.predict import correct_predictions


# Pre:  argv is None (use sys.argv) or a list of CLI tokens.
# Post: returns parsed args with .obo, .infile, .outfile string paths.
# Inputs:  argv (Sequence[str] | None)
# Outputs: argparse.Namespace
def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hierarchically correct prediction rows.")
    p.add_argument("--obo", required=True, help="path to go-basic.obo")
    p.add_argument("--in", dest="infile", required=True, help="pred.tsv in")
    p.add_argument("--out", dest="outfile", required=True, help="pred.corrected.tsv out")
    return p.parse_args(argv)


# Pre:  the OBO and input TSV named in argv exist and are readable; the output
#       directory exists; input scores parse as floats.
# Post: writes pred.corrected.tsv (accession, go_id, score) and prints the row
#       count. Returns process exit code 0 on success.
# Inputs:  argv (Sequence[str] | None)
# Outputs: int — exit code
def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    dag = GoDag.from_obo(args.obo)
    grouped = group_by_first(read_tsv3(args.infile))
    rows = correct_predictions(grouped, dag)
    n = write_tsv3(args.outfile, rows)
    print(f"[correct] {len(grouped)} proteins -> {n} corrected rows -> {args.outfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
