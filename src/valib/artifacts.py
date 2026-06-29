"""Artifact-directory IO: the conventional file formats the tools exchange.

The pipeline is wired through named files in a working directory (see the plan).
This module centralises the few simple formats so every tool reads and writes them
identically:

  * 3-column TSV — `accession <tab> go_id <tab> field` — used by annotations.tsv,
    labels.tsv, and (as accession/go_id/score) the prediction files.

All functions are pure IO with no domain logic.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Iterator


# Pre:  path is a readable 3-column TSV; blank lines are skipped.
# Post: yields one tuple per non-blank line; raises ValueError on a line that
#       does not have exactly three tab-separated fields.
# Inputs:  path (str | Path) — TSV file
# Outputs: Iterator[tuple[str, str, str]] — (col0, col1, col2)
def read_tsv3(path: str | Path) -> Iterator[tuple[str, str, str]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"expected 3 columns, got {len(parts)}: {line!r}")
            yield parts[0], parts[1], parts[2]


# Pre:  rows is an iterable of 3-tuples of strings; parent dir of path exists.
# Post: writes the rows as a 3-column TSV (no header), one row per line.
# Inputs:  path (str | Path); rows (Iterable[tuple[str, str, str]])
# Outputs: int — number of rows written
def write_tsv3(path: str | Path, rows: Iterable[tuple[str, str, str]]) -> int:
    n = 0
    with Path(path).open("w", encoding="utf-8") as fh:
        for a, b, c in rows:
            fh.write(f"{a}\t{b}\t{c}\n")
            n += 1
    return n


# Pre:  rows is an iterable of 3-tuples (key, value, tag).
# Post: returns an insertion-ordered map key -> list of (value, tag), preserving
#       row order within each key. Does not deduplicate.
# Inputs:  rows (Iterable[tuple[str, str, str]])
# Outputs: OrderedDict[str, list[tuple[str, str]]]
def group_by_first(rows: Iterable[tuple[str, str, str]]) -> "OrderedDict[str, list[tuple[str, str]]]":
    grouped: "OrderedDict[str, list[tuple[str, str]]]" = OrderedDict()
    for key, value, tag in rows:
        grouped.setdefault(key, []).append((value, tag))
    return grouped
