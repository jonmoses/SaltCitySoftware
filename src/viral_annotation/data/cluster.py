"""Sequence-identity clustering via MMseqs2 (docs/03).

Used to build the rigorous split: cluster all sequences at 30% identity, then
assign whole clusters to train/val/test so no test protein has a close homolog in
training (the standard defence against inflated AFP numbers, Park & Marcotte 2012).

Requires the `mmseqs` binary on PATH (`brew install mmseqs2`).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from viral_annotation.config import (
    CLUSTER_COVERAGE,
    CLUSTER_MIN_SEQ_ID,
    CLUSTER_WORKDIR,
)


def mmseqs_available() -> bool:
    return shutil.which("mmseqs") is not None


def cluster_sequences(
    records,
    min_seq_id: float = CLUSTER_MIN_SEQ_ID,
    coverage: float = CLUSTER_COVERAGE,
    workdir: Path = CLUSTER_WORKDIR,
) -> dict[str, str]:
    """Cluster `records` (objects with .accession/.sequence) by identity.

    Returns a dict mapping each accession to its cluster representative accession
    (members of the same cluster share a representative). Every input sequence is
    assigned (singletons represent themselves).
    """
    if not mmseqs_available():
        raise RuntimeError("mmseqs not found on PATH — install with `brew install mmseqs2`")

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    fasta = workdir / "input.fasta"
    with fasta.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(f">{r.accession}\n{r.sequence}\n")

    prefix = workdir / "clu"
    tmp = workdir / "tmp"
    # easy-cluster: cascaded clustering; --min-seq-id is the identity threshold,
    # -c the coverage. Output <prefix>_cluster.tsv has rep<TAB>member per line.
    subprocess.run(
        ["mmseqs", "easy-cluster", str(fasta), str(prefix), str(tmp),
         "--min-seq-id", str(min_seq_id), "-c", str(coverage)],
        check=True, capture_output=True, text=True,
    )

    clusters: dict[str, str] = {}
    with (workdir / "clu_cluster.tsv").open(encoding="utf-8") as fh:
        for line in fh:
            rep, member = line.rstrip("\n").split("\t")
            clusters[member] = rep
    return clusters
