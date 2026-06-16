"""BLAST-KNN homology component: transfer GO labels from sequence neighbours.

For each query protein we MMseqs2-search it against an annotated database (the
manual-having training proteins), then score each GO term by the bitscore-weighted
fraction of hits annotated with it (the GOLabeler/NetGO formulation):

    score(q, t) = sum_{h : t in labels(h)} bitscore(q,h)  /  sum_h bitscore(q,h)

This transfers the neighbours' MANUAL labels, so it is aligned with our manual-only
evaluation. Strongest where annotated relatives exist; for a held-out family it
transfers conserved function from other families. Reuses MMseqs2 (see cluster.py).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from viral_annotation.config import CLUSTER_WORKDIR
from viral_annotation.data.cluster import mmseqs_available

# Permissive search so distant (cross-family) relatives still produce hits —
# that's exactly the zero-shot regime we want signal in.
SEARCH_SENSITIVITY = 7.5
SEARCH_EVALUE = 10.0


def homology_scores(query_records, db_records, dag, vocab,
                    workdir: Path = CLUSTER_WORKDIR / "homology"):
    """Return [len(query) x len(vocab)] bitscore-weighted label-transfer scores.

    db_records carry the labels to transfer (use manual-having train proteins;
    their `terms_manual` are propagated and restricted to `vocab`).
    """
    import numpy as np

    if not mmseqs_available():
        raise RuntimeError("mmseqs not found on PATH — `brew install mmseqs2`")

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    q_fasta, db_fasta = workdir / "query.fasta", workdir / "db.fasta"
    _write_fasta(query_records, q_fasta)
    _write_fasta(db_records, db_fasta)

    out = workdir / "hits.m8"
    subprocess.run(
        ["mmseqs", "easy-search", str(q_fasta), str(db_fasta), str(out), str(workdir / "tmp"),
         "-s", str(SEARCH_SENSITIVITY), "-e", str(SEARCH_EVALUE),
         "--format-output", "query,target,bits"],
        check=True, capture_output=True, text=True,
    )

    # db label sets restricted to the vocab (terms outside vocab can't be scored).
    col = vocab.index
    db_terms = {r.accession: [col[t] for t in r.terms_manual if t in col] for r in db_records}

    # query accession -> list of (target, bits)
    hits: dict[str, list] = {r.accession: [] for r in query_records}
    with out.open(encoding="utf-8") as fh:
        for line in fh:
            q, target, bits = line.rstrip("\n").split("\t")
            if q == target or q not in hits:   # skip self-hits
                continue
            hits[q].append((target, float(bits)))

    return _aggregate([r.accession for r in query_records], hits, db_terms, len(vocab))


def _aggregate(query_accs, hits, db_term_cols, n_vocab):
    """Bitscore-weighted transfer -> [len(query) x n_vocab]. Pure (testable)."""
    import numpy as np

    S = np.zeros((len(query_accs), n_vocab), dtype="float32")
    for i, acc in enumerate(query_accs):
        h = hits.get(acc, [])
        total = sum(b for _, b in h)
        if total <= 0:
            continue
        for target, bits in h:
            for c in db_term_cols.get(target, ()):
                S[i, c] += bits
        S[i] /= total
    return S


def _write_fasta(records, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(f">{r.accession}\n{r.sequence}\n")
