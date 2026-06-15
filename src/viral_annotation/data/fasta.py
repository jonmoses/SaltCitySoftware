"""FASTA I/O.

Uses Biopython if installed (the `[bio]` extra), else a small stdlib parser so
the pipeline's most basic step never blocks on a heavy install.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class ProteinRecord:
    """One protein sequence. `id` is the first whitespace-delimited token."""

    id: str
    sequence: str
    description: str = ""


def read_fasta(path: str | Path) -> Iterator[ProteinRecord]:
    """Yield ProteinRecord per entry. Prefers Biopython when available."""
    path = Path(path)
    try:
        from Bio import SeqIO  # type: ignore

        for rec in SeqIO.parse(str(path), "fasta"):
            yield ProteinRecord(
                id=rec.id,
                sequence=str(rec.seq),
                description=rec.description,
            )
        return
    except ImportError:
        yield from _read_fasta_stdlib(path)


def _read_fasta_stdlib(path: Path) -> Iterator[ProteinRecord]:
    header: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield _make_record(header, chunks)
                header = line[1:]
                chunks = []
            elif line:
                chunks.append(line.strip())
    if header is not None:
        yield _make_record(header, chunks)


def _make_record(header: str, chunks: list[str]) -> ProteinRecord:
    ident = header.split(None, 1)[0] if header else ""
    return ProteinRecord(id=ident, sequence="".join(chunks), description=header)
