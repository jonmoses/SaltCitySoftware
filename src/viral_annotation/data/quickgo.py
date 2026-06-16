"""Dated experimental GO annotations from QuickGO (for the temporal benchmark).

NetGO/CAFA benchmark on a TIME split — train on annotations before a cutoff, test
on proteins that gain their first experimental annotation later. That needs dated,
experimental-evidence annotations, which UniProt's flat fields don't give but
QuickGO does. We pull viral (taxon 10239) annotations with experimental evidence
(ECO:0000269 + descendants = EXP/IDA/IPI/IMP/IGI/IEP), each with its assertion date.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from viral_annotation.config import DATA_DIR

_USER_AGENT = "viral-annotation/0.0.1 (SBIR DPA26BZ03-DV014)"
_URL = ("https://www.ebi.ac.uk/QuickGO/services/annotation/downloadSearch"
        "?taxonId=10239&taxonUsage=descendants"
        "&evidenceCode=ECO:0000269&evidenceCodeUsage=descendants"
        "&downloadLimit={limit}")
_ASPECT = {"F": "molecular_function", "P": "biological_process", "C": "cellular_component"}

QUICKGO_PATH = DATA_DIR / "quickgo_viral_exp.jsonl"


@dataclass
class ExpAnnotation:
    accession: str
    go_id: str
    namespace: str   # molecular_function / biological_process / cellular_component
    evidence: str    # EXP/IDA/IPI/IMP/IGI/IEP
    date: int        # YYYYMMDD as int, for easy temporal comparison


def fetch(limit: int = 50000) -> list[ExpAnnotation]:
    """Download viral experimental annotations from QuickGO (TSV) -> list."""
    req = urllib.request.Request(_URL.format(limit=limit),
                                 headers={"User-Agent": _USER_AGENT, "Accept": "text/tsv"})
    with urllib.request.urlopen(req) as resp:
        lines = resp.read().decode("utf-8").splitlines()

    header = lines[0].split("\t")
    col = {name: i for i, name in enumerate(header)}
    out: list[ExpAnnotation] = []
    for line in lines[1:]:
        f = line.split("\t")
        if "NOT" in f[col["QUALIFIER"]]:            # skip negative annotations
            continue
        aspect = _ASPECT.get(f[col["GO ASPECT"]])
        date = f[col["DATE"]].strip()
        if aspect is None or not date.isdigit():
            continue
        out.append(ExpAnnotation(
            accession=f[col["GENE PRODUCT ID"]],
            go_id=f[col["GO TERM"]],
            namespace=aspect,
            evidence=f[col["GO EVIDENCE CODE"]],
            date=int(date),
        ))
    return out


def save(annotations, path: Path = QUICKGO_PATH) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for a in annotations:
            fh.write(json.dumps(asdict(a)) + "\n")
    return len(annotations)


def load(path: Path = QUICKGO_PATH) -> Iterator[ExpAnnotation]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            yield ExpAnnotation(**json.loads(line))


def fetch_or_load(path: Path = QUICKGO_PATH, limit: int = 50000) -> list[ExpAnnotation]:
    """Cached fetch: load the JSONL if present, else download and save."""
    if path.exists():
        return list(load(path))
    ann = fetch(limit)
    save(ann, path)
    return ann
