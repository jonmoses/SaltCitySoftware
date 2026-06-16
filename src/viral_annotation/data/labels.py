"""Fetch viral reviewed Swiss-Prot entries and build GO label records.

Single source: the UniProt REST API. Each entry's GO cross-references carry a
`GoEvidenceType` (e.g. "IEA:InterPro", "IDA:..."); we KEEP every annotation but
tag it manual vs iea. Two propagated term sets are produced per protein:

    terms_all    = propagate(manual + iea)   -> training labels
    terms_manual = propagate(manual only)    -> val/test labels (non-circular)

Fetch (network) and labeling (needs the GO DAG) are separate so this module
doesn't load the 31 MB ontology unless you actually propagate.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from viral_annotation.config import (
    IEA_EVIDENCE_PREFIX,
    UNIPROT_FIELDS,
    UNIPROT_VIRAL_QUERY,
    VIRAL_RECORDS_PATH,
)

_USER_AGENT = "viral-annotation/0.0.1 (SBIR DPA26BZ03-DV014; +https://www.uniprot.org)"
_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
_PAGE_SIZE = 500  # UniProt max per page


@dataclass
class RawProtein:
    """One protein as fetched: sequence + raw (go_id, tier) annotations."""

    accession: str
    sequence: str
    organism: str
    lineage: list[str] = field(default_factory=list)
    # (go_id, tier) where tier in {"manual", "iea"}.
    annotations: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class LabeledProtein:
    """A protein with propagated, tier-split GO label sets."""

    accession: str
    sequence: str
    organism: str
    lineage: list[str]
    terms_all: frozenset[str]      # propagate(manual + iea) -> train labels
    terms_manual: frozenset[str]   # propagate(manual)       -> val/test labels
    n_manual: int                  # raw manual annotations (pre-propagation)
    n_iea: int                     # raw iea annotations (pre-propagation)
    # propagate(iea) — the InterPro2GO/UniRule signal, independent of manual
    # labels. A test-time feature for the ensemble; NEVER a training label.
    terms_iea: frozenset[str] = frozenset()

    @property
    def has_manual(self) -> bool:
        return self.n_manual > 0


# --- fetch ------------------------------------------------------------------
_REQUEST_TIMEOUT = 60     # seconds per page
_MAX_RETRIES = 4          # the full viral set is ~35 pages; one flaky page shouldn't kill the run


def _get(url: str) -> tuple[dict, str | None]:
    """GET JSON + the rel="next" cursor URL from the Link header, with retries.

    UniProt's cursor stream is dozens of pages for the full viral set; a single
    transient timeout would otherwise abort the whole fetch, so retry with backoff.
    """
    import time

    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                return payload, _parse_next_link(resp.headers.get("Link"))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == _MAX_RETRIES:
                raise
            time.sleep(2 ** attempt)  # 2s, 4s, 8s backoff


def _parse_next_link(link_header: str | None) -> str | None:
    # Format: <https://...fields=a,b,c&cursor=...>; rel="next"
    # The URL contains commas (the fields list), so match within the angle
    # brackets rather than splitting on ",".
    if not link_header:
        return None
    m = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
    return m.group(1) if m else None


def _parse_entry(result: dict) -> RawProtein:
    organism = result.get("organism", {}) or {}
    annotations: list[tuple[str, str]] = []
    for xref in result.get("uniProtKBCrossReferences", []):
        if xref.get("database") != "GO":
            continue
        evidence = ""
        for prop in xref.get("properties", []):
            if prop.get("key") == "GoEvidenceType":
                evidence = prop.get("value", "")
                break
        tier = "iea" if evidence.upper().startswith(IEA_EVIDENCE_PREFIX) else "manual"
        annotations.append((xref["id"], tier))
    return RawProtein(
        accession=result.get("primaryAccession", ""),
        sequence=(result.get("sequence", {}) or {}).get("value", ""),
        organism=organism.get("scientificName", ""),
        lineage=list(organism.get("lineage", []) or []),
        annotations=annotations,
    )


def fetch_raw(limit: int | None = None, query: str = UNIPROT_VIRAL_QUERY) -> Iterator[RawProtein]:
    """Stream RawProtein records from UniProt, paging by cursor.

    Stops after `limit` records if given. Network-only; no ontology needed.
    """
    params = {
        "query": query,
        "format": "json",
        "fields": ",".join(UNIPROT_FIELDS),
        "size": str(min(_PAGE_SIZE, limit) if limit else _PAGE_SIZE),
    }
    url: str | None = f"{_SEARCH_URL}?{urllib.parse.urlencode(params)}"
    yielded = 0
    while url:
        payload, url = _get(url)
        for result in payload.get("results", []):
            yield _parse_entry(result)
            yielded += 1
            if limit and yielded >= limit:
                return


# --- cache ------------------------------------------------------------------
def save_raw(records: Iterable[RawProtein], path: Path = VIRAL_RECORDS_PATH) -> int:
    """Write RawProtein records to JSONL. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(asdict(rec)) + "\n")
            n += 1
    return n


def load_raw(path: Path = VIRAL_RECORDS_PATH) -> Iterator[RawProtein]:
    """Read RawProtein records from a JSONL cache."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            # annotations come back as lists; restore tuples.
            d["annotations"] = [tuple(a) for a in d["annotations"]]
            yield RawProtein(**d)


# --- labeling (needs the GO DAG) -------------------------------------------
def label_proteins(raw: Iterable[RawProtein], dag) -> list[LabeledProtein]:
    """Propagate each protein's tier-split annotations through the GO DAG.

    `dag` is a viral_annotation.ontology.GoDag. Annotations to terms absent from
    the (non-obsolete) DAG are dropped by GoDag.propagate's resolution.
    """
    out: list[LabeledProtein] = []
    for r in raw:
        manual = [gid for gid, tier in r.annotations if tier == "manual"]
        iea = [gid for gid, tier in r.annotations if tier == "iea"]
        terms_manual = frozenset(dag.propagate(manual))
        terms_all = frozenset(dag.propagate(manual + iea))
        terms_iea = frozenset(dag.propagate(iea))
        out.append(
            LabeledProtein(
                accession=r.accession,
                sequence=r.sequence,
                organism=r.organism,
                lineage=r.lineage,
                terms_all=terms_all,
                terms_manual=terms_manual,
                n_manual=len(manual),
                n_iea=len(iea),
                terms_iea=terms_iea,
            )
        )
    return out
