"""Fetch a *target* virus proteome to annotate (the Stage-3 threat use case).

Unlike `labels.py` (which pulls the whole reviewed viral training set), this pulls
one named virus by taxonomy id — including **unreviewed (TrEMBL)** entries — to
simulate a freshly sequenced, never-curated pathogen where our model is the only
annotation signal. Reuses `labels.fetch_raw` / `save_raw` / `load_raw`; the only
new logic is the target registry, a per-virus JSONL cache, and exact-sequence
dedup (TrEMBL is dominated by near-identical strain copies).

Taxonomy ids verified against UniProt 2026-06 (the NCBI taxonomy renumbered
several of these — e.g. Nipah/Lassa/Marburg now sit under new species ids).
"""

from __future__ import annotations

from dataclasses import dataclass

from viral_annotation.config import DATA_DIR
from viral_annotation.data.labels import RawProtein, fetch_raw, load_raw, save_raw

TARGETS_DIR = DATA_DIR / "targets"


@dataclass(frozen=True)
class TargetVirus:
    """A candidate-dangerous virus to annotate. `family` is informational only."""

    name: str
    taxon_id: int
    family: str
    note: str = ""


# Hemorrhagic-fever panel (BSL-4 agents). NOTE: only Coronaviridae is held out of
# training, so these families ARE in-distribution — this is "annotate + triage an
# unknown sample", not a novel-family zero-shot test. Ebola has 0 reviewed entries
# in UniProt, so it is a true "uncurated proteome" case.
TARGET_VIRUSES: dict[str, TargetVirus] = {
    "ebola": TargetVirus("Zaire ebolavirus", 186538, "Filoviridae",
                         "0 reviewed entries — fully uncurated"),
    "nipah": TargetVirus("Nipah virus", 3052225, "Paramyxoviridae"),
    "lassa": TargetVirus("Mammarenavirus lassaense", 3052310, "Arenaviridae"),
    "marburg": TargetVirus("Orthomarburgvirus marburgense", 3052505, "Filoviridae"),
}

# Bacterial select-agent panel. Taxon ids verified against UniProt (2026-06), all
# species rank. `family` is informational EXCEPT Francisellaceae, which is the
# bacterial model's held-out family — so `tularemia` is the genuine zero-shot target
# (mirroring SARS-CoV-2 / Coronaviridae on the viral side). `family` reuses the
# TargetVirus field; it is a target registry, not a virus-only structure.
TARGET_BACTERIA: dict[str, TargetVirus] = {
    "anthrax": TargetVirus("Bacillus anthracis", 1392, "Bacillaceae",
                           "anthrax; Tier-1 select agent"),
    "plague": TargetVirus("Yersinia pestis", 632, "Yersiniaceae",
                          "plague; Tier-1 select agent"),
    "tularemia": TargetVirus("Francisella tularensis", 263, "Francisellaceae",
                             "tularemia; held-out family -> zero-shot target"),
    "melioidosis": TargetVirus("Burkholderia pseudomallei", 28450, "Burkholderiaceae",
                               "melioidosis; Tier-1 select agent"),
}

# Domain -> its target panel (selected by the threat CLI's --domain).
TARGETS_BY_DOMAIN: dict[str, dict[str, TargetVirus]] = {
    "viral": TARGET_VIRUSES,
    "bacterial": TARGET_BACTERIA,
}


def target_registry(domain: str = "viral") -> dict[str, TargetVirus]:
    """The named-target panel for a pathogen domain."""
    if domain not in TARGETS_BY_DOMAIN:
        raise KeyError(f"no target panel for domain {domain!r}; "
                       f"choose from {list(TARGETS_BY_DOMAIN)}")
    return TARGETS_BY_DOMAIN[domain]


def target_query(taxon_id: int, reviewed: bool = False) -> str:
    """UniProt query for a virus by taxonomy id; unreviewed included by default."""
    q = f"(taxonomy_id:{taxon_id})"
    return f"{q} AND (reviewed:true)" if reviewed else q


def _dedup_by_sequence(records: list[RawProtein]) -> list[RawProtein]:
    """Collapse exact-duplicate sequences (strain copies), keeping the first seen."""
    seen: set[str] = set()
    out: list[RawProtein] = []
    for r in records:
        if not r.sequence or r.sequence in seen:
            continue
        seen.add(r.sequence)
        out.append(r)
    return out


def fetch_target(
    name: str,
    reviewed: bool = False,
    limit: int | None = None,
    use_cache: bool = True,
    dedup: bool = True,
    registry: dict[str, TargetVirus] | None = None,
) -> list[RawProtein]:
    """Fetch (and cache) one target pathogen's proteins as RawProtein records.

    Args:
        name: a key of `registry` (default TARGET_VIRUSES), or a bare taxonomy id.
        registry: the named-target panel to resolve `name` against (pass
            `target_registry("bacterial")` for the bacterial panel).
        reviewed: restrict to Swiss-Prot (default False -> include TrEMBL).
        dedup: collapse exact-duplicate sequences (TrEMBL strain copies).

    Raises SystemExit if the taxon returns no proteins (fail loud).
    """
    registry = registry if registry is not None else TARGET_VIRUSES
    if name in registry:
        target = registry[name]
        taxon_id, label = target.taxon_id, name
    else:
        taxon_id, label = int(name), f"taxon{name}"

    cache = TARGETS_DIR / f"{label}{'_reviewed' if reviewed else ''}.jsonl"
    if use_cache and cache.exists():
        records = list(load_raw(cache))
    else:
        records = list(fetch_raw(limit=limit, query=target_query(taxon_id, reviewed)))
        TARGETS_DIR.mkdir(parents=True, exist_ok=True)
        save_raw(records, cache)

    n_raw = len(records)
    if dedup:
        records = _dedup_by_sequence(records)
    if not records:
        raise SystemExit(
            f"no proteins for target {name!r} (taxon {taxon_id}); check the id."
        )
    print(f"[target] {label}: {len(records)} unique proteins "
          f"({n_raw} fetched{', deduped' if dedup else ''})", flush=True)
    return records
