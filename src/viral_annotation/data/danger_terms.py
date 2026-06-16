"""Curated GO "danger" ontology — the threat-characterization knowledge base.

Maps human-meaningful **danger categories** to a small set of high-level GO *root*
terms. At load time (`threat.build_danger_map`) each root is expanded to all its
descendants over the live DAG, so a prediction on any specific term (e.g. "fusion of
virus membrane with host endosome membrane") counts as a hit for its category
("Host-cell entry & membrane fusion").

This is a HAND-CURATED ontology subset, not an authoritative danger list — it is
meant to be auditable and expert-reviewable, which is why every root carries a name
comment and a rationale. All ids were verified present and **non-obsolete** in
go-basic (2026-06); note GO renamed the classic "by virus of host" terms to
"symbiont-mediated", and several old ids (e.g. GO:0019048) are now obsolete — only
live ids appear here. `threat.build_danger_map` re-asserts this at load so a future
ontology update that obsoletes a root fails loudly instead of silently going dark.

Reality check (against the shipped model's 695-term vocab): the danger signal lives
almost entirely in **biological_process** (host-interaction mechanisms). Classic
toxin MF terms are absent from the viral training vocab, so that category rarely
fires — kept for completeness and future non-viral use.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DangerCategory:
    """A named danger mechanism and the GO roots that define it."""

    key: str
    label: str
    roots: list[str]              # high-level GO ids; expanded to descendants at load
    rationale: str = ""
    root_names: dict[str, str] = field(default_factory=dict)  # id -> name, for audit


# Ordered roughly by directness of harm. Roots are deliberately high-level; the DAG
# expansion pulls in the specific mechanisms beneath each.
DANGER_CATEGORIES: list[DangerCategory] = [
    DangerCategory(
        key="toxin",
        label="Toxin activity",
        roots=["GO:0090729"],
        rationale="Direct molecular toxicity toward host cells/tissues.",
        root_names={"GO:0090729": "toxin activity"},
    ),
    DangerCategory(
        key="entry_fusion",
        label="Host-cell entry & membrane fusion",
        roots=["GO:0044409", "GO:0039663", "GO:0019064", "GO:0039654",
               "GO:0075503", "GO:0098997", "GO:0075732"],
        rationale="Getting into the host cell — attachment, membrane fusion, "
                  "envelope/membrane disruption, nuclear penetration. The first "
                  "step of infection and a prime intervention target.",
        root_names={
            "GO:0044409": "symbiont entry into host",
            "GO:0039663": "membrane fusion involved in viral entry into host cell",
            "GO:0019064": "fusion of virus membrane with host plasma membrane",
            "GO:0039654": "fusion of virus membrane with host endosome membrane",
            "GO:0075503": "fusion of virus membrane with host macropinosome membrane",
            "GO:0098997": "fusion of virus membrane with host outer membrane",
            "GO:0075732": "viral penetration into host nucleus",
        },
    ),
    DangerCategory(
        key="kill_lyse",
        label="Host-cell killing / lysis",
        roots=["GO:0001907", "GO:0001897", "GO:0019835", "GO:0051715"],
        rationale="Killing or lysing host cells — directly cytopathic, the basis "
                  "of tissue damage in acute viral disease.",
        root_names={
            "GO:0001907": "symbiont-mediated killing of host cell",
            "GO:0001897": "symbiont-mediated cytolysis of host cell",
            "GO:0019835": "cytolysis",
            "GO:0051715": "cytolysis in another organism",
        },
    ),
    DangerCategory(
        key="immune_evasion",
        label="Immune evasion / host-defense perturbation",
        roots=["GO:0030682", "GO:0042783", "GO:0052167", "GO:0140886"],
        rationale="Disabling host defenses — interferon antagonism, innate-immune "
                  "suppression, evasion of pattern-recognition. Drives virulence "
                  "and is the hallmark of dangerous emerging viruses (e.g. Ebola "
                  "VP35/VP24).",
        root_names={
            "GO:0030682": "symbiont-mediated perturbation of host defenses",
            "GO:0042783": "symbiont-mediated evasion of host immune response",
            "GO:0052167": "symbiont-mediated perturbation of host innate immune response",
            "GO:0140886": "symbiont-mediated suppression of host interferon-mediated signaling pathway",
        },
    ),
    DangerCategory(
        key="apoptosis",
        label="Apoptosis manipulation",
        roots=["GO:0052150"],
        rationale="Forcing or blocking host programmed cell death to favor "
                  "replication/spread.",
        root_names={"GO:0052150": "symbiont-mediated perturbation of host apoptosis"},
    ),
    DangerCategory(
        key="host_hijack",
        label="Host gene-expression / machinery hijack",
        roots=["GO:0039656", "GO:0039648", "GO:0039699"],
        rationale="Commandeering host transcription/translation and "
                  "ubiquitin/cap machinery — host shutoff and resource takeover.",
        root_names={
            "GO:0039656": "symbiont-mediated perturbation of host gene expression",
            "GO:0039648": "symbiont-mediated perturbation of host ubiquitin-like protein modification",
            "GO:0039699": "symbiont-mediated evasion of mRNA degradation by host via mRNA cap methylation",
        },
    ),
]


def all_roots() -> list[str]:
    """Flat list of every danger root id (for assertions/tests)."""
    return [r for cat in DANGER_CATEGORIES for r in cat.roots]
