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

Reality check (against the shipped viral model's 695-term vocab): the viral danger
signal lives almost entirely in **biological_process** (host-interaction mechanisms).
Classic toxin MF terms are absent from the viral training vocab, so that category
rarely fires for viruses — but it is central to the BACTERIAL set below (anthrax /
diphtheria / cholera toxins), along with secretion systems, antimicrobial resistance,
iron piracy, and biofilm. Categories are domain-keyed (`danger_categories(domain)`);
the threat engine in `threat.py` is given a category list and is otherwise identical
across domains.
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
VIRAL_DANGER_CATEGORIES: list[DangerCategory] = [
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


# Bacterial pathogenicity. Distinct mechanisms from viruses: classic protein
# TOXINS (which viruses lack — the viral toxin category is always 0.00), secretion
# systems that inject effectors, adhesion/invasion, antimicrobial resistance, iron
# piracy, and biofilm persistence; immune-evasion and host-killing roots are shared
# with the viral set (the same symbiont-mediated mechanisms). All ids verified
# present and non-obsolete in go-basic (2026-06); build_danger_map re-asserts at load.
BACTERIAL_DANGER_CATEGORIES: list[DangerCategory] = [
    DangerCategory(
        key="toxin",
        label="Toxin activity",
        roots=["GO:0090729"],
        rationale="Direct molecular toxicity toward host cells/tissues — the "
                  "anthrax / diphtheria / cholera / Shiga toxin class. A central "
                  "bacterial virulence mechanism (viruses lack classic toxins).",
        root_names={"GO:0090729": "toxin activity"},
    ),
    DangerCategory(
        key="secretion",
        label="Secretion-system effector delivery",
        roots=["GO:0030254", "GO:0030255", "GO:0033103", "GO:0044315"],
        rationale="Type III/IV/VI/VII secretion systems inject effector proteins "
                  "into host cells — the molecular syringes of Gram-negative and "
                  "mycobacterial (ESX) pathogens, and a prime intervention target.",
        root_names={
            "GO:0030254": "protein secretion by the type III secretion system",
            "GO:0030255": "protein secretion by the type IV secretion system",
            "GO:0033103": "protein secretion by the type VI secretion system",
            "GO:0044315": "protein secretion by the type VII secretion system",
        },
    ),
    DangerCategory(
        key="adhesion_invasion",
        label="Host adhesion & cell invasion",
        roots=["GO:0044406", "GO:0044650", "GO:0044409", "GO:0085017"],
        rationale="Attaching to and entering host cells — adhesins, and invasion "
                  "via a symbiont-containing vacuole (Salmonella/Listeria style). "
                  "The first step of infection.",
        root_names={
            "GO:0044406": "adhesion of symbiont to host",
            "GO:0044650": "adhesion of symbiont to host cell",
            "GO:0044409": "symbiont entry into host",
            "GO:0085017": "entry into host cell by a symbiont-containing vacuole",
        },
    ),
    DangerCategory(
        key="immune_evasion",
        label="Immune evasion / host-defense subversion",
        roots=["GO:0042783", "GO:0141043", "GO:0030682", "GO:0099018"],
        rationale="Disabling host defenses — innate-immune evasion, perturbation "
                  "of host defenses, restriction-modification evasion. Drives "
                  "virulence and persistence.",
        root_names={
            "GO:0042783": "symbiont-mediated evasion of host immune response",
            "GO:0141043": "symbiont-mediated evasion of host innate immune response",
            "GO:0030682": "symbiont-mediated perturbation of host defenses",
            "GO:0099018": "symbiont-mediated evasion of host restriction-modification system",
        },
    ),
    DangerCategory(
        key="kill_lyse",
        label="Host-cell killing / lysis",
        roots=["GO:0001907", "GO:0001897", "GO:0019835", "GO:0051715"],
        rationale="Killing or lysing host cells — hemolysins and pore-forming "
                  "cytolysins; the basis of tissue damage in acute infection.",
        root_names={
            "GO:0001907": "symbiont-mediated killing of host cell",
            "GO:0001897": "symbiont-mediated cytolysis of host cell",
            "GO:0019835": "cytolysis",
            "GO:0051715": "cytolysis in another organism",
        },
    ),
    DangerCategory(
        key="amr",
        label="Antimicrobial resistance",
        roots=["GO:0046677", "GO:0008800"],
        rationale="Resistance to antibiotics — beta-lactamases and the broader "
                  "antibiotic-response machinery. A treatability danger axis "
                  "specific to bacterial threats.",
        root_names={
            "GO:0046677": "response to antibiotic",
            "GO:0008800": "beta-lactamase activity",
        },
    ),
    DangerCategory(
        key="iron_piracy",
        label="Iron / nutrient piracy",
        roots=["GO:0019290", "GO:0015891"],
        rationale="Siderophore-mediated iron acquisition — wresting iron from host "
                  "sequestration (nutritional immunity), an established virulence "
                  "requirement for many pathogens.",
        root_names={
            "GO:0019290": "siderophore biosynthetic process",
            "GO:0015891": "siderophore transport",
        },
    ),
    DangerCategory(
        key="biofilm",
        label="Biofilm / persistence",
        roots=["GO:0042710"],
        rationale="Biofilm formation — chronic, antibiotic-tolerant, immune-evasive "
                  "communities underlying persistent and device-associated infection.",
        root_names={"GO:0042710": "biofilm formation"},
    ),
]


# Domain -> its danger categories. `DANGER_CATEGORIES` stays bound to the viral set
# for back-compat (existing imports/tests); new code selects via danger_categories().
DANGER_CATEGORIES_BY_DOMAIN: dict[str, list[DangerCategory]] = {
    "viral": VIRAL_DANGER_CATEGORIES,
    "bacterial": BACTERIAL_DANGER_CATEGORIES,
}
DANGER_CATEGORIES = VIRAL_DANGER_CATEGORIES


def danger_categories(domain: str = "viral") -> list[DangerCategory]:
    """The danger-category set for a pathogen domain."""
    if domain not in DANGER_CATEGORIES_BY_DOMAIN:
        raise KeyError(
            f"no danger ontology for domain {domain!r}; "
            f"choose from {list(DANGER_CATEGORIES_BY_DOMAIN)}"
        )
    return DANGER_CATEGORIES_BY_DOMAIN[domain]


def all_roots(categories: list[DangerCategory] | None = None) -> list[str]:
    """Flat list of every danger root id in `categories` (for assertions/tests).

    Defaults to the viral set; pass a domain's list to check it instead.
    """
    cats = categories if categories is not None else VIRAL_DANGER_CATEGORIES
    return [r for cat in cats for r in cat.roots]
