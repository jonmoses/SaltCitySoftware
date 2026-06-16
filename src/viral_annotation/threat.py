"""Threat characterization: turn predicted GO terms into a danger profile.

Given a proteome annotated by the GO classifier (`classifier.serving.GoAnnotator`),
intersect each protein's predicted terms with the curated danger ontology
(`data.danger_terms`) expanded over the DAG, and roll the hits up into a
per-protein and per-proteome threat profile.

The output is a **triage signal**, not a determination: a danger hit is a hypothesis
that a protein participates in a harmful mechanism.

IMPORTANT — base rates. High-level viral terms ("symbiont entry into host cell",
"perturbation of host innate immune response") sit on *most* viral proteins after
true-path propagation, so their absolute probability is high for almost everything
(the same effect that flatters Naive in the NetGO benchmark, docs/06). A raw
threshold therefore flags every protein and is useless. We report two views:

  * **category fingerprint** — peak absolute confidence per danger category: "which
    dangerous mechanisms are present in this proteome" (comparative across viruses);
  * **standout proteins** — ranked by *lift over the proteome background* (a term's
    mean predicted probability across this proteome), which surfaces the specific
    proteins that drive a mechanism above the viral crowd (the entry glycoprotein,
    the interferon antagonist), not the universal baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from viral_annotation.data.danger_terms import DANGER_CATEGORIES


# --- danger map (roots -> descendant term sets) -----------------------------
def build_danger_map(dag, categories=DANGER_CATEGORIES) -> dict[str, frozenset[str]]:
    """Expand each danger category's roots to all descendant terms over `dag`.

    `categories` selects the pathogen-domain ontology (viral by default; pass
    `danger_categories("bacterial")` for the bacterial set). Asserts every root is
    present and non-obsolete (GoDag drops obsolete terms on load, so a missing root
    means it was obsoleted/renamed — fail loudly rather than silently characterizing
    nothing).
    """
    out: dict[str, frozenset[str]] = {}
    for cat in categories:
        terms: set[str] = set()
        for root in cat.roots:
            if root not in dag:
                raise ValueError(
                    f"danger root {root} ({cat.root_names.get(root, '?')}) absent from "
                    f"the GO DAG — likely obsoleted/renamed; update data/danger_terms.py"
                )
            terms |= dag.descendants(root)
        out[cat.key] = frozenset(terms)
    return out


def _danger_terms(danger_map) -> set[str]:
    out: set[str] = set()
    for s in danger_map.values():
        out |= s
    return out


def background_rates(annotated_list, danger_map) -> dict[str, float]:
    """Mean predicted probability per danger term across the proteome (missing=0).

    The "average viral protein" baseline a standout must clear — self-contained, so
    no training-set refetch is needed.
    """
    terms = _danger_terms(danger_map)
    n = max(len(annotated_list), 1)
    total: dict[str, float] = {t: 0.0 for t in terms}
    for a in annotated_list:
        for t, p in a.terms.items():
            if t in total:
                total[t] += p
    return {t: total[t] / n for t in terms}


# --- result types -----------------------------------------------------------
@dataclass
class TermHit:
    """A single predicted danger term, with its lift over the proteome background."""

    go_id: str
    name: str
    prob: float
    namespace: str
    lift: float = 0.0  # prob - background mean for this term across the proteome


@dataclass
class CategoryHit:
    """All danger terms a protein hit within one category, sorted by confidence."""

    key: str
    label: str
    terms: list[TermHit] = field(default_factory=list)

    @property
    def peak(self) -> float:
        return max((t.prob for t in self.terms), default=0.0)

    @property
    def peak_lift(self) -> float:
        return max((t.lift for t in self.terms), default=0.0)


@dataclass
class ProteinThreat:
    """A protein's danger profile: category key -> CategoryHit (only ones that fired)."""

    accession: str
    organism: str
    categories: dict[str, CategoryHit] = field(default_factory=dict)

    def peak(self) -> float:
        return max((c.peak for c in self.categories.values()), default=0.0)

    def standout(self) -> float:
        """Max lift over background across categories — the triage ranking score."""
        return max((c.peak_lift for c in self.categories.values()), default=0.0)


@dataclass
class ProteomeThreat:
    """Proteome-level roll-up across all proteins."""

    name: str
    n_proteins: int
    proteins: list[ProteinThreat]
    background: dict[str, float] = field(default_factory=dict)
    standout_threshold: float = 0.15
    # The danger-category set this profile was built against (viral by default).
    categories: list = field(default_factory=lambda: DANGER_CATEGORIES)

    def category_peaks(self) -> dict[str, float]:
        """Category key -> max confidence seen anywhere in the proteome."""
        peaks = {cat.key: 0.0 for cat in self.categories}
        for p in self.proteins:
            for key, ch in p.categories.items():
                peaks[key] = max(peaks[key], ch.peak)
        return peaks

    def ranked(self, key: str | None = None, n: int = 5) -> list[tuple[ProteinThreat, float]]:
        """Proteins ranked by lift — overall (key=None) or within one category."""
        if key is None:
            scored = [(p, p.standout()) for p in self.proteins if p.categories]
        else:
            scored = [(p, p.categories[key].peak_lift)
                      for p in self.proteins if key in p.categories]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]


# --- characterization -------------------------------------------------------
def characterize_protein(annotated, danger_map, dag, background=None,
                         display_floor: float = 0.0,
                         categories=DANGER_CATEGORIES) -> ProteinThreat:
    """Intersect one AnnotatedProtein's terms with each danger category."""
    background = background or {}
    pt = ProteinThreat(accession=annotated.accession, organism=annotated.organism)
    for cat in categories:
        hits = [
            TermHit(go_id=t, name=(dag.get(t).name if dag.get(t) else t),
                    prob=prob, namespace=(dag.namespace_of(t) or ""),
                    lift=prob - background.get(t, 0.0))
            for t, prob in annotated.terms.items()
            if t in danger_map[cat.key] and prob >= display_floor
        ]
        if hits:
            hits.sort(key=lambda h: h.prob, reverse=True)
            pt.categories[cat.key] = CategoryHit(key=cat.key, label=cat.label, terms=hits)
    return pt


def characterize_proteome(name, annotated_list, danger_map, dag,
                          standout_threshold: float = 0.15,
                          display_floor: float = 0.05,
                          categories=DANGER_CATEGORIES) -> ProteomeThreat:
    """Characterize every annotated protein and roll up to a ProteomeThreat."""
    background = background_rates(annotated_list, danger_map)
    proteins = [characterize_protein(a, danger_map, dag, background, display_floor, categories)
                for a in annotated_list]
    return ProteomeThreat(name=name, n_proteins=len(annotated_list), proteins=proteins,
                          background=background, standout_threshold=standout_threshold,
                          categories=categories)


# --- reporting --------------------------------------------------------------
def category_label(key: str, categories=DANGER_CATEGORIES) -> str:
    for cat in categories:
        if cat.key == key:
            return cat.label
    return key


def format_report(pt: ProteomeThreat, max_terms: int = 4) -> str:
    """Human-readable per-proteome threat report (fingerprint + standouts)."""
    lines: list[str] = []
    lines.append(f"\n{'=' * 72}")
    lines.append(f"THREAT PROFILE — {pt.name}  ({pt.n_proteins} proteins annotated)")
    lines.append(f"{'=' * 72}")

    peaks = pt.category_peaks()
    lines.append("Danger-category fingerprint (peak confidence; which mechanisms are present),")
    lines.append("with the proteins that drive each ABOVE this proteome's baseline (lift):")
    any_standout = False
    for cat in pt.categories:
        peak = peaks[cat.key]
        bar = "#" * int(round(peak * 24))
        lines.append(f"\n  {cat.label:42s} {peak:5.2f} {bar}")
        for p, lift in pt.ranked(cat.key, n=3):
            if lift < pt.standout_threshold:
                break
            any_standout = True
            ch = p.categories[cat.key]
            top = "; ".join(f"{t.name} ({t.prob:.2f})" for t in ch.terms[:max_terms])
            lines.append(f"        {p.accession:12s} lift {lift:+.2f} | {top}")
    if not any_standout:
        lines.append("\n  (no protein rises notably above background — homogeneous proteome)")
    lines.append("\n  note: every viral protein scores high on entry/immune terms by base rate;")
    lines.append("  the peak shows a mechanism is PRESENT, the lift shows WHICH proteins are")
    lines.append("  distinctive for it (the entry glycoprotein, the interferon antagonist).")
    return "\n".join(lines)


def to_dict(pt: ProteomeThreat) -> dict:
    """JSON-serializable proteome threat record."""
    return {
        "name": pt.name,
        "n_proteins": pt.n_proteins,
        "standout_threshold": pt.standout_threshold,
        "category_peaks": pt.category_peaks(),
        "background": {t: r for t, r in pt.background.items() if r > 0.0},
        "proteins": [
            {
                "accession": p.accession,
                "organism": p.organism,
                "peak": p.peak(),
                "standout": p.standout(),
                "categories": {
                    key: {
                        "label": ch.label,
                        "peak": ch.peak,
                        "peak_lift": ch.peak_lift,
                        "terms": [
                            {"go_id": t.go_id, "name": t.name, "prob": t.prob,
                             "lift": t.lift, "namespace": t.namespace}
                            for t in ch.terms
                        ],
                    }
                    for key, ch in p.categories.items()
                },
            }
            for p in sorted(pt.proteins, key=lambda x: x.standout(), reverse=True)
            if p.categories
        ],
    }
