"""Gene Ontology DAG: parse go-basic.obo and apply the true-path rule.

The true-path rule is central to GO-based annotation (docs/01, docs/02): if a
protein is annotated with a term, it is implicitly annotated with all that term's
ancestors. We therefore:

  * propagate training labels UP the DAG before training, and
  * correct predicted probabilities so a parent is never less likely than its
    most-likely child.

This module is pure stdlib so it runs and is tested without any heavy install.
Only `is_a` and `part_of` edges are followed (config.PROPAGATION_RELATIONS),
which is the standard, safe choice on the go-basic graph.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from viral_annotation.config import PROPAGATION_RELATIONS


@dataclass
class Term:
    """A single GO term and its direct parents (is_a + part_of)."""

    id: str
    name: str = ""
    namespace: str = ""
    parents: set[str] = field(default_factory=set)
    is_obsolete: bool = False
    # Term ids that this term replaces (alt_id), so lookups resolve old ids.
    alt_ids: set[str] = field(default_factory=set)


class GoDag:
    """In-memory GO DAG with ancestor lookup and true-path operations."""

    def __init__(self, terms: dict[str, Term]):
        self._terms = terms
        # Map alt_id -> primary id so callers can use either.
        self._alias: dict[str, str] = {}
        for t in terms.values():
            for alt in t.alt_ids:
                self._alias[alt] = t.id
        self._ancestor_cache: dict[str, frozenset[str]] = {}
        # Child index (inverse of parents), built lazily on first descendants() call.
        self._children: dict[str, set[str]] | None = None
        self._descendant_cache: dict[str, frozenset[str]] = {}

    # --- construction -------------------------------------------------------
    @classmethod
    def from_obo(cls, path: str | Path, include_obsolete: bool = False) -> "GoDag":
        """Parse an OBO file (e.g. go-basic.obo) into a GoDag."""
        terms = _parse_obo(Path(path))
        if not include_obsolete:
            terms = {tid: t for tid, t in terms.items() if not t.is_obsolete}
            # Drop parent edges pointing at removed obsolete terms.
            valid = set(terms)
            for t in terms.values():
                t.parents &= valid
        return cls(terms)

    # --- basic access -------------------------------------------------------
    def __len__(self) -> int:
        return len(self._terms)

    def __contains__(self, term_id: str) -> bool:
        return self.resolve(term_id) in self._terms

    def resolve(self, term_id: str) -> str:
        """Resolve a possibly-secondary (alt) id to its primary id."""
        return self._alias.get(term_id, term_id)

    def get(self, term_id: str) -> Term | None:
        return self._terms.get(self.resolve(term_id))

    def namespace_of(self, term_id: str) -> str | None:
        t = self.get(term_id)
        return t.namespace if t else None

    # --- ancestors / true-path ---------------------------------------------
    def ancestors(self, term_id: str, include_self: bool = True) -> frozenset[str]:
        """All ancestors of a term via is_a/part_of, memoized.

        include_self=True returns the term itself plus its ancestors, which is
        the set you union in for true-path propagation.
        """
        primary = self.resolve(term_id)
        if primary not in self._terms:
            return frozenset({primary}) if include_self else frozenset()

        if primary not in self._ancestor_cache:
            seen: set[str] = set()
            queue: deque[str] = deque(self._terms[primary].parents)
            while queue:
                cur = queue.popleft()
                if cur in seen or cur not in self._terms:
                    continue
                seen.add(cur)
                queue.extend(self._terms[cur].parents)
            self._ancestor_cache[primary] = frozenset(seen)

        anc = self._ancestor_cache[primary]
        return frozenset(anc | {primary}) if include_self else anc

    def descendants(self, term_id: str, include_self: bool = True) -> frozenset[str]:
        """All descendants of a term via is_a/part_of (the inverse of ancestors).

        Memoized like ``ancestors``. Used to expand a high-level "root" term (e.g.
        a danger category like toxin activity) to every specific term beneath it,
        so a prediction on any descendant counts as a hit for that category.
        include_self=True returns the term itself plus its descendants.
        """
        primary = self.resolve(term_id)
        if primary not in self._terms:
            return frozenset({primary}) if include_self else frozenset()

        if self._children is None:
            self._children = {}
            for t in self._terms.values():
                for parent in t.parents:
                    self._children.setdefault(parent, set()).add(t.id)

        if primary not in self._descendant_cache:
            seen: set[str] = set()
            queue: deque[str] = deque(self._children.get(primary, ()))
            while queue:
                cur = queue.popleft()
                if cur in seen:
                    continue
                seen.add(cur)
                queue.extend(self._children.get(cur, ()))
            self._descendant_cache[primary] = frozenset(seen)

        desc = self._descendant_cache[primary]
        return frozenset(desc | {primary}) if include_self else desc

    def propagate(self, term_ids: Iterable[str]) -> set[str]:
        """True-path propagation: a label set -> set closed under ancestors.

        Use on each training protein's annotations before building label vectors
        (docs/01 Step 3).
        """
        out: set[str] = set()
        for tid in term_ids:
            out |= self.ancestors(tid, include_self=True)
        return out

    def most_specific(self, term_ids: Iterable[str]) -> frozenset[str]:
        """Reduce a label set to its lowest-level (leaf-of-set) terms.

        The inverse intent of `propagate`: instead of adding ancestors, drop any
        term that is a *proper ancestor* of another term in the same set, keeping
        only the most specific annotations a protein actually carries. Used to
        build leaf-only training labels (no hierarchy, no true-path rule).

        Terms absent from the (non-obsolete) DAG are dropped, mirroring
        `propagate`. A term with no descendants in the set survives; a parent that
        is redundant given a more specific child does not.
        """
        present = {self.resolve(t) for t in term_ids}
        present = {t for t in present if t in self._terms}
        redundant: set[str] = set()
        for t in present:
            redundant |= self.ancestors(t, include_self=False) & present
        return frozenset(present - redundant)

    def correct_scores(self, scores: dict[str, float]) -> dict[str, float]:
        """Hierarchical (post-hoc) correction of predicted probabilities.

        Enforces that a parent's probability >= max over its children's, so the
        output never violates the DAG (docs/01 Step 4). Implemented as: each
        term's corrected score = max of its own score and all *descendant*
        scores present in `scores`.

        Returns a new dict over exactly the input term ids.
        """
        # Build descendant relation only over the terms we actually scored,
        # by inverting ancestor lookups within the keyed set.
        keys = [self.resolve(t) for t in scores]
        norm = {self.resolve(t): s for t, s in scores.items()}
        corrected = dict(norm)
        for child in keys:
            child_score = norm[child]
            for anc in self.ancestors(child, include_self=False):
                if anc in corrected and child_score > corrected[anc]:
                    corrected[anc] = child_score
        return corrected


# --- OBO parsing ------------------------------------------------------------
def _parse_obo(path: Path) -> dict[str, Term]:
    """Minimal OBO 1.2 parser: enough of the [Term] stanza for the DAG.

    Reads id, name, namespace, is_a, relationship: part_of, is_obsolete, alt_id.
    Ignores everything else. Streams line-by-line so a full go-basic.obo
    (~40k terms) parses without loading structure beyond the term dict.
    """
    terms: dict[str, Term] = {}
    cur: Term | None = None
    in_term = False

    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("["):
                # Stanza boundary: commit the previous term, start fresh.
                if cur is not None and in_term:
                    terms[cur.id] = cur
                in_term = line == "[Term]"
                cur = None
                continue
            if not in_term or not line or line.startswith("!"):
                continue

            key, _, value = line.partition(":")
            value = value.strip()
            # Strip trailing "! comment" that OBO appends to id-bearing fields.
            value_id = value.split("!", 1)[0].strip()

            if key == "id":
                cur = Term(id=value_id)
            elif cur is None:
                continue
            elif key == "name":
                cur.name = value
            elif key == "namespace":
                cur.namespace = value
            elif key == "is_a":
                cur.parents.add(value_id)
            elif key == "alt_id":
                cur.alt_ids.add(value_id)
            elif key == "is_obsolete":
                cur.is_obsolete = value.lower() == "true"
            elif key == "relationship":
                rel, _, target = value.partition(" ")
                if rel in PROPAGATION_RELATIONS:
                    cur.parents.add(target.split("!", 1)[0].strip())

        if cur is not None and in_term:
            terms[cur.id] = cur

    return terms
