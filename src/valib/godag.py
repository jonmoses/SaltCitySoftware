"""Gene Ontology DAG: parse go-basic.obo and apply the true-path rule.

The true-path rule: if a protein is annotated with a term, it is implicitly
annotated with all that term's ancestors. We use this to (a) propagate training
labels up the DAG and (b) correct predicted probabilities so a parent is never
less likely than its most-likely descendant. Pure stdlib: runs and tests with no
heavy install. Only `is_a` and `part_of` edges are followed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Relations treated as "child -> parent" edges for propagation. Standard safe
# choice on the go-basic graph; kept local so valib has no config dependency.
PROPAGATION_RELATIONS = ("is_a", "part_of")


@dataclass
class Term:
    """A single GO term and its direct parents (is_a + part_of)."""

    id: str
    name: str = ""
    namespace: str = ""
    parents: set[str] = field(default_factory=set)
    is_obsolete: bool = False
    # Term ids this term replaces (alt_id), so lookups resolve old ids.
    alt_ids: set[str] = field(default_factory=set)


# Pre:  term is the [Term] stanza being built; key/value are one parsed OBO line
#       with the trailing "! comment" already stripped from value_id.
# Post: term is mutated in place to record the field; unknown keys are ignored.
# Inputs:  term (Term); key (str); value (str) raw; value_id (str) id-only form
# Outputs: None (term mutated)
def _apply_field(term: Term, key: str, value: str, value_id: str) -> None:
    if key == "name":
        term.name = value
    elif key == "namespace":
        term.namespace = value
    elif key == "is_a":
        term.parents.add(value_id)
    elif key == "alt_id":
        term.alt_ids.add(value_id)
    elif key == "is_obsolete":
        term.is_obsolete = value.lower() == "true"
    elif key == "relationship":
        rel, _, target = value.partition(" ")
        if rel in PROPAGATION_RELATIONS:
            term.parents.add(target.split("!", 1)[0].strip())


# Pre:  path points at a readable OBO 1.2 file (e.g. go-basic.obo).
# Post: returns every [Term] stanza as a Term; obsolete terms are included here
#       (callers filter). Streams line-by-line; non-[Term] stanzas are skipped.
# Inputs:  path (Path) — OBO file
# Outputs: dict[str, Term] keyed by primary GO id
def _parse_obo(path: Path) -> dict[str, Term]:
    terms: dict[str, Term] = {}
    cur: Term | None = None
    in_term = False
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("["):
                if cur is not None and in_term:
                    terms[cur.id] = cur
                in_term = line == "[Term]"
                cur = None
                continue
            if not in_term or not line or line.startswith("!"):
                continue
            key, _, value = line.partition(":")
            value = value.strip()
            value_id = value.split("!", 1)[0].strip()
            if key == "id":
                cur = Term(id=value_id)
            elif cur is not None:
                _apply_field(cur, key, value, value_id)
        if cur is not None and in_term:
            terms[cur.id] = cur
    return terms


class GoDag:
    """In-memory GO DAG with ancestor/descendant lookup and true-path ops."""

    # Pre:  terms maps primary GO id -> Term (parents already restricted to
    #       valid ids by the caller if obsolete terms were dropped).
    # Post: a GoDag with an alt_id->primary alias map and empty lazy caches.
    # Inputs:  terms (dict[str, Term])
    # Outputs: GoDag
    def __init__(self, terms: dict[str, Term]):
        self._terms = terms
        self._alias: dict[str, str] = {}
        for t in terms.values():
            for alt in t.alt_ids:
                self._alias[alt] = t.id
        self._ancestor_cache: dict[str, frozenset[str]] = {}
        self._children: dict[str, set[str]] | None = None
        self._descendant_cache: dict[str, frozenset[str]] = {}

    # Pre:  path is a readable OBO file.
    # Post: returns a GoDag; if not include_obsolete, obsolete terms and parent
    #       edges pointing at them are removed.
    # Inputs:  path (str | Path); include_obsolete (bool)
    # Outputs: GoDag
    @classmethod
    def from_obo(cls, path: str | Path, include_obsolete: bool = False) -> "GoDag":
        terms = _parse_obo(Path(path))
        if not include_obsolete:
            terms = {tid: t for tid, t in terms.items() if not t.is_obsolete}
            valid = set(terms)
            for t in terms.values():
                t.parents &= valid
        return cls(terms)

    # Pre/Post: trivial accessors.
    # Inputs:  (self)  Outputs: int — number of terms
    def __len__(self) -> int:
        return len(self._terms)

    # Inputs: term_id (str)  Outputs: bool — whether the (resolved) id is known
    def __contains__(self, term_id: str) -> bool:
        return self.resolve(term_id) in self._terms

    # Pre:  term_id may be a primary or secondary (alt) id.
    # Post: returns the primary id; unknown ids pass through unchanged.
    # Inputs: term_id (str)  Outputs: str — primary id
    def resolve(self, term_id: str) -> str:
        return self._alias.get(term_id, term_id)

    # Inputs: term_id (str)  Outputs: Term | None — the term, resolving alt ids
    def get(self, term_id: str) -> Term | None:
        return self._terms.get(self.resolve(term_id))

    # Inputs: term_id (str)  Outputs: str | None — namespace, or None if unknown
    def namespace_of(self, term_id: str) -> str | None:
        t = self.get(term_id)
        return t.namespace if t else None

    # Pre:  term_id may be unknown (returns just itself / empty per include_self).
    # Post: returns all ancestors via is_a/part_of, memoized. include_self adds
    #       the term itself — the set to union for true-path propagation.
    # Inputs:  term_id (str); include_self (bool)
    # Outputs: frozenset[str] of ancestor ids
    def ancestors(self, term_id: str, include_self: bool = True) -> frozenset[str]:
        primary = self.resolve(term_id)
        if primary not in self._terms:
            return frozenset({primary}) if include_self else frozenset()
        if primary not in self._ancestor_cache:
            self._ancestor_cache[primary] = self._walk_ancestors(primary)
        anc = self._ancestor_cache[primary]
        return frozenset(anc | {primary}) if include_self else anc

    # Pre:  primary is a known term id.
    # Post: returns its strict ancestors (excluding itself) via BFS over parents.
    # Inputs:  primary (str)
    # Outputs: frozenset[str] of strict ancestor ids
    def _walk_ancestors(self, primary: str) -> frozenset[str]:
        seen: set[str] = set()
        queue: deque[str] = deque(self._terms[primary].parents)
        while queue:
            cur = queue.popleft()
            if cur in seen or cur not in self._terms:
                continue
            seen.add(cur)
            queue.extend(self._terms[cur].parents)
        return frozenset(seen)

    # Pre:  builds the child index lazily on first call.
    # Post: returns all descendants via is_a/part_of (inverse of ancestors),
    #       memoized. Used to expand a high-level root to every term beneath it.
    # Inputs:  term_id (str); include_self (bool)
    # Outputs: frozenset[str] of descendant ids
    def descendants(self, term_id: str, include_self: bool = True) -> frozenset[str]:
        primary = self.resolve(term_id)
        if primary not in self._terms:
            return frozenset({primary}) if include_self else frozenset()
        if self._children is None:
            self._build_children()
        if primary not in self._descendant_cache:
            self._descendant_cache[primary] = self._walk_descendants(primary)
        desc = self._descendant_cache[primary]
        return frozenset(desc | {primary}) if include_self else desc

    # Pre:  self._children is None.
    # Post: self._children maps parent id -> set of direct child ids.
    # Inputs:  (self)  Outputs: None (sets self._children)
    def _build_children(self) -> None:
        self._children = {}
        for t in self._terms.values():
            for parent in t.parents:
                self._children.setdefault(parent, set()).add(t.id)

    # Pre:  self._children is built; primary is a known term id.
    # Post: returns strict descendants (excluding itself) via BFS over children.
    # Inputs:  primary (str)
    # Outputs: frozenset[str] of strict descendant ids
    def _walk_descendants(self, primary: str) -> frozenset[str]:
        assert self._children is not None
        seen: set[str] = set()
        queue: deque[str] = deque(self._children.get(primary, ()))
        while queue:
            cur = queue.popleft()
            if cur in seen:
                continue
            seen.add(cur)
            queue.extend(self._children.get(cur, ()))
        return frozenset(seen)

    # Pre:  term_ids is any iterable of (possibly alt) GO ids.
    # Post: returns the set closed under ancestors (true-path propagation).
    # Inputs:  term_ids (Iterable[str])
    # Outputs: set[str] of propagated ids
    def propagate(self, term_ids: Iterable[str]) -> set[str]:
        out: set[str] = set()
        for tid in term_ids:
            out |= self.ancestors(tid, include_self=True)
        return out

    # Pre:  scores maps GO id -> predicted probability.
    # Post: returns a new dict over the same ids where each term's score is
    #       raised to >= the max score of any of its descendants present in
    #       scores, so the output never violates the DAG.
    # Inputs:  scores (dict[str, float])
    # Outputs: dict[str, float] corrected scores
    def correct_scores(self, scores: dict[str, float]) -> dict[str, float]:
        norm = {self.resolve(t): s for t, s in scores.items()}
        corrected = dict(norm)
        for child, child_score in norm.items():
            for anc in self.ancestors(child, include_self=False):
                if anc in corrected and child_score > corrected[anc]:
                    corrected[anc] = child_score
        return corrected
