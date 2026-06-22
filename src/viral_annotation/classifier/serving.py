"""Load the persisted GO classifier and annotate new protein sequences.

Training persists per-namespace heads to ``models/go_classifier.pt`` (state dicts
keyed by namespace) + ``go_classifier.meta.json`` (esm model, pooling, and each
head's ordered vocab). This is the serving counterpart: rebuild each head, embed
new sequences with the *same* ESM config, predict, and apply true-path correction —
so a caller can annotate an arbitrary virus proteome without retraining.

The forward path mirrors training exactly (`embed_records` + `predict_proba` +
`GoDag.correct_scores`), so serving and evaluation can't silently diverge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from viral_annotation.classifier.model import build_classifier, predict_proba
from viral_annotation.config import GO_OBO_PATH, MODELS_DIR
from viral_annotation.embeddings.cache import embed_records
from viral_annotation.ontology import GoDag


@dataclass
class AnnotatedProtein:
    """One annotated protein: accession + corrected GO-term probabilities."""

    accession: str
    sequence: str
    organism: str = ""
    # GO id -> hierarchically-corrected probability, terms below threshold dropped.
    terms: dict[str, float] = field(default_factory=dict)


class GoAnnotator:
    """A loaded GO classifier ready to annotate sequences."""

    def __init__(self, heads, terms_by_ns, esm_model, pooling, dag, correct=True):
        self.heads = heads                # ns -> torch.nn.Module (eval mode)
        self.terms_by_ns = terms_by_ns    # ns -> list[str] (column order)
        self.esm_model = esm_model
        self.pooling = pooling
        self.dag = dag
        self.correct = correct            # apply true-path correction (off for leaf-only models)

    @classmethod
    def load(cls, models_dir=MODELS_DIR, dag=None, obo_path=GO_OBO_PATH) -> "GoAnnotator":
        """Rebuild the per-namespace heads from the saved state dicts + meta."""
        import torch

        models_dir = Path(models_dir)
        meta = json.loads((models_dir / "go_classifier.meta.json").read_text())
        states = torch.load(models_dir / "go_classifier.pt", map_location="cpu")
        dag = dag or GoDag.from_obo(obo_path)
        hidden_dims = meta.get("hidden_dims") or []

        heads, terms_by_ns = {}, {}
        for ns, info in meta["namespaces"].items():
            terms = info["terms"]
            state = states[ns]
            # Infer the embedding width from the first linear weight [out, in], so
            # we don't have to re-derive the pooling's feature dimension here.
            first_w = next(v for k, v in state.items() if k.endswith("weight"))
            model = build_classifier(int(first_w.shape[1]), len(terms), hidden_dims=hidden_dims)
            model.load_state_dict(state)
            model.eval()
            heads[ns], terms_by_ns[ns] = model, terms

        # Leaf-only models carry no hierarchy, so skip true-path correction at serve.
        correct = meta.get("hierarchical_correction", not meta.get("leaf_only", False))
        return cls(heads, terms_by_ns, meta["esm_model"], meta["pooling"], dag, correct)

    def annotate(self, records, threshold: float = 0.01) -> list[AnnotatedProtein]:
        """Annotate `records` (objects with .accession/.sequence) -> AnnotatedProtein.

        Embeds once with the model's pooling, predicts every head, merges the
        per-namespace probabilities into one term->prob map per protein, applies
        true-path correction (unless the model is leaf-only), and drops terms below
        `threshold`.
        """
        records = list(records)
        if not records:
            return []
        # One shared embed pass (cached, windowed for long proteins).
        _, X = embed_records(records, self.esm_model, self.pooling, None, window=True)
        per_ns_prob = {ns: predict_proba(self.heads[ns], X) for ns in self.heads}
        return _assemble(records, self.terms_by_ns, per_ns_prob, self.dag, threshold, self.correct)


def _assemble(records, terms_by_ns, per_ns_prob, dag, threshold, correct=True) -> list[AnnotatedProtein]:
    """Merge per-namespace [P x N] probabilities -> (optionally corrected) thresholded annotations."""
    out: list[AnnotatedProtein] = []
    for i, r in enumerate(records):
        scores: dict[str, float] = {}
        for ns, terms in terms_by_ns.items():
            probs = per_ns_prob[ns][i]
            for col, term in enumerate(terms):
                scores[term] = float(probs[col])
        corrected = dag.correct_scores(scores) if correct else scores
        kept = {t: p for t, p in corrected.items() if p >= threshold}
        out.append(AnnotatedProtein(
            accession=getattr(r, "accession", getattr(r, "id", "")),
            sequence=r.sequence,
            organism=getattr(r, "organism", ""),
            terms=kept,
        ))
    return out


class FinetunedAnnotator:
    """Serving wrapper for a LoRA fine-tuned model (`--finetune lora`).

    Heavier than `GoAnnotator`: it loads the full ESM-2 backbone + LoRA adapter and
    runs it end-to-end, so it can't be served by the lightweight state-dict loader.
    The forward path (predict -> true-path correction -> threshold) is otherwise
    identical, so serving and training can't diverge.
    """

    def __init__(self, model, tokenizer, terms_by_ns, dag, max_length, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.terms_by_ns = terms_by_ns
        self.dag = dag
        self.max_length = max_length
        self.device = device

    @classmethod
    def load(cls, model_dir, dag=None, obo_path=GO_OBO_PATH, device="cpu") -> "FinetunedAnnotator":
        from viral_annotation.training.finetune import load_finetuned

        model_dir = Path(model_dir)
        model, tokenizer, meta = load_finetuned(model_dir, device=device)
        dag = dag or GoDag.from_obo(obo_path)
        terms_by_ns = {ns: info["terms"] for ns, info in meta["namespaces"].items()}
        return cls(model, tokenizer, terms_by_ns, dag, meta["max_length"], device)

    def annotate(self, records, threshold: float = 0.01) -> list[AnnotatedProtein]:
        from viral_annotation.training.finetune import _predict_all

        records = list(records)
        if not records:
            return []
        per_ns_prob = _predict_all(self.model, records, self.tokenizer, self.device,
                                   self.max_length, batch_size=8)
        return _assemble(records, self.terms_by_ns, per_ns_prob, self.dag, threshold)
