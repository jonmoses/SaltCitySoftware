# Viral Protein Annotation Pipeline

Functional-annotation stage of SBIR topic **DPA26BZ03-DV014: Real-Time
Pathogen-Host Interactome Prediction**. Takes a viral protein sequence and
produces functional annotation — GO terms, subcellular localization, and
(set-level) pathway enrichment.

> Design rationale, data sources, evaluation protocol, and references live in
> [`docs/`](docs/README.md). The original solicitation is the root PDF; the
> working design is [`OUTLINE.md`](OUTLINE.md).

## Pipeline (target)

```
sequence (FASTA)
  -> ESM-2 per-residue embeddings        viral_annotation.embeddings.esm
  -> mean-pool to one vector
  -> linear/MLP sigmoid multi-label head viral_annotation.classifier.model
  -> GO-term probabilities
  -> true-path hierarchical correction   viral_annotation.ontology.go_dag
  -> (optional) per-term thresholds
  -> GO annotations  + localization + pathway enrichment
```

## What's implemented now

| Module | Status |
|--------|--------|
| `ontology.go_dag` — OBO parse, ancestor lookup, true-path propagation/correction | **Working, tested** (pure stdlib) |
| `evaluation.metrics` — protein-centric Fmax | **Working, tested** (pure stdlib) |
| `data.download` — fetch `go-basic.obo` | **Working** (stdlib urllib) |
| `data.fasta` — FASTA read/write | **Working** (stdlib fallback) |
| `config` — model registry, paths, constants | **Working** |
| `data.labels` — UniProt fetch + manual/IEA evidence tiers + propagation | **Working, tested** |
| `data.split` — asymmetric random split (val/test manual-only) | **Working, tested** |
| `data.dataset` — term-vocab selection + multi-hot matrices | **Working, tested** |
| `embeddings.esm` / `embeddings.cache` — ESM-2 pooled embeddings, cached | **Working** (`[ml]`); length-safe batching |
| `classifier.model` — linear multi-label head + `predict_proba` | **Working** (`[ml]`); MLP via `hidden_dims` |
| `training.train` — end-to-end train + per-namespace Fmax eval | **Working** (`[ml]`); `python -m viral_annotation.training.train` |
| localization / enrichment | Planned — see `docs/01-annotation-pipeline-design.md` |

### First GO classifier — dry-run result (400 proteins)

Validated end-to-end on a 400-protein sample (linear head, ESM-2 650M, manual-only
test labels, hierarchically corrected). Fmax: MF 0.25, BP 0.23, CC 0.13, overall
0.27. These are underfit (val Fmax still climbing at the epoch cap; tiny sample) —
a machinery check, **not** final numbers. Real numbers need the full set, more
epochs, and the cluster split (see plan / `docs/03`).

## Setup

```bash
.venv/bin/pip install -e ".[dev,bio,ml]"   # ml is heavy (torch); omit to start light
.venv/bin/pip install -e ".[dev]"          # enough to run the current tests
```

## Run the tests

```bash
.venv/bin/python -m pytest
```

## First real step

```bash
.venv/bin/python -m viral_annotation.data.download   # pulls go-basic.obo into data/
```

Then load and propagate (see `tests/test_go_dag.py` for usage).
