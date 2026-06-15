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
| `embeddings.esm` — ESM-2 mean-pooled embeddings | Scaffolded; needs `[ml]` deps |
| `classifier.model` — multi-label head | Scaffolded; architecture TBD by sweep |
| localization / enrichment | Planned — see `docs/01-annotation-pipeline-design.md` |

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
