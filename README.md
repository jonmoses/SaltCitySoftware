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
| `data.cluster` — MMseqs2 30%-identity clustering | **Working** (needs `mmseqs`) |
| `data.split` — asymmetric random + cluster split + family holdout | **Working, tested** |
| `data.dataset` — term-vocab selection + multi-hot matrices | **Working, tested** |
| `embeddings.esm` / `embeddings.cache` — ESM-2 pooled embeddings, cached | **Working** (`[ml]`); length-safe batching |
| `classifier.model` — linear multi-label head + `predict_proba` | **Working** (`[ml]`); MLP via `hidden_dims` |
| `training.train` — per-namespace heads, cluster split, test + zero-shot Fmax | **Working** (`[ml]`); `python -m viral_annotation.training.train` |
| localization / enrichment | Planned — see `docs/01-annotation-pipeline-design.md` |

### GO classifier — full-set result (17,517 viral reviewed proteins)

Linear heads on ESM-2 650M, **per-namespace evidence policy**, manual-only test
labels, hierarchically corrected, **30%-identity cluster split** (MMseqs2) with
**Coronaviridae held out** for zero-shot.

Every number is reported against a **Naive baseline** (predict each term's training
frequency) — the floor a real model must clear. `lift` = model − naive.

**In-distribution test (leakage-safe cluster split):**

| Namespace | N | Policy | Fmax | naive | lift |
|-----------|---|--------|------|-------|------|
| Molecular Function | 45  | manual-only      | 0.153 | 0.135 | +0.02 |
| Biological Process | 545 | asymmetric       | 0.344 | 0.293 | +0.05 |
| Cellular Component | 105 | asymmetric       | 0.251 | 0.205 | +0.05 |
| **overall**        | 695 | —                | **0.376** | 0.293 | **+0.08** |

In-distribution the model beats the prior by a clear margin (overall +0.08),
driven by BP/CC; **MF barely clears naive** (+0.02) — viral molecular function is
hard (manual-MF ≈ protein binding, closer to the PPI problem; see `docs/01`).

**Zero-shot — held-out Coronaviridae (69 proteins, never trained on):**

| Namespace | Fmax | naive | lift |
|-----------|------|-------|------|
| Molecular Function | 0.249 | 0.344 | −0.09 |
| Biological Process | 0.204 | 0.192 | +0.01 |
| Cellular Component | 0.276 | 0.321 | −0.05 |
| **overall**        | **0.257** | 0.287 | **−0.03** |

**Honest caveat:** against the Naive baseline the model does **not** yet beat the
prior zero-shot (overall −0.04). Coronaviruses are enriched for common, predictable
functions, so the prior is strong; recovering an unseen family's function *better
than base rates* is unsolved here and is real future work — not the win an earlier
draft of this README claimed. (Adding a homology/InterPro ensemble component is the
likely lever; see the NetGO 3.0 lessons.)

**Why per-namespace:** a joint head trained on IEA-dominated labels collapsed MF
to 0.09 — viral IEA-MF (domain-rule ligand binding) is nearly disjoint from
manual-MF (curated protein binding). Training MF manual-only fixes it; independent
heads also lifted BP/CC. See `docs/01` + project memory.

**Rigorous separation:** whole 30%-identity clusters go to one split bucket, so no
test protein has a close homolog in train (the cluster split dropped ~1,570 IEA
homologs of val/test proteins that a random split would have leaked into train).
Coronaviridae is held out entirely. Long proteins (>1022 aa, 7.8% of the set) are
embedded by **non-overlapping windows + length-weighted pooling** rather than
truncated. Runs are seeded; embeddings are cached, so a full re-run is ~90s.

## Setup

```bash
.venv/bin/pip install -e ".[dev,bio,ml]"   # ml is heavy (torch); omit to start light
.venv/bin/pip install -e ".[dev]"          # enough to run the current tests
brew install mmseqs2                        # for the 30%-identity cluster split
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
