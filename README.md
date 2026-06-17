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
| `training.{pipeline,heads,train}` — one config-driven trainer: pooling (mean/stats/attention/per-namespace) + optional homology ensemble, cluster split, test + zero-shot | **Working** (`[ml]`); `va-train [--pooling P] [--ensemble homology]` |
| `evaluation.report` — shared per-namespace + overall Fmax-vs-naive tables | **Working** |
| `benchmark.run` — virus-only NetGO temporal benchmark (Fmax/M-AUPR/Smin) | **Working** (`[ml]`); `va-benchmark`; see `docs/06-benchmark.md` |
| `classifier.serving` — load saved heads + annotate new sequences | **Working** (`[ml]`) |
| `data.proteomes` — fetch a target virus proteome (TrEMBL incl.) by taxon | **Working** |
| `threat` / `data.danger_terms` — map predicted GO → danger categories | **Working, tested** |
| `cli.threat` — **Stage 3 demo**: annotate a virus → threat profile | **Working** (`[ml]`); `va-threat --panel`; see `docs/07-threat-characterization.md` |
| `config.PathogenDomain` — per-domain profiles (viral/bacterial): taxon, family holdout, evidence/pooling policy, danger ontology, target panel, models dir | **Working**; `va-train --domain bacterial`, `va-threat --domain bacterial`; bacterial model not yet trained; see `docs/08-bacterial-extension.md` |
| localization / enrichment | Planned — see `docs/01-annotation-pipeline-design.md` |

### GO classifier — full-set result (17,517 viral reviewed proteins)

ESM-2 650M, **per-namespace evidence policy** AND **per-namespace pooling**
(`va-train --pooling per-namespace`), manual-only test labels, hierarchically corrected,
**30%-identity cluster split** (MMseqs2), **Coronaviridae held out** for zero-shot,
seeded. Every number is reported against a **Naive baseline** (predict each term's
training frequency) — the floor a real model must clear. `lift` = model − naive.

| Namespace | N | pooling | test (naive) | zero-shot (naive) |
|-----------|---|---------|--------------|-------------------|
| Molecular Function | 45  | **attention** | 0.186 (0.135) | **0.463 (0.344)** |
| Biological Process | 545 | mean          | 0.355 (0.293) | 0.190 (0.192) |
| Cellular Component | 105 | mean          | 0.232 (0.205) | 0.277 (0.321) |
| **overall**        | 695 | —             | **0.391** (0.293, **+0.10**) | 0.278 (0.287, −0.01) |

**In-distribution: overall 0.391, +0.10 over naive** — the best single-model config
(mean 0.376, stats 0.357).

#### Ensemble — pLM + homology (BLAST-KNN), `va-train --ensemble homology`

Late-fusion of the pLM heads with a homology component (MMseqs2 search → bitscore-
weighted transfer of neighbours' manual labels), weights grid-searched per namespace
on validation. **This closes the zero-shot gap:**

| | test (naive 0.293) | zero-shot (naive 0.287) |
|---|---|---|
| pLM-only | 0.388 (+0.10) | 0.277 (**−0.01**, below prior) |
| **+ homology** | **0.412 (+0.12)** | **0.345 (+0.06**, above prior) |

Homology flips overall zero-shot from below the naive prior to clearly above it
(driven by BP, 0.356→0.398) — the system recovers an **unseen viral family's**
function above base rates, not just MF. A third component (InterPro2GO via the
UniProt IEA proxy, `terms_iea`) is wired in but gets weight 0: validation-tuned
weights don't reward its specifically-zero-shot value (a zero-shot tuning holdout
would be the fix). Homology alone closes the gap.

**Zero-shot is the interesting result.** Held-out coronavirus **molecular function
recovers strongly — 0.463, +0.12 over naive** — because learned attention pooling
locks onto conserved catalytic/binding residues that transfer across an unseen
family. BP/CC do *not* beat their priors zero-shot, and they dominate the term count
(650 of 695), so the *aggregate* zero-shot sits at naive (−0.01). The honest reading:
**what a viral protein *does* (molecular function) is recoverable zero-shot from
sequence alone; where it localizes and which process it joins are not.** MF is the
most actionable annotation for threat characterization.

**Why per-namespace pooling:** a full comparison (project memory: pooling-comparison)
showed learned attention pooling wins zero-shot MF decisively (0.46 vs 0.25 mean /
0.37 stats), while mean is best for BP/CC and far cheaper (attention needs a ~20 GB
per-residue cache). So MF uses attention, BP/CC use mean.

**Why per-namespace evidence:** a joint head trained on IEA-dominated labels
collapsed MF to 0.09 — viral IEA-MF (domain-rule ligand binding) is nearly disjoint
from manual-MF (curated protein binding). MF trains manual-only; BP/CC asymmetric.
See `docs/01` + project memory.

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
.venv/bin/va-download-go   # pulls go-basic.obo into data/
```

Then load and propagate (see `tests/test_go_dag.py` for usage).
