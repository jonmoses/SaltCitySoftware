# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code style (MANDATORY — applies to all code, all agents and subagents)

Every function written for this project MUST follow these rules. They are
non-negotiable and apply equally to any agent or subagent that writes code here.

1. **Preconditions and postconditions are commented before every function.** A
   short comment block immediately above each `def` states what must hold on entry
   (preconditions) and what is guaranteed on exit (postconditions).
2. **Inputs and outputs are explicitly labeled.** The same block has an `Inputs:`
   line and an `Outputs:` line giving the types and meaning of each.
3. **No function longer than ~40 lines** (one printed sheet: one line per statement,
   one line per declaration). Going over requires a VERY justifiable, commented
   reason — default to splitting into named helpers instead.
4. **Simple control flow only.** Straight-line code; simple `if`; simple, bounded
   `for`/`while`. No deep nesting, no clever one-liners, no recursion where a loop
   works. Decompose rather than nest.

Template:

```python
# Pre:  <what must be true of the inputs / state on entry>
# Post: <what is guaranteed on return>
# Inputs:  name (type) — meaning; ...
# Outputs: (type) — meaning
def do_one_thing(...):
    ...
```

## What this is

The **functional-annotation stage** of SBIR topic DPA26BZ03-DV014 (Real-Time
Pathogen-Host Interactome Prediction). Takes a pathogen protein sequence → ESM-2
embedding → multi-label GO-term head → hierarchically-corrected GO annotations,
then maps predictions onto a danger ontology for threat characterization. Package
is `viral_annotation` (in `src/`), despite the work now spanning viral **and**
bacterial domains.

Design rationale, data sources, and evaluation protocol live in `docs/` (numbered
00–08). The original solicitation is the root PDF. Read `docs/01` and `docs/03`
before changing the model or evaluation; read `docs/08` before touching the
domain abstraction.

## Commands

```bash
# Install. `ml` is heavy (torch) — omit it to run the stdlib/numpy-only modules + tests.
.venv/bin/pip install -e ".[dev,bio,ml]"
.venv/bin/pip install -e ".[dev]"          # enough for most tests
brew install mmseqs2                        # needed for the 30%-identity cluster split

.venv/bin/python -m pytest                  # all tests
.venv/bin/python -m pytest tests/test_go_dag.py::test_name   # single test

.venv/bin/va-download-go                    # fetch go-basic.obo into data/ (do this first)
.venv/bin/va-train                          # train (viral, mean pooling, saves servable model)
.venv/bin/va-train --domain bacterial       # bacterial profile
.venv/bin/va-train --pooling per-namespace --ensemble homology   # best-result config
.venv/bin/va-train --limit 400 --no-save    # quick subset dry run
.venv/bin/va-threat --panel                 # Stage-3 demo: annotate + triage a virus panel
.venv/bin/va-benchmark                       # virus-only NetGO temporal benchmark
```

Entry points (`pyproject.toml [project.scripts]`) are thin `cli/` wrappers around
`training.train.run`, `cli.threat`, and `benchmark.run`.

## Architecture

**One config-driven trainer, not a script per experiment.** `training/train.py:run`
is the single training path. Pooling strategy, homology ensemble, ESM model, split
type, and pathogen domain are all flags/parameters — the documented experiments
reproduce from this one function. Don't add parallel trainers; add a knob.

Flow: `pipeline.load_proteins` (UniProt fetch or cached JSONL) → `pipeline.make_split`
(30%-identity MMseqs2 cluster split + family holdout) → per-namespace
`heads.fit_namespace` (select vocab, fit head, hierarchical-correct, Fmax-vs-Naive) →
test + zero-shot report → `_save` (pooled heads only).

**Per-namespace policy is the core design decision.** The three GO sub-ontologies
(molecular_function, biological_process, cellular_component) are trained, pooled,
and scored *independently*, each with its own evidence and pooling policy
(`config.NAMESPACE_POLICY`). This exists because IEA training poisons viral MF
(IEA-MF = domain-rule ligand binding ≈ disjoint from manual-MF = curated protein
binding; collapses MF Fmax to 0.09), so **MF trains manual-only with learned
attention pooling; BP/CC train on manual+IEA with mean pooling.** Val/test *always*
score against manual-only labels regardless of training policy. Every metric is
reported against a Naive baseline (predict each term's train frequency) — that's the
floor, not a formality.

**Pathogen domains** (`config.PathogenDomain` / `DOMAINS`): viral and bacterial
profiles bundle everything domain-specific — UniProt taxon/query, family-holdout
rank suffix (`viridae`/`aceae`), evidence/pooling policy, term-count floor, danger
ontology, models subdir. The embedding/clustering/training/threat machinery is
domain-agnostic and reused unchanged. Unspecified CLI flags resolve from the
selected domain profile, so the viral path stays the default. Models are trained
and served **per domain** (separate heads + vocab).

### Module map (`src/viral_annotation/`)

- `config.py` — all paths, model registry, constants, `NAMESPACE_POLICY`, `DOMAINS`. Start here.
- `ontology/go_dag.py` — OBO parse, ancestor lookup, true-path propagation/correction (pure stdlib).
- `data/` — `labels` (UniProt fetch + manual/IEA evidence tiers), `cluster` (MMseqs2), `split` (cluster split + family holdout), `dataset` (vocab + multi-hot matrices), `homology` (BLAST-KNN), `proteomes`/`danger_terms` (threat inputs).
- `embeddings/` — `esm` (ESM-2 pooled, fp16 autocast on CUDA, length-safe windowed batching for >1022aa), `cache`/`residue_cache` (the per-residue cache is large — ~20GB viral — and only built when a namespace uses attention pooling).
- `classifier/` — `model` (linear/MLP head), `pooling`, `ensemble` (late fusion), `serving` (load saved heads + annotate new sequences).
- `training/` — `pipeline` (shared load/split helpers for train + benchmark), `heads` (`fit_namespace`), `train` (the `run` orchestrator).
- `evaluation/` — `metrics` (protein-centric Fmax, hierarchical correction; pure stdlib), `report` (shared Fmax-vs-naive tables).
- `benchmark/` — virus-only NetGO temporal-split suite (Fmax/M-AUPR/Smin).
- `threat.py` + `cli/threat.py` — Stage-3 demo: predicted GO → danger categories.

## Conventions specific to this repo

- **Lazy heavy imports.** `torch`/`transformers`/`sklearn` are imported *inside*
  functions, not at module top, so `import viral_annotation` and the stdlib modules
  (ontology, metrics) stay usable without the `ml` extra. Preserve this.
- **Reproducibility is load-bearing.** Runs are seeded (`TRAIN_SEED`/`SPLIT_SEED = 1337`);
  embeddings are cached so a full re-run is ~90s. Keep new randomness seeded.
- **Cluster split, not random.** Whole sequence-identity clusters go to one bucket so
  no test protein has a ≥30%-identity homolog in train. `--random-split` exists for
  quick checks only and is *not* leakage-safe — never report results from it.
- Trained artifacts (`go_classifier.pt` + `.meta.json`) land under the domain's
  `models_dir`. Attention-pooled heads are **not servable** by the lightweight loader,
  so `_save` skips them — use `--pooling mean` for a deployable model.
- `data/`, `models/`, `results/`, and embedding caches are gitignored (large/generated).
  Note `/data/` is anchored so it ignores the repo-root data dir, NOT the `data/` package.
- `notebooks/kaggle_bacterial_train.ipynb` runs bacterial GPU training on Kaggle (T4);
  use `va-train --records <cached.jsonl>` to skip the slow rate-limited UniProt fetch there.
