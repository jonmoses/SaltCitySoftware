# Reference Library — Viral Protein Annotation Pipeline

This `docs/` directory is the gathered reference material for building the
**functional-annotation component** of the SBIR topic *DPA26BZ03-DV014:
Real-Time Pathogen-Host Interactome Prediction*.

Source material lives at the repo root:
- `OUTLINE.md` — the working design for the annotation pipeline (ESM-2 → GO classifier).
- `topic_DPA26BZ03-DV014_*.PDF` — the original DARPA SBIR solicitation.

## What we are building (scope)

The full SBIR system has three stages (see `00-sbir-topic-summary.md`):

1. **Characterize taxonomy** — what pathogen family the genome belongs to.
2. **Annotate proteins of interest** — GO terms, subcellular localization, pathways. ← **WE BUILD THIS FIRST**
3. **Predict host-pathogen interactions** and characterize the threat.

This reference set is centered on **Stage 2 (annotation)** because it is the
component `OUTLINE.md` specifies and it directly satisfies the solicitation's
"Functional Annotation Capability" feasibility requirement. The PPI stage (3)
is the other major pillar and is summarized but not the current build target.

## Index

| File | Contents |
|------|----------|
| `00-sbir-topic-summary.md`        | Distilled solicitation + requirements traceability matrix |
| `01-annotation-pipeline-design.md`| End-to-end annotation architecture, decisions, open hyperparameters |
| `02-data-sources.md`              | Datasets & databases: URLs, formats, access notes |
| `03-evaluation-protocol.md`       | Fmax/AUPR/Smin, identity-based splits, zero-shot holdout |
| `04-models-and-tools.md`          | ESM-2 / ESM-C variants, DeepLoc, downstream tools, hardware/timing |
| `05-references.md`                | Annotated bibliography with URLs |
| `06-benchmark.md`                 | Virus-only NetGO-3.0-style temporal benchmark (methodology + result) |
| `07-threat-characterization.md`   | Stage-3 demo: annotate a virus → danger-category threat profile |
| `08-bacterial-extension.md`       | Extending the pipeline to bacteria: per-domain profiles, bacterial danger ontology, re-validation |
| `glossary.md`                     | Domain term definitions |

## Status

Reference-gathering complete. No pipeline code written yet — these files are
the foundation to build against.
