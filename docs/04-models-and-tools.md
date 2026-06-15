# Models & Tools Reference

Backbone models, off-the-shelf annotation tools, and the libraries we'll lean on.

## ESM-2 (primary embedding backbone, per OUTLINE.md)

Transformer pLMs from Meta/FAIR. Embedding dimension `d` = the per-residue vector
size you pool. Weights on HuggingFace (`facebook/esm2_t*`) and the `fair-esm` /
`esm` packages.

| Model | Params | Layers | Embedding dim `d` |
|-------|--------|--------|-------------------|
| ESM-2 8M   | 8M   | 6  | 320  |
| ESM-2 35M  | 35M  | 12 | 480  |
| ESM-2 150M | 150M | 30 | 640  |
| ESM-2 650M | 650M | 33 | **1280** |
| ESM-2 3B   | 3B   | 36 | **2560** |
| ESM-2 15B  | 15B  | 48 | 5120 |

**Recommendation for the prototype:** start at **650M (d=1280)** — the
accuracy/throughput sweet spot and what fits the 15-min timing budget for a
viral proteome (tens–hundreds of proteins). Scale to 3B only if Fmax demands it.
Remember "which layer to pool from" is a hyperparameter — not necessarily the last.

## ESM-C / ESM Cambrian (alternative backbone to evaluate)

EvolutionaryScale's representation-focused family (released Dec 2024), parallel to
ESM3. Variants: **300M, 600M, 6B**. ESM-C 600M reportedly performs near much larger
models, and ESM-C has beaten ESM-2 15B on contact prediction at far smaller size.
Worth benchmarking as a drop-in for the embedding step if license/access permits.
Blog: https://www.evolutionaryscale.ai/blog/esm-cambrian

> Decision: ESM-2 650M is the default per OUTLINE.md. Add ESM-C 600M as a
> comparison arm once the linear classifier baseline exists — same harness, swap
> the embedding source.

## Functional-annotation prior art (methods to mirror / benchmark against)

| Tool | Approach | Relevance |
|------|----------|-----------|
| **NetGO 3.0** | LR-ESM: logistic regression on ESM-1b embeddings → GO terms (+ ensemble) | Direct precedent for our linear-on-pLM design |
| **DeepGOPlus** | CNN on sequence + diamond, multi-label GO; ~40 proteins/sec | Fast baseline; framing AFP as large-scale multi-label classification |
| **CAFA 5 solutions** | pLM embeddings + MLP heads, ensembles, IA-weighted term sets | Practical templates for term-set sizing and ensembling |

## Subcellular localization (F4 sub-requirement)

| Tool | Approach | URL |
|------|----------|-----|
| **DeepLoc 2.0** | pLM → multi-label subcellular localization + sorting-signal prediction | https://services.healthtech.dtu.dk/services/DeepLoc-2.0/ |
| DeepLoc 2.1 | Adds membrane-protein-type prediction | (same service host) |

Decision to make: **call DeepLoc as an external tool** vs. **train our own
localization head** on the shared ESM embeddings. External tool is faster to a
demo; a shared-embedding head is cleaner for the unified <15-min pipeline and lets
us use viral host-relative compartment labels. Default: prototype with our own head
on the existing embeddings, validate against DeepLoc.

## Pathway enrichment (F4 sub-requirement)

- **Reactome** analysis service / `reactome2py`; **KEGG** via `KEGG REST` / `bioservices`.
- Python enrichment: `gseapy`, `goatools` (GO enrichment).
- Operates over the predicted term *set* for a proteome → feeds mechanistic hypotheses.

## Core Python libraries

| Need | Library |
|------|---------|
| ESM inference | `transformers` (HF) or `fair-esm` / `esm`; `torch` |
| GO DAG parsing / propagation / enrichment | `goatools`, `obonet`, `pronto` |
| Sequence clustering for splits | MMseqs2 (CLI), CD-HIT |
| Sequence I/O | `biopython` |
| Classifier / metrics | `scikit-learn`, `torch`, `numpy`, `scipy` |
| Experiment tracking | `wandb` or `mlflow` (optional) |

## Hardware & timing notes (F3: <15 min core, <1 hr full report)

- ESM forward passes are the bottleneck — GPU strongly preferred; **cache
  embeddings** so re-runs and threshold tuning are free.
- Batch sequences by length to minimize padding waste.
- A viral proteome is small (tens–few hundred proteins), so 650M embeddings for a
  whole proteome are well within 15 min on a single modern GPU. The 15-min budget
  is the real constraint when scaling to larger proteomes / higher-consequence
  agents in Phase II — pick model size accordingly.
- "Standard computing hardware" in the solicitation — avoid a design that *requires*
  the 15B model; keep a smaller-model fallback path.
