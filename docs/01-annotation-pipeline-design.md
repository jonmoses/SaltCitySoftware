# Annotation Pipeline — Design Reference

Distilled and structured from `OUTLINE.md`, cross-checked against the methods
used by NetGO 3.0, DeepGOPlus, and DeepLoc 2.0. This is the architecture we
build against. Citations in `05-references.md`.

## Goal

Input: a viral protein **amino-acid sequence** (FASTA).
Output: a **functional annotation** of that protein —
- ranked **GO terms** (Molecular Function, Biological Process, Cellular Component),
- **subcellular localization**,
- and (downstream, set-level) **pathway enrichment**.

This satisfies feasibility requirement **F4** and feeds the mechanistic-hypothesis
stage of the larger system.

## End-to-end recipe

```
sequence (FASTA)
   │  ESM-2 (or ESM-C) forward pass
   ▼
[L × d] per-residue embeddings
   │  mean-pool over length (CLS pooling is the alt)
   ▼
[d] protein vector
   │  linear layer  W[N×d] + b   (sigmoid, NOT softmax)
   ▼
N GO-term logits → per-term probabilities
   │  hierarchical (true-path) correction
   ▼
consistent GO probabilities
   │  per-term thresholds (optional; ranking often preferred)
   ▼
discrete GO annotations  ──►  subcellular localization (DeepLoc-style head/tool)
                          ──►  pathway enrichment (Reactome/KEGG over the term set)
```

## Step-by-step decisions

### 1. Per-protein embedding
- Backbone: **ESM-2** per `OUTLINE.md`. 650M model gives d=1280; 3B gives d=2560.
  See `04-models-and-tools.md` for the full size/dim table and the ESM-C option.
- **Mean pooling** across residues is the default protein-level summary; CLS-token
  pooling is the alternative. NetGO 3.0's LR-ESM uses pLM embeddings into a linear
  classifier — direct precedent for this exact move.
- **Which layer to pool** is a small hyperparameter — the most informative layer is
  not always the last. Treat as a sweep.

### 2. Classifier
- **Linear baseline first:** one layer `z = Wx + b`, shape `W[N×d]`, then
  per-logit **sigmoid**. This is multi-label, so each GO term is an independent
  binary question — sigmoid, never softmax. NetGO 3.0 found logistic regression
  on ESM embeddings (LR-ESM) competitive; start here.
- **MLP upgrade only if it earns Fmax:** insert 1–2 ReLU hidden layers + dropout.
  More capacity, more overfit/data-hunger risk. Establish the linear baseline,
  then test whether the MLP measurably improves Fmax.

### 3. Labels & loss
- Labels: binary vector length N per protein from UniProt/Swiss-Prot GO annotations.
- **True-path rule (critical):** propagate each annotation UP the GO DAG — set all
  ancestor terms to 1 before training. Skipping this trains on incoherent labels
  (positive child, negative parent).
- Loss: **binary cross-entropy**, per-label, averaged over N terms and all proteins.
- **Class imbalance is the main hazard** — label vectors are overwhelmingly zero.
  Mitigations: positive-weighting in BCE, focal loss, and/or restricting N to GO
  terms frequent enough to be learnable. CAFA 5 capped term sets near ~1500 (BPO)
  / ~800 (CCO) / ~800 (MFO) by frequency + Information Accretion — a sane template.

### 4. Hierarchical consistency at inference
- Raw sigmoids can violate the DAG (child > parent), which is impossible under the
  true-path rule.
- **Post-hoc correction** (prototype default): force a parent's probability ≥ max of
  its children's. Simple, interpretable, consistent.
- Architectural/loss-baked constraints are the sophisticated alternative — defer.

### 5. Thresholding → discrete annotations
- Sigmoid gives probabilities; a discrete annotation needs a yes/no cut.
- **Per-term thresholds** tuned on validation beat a single global 0.5 (terms have
  different base rates / calibration).
- You don't always need to threshold: for pathway enrichment and analyst ranking,
  the ranked probabilities are more useful than a hard cut. Threshold mainly for
  discrete reporting and certain metrics.

### 6. Subcellular localization (F4 sub-requirement)
- Same embedding backbone can drive a localization head; **DeepLoc 2.0** is the
  reference design (pLM → multi-label localization, with sorting-signal output).
- For viral proteins, localization vocabulary should include **host-cell
  compartments** (e.g. "host cell cytoplasm") — UniProt annotates viral proteins
  with host-relative terms. Note this when building the localization label set.
- Decision to make: train our own head vs. call DeepLoc as an external tool. See
  `04-models-and-tools.md`.

### 7. Pathway enrichment (F4 sub-requirement)
- Operates at the protein-**set** level (a predicted proteome), not per protein:
  map predicted GO/pathway terms onto **Reactome / KEGG** and test enrichment.
- This is the bridge from annotation → "ranked mechanistic hypotheses about
  infection pathways" (system requirement #4).

## Timing budget (must fit F3)

Core predictions **< 15 min**, full report **< 1 hr**, on standard hardware.
Implications for design:
- ESM embedding extraction is the compute bottleneck; pick model size with the
  timing budget in mind (650M is a reasonable accuracy/throughput balance — a
  typical viral proteome is only tens to a few hundred proteins).
- Cache embeddings; batch by length; consider the smaller ESM-2 / ESM-C variants
  if the largest model blows the budget for little Fmax gain.

## Open hyperparameters to sweep (none are hand-tunable a priori)

- ESM model size and **which layer** to pool from.
- Pooling: mean vs. CLS.
- Classifier: linear vs. MLP (depth, width, dropout).
- Loss: positive weight vs. focal-loss gamma.
- Term-set size N and inclusion frequency cutoff.
- Per-term decision thresholds.

## Honest caveats (carried from OUTLINE.md)

1. This is the standard, defensible methodology — a sound starting architecture,
   but the listed hyperparameters are tuned empirically on our own validation data.
2. To exactly reproduce NetGO 3.0 numbers, use their paper's methods/released code;
   what we build is a clean from-first-principles equivalent, not a line-by-line
   reimplementation.
