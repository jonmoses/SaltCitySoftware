# Evaluation Protocol Reference

How we measure annotation performance and — just as important — how we **split**
data so the numbers are real. This directly underwrites SBIR feasibility items F2
(zero-shot) and the program-wide mandate to prove generalization, not memorization.

## Metrics (CAFA-aligned)

| Metric | What it measures | Why it's here |
|--------|------------------|---------------|
| **Fmax** | Max F1 over the full threshold sweep, computed protein-centric, per sub-ontology | The field standard from CAFA; report it to speak the field's language |
| **AUPR** | Area under precision-recall | Robust under extreme class imbalance (label vectors are mostly zeros) |
| **Smin** | Semantic-distance minimum; weights terms by Information Accretion | Credits getting *informative* terms right, not just frequent ones |
| **per-term AUROC** | Discrimination per GO term | Diagnostic; find which terms are learnable |

Report Fmax and AUPR at minimum; Smin and per-term AUROC as complementary. Compute
metrics **separately for MFO / BPO / CCO** — they behave very differently (MFO
typically highest). CAFA 5 reference band: mean Fmax ≈ 0.65, MFO up to ~0.8.

### Fmax definition (for implementation)
Sweep threshold t ∈ [0,1]. At each t, for each protein compute precision and recall
over predicted vs. true (propagated) term sets, average across proteins, take the
harmonic mean (F1). **Fmax = max over t** of that averaged F1. Per CAFA, average
precision only over proteins with ≥1 prediction at t; average recall over all proteins.

## Splitting — the part that makes numbers honest

### Hazard
Naive random splits leak: a near-identical homolog of a test protein sitting in
train lets the model memorize, inflating reported numbers. (This is the C3 data-
stratification concern from Park & Marcotte, 2012 — a topic reference.)

### Standard fix — sequence-identity-based splitting
1. Cluster ALL proteins at a sequence-identity threshold (commonly **30%**) using
   MMseqs2 / CD-HIT.
2. Assign **whole clusters** to train OR test — never split a cluster.
3. Result: no test protein has a close relative in training. This is the
   separation standard to cite.

### Time-based split (CAFA style, complementary)
Train on annotations existing before a cutoff date; evaluate on annotations added
after. Mirrors the real "predict the unknown" use case. Useful to report alongside
identity splits.

## Zero-shot validation (feasibility item F2)

Go beyond cluster splitting: **hold out an entire organism or viral family.**

- Train on everything *except* one held-out virus (or family).
- Show the model recovers that virus's known functional annotations / known host
  interactions despite never having seen it.
- This experimental design *is* the evidence the topic asks for. ESM's self-
  supervised pretraining is what makes it plausible — embeddings are meaningful
  even for sequences with no annotated relatives in training.

Candidate held-out target: **SARS-CoV-2** (Gordon et al. 2020 gives a clean known
interactome to recover against) — strong, recognizable zero-shot demonstration.

## Evaluation checklist

- [ ] Propagate BOTH predictions and ground truth up the DAG before scoring.
- [ ] Compute Fmax / AUPR / Smin per sub-ontology (MFO, BPO, CCO) separately.
- [ ] Report under **30% identity cluster split** (primary) + time-based split (secondary).
- [ ] Run at least one **whole-organism / whole-family holdout** for the zero-shot claim.
- [ ] State term-set size N and inclusion cutoff alongside every score.
- [ ] Keep the held-out organism's data out of *embedding-layer* selection and
      threshold tuning too — no leakage through hyperparameter choice.
