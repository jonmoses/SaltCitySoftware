# Walkthrough 2 — For a Biology Undergraduate

This walkthrough assumes you know biology — amino acids, proteins, the Gene Ontology,
homology, BLAST, viral life cycles — but **not** machine learning. Its job is to
explain the computational half: what a "protein language model" is, how we turn
sequences into a function classifier, and why each modeling decision was made (often
*for a biological reason*).

If you want a gentler, biology-free framing first, read Walkthrough 1. For the full
implementation with file references, read Walkthrough 3.

---

## 0. The goal, stated precisely

We are building **Stage 2** of a three-stage DARPA SBIR pipeline (the topic is
real-time pathogen–host interactome prediction). Stage 2 is *functional annotation*:
given a viral protein sequence, predict its **GO terms** across all three aspects
(MF, BP, CC). Stage 3 (a demo here) turns those annotations into a **threat profile**.

You already know that GO annotation is normally done by homology transfer (BLAST your
sequence against annotated proteins and copy their terms) or by domain models
(InterPro/Pfam → GO mappings). Those work well *when a well-annotated relative
exists*. The whole point of this project is the regime where it might not — a newly
emerged virus with no close, curated neighbor. That's why the core method is not
homology; it's a **protein language model**.

## 1. What a protein language model is (the one big new idea)

Think of how you learned protein intuition: after seeing enough sequences and
structures, you can glance at a sequence and guess "that looks like a kinase" or "that
hydrophobic stretch is a transmembrane helix." A **protein language model (pLM)** is
software that has done the same thing at enormous scale.

Concretely, **ESM-2** (the model we use) is a *transformer* neural network trained on
~250 million protein sequences by a single self-supervised task: hide a residue and
predict it from its context (masked-residue prediction — the protein analogue of fill-
in-the-blank). To get good at that, the network is forced to internalize the
"grammar" of proteins: which residues co-vary (i.e. are in contact in the folded
structure), which motifs are catalytic, which patterns are signal peptides. It learns
this from sequence alone, with **no labels**.

The output we care about: for any sequence you give it, ESM-2 returns a **vector of
~1280 numbers for every residue** (this is for the 650M-parameter variant we default
to; see `config.py`). These vectors are called **embeddings**. The key property is
that they are *contextual* and *evolutionarily informed* — two residues that play the
same functional role get similar embeddings even if the amino acids differ, because
the model learned that from homologous families during pretraining. In effect, ESM-2
gives us "evolutionary context for free," which is exactly what BLAST gets from an
alignment — but ESM-2 can produce it even when no good alignment exists.

We use ESM-2 **frozen** — we never update its weights. It is a fixed feature
extractor. All the learning in our project happens in a small classifier on top.

## 2. From a variable-length protein to a fixed feature vector

A classifier needs a fixed-size input, but proteins vary in length and ESM-2 gives us
one vector *per residue*. So we **pool** the [length × 1280] matrix down to a single
[1280] vector per protein. How we pool turns out to matter biologically, and it's one
of the more interesting results in the project.

- **Mean pooling** — average across residues. Robust, cheap, and the default. Good
  when the functional signal is spread across the whole protein.
- **Attention pooling** — instead of averaging equally, the model *learns a weighting*
  over residues and takes a weighted average (it can also learn several weightings at
  once, each free to focus on a different region). Biologically, this lets the model
  concentrate on the residues that actually carry the function — a catalytic triad, a
  binding pocket, a sorting signal — and ignore the structural filler.

We found these behave differently per GO aspect, so we use **per-namespace pooling**:

| Aspect | Pooling | Why |
|--------|---------|-----|
| Molecular Function | **attention** | MF is *local* — it lives in a few catalytic/binding residues. Attention locks onto them, and because those residues are conserved, the signal **transfers to unseen virus families**. On held-out coronaviruses, attention recovers MF at Fmax 0.46 vs 0.25 for mean pooling. |
| Biological Process | mean | BP is a whole-protein, distributed property; there's no localized site to attend to, and mean is far cheaper. |
| Cellular Component | mean | Same reasoning as BP. |

This is a genuine biological finding dressed up in ML clothing: **what a protein does
mechanically is written in a handful of conserved residues, and a learned attention
pooler can read it even on a virus family it never trained on.**

(Long proteins exceed ESM-2's input limit of ~1022 residues. Rather than truncate and
throw away a third of a polyprotein, we embed it in non-overlapping **windows** and
recombine — so the whole sequence is seen. Details in Walkthrough 3.)

## 3. The classifier itself

On top of the pooled embedding sits the actual model we train: a **multi-label
classifier**. For each of the ~700 GO terms in our vocabulary it outputs an
independent probability via a **sigmoid** (not a softmax — terms aren't mutually
exclusive; a protein has many at once). Architecturally it's as simple as logistic
regression (`classifier/model.py`); we keep it small on purpose, because the pLM
embedding already did the hard representational work and a bigger head mostly overfits.

Training is standard supervised learning: show it pooled embeddings with their known
GO labels, and adjust the head's weights to match. Two biology-driven wrinkles:

**Class imbalance.** Most GO terms are rare — any given term is "absent" on the vast
majority of proteins. Left alone, the model would learn to always say "absent." We
counter this by up-weighting the positive examples for each term in the loss function,
proportional to how rare the term is.

**The true-path rule, enforced twice.** You know that GO annotations obey the true-
path rule: if a protein has a child term, it implicitly has all ancestor terms. We use
this in both directions (`ontology/go_dag.py`):
- *Before training*, we **propagate** every protein's annotations up to the root, so
  the labels are complete.
- *After prediction*, we **correct** the output so a parent's probability is never
  lower than its most confident child — guaranteeing the predictions are a valid GO
  annotation set, not something that violates the ontology.

## 4. The data, and a subtle trap about evidence codes

Training labels come from **UniProt** — viral reviewed (Swiss-Prot) entries, fetched
with their GO cross-references (`data/labels.py`). Here's where your biology knowledge
matters: each GO annotation carries an **evidence code**. We split them into two
tiers:
- **Manual** — experimental or curator-reviewed (EXP, IDA, IPI, IMP, IGI, IEP, TAS,
  IC…). High trust.
- **IEA** — Inferred from Electronic Annotation (no human looked at it). Lower trust,
  but abundant.

**We always evaluate against manual labels only** — IEA labels would be circular
ground truth (they're guesses themselves). But can we *train* on IEA labels for extra
coverage? Here's the trap, and it's biological:

For **MF**, no. We discovered IEA-MF and manual-MF are *nearly disjoint vocabularies in
viruses*. Electronic MF annotations are dominated by InterPro/UniRule domain rules
that emit generic terms like "nucleotide binding" or "metal ion binding." Curated
viral MF, by contrast, is about specific **protein–protein binding** and adaptor
roles. Train on the electronic version and you learn the wrong distribution — MF
collapses to Fmax 0.09. Training MF on **manual labels only** doubles it.

For **BP and CC**, the electronic and manual vocabularies overlap fine, and the extra
IEA data helps. So we apply a **per-namespace evidence policy** (`config.py`,
`NAMESPACE_POLICY`): MF trains manual-only; BP/CC train on manual+IEA. This is the
single most important data decision in the project and it's driven entirely by viral
biology.

## 5. Evaluating honestly (this is where most annotation papers cheat)

**The metric: Fmax.** This is the CAFA-standard metric for protein function
prediction (`evaluation/metrics.py`). It's a protein-centric F1: at each probability
threshold, compute precision and recall of predicted vs. true terms (per protein, then
averaged), and report the best F1 over all thresholds. We also compute **M-AUPR**
(term-centric average precision) and **Smin** (an information-content-weighted error)
for the formal benchmark, mirroring NetGO 3.0.

**The baseline: Naive.** The honest floor is predicting every term at its training
frequency. A lot of "good" annotation numbers are really just the Naive baseline in
disguise, because a handful of generic terms sit on almost every protein after
propagation. We report **lift over Naive** everywhere, so you can see whether the
model learned anything real.

**The split: identity-aware, leakage-safe.** This is the biology-specific rigor.
Proteins have homologs; if a test protein's homolog is in the training set, a high
score just means the model memorized a family it had already seen. So we cluster all
sequences at **30% identity with MMseqs2** (`data/cluster.py`) and assign **whole
clusters** to train/val/test (`data/split.py`), guaranteeing no test protein has a
≥30%-identity relative in training. (This is the standard Park & Marcotte defense
against inflated function-prediction numbers.)

**The hard test: zero-shot on a held-out family.** We remove **all of
Coronaviridae** from training and score the model on recovering its known functions
cold — a stand-in for "a new pathogen emerges." The result is the most interesting
finding in the repo:

- **Molecular Function transfers.** Held-out coronavirus MF recovers at Fmax 0.46,
  well above its Naive floor — because attention pooling reads conserved catalytic/
  binding residues that are shared across families.
- **Biological Process and Cellular Component do not** beat their Naive priors zero-
  shot — *where* a protein localizes and *which* host process it joins are
  family-specific and not recoverable from sequence alone.

The honest reading: **on a never-before-seen virus family, sequence alone tells you
what its proteins *do*, but not where they act or what process they join.** For threat
characterization, "what it does" is the actionable part.

## 6. Adding homology back in (the ensemble)

A pLM and BLAST have complementary failure modes, so we combine them. The homology
component (`data/homology.py`) is a **BLAST-KNN** label-transfer (the GOLabeler/NetGO
formulation): MMseqs2-search the query against annotated training proteins, then score
each GO term by the **bitscore-weighted fraction** of hits that carry it. It transfers
the neighbors' *manual* labels, so it stays consistent with our manual-only
evaluation.

We then **late-fuse** the pLM and homology score matrices with weights grid-searched
per namespace on the validation set (`classifier/ensemble.py`). The effect: homology
**flips zero-shot from below the Naive floor to clearly above it** (driven by BP,
0.356 → 0.398). Even with no close relative, a permissive cross-family search finds
distant homologs that carry conserved process information the pLM misses. The two
signals together recover an unseen family's function above base rates.

## 7. Stage 3 — turning annotations into a threat profile

Finally, the demo (`threat.py`, `data/danger_terms.py`, run via `va-threat --panel`).
A curated, auditable map pins human-meaningful **danger categories** — host-cell
entry & membrane fusion, immune evasion, host-cell killing, apoptosis manipulation,
host-machinery hijack, toxin activity — to a few high-level GO roots each. At load
time every root is expanded over the live DAG to *all* its descendant terms, so a
prediction on any specific mechanism (e.g. "fusion of virus membrane with host
endosome membrane") counts toward its category.

The base-rate problem you'd anticipate as a biologist is real: after propagation,
generic terms like "symbiont entry into host cell" sit on nearly every viral protein,
so absolute confidence flags everything. So the report shows two views:
- a **category fingerprint** (peak confidence per danger category — *which* dangerous
  mechanisms are present, comparable across viruses), and
- **standout proteins**, ranked by **lift over the proteome's own background** — which
  surfaces the *specific* proteins that drive a mechanism above the viral crowd: the
  entry glycoprotein, the interferon antagonist (think Ebola VP35/VP24), not the
  universal baseline.

The output is explicitly a **triage hypothesis, not a determination** — "this protein
likely participates in this harmful mechanism; look here first."

---

### The biology-to-computation cheat sheet

| Biological idea | Computational realization |
|---|---|
| Evolutionary context from homologs | ESM-2 embeddings (learned from 250M sequences, no alignment needed) |
| Function lives in conserved active-site residues | Attention pooling for MF |
| True-path rule | Label propagation up the DAG + post-hoc score correction |
| Evidence codes (manual vs IEA) | Per-namespace evidence policy (MF manual-only) |
| Homologs inflate test scores | 30%-identity cluster split, whole clusters to one bucket |
| A novel pathogen has no curated relative | Held-out-family zero-shot evaluation |
| BLAST label transfer | BLAST-KNN homology component, late-fused with the pLM |
| Pathogenicity mechanisms | Curated GO danger-category map + lift-over-background ranking |
