# Bacterial extension — annotating & triaging bacterial pathogens

The SBIR topic spans **viral, bacterial, and parasitic** pathogen classes
(`00-sbir-topic-summary.md`, requirement #1). The first build was viral-only; this
adds **bacteria** as a second, separately-trained domain. Parasites are still future
work.

## Design: one pipeline, per-domain profiles

A `PathogenDomain` profile (`config.py`, `DOMAINS` registry) bundles everything that
is domain-specific; the rest of the pipeline is domain-agnostic and reused unchanged.
A `--domain {viral,bacterial}` flag on `va-train` / `va-threat` selects the profile;
unspecified knobs fall back to the profile, so the **viral path is byte-for-byte
unchanged** (its model still lives at `models/go_classifier.pt`).

| Profile field | viral | bacterial |
|---|---|---|
| `taxon_id` / `uniprot_query` | 10239 | **2** |
| `family_suffixes` (holdout rank) | `viridae` | **`aceae`** |
| `holdout_family` (zero-shot) | Coronaviridae | **Francisellaceae** |
| `namespace_policy` | MF manual-only, attn MF | **all asymmetric; attn MF, mean BP/CC** (see below) |
| `min_term_count` | 10 | **15** (larger corpus) |
| `default_pooling` | mean | **mean** (the servable model; LoRA path uses the policy) |
| `models_subdir` | `""` (root) | **`bacterial/`** |

**Reused unchanged:** ESM embedding + caches (`embeddings/`), classifier heads +
attention pooler (`classifier/`), the `Head`/trainer loop (`training/heads.py`),
MMseqs2 cluster split (`data/cluster.py`, `data/split.py`), vocab + label matrices
(`data/dataset.py`), GO DAG + true-path correction (`ontology/go_dag.py`), metrics +
report (`evaluation/`), the homology ensemble (`data/homology.py`,
`classifier/ensemble.py`), serving (`classifier/serving.py`), and the threat *engine*
(`threat.py`). Embedding caches are keyed by accession (globally unique in UniProt), so
bacterial proteins simply add rows — no cache split needed.

## Bacterial danger ontology (`data/danger_terms.py`)

Danger categories are domain-keyed (`danger_categories("bacterial")`). The threat
engine is otherwise identical — it is handed a category list. Every root was verified
present and non-obsolete in go-basic (2026-06); `threat.build_danger_map` re-asserts at
load (tested in `tests/test_bacterial_domain.py`).

| Category | Roots (GO) | Note |
|---|---|---|
| Toxin activity | GO:0090729 | The viral toxin category is always 0.00; for bacteria this **fires** (anthrax/diphtheria/cholera/Shiga). |
| Secretion-system effector delivery | GO:0030254/0030255/0033103/0044315 | Type III/IV/VI/VII secretion — molecular syringes. |
| Host adhesion & cell invasion | GO:0044406/0044650/0044409/0085017 | Adhesins + vacuole-mediated invasion. |
| Immune evasion / host-defense subversion | GO:0042783/0141043/0030682/0099018 | Shared symbiont-mediated roots + RM-system evasion. |
| Host-cell killing / lysis | GO:0001907/0001897/0019835/0051715 | Hemolysins, pore-forming cytolysins. |
| Antimicrobial resistance | GO:0046677/0008800 | Treatability axis — beta-lactamase + antibiotic response. |
| Iron / nutrient piracy | GO:0019290/0015891 | Siderophore biosynthesis + transport. |
| Biofilm / persistence | GO:0042710 | Chronic, antibiotic-tolerant communities. |

## Target panel & zero-shot (`data/proteomes.py`)

`TARGET_BACTERIA` is a select-agent panel (taxon ids verified against UniProt 2026-06):
`anthrax` (1392), `plague` (632), `tularemia` (263), `melioidosis` (28450).
**`tularemia` (Francisella tularensis) is the genuine zero-shot target** — its family
`Francisellaceae` is the model's held-out family, mirroring SARS-CoV-2 / Coronaviridae
on the viral side. The other three are in-distribution "annotate + triage" cases.

## Re-validation — virus findings are NOT inherited

Two empirical viral findings were virus-specific and must be re-derived for bacteria:

1. **Evidence policy.** The viral MF-manual-only fix came from viral IEA-MF (domain
   rules → generic ligand binding) being nearly disjoint from curated manual-MF.
   Bacterial IEA is rich and reliable (orthology + curated rules), so the bacterial
   policy **starts asymmetric (manual+IEA) for all three namespaces** and is only
   specialized if the IEA-vs-manual MF diagnostic shows the same collapse. The
   bacterial `NAMESPACE_POLICY` is an *output* of that run, not an assumption.
2. **Pooling.** The plain `va-train --domain bacterial` model defaults to **mean** —
   the servable config, and a *frozen* attention per-residue cache (~hundreds of GB at
   bacterial scale) is impractical. Attention-for-MF is instead reached through the
   end-to-end **LoRA fine-tune** path (below), where residues are computed live and
   never cached.

## Architecture: LoRA fine-tune (beyond the frozen linear head)

The frozen pLM + linear head tops out around **overall Fmax 0.49** (MF 0.48 / BP 0.45 /
CC 0.57) — only ~0.10–0.15 over Naive. The backbone never adapts to bacterial sequence
statistics, and mean pooling dilutes the localized motifs that determine function. The
`--finetune lora` mode (`training/finetune.py`) unfreezes ESM-2 650M with **low-rank
adapters** and trains it jointly with the per-namespace pooling + heads:

- **One shared backbone, three per-namespace heads (multi-task).** A single backbone
  forward yields hidden states `[B,L,d]` that feed an **attention pooler for MF** and a
  **mean pool for BP/CC** — far cheaper than three separate fine-tunes, and it fits a
  Kaggle T4 (adapters-only params + gradient checkpointing + fp16).
- **Per-namespace evidence policy preserved by masking,** not by splitting the backbone:
  each protein contributes loss to a namespace only if it is in that namespace's
  `train_pool` (so MF can stay manual-only while BP/CC train manual+IEA). Val/test still
  score manual-only, exactly as the frozen path.
- **No per-residue disk cache** — residues are computed live and back-propagated.
- **Asymmetric multi-label loss** (`--loss asl`) replaces the crude clamped `pos_weight`,
  and the heads gain LayerNorm/GELU/dropout (`build_classifier` flags).
- **Serving is heavier:** a LoRA model loads ESM-2 + the adapter via
  `serving.FinetunedAnnotator` (not the lightweight `GoAnnotator`); artifacts land in
  `models/bacterial/finetuned/` (adapter + `heads.pt` + meta).

The frozen path is unchanged and remains the default; this is a `--finetune lora` knob on
the one trainer, validated against the same Fmax-vs-Naive tables.

## Scale (all reviewed bacterial Swiss-Prot, ~20× viral)

- Embedding is a long one-time job; the pooled cache checkpoints per chunk. A *frozen*
  per-residue (attention) cache is impractical here → mean pooling for the servable
  model; the LoRA path avoids the cache entirely by computing residues live.
- MMseqs2 clustering scales fine. Model-organism over-representation (E. coli, B.
  subtilis) is partly handled by the cluster split removing near-duplicates; per-cluster
  subsampling is a future option.
- `build_labels` makes a dense `[P × N]` matrix; with a larger bacterial vocab this is a
  memory watch-item (switch to sparse if needed).
- The 15-min inference budget (F3) is tighter for bacterial proteomes (hundreds–
  thousands of proteins vs tens). `default_esm_model` is per-domain; drop to a smaller
  ESM for bacterial serving if a real proteome blows the budget.

## Running it

```bash
# Frozen baseline (writes models/bacterial/go_classifier.pt[.meta.json]).
va-train --domain bacterial --limit 2000      # dry run first (subset)
va-train --domain bacterial                    # full reviewed bacterial set

# LoRA fine-tune — the higher-accuracy architecture (writes models/bacterial/finetuned/).
# --train-pool-cap bounds the manual+IEA pool to fit a single Kaggle T4 session.
va-train --domain bacterial --finetune lora --loss asl --pooling per-namespace \
         --train-pool-cap 100000

# Threat-triage a bacterial proteome.
va-threat --domain bacterial --panel           # anthrax / plague / tularemia / melioidosis
va-threat --domain bacterial --taxon 632 --name plague
```

## Status

**Infrastructure complete and tested** (`tests/test_bacterial_domain.py`,
`tests/test_training_refactor.py`, `tests/test_finetune.py`; the viral path is unchanged
and still passes). Both the frozen baseline (~Fmax 0.49) and the new **LoRA fine-tune**
path are wired into the one trainer. The fine-tuned model still needs a full Kaggle-T4
run (`notebooks/kaggle_bacterial_train.ipynb`), after which this doc gets its
per-namespace Fmax-vs-Naive and panel tables (as `docs/06`/`docs/07` have for viruses)
and the frozen-vs-fine-tuned comparison + evidence-policy diagnostic.

## Out of scope (this iteration)

NetGO temporal benchmark for bacteria (`benchmark/` + `quickgo.py` taxon); a joint
cross-kingdom model; the parasitic class; renaming the `va-*` scripts / `viral-`
package away from their viral-era names.
