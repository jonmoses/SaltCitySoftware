# Threat characterization — "does this virus have dangerous effects?"

The SBIR's Stage-3 payoff: take a virus proteome, annotate it with the GO
classifier, and surface the **dangerous mechanisms** it encodes. Run:

```bash
va-threat --panel              # hemorrhagic-fever panel
va-threat --taxon 2697049 --name sars2
va-threat --fasta proteome.faa --name sample
```

## How it works

1. **Fetch** the target proteome by taxonomy id, **TrEMBL included**
   (`data/proteomes.py` → `labels.fetch_raw`), to simulate an uncurated sample.
   Exact-duplicate strain sequences are collapsed.
2. **Annotate** each protein with the persisted GO classifier
   (`classifier/serving.py` rebuilds the saved per-namespace heads, embeds with the
   same ESM-2 config, predicts, and applies true-path correction). No retraining.
3. **Map to danger categories.** A curated set of GO *roots* (`data/danger_terms.py`)
   is expanded to all descendants over the DAG (`GoDag.descendants`); each protein's
   predicted terms are intersected with those sets (`threat.py`). Categories:
   toxin activity, host-cell entry & membrane fusion, host-cell killing/lysis,
   immune evasion / host-defense perturbation, apoptosis manipulation, and host
   gene-expression/machinery hijack.

The curation uses **live, non-obsolete** GO ids (GO renamed the classic
"by virus of host" terms to "symbiont-mediated"; several old ids are obsolete).
`build_danger_map` re-asserts every root exists at load, so a future ontology
update that obsoletes a root fails loudly instead of silently going dark.

## Reading the output: fingerprint + lift, not a binary flag

High-level viral terms ("symbiont entry into host cell", "perturbation of host
innate immune response") sit on **most** viral proteins after true-path propagation,
so their absolute probability is high for nearly everything — the same base-rate
effect that flatters Naive in the temporal benchmark (`docs/06`). A raw threshold
therefore flags every protein and is useless. The report gives two honest views:

- **Category fingerprint** — peak confidence per category: *which* dangerous
  mechanisms are present, comparable across viruses.
- **Standout proteins** — ranked by **lift over the proteome background** (a term's
  mean predicted probability across this proteome). This surfaces the specific
  proteins that drive a mechanism *above the viral crowd* — the entry glycoprotein,
  the interferon antagonist — rather than the universal baseline.

## Result — hemorrhagic-fever panel (Ebola / Nipah / Lassa / Marburg, TrEMBL incl.)

Peak confidence per danger category (the fingerprint):

| category | ebola | nipah | lassa | marburg |
|----------|------:|------:|------:|--------:|
| Toxin activity | 0.00 | 0.00 | 0.00 | 0.00 |
| Host-cell entry & membrane fusion | 0.95 | 0.92 | 0.92 | 0.91 |
| Host-cell killing / lysis | 0.38 | 0.63 | 0.38 | 0.44 |
| Immune evasion / host-defense perturbation | 0.76 | 0.80 | **0.95** | 0.80 |
| Apoptosis manipulation | 0.68 | 0.82 | 0.62 | 0.68 |
| Host gene-expression / machinery hijack | 0.59 | 0.45 | 0.70 | 0.56 |

**The standouts land on the right proteins** (verified against UniProt names):

- **Nipah** — entry/fusion standout **Q4VCP6 = Fusion glycoprotein F0**; immune-evasion
  and apoptosis standouts **Q997F1 / Q4VCP8 = C protein** and **Q997F2 = V protein**
  — the P/V/W/C-gene products that are Nipah's known interferon antagonists
  (MDA5/MAVS/STAT inhibition). The model put the danger flags on exactly the
  proteins responsible for them.
- **Lassa** — immune evasion peaks at **0.95** on the nucleoprotein, whose
  exonuclease domain is a documented type-I-interferon antagonist.
- **Ebola** — distinct proteins top entry (the GP glycoprotein) versus immune
  evasion versus host-shutoff, matching the division of labor in the filovirus
  proteome.
- **Toxin activity is 0.00 everywhere** — correct: viruses don't carry the classic
  toxin MF terms, and none are in the model's training vocab.

## Zero-shot validation — SARS-CoV-2 (the genuine novel-family test)

The persisted model was trained with `use_cluster=True` and **Coronaviridae held
out** (`HOLDOUT_FAMILY`), so its heads never saw a coronavirus. Running
`--taxon 2697049` (291 unique proteins, TrEMBL incl.) is therefore a true
novel-family zero-shot test — and the lift ranking still lands on the right
virulence factors (accessions verified against UniProt):

| standout | protein | flagged for | biology |
|----------|---------|-------------|---------|
| A0A873P8T4 | Surface glycoprotein (**Spike**, S) | entry/fusion 0.90 | the entry/fusion protein |
| A0A6M3HM27 | **ORF1ab** polyprotein | host ubiquitin hijack 0.82 | encodes nsp3/PLpro, a deubiquitinase/deISGylase |
| A0A7U3N901 | **ORF6** | killing/apoptosis | a top SARS-CoV-2 interferon antagonist (STAT1 / mRNA-export block) |
| P0DTF1 / P0DTG1 / P0DTD3 | **ORF3b / ORF3c / ORF9c** | killing/apoptosis | accessory ORFs, known apoptosis inducers |
| A0A8A8QES1 | **Nucleoprotein** (N) | nuclear penetration 0.85 | N traffics through the nucleus |

Immune evasion peaks at 0.87 (innate-immune + pattern-recognition suppression);
toxin is 0.00. The danger signal lives in biological_process — the project's
weaker zero-shot namespace by Fmax — yet the per-protein **lift ranking** still
recovered SARS-CoV-2's actual danger biology from sequence alone, never having seen
the family. This is the strongest demonstration of the Stage-3 use case.

## Honest scope & caveats

- **The panel is in-distribution; SARS-CoV-2 is the zero-shot case.** Only
  Coronaviridae is held out of training (`HOLDOUT_FAMILY`), so the hemorrhagic-fever
  panel (Filo/Paramyxo/Arenaviridae) demonstrates "annotate + triage an unknown
  sample", not "unseen family". The SARS-CoV-2 run above is the genuine novel-family
  zero-shot test.
- **The danger ontology is almost entirely biological_process** (host-interaction
  mechanisms). Per the project's pooling finding, molecular function is the most
  reliable zero-shot signal — but the danger signal lives in BP, which is weaker
  out-of-family. In-distribution (this panel) BP is solid.
- **Persisted model is mean-pooled linear** (overall Fmax 0.376), not the attention
  MF production head. Retraining with `va-train --pooling per-namespace` would sharpen calls.
- **Triage, not determination.** A danger hit is a confidence-ranked hypothesis that
  a protein participates in a harmful mechanism — a screening signal, not a wet-lab
  result. The danger-term list is a hand-curated, auditable ontology subset.
