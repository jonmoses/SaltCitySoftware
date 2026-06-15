# SBIR Topic Summary — DPA26BZ03-DV014

**Title:** Real-Time Pathogen-Host Interactome Prediction
**Agency:** DARPA (Chem Bio Defense; Modernization Priority: Biotechnology)
**Type:** Direct-to-Phase-II (DP2). Phase I feasibility must be demonstrated via prior work.
**CMMC:** Level 2 (Self).

## Objective

Rapidly characterize host–pathogen interactions **from pathogen protein sequence
alone**, enabling timely medical-countermeasure prioritization and force health
protection against novel or emerging biological threats.

## The capability gap

Characterizing how a novel pathogen interacts with human hosts currently takes
weeks to months of experimental work and yields incomplete understanding. The
solicitation bets that recent protein language models (pLMs) + large-scale PPI
prediction make computational threat characterization feasible.

## System requirements (from DESCRIPTION)

The system must:
1. Predict host-pathogen PPIs with high accuracy across **viral, bacterial, and parasitic** classes.
2. Demonstrate **zero-shot** prediction on previously unseen pathogens.
3. Provide **comprehensive functional annotation of both pathogen and host proteins**.
4. Generate **ranked mechanistic hypotheses** about infection pathways via automated analysis.
5. Complete **core predictions within 15 minutes** and **full characterization reports within 1 hour** on standard computing hardware.

Cross-cutting mandates: rigorous evaluation that proves generalization (not
memorization), benchmarking against established interaction databases, and
experimental validation via standard binding-assay techniques.

## Phase I feasibility requirements (what DP2 proposers must already show)

| # | Requirement | How our annotation build relates |
|---|-------------|----------------------------------|
| F1 | **Benchmark Performance Data** — quantified PPI results on ≥1 pathogen class with rigorous data-separation methods | PPI stage (not this build); annotation reuses the same data-separation discipline |
| F2 | **Zero-Shot Validation** — recover known host-pathogen interactions without training on that specific pathogen | Annotation analog: whole-organism / whole-family holdout (see `03-evaluation-protocol.md`) |
| F3 | **Pipeline Demonstration** — one complete end-to-end run, sequence → mechanistic report, meeting timing | Annotation is a stage of that pipeline; must fit the 15-min / 1-hr budget |
| F4 | **Functional Annotation Capability** — operational tools for GO terms, subcellular localization, AND pathway enrichment | **← This is exactly what we build first** |

## Phase II structure (context for where this goes)

- **DP2 Base (9 mo):** scale pipeline across expanded pathogen coverage incl.
  higher-consequence agents; experimental validation of ≥25 novel predicted
  interactions (≥30% hit rate); drug-repurposing demonstration.
- **DP2 Option (9 mo):** transition-ready software, full docs, live demo to
  DARPA with prospective run on a Government-selected pathogen.

Key payable milestones reference an **evaluation dataset covering higher-
consequence pathogens** (Month 4) and **experimental validation of ≥25 novel
interactions** (Month 9) — the annotation layer feeds the hypotheses that those
validations test.

## Why annotation first

- It is the most self-contained, independently demonstrable feasibility item (F4).
- Its outputs (GO function, localization, pathways) are inputs to the mechanistic
  hypothesis generation in requirement #4.
- It shares the rigorous-split and zero-shot methodology the whole program needs,
  so getting it right establishes the evaluation discipline early.

## Dual-use (Phase III)

Military: biosurveillance, countermeasure prioritization, force health
protection, intel analysis. Commercial: drug discovery/repurposing, vaccine
target ID, diagnostic biomarkers, veterinary/agricultural biosecurity.
