# Annotated Bibliography

Combines the solicitation's reference list with the key method/model papers for
the annotation pipeline. Links verified June 2026.

## From the SBIR solicitation

1. **Hallee, L. & Gleghorn, J.P. (2023).** "Protein-Protein Interaction Prediction
   is Achievable with Large Language Models." *bioRxiv* 2023.06.07.544109.
   https://doi.org/10.1101/2023.06.07.544109 — pLMs reach SOTA on PPI given a large
   balanced dataset; novel synthetic augmentation of BioGRID + Negatome. *Anchor for
   the PPI stage; same authors as the topic's biophysical-characterization refs.*

2. **Gleghorn, J.P. et al.** University of Delaware, Biomedical Engineering —
   host-pathogen biophysical characterization publications. *Domain grounding.*

3. **Park, Y. & Marcotte, E.M. (2012).** "Flaws in evaluation schemes for pair-input
   computational predictions." *Nature Methods.* — The **C3 data-stratification
   standard** for PPI evaluation; the reason we use identity-based cluster splits.

4. **Evans, R. et al. (2022).** "Protein complex prediction with AlphaFold-Multimer."
   *bioRxiv.* — Structural baseline for benchmarking interaction predictions.

5. **PHI-base; VirHostNet; HPIDB; IntAct** — pathogen-host interaction databases.
   See `02-data-sources.md` for URLs/formats.

6. **Gordon, D.E. et al. (2020).** "A SARS-CoV-2 protein interaction map reveals
   targets for drug repurposing." *Nature.* — 332-interaction reference interactome;
   our candidate **zero-shot validation** target.

7. **Reactome, KEGG, Gene Ontology Consortium** — pathway-enrichment reference DBs.

## Annotation methods & models (added)

8. **Lin, Z. et al. (2023).** "Evolutionary-scale prediction of atomic-level protein
   structure with a language model" (**ESM-2 / ESMFold**). *Science.* — The
   embedding backbone. Model sizes/dims in `04-models-and-tools.md`.

9. **EvolutionaryScale (2024).** "ESM Cambrian (ESM-C)." Blog + release.
   https://www.evolutionaryscale.ai/blog/esm-cambrian — Representation-focused pLM
   family (300M/600M/6B); alternative backbone to benchmark.

10. **Yao, S. et al. (2023).** "NetGO 3.0: Protein Language Model Improves Large-
    Scale Functional Annotations." *Genomics, Proteomics & Bioinformatics* 21(2):349.
    https://academic.oup.com/gpb/article/21/2/349/7585485 — **LR-ESM**: logistic
    regression on ESM embeddings → GO terms. Direct precedent for our linear design.
    Local copy: [`NetGO-3.0-paper.pdf`](NetGO-3.0-paper.pdf).

11. **Kulmanov, M. & Hoehndorf, R. (2021).** "DeepGOPlus: improved protein function
    prediction from sequence." *Bioinformatics.* — AFP as large-scale multi-label
    classification; fast sequence-only baseline.

12. **Thumuluri, V. et al. (2022).** "DeepLoc 2.0: multi-label subcellular
    localization prediction using protein language models." *Nucleic Acids Research*
    50(W1):W228. https://academic.oup.com/nar/article/50/W1/W228/6576357 —
    Reference design for the localization sub-requirement.

13. **CAFA 5 (2023–24).** Critical Assessment of Functional Annotation, 5th ed.
    https://www.kaggle.com/competitions/cafa-5-protein-function-prediction and
    https://biofunctionprediction.org/cafa/ — Fmax/Smin scoring, IA-weighted term
    sets, time-based holdout. Our evaluation lingua franca.

14. **Radivojac, P. et al. (2013) & Zhou, N. et al. (2019).** Original & CAFA2/3
    "large-scale evaluation of computational protein function prediction" papers
    (*Nat. Methods* / *Genome Biology*) — define Fmax, Smin, and the evaluation
    methodology we adopt.

## Standards / data infrastructure

15. **The Gene Ontology Consortium.** GO resource + `go-basic.obo`.
    https://geneontology.org/ — DAG for true-path propagation and ancestor lookup.

16. **UniProt Consortium (2019+).** "UniProt: a worldwide hub of protein knowledge."
    *Nucleic Acids Research.* https://www.uniprot.org/ — sequences + GOA annotations
    (training labels). Note viral host-relative GO terms (e.g. "host cell cytoplasm").
