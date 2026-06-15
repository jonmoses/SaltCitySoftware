# Glossary

**AFP** — Automated Function Prediction. The task of assigning functional labels
(GO terms) to proteins computationally.

**AUPR** — Area Under the Precision-Recall curve. Preferred over AUROC under
extreme class imbalance (most GO labels are negative).

**BCE** — Binary Cross-Entropy. Per-label loss summed/averaged over all terms;
the loss for multi-label sigmoid classification.

**CAFA** — Critical Assessment of Functional Annotation. Community benchmark that
standardized Fmax/Smin and rigorous time-based evaluation. CAFA 5 = the 2023–24 Kaggle edition.

**CCO / BPO / MFO** — The three GO sub-ontologies: Cellular Component, Biological
Process, Molecular Function. Scored separately.

**CLS pooling** — Using the special start-of-sequence token's representation as the
protein-level summary (BERT-style). Alternative to mean pooling.

**DAG** — Directed Acyclic Graph. The GO's structure; terms have multiple parents.

**ESM-2 / ESM-C** — Protein language models (Meta / EvolutionaryScale). Produce
per-residue embeddings of dimension `d`. See `04-models-and-tools.md`.

**Fmax** — Maximum protein-centric F1 over a threshold sweep. The headline AFP metric.

**GO** — Gene Ontology. Controlled vocabulary of protein function as a DAG of terms.

**GOA** — GO Annotation database (UniProt-GOA): evidence-coded protein→GO links.

**IA (Information Accretion)** — A term's information content given its parents;
used to weight Smin and to select informative term sets.

**IEA** — Inferred from Electronic Annotation (a GO evidence code). Non-experimental;
include/exclude is a deliberate data-prep choice.

**Mean pooling** — Averaging per-residue embeddings over length to get one
`d`-dimensional protein vector. Default protein-level summary.

**Multi-label classification** — Each protein can have many labels simultaneously;
labels are not mutually exclusive ⇒ **sigmoid** per term, not softmax.

**pLM** — protein Language Model.

**PPI** — Protein-Protein Interaction. The interactome-prediction stage of the system.

**Sigmoid (vs. softmax)** — Sigmoid treats each GO term as an independent binary
question (correct for multi-label). Softmax forces outputs to sum to 1 (single-label only).

**Smin** — Semantic-distance metric; minimum over thresholds of an IA-weighted
distance between predicted and true term sets.

**True-path rule** — If a protein has a GO term, it implicitly has all that term's
ancestors. Annotations are **propagated up** the DAG before training; predictions
are **corrected** to respect it at inference.

**Zero-shot** — Predicting for a pathogen/organism with no examples in training
(here: whole-organism or whole-family holdout). The SBIR's key feasibility demand.
