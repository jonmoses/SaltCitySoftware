"""Central configuration: paths, model registry, and pipeline constants.

Values here are the *defaults / knowns*. Anything marked as a hyperparameter in
docs/01-annotation-pipeline-design.md (e.g. which ESM layer to pool, classifier
depth, thresholds) is deliberately NOT hard-coded as a final value — it is a
sweep, and lives with the training config when that module is built.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Repository paths -------------------------------------------------------
# config.py is at src/viral_annotation/config.py -> repo root is parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
GO_OBO_PATH = DATA_DIR / "go-basic.obo"
EMBEDDINGS_CACHE = DATA_DIR / "embeddings_cache"

# Canonical, always-current go-basic ontology (filtered DAG safe for upward
# is_a/part_of propagation). See docs/02-data-sources.md.
GO_BASIC_URL = "https://current.geneontology.org/ontology/go-basic.obo"

# GO sub-ontologies. Scored separately; see docs/03-evaluation-protocol.md.
GO_NAMESPACES = ("molecular_function", "biological_process", "cellular_component")

# The three ontology roots. After true-path propagation these sit on essentially
# every protein, so they are trivial (uninformative) prediction targets and are
# excluded from the classifier's term set.
GO_ROOTS = frozenset({"GO:0003674", "GO:0008150", "GO:0005575"})

# Relationship types propagated under the true-path rule. go-basic restricts the
# graph so traversing these stays a clean DAG.
PROPAGATION_RELATIONS = ("is_a", "part_of")


# --- Training-data acquisition (UniProt) ------------------------------------
# Viral reviewed Swiss-Prot entries. Taxonomy 10239 = Viruses (descendants
# included by UniProt). See docs/02-data-sources.md.
UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
VIRUSES_TAXON_ID = 10239
UNIPROT_VIRAL_QUERY = "(reviewed:true) AND (taxonomy_id:10239)"
# Fields returned in the JSON stream. `go` carries per-annotation evidence.
UNIPROT_FIELDS = ("accession", "sequence", "organism_name", "lineage", "go_id")

# Evidence handling (docs decision): we KEEP all annotations and tag the tier.
# An annotation is "iea" (electronic) if its GO evidence code starts with "IEA";
# everything else (EXP/IDA/IPI/IMP/IGI/IEP, TAS/IC, ...) is "manual".
IEA_EVIDENCE_PREFIX = "IEA"

# Local cache of the raw fetched records (JSONL), so re-runs skip the network.
VIRAL_RECORDS_PATH = DATA_DIR / "viral_reviewed.jsonl"


# --- Term-set selection -----------------------------------------------------
# Keep a GO term as a prediction target only if at least this many TRAIN proteins
# carry it (after propagation). Floors class imbalance; N is data-dependent.
MIN_TERM_COUNT = 10

# --- Per-namespace evidence policy ------------------------------------------
# A full-set experiment showed the asymmetric IEA-train/manual-test policy
# BACKFIRES for Molecular Function in viruses: IEA-MF (InterPro/UniRule domain
# rules -> generic ligand/nucleotide binding) is nearly disjoint from manual-MF
# (curated protein-binding/adaptor terms), so training on IEA poisons MF
# (Fmax 0.09). Training MF manual-only ~doubles it (0.20). BP/CC are robust
# because their IEA and manual vocabularies overlap. So each namespace gets its
# own policy. See memory: iea-manual-mf-distribution-shift.
#
#   train_pool:  "all" (every train protein) | "manual_having" (>=1 manual term)
#   train_field: which label set trains the head ("terms_all" = manual+IEA,
#                "terms_manual" = manual only)
#   vocab_field: which label set the term vocabulary is selected from
# Validation/test always score against manual-only labels regardless of policy.
# `pooling` selects the residue->protein reduction per namespace. A full-set
# comparison (memory: pooling-comparison) showed learned ATTENTION pooling wins
# zero-shot MF decisively (0.46 vs 0.25 mean) — the conserved catalytic/binding
# residues it focuses on transfer to unseen families — while mean is best for
# BP/CC (no localized signal, and attention costs a ~20GB per-residue cache).
NAMESPACE_POLICY = {
    "molecular_function": {
        "train_pool": "manual_having", "train_field": "terms_manual",
        "vocab_field": "terms_manual", "pooling": "attention",
    },
    "biological_process": {
        "train_pool": "all", "train_field": "terms_all",
        "vocab_field": "terms_all", "pooling": "mean",
    },
    "cellular_component": {
        "train_pool": "all", "train_field": "terms_all",
        "vocab_field": "terms_all", "pooling": "mean",
    },
}


# --- Train / val / test split ----------------------------------------------
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, val, test
SPLIT_SEED = 1337

# Sequence-identity clustering for the rigorous split (docs/03). Whole clusters
# go to one bucket, so no test protein has a >=30%-identity homolog in train.
CLUSTER_MIN_SEQ_ID = 0.30
CLUSTER_COVERAGE = 0.80            # MMseqs2 -c (alignment coverage)
CLUSTER_WORKDIR = DATA_DIR / "cluster_work"

# Held-out viral family for zero-shot validation (docs/03, SBIR F2). Coronaviridae
# = the SARS-CoV-2 family the topic cites (Gordon et al. 2020); 69 manual-having
# proteins, ~2.8% of the set. Entirely excluded from train/val/test, then the
# model is scored on recovering its known (in-vocab) functions.
HOLDOUT_FAMILY = "Coronaviridae"


# --- Classifier training hyperparameters ------------------------------------
# First-round defaults for the linear head; these are starting points, not tuned.
TRAIN_LR = 1e-3
TRAIN_WEIGHT_DECAY = 1e-4
TRAIN_EPOCHS = 100
TRAIN_BATCH_SIZE = 256
TRAIN_EARLY_STOP_PATIENCE = 10   # epochs without val-Fmax improvement
POS_WEIGHT_CLAMP = 100.0         # cap per-term BCE pos_weight to avoid blow-ups
TRAIN_SEED = 1337                # seed weight init + minibatch shuffle for reproducible runs

# Where trained artifacts land (state_dict + term index + run config).
MODELS_DIR = REPO_ROOT / "models"


# --- ESM-2 backbone registry ------------------------------------------------
@dataclass(frozen=True)
class ESMModel:
    """An ESM-2 variant. `dim` is the per-residue embedding width `d`."""

    hf_name: str
    params: str
    layers: int
    dim: int


# HuggingFace ids + dims verified against the ESM-2 release (docs/04-models-and-tools.md).
ESM2_MODELS: dict[str, ESMModel] = {
    "8M": ESMModel("facebook/esm2_t6_8M_UR50D", "8M", 6, 320),
    "35M": ESMModel("facebook/esm2_t12_35M_UR50D", "35M", 12, 480),
    "150M": ESMModel("facebook/esm2_t30_150M_UR50D", "150M", 30, 640),
    "650M": ESMModel("facebook/esm2_t33_650M_UR50D", "650M", 33, 1280),
    "3B": ESMModel("facebook/esm2_t36_3B_UR50D", "3B", 36, 2560),
    "15B": ESMModel("facebook/esm2_t48_15B_UR50D", "15B", 48, 5120),
}

# Default per OUTLINE.md / docs: 650M is the accuracy/throughput sweet spot and
# fits the <15-min timing budget for a viral proteome.
DEFAULT_ESM_MODEL = "650M"

# Pooling strategy for collapsing [L x d] -> [d]. "mean" is the robust default;
# "cls" (start-of-sequence token) is the alternative to benchmark.
DEFAULT_POOLING = "mean"


# --- Pathogen-domain profiles -----------------------------------------------
# The SBIR topic spans viral, bacterial, and parasitic pathogen classes (docs/00,
# requirement #1). Everything that is domain-specific — the UniProt taxon, the
# family-rank suffix used for the zero-shot holdout, the held-out family, the
# per-namespace evidence/pooling policy, and where the trained model lands — is
# bundled into a PathogenDomain profile so the (domain-agnostic) embedding,
# clustering, training, and threat machinery can be reused unchanged. Models are
# trained and served PER DOMAIN (separate heads + vocab), selected by taxonomy.

# Bacteria. Taxonomy 2 = Bacteria. The bacterial set is ~20x the viral one, so:
#   * pooling defaults to "mean" — the servable config, and the learned-attention
#     per-residue cache (~hundreds of GB at this scale) is impractical here;
#   * the term-frequency floor is raised (bigger corpus -> larger vocab otherwise);
#   * the evidence policy starts ASYMMETRIC for all three namespaces (train on
#     manual+IEA). Bacterial IEA is rich and reliable (orthology + curated domain
#     rules), so the viral MF-manual-only fix is NOT assumed to carry over — it is
#     re-derived by re-running the IEA-vs-manual MF diagnostic and only then
#     specialized if MF collapses. See docs/08-bacterial-extension.md.
BACTERIAL_NAMESPACE_POLICY = {
    ns: {
        "train_pool": "all", "train_field": "terms_all",
        "vocab_field": "terms_all", "pooling": "mean",
    }
    for ns in GO_NAMESPACES
}


@dataclass(frozen=True)
class PathogenDomain:
    """A pathogen class the pipeline can be trained/served for.

    `models_subdir` is "" for the viral domain (artifacts stay at MODELS_DIR root,
    preserving the existing models/go_classifier.pt); other domains nest under it.
    """

    key: str
    taxon_id: int
    uniprot_query: str
    family_suffixes: tuple[str, ...]   # lineage-clade suffixes that mark the holdout rank
    holdout_family: str | None
    namespace_policy: dict
    min_term_count: int
    default_pooling: str
    default_esm_model: str
    models_subdir: str

    @property
    def models_dir(self) -> Path:
        return MODELS_DIR / self.models_subdir if self.models_subdir else MODELS_DIR


DOMAINS: dict[str, PathogenDomain] = {
    "viral": PathogenDomain(
        key="viral",
        taxon_id=VIRUSES_TAXON_ID,
        uniprot_query=UNIPROT_VIRAL_QUERY,
        family_suffixes=("viridae",),          # ICTV family rank
        holdout_family=HOLDOUT_FAMILY,
        namespace_policy=NAMESPACE_POLICY,
        min_term_count=MIN_TERM_COUNT,
        default_pooling=DEFAULT_POOLING,
        default_esm_model=DEFAULT_ESM_MODEL,
        models_subdir="",                      # back-compat: MODELS_DIR root
    ),
    "bacterial": PathogenDomain(
        key="bacterial",
        taxon_id=2,
        uniprot_query="(reviewed:true) AND (taxonomy_id:2)",
        family_suffixes=("aceae",),            # LPSN/NCBI bacterial family rank
        holdout_family="Francisellaceae",      # tularemia agent; contained, BSL-3
        namespace_policy=BACTERIAL_NAMESPACE_POLICY,
        min_term_count=25,                     # higher floor for the ~20x-larger corpus
        default_pooling="mean",
        default_esm_model=DEFAULT_ESM_MODEL,
        models_subdir="bacterial",
    ),
}

DEFAULT_DOMAIN = "viral"


def get_domain(key: str = DEFAULT_DOMAIN) -> PathogenDomain:
    """Resolve a domain profile by key, with a helpful error."""
    if key not in DOMAINS:
        raise KeyError(f"unknown pathogen domain {key!r}; choose from {list(DOMAINS)}")
    return DOMAINS[key]
