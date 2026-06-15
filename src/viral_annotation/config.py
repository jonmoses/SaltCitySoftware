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


# --- Train / val / test split ----------------------------------------------
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, val, test
SPLIT_SEED = 1337


# --- Classifier training hyperparameters ------------------------------------
# First-round defaults for the linear head; these are starting points, not tuned.
TRAIN_LR = 1e-3
TRAIN_WEIGHT_DECAY = 1e-4
TRAIN_EPOCHS = 100
TRAIN_BATCH_SIZE = 256
TRAIN_EARLY_STOP_PATIENCE = 10   # epochs without val-Fmax improvement
POS_WEIGHT_CLAMP = 100.0         # cap per-term BCE pos_weight to avoid blow-ups

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
