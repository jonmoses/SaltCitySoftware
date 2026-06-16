# Walkthrough 3 — Full Engineering Deep Dive

This is the complete, implementation-level walkthrough. It assumes the conceptual
background from Walkthroughs 1 and 2 (protein language models, GO, the true-path rule,
evidence tiers, leakage-safe splits) and instead traces **every major step from data
ingest to output**, naming the specific module responsible for each. It stops short of
reproducing code, but it points you at the exact file (and function) to read next.

Companion reference docs live alongside this one: `docs/01-annotation-pipeline-design.md`
(architecture + open hyperparameters), `docs/03-evaluation-protocol.md` (metrics and
splits), `docs/06-benchmark.md` (the temporal benchmark), and
`docs/07-threat-characterization.md` (Stage 3).

---

## 0. Package layout and cross-cutting conventions

```
src/viral_annotation/
  config.py              # single source of paths, model registry, constants, policies
  ontology/go_dag.py     # OBO parse + true-path propagation/correction (pure stdlib)
  data/                  # ingest, labeling, clustering, splitting, vocab, homology, targets
  embeddings/            # ESM-2 wrapper + pooled cache + per-residue cache
  classifier/            # the head architecture, attention pooler, ensemble, serving
  training/              # the config-driven trainer + shared head/pipeline helpers
  evaluation/            # Fmax / M-AUPR / Smin metrics + the shared report renderer
  benchmark/             # NetGO-style temporal split + runner
  cli/                   # argparse entry points (va-train, va-benchmark, va-threat)
  threat.py              # Stage-3 danger-category characterization
```

Two conventions appear everywhere and are worth stating once:

- **Lazy heavy imports.** `torch`, `transformers`, `numpy`, and `sklearn` are imported
  *inside* functions, never at module top level. This lets the pure-stdlib core
  (`ontology/go_dag.py`, `evaluation/metrics.py`'s set API, `data/labels.py` fetch) be
  imported, tested, and run without the heavy `[ml]` extra installed. The `[ml]` extra
  is opt-in.
- **Config centralization.** Every path, threshold, seed, and the model registry live
  in `config.py`. Anything described there as a hyperparameter (ESM layer, head depth)
  is deliberately *not* frozen as a final value.

The forward path (`embed → predict → correct`) is written **once** and shared between
training-time evaluation and serving, so the two cannot silently diverge — this is an
explicit design goal noted in `classifier/serving.py`.

---

## 1. Data ingest — UniProt → labeled proteins

### 1.1 Fetch (`data/labels.py`)

`labels.fetch_raw` streams viral reviewed Swiss-Prot entries from the UniProt REST
API. Details that matter:

- Query and fields come from `config.py` (`UNIPROT_VIRAL_QUERY` = `(reviewed:true) AND
  (taxonomy_id:10239)`, where 10239 is the Viruses taxon; `UNIPROT_FIELDS` requests
  accession, sequence, organism, lineage, and `go_id` with evidence).
- Paging is **cursor-based**: each response's `Link: ...rel="next"` header carries the
  next page URL. `_parse_next_link` matches inside the angle brackets because the URL
  itself contains commas (the fields list).
- `_get` retries with exponential backoff (2/4/8s, `_MAX_RETRIES`) because the full
  set is ~35 pages and one transient timeout shouldn't abort the run.
- `_parse_entry` walks each entry's `uniProtKBCrossReferences`, keeps `database == "GO"`
  rows, reads the `GoEvidenceType` property, and tags each annotation **manual** vs
  **iea** by `config.IEA_EVIDENCE_PREFIX` ("IEA"). It produces a `RawProtein`
  (accession, sequence, organism, lineage, list of `(go_id, tier)`).

Raw records can be cached to JSONL (`save_raw`/`load_raw`, `config.VIRAL_RECORDS_PATH`)
so re-runs skip the network. Fetch is intentionally network-only — it does **not**
load the 31 MB ontology.

### 1.2 Labeling (`data/labels.py` → `label_proteins`)

`label_proteins` needs the GO DAG and produces, per protein, **three** propagated term
sets on a `LabeledProtein`:

- `terms_manual = propagate(manual)` — the val/test ground truth (non-circular).
- `terms_all = propagate(manual + iea)` — candidate training labels.
- `terms_iea = propagate(iea)` — the InterPro2GO/UniRule signal in isolation; a
  *test-time feature* for the ensemble, **never** a training label.

`n_manual`/`n_iea` retain the pre-propagation raw counts; `has_manual` (≥1 manual
annotation) is the predicate the split uses to decide eligibility for val/test.

### 1.3 The ontology (`ontology/go_dag.py`)

`GoDag.from_obo` parses `go-basic.obo` (fetched by `va-download-go`, URL in
`config.GO_BASIC_URL`). `_parse_obo` is a minimal streaming OBO 1.2 parser reading
only `id`, `name`, `namespace`, `is_a`, `relationship: part_of`, `is_obsolete`, and
`alt_id`. Only `is_a` and `part_of` edges are treated as parents
(`config.PROPAGATION_RELATIONS`) — the standard safe choice on go-basic. Obsolete
terms are dropped by default and dangling parent edges to them are pruned.

The class exposes the operations the rest of the pipeline relies on:
- `resolve` maps secondary (alt) ids to primary, so old ids still work.
- `ancestors` / `descendants` are BFS with memoization; `descendants` lazily builds the
  inverted child index on first call (used to expand danger roots in Stage 3).
- `propagate(term_ids)` = union of ancestors-including-self — the true-path rule for
  building labels.
- `correct_scores(scores)` = post-hoc hierarchical correction: each ancestor's score is
  lifted to at least its most confident descendant's. This is the inverse direction of
  `propagate` and runs on predictions.

This module is pure stdlib and is the most heavily unit-tested piece (see
`tests/test_go_dag.py`).

---

## 2. Splitting — leakage-safe, asymmetric, family-holdout

### 2.1 Clustering (`data/cluster.py`)

`cluster_sequences` shells out to **MMseqs2** `easy-cluster` at
`config.CLUSTER_MIN_SEQ_ID` (0.30) and `config.CLUSTER_COVERAGE` (0.80), parses
`clu_cluster.tsv`, and returns `accession → cluster-representative`. Requires the
`mmseqs` binary on PATH; `mmseqs_available` guards it.

### 2.2 The split (`data/split.py`)

`cluster_split` combines three constraints simultaneously:

1. **Family holdout.** Every protein whose lineage family (`family_of` — first clade
   ending in `viridae`) equals `holdout_family` is pulled out entirely; its
   manual-having members become the zero-shot `holdout` set. Default
   `config.HOLDOUT_FAMILY = "Coronaviridae"`.
2. **Cluster integrity.** Proteins are grouped by cluster representative; whole clusters
   are assigned to train/val/test by `config.SPLIT_RATIOS` (0.70/0.15/0.15) over the
   *manual-having* clusters, seeded by `config.SPLIT_SEED` (1337). No val/test protein
   can have a ≥30%-identity homolog in train.
3. **Asymmetric evidence.** val/test contain only manual-having proteins. IEA-only
   members of a val/test cluster are **dropped** (they can't go to train without
   leaking); IEA-only clusters feed train (their IEA labels are still signal).

`split_proteins` is the non-leakage-safe random fallback (`--random-split`), kept for
quick checks. `Split.summary()` reports the train/val/test/holdout breakdown.

> This split is the reason the headline numbers are trustworthy: the README notes the
> cluster split dropped ~1,570 IEA homologs of val/test proteins that a random split
> would have leaked into train.

---

## 3. Vocabulary and label matrices (`data/dataset.py`)

`select_vocab` chooses the classifier's output columns from the **training** labels: a
term is kept iff at least `config.MIN_TERM_COUNT` (10) training proteins carry it
(after propagation), it is not one of the three ontology roots (`config.GO_ROOTS`, which
sit on everything and are uninformative), and it is in the requested namespace(s). The
`field` argument selects which label set to count (`terms_all` vs `terms_manual`) —
this is how the per-namespace evidence policy is honored at vocab-selection time. Terms
are sorted by `(namespace, id)` for stable columns. The returned `TermVocab` carries
`terms` (column order), `index` (term→column), `namespaces` (per column), and
`columns_by_namespace()` for per-aspect scoring.

`build_labels` turns a protein list + vocab into a multi-hot `[P × N]` matrix, reading
whichever `field` the policy dictates (train uses `terms_all`/`terms_manual` per
policy; val/test always use `terms_manual`).

---

## 4. Embeddings — the ESM-2 feature extractor

### 4.1 The model wrapper (`embeddings/esm.py`)

`ESMEmbedder` wraps a HuggingFace ESM-2 model (registry in `config.ESM2_MODELS`;
default `650M` → `facebook/esm2_t33_650M_UR50D`, `dim=1280`). `_ensure_loaded` lazily
loads tokenizer + model and picks the device (`_auto_device`: CUDA → Apple MPS → CPU).
`repr_layer=None` means the last hidden layer (which layer to pool is flagged as a
hyperparameter, not necessarily optimal).

Key engineering in the forward path:

- **Token-budget batching** (`_forward_batches`). Sequences are length-sorted, then
  greedily packed into batches under `max_tokens` (default 4096, = count × longest),
  so a few long sequences can't blow up the O(L²) attention memory. GPU buffers are
  freed (`empty_cache`) after each batch. Only the needed layer is materialized.
- **Three pooling modes**, selected at construction:
  - `mean` / `cls` via `_embed_flat` → `_pool` (`cls` = token 0; `mean` = masked mean
    over real residues; the attention mask covers BOS/EOS, a documented baseline
    choice).
  - `stats` via `_embed_stats` → 4×d = `concat(mean, max, min, std)` over residues. Max
    catches active-site-like signals the mean dilutes; std catches heterogeneity.
- **Windowing for long proteins.** ESM-2's input cap is `max_length` (1022). With
  `window=True`, `_windows` splits a sequence into non-overlapping chunks; the protein
  is embedded as windows and recombined to **one** per-protein vector:
  - mean: length-weighted average of window vectors;
  - stats: `_finalize_stats` combines per-window **sufficient statistics** — `max`/`min`
    taken element-wise *across* windows and `std` computed over *all* residues — so the
    statistics are exact, not per-window averages.
  The prediction is never made on a fragment; only the embedding is windowed (README
  notes ~7.8% of the set exceeds 1022 aa).
- **Per-residue output** (`embed_residues`). For attention pooling we need the full
  `[L × d]` matrix, BOS/EOS stripped, windows concatenated in order, stored as
  **float16**. This is what the attention pooler ranges over.

### 4.2 Two caches

ESM is frozen, so every sequence is embedded **once** and reused across runs, threshold
sweeps, and classifier variants.

- **Pooled cache** (`embeddings/cache.py`). One `.npz` per `(model, pooling, layer,
  windowed?)` config holding `{ids, embeddings}`. `embed_records` is incremental
  (computes only missing accessions), checkpoints every 1024-protein chunk, and writes
  via temp-file-then-rename so a crash can't corrupt the cache. A neat optimization:
  short proteins (≤ one window) embed identically whether windowed or not, so the
  windowed cache **seeds** them from the truncated cache instead of recomputing — only
  genuinely long proteins need the windowed forward pass.
- **Per-residue cache** (`embeddings/residue_cache.py`). One float16 `.npy` per protein
  (`[L × d]`) under a per-config directory. fp16 halves the footprint (~20 GB for the
  full set, per the README); written per-protein so a long run checkpoints. This cache
  is only built when some namespace uses attention pooling.

---

## 5. The classifier heads

### 5.1 Pooled head (`classifier/model.py`)

`build_classifier(input_dim, num_terms, hidden_dims)` returns a plain
`torch.nn.Sequential`: a linear baseline when `hidden_dims` is empty (the default — the
README notes NetGO 3.0's LR-ESM shows logistic regression on pLM embeddings is
competitive), or an MLP with ReLU/optional dropout when given hidden widths. Output is
**raw logits**; sigmoid is applied at inference (`predict_proba`) and internally by
`BCEWithLogitsLoss` at train time. `predict_proba` is the shared, batched forward used
by both evaluation and serving.

### 5.2 Attention pooler (`classifier/pooling.py`)

`build_attn_classifier` returns an `AttnPoolClassifier`: `H` learned query vectors
(`config`-driven `ATTN_HEADS=8`), each producing a softmax over the protein's residues
(padding masked to `-inf`) and a weighted-sum vector; the `H` pooled vectors are
concatenated (`H·d`) and fed to a `build_classifier` head. Different heads can
specialize (catalytic site, sorting signal). It takes `[B, L, d]` + a validity mask and
emits `[B, num_terms]` logits — so pooling and classification are trained **jointly**.

### 5.3 The unifying `Head` abstraction (`training/heads.py`)

This module is the single home for "ESM features → GO scores," replacing what used to
be three separate trainer files. It exposes:

- `fit_pooled_head` — trains a linear/MLP head on **fixed pooled** features
  (`Xtr/Ytr`), early-stopping on **validation Fmax** (`config.TRAIN_EARLY_STOP_PATIENCE`
  = 10), with `BCEWithLogitsLoss` and per-term positive weights.
- `fit_attention_head` — trains the attention pooler + head **jointly** over a
  per-residue `Dataset` (`make_residue_dataset` + `collate_residues`, which pads
  variable-length residue tensors and builds the mask). `predict_residues` runs it in
  eval mode.
- `compute_pos_weight` — per-term BCE positive weight = neg/pos, clamped to
  `config.POS_WEIGHT_CLAMP` (100) — the class-imbalance counter.
- `cap_pool` — for attention only, bounds per-epoch disk reads by subsampling the train
  pool to a cap while keeping **all** manual-having proteins plus a seeded sample of the
  IEA-only remainder.
- `fit_namespace` — the **dispatch**. Given a namespace, its policy, and a pooling, it
  selects the vocab from the policy's pool/field, builds labels, computes pos-weights,
  trains the right head, and returns a uniform `Head` dataclass: `vocab`, a `predict()`
  closure, the training `prior` (term frequencies, for the Naive baseline), `val_fmax`,
  `epochs`, `pooling`, and `state` (the torch state_dict — `None` for attention heads,
  which aren't servable). The `predict()` closure hides whether the head re-embeds
  pooled features or runs the residue dataset, so the trainer treats both identically.

---

## 6. Training orchestration (`training/train.py`, `training/pipeline.py`)

`pipeline.py` holds the shared front half (so the trainer and benchmark don't
copy-paste it): `auto_device`, `load_proteins` (fetch + label), `make_split`
(cluster-or-random + family holdout), and `annotation_stats`.

`train.run` is the single config-driven path (`va-train`, `cli/train.py`):

1. Seed everything (`config.TRAIN_SEED` = 1337), pick device, assemble the
   hyperparameter `SimpleNamespace`.
2. Resolve pooling. `--pooling per-namespace` reads `config.NAMESPACE_POLICY` to get
   `{MF: attention, BP: mean, CC: mean}`; a single value applies it everywhere.
3. Load the DAG, fetch+label proteins, build the split. Define two train **pools**:
   `all` (everything) and `manual_having` (the homology DB and the MF train pool).
4. If any namespace uses attention, cache per-residue embeddings for the relevant pools
   + val + test + holdout (`cache_residues`).
5. **Per namespace** (`GO_NAMESPACES`), `fit_namespace` per `NAMESPACE_POLICY[ns]`,
   optionally fit ensemble weights (§7), then score test and (if present) the held-out
   family via the local `scored()` closure, which applies `apply_hierarchical_correction`
   and — when ensembling — fuses in homology before a second correction. `_eval_split`
   accumulates `[P × N]` blocks (model / true / naive / pLM-only) for the overall
   metric and the per-namespace report rows.
6. `_print_reports` renders the TEST and ZERO-SHOT tables (`evaluation/report.py`),
   including the ensemble-vs-pLM lift lines.
7. `_save` persists **pooled** heads only — `models/go_classifier.pt` (state dicts keyed
   by namespace) + `go_classifier.meta.json` (esm model, pooling, per-head ordered
   vocab, metrics). It refuses to save if any head used attention pooling (those aren't
   servable by the lightweight loader) and tells you to use `--pooling mean` for a
   deployable model.

The evidence policy (`config.NAMESPACE_POLICY`) is *always* applied: MF trains
manual-only (`train_pool=manual_having`, `train_field=terms_manual`), BP/CC train on
manual+IEA (`train_pool=all`, `train_field=terms_all`); val/test always score
manual-only. This is the fix for the IEA/manual MF distribution shift (MF Fmax 0.09 →
~0.20) documented in `docs/01` and project memory.

---

## 7. The homology ensemble

### 7.1 BLAST-KNN component (`data/homology.py`)

`homology_scores(query, db, dag, vocab)` MMseqs2 `easy-search`es the queries against an
annotated database (the manual-having training proteins) with deliberately **permissive**
settings (`SEARCH_SENSITIVITY=7.5`, `SEARCH_EVALUE=10.0`) so distant cross-family
relatives still produce hits — exactly the zero-shot regime we want signal in. It then
scores each term by the **bitscore-weighted fraction** of a query's hits that carry it
(the GOLabeler/NetGO formula, `_aggregate`), restricted to the namespace vocab and
transferring the neighbors' *manual* labels (self-hits skipped). Returns `[len(query) ×
len(vocab)]`.

### 7.2 Late fusion (`classifier/ensemble.py`)

`fuse` is a weighted sum of component score matrices; `search_weights` grid-searches the
non-pLM components' weights (`WEIGHT_GRID = 0, 0.25, 0.5, 1.0, 2.0`) to maximize
**validation Fmax**, with the pLM pinned at 1.0 (Fmax sweeps the threshold, so absolute
scale is absorbed). Weights are fit **per namespace** in `train._fit_ensemble_weights`,
so e.g. the IEA/InterPro component can get weight 0 where it doesn't help.

The documented effect: homology flips overall zero-shot from below the Naive prior to
clearly above it (BP 0.356 → 0.398). A third component (InterPro2GO via the `terms_iea`
proxy) is wired in but tunes to weight 0 — validation-tuned weights don't reward its
specifically-zero-shot value; a zero-shot tuning holdout would be the fix.

---

## 8. Evaluation (`evaluation/metrics.py`, `evaluation/report.py`)

- **Fmax** — `fmax` (set-of-terms API, pure stdlib) and `fmax_matrix` (vectorized,
  numpy, fast enough to call every epoch) implement the CAFA protein-centric
  convention: at each threshold τ, precision is averaged **only** over proteins with ≥1
  prediction at τ, recall over **all** proteins; report the best F1 over τ. Inputs are
  assumed already true-path corrected.
- **`apply_hierarchical_correction`** — applies `GoDag.correct_scores` row-by-row to a
  `[P × N]` matrix before scoring (the §1.3 post-hoc correction at matrix scale).
- **M-AUPR** — `m_aupr`, term-centric mean average precision (NetGO's metric), via
  sklearn.
- **Smin** — `smin` with `information_accretion` (IA(t) = −log₂ P(t | parents(t)) from a
  reference corpus); the CAFA information-content-weighted distance, lower is better.
- **`fmax_by_namespace`** — per-namespace + micro-overall using
  `vocab.columns_by_namespace()`.
- **Report** — `report.overall_fmax` concatenates the disjoint per-namespace column
  blocks and scores the whole matrix; `report.print_table` renders the uniform
  "Fmax vs naive, per namespace and overall" table used by both the trainer and the
  benchmark.

---

## 9. The temporal benchmark (`benchmark/`, `va-benchmark`)

A separate, NetGO-3.0-style evaluation that does **not** reuse the cluster split — it
uses a **time** split, which is the CAFA gold standard.

- **Dated annotations** (`data/quickgo.py`). UniProt's flat fields aren't dated, so
  `quickgo.fetch_or_load` pulls viral (taxon 10239) **experimental-evidence**
  annotations (`ECO:0000269` + descendants) from QuickGO, each with an assertion date.
- **Temporal split** (`benchmark/temporal.py`). Per ontology, a protein is **train** if
  it has an experimental annotation *before* the cutoff (labels = propagate of
  before-cutoff terms); it is a **no-knowledge test** protein if it has none before but
  gains one after (labels = propagate of all its terms). `BenchProtein` deliberately
  names its field `terms_manual` so it drops into `homology_scores` / `build_labels` /
  `select_vocab` / `embed_records` unchanged. The caveat (QuickGO's date is last-update,
  not original assertion) is documented in the module.
- **Runner** (`benchmark/run.py`). Per ontology it evaluates four methods — **Naive**,
  **BLAST-KNN**, **LR-ESM** (a mean-pooled pooled head), and **Ensemble** (their fusion,
  weights tuned on a carved-out validation slice) — reporting **Fmax / M-AUPR / Smin**,
  the structure of NetGO 3.0's Table 1. The finding (see `docs/06-benchmark.md`): Fmax
  flatters Naive on the small viral test set, while the Ensemble wins the term-centric
  M-AUPR.

---

## 10. Serving (`classifier/serving.py`)

`GoAnnotator.load` rebuilds the per-namespace heads from `models/go_classifier.pt` +
`go_classifier.meta.json`, inferring each head's embedding width from the first linear
weight's shape (so the pooling's feature dimension needn't be re-derived). `annotate`:

1. Embeds the input records **once** with the model's saved pooling, windowed, via the
   shared `embed_records` (cached).
2. Runs every namespace head (`predict_proba`), merges the per-namespace probabilities
   into one `term → prob` map per protein.
3. Applies `GoDag.correct_scores` (true-path correction) and drops terms below a
   threshold.

It returns `AnnotatedProtein` records. Because it calls the *same* `embed_records` +
`predict_proba` + `correct_scores` as training, serving and evaluation cannot diverge.

---

## 11. Stage 3 — threat characterization (`threat.py`, `data/danger_terms.py`, `va-threat`)

### 11.1 The danger ontology (`data/danger_terms.py`)

A **hand-curated, auditable** map: `DANGER_CATEGORIES` lists six `DangerCategory`
records (toxin, host-cell entry & fusion, host-cell killing/lysis, immune evasion,
apoptosis manipulation, host-machinery hijack), each pinned to a few high-level GO
**root** ids with a `rationale` and per-root name comments for audit. All ids were
verified live and non-obsolete in go-basic (2026-06); the module notes GO's rename of
"by virus of host" → "symbiont-mediated" and that several classic ids are now obsolete.

### 11.2 Characterization (`threat.py`)

- `build_danger_map` expands every category's roots to **all descendants** over the live
  DAG (`GoDag.descendants`), and **re-asserts** each root is present — a future ontology
  update that obsoletes a root fails loudly rather than silently characterizing nothing.
- `background_rates` computes the mean predicted probability of each danger term across
  the proteome — the "average viral protein" baseline.
- `characterize_protein` intersects one `AnnotatedProtein`'s predicted terms with each
  category's term set, recording `TermHit`s with their **lift over background**
  (`prob − background`).
- `characterize_proteome` rolls these up into a `ProteomeThreat` with two views, because
  raw absolute probability flags everything (generic entry/immune terms sit on nearly
  every viral protein after propagation — the same base-rate effect that flatters Naive
  in the benchmark):
  - **category fingerprint** — `category_peaks()`, the peak confidence per category
    (which mechanisms are present; comparable across viruses);
  - **standout proteins** — `ranked()` by **lift over the proteome background**, which
    surfaces the specific proteins driving a mechanism above the viral crowd (the entry
    glycoprotein, the interferon antagonist) rather than the universal baseline.

`format_report` renders the human-readable profile; `to_dict` emits the JSON record. The
output is explicitly a **triage hypothesis, not a determination**.

### 11.3 Target acquisition (`data/proteomes.py`) and the CLI (`cli/threat.py`)

`proteomes.fetch_target` pulls one named virus by taxonomy id — **including unreviewed
TrEMBL** — to simulate a freshly sequenced, never-curated pathogen, reusing
`labels.fetch_raw`/`save_raw`/`load_raw` with a per-virus cache and exact-sequence dedup
(TrEMBL is dominated by near-identical strain copies). `TARGET_VIRUSES` is the
hemorrhagic-fever panel (Ebola, Nipah, Lassa, Marburg); Ebola has 0 reviewed entries, so
it's a true uncurated-proteome case. `cli/threat.py` wires it together: `va-threat
--panel` annotates the whole panel and prints a side-by-side category×virus peak table;
`--taxon`/`--fasta`/`--name` run a single target. **Honest-scope note** (in the module
docstring): only Coronaviridae is held out of training, so the panel families are
*in-distribution* — this is "annotate + triage an unknown sample," not a novel-family
zero-shot test.

---

## 12. End-to-end, as a single trace

```
va-train
  pipeline.load_proteins                 labels.fetch_raw → label_proteins → propagate (go_dag)
  pipeline.make_split                    cluster_sequences (MMseqs2) → cluster_split (+ Coronaviridae holdout)
  [if attention] cache_residues          esm.embed_residues → residue_cache (fp16 [L×d])
  per namespace:
    fit_namespace                        select_vocab → build_labels → embed_records (esm, cached)
                                         → fit_pooled_head | fit_attention_head (early-stop on val Fmax)
    [if --ensemble] search_weights       homology_scores (MMseqs2 BLAST-KNN) → grid-search on val
    scored()                             apply_hierarchical_correction → [fuse] → correct again
    _eval_split                          fmax_matrix vs Naive prior, accumulate blocks
  _print_reports                         report.print_table (TEST + ZERO-SHOT)
  _save                                  models/go_classifier.pt (+ meta) — pooled heads only

va-threat --panel
  proteomes.fetch_target                 UniProt by taxon, TrEMBL incl., dedup
  GoAnnotator.load + annotate            embed_records → per-ns predict_proba → correct_scores
  threat.characterize_proteome           danger_map (descendants) ∩ predictions, lift over background
  format_report / _panel_table           fingerprint + standout proteins
```

That is the whole system, from a UniProt query to a per-virus threat profile, with the
specific module responsible for each step. To go deeper on *why* a given choice was made
(rather than *where* it lives), the design rationale is in `docs/01`, the evaluation
rationale in `docs/03`, and the empirical findings in the README's results section and
project memory.
