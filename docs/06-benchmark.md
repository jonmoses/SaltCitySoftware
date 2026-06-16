# Virus-only NetGO-3.0-style benchmark

A faithful-as-possible replication of NetGO 3.0's benchmarking methodology, adapted
for viruses. `python -m viral_annotation.benchmark.run [--cutoff YYYYMMDD] [--min-count N]`.

## Methodology (what we replicate)

- **Temporal (CAFA) split** — the defining NetGO method. Dated experimental
  annotations (QuickGO, ECO:0000269 + descendants = EXP/IDA/IPI/IMP/IGI/IEP) split
  by a cutoff date: train = annotations before the cutoff; test = **no-knowledge**
  proteins (no experimental annotation in that ontology before the cutoff, gains
  one after). `data/quickgo.py`, `benchmark/temporal.py`.
- **Three metrics, per ontology** (MFO/BPO/CCO): protein-centric **Fmax**,
  term-centric **M-AUPR** (mean average-precision per GO term), and **Smin**
  (Information-Accretion-weighted semantic distance). `evaluation/metrics.py`.
- **Methods** mirroring NetGO's components: Naive, BLAST-KNN (homology), LR-ESM
  (mean-pool ESM + linear), Ensemble (LR-ESM + homology, weights tuned on a val
  carve). `benchmark/run.py`.
- Rare terms kept (low `min_count`), as NetGO advocates.

## Result (cutoff 2024-01-01; MFO/BPO/CCO test 457/70/271)

| metric | method | MFO | BPO | CCO |
|--------|--------|-----|-----|-----|
| Fmax ↑   | Naive | **0.724** | **0.516** | **0.753** |
|          | Ensemble | 0.518 | 0.298 | 0.659 |
| M-AUPR ↑ | Naive | 0.038 | 0.099 | 0.196 |
|          | **Ensemble** | **0.225** | **0.206** | **0.416** |

**Key finding:** Naive wins Fmax/Smin but collapses on M-AUPR; the Ensemble leads
M-AUPR across all three ontologies. On a small, homogeneous viral no-knowledge test
set, true-path propagation makes every protein share high-level terms, so
protein-centric Fmax flatters the prior — exactly why CAFA/NetGO report the
term-centric M-AUPR too. On that discriminative metric our methods clearly beat
Naive, mirroring NetGO 3.0 being best in its Table 1.

## Honest caveats (virus-only "as close as we can")

- QuickGO `date` is the LAST-UPDATE date, not the original assertion date, so NK
  selection is approximate (a 2026 bulk re-dating spike inflates the test set —
  MFO test 457 > train 447 is a tell). CAFA uses dated DB snapshots we can't easily
  obtain.
- Small viral test sets (BPO test = 70) → noisy, wide implicit CIs.
- No HUMAN/MOUSE species split (all viral). Not included from full NetGO: difficult
  proteins, term-frequency groups, bootstrap CIs (deferred — core methodology first).
