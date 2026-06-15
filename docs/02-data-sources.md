# Data Sources Reference

Datasets and databases for training, labels, ontology, and benchmarking. Verify
licenses before redistribution; most are open for research use.

## Ontology (the GO DAG itself)

| Resource | What | Format | URL |
|----------|------|--------|-----|
| Gene Ontology | The ontology / DAG used for true-path propagation and ancestor lookup | OBO / OWL (`go.obo`, `go-basic.obo`) | https://geneontology.org/docs/download-ontology/ |

Use `go-basic.obo` for propagation — it is filtered to be a clean DAG safe for
upward ancestor traversal (`is_a` / `part_of`). Parse with `obonet`, `goatools`,
or `pronto` in Python.

## Training labels (protein → GO annotations)

| Resource | What | Format | URL |
|----------|------|--------|-----|
| UniProtKB / Swiss-Prot | Manually reviewed proteins + sequences; primary training set | FASTA, XML, TSV | https://www.uniprot.org/ |
| UniProt-GOA | Evidence-coded GO annotations for UniProtKB | GAF / GPAD | https://www.ebi.ac.uk/GOA/ |
| UniProt Viruses proteome sets | Reference proteomes for viral organisms | FASTA | https://www.uniprot.org/proteomes (filter Viruses) |

Notes:
- Prefer **experimental evidence codes** (EXP, IDA, IPI, IMP, IGI, IEP) for the
  gold standard; decide explicitly whether to include electronic (IEA) annotations.
- Viral GO annotations use **host-relative terms** (e.g. "host cell cytoplasm")
  — keep these distinct from the host's own-compartment terms in the label set.

## Benchmark / evaluation datasets

| Resource | What | URL |
|----------|------|-----|
| CAFA 5 (Kaggle) | Standard AFP benchmark; train/test splits, IA values, Fmax scoring | https://www.kaggle.com/competitions/cafa-5-protein-function-prediction |
| CAFA (consortium) | Methodology, time-based holdout design | https://biofunctionprediction.org/cafa/ |

CAFA 5 used term-set caps of ~1500 (BPO) / ~800 (CCO) / ~800 (MFO) by frequency +
Information Accretion — a reasonable template for choosing N. Mean Fmax across
teams approached ~0.65 (MFO best, up to ~0.8). Useful target band to position against.

## Host-pathogen interaction databases (for the PPI stage / zero-shot validation)

Not needed for the first annotation build, but the same proteins/organisms recur,
and these define the zero-shot validation sets the program requires.

| Resource | What | Format | URL |
|----------|------|--------|-----|
| VirHostNet (3.0-beta) | Curated virus/host molecular interaction network; aggregates IntAct, MINT, DIP, BioGRID, UniProt, MatrixDB, HPIDB, IMEx via PSICQUIC | PSI-MITAB 2.5 (single TSV download) | https://virhostnet.prabi.fr/ |
| HPIDB 2.0 | Curated host-pathogen interactions (much of it from VirHostNet + IntAct) | PSI-MITAB | https://hpidb.igbb.msstate.edu/ |
| IntAct | General curated molecular interactions | PSI-MITAB / PSI-XML | https://www.ebi.ac.uk/intact/ |
| PHI-base | Pathogen-host interactions w/ phenotype annotation (cited by topic) | CSV / web | http://www.phi-base.org/ |
| BioGRID | General PPI; used as positives in Hallee et al. PPI work | TAB3 / PSI-MITAB | https://thebiogrid.org/ |
| Negatome | Curated non-interacting pairs (negatives for PPI training) | TSV | http://mips.helmholtz-muenchen.de/proj/ppi/negatome/ |

### SARS-CoV-2 reference interactome (zero-shot demo target)
- **Gordon et al. 2020 (Nature)** — SARS-CoV-2 protein interaction map, 332 high-
  confidence human-viral interactions; cited by the topic as a zero-shot validation
  reference. A clean candidate held-out organism for the zero-shot claim.

## Downstream pathway / enrichment references

| Resource | What | URL |
|----------|------|-----|
| Reactome | Curated human pathways; enrichment analysis | https://reactome.org/ |
| KEGG | Pathways / molecular interaction maps | https://www.genome.jp/kegg/ |
| Gene Ontology Consortium | (same as ontology above) | https://geneontology.org/ |

## Data-prep checklist (annotation build)

- [ ] Download `go-basic.obo`; build ancestor map per sub-ontology.
- [ ] Pull Swiss-Prot sequences + GOA annotations for the target scope (viral + relevant host).
- [ ] Filter by evidence code; decide IEA inclusion.
- [ ] **Propagate annotations up the DAG (true-path rule).**
- [ ] Choose term set N (frequency + IA cutoff).
- [ ] Build binary label matrix.
- [ ] Cluster sequences for identity-based splits (see `03-evaluation-protocol.md`).
- [ ] Reserve a whole virus / viral family for zero-shot holdout.
