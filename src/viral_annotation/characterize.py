"""CLI: annotate a virus proteome and characterize its dangerous effects.

This is the Stage-3 demo end-to-end (docs/README): fetch a target virus's proteins
(UniProt, unreviewed included), annotate them with the trained GO classifier
(`classifier.serving`), and map the predictions onto the curated danger ontology
(`threat`) to produce a per-virus threat profile + a side-by-side panel table.

    python -m viral_annotation.characterize --panel
    python -m viral_annotation.characterize --taxon 2697049 --name sars2
    python -m viral_annotation.characterize --fasta path/to/proteome.faa --name sample

HONEST SCOPE: only Coronaviridae is held out of training, so the hemorrhagic-fever
panel (Filo/Paramyxo/Arenaviridae) is IN-DISTRIBUTION — this simulates "annotate +
triage an unknown sample", not a novel-family zero-shot test. Predicted danger terms
are triage hypotheses ranked by confidence, not determinations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from viral_annotation.config import REPO_ROOT
from viral_annotation.data.danger_terms import DANGER_CATEGORIES
from viral_annotation.data.proteomes import TARGET_VIRUSES, fetch_target

RESULTS_DIR = REPO_ROOT / "results" / "threat"


def _load_records(args):
    """Return (display_name, list of records with .accession/.sequence/.organism)."""
    if args.fasta:
        from viral_annotation.data.fasta import read_fasta

        name = args.name or Path(args.fasta).stem
        recs = [SimpleNamespace(accession=r.id, sequence=r.sequence, organism=name)
                for r in read_fasta(args.fasta)]
        if not recs:
            raise SystemExit(f"no sequences in {args.fasta}")
        print(f"[target] {name}: {len(recs)} proteins from FASTA", flush=True)
        return name, recs

    key = args.taxon or args.name
    recs = fetch_target(key, reviewed=args.reviewed, limit=args.limit)
    name = args.name or (key if key in TARGET_VIRUSES else f"taxon{key}")
    return name, recs


def _run_one(name, records, annotator, danger_map, threat_mod, standout):
    annotated = annotator.annotate(records)
    pt = threat_mod.characterize_proteome(name, annotated, danger_map, annotator.dag,
                                          standout_threshold=standout)
    print(threat_mod.format_report(pt))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{name}.json"
    out.write_text(json.dumps(threat_mod.to_dict(pt), indent=2))
    print(f"\n[saved] {out}")
    return pt


def _panel_table(profiles):
    """Side-by-side category x virus peak-confidence table."""
    lines = ["\n" + "=" * 70, "PANEL SUMMARY — peak confidence per danger category", "=" * 70]
    names = [pt.name for pt in profiles]
    header = f"  {'category':42s}" + "".join(f"{n[:8]:>9s}" for n in names)
    lines.append(header)
    peaks = {pt.name: pt.category_peaks() for pt in profiles}
    for cat in DANGER_CATEGORIES:
        row = f"  {cat.label:42s}"
        for n in names:
            row += f"{peaks[n][cat.key]:9.2f}"
        lines.append(row)
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Annotate a virus and flag dangerous effects.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--panel", action="store_true",
                   help="run the full hemorrhagic-fever target panel")
    g.add_argument("--taxon", help="a UniProt taxonomy id to fetch")
    g.add_argument("--fasta", help="annotate a local proteome FASTA instead of fetching")
    ap.add_argument("--name", help="display/output name (key of TARGET_VIRUSES or free text)")
    ap.add_argument("--reviewed", action="store_true",
                    help="restrict the fetch to Swiss-Prot (default: include TrEMBL)")
    ap.add_argument("--limit", type=int, default=None, help="cap proteins fetched")
    ap.add_argument("--standout", type=float, default=0.15,
                    help="min lift-over-background to call a protein a standout (default 0.15)")
    args = ap.parse_args(argv)

    # Import torch-dependent serving lazily so --help works without the [ml] extra.
    from viral_annotation import threat as threat_mod
    from viral_annotation.classifier.serving import GoAnnotator

    print("[load] rebuilding GO classifier from models/go_classifier.pt …", flush=True)
    annotator = GoAnnotator.load()
    danger_map = threat_mod.build_danger_map(annotator.dag)

    if args.panel:
        targets = list(TARGET_VIRUSES)
    elif args.fasta or args.taxon or args.name:
        targets = [None]  # single run resolved in _load_records
    else:
        ap.error("specify --panel, --taxon, --fasta, or --name")

    profiles = []
    for t in targets:
        run_args = args if t is None else SimpleNamespace(**{**vars(args), "name": t,
                                                             "taxon": None, "fasta": None})
        name, records = _load_records(run_args)
        profiles.append(_run_one(name, records, annotator, danger_map, threat_mod,
                                 args.standout))

    if len(profiles) > 1:
        print(_panel_table(profiles))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
