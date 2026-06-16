"""CLI: annotate a pathogen proteome and characterize its dangerous effects.

The Stage-3 demo end-to-end (docs/07): fetch a target pathogen's proteins (UniProt,
unreviewed included), annotate them with the trained GO classifier for that domain
(`classifier.serving`), and map predictions onto the domain's curated danger ontology
(`threat`) to produce a per-organism threat profile + a side-by-side panel table.

    va-threat --panel                         # viral hemorrhagic-fever panel
    va-threat --domain bacterial --panel      # bacterial select-agent panel
    va-threat --taxon 2697049 --name sars2
    va-threat --domain bacterial --taxon 632 --name plague
    va-threat --fasta path/to/proteome.faa --name sample

HONEST SCOPE: the model holds out one family per domain (viral Coronaviridae; bacterial
Francisellaceae). Panel members from other families are IN-DISTRIBUTION — "annotate +
triage an unknown sample". The held-out family's target (SARS-CoV-2 / tularemia) is the
genuine novel-family zero-shot case. Predicted danger terms are triage hypotheses ranked
by confidence, not determinations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from viral_annotation.config import DEFAULT_DOMAIN, DOMAINS, REPO_ROOT, get_domain
from viral_annotation.data.danger_terms import danger_categories
from viral_annotation.data.proteomes import fetch_target, target_registry

RESULTS_DIR = REPO_ROOT / "results" / "threat"


def _load_records(args, registry):
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
    recs = fetch_target(key, reviewed=args.reviewed, limit=args.limit, registry=registry)
    name = args.name or (key if key in registry else f"taxon{key}")
    return name, recs


def _run_one(name, records, annotator, danger_map, threat_mod, standout, categories):
    annotated = annotator.annotate(records)
    pt = threat_mod.characterize_proteome(name, annotated, danger_map, annotator.dag,
                                          standout_threshold=standout, categories=categories)
    print(threat_mod.format_report(pt))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{name}.json"
    out.write_text(json.dumps(threat_mod.to_dict(pt), indent=2))
    print(f"\n[saved] {out}")
    return pt


def _panel_table(profiles, categories):
    """Side-by-side category x organism peak-confidence table."""
    lines = ["\n" + "=" * 70, "PANEL SUMMARY — peak confidence per danger category", "=" * 70]
    names = [pt.name for pt in profiles]
    lines.append(f"  {'category':42s}" + "".join(f"{n[:8]:>9s}" for n in names))
    peaks = {pt.name: pt.category_peaks() for pt in profiles}
    for cat in categories:
        row = f"  {cat.label:42s}" + "".join(f"{peaks[n][cat.key]:9.2f}" for n in names)
        lines.append(row)
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Annotate a pathogen and flag dangerous effects.")
    ap.add_argument("--domain", default=DEFAULT_DOMAIN, choices=list(DOMAINS),
                    help="pathogen domain profile (default viral)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--panel", action="store_true",
                   help="run the domain's full target panel")
    g.add_argument("--taxon", help="a UniProt taxonomy id to fetch")
    g.add_argument("--fasta", help="annotate a local proteome FASTA instead of fetching")
    ap.add_argument("--name", help="display/output name (a panel key or free text)")
    ap.add_argument("--reviewed", action="store_true",
                    help="restrict the fetch to Swiss-Prot (default: include TrEMBL)")
    ap.add_argument("--limit", type=int, default=None, help="cap proteins fetched")
    ap.add_argument("--standout", type=float, default=0.15,
                    help="min lift-over-background to call a protein a standout (default 0.15)")
    args = ap.parse_args(argv)

    dom = get_domain(args.domain)
    registry = target_registry(args.domain)
    categories = danger_categories(args.domain)

    # Import torch-dependent serving lazily so --help works without the [ml] extra.
    from viral_annotation import threat as threat_mod
    from viral_annotation.classifier.serving import GoAnnotator

    print(f"[load] rebuilding {args.domain} GO classifier from "
          f"{dom.models_dir / 'go_classifier.pt'} …", flush=True)
    annotator = GoAnnotator.load(models_dir=dom.models_dir)
    danger_map = threat_mod.build_danger_map(annotator.dag, categories)

    if args.panel:
        targets = list(registry)
    elif args.fasta or args.taxon or args.name:
        targets = [None]  # single run resolved in _load_records
    else:
        ap.error("specify --panel, --taxon, --fasta, or --name")

    profiles = []
    for t in targets:
        run_args = args if t is None else SimpleNamespace(**{**vars(args), "name": t,
                                                             "taxon": None, "fasta": None})
        name, records = _load_records(run_args, registry)
        profiles.append(_run_one(name, records, annotator, danger_map, threat_mod,
                                 args.standout, categories))

    if len(profiles) > 1:
        print(_panel_table(profiles, categories))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
