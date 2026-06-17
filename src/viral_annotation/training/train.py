"""Train the viral GO classifier — one config-driven path (docs/01).

Per-namespace **evidence policy** (config.NAMESPACE_POLICY) is always applied: a
full-set experiment showed IEA training poisons Molecular Function in viruses
(manual-MF is curated protein binding, IEA-MF is domain-rule ligand binding —
nearly disjoint, MF Fmax 0.09 under IEA training), so MF trains manual-only while
BP/CC train on manual+IEA. Val/test always score on manual-only labels.

Pooling and the optional homology ensemble are selectable, so the documented
experiments reproduce from this one entry point:
  * --pooling mean      (default; the servable model — best single-pooling)
  * --pooling stats     (mean|max|min|std, 4d features)
  * --pooling attention (learned per-residue pooling — wins zero-shot MF; not servable)
  * --pooling per-namespace  (NAMESPACE_POLICY's choice: attention MF, mean BP/CC)
  * --ensemble homology (late-fuse a BLAST-KNN component — closes the zero-shot gap)

Pipeline: load proteins -> cluster split (+ family holdout) -> per namespace
{select vocab, fit head, hierarchically correct, Fmax vs Naive} -> test + zero-shot
report -> save (pooled models only). See memory: iea-manual-mf-distribution-shift.

Domain: `--domain {viral,bacterial}` selects a config.PathogenDomain profile (taxon,
family-holdout rank, evidence/pooling policy, models dir). Defaults resolve from the
profile, so the viral path is unchanged. See docs/08-bacterial-extension.md.

Run:  python -m viral_annotation.cli.train [--domain D] [--pooling P] [--ensemble homology] ...
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

from viral_annotation.config import (
    DEFAULT_DOMAIN,
    ESM2_MODELS,
    FT_BACKBONE_LR,
    FT_BATCH_SIZE,
    FT_DROPOUT,
    FT_EPOCHS,
    FT_GRAD_ACCUM,
    FT_HEAD_LR,
    FT_HIDDEN_DIMS,
    FT_MAX_LENGTH,
    GO_NAMESPACES,
    GO_OBO_PATH,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_LR,
    TRAIN_SEED,
    get_domain,
)
from viral_annotation.data.dataset import build_labels
from viral_annotation.embeddings.residue_cache import cache_residues, residue_cache_dir
from viral_annotation.evaluation import report
from viral_annotation.evaluation.metrics import apply_hierarchical_correction, fmax_matrix
from viral_annotation.ontology import GoDag
from viral_annotation.training import pipeline
from viral_annotation.training.heads import fit_namespace

# Attention-pooling defaults (only used when --pooling attention/per-namespace).
ATTN_EPOCHS = 100
ATTN_BATCH = 16
ATTN_HEADS = 8
MAX_RESIDUES = 2048

POOLING_CHOICES = ("mean", "stats", "attention", "per-namespace")

# Sentinel: "caller didn't specify, fall back to the domain profile's value".
_USE_DOMAIN = object()


def _pooling_per_namespace(pooling: str, policy: dict) -> dict[str, str]:
    """Resolve the --pooling choice to a concrete pooling per namespace."""
    if pooling == "per-namespace":
        return {ns: policy[ns]["pooling"] for ns in GO_NAMESPACES}
    return {ns: pooling for ns in GO_NAMESPACES}


def run(limit=None, domain=DEFAULT_DOMAIN, model_key=None, pooling=None, ensemble=None,
        min_count=None, hidden_dims=None, epochs=TRAIN_EPOCHS, lr=TRAIN_LR,
        batch_size=TRAIN_BATCH_SIZE, holdout_family=_USE_DOMAIN, use_cluster=True,
        save=True, records_path=None, finetune="none", loss="bce",
        max_length=FT_MAX_LENGTH, grad_accum=FT_GRAD_ACCUM, train_pool_cap=None):
    import numpy as np
    import torch

    # Resolve unspecified knobs from the pathogen-domain profile (viral by default).
    dom = get_domain(domain)
    policy = dom.namespace_policy
    model_key = model_key or dom.default_esm_model
    pooling = pooling or dom.default_pooling
    min_count = dom.min_term_count if min_count is None else min_count
    holdout_family = dom.holdout_family if holdout_family is _USE_DOMAIN else holdout_family

    t0 = time.time()
    torch.manual_seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    device = pipeline.auto_device(torch)
    hp = SimpleNamespace(model_key=model_key, min_count=min_count, hidden=hidden_dims,
                         epochs=epochs, attn_epochs=ATTN_EPOCHS, lr=lr, batch=batch_size,
                         attn_batch=ATTN_BATCH, heads=ATTN_HEADS, max_residues=MAX_RESIDUES,
                         train_pool_cap=train_pool_cap, input_dim=ESM2_MODELS[model_key].dim,
                         loss=loss, max_length=max_length, ft_batch=FT_BATCH_SIZE,
                         ft_grad_accum=grad_accum, ft_epochs=FT_EPOCHS,
                         ft_hidden=list(FT_HIDDEN_DIMS), ft_dropout=FT_DROPOUT,
                         ft_backbone_lr=FT_BACKBONE_LR, ft_head_lr=FT_HEAD_LR)
    pooling_by_ns = _pooling_per_namespace(pooling, policy)

    print(f"[1/5] domain={domain} | device={device} | pooling={pooling_by_ns} | "
          f"finetune={finetune} | loss={loss} | ensemble={ensemble or 'none'} | loading GO DAG …")
    dag = GoDag.from_obo(GO_OBO_PATH)

    src = f"cached records {records_path}" if records_path else f"{domain} reviewed proteins"
    print(f"[2/5] loading {src} (limit={limit}) …")
    proteins = pipeline.load_proteins(dag, limit, query=dom.uniprot_query,
                                      records_path=records_path)
    print(f"       {pipeline.annotation_stats(proteins)}")

    split = pipeline.make_split(proteins, use_cluster=use_cluster, holdout_family=holdout_family,
                                family_suffixes=dom.family_suffixes)
    train_prots = split.train
    if finetune == "lora" and train_pool_cap:
        from viral_annotation.training.heads import cap_pool
        train_prots = cap_pool(split.train, train_pool_cap, TRAIN_SEED)
        print(f"       fine-tune train pool capped {len(split.train)} -> {len(train_prots)}")
    pools = {"all": train_prots, "manual_having": [p for p in train_prots if p.has_manual]}
    print(f"[3/5] {'cluster' if use_cluster else 'random'} split: {split.summary()} "
          f"| manual-having train {len(pools['manual_having'])}")
    if not split.val or not split.test:
        raise SystemExit("val/test empty — too few manual-having proteins. Increase --limit.")

    # Per-residue cache is only needed for the FROZEN attention path. The LoRA
    # fine-tune path computes residues live and back-propagates, so no cache.
    cache_dir = None
    if "attention" in pooling_by_ns.values() and finetune != "lora":
        print("[4/5] caching per-residue embeddings for attention namespaces …")
        attn_pools = {policy[ns]["train_pool"] for ns in GO_NAMESPACES
                      if pooling_by_ns[ns] == "attention"}
        groups = [pools[p] for p in attn_pools] + [split.val, split.test]
        groups += [split.holdout] if split.holdout else []
        for g in groups:
            cache_residues(g, model_key, None)
        cache_dir = residue_cache_dir(model_key, None)
    else:
        print("[4/5] (pooled features — no per-residue cache needed)")

    homology_db = pools["manual_having"]   # homology transfers these proteins' manual labels
    # The LoRA path trains one shared-backbone multi-task model up front, then wraps
    # each namespace as a Head so the eval/report path below is reused unchanged.
    ft_heads, ft_artifacts = {}, None
    if finetune == "lora":
        from viral_annotation.training.finetune import run_finetune
        print("[5/5] fine-tuning shared LoRA backbone + per-namespace heads …")
        ft_heads, ft_artifacts = run_finetune(split, pools, policy, pooling_by_ns,
                                              dag, device, hp)
    else:
        print(f"[5/5] training per-namespace heads{' + homology ensemble' if ensemble else ''} …")
    heads: dict[str, object] = {}
    # Accumulated [P x N_ns] blocks for the across-namespace overall metric.
    test = {"model": [], "true": [], "naive": [], "plm": []}
    zero = {"model": [], "true": [], "naive": [], "plm": []}
    test_rows, zero_rows = [], []

    for ns in GO_NAMESPACES:
        head = (ft_heads.get(ns) if finetune == "lora"
                else fit_namespace(ns, policy[ns], pooling_by_ns[ns], split, pools,
                                   dag, device, cache_dir, hp))
        if head is None:
            print(f"       {report.NS_SHORT[ns]}: empty vocab — skipped")
            continue
        vocab = head.vocab
        heads[ns] = head

        weights = _fit_ensemble_weights(head, split, homology_db, dag, ensemble)

        def scored(prots):
            """(pLM prob, final model prob) for a protein set, hierarchically corrected."""
            plm = apply_hierarchical_correction(head.predict(prots), vocab, dag)
            if ensemble == "homology":
                from viral_annotation.classifier.ensemble import fuse
                from viral_annotation.data.homology import homology_scores
                fused = fuse({"plm": plm, "homology": homology_scores(prots, homology_db, dag, vocab)},
                             weights)
                return plm, apply_hierarchical_correction(fused, vocab, dag)
            return plm, plm

        zline = ""
        _eval_split(ns, head, split.test, scored, dag, test, test_rows, zero_shot=False)
        if split.holdout:
            zline = _eval_split(ns, head, split.holdout, scored, dag, zero, zero_rows, zero_shot=True)
        print(f"       {report.NS_SHORT[ns]} [{head.pooling}]: N={len(vocab):4d} "
              f"test {test_rows[-1][1].fmax:.4f} (naive {test_rows[-1][2].fmax:.4f}){zline} "
              f"val={head.val_fmax:.3f} ep={head.epochs}", flush=True)

    _print_reports(test, test_rows, zero, zero_rows, ensemble, holdout_family, split)
    print(f"\n[done] elapsed {time.time() - t0:.1f}s")

    if save:
        if finetune == "lora":
            if ft_artifacts is not None:
                from viral_annotation.training.finetune import save_finetuned
                out = save_finetuned(ft_artifacts, dom.models_dir / "finetuned", test_rows)
                print(f"[saved] {out} (LoRA adapter + heads + meta.json)")
        else:
            _save(heads, pooling, model_key, hidden_dims, test, test_rows, dom.models_dir, policy)
    return heads


def _fit_ensemble_weights(head, split, homology_db, dag, ensemble):
    """Grid-search per-namespace fusion weights on validation (None if no ensemble)."""
    if ensemble != "homology":
        return None
    from viral_annotation.classifier.ensemble import search_weights
    from viral_annotation.data.homology import homology_scores

    vocab = head.vocab
    val_comps = {
        "plm": apply_hierarchical_correction(head.predict(split.val), vocab, dag),
        "homology": homology_scores(split.val, homology_db, dag, vocab),
    }
    weights, _ = search_weights(val_comps, build_labels(split.val, vocab, "terms_manual"))
    return weights


def _eval_split(ns, head, prots, scored, dag, acc, rows, *, zero_shot) -> str:
    """Score `prots`, append to the accumulators + report rows, and return a
    zero-shot log fragment (empty for the test split)."""
    import numpy as np

    Y = build_labels(prots, head.vocab, "terms_manual")
    plm, model = scored(prots)
    naive = np.tile(head.prior, (Y.shape[0], 1))
    res, naive_res = fmax_matrix(model, Y), fmax_matrix(naive, Y)
    acc["model"].append(model); acc["true"].append(Y)
    acc["naive"].append(naive); acc["plm"].append(plm)
    rows.append((ns, res, naive_res))
    return f" | zero-shot {res.fmax:.4f} (naive {naive_res.fmax:.4f})" if zero_shot else ""


def _print_reports(test, test_rows, zero, zero_rows, ensemble, holdout_family, split):
    overall = report.overall_fmax(test["model"], test["true"])
    overall_naive = report.overall_fmax(test["naive"], test["true"])
    report.print_table("TEST (manual-only labels, hierarchically corrected)",
                       test_rows, overall, overall_naive)
    if ensemble:
        plm_overall = report.overall_fmax(test["plm"], test["true"])
        print(f"  (pLM-only overall {plm_overall.fmax:.4f}; ensemble lift "
              f"{overall.fmax - plm_overall.fmax:+.4f})")

    if split.holdout and zero_rows:
        z_overall = report.overall_fmax(zero["model"], zero["true"])
        z_naive = report.overall_fmax(zero["naive"], zero["true"])
        report.print_table(f"ZERO-SHOT — held-out {holdout_family} "
                           f"({len(split.holdout)} proteins, never trained on)",
                           zero_rows, z_overall, z_naive)
        if ensemble:
            z_plm = report.overall_fmax(zero["plm"], zero["true"])
            print(f"  (pLM-only overall {z_plm.fmax:.4f}; ensemble lift "
                  f"{z_overall.fmax - z_plm.fmax:+.4f})")


def _save(heads, pooling, model_key, hidden_dims, test, test_rows, models_dir, policy):
    """Persist pooled heads (state_dict + meta) under the domain's models dir.
    Attention heads aren't servable by the lightweight loader, so skip saving when
    any head used attention pooling."""
    import torch

    if any(h.state is None for h in heads.values()):
        print("[save] skipped — attention heads aren't servable; use --pooling mean "
              "for a deployable model.")
        return

    overall = report.overall_fmax(test["model"], test["true"])
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save({ns: heads[ns].state for ns in heads}, models_dir / "go_classifier.pt")
    naive_by_ns = {ns: nv for ns, _res, nv in test_rows}
    res_by_ns = {ns: res for ns, res, _nv in test_rows}
    meta = {
        "esm_model": model_key,
        "pooling": pooling,                 # uniform across heads when saved
        "hidden_dims": hidden_dims or [],
        "overall_fmax": overall.fmax,
        "namespaces": {
            ns: {
                "policy": policy[ns],
                "fmax": res_by_ns[ns].fmax,
                "naive_fmax": naive_by_ns[ns].fmax,
                "terms": heads[ns].vocab.terms,
            }
            for ns in heads
        },
    }
    (models_dir / "go_classifier.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[saved] {models_dir / 'go_classifier.pt'} (+ meta.json)")
