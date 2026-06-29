"""Head-to-head benchmark of two ESM-2 backbones (e.g. 650M vs 3B).

Answers "how much do I really gain from the bigger embedding?" with **matched
controls** and **bootstrap confidence intervals**, across two protocols:

  * cluster   — the leakage-safe 30%-identity cluster split + held-out-family
                zero-shot (the repo's gold standard); viral AND bacterial.
  * temporal  — the NetGO/CAFA no-knowledge temporal split; viral only (QuickGO is
                taxon 10239). Reports Naive / BLAST-KNN / LR-ESM / Ensemble.

The split and the GO vocabulary depend only on sequences, not on embeddings, so
they are computed ONCE per (domain, protocol) and shared by both models — only the
feature matrix differs. The head is the linear LR-ESM head (`fit_pooled_head`,
`hidden_dims=None`), matching `va-train --pooling mean`. Each (model, namespace) is
fit over several seeds (head init + minibatch order) to expose optimization noise;
a paired protein-bootstrap then puts a 95% CI on the per-metric delta.

Everything runs off the cached embeddings + cached records — no GPU, no UniProt
fetch. Each model's cache `.npz` is loaded into memory once and reused.

Run via:  va-compare-embeddings  (see cli/compare.py)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from viral_annotation.config import (
    DATA_DIR,
    GO_NAMESPACES,
    GO_OBO_PATH,
    REPO_ROOT,
    TRAIN_BATCH_SIZE,
    TRAIN_EARLY_STOP_PATIENCE,
    TRAIN_EPOCHS,
    TRAIN_LR,
    VIRAL_RECORDS_PATH,
    get_domain,
)
from viral_annotation.classifier.model import predict_proba
from viral_annotation.data import labels as labels_mod
from viral_annotation.data import quickgo
from viral_annotation.data.dataset import build_labels, select_vocab
from viral_annotation.embeddings.cache import cache_path
from viral_annotation.evaluation.metrics import (
    apply_hierarchical_correction,
    fmax_matrix,
    information_accretion,
    m_aupr,
    paired_bootstrap_fmax_smin,
    smin,
)
from viral_annotation.ontology import GoDag
from viral_annotation.training.heads import compute_pos_weight, fit_pooled_head
from viral_annotation.training.pipeline import auto_device, load_proteins, make_split

# Deterministic seed pool; --seeds N takes the first N. Capturing both head-init and
# minibatch-shuffle randomness, with the split held fixed.
SEED_POOL = (1337, 7, 42, 101, 2024, 13, 271, 99)
DEFAULT_SEEDS = {"viral": 5, "bacterial": 3}   # bacterial fits are big — fewer seeds
NS_ABBR = {"molecular_function": "MFO", "biological_process": "BPO", "cellular_component": "CCO"}
RESULTS_DIR = REPO_ROOT / "results"

_EMB_CACHE: dict[str, dict] = {}   # model_key -> {accession: vector}, loaded once per run


# --------------------------------------------------------------------------- #
# embedding access (cache hit only — no live embedding)                        #
# --------------------------------------------------------------------------- #
def _load_emb(model_key: str) -> dict:
    """Load a model's windowed mean-pooled cache into {accession: vector}, once."""
    import numpy as np

    if model_key in _EMB_CACHE:
        return _EMB_CACHE[model_key]
    path = cache_path(model_key, "mean", None, window=True)
    if not path.exists():
        raise SystemExit(
            f"missing embedding cache {path} — build it (scripts/embed_{model_key.lower()}.py "
            "or va-train) before comparing.")
    data = np.load(path, allow_pickle=False)
    mat = data["embeddings"]
    emb = {acc: mat[i] for i, acc in enumerate(data["ids"].tolist())}
    _EMB_CACHE[model_key] = emb
    return emb


def _free_emb(model_key: str) -> None:
    """Drop a model's cache from RAM (the .npz reloads on demand if needed again).

    Holding both the 650M (~1.8GB) and 3B (~3.6GB) caches at once peaks past what a
    16GB box tolerates during the bacterial summarise — so free each after its fits.
    """
    import gc

    _EMB_CACHE.pop(model_key, None)
    gc.collect()


def _lookup(prots, emb: dict):
    """Stack cached vectors for `prots` (errors loudly if any are uncached)."""
    import numpy as np

    missing = [p.accession for p in prots if p.accession not in emb]
    if missing:
        raise SystemExit(f"{len(missing)} proteins not in cache (e.g. {missing[:3]}) — "
                         "the cache is incomplete; rebuild it before comparing.")
    return np.stack([emb[p.accession] for p in prots]).astype("float32")


def _seed(s: int) -> None:
    import numpy as np
    import torch

    torch.manual_seed(s)
    np.random.seed(s)


def _resolve_seeds(domain: str, seeds: int | None) -> tuple[int, ...]:
    n = DEFAULT_SEEDS.get(domain, 3) if seeds is None else seeds
    return SEED_POOL[:n]


def _records_path(domain: str) -> Path:
    return VIRAL_RECORDS_PATH if domain == "viral" else DATA_DIR / f"{domain}_reviewed.jsonl"


# --------------------------------------------------------------------------- #
# metric summarisation                                                        #
# --------------------------------------------------------------------------- #
def _seed_stats(probs_seeds, Y, ia, terms):
    """Per-seed Fmax/M-AUPR/Smin -> mean/std; plus the seed-averaged prob matrix."""
    import numpy as np

    fm, ma, sm = [], [], []
    for p in probs_seeds:
        fm.append(fmax_matrix(p, Y).fmax)
        ma.append(m_aupr(p, Y))
        sm.append(smin(p, Y, ia, terms))
    avg = np.mean(np.stack(probs_seeds), axis=0)
    return {
        "fmax_mean": float(np.mean(fm)), "fmax_std": float(np.std(fm)),
        "m_aupr_mean": float(np.mean(ma)), "m_aupr_std": float(np.std(ma)),
        "smin_mean": float(np.mean(sm)), "smin_std": float(np.std(sm)),
        "fmax_seeds": [float(x) for x in fm],
    }, avg


def _delta_ci(avg_a, avg_b, Y, ia, terms, n_boot):
    """Paired bootstrap deltas (b - a) for Fmax and Smin on seed-averaged probs.

    Uses the precompute-once fast bootstrap (`paired_bootstrap_fmax_smin`) so even
    n_boot=1000 on the wide bacterial matrices is seconds, not tens of minutes.
    (M-AUPR is term-centric — bootstrapping proteins for it is non-standard and the
    per-column sklearn AP is slow — so it gets seed-variance only, not a CI.)
    """
    return paired_bootstrap_fmax_smin(avg_a, avg_b, Y, ia, terms, n_boot=n_boot)


def _naive(prior, Y, ia, terms):
    import numpy as np

    naive = np.tile(prior, (Y.shape[0], 1))
    return {"fmax": float(fmax_matrix(naive, Y).fmax), "m_aupr": float(m_aupr(naive, Y)),
            "smin": float(smin(naive, Y, ia, terms))}


# --------------------------------------------------------------------------- #
# cluster protocol (viral + bacterial)                                        #
# --------------------------------------------------------------------------- #
def _run_cluster(domain, models, seeds, dag, device, n_boot):
    dom = get_domain(domain)
    print(f"  [cluster/{domain}] loading {_records_path(domain).name} …", flush=True)
    proteins = load_proteins(dag, query=dom.uniprot_query, records_path=str(_records_path(domain)))
    split = make_split(proteins, use_cluster=True, holdout_family=dom.holdout_family,
                       family_suffixes=dom.family_suffixes)
    pools = {"all": split.train, "manual_having": [p for p in split.train if p.has_manual]}
    print(f"  [cluster/{domain}] {split.summary()}", flush=True)

    # Per-namespace setup is model-independent: vocab, labels, IA, prior computed once.
    setup = {}
    for ns in GO_NAMESPACES:
        policy = dom.namespace_policy[ns]
        train_prots = pools[policy["train_pool"]]
        vocab = select_vocab(train_prots, dag, dom.min_term_count,
                             field=policy["vocab_field"], namespaces=[ns])
        if len(vocab) == 0:
            continue
        setup[ns] = dict(
            policy=policy, train_prots=train_prots, vocab=vocab,
            Ytr=build_labels(train_prots, vocab, policy["train_field"]),
            Yva=build_labels(split.val, vocab, "terms_manual"),
            Yte=build_labels(split.test, vocab, "terms_manual"),
            Yzs=build_labels(split.holdout, vocab, "terms_manual") if split.holdout else None,
            ia=information_accretion([p.terms_manual for p in train_prots], dag),
        )

    # Fit each model (cache loaded once), collecting seed-corrected prob matrices.
    probs = {ns: {m: {"test": [], "zs": []} for m in models} for ns in setup}
    for model in models:
        emb = _load_emb(model)
        for ns, S in setup.items():
            Xtr, Xva = _lookup(S["train_prots"], emb), _lookup(split.val, emb)
            Xte = _lookup(split.test, emb)
            Xzs = _lookup(split.holdout, emb) if split.holdout else None
            pos_weight = compute_pos_weight(S["Ytr"])
            for s in seeds:
                _seed(s)
                mdl, _, _ = fit_pooled_head(
                    Xtr, S["Ytr"], Xva, S["Yva"], hidden_dims=None, epochs=TRAIN_EPOCHS,
                    lr=TRAIN_LR, batch_size=TRAIN_BATCH_SIZE, device=device,
                    patience=TRAIN_EARLY_STOP_PATIENCE, pos_weight=pos_weight)
                probs[ns][model]["test"].append(
                    apply_hierarchical_correction(predict_proba(mdl, Xte), S["vocab"], dag))
                if Xzs is not None:
                    probs[ns][model]["zs"].append(
                        apply_hierarchical_correction(predict_proba(mdl, Xzs), S["vocab"], dag))
            print(f"  [cluster/{domain}] {NS_ABBR[ns]} {model}: N={len(S['vocab'])} "
                  f"fit {len(seeds)} seeds", flush=True)
        _free_emb(model)   # cap peak RAM at one cache; none resident during summarise

    return _summarise_cluster(domain, models, setup, split, probs, n_boot)


def _summarise_cluster(domain, models, setup, split, probs, n_boot):
    import numpy as np

    out = {"protocol": "cluster", "domain": domain, "split": split.summary(),
           "models": list(models), "namespaces": {}, "overall": {}}
    # accumulate seed-averaged probs / Y across namespaces for the overall metric.
    acc = {sp: {m: [] for m in models} for sp in ("test", "zs")}
    accY = {sp: [] for sp in ("test", "zs")}
    acc_terms, acc_ia = [], {}
    acc_prior = []

    for ns, S in setup.items():
        ns_out = {"N": len(S["vocab"]), "test": {}, "zero_shot": {}}
        for sp, key, Y in (("test", "test", S["Yte"]), ("zero_shot", "zs", S["Yzs"])):
            if Y is None or Y.shape[0] == 0:
                continue
            avgs = {}
            for m in models:
                stats, avg = _seed_stats(probs[ns][m][key], Y, S["ia"], S["vocab"].terms)
                ns_out[sp][m] = stats
                avgs[m] = avg
            ns_out[sp]["naive"] = _naive(S["Ytr"].mean(axis=0), Y, S["ia"], S["vocab"].terms)
            if len(models) == 2:
                ns_out[sp]["delta"] = _delta_ci(avgs[models[0]], avgs[models[1]], Y,
                                                S["ia"], S["vocab"].terms, n_boot)
            # stash for overall (only the test/zs splits that exist for every ns)
            for m in models:
                acc[key][m].append(avgs[m])
        out["namespaces"][ns] = ns_out
        # overall accumulators (test always present; zs present iff holdout)
        accY["test"].append(S["Yte"])
        if S["Yzs"] is not None:
            accY["zs"].append(S["Yzs"])
        acc_terms.extend(S["vocab"].terms)
        acc_ia.update(S["ia"])
        acc_prior.append(S["Ytr"].mean(axis=0))

    # overall = concatenate columns across namespaces (same proteins per split).
    for sp, key in (("test", "test"), ("zero_shot", "zs")):
        if not accY[key] or any(len(acc[key][m]) != len(accY[key]) for m in models):
            continue
        Y = np.concatenate(accY[key], axis=1)
        avgs = {m: np.concatenate(acc[key][m], axis=1) for m in models}
        prior = np.concatenate(acc_prior)
        ov = {}
        for m in models:
            ov[m] = {"fmax": float(fmax_matrix(avgs[m], Y).fmax),
                     "m_aupr": float(m_aupr(avgs[m], Y)),
                     "smin": float(smin(avgs[m], Y, acc_ia, acc_terms))}
        ov["naive"] = _naive(prior, Y, acc_ia, acc_terms)
        if len(models) == 2:
            ov["delta"] = _delta_ci(avgs[models[0]], avgs[models[1]], Y, acc_ia,
                                    acc_terms, n_boot)
        out["overall"][sp] = ov
    return out


# --------------------------------------------------------------------------- #
# temporal protocol (viral only)                                              #
# --------------------------------------------------------------------------- #
def _run_temporal(models, seeds, dag, device, n_boot, cutoff, min_count=3):
    from viral_annotation.classifier.ensemble import fuse, search_weights
    from viral_annotation.data.homology import homology_scores
    from viral_annotation.benchmark.temporal import build_temporal_split
    import numpy as np

    print("  [temporal/viral] QuickGO experimental annotations (cached) + sequences …", flush=True)
    ann = quickgo.fetch_or_load()
    proteins = labels_mod.label_proteins(list(labels_mod.load_raw(VIRAL_RECORDS_PATH)), dag)
    seq_by_acc = {p.accession: p.sequence for p in proteins if p.sequence}
    split = build_temporal_split(ann, seq_by_acc, dag, cutoff)
    print(f"  [temporal/viral] cutoff {cutoff}: {split.summary()}", flush=True)

    out = {"protocol": "temporal", "domain": "viral", "cutoff": cutoff,
           "models": list(models), "namespaces": {}}
    # Pre-resolve per-namespace setup (model-independent).
    setup = {}
    for ns in GO_NAMESPACES:
        train_ns, test_ns = split.train[ns], split.test[ns]
        if not train_ns or not test_ns:
            continue
        vocab = select_vocab(train_ns, dag, min_count, field="terms_manual", namespaces=[ns])
        if len(vocab) == 0:
            continue
        idx = np.random.RandomState(1337).permutation(len(train_ns))
        n_val = max(1, int(0.15 * len(train_ns)))
        val_ns = [train_ns[i] for i in idx[:n_val]]
        tr_ns = [train_ns[i] for i in idx[n_val:]]
        setup[ns] = dict(vocab=vocab, tr=tr_ns, val=val_ns, test=test_ns,
                         Ytr=build_labels(tr_ns, vocab, "terms_manual"),
                         Yva=build_labels(val_ns, vocab, "terms_manual"),
                         Yte=build_labels(test_ns, vocab, "terms_manual"),
                         ia=information_accretion([p.terms_manual for p in train_ns], dag))

    # BLAST-KNN homology is model-independent — compute once per namespace.
    hom = {}
    for ns, S in setup.items():
        hom[ns] = {"val": homology_scores(S["val"], S["tr"], dag, S["vocab"]),
                   "test": homology_scores(S["test"], S["tr"], dag, S["vocab"])}

    lr_probs = {ns: {m: [] for m in models} for ns in setup}    # corrected test probs/seed
    ens_probs = {ns: {m: [] for m in models} for ns in setup}
    for model in models:
        emb = _load_emb(model)
        for ns, S in setup.items():
            Xtr, Xva, Xte = _lookup(S["tr"], emb), _lookup(S["val"], emb), _lookup(S["test"], emb)
            pos_weight = compute_pos_weight(S["Ytr"])
            for s in seeds:
                _seed(s)
                mdl, _, _ = fit_pooled_head(
                    Xtr, S["Ytr"], Xva, S["Yva"], hidden_dims=None, epochs=TRAIN_EPOCHS,
                    lr=TRAIN_LR, batch_size=TRAIN_BATCH_SIZE, device=device,
                    patience=TRAIN_EARLY_STOP_PATIENCE, pos_weight=pos_weight)
                lr_te = predict_proba(mdl, Xte)
                w, _ = search_weights({"plm": predict_proba(mdl, Xva), "homology": hom[ns]["val"]},
                                      S["Yva"])
                ens_te = fuse({"plm": lr_te, "homology": hom[ns]["test"]}, w)
                lr_probs[ns][model].append(apply_hierarchical_correction(lr_te, S["vocab"], dag))
                ens_probs[ns][model].append(apply_hierarchical_correction(ens_te, S["vocab"], dag))
            print(f"  [temporal/viral] {NS_ABBR[ns]} {model}: N={len(S['vocab'])} "
                  f"test={len(S['test'])} fit {len(seeds)} seeds", flush=True)
        _free_emb(model)

    for ns, S in setup.items():
        Y, ia, terms = S["Yte"], S["ia"], S["vocab"].terms
        ns_out = {"N": len(S["vocab"]), "test": len(S["test"]),
                  "Naive": _naive(S["Ytr"].mean(axis=0), Y, ia, terms),
                  "BLAST-KNN": {"fmax": float(fmax_matrix(
                      apply_hierarchical_correction(hom[ns]["test"], S["vocab"], dag), Y).fmax)},
                  "LR-ESM": {}, "Ensemble": {}}
        for label, store in (("LR-ESM", lr_probs), ("Ensemble", ens_probs)):
            avgs = {}
            for m in models:
                stats, avg = _seed_stats(store[ns][m], Y, ia, terms)
                ns_out[label][m] = stats
                avgs[m] = avg
            if len(models) == 2:
                ns_out[label]["delta"] = _delta_ci(avgs[models[0]], avgs[models[1]], Y,
                                                   ia, terms, n_boot)
        out["namespaces"][ns] = ns_out
    return out


# --------------------------------------------------------------------------- #
# orchestration + reporting                                                   #
# --------------------------------------------------------------------------- #
def run_comparison(domains=("viral", "bacterial"), protocols=("cluster", "temporal"),
                   models=("650M", "3B"), seeds=None, n_boot=1000, cutoff=20240101,
                   out_dir=RESULTS_DIR):
    import torch

    t0 = time.time()
    device = auto_device(torch)
    print(f"[compare] device={device} | models={list(models)} | protocols={list(protocols)} "
          f"| domains={list(domains)} | n_boot={n_boot}")
    print("[compare] loading GO DAG …")
    dag = GoDag.from_obo(GO_OBO_PATH)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "embedding_comparison.json"

    results = []
    for domain in domains:
        for protocol in protocols:
            if protocol == "temporal" and domain != "viral":
                continue   # QuickGO experimental corpus is viral (taxon 10239)
            sds = _resolve_seeds(domain, seeds)
            if protocol == "cluster":
                r = _run_cluster(domain, models, sds, dag, device, n_boot)
            else:
                r = _run_temporal(models, sds, dag, device, n_boot, cutoff)
            results.append(r)
            # print + persist after each protocol so a kill never loses finished work.
            (_print_cluster if r["protocol"] == "cluster" else _print_temporal)(r)
            out_path.write_text(json.dumps({"models": list(models), "n_boot": n_boot,
                                            "results": results}, indent=2))
            print(f"\n[compare] wrote {out_path} ({len(results)} protocol(s) so far)", flush=True)

    print(f"[done] elapsed {time.time() - t0:.1f}s")
    return results


def _fmt_cell(stats):
    return f"{stats['fmax_mean']:.4f}±{stats['fmax_std']:.3f}"


def _fmt_delta(d):
    sig = "*" if (d["lo"] > 0 or d["hi"] < 0) else " "
    return f"{d['delta']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]{sig}"


def _print_cluster(r):
    models = r["models"]
    print(f"\n=== CLUSTER — {r['domain']} (Fmax mean±std over seeds; Δ=3B−650M w/ 95% bootstrap CI, "
          f"* excludes 0) ===")
    print(f"    {r['split']}")
    for sp_key, sp_name in (("test", "TEST"), ("zero_shot", "ZERO-SHOT")):
        rows = []
        for ns, nd in r["namespaces"].items():
            if sp_name == "TEST" and "test" in nd and nd["test"]:
                rows.append((NS_ABBR[ns], nd["test"], nd["N"]))
            elif sp_name == "ZERO-SHOT" and nd.get("zero_shot"):
                rows.append((NS_ABBR[ns], nd["zero_shot"], nd["N"]))
        ov = r["overall"].get(sp_key)
        if not rows and not ov:
            continue
        print(f"\n  -- {sp_name} (manual-only, hierarchically corrected) --")
        hdr = "  " + f"{'ns':6s}" + "".join(f"{m:>16s}" for m in models) + \
              f"{'naive':>10s}" + f"{'Δ Fmax [95% CI]':>30s}"
        print(hdr)
        for abbr, cell, n in rows:
            line = f"  {abbr:6s}" + "".join(f"{_fmt_cell(cell[m]):>16s}" for m in models)
            line += f"{cell['naive']['fmax']:>10.4f}"
            if "delta" in cell:
                line += f"{_fmt_delta(cell['delta']['fmax']):>30s}"
            print(line)
        if ov:
            line = f"  {'OVR':6s}" + "".join(f"{ov[m]['fmax']:>16.4f}" for m in models)
            line += f"{ov['naive']['fmax']:>10.4f}"
            if "delta" in ov:
                line += f"{_fmt_delta(ov['delta']['fmax']):>30s}"
            print(line)


def _print_temporal(r):
    models = r["models"]
    print(f"\n=== TEMPORAL CAFA — viral (cutoff {r['cutoff']}; Fmax mean±std; "
          f"Δ=3B−650M w/ 95% CI) ===")
    for ns, nd in r["namespaces"].items():
        print(f"\n  -- {NS_ABBR[ns]} (N={nd['N']} test={nd['test']}) --")
        print(f"  {'method':10s}" + "".join(f"{m:>16s}" for m in models) +
              f"{'Δ Fmax [95% CI]':>30s}")
        print(f"  {'Naive':10s}{nd['Naive']['fmax']:>16.4f}")
        print(f"  {'BLAST-KNN':10s}{nd['BLAST-KNN']['fmax']:>16.4f}")
        for label in ("LR-ESM", "Ensemble"):
            cell = nd[label]
            line = f"  {label:10s}" + "".join(f"{_fmt_cell(cell[m]):>16s}" for m in models)
            if "delta" in cell:
                line += f"{_fmt_delta(cell['delta']['fmax']):>30s}"
            print(line)
