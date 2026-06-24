"""One-click ESM-2 3B embedding of the viral + bacterial datasets (GPU box).

Computes windowed, mean-pooled per-protein embeddings with the 3B backbone
(`facebook/esm2_t36_3B_UR50D`) for every protein in the viral and bacterial
record dumps, and writes them to the repo's shared embedding cache:

    data/embeddings_cache/esm2_3B_mean_layer-last_win.npz

That is exactly the cache `va-train` (mean pooling) and the serving loader look
for, so once this finishes you can train/serve the 3B viral, bacterial, OR
unified models with no further embedding work. The cache is keyed by accession
and shared across domains, so both datasets go into the one file.

Why this config:
  * window=True, pooling="mean"  -> matches training (heads.py) and serving.
  * It does NOT build the per-residue attention cache (that would be hundreds of
    GB at 337K proteins). Train MF with `--pooling mean` to use this cache, or
    accept that the default attention-MF path recomputes residues live.

Robustness:
  * Embeddings are checkpointed every 1024 proteins (atomic temp-replace), so a
    crash or reboot loses at most one chunk — just re-run to resume.
  * First run downloads the 3B weights (~5.6 GB fp16) from Hugging Face.

Hardware: built for an RTX 4080 (16 GB). 3B inference fits comfortably under
fp16 autocast with the default 4096-token batch budget. Falls back to CPU with a
loud warning (don't — it would take days).

Run it:  python scripts/embed_3b.py        (or double-click scripts/embed_3b.bat)
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# --- Make the package importable whether or not it's pip-installed ----------
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MODEL_KEY = "3B"          # facebook/esm2_t36_3B_UR50D, dim 2560
POOLING = "mean"
WINDOW = True             # windowed pooling -> esm2_3B_mean_layer-last_win.npz

DATA_DIR = REPO_ROOT / "data"
# Each dataset ships in the repo gzipped (.jsonl.gz). An uncompressed .jsonl, if
# present, wins — but you do NOT need to gunzip anything; the .gz is read directly.
RECORD_STEMS = ["viral_reviewed", "bacterial_reviewed"]


def _check_deps() -> None:
    missing = []
    for mod in ("torch", "transformers", "numpy"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(
            "\nERROR: missing required packages: " + ", ".join(missing) + "\n\n"
            "Install them into your Python environment, e.g.:\n"
            "  pip install -e .[ml]        (from the repo root)\n"
            "or, for a CUDA-enabled PyTorch on Windows:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
            "  pip install transformers numpy\n",
            file=sys.stderr,
        )
        sys.exit(1)


def _report_device() -> None:
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Device: CUDA -> {name} ({vram:.0f} GB)")
    else:
        print(
            "\n*** WARNING: no CUDA GPU detected — this will run on CPU and take\n"
            "*** DAYS for 337K proteins. Stop now (Ctrl-C) unless you meant to.\n"
        )
        time.sleep(5)


def _resolve(stem: str) -> Path:
    """Find <stem>.jsonl (preferred) or <stem>.jsonl.gz under data/, or exit."""
    plain, gz = DATA_DIR / f"{stem}.jsonl", DATA_DIR / f"{stem}.jsonl.gz"
    if plain.exists():
        return plain
    if gz.exists():
        return gz
    print(
        f"\nERROR: neither {plain.name} nor {gz.name} found in {DATA_DIR}.\n"
        "Both datasets ship in the repo as .jsonl.gz — make sure you cloned the\n"
        "branch that contains them and that the data/ folder came along.\n",
        file=sys.stderr,
    )
    sys.exit(1)


def _load_records(path: Path) -> list[SimpleNamespace]:
    """Read accession+sequence from a (optionally gzipped) record JSONL."""
    opener = gzip.open if path.suffix == ".gz" else open
    out = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(SimpleNamespace(accession=d["accession"], sequence=d["sequence"]))
    print(f"  {path.name}: {len(out)} proteins")
    return out


def main() -> None:
    print("=" * 70)
    print("ESM-2 3B embedding (windowed mean pooling) — viral + bacterial")
    print("=" * 70)
    _check_deps()
    _report_device()

    from viral_annotation.embeddings.cache import cache_path, embed_records

    out_path = cache_path(MODEL_KEY, POOLING, None, window=WINDOW)
    print(f"Output cache: {out_path}")

    print("\nLoading records …")
    records: list[SimpleNamespace] = []
    seen: set[str] = set()
    for stem in RECORD_STEMS:
        for r in _load_records(_resolve(stem)):
            if r.accession not in seen:        # cache is keyed by accession; dedup
                seen.add(r.accession)
                records.append(r)
    print(f"  total unique proteins: {len(records)}")

    print("\nEmbedding (checkpoints every 1024 proteins; safe to re-run to resume) …")
    t0 = time.time()
    ids, X = embed_records(
        records, model_key=MODEL_KEY, pooling=POOLING, repr_layer=None, window=WINDOW
    )
    dt = time.time() - t0

    print("\n" + "=" * 70)
    print(f"DONE: {X.shape[0]} proteins x dim {X.shape[1]} in {dt / 60:.1f} min")
    print(f"Cache written: {out_path}")
    print("Next: va-train --domain bacterial --model 3B --pooling mean --records "
          "data/bacterial_reviewed.jsonl")
    print("=" * 70)


if __name__ == "__main__":
    main()
