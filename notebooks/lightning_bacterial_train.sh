#!/usr/bin/env bash
# Bacterial GO classifier — Lightning AI (SSH) training run.
#
# The SSH/tmux equivalent of notebooks/kaggle_bacterial_train.ipynb, for running the
# bacterial LoRA fine-tune on a Lightning AI Studio after exhausting Kaggle's free GPU
# hours. ESM-2 650M needs ~2.5 GB of weights and the run peaks ~14 GB VRAM, so a 16 GB
# T4 is enough; an L4/A10G (24 GB) is faster and still within the free monthly credits
# for a single run. Use a NON-interruptible machine so a preemption can't lose the run.
#
# Usage (run INSIDE tmux so it survives SSH disconnects):
#   export GITHUB_TOKEN=...   # fine-grained PAT, read-only Contents on jonmoses/SaltCitySoftware
#   export HF_TOKEN=...       # HuggingFace read token (unauth ESM-2 pulls get throttled)
#   bash lightning_bacterial_train.sh setup   # clone + deps + mmseqs + GO ontology  (once)
#   # --- then upload the LOCAL records (data/ is gitignored, so the clone lacks it) ---
#   #   the bacterial Swiss-Prot pull is already cached locally; ship it instead of
#   #   re-fetching from UniProt. From your LAPTOP (the .gz is ~73 MB vs 229 MB raw):
#   #     scp data/bacterial_reviewed.jsonl.gz <lightning-ssh-host>:SaltCitySoftware/data/
#   #   (or drag-drop the file into the Studio file browser, into the repo's data/ dir).
#   bash lightning_bacterial_train.sh records # verify/unzip the uploaded local records (once)
#   bash lightning_bacterial_train.sh diag    # ~10-15m capped run: read real it/s before committing
#   bash lightning_bacterial_train.sh train   # full run, logged to ~/bacterial_train_*.log
#
# After: models/bacterial/finetuned/{adapter,heads.pt,finetuned.meta.json} (~10 MB).
set -euo pipefail

REPO="jonmoses/SaltCitySoftware"
BRANCH="main"
WORKDIR="${WORKDIR:-$HOME/SaltCitySoftware}"
RECORDS="data/bacterial_reviewed.jsonl"

setup() {
  : "${GITHUB_TOKEN:?set GITHUB_TOKEN (read-only PAT) before running setup}"
  : "${HF_TOKEN:?set HF_TOKEN (HuggingFace read token) before running setup}"

  # 1) Clone the private repo (token is used only for the clone URL, not persisted).
  if [ ! -d "$WORKDIR/.git" ]; then
    git clone --depth 1 --branch "$BRANCH" \
      "https://x-access-token:${GITHUB_TOKEN}@github.com/${REPO}.git" "$WORKDIR"
  fi
  cd "$WORKDIR"
  git log --oneline -1

  # 2) Install the package. Lightning images ship CUDA-matched torch/transformers; the
  #    ml extra is a no-op if already present. peft + accelerate power the LoRA path.
  pip -q install -e ".[dev,bio,ml]"
  pip -q install biopython peft accelerate
  python - <<'PY'
import torch, peft
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
      "| peft", peft.__version__)
PY

  # 3) MMseqs2 static binary for the 30%-identity cluster split (AVX2; swap to
  #    mmseqs-linux-sse41.tar.gz on 'Illegal instruction'). Persist on PATH for later steps.
  if [ ! -x /tmp/mmseqs/bin/mmseqs ]; then
    wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -O /tmp/mmseqs.tar.gz
    tar -xzf /tmp/mmseqs.tar.gz -C /tmp
  fi
  export PATH="/tmp/mmseqs/bin:$PATH"
  mmseqs version

  # 4) GO ontology (go-basic.obo -> data/).
  va-download-go
  echo ">>> setup complete. Remember: export PATH=/tmp/mmseqs/bin:\$PATH in new shells."
}

# Use the LOCAL bacterial records (uploaded into the repo's data/ dir) instead of a
# fresh UniProt pull. Decompress the .gz if that's what was shipped, then verify.
records() {
  cd "$WORKDIR"
  if [ ! -f "$RECORDS" ] && [ -f "$RECORDS.gz" ]; then
    echo ">>> decompressing $RECORDS.gz"
    gunzip -k "$RECORDS.gz"
  fi
  if [ ! -f "$RECORDS" ]; then
    echo "ERROR: $RECORDS not found. Upload it from your laptop first, e.g.:" >&2
    echo "  scp data/bacterial_reviewed.jsonl.gz <lightning-ssh-host>:$WORKDIR/data/" >&2
    exit 1
  fi
  echo ">>> using local records: $RECORDS ($(wc -l < "$RECORDS") proteins)"
}

# Diagnostic: tiny capped fine-tune. Watch the 'fit:' line and per-step 'it/s | ETA' to
# settle whether the Kaggle 12 h timeout was a hang or just slow, BEFORE committing hours.
diag() {
  cd "$WORKDIR"; export PATH="/tmp/mmseqs/bin:$PATH"; export HF_TOKEN="${HF_TOKEN:?}"
  records
  python -u -m viral_annotation.cli.train --domain bacterial --finetune lora \
    --loss asl --pooling per-namespace --min-count 15 \
    --records "$RECORDS" --train-pool-cap 4000
}

# Full run. ~12.5k steps/epoch x 4 epochs; ~4-6 h on T4, faster on L4/A10G. Logged + teed.
# If diag showed T4 too slow: switch machine to L4/A10G, or lower --train-pool-cap.
train() {
  cd "$WORKDIR"; export PATH="/tmp/mmseqs/bin:$PATH"; export HF_TOKEN="${HF_TOKEN:?}"
  records
  local log="$HOME/bacterial_train_$(date +%Y%m%d_%H%M).log"
  echo ">>> logging to $log"
  python -u -m viral_annotation.cli.train --domain bacterial --finetune lora \
    --loss asl --pooling per-namespace --min-count 15 \
    --records "$RECORDS" --train-pool-cap 100000 2>&1 | tee "$log"
  echo ">>> artifacts:"; find models/bacterial/finetuned -type f 2>/dev/null
}

case "${1:-}" in
  setup)   setup ;;
  records) records ;;
  diag)    diag ;;
  train)   train ;;
  *) echo "usage: $0 {setup|records|diag|train}"; exit 2 ;;
esac
