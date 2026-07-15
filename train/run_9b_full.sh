#!/bin/bash
# =============================================================================
# CaT-Instruct SFT — LoRA, multi-GPU DDP. Runs INSIDE the cat-train container
# (torch/transformers-5.x/peft/liger/fla present; base model mounted, e.g. /model9).
#
# Same recipe for BOTH Qwen3.5-9B and tmax-9B (identical qwen3_5 arch); only MODEL differs.
# Data = pre-tokenized {input_ids, labels, length} with assistant-only loss mask
#        (built by: merge folds -> stage_b4 stitch (fold ⊕ raw) -> cat_dataset.py).
#
# Env (override as needed):
#   DATA      tokenized dataset dir            [data/tokenized/cat_final]  (NEW pipeline output, <=65536)
#   MODEL     base model dir (in-container)    [/model9]   (Qwen3.5-9B; or mount tmax-9b-full)
#   OUT       adapter output dir               [ckpts/cat-9b-lora]
#   NPROC     GPUs / DDP ranks                 [8]
#   MAXLEN    extra length cap (0 = none)      [0]   ← big cards: 0 (keep full 65536).
#                                                     48GB-4090: set 45056 (cap-44k) to fit.
#   EPOCHS / LR / GRAD_ACCUM                   [3 / 1e-4 / 8]   (eff-batch = 1*GRAD_ACCUM*NPROC)
#   MAX_RETRY auto-resume retries after crash  [8]
#
# NOTE on "折断"/crashes: the winning 8x4090 run completed on attempt 1 (zero OOM). The
# retry loop + sft.py get_last_checkpoint() auto-resume are INSURANCE for the cap-44k
# memory edge (peak 98-99% of 48GB). Sequences are DROPPED if too long, never truncated,
# so there are no broken half-sequences. On larger cards with MAXLEN=0 this won't trigger.
# =============================================================================
set -o pipefail
DATA=${DATA:-data/tokenized/cat_final}
MODEL=${MODEL:-/model9}
OUT=${OUT:-ckpts/cat-9b-lora}
NPROC=${NPROC:-8}
MAXLEN=${MAXLEN:-0}
EPOCHS=${EPOCHS:-3}; LR=${LR:-1e-4}; GRAD_ACCUM=${GRAD_ACCUM:-8}
MAX_RETRY=${MAX_RETRY:-8}

# ---- optional extra length cap (data is already <=65536; only cap further on small cards) ----
DATA_USED="$DATA"
if [ "$MAXLEN" -gt 0 ]; then
  DATA_USED="${DATA}_cap${MAXLEN}"
  if [ ! -d "$DATA_USED" ]; then
    echo "[cap] $DATA -> $DATA_USED (<=$MAXLEN tok)"
    python3 - "$DATA" "$DATA_USED" "$MAXLEN" <<'PY'
import sys
from datasets import load_from_disk
src, out, m = sys.argv[1], sys.argv[2], int(sys.argv[3])
ds = load_from_disk(src); before = len(ds)
ds = ds.filter(lambda e: e["length"] <= m, num_proc=8)
print(f"[cap {m}] kept {len(ds)}/{before} ({100*len(ds)/before:.1f}%)", flush=True)
ds.save_to_disk(out)
PY
  fi
fi

echo "=== CaT SFT | model=$MODEL data=$DATA_USED out=$OUT nproc=$NPROC eff_batch=$((GRAD_ACCUM*NPROC)) ==="
attempt=0; rc=1
while [ $attempt -le $MAX_RETRY ]; do
  attempt=$((attempt+1))
  echo "=== [launch attempt $attempt] $(date) ==="
  torchrun --nproc_per_node "$NPROC" train/sft.py \
    --model "$MODEL" --data "$DATA_USED" --out "$OUT" --fused-ce \
    --epochs "$EPOCHS" --grad-accum "$GRAD_ACCUM" --lr "$LR" \
    --save-steps 50 --log-steps 5
  rc=$?
  echo "=== [attempt $attempt] torchrun exit=$rc ==="
  [ $rc -eq 0 ] && break
  echo "crash; sft.py auto-resumes from latest checkpoint in 30s..."; sleep 30
done
echo "EXIT=$rc  (adapter -> $OUT)"
