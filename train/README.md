# Stage D — SFT (CAT-Model & Base-Model)

Same recipe for both models; only `--data` differs. This is the controlled comparison:
identical base model, hyperparams, and data distribution — the ONLY variable is whether
the training data contains the `context` folding behavior.

## Env (separate from eval `cat` env)
```bash
conda create -y -n cattrain python=3.11 && conda activate cattrain
pip install torch transformers datasets accelerate peft deepspeed bitsandbytes
# flash-attn NOT required: default --attn sdpa (built-in; safe for the hybrid arch).
# Install flash-attn only if you explicitly pass --attn flash_attention_2.
```

## Step 1 — tokenize + assistant-only loss mask (run once per dataset)
```bash
# CAT-Instruct (has the context tool + M folds)
python train/cat_dataset.py \
  --data data/cat_instruct/cat_instruct.jsonl \
  --tools data/cat_instruct/tools.json \
  --tokenizer /data/liqingyang/models/Qwen3.6-27B \
  --out-dir data/tokenized/cat --max-len 65536

# Base-Instruct (plain ReAct, no context tool) — same flags, base tools.json
python train/cat_dataset.py \
  --data data/base_instruct/base_instruct.jsonl \
  --tools data/base_instruct/tools.json \
  --out-dir data/tokenized/base --max-len 65536
```
Loss is on assistant tokens only (Thought + Action + M); system/user/all tool
observations (incl. the context ack) are masked to -100. Over-length convs are dropped
(report printed).

## Step 2 — train
**Pipeline validation — QLoRA on ONE 48GB card (27B in 4-bit ~14GB, no deepspeed):**
```bash
# tokenize a small subset at a small ctx so it fits one card for the smoke
python train/cat_dataset.py --data data/cat_instruct/cat_instruct.jsonl \
  --tools data/cat_instruct/tools.json --out-dir data/tokenized/cat_24k --max-len 24576
CUDA_VISIBLE_DEVICES=0 python train/sft.py --qlora \
  --data data/tokenized/cat_24k --out ckpts/cat-qlora --max-steps 50
```
(plain `python train/sft.py` WITHOUT --qlora/--deepspeed loads 27B bf16 onto one card →
OOM; always use --qlora for single-GPU or --deepspeed for multi-GPU.)

**bf16 LoRA @ 64k on 8 GPUs (ZeRO-3 shards the frozen base) — fallback if full-FT OOMs:**
```bash
deepspeed --num_gpus 8 train/sft.py --deepspeed train/ds_config_zero3.json \
  --data data/tokenized/cat --out ckpts/cat-lora
```

**Full fine-tune (paper-faithful), 27B @ 64k via DeepSpeed ZeRO-3 + CPU offload.**
On 8×48GB this is the hard case — ZeRO-3 shards/offloads params+optimizer, but ZeRO-3
does NOT shard activations, so 64k activations may OOM. Two prerequisites:
  * CPU RAM ≳ ~380GB (optimizer offload: 27B AdamW fp32 states are huge);
  * activations at 64k fit per-GPU (the risk). Probe FAIL-FAST before committing.

Staged attempt (fail-fast):
```bash
# (0) tiny tokenized subset for the probe
python train/cat_dataset.py --data data/cat_instruct/cat_instruct.jsonl \
  --tools data/cat_instruct/tools.json --out-dir data/tokenized/cat_probe \
  --max-len 65536 --limit 64

# (1) PROBE: does full-FT init (zero3 load + optimizer offload) and survive a few
#     64k steps without OOM?  (3 steps, ~minutes; OOM shows up immediately)
deepspeed --num_gpus 8 train/sft.py --full --deepspeed train/ds_config_zero3.json \
  --data data/tokenized/cat_probe --out /tmp/probe --max-steps 3

# (2) if the probe survives -> full run on the full tokenized data
deepspeed --num_gpus 8 train/sft.py --full --deepspeed train/ds_config_zero3.json \
  --data data/tokenized/cat --out ckpts/cat-full
deepspeed --num_gpus 8 train/sft.py --full --deepspeed train/ds_config_zero3.json \
  --data data/tokenized/base --out ckpts/base-full
```
If the probe OOMs at 64k: first retry the probe at a smaller ctx (re-tokenize with
`--max-len 24576`) to confirm full-FT *machinery* works, then fall back to LoRA@64k
(sequence parallelism) for the real run.

**LoRA fallback (if full-FT @ 64k OOMs):** see plan A — LoRA via a sequence-parallel
framework (LLaMA-Factory).

## Hyperparams (paper defaults — ⚠️VERIFY for Qwen3.6)
3 epochs, AdamW wd 0.01, cosine, warmup 0.1. LR: full 1e-5 (paper 5e-5), LoRA 1e-4.
ctx 65536 (matches paper; keeps ~98% of data, median ~35k / p90 ~48k tokens). 27B @ 64k
is memory-heavy — see GPU feasibility note in chat; may require sequence parallelism on
48GB cards or A100/H100-80GB.

## Notes
- `--enable-thinking` is OFF by default: we strip the empty `<think></think>` so the
  agent doesn't learn to emit it each turn (M / reasoning live in content + tool_calls).
- Qwen3.6 renders tool calls as `<tool_call><function=...><parameter=...>` (NOT JSON);
  inference (Stage E/F via OpenHands) must use the same template/parser — VERIFY there.
- `context` tool: defined in data/cat_instruct/tools.json; the CAT-Model learns to call
  it and emit M as its `summary` arg. Base-Model never sees it.
