# Training — CaT-Instruct LoRA SFT (Qwen3.5-9B / tmax-9B)

LoRA fine-tune, **multi-GPU pure DDP** (torchrun, no DeepSpeed). Both models are the same
`qwen3_5` dense arch, so one recipe — only `MODEL` differs.

## Files
- `cat_dataset.py` — tokenize stitched conversations → `{input_ids, labels, length}` with an
  **assistant-only loss mask** (Thought + tool_calls incl. the `context(M)` call are trained;
  system / user / tool observations masked). The Qwen chat template has no `{% generation %}`
  tags, so masking scans `<|im_start|> role … <|im_end|>` spans manually — use this, don't
  reimplement the mask.
- `sft.py` — SFT engine: `force_fla` (route gated-delta to fla), `--fused-ce` (liger
  fused-linear-CE for the 248320-vocab logits wall), gradient checkpointing, DDP, auto-resume.
- `run_9b_full.sh` — env-driven launcher (`MODEL / DATA / OUT / NPROC / MAXLEN / GRAD_ACCUM`).

## Run
```bash
# Step 1 — tokenize (per model's own tokenizer)
python3 cat_dataset.py --data <cat_instruct.jsonl> --tools <tools.json> \
  --tokenizer /path/to/Qwen3.5-9B --out-dir data/tokenized/cat_final --max-len 65536

# Step 2 — train (inside the cat-train container, 8 GPUs)
MODEL=/path/to/Qwen3.5-9B DATA=data/tokenized/cat_final OUT=ckpts/qwen35-9b-cat \
  NPROC=8 MAXLEN=0 GRAD_ACCUM=8  bash run_9b_full.sh
```

## Recipe (validated)
LoRA r32 / α64 / dropout0.05, target = q,k,v,o,gate,up,down · lr 1e-4 cosine, warmup 0.1,
wd 0.01 · 3 epochs · per-device 1, **eff-batch ≈ 64** = `1 × GRAD_ACCUM × NPROC` · bf16 ·
gradient checkpointing · **`--fused-ce` (required)** · attn sdpa.

**Hardware knobs:** big cards (80G) → `MAXLEN=0` (full 65536, drop nothing); 48G-4090 →
`MAXLEN=45056` (cap-44k, else the longest sequence OOMs). Match eff-batch 64 by adjusting
`GRAD_ACCUM` for your GPU count (8→8, 4→16).

## Gotchas
1. **`--fused-ce` required** — without liger fused-linear-CE the `[seq × 248320]` logits alone
   are ~32 GB → OOM.
2. **fla** — `force_fla()` forces gated-delta onto the fla fast kernel (transformers gates it on
   a package name that doesn't match the wheel). Log line `[fla] forced availability=True` = OK.
3. **Long sequences are dropped, never truncated.** Peak memory is set by the single longest
   sequence; if it OOMs, cap lower (~38k) — small drops from 48k barely help. Crashes
   auto-resume from the latest checkpoint (`run_9b_full.sh` retry loop).
4. **Serving:** `qwen3_5` is a multimodal shell; `AutoModelForCausalLM` loads the text view, so
   adapter keys are `model.layers.N` and **vLLM `--enable-lora` silently serves the base model**.
   Deploy by **merge + reconstruct** the full multimodal checkpoint and serve `--model /merged`.

## Expected
Trainable ≈ 58M (0.65%) · train_loss ≈ 0.28 → 0.18 over 3 epochs · smoke (2 steps) shows
`[fla]`, `[fused-ce]`, `trainable params: 58,195,968`, `train_loss ~0.2x`, `saved ->`, exit 0.
