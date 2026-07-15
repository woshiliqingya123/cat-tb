# CaT-TB — Compress-as-you-go Memory Folding for Long-Horizon Agents

Pipeline + training code for **CaT (Compress-as-you-go)**: at semantically-triggered
checkpoints in an agent trajectory, raw history is compressed into a structured memory
**M**, so the working context `C(t) = (Q, M(t), recent-k steps)` stays bounded and
**training matches inference**.

This repo holds the **data-generation pipeline** (turn raw agent trajectories into
CaT-Instruct) and the **SFT code** (LoRA fine-tune a base model to fold). Large artifacts
(data, checkpoints, models, logs) are git-ignored — see paths below.

---

## Pipeline (`catgen/`)

Raw trajectories → fold plans → memory M → sufficiency gate → stitched CaT-Instruct.

| Stage | Script | What it does |
|---|---|---|
| B1/2 | `stage_b12_fold_points.py` | Detect **domain-general** fold checkpoints from action *semantics* (expansion / error-recovery / first-action / verified / plan), not tool names. Emits compressible step ranges. |
| B3 | `stage_b3_generate_memory.py` (+`_parallel`) | Generate memory **M** for each fold (teacher summarizer, `enable_thinking=False`). |
| B3-v2 | `stage_b3v2_forward_memory.py` | **Forward-looking M** (7 sections: Goal / Attempts / Env / Constraints + Dead-Ends / Open-Threads / Next + Pinned). |
| B3-v3 | `stage_b3v3_sufficiency.py` | **Sufficiency gate (rejection sampling)**: counterfactual per-token ΔNLL of the future under folded-vs-raw context, scored by the *student*. Keeps M iff `ΔNLL ≤ τ`. Optional `--repair` closed loop (teacher rewrites, re-score). |
| B4-v5 | `stage_b45v5_recursive.py` | **Recursive rolling M** (`M_i = compress(M_{i-1} + new steps)`) — training==inference variant. |
| B4 | `stage_b4_stitch_cat_instruct.py` | **Stitch**: merge the accepted folds' M back into the raw trajectory (raw steps ⊕ `context(M)` calls) → one conversation per trajectory. |

**Sufficiency intuition:** `ΔNLL_per_tok = (NLL(future | Q,M,recent) − NLL(future | Q,raw,recent)) / |future tokens|`.
`≈0` means M preserved everything the model needs to predict the future → M is a
*sufficient statistic* of the past for the future. Calibrate the scorer to the **deployed**
model (weaker reader ⇒ conservative M).

---

## Training (`train/`)

| File | Role |
|---|---|
| `cat_dataset.py` | Tokenize stitched conversations → `{input_ids, labels, length}` with **assistant-only loss mask** (Thought + tool_calls incl. the `context(M)` call are trained; system/user/tool observations masked). |
| `sft.py` | LoRA SFT engine: `force_fla` (route gated-delta to fla), `--fused-ce` (liger fused-linear-CE for the 248k-vocab wall), gradient checkpointing, DDP, auto-resume. |
| `run_9b_full.sh` | Env-driven launcher (8-GPU DDP). Same recipe for **Qwen3.5-9B** and **tmax-9B** (both `qwen3_5` arch); only `MODEL` differs. |
| `ds_config_zero3*.json` | DeepSpeed ZeRO-3 configs (for full fine-tune / large-ctx variants). |
| `README.md` | Training details + gotchas. |

**Recipe:** LoRA r32/α64, lr 1e-4, cosine, 3 epochs, bf16, grad-ckpt, `--fused-ce`,
eff-batch ≈ 64 (`per_device 1 × grad_accum × nproc`). See `train/README.md`.

**Serving gotcha:** `qwen3_5` is a multimodal shell; `AutoModelForCausalLM` loads the text
view, so adapter keys are `model.layers.N` and **vLLM online-LoRA silently serves base**.
Deploy by **merge + reconstruct** the full multimodal checkpoint, not `--enable-lora`.

---

## Local artifact paths (git-ignored)

- `data/raw/tbase_pool_min50.jsonl` — raw agent trajectories (backbone for stitch).
- `data/fold_plans/` — fold points, M, sufficiency scores.
- `data/cat_instruct/` — stitched `cat_instruct_*.jsonl` + `tools_*.json`.
- `data/tokenized/` — HF datasets (`save_to_disk`) fed to `sft.py`.
- `ckpts/` — LoRA adapters.

## Environment

Runs in the `cat-train` docker image (torch / transformers 5.x / peft / datasets /
liger-kernel / flash-linear-attention). `transformers` must support the `qwen3_5` arch.
