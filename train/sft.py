#!/usr/bin/env python3
"""Stage D — SFT for CAT-Model / Base-Model (SAME recipe; only --data differs).

Trains Qwen3.6-27B on pre-tokenized {input_ids, labels} (see cat_dataset.py), with
loss already restricted to assistant tokens (labels=-100 elsewhere). Supports LoRA
(default, to validate the pipeline cheaply) and full fine-tuning (--full, via
DeepSpeed ZeRO-3).

Fairness: run this once on CaT-Instruct -> CAT-Model and once on Base-Instruct ->
Base-Model with identical flags. Only the dataset changes.
"""
from __future__ import annotations
import argparse, sys, os
from dataclasses import dataclass

import torch
from datasets import load_from_disk
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)


def force_fla():
    """transformers gates the gated-delta-rule fast path on is_flash_linear_attention_available(),
    which checks `metadata.version("fla")` — but the fla wheel's DISTRIBUTION name is
    `flash-linear-attention`, so that lookup raises PackageNotFoundError and the check returns
    False even when fla is installed. Result: gated-delta silently runs the slow torch fallback.
    Here we (1) force the availability flag True and (2) rebind the modeling module's globals to
    fla's kernels, so layers constructed during from_pretrained use fla. (causal_conv1d is a
    separate, small fallback and is left as-is.)"""
    try:
        import fla  # noqa: F401
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    except Exception as e:
        print(f"[fla] not importable, staying on torch fallback: {e}", flush=True)
        return
    import sys
    import transformers.utils.import_utils as iu
    iu.is_flash_linear_attention_available = lambda: True
    try:
        from fla.modules import FusedRMSNormGated
    except Exception:
        FusedRMSNormGated = None
    # rebind globals in EVERY already-imported qwen3_5* modeling module (dense `qwen3_5`
    # AND moe `qwen3_5_moe` use separate modules with their own globals)
    patched = []
    for name, mod in list(sys.modules.items()):
        if "modeling_qwen3_5" in name and mod is not None and hasattr(mod, "torch_chunk_gated_delta_rule"):
            mod.chunk_gated_delta_rule = chunk_gated_delta_rule
            mod.fused_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule
            if FusedRMSNormGated is not None:
                mod.FusedRMSNormGated = FusedRMSNormGated
            patched.append(name)
    print(f"[fla] forced availability=True; rebound gated-delta to fla in: {patched or '(none yet; will bind on import)'}", flush=True)


def patch_gdr_chunk(size):
    """Force a smaller chunk_size in the gated-delta-rule linear attention. The within-chunk
    work is O(S * chunk_size); shrinking it lowers the transient high-water mark in
    forward/backward. Lossless (no data dropped, no extra stored activation); cost is a few %
    slower linear attention (more chunks). The call site hardcodes chunk_size=64, so we wrap
    the module function to override it."""
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as mod
    orig = mod.torch_chunk_gated_delta_rule
    def wrapped(*a, chunk_size=64, **kw):
        return orig(*a, chunk_size=size, **kw)
    mod.torch_chunk_gated_delta_rule = wrapped
    print(f"[gdr] forced gated-delta-rule chunk_size -> {size}", flush=True)


def apply_fused_ce(model):
    """Fuse ONLY the LM cross-entropy via liger (kills the 248k-vocab logits wall), and
    explicitly DISABLE liger's MoE/swiglu fusion — that expert-MLP kernel pre-allocates a
    large dgate_up_proj buffer in backward that was the last ~1GB pushing us OOM. With
    swiglu off, the MoE experts run native (under gradient checkpointing, no big buffer).
    The CE fusion runs inside the model's own forward, so it stays correct under ZeRO-3
    (deepspeed gathers the lm_head during forward)."""
    import inspect
    from liger_kernel.transformers import monkey_patch as mp
    mt = model.config.model_type  # 'qwen3_5' (dense 9B) or 'qwen3_5_moe' (35B-A3B)
    fn = getattr(mp, f"apply_liger_kernel_to_{mt}", None)
    if fn is None:
        from liger_kernel.transformers import _apply_liger_kernel_to_instance
        _apply_liger_kernel_to_instance(model=model)
        print(f"[fused-ce] generic liger patch for model_type={mt}", flush=True)
        return
    kw = dict(model=model, fused_linear_cross_entropy=True, cross_entropy=False,
              rms_norm=False, rope=False)
    if "swiglu" in inspect.signature(fn).parameters:
        # disable expert-MLP fusion ONLY for MoE (its backward buffer OOMs); harmless/faster on dense
        kw["swiglu"] = ("moe" not in mt)
    fn(**kw)
    print(f"[fused-ce] liger fused-CE on model_type={mt} (swiglu={kw.get('swiglu')})", flush=True)


@dataclass
class PadCollator:
    pad_id: int
    def __call__(self, feats):
        maxlen = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            ids = f["input_ids"]; lab = f["labels"]; pad = maxlen - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/liqingyang/models/Qwen3.6-27B")
    ap.add_argument("--data", required=True, help="tokenized dataset dir (save_to_disk)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--full", action="store_true", help="full fine-tune (needs deepspeed); else LoRA")
    ap.add_argument("--qlora", action="store_true",
                    help="4-bit QLoRA: fits 27B on ONE 48GB card (~14GB). Single-GPU; no deepspeed.")
    ap.add_argument("--deepspeed", default=None,
                    help="deepspeed config json. Use for --full, or for bf16 LoRA on multi-GPU/64k.")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"],
                    help="attention impl. Default sdpa (built-in; safe for the hybrid arch / no flash-attn).")
    ap.add_argument("--gdr-chunk", type=int, default=0,
                    help="override gated-delta-rule chunk_size (default 64). Smaller = lower "
                         "transient memory (O(S*chunk_size)), lossless, slightly slower. e.g. 16.")
    ap.add_argument("--fused-ce", action="store_true",
                    help="fused-linear cross-entropy (liger): never materializes the full "
                         "[seq, 248320-vocab] logits -> required to fit long ctx on 48GB. "
                         "Bypasses the model's own LM head/loss (and MoE aux loss, which is "
                         "frozen under LoRA anyway).")
    # hyperparams: paper defaults (Qwen2.5-Coder-32B); VERIFY for Qwen3.6
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=None, help="default 1e-5 full / 1e-4 LoRA")
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--log-steps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="cap steps for a fail-fast OOM/feasibility probe (e.g. 3)")
    ap.add_argument("--local_rank", type=int, default=-1,
                    help="injected by the deepspeed launcher; Trainer reads rank from env")
    args = ap.parse_args()

    lr = args.lr if args.lr is not None else (1e-5 if args.full else 1e-4)
    force_fla()  # make transformers actually route gated-delta to fla (before model build)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # IMPORTANT: build TrainingArguments (with the deepspeed config) BEFORE loading the
    # model. For ZeRO-3 this activates the zero.Init context so from_pretrained shards
    # the 27B weights across ranks instead of materializing the full 54GB on every GPU
    # (which would OOM at load time).
    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=True,
        # non-reentrant (preserves RNG -> lora_dropout reproduced on recompute), but skip the
        # strict metadata-equality check that misfires on MoE dynamic expert dispatch.
        # Safe here: routing is deterministic (no jitter / dropout 0 / dropless).
        gradient_checkpointing_kwargs={"use_reentrant": False, "determinism_check": "none"},
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        deepspeed=(None if args.qlora else args.deepspeed),  # zero3 not compatible with 4-bit
        dataloader_num_workers=4,
        # LoRA has no truly-unused params; the extra autograd traversal just wastes time/mem.
        ddp_find_unused_parameters=False,
    )

    ds = load_from_disk(args.data)
    print(f"[data] {len(ds)} examples")

    quant_config = None
    model_kwargs = {}
    if args.qlora:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            # keep the LM head in bf16 so fused-linear CE can use its weight directly
            llm_int8_skip_modules=["lm_head"] if args.fused_ce else None)
        # place the (small, 4-bit) weights on this process's GPU
        model_kwargs["device_map"] = {"": int(__import__("os").environ.get("LOCAL_RANK", 0))}

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation=args.attn, trust_remote_code=True,
        quantization_config=quant_config, **model_kwargs,
    )
    model.config.use_cache = False  # set on config (transformers 5.x rejects use_cache= in from_pretrained for this arch); required for gradient checkpointing
    if args.gdr_chunk > 0:
        patch_gdr_chunk(args.gdr_chunk)
    if args.fused_ce:
        apply_fused_ce(model)
    if args.qlora:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()

    if not args.full:
        from peft import LoraConfig, get_peft_model
        lcfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()

    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=PadCollator(pad_id))
    # auto-resume: if the out dir already holds a checkpoint (e.g. after an OOM restart),
    # pick up from the latest one instead of restarting from step 0.
    resume = None
    if os.path.isdir(args.out):
        from transformers.trainer_utils import get_last_checkpoint
        resume = get_last_checkpoint(args.out)
        if resume:
            print(f"[resume] continuing from {resume}", flush=True)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
