#!/usr/bin/env python3
"""Stage D data prep for the RECURSIVE rolling data (idea #5) — same assistant-only
loss mask as cat_dataset.py, but each example additionally masks everything BEFORE
`target_from_msg`, so the prior memory M_{i-1} and the raw activity being folded are
CONTEXT (masked) and only the emitted M_i + its continuation carry loss. This teaches
memory MERGING rather than re-teaching the prior memory.

Input: cat_instruct_v5_a3b.jsonl lines = {messages, target_from_msg, ...}.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cat_dataset import make_masker  # reuse the assistant-span masker  # noqa: E402


def build_example(messages, target_from_msg, tools, tok, mask_labels, max_len):
    # token offset where the TARGET (M_i onward) begins: length of the rendered context
    # return_dict=True + ["input_ids"] REQUIRED: bare tokenize=True returns a BatchEncoding
    # (len==2), not a flat id list — matches the working cat_dataset.py pattern.
    ctx_ids = tok.apply_chat_template(
        messages[:target_from_msg], tools=tools, tokenize=True, return_dict=True,
        add_generation_prompt=False, enable_thinking=False)["input_ids"]
    full_ids = tok.apply_chat_template(
        messages, tools=tools, tokenize=True, return_dict=True,
        add_generation_prompt=False, enable_thinking=False)["input_ids"]
    if len(full_ids) > max_len:
        return None
    # robust start: longest common prefix (guards against any template drift)
    n = min(len(ctx_ids), len(full_ids)); i = 0
    while i < n and ctx_ids[i] == full_ids[i]:
        i += 1
    target_start = i
    labels = mask_labels(full_ids)            # assistant tokens unmasked, rest -100
    for p in range(min(target_start, len(labels))):
        labels[p] = -100                      # mask everything before the target fold
    if not any(l != -100 for l in labels):
        return None
    return {"input_ids": full_ids, "labels": labels, "length": len(full_ids)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="cat_instruct_v5_a3b.jsonl")
    ap.add_argument("--tools", required=True, help="tools_v5.json")
    ap.add_argument("--tokenizer", default="/data/liqingyang/models/Qwen3.6-27B")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-len", type=int, default=65536)
    ap.add_argument("--num-proc", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from datasets import Dataset
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    mask_labels, _, _ = make_masker(tok)
    tools = json.load(open(args.tools))

    rows = []
    with open(args.data) as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            r = json.loads(line)
            rows.append({"messages": r["messages"],
                         "target_from_msg": r.get("target_from_msg", 0),
                         "trajectory_id": r.get("trajectory_id")})
    print(f"[load] {len(rows)} rolling examples")
    ds = Dataset.from_list(rows)

    def _map(ex):
        e = build_example(ex["messages"], ex["target_from_msg"], tools, tok,
                          mask_labels, args.max_len)
        if e is None:
            return {"input_ids": [], "labels": [], "length": 0, "ok": False}
        return {**e, "ok": True}

    ds = ds.map(_map, num_proc=args.num_proc, remove_columns=["messages"],
                desc="tokenize+mask(v5)")
    before = len(ds)
    ds = ds.filter(lambda e: e["ok"], num_proc=args.num_proc).remove_columns(["ok"])
    lens = ds["length"]
    print(f"[done] kept {len(ds)}/{before} (dropped over-len/empty: {before-len(ds)})")
    if lens:
        import statistics as st
        print(f"length tokens: median={int(st.median(lens))} "
              f"p95={sorted(lens)[int(len(lens)*0.95)]} max={max(lens)}")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(args.out_dir)
    print(f"saved -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
