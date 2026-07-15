#!/usr/bin/env python3
"""Stage D data prep — tokenize CaT-Instruct / Base-Instruct conversations into
{input_ids, labels} with an ASSISTANT-ONLY loss mask, and cache to disk.

Loss design (validated against the data):
  * loss is computed ONLY on assistant-generated tokens: the Thought (content) and
    the Action (tool_calls), which INCLUDES the `context` call whose `summary`
    argument is the memory block M -> the model learns to generate M.
  * everything else is masked to -100: system prompt (Q), user/task, and EVERY tool
    observation (bash/editor outputs AND the context ack). The assistant role header
    `<|im_start|>assistant\\n` is masked; the closing `<|im_end|>` is kept so the model
    learns to stop.

The Qwen3.6 chat template has no `{% generation %}` tags, so we mask manually by
scanning `<|im_start|> role ... <|im_end|>` spans and unmasking only assistant spans.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from transformers import AutoTokenizer


def make_masker(tok):
    IM_START = tok.convert_tokens_to_ids("<|im_start|>")
    IM_END = tok.convert_tokens_to_ids("<|im_end|>")

    def mask_labels(ids: list[int]) -> list[int]:
        n = len(ids); labels = [-100] * n; i = 0
        while i < n:
            if ids[i] == IM_START:
                # read role text until the header newline
                j = i + 1; role = ""
                while j < n:
                    t = tok.decode([ids[j]]); j += 1
                    if "\n" in t:
                        role += t.split("\n")[0]; break
                    role += t
                k = j
                while k < n and ids[k] != IM_END:
                    k += 1
                if role.strip() == "assistant":
                    for p in range(j, min(k + 1, n)):   # content + tool_calls + im_end
                        labels[p] = ids[p]
                i = k + 1
            else:
                i += 1
        return labels

    return mask_labels, IM_START, IM_END


def build_example(messages, tools, tok, mask_labels, max_len, enable_thinking):
    out = tok.apply_chat_template(
        messages, tools=tools, tokenize=True, return_dict=True,
        add_generation_prompt=False, enable_thinking=enable_thinking,
    )
    ids = out["input_ids"]
    if len(ids) > max_len:
        return None  # drop over-length (truncating mid-conversation would corrupt structure)
    labels = mask_labels(ids)
    if not any(l != -100 for l in labels):
        return None  # no trainable tokens
    return {"input_ids": ids, "labels": labels, "length": len(ids)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="cat_instruct.jsonl or base_instruct.jsonl")
    ap.add_argument("--tools", required=True, help="tools.json (incl. context for CAT)")
    ap.add_argument("--tokenizer", default="/data/liqingyang/models/Qwen3.6-27B")
    ap.add_argument("--out-dir", required=True, help="HF dataset save_to_disk dir")
    ap.add_argument("--max-len", type=int, default=65536,
                    help="paper used 65536; keeps ~98%% of CaT-Instruct (median ~35k, p90 ~48k)")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="keep the <think> block (default off: agent does not emit empty think)")
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
            rows.append(json.loads(line))
    print(f"[load] {len(rows)} conversations")

    kept = []
    dropped_long = dropped_empty = 0
    # NOTE: tokenization is single-process here for simplicity/determinism; for the full
    # set, run with --num-proc via datasets.map below instead.
    ds = Dataset.from_list([{"messages": r["messages"], "trajectory_id": r.get("trajectory_id")} for r in rows])

    def _map(ex):
        e = build_example(ex["messages"], tools, tok, mask_labels, args.max_len, args.enable_thinking)
        if e is None:
            return {"input_ids": [], "labels": [], "length": 0, "ok": False}
        return {**e, "ok": True}

    ds = ds.map(_map, num_proc=args.num_proc, remove_columns=["messages"],
                desc="tokenize+mask")
    before = len(ds)
    ds = ds.filter(lambda e: e["ok"], num_proc=args.num_proc)
    ds = ds.remove_columns(["ok"])
    lens = ds["length"]
    print(f"[done] kept {len(ds)}/{before}  (dropped over-len/empty: {before-len(ds)})")
    if lens:
        import statistics as st
        print(f"length tokens: min={min(lens)} median={int(st.median(lens))} "
              f"p95={sorted(lens)[int(len(lens)*0.95)]} max={max(lens)}")
        avg_un = None
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(args.out_dir)
    print(f"saved -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
