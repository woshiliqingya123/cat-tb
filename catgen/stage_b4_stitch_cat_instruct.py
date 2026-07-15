#!/usr/bin/env python3
"""Stage B4 — trajectory stitching: turn (fold plan + M) into CaT-Instruct.

CAT-GENERATOR Phase II step (4). Takes B3's output (fold points with generated
memory blocks M) plus the original trajectories, and produces the final SFT data:
complete OpenHands-format conversations in which, at each fold point,

  * the agent emits a brief Thought + a call to the `context` tool whose `summary`
    argument IS the memory block M  (so M lands on the ASSISTANT turn and is a
    learnable target, not a loss-masked observation), followed by
  * a short tool acknowledgement, and
  * the compressed raw steps are PHYSICALLY REMOVED from the sequence.

The result is the "compressed-state" trajectory the model must reproduce at
inference: history is replaced by M, so training == inference.

Outputs:
  data/cat_instruct/cat_instruct.jsonl   one conversation per line {messages, ...}
  data/cat_instruct/tools.json           the OpenHands tool set + the `context` tool
"""
from __future__ import annotations
import argparse, json, sys, uuid
from pathlib import Path
# Make load_steps importable whether stage_b12_fold_points.py sits next to this
# script (e.g. catgen/b4/) or in the parent catgen/ dir.
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from stage_b12_fold_points import load_steps

CONTEXT_TOOL = {
    "type": "function",
    "function": {
        "name": "context",
        "description": (
            "Condense the earlier conversation history into a structured long-term "
            "memory, REPLACING the raw history so the working context stays compact. "
            "Invoke at natural milestones: a subtask is complete, you have recovered "
            "from repeated failures, or the history has grown large and a concise "
            "summary now serves reasoning better than verbose logs. Provide the full "
            "memory block as `summary`; the earlier raw steps are then removed and "
            "`summary` becomes your long-term memory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": (
                        "The structured long-term memory, with sections [Goal] / "
                        "[Attempts & Results] / [Environment Feedback] / [Active "
                        "Constraints]. Preserve exact paths, line numbers, symbols, "
                        "commands, errors, test results, code edits, and failed "
                        "attempts; drop reconstructible or redundant logs."
                    ),
                }
            },
            "required": ["summary"],
        },
    },
}

# Domain-NEUTRAL fold-trigger phrases (no tests/code/implementation wording) so the learned
# fold behavior isn't tied to SWE surface cues and can transfer to other agent domains.
THOUGHT = {
    "structural:plan": "I'm at a planning milestone, so I'll consolidate the earlier history into long-term memory.",
    "structural:verified": "I've confirmed this subtask is on track, so I'll consolidate the work so far into long-term memory before continuing.",
    "structural:first_action": "I've finished gathering context and am moving to act, so I'll consolidate what I've learned into long-term memory.",
    "error_recovery": "I've recovered from the earlier failures and found a workable direction, so I'll consolidate that history into long-term memory.",
    "expansion": "The context has grown large, so I'll consolidate the earlier history into long-term memory to stay focused.",
    # backward-compat aliases (old signal names) -> neutral phrasing
    "structural:testpass": "I've confirmed this subtask is on track, so I'll consolidate the work so far into long-term memory before continuing.",
    "structural:first_write": "I've finished gathering context and am moving to act, so I'll consolidate what I've learned into long-term memory.",
}
DEFAULT_THOUGHT = "I'll consolidate the earlier history into long-term memory to keep my working context compact."
ACK = ("Context condensed. The summary has been saved as your long-term memory and the "
       "corresponding earlier raw history has been removed from the working context.")


def context_messages(M: str, signal: str) -> list[dict]:
    tcid = "call_ctx_" + uuid.uuid4().hex[:24]
    assistant = {
        "role": "assistant",
        "content": THOUGHT.get(signal, DEFAULT_THOUGHT),
        "tool_calls": [{
            "id": tcid, "type": "function",
            "function": {"name": "context", "arguments": {"summary": M}},
        }],
    }
    tool = {"role": "tool", "name": "context", "tool_call_id": tcid, "content": ACK}
    return [assistant, tool]


def stitch(head, steps, folds) -> tuple[list[dict], int]:
    """folds: list of dicts with compressible_steps [lo,hi], M, signal."""
    blocks = {}
    for fp in folds:
        cs = fp.get("compressible_steps")
        if not cs or not fp.get("M"):
            continue
        lo, hi = cs
        if 0 <= lo <= hi < len(steps):
            blocks[lo] = (hi, fp["M"], fp.get("signal", ""))
    msgs = list(head)
    i, n, used = 0, len(steps), 0
    while i < n:
        if i in blocks:
            hi, M, sig = blocks[i]
            msgs.extend(context_messages(M, sig))
            used += 1
            i = hi + 1
        else:
            msgs.extend(steps[i]["raw"])
            i += 1
    return msgs, used


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-m", default="data/cat_instruct/fold_plans_with_M.jsonl")
    ap.add_argument("--pool", default="data/raw/tbase_pool_min50.jsonl")
    ap.add_argument("--out", default="data/cat_instruct/cat_instruct.jsonl")
    ap.add_argument("--tools-in", default=str(Path(__file__).parent / "tools.json"),
                    help="original OpenHands tool schemas (the 5 base tools)")
    ap.add_argument("--tools-out", default="data/cat_instruct/tools.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # augmented tool set = original OpenHands tools + context
    base_tools = json.load(open(args.tools_in))
    # tools.json may be a bare list of {function:...}; normalize to {type,function}
    norm = []
    for t in base_tools:
        norm.append(t if "type" in t else {"type": "function", "function": t["function"] if "function" in t else t})
    aug = norm + [CONTEXT_TOOL]
    Path(args.tools_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(aug, open(args.tools_out, "w"), ensure_ascii=False, indent=2)
    print(f"[tools] wrote {len(aug)} tools (incl. context) -> {args.tools_out}")

    print(f"[load] indexing pool {args.pool}")
    pool = {}
    with open(args.pool) as f:
        for line in f:
            r = json.loads(line)
            pool[r["trajectory_id"]] = r["trajectory"]
    print(f"[load] {len(pool)} trajectories")

    out_p = Path(args.out)
    out_f = out_p.open("w")
    n = n_folds = 0
    msgs_before = msgs_after = 0
    with open(args.with_m) as f:
        for line in f:
            if args.limit and n >= args.limit:
                break
            rec = json.loads(line)
            traj = pool.get(rec["trajectory_id"])
            if traj is None:
                continue
            head, steps = load_steps(traj)
            msgs, used = stitch(head, steps, rec["fold_points"])
            if used == 0:
                continue
            n += 1; n_folds += used
            msgs_before += len(traj); msgs_after += len(msgs)
            out_f.write(json.dumps({
                "trajectory_id": rec["trajectory_id"],
                "instance_id": rec.get("instance_id"),
                "num_folds": used,
                "messages": msgs,
            }, ensure_ascii=False) + "\n")
    out_f.close()
    print(f"\n===== Stage B4 summary =====")
    print(f"conversations written : {n}")
    print(f"context folds injected: {n_folds}  (avg {n_folds/max(1,n):.2f}/conv)")
    print(f"avg messages: {msgs_before/max(1,n):.0f} (raw) -> {msgs_after/max(1,n):.0f} (folded)")
    print(f"out  -> {out_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
