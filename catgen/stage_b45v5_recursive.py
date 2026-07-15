#!/usr/bin/env python3
"""Stage B4/5-v5 — RECURSIVE rolling-memory generation + stitch (idea #5).

The paper's C(t)=(Q, M(t), I^k) is a SINGLE evolving memory, but our v1/v2 data trains
independent per-window M's the model never learns to MAINTAIN. Here each memory is
produced recursively and each fold becomes ONE bounded rolling-workspace example that
matches inference exactly:

    M_i  = summarize( M_{i-1}  +  raw steps since the last fold )      # update, not recreate
    example_i = [ Q,
                  ctx(M_{i-1}), ack,            # prior memory (context, masked)
                  raw steps (prev_hi+1 .. hi_i),# the activity being folded (context, masked)
                  ctx(M_i), ack,                # <<< TARGET: learn to emit the updated memory
                  raw steps (hi_i+1 .. +horizon)]  # continuation: learn to USE M_i (trained)

Only tokens from `target_from_msg` onward carry loss (train cat_dataset_v5.py handles
this), so M_{i-1} is context and M_i + its use are the target — teaching memory
MERGING (carry-forward + integrate + drop-resolved), which is what "reuse compressed
representations" needs. Q (task) is always verbatim; recursion decay is bounded per-step
by idea #1's sufficiency check run on the output.

Output: cat_instruct_v5_a3b.jsonl, one rolling example per fold, {messages,
target_from_msg, ...} — plus tools.json (same 6 tools incl. context). Sequential within
a trajectory (recursion), parallel across trajectories.

Needs the STRONG summarizer endpoint (--base-url), same as stage_b3v2.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
from stage_b12_fold_points import load_steps                         # noqa: E402
from stage_b4_stitch_cat_instruct import context_messages, CONTEXT_TOOL  # noqa: E402
import stage_b3_generate_memory as b3                                # noqa: E402
import stage_b3v2_forward_memory as v2                               # noqa: E402

RECURSIVE_TEMPLATE = """# Task
{task}

# Pinned anchors (STABLE facts that never change for this task — workspace/repo path,
# core file paths, task invariants). Reproduce these VERBATIM as a [Pinned] section at
# the TOP of your memory. NEVER alter or drop a pinned line; append a new line only if a
# new permanent anchor has appeared.
{pinned}

# Current long-term memory (your memory so far — carry forward what is still relevant)
{prev_memory}

# New steps since the last memory update (integrate these)
{new_steps}

# What happens NEXT (steps after the fold) — for [Next]/[Dead Ends] ONLY
# Infer the immediate PLAN; abstract to subgoal level, do NOT copy concrete commands.
{continuation}

# Instructions
Produce an UPDATED memory that SUPERSEDES the current memory. Begin with a [Pinned]
section reproducing the pinned anchors verbatim, then the seven sections. Carry forward
still-relevant facts, integrate the new steps, and DROP items that are now resolved or
obsolete. Preserve exact paths, line numbers, errors, commands, and code edits. Target
about {budget} tokens for the four retrospective sections; keep [Pinned] and the three
forward sections brief.

# Output
Return the [Pinned] section followed by the seven-section memory block, nothing else."""

_FIRST_MEM = "(none — this is the first memory for this trajectory)"
_FIRST_PINNED = ("(none yet — create [Pinned] now: the workspace/repo path plus any core "
                 "file paths or task invariants that stay constant for the whole task)")

# [Pinned] + the 7 forward-schema headers (v2._REQUIRED_HEADERS). Used to slice sections.
_V5_ALL_HEADERS = ("[Pinned]",) + v2._REQUIRED_HEADERS


def extract_section(M: str, header: str) -> str:
    """Return the text under `header` in M (excluding the header line), or ''."""
    i = M.find(header)
    if i < 0:
        return ""
    tail = M[i + len(header):]
    ends = [e for e in (tail.find(h) for h in _V5_ALL_HEADERS if h != header) if e >= 0]
    return (tail[:min(ends)] if ends else tail).strip(": \n")


def set_pinned(M: str, content: str) -> str:
    """Force M's [Pinned] section to `content` (insert at top if absent, else replace)."""
    block = f"[Pinned]\n{content}".rstrip()
    i = M.find("[Pinned]")
    if i < 0:
        return block + "\n\n" + M.strip()
    tail = M[i + len("[Pinned]"):]
    ends = [e for e in (tail.find(h) for h in _V5_ALL_HEADERS if h != "[Pinned]") if e >= 0]
    cut = i + len("[Pinned]") + min(ends) if ends else len(M)
    return (M[:i] + block + "\n\n" + M[cut:]).strip()


def merge_pinned(carried: str, model_pinned: str) -> str:
    """Monotonic carry-forward: prior pinned lines verbatim + any new anchor lines."""
    out, seen = [], set()
    for src in (carried, model_pinned):
        for ln in (src or "").splitlines():
            s = ln.strip()
            if s and s.lower().strip("-• ") not in ("", "none", "none yet") and s not in seen:
                seen.add(s); out.append(ln.rstrip())
    return "\n".join(out) if out else "None yet"


def render_steps_range(steps, a, b, obs_cap=None):
    parts = []
    for i in range(a, b):
        s = b3.render_step(steps[i])
        if obs_cap and "Observation:" in s and len(s) > obs_cap:
            head, _, obs = s.partition("Observation:")
            s = head + "Observation: " + (obs.strip()[:obs_cap] + " …[truncated]")
        parts.append(s)
    return "\n\n".join(parts)


def gen_recursive_M(client, args, task, prev_M, prev_pinned, steps, prev_hi, hi, horizon):
    """Returns (M, pinned). M carries a verbatim-forwarded [Pinned] block, has all seven
    forward-schema headers, and is size-bounded by the ratio self-heal."""
    new_steps = render_steps_range(steps, prev_hi + 1, hi + 1)
    continuation = render_steps_range(steps, hi + 1, min(len(steps), hi + 1 + horizon),
                                      obs_cap=args.cont_obs_cap) or \
        "(none — near the end of the trajectory; the task is being finalized)"
    budget = max(128, min(int(len(new_steps.split()) * 1.3 * args.target_ratio),
                          args.max_m_tokens))
    out_cap = min(args.max_m_tokens + 768, max(1280, budget * 2))
    prompt = RECURSIVE_TEMPLATE.format(
        task=task[:args.max_task_chars], pinned=prev_pinned or _FIRST_PINNED,
        prev_memory=prev_M or _FIRST_MEM,
        new_steps=new_steps[:args.max_slice_chars], continuation=continuation,
        budget=budget)

    def _gen(p):
        return b3.summarize(client, args.model, v2.SUMMARIZER_SYSTEM_V2, p,
                            max_tokens=out_cap, temperature=args.temperature,
                            extra_body=b3.NO_THINK)

    M = _gen(prompt)
    # (a) self-heal a missing section once
    if not all(h in M for h in v2._REQUIRED_HEADERS):
        M = _gen(prompt + ("\n\nYour memory was missing one or more of the seven sections. "
                           "Rewrite with the [Pinned] section then ALL seven headers verbatim "
                           "(use 'None'/'None yet' where empty); keep [Next] plan-level."))
    # (b) #3 ratio self-heal — word-proxy vs the material being consolidated (prev M + new
    #     steps). max_ratio<1 makes the recursion a contraction, so M stays bounded.
    seg_words = len(new_steps.split()) + len((prev_M or "").split())
    ratio = len(M.split()) / max(1, seg_words)
    if ratio > args.max_ratio:
        M = _gen(prompt + (
            f"\n\nYour memory was too long. Rewrite the four retrospective sections under "
            f"{budget} tokens — drop only filler (logs, dir listings, file dumps, navigation); "
            f"keep every path, symbol, error, command, code edit, failed attempt, and "
            f"constraint. Keep [Pinned] and the three forward sections brief. All headers stay."))
    elif ratio < args.min_ratio:
        M = _gen(prompt + (
            "\n\nYour memory was too short and likely dropped important detail. Rewrite adding "
            "back the specific paths, line numbers, commands, code edits, test results, and "
            "failed attempts. Keep [Pinned] and all seven headers; forward sections stay brief."))
    # (c) #1 pinned facts — force verbatim monotonic carry-forward regardless of the model
    merged_pinned = merge_pinned(prev_pinned, extract_section(M, "[Pinned]"))
    M = set_pinned(M, merged_pinned)
    return M, merged_pinned


def build_rolling_example(head, steps, prev_M, prev_sig, prev_hi, fp, M_i, horizon):
    """One bounded rolling-workspace example for this fold. Returns (messages,
    target_from_msg)."""
    lo, hi = fp["compressible_steps"]
    msgs = list(head)                                    # Q
    if prev_M is not None:
        msgs += context_messages(prev_M, prev_sig)       # prior memory (context)
    for i in range(prev_hi + 1, hi + 1):                 # the raw activity being folded
        msgs += steps[i]["raw"]
    target_from = len(msgs)                              # <<< loss starts here
    msgs += context_messages(M_i, fp.get("signal", ""))  # TARGET: emit updated memory
    for i in range(hi + 1, min(len(steps), hi + 1 + horizon)):  # continuation (trained)
        msgs += steps[i]["raw"]
    return msgs, target_from


def process_traj(client, args, plan, pool):
    tid = plan["trajectory_id"]
    traj = pool.get(tid)
    if traj is None:
        return []
    head, steps = load_steps(traj)
    task = b3.task_text(head)
    folds = [fp for fp in plan["fold_points"] if fp.get("compressible_steps")]
    folds.sort(key=lambda fp: fp["compressible_steps"][0])
    examples = []
    prev_M = None; prev_sig = ""; prev_hi = -1; prev_pinned = ""
    for fp in folds:
        lo, hi = fp["compressible_steps"]
        if lo <= prev_hi:                                 # overlapping/degenerate -> skip
            continue
        try:
            M_i, prev_pinned = gen_recursive_M(client, args, task, prev_M, prev_pinned,
                                               steps, prev_hi, hi, args.horizon)
        except Exception as exc:  # noqa: BLE001
            return {"_error": f"{type(exc).__name__}: {exc}", "tid": tid}
        msgs, tfrom = build_rolling_example(head, steps, prev_M, prev_sig, prev_hi, fp,
                                            M_i, args.horizon)
        examples.append({"trajectory_id": tid, "instance_id": plan.get("instance_id"),
                         "fold_signal": fp.get("signal"), "M": M_i,
                         "target_from_msg": tfrom, "messages": msgs})
        prev_M, prev_sig, prev_hi = M_i, fp.get("signal", ""), hi
    return examples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-plans", default="data/fold_plans/fold_plans.jsonl")
    ap.add_argument("--pool", default="data/raw/tbase_pool_min50.jsonl")
    ap.add_argument("--out", default="data/cat_instruct/cat_instruct_v5_a3b.jsonl")
    ap.add_argument("--tools-in", default=str(_here / "tools.json"))
    ap.add_argument("--tools-out", default="data/cat_instruct/tools_v5.json")
    ap.add_argument("--base-url", default="http://localhost:8000/v1",
                    help="STRONG summarizer endpoint (teacher), not the 9B student")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="Qwen3.6-27B")
    ap.add_argument("--target-ratio", type=float, default=0.30)
    ap.add_argument("--max-ratio", type=float, default=0.62,
                    help="#3 upper bound on M/(prevM+newsteps) word-ratio; <1 keeps recursion bounded")
    ap.add_argument("--min-ratio", type=float, default=0.18,
                    help="#3 lower bound; below this, M likely dropped detail -> expand")
    ap.add_argument("--max-task-chars", type=int, default=12000,
                    help="#4 Q anchor cap (was hard 3000); ~verbatim for virtually all tasks")
    ap.add_argument("--max-m-tokens", type=int, default=4000)
    ap.add_argument("--max-slice-chars", type=int, default=48000)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--horizon", type=int, default=6, help="continuation steps kept raw + used")
    ap.add_argument("--cont-obs-cap", type=int, default=600)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=5, timeout=600)

    # tools.json = base tools + context (same as b4). Robust to a tools-in that
    # ALREADY contains context (e.g. tools_a3b.json): don't double-add it.
    base_tools = json.load(open(args.tools_in))
    norm = [t if "type" in t else {"type": "function",
            "function": t.get("function", t)} for t in base_tools]
    has_ctx = any((t.get("function", t).get("name")) == "context" for t in norm)
    aug = norm if has_ctx else norm + [CONTEXT_TOOL]
    Path(args.tools_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(aug, open(args.tools_out, "w"), ensure_ascii=False, indent=2)

    print(f"[load] indexing pool {args.pool}", flush=True)
    pool = {}
    with open(args.pool) as f:
        for line in f:
            r = json.loads(line); pool[r["trajectory_id"]] = r["trajectory"]
    print(f"[load] pool indexed: {len(pool)}", flush=True)

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path = out_path.with_suffix(out_path.suffix + ".failed")
    done = set()
    if out_path.exists():
        for line in out_path.open():
            try: done.add(json.loads(line)["trajectory_id"])
            except Exception: pass
        print(f"[resume] {len(done)} trajectories already done", flush=True)

    write_lock = threading.Lock(); stat_lock = threading.Lock()
    g = out_path.open("a"); gf = failed_path.open("a")
    stat = {"traj": 0, "examples": 0, "failed": 0}
    t0 = time.time()

    def handle(plan):
        res = process_traj(client, args, plan, pool)
        if isinstance(res, dict) and res.get("_error"):
            with write_lock:
                gf.write(json.dumps({"trajectory_id": res["tid"],
                                     "error": res["_error"]}) + "\n"); gf.flush()
            with stat_lock: stat["failed"] += 1
            return
        with write_lock:
            for ex in res:
                g.write(json.dumps(ex, ensure_ascii=False) + "\n")
            g.flush()
        with stat_lock:
            stat["traj"] += 1; stat["examples"] += len(res)
            if stat["traj"] % 25 == 0:
                rate = stat["traj"] / max(1e-6, time.time() - t0)
                print(f"  traj={stat['traj']} examples={stat['examples']} "
                      f"failed={stat['failed']} ({rate:.2f} traj/s)", flush=True)

    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        for line in open(args.fold_plans):
            if args.limit and n >= args.limit:
                break
            plan = json.loads(line)
            if plan["trajectory_id"] in done:
                continue
            n += 1
            futs.append(ex.submit(handle, plan))
        for fu in futs:
            fu.result()
    g.close(); gf.close()
    print("\n===== Stage B4/5-v5 (recursive rolling) summary =====")
    print(f"trajectories : {stat['traj']} (failed {stat['failed']})")
    print(f"rolling examples (one per fold): {stat['examples']}")
    print(f"out    -> {out_path}")
    print(f"tools  -> {args.tools_out}")
    print(f"failed -> {failed_path}")
    print("NOTE: train with cat_dataset_v5.py (masks tokens before target_from_msg).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
