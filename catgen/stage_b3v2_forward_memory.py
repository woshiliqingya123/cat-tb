#!/usr/bin/env python3
"""Stage B3-v2 — FORWARD-LOOKING structured memory M (idea #4).

Same faithful, hardened M-generation as stage_b3_generate_memory_parallel.py, but M
gains three FORWARD sections on top of the four retrospective ones, so the folded
memory carries not just "what happened" but "what to do next / what's unresolved /
what NOT to retry". This is what lets the agent skip re-deriving the plan after a
fold (the paper's "reuse compressed representations", which our probe showed did NOT
transfer), and the [Dead Ends] section directly attacks the repeat-action death loop.

Seven sections (first four identical to v1, verbatim headers):
    [Goal]                 subtask this slice pursued
    [Attempts & Results]   strategies tried + outcomes + confirmed root cause / hypothesis
    [Environment Feedback] salient env feedback: errors, paths, configs, results
    [Active Constraints]   invariants / contracts still binding
    [Dead Ends]            approaches proven not to work -> imperative "DO NOT retry X (Y)"
    [Open Threads]         unresolved subgoals / partial work / hypotheses still to verify
    [Next]                 the immediate PLAN, at SUBGOAL level (never concrete commands)

The three forward sections are grounded in the trajectory AFTER the fold (steps
hi+1..hi+horizon), shown to the summarizer for [Next]/[Dead Ends] inference ONLY.
Strict anti-leakage: [Next] must be plan-level; concrete future commands / edits /
code are NEVER copied verbatim, or the student just memorizes the gold path instead
of learning to plan.

Output schema is byte-compatible with fold_plans_with_M*.jsonl, so stage_b4 stitch
is UNCHANGED. The regenerated M is the input for idea #1 (M-sufficiency rejection
sampling) and idea #5 (recursive M update), which run on top of this file.

Reuses (imported, not reimplemented) from stage_b3_generate_memory:
    render_step / task_text / count_tokens / summarize / NO_THINK
and the entire verified COMPRESSION POLICY tail of SUMMARIZER_SYSTEM (sliced in, so
the dataset-specific keep/drop rules validated against 1.46M steps are preserved).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transformers import AutoTokenizer
from openai import OpenAI

from stage_b12_fold_points import load_steps
import stage_b3_generate_memory as b3

INFLIGHT_FACTOR = 2

# ---- v2 prompt: new role + 7-section schema + forward/anti-leak rules, then the
# proven COMPRESSION POLICY tail (governing test, importance-not-length, always-
# preserve, dataset rules) reused verbatim from v1 so nothing validated is lost.
_POLICY_MARKER = "COMPRESSION POLICY"
_i = b3.SUMMARIZER_SYSTEM.find(_POLICY_MARKER)
if _i < 0:
    raise RuntimeError("could not locate COMPRESSION POLICY tail in b3.SUMMARIZER_SYSTEM")
_POLICY_TAIL = b3.SUMMARIZER_SYSTEM[_i:]

_V2_HEADER = """You are the long-term memory writer for a software-engineering ReAct agent.

Your job: compress a soon-to-be-removed slice of the agent's own trajectory into a
structured memory block the agent will rely on for ALL future reasoning. This memory
REPLACES the raw history. Beyond faithfully recording what happened, you also write
FORWARD guidance so the agent can continue WITHOUT re-deriving its plan. If you omit a
critical path, line number, symbol, command, error, result, patch, failed attempt, or
constraint, the future agent may repeat mistakes or break the task.

Output ONLY the final memory block. No analysis, no chain-of-thought, no meta
commentary, no <think> or </think>, no "Let me analyze". Write EVERYTHING in English,
including the section headers. Use exactly these SEVEN headers verbatim, in this order:

[Goal]                 the goal / subtask this slice was pursuing
[Attempts & Results]   strategies tried and their outcomes; ALSO root-cause conclusions,
                       the current working hypothesis, and disproven hypotheses
[Environment Feedback] salient environment feedback: errors, paths, configs, results
[Active Constraints]   constraints / invariants / contracts that still bind later steps
[Dead Ends]            approaches ALREADY PROVEN not to work, as imperative guardrails:
                       "DO NOT retry <approach> — <why it failed>". One line each. This is
                       what stops the agent from looping on a known-bad action. If nothing
                       has been ruled out yet, write "None yet."
[Open Threads]         still-unresolved subgoals, partially-done work, and hypotheses not
                       yet verified — what remains to close out. If none, write "None."
[Next]                 the immediate PLAN going forward, at SUBGOAL level: 1-4 short bullets
                       describing WHAT to accomplish next and HOW to verify it — e.g.
                       "Verify the fix by re-running the previously failing test", NOT the
                       literal command. If the task is essentially done, say "Finalize:
                       <what remains to submit>".

FORWARD-SECTION RULES (critical — violating these makes the data harmful):
- [Next] is a PLAN, not a script. State the subgoal and the verification intent. NEVER
  copy a concrete future command, file edit, diff, code snippet, or exact argument from
  the "What happens NEXT" section — abstract it to intent. The agent must still choose
  the concrete action itself.
- [Dead Ends] and [Open Threads] come from what has ALREADY happened in the slice (and,
  for [Next], the immediate direction). Do not invent steps that are not supported.
- Keep the three forward sections BRIEF (plan-level bullets). The retrospective four
  sections carry the detailed facts under the length target below.

""" + "\n"  # keep a blank line before the policy tail

SUMMARIZER_SYSTEM_V2 = _V2_HEADER + _POLICY_TAIL

SUMMARIZER_TEMPLATE_V2 = """# Task
{task}

# Fold location
Compress trajectory steps {lo} through {hi}. This raw history is removed after compression.

# Raw trajectory slice (steps {lo}-{hi}) — the history being compressed
{slice_text}

# What happens NEXT (steps after {hi}) — for writing [Next]/[Dead Ends] ONLY
# Infer the immediate PLAN from this. Abstract to subgoal level; do NOT copy these
# concrete commands, edits, or code into [Next]. If empty, the fold is near the end.
{continuation_text}

# Length target
About {budget} tokens for the four retrospective sections; for dense technical evidence
0.25-0.5 of the slice is fine. The three forward sections must be brief plan-level bullets.
If the budget is too small to keep all critical facts, exceed it rather than dropping them.

# Output
Return exactly the seven-section memory block and nothing else."""


def render_continuation(steps, hi, horizon, obs_cap):
    """Render steps hi+1..hi+horizon compactly (thought + action + truncated obs) for
    forward-section grounding. Observations are truncated because [Next] needs the
    DIRECTION, not full outputs — and truncation further discourages verbatim leakage."""
    end = min(len(steps), hi + 1 + horizon)
    if hi + 1 >= end:
        return "(none — this fold is near the end of the trajectory; the task is being finalized)"
    parts = []
    for i in range(hi + 1, end):
        full = b3.render_step(steps[i])
        # truncate only the Observation portion, keep Thought/Action intact
        if "Observation:" in full and len(full) > obs_cap:
            head, _, obs = full.partition("Observation:")
            obs = obs.strip()
            if len(obs) > obs_cap:
                obs = obs[:obs_cap] + " …[truncated]"
            full = head + "Observation: " + obs
        parts.append(f"[future step {i}]\n{full}")
    return "\n\n".join(parts)


def build_fold_job(tok, steps, task, fp, args):
    lo, hi = fp["compressible_steps"]
    slice_text = "\n\n".join(b3.render_step(steps[i]) for i in range(lo, hi + 1))
    continuation_text = render_continuation(steps, hi, args.horizon, args.cont_obs_cap)
    seg_tokens = fp["compressible_tokens"]
    budget = max(128, min(int(seg_tokens * args.target_ratio), args.max_m_tokens))
    out_cap = min(args.max_m_tokens + 768, max(1280, budget * 2))  # +room for 3 fwd sections
    cont_tok = b3.count_tokens(tok, continuation_text)
    max_in = args.max_model_len - out_cap - cont_tok - 384
    slice_text, truncated = _truncate(tok, slice_text, max_in)
    user_prompt = SUMMARIZER_TEMPLATE_V2.format(
        task=task, lo=lo, hi=hi, slice_text=slice_text,
        continuation_text=continuation_text, budget=budget)
    return {"fp": fp, "user_prompt": user_prompt, "budget": budget,
            "out_cap": out_cap, "seg_tokens": seg_tokens, "truncated": truncated}


def _truncate(tok, slice_text, max_in):
    ids = tok(slice_text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_in:
        return slice_text, False
    head_n = int(max_in * 0.60); tail_n = max(0, max_in - head_n)
    return (tok.decode(ids[:head_n]) + "\n\n[... trajectory middle elided for length ...]\n\n"
            + tok.decode(ids[-tail_n:])), True


_REQUIRED_HEADERS = ("[Goal]", "[Attempts & Results]", "[Environment Feedback]",
                     "[Active Constraints]", "[Dead Ends]", "[Open Threads]", "[Next]")


def run_fold(client, tok, args, job):
    """Never raises. Returns (out_dict, ratio, ok)."""
    try:
        fp, user_prompt = job["fp"], job["user_prompt"]
        budget, seg_tokens = job["budget"], job["seg_tokens"]
        m = b3.summarize(client, args.model, SUMMARIZER_SYSTEM_V2, user_prompt,
                         max_tokens=job["out_cap"], temperature=args.temperature,
                         extra_body=b3.NO_THINK)
        # self-heal a missing forward section once (weak summarizers sometimes stop
        # after the four retrospective sections).
        if not all(h in m for h in _REQUIRED_HEADERS):
            fix = user_prompt + (
                "\n\nYour previous memory was missing one or more of the required seven "
                "sections. Rewrite it with ALL seven headers verbatim, including the "
                "forward sections [Dead Ends], [Open Threads], [Next] (use 'None'/'None "
                "yet' where truly empty). Keep [Next] plan-level.")
            m = b3.summarize(client, args.model, SUMMARIZER_SYSTEM_V2, fix,
                             max_tokens=job["out_cap"], temperature=args.temperature,
                             extra_body=b3.NO_THINK)
        m_tok = b3.count_tokens(tok, m)
        ratio = m_tok / max(1, seg_tokens)
        if ratio > args.max_ratio:
            strict = user_prompt + (
                f"\n\nYour previous memory was too long. Rewrite the four retrospective "
                f"sections in under {budget} tokens — drop only filler (logs, dir listings, "
                f"file dumps, navigation), keep every path, symbol, error, command, code "
                f"edit, failed attempt, and constraint. Keep the three forward sections "
                f"brief. All seven headers must remain.")
            m = b3.summarize(client, args.model, SUMMARIZER_SYSTEM_V2, strict,
                             max_tokens=job["out_cap"], temperature=args.temperature,
                             extra_body=b3.NO_THINK)
            m_tok = b3.count_tokens(tok, m); ratio = m_tok / max(1, seg_tokens)
        elif ratio < args.min_ratio and not job["truncated"]:
            expand = user_prompt + (
                "\n\nYour previous memory was too short and likely dropped important detail. "
                "Rewrite it adding back the specific paths, line numbers, commands, code "
                "edits, test results, and failed attempts from the slice. Keep all seven "
                "headers; keep the forward sections brief.")
            m = b3.summarize(client, args.model, SUMMARIZER_SYSTEM_V2, expand,
                             max_tokens=job["out_cap"], temperature=args.temperature,
                             extra_body=b3.NO_THINK)
            m_tok = b3.count_tokens(tok, m); ratio = m_tok / max(1, seg_tokens)
        out = {**fp, "M": m, "M_tokens": m_tok, "compression_ratio": round(ratio, 3),
               "schema": "v2-forward",
               "has_all_sections": all(h in m for h in _REQUIRED_HEADERS)}
        if job["truncated"]:
            out["slice_truncated"] = True
        return out, ratio, True
    except Exception as exc:  # noqa: BLE001 — must not propagate into callback
        return {"_error": f"{type(exc).__name__}: {exc}"}, None, False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-plans", default="data/fold_plans/fold_plans.jsonl",
                    help="pre-M fold plans (step selection); M is regenerated here")
    ap.add_argument("--pool", default="data/raw/tbase_pool_min50.jsonl")
    ap.add_argument("--out", default="data/fold_plans/fold_plans_with_M_v2_a3b.jsonl")
    ap.add_argument("--tokenizer", default="/data/liqingyang/models/Qwen3.6-27B",
                    help="MUST match the summarizer's tokenizer for accurate ratios")
    ap.add_argument("--base-url", default="http://localhost:8000/v1",
                    help="summarizer (teacher) endpoint — use a STRONG model, not the 9B")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="Qwen3.6-27B")
    ap.add_argument("--target-ratio", type=float, default=0.30)
    ap.add_argument("--max-ratio", type=float, default=0.62,
                    help="slightly above v1 0.60: the 3 forward sections add a little length")
    ap.add_argument("--min-ratio", type=float, default=0.18)
    ap.add_argument("--max-m-tokens", type=int, default=6000)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--horizon", type=int, default=4,
                    help="how many post-fold steps to show for [Next]/[Dead Ends] grounding")
    ap.add_argument("--cont-obs-cap", type=int, default=600,
                    help="per-step observation char cap in the continuation (limits leakage)")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=5, timeout=600)

    print(f"[load] indexing pool {args.pool}", flush=True)
    pool: dict[str, list[dict]] = {}
    with open(args.pool) as f:
        for line in f:
            rec = json.loads(line)
            pool[rec["trajectory_id"]] = rec["trajectory"]
    print(f"[load] pool indexed: {len(pool)} trajectories", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path = out_path.with_suffix(out_path.suffix + ".failed")
    done: set[str] = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["trajectory_id"])
                except Exception:
                    pass
        print(f"[resume] {len(done)} trajectories already done", flush=True)

    write_lock = threading.Lock()
    stat_lock = threading.Lock()
    sem = threading.Semaphore(args.workers * INFLIGHT_FACTOR)
    g = out_path.open("a"); gf = failed_path.open("a")
    ratios: list[float] = []
    n_traj = [0]; n_folds = [0]; n_failed_folds = [0]; n_missing = [0]
    t0 = time.time()

    def flush_traj(plan, results, n_jobs, any_fail):
        ordered = [results[i] for i in sorted(results) if results[i] is not None]
        if any_fail or len(ordered) < n_jobs:
            with write_lock:
                gf.write(json.dumps({"trajectory_id": plan["trajectory_id"],
                                     "instance_id": plan.get("instance_id")},
                                    ensure_ascii=False) + "\n"); gf.flush()
            return
        with write_lock:
            g.write(json.dumps({
                "trajectory_id": plan["trajectory_id"],
                "instance_id": plan.get("instance_id"),
                "num_steps": plan["num_steps"],
                "fold_points": ordered,
            }, ensure_ascii=False) + "\n"); g.flush()
        with stat_lock:
            n_traj[0] += 1
            if n_traj[0] % 25 == 0:
                avg = sum(ratios) / max(1, len(ratios))
                rt = n_traj[0] / (time.time() - t0); rf = n_folds[0] / (time.time() - t0)
                print(f"  ... traj={n_traj[0]} folds={n_folds[0]} "
                      f"failed={n_failed_folds[0]} missing_section={n_missing[0]} "
                      f"avg_ratio={avg:.3f} ({rt:.2f} traj/s, {rf:.2f} folds/s)", flush=True)

    new_traj = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        with open(args.fold_plans) as f:
            for line in f:
                if args.limit and new_traj >= args.limit:
                    break
                plan = json.loads(line)
                tid = plan["trajectory_id"]
                if tid in done:
                    continue
                traj = pool.get(tid)
                if traj is None:
                    continue
                new_traj += 1
                head, steps = load_steps(traj)
                task = b3.task_text(head)
                jobs = [fp for fp in plan["fold_points"] if fp.get("compressible_steps")]
                if not jobs:
                    flush_traj(plan, {}, 0, False)
                    continue

                state = {"remaining": len(jobs), "any_fail": False}
                results: dict[int, dict] = {}
                rlock = threading.Lock()

                def make_cb(idx, plan_ref, state_ref, results_ref, rlock_ref, n_jobs):
                    def _cb(fut):
                        try:
                            out, ratio, ok = fut.result()
                        except Exception as exc:
                            out, ratio, ok = {"_error": str(exc)}, None, False
                        finally:
                            sem.release()
                        with stat_lock:
                            n_folds[0] += 1
                            if ok:
                                ratios.append(ratio)
                                if not out.get("has_all_sections", True):
                                    n_missing[0] += 1
                            else:
                                n_failed_folds[0] += 1
                        with rlock_ref:
                            results_ref[idx] = out if ok else None
                            if not ok:
                                state_ref["any_fail"] = True
                            state_ref["remaining"] -= 1
                            last = state_ref["remaining"] == 0
                        if last:
                            flush_traj(plan_ref, results_ref, n_jobs, state_ref["any_fail"])
                    return _cb

                for idx, fp in enumerate(jobs):
                    job = build_fold_job(tok, steps, task, fp, args)
                    sem.acquire()
                    fut = ex.submit(run_fold, client, tok, args, job)
                    fut.add_done_callback(
                        make_cb(idx, plan, state, results, rlock, len(jobs)))

    g.close(); gf.close()
    avg = sum(ratios) / max(1, len(ratios))
    print("\n===== Stage B3-v2 (forward-looking M) summary =====")
    print(f"trajectories written : {n_traj[0]}")
    print(f"folds done           : {n_folds[0]} (failed: {n_failed_folds[0]}, "
          f"missing-section: {n_missing[0]})")
    print(f"avg compression ratio: {avg:.3f}  (target {args.target_ratio})")
    print(f"out     -> {out_path}")
    print(f"failed  -> {failed_path} (rerun to retry these trajectories)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
