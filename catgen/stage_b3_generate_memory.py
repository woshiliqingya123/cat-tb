#!/usr/bin/env python3
"""Stage B3 — structured long-term memory (M) generation via the summarizer LLM.

CAT-GENERATOR Phase II step (3). For every fold point produced by B1/B2, take the
compressible history segment (the raw steps between the previous fold and the
recent-k window) and ask the summarizer (Qwen3.6-27B, same backbone as the target
model) to condense it into a STRUCTURED memory block M with four fixed sections:

    [Goal]                 what this phase was trying to achieve
    [Attempts & Results]   strategies attempted and their outcomes
    [Environment Feedback] salient env feedback: errors, paths, configs, file/symbol names
    [Active Constraints]   constraints that still bind subsequent reasoning

Target compression ratio ~30% (paper Table 1: 15585 -> 4676 tokens). If M exceeds
--max-ratio it is regenerated once stricter; if M falls below --min-ratio (and the
slice was not truncated) it is regenerated once asking for more specifics, so dense
segments are not over-compressed. M becomes the Observation of the injected `context`
tool call in B4.

Runs against an OpenAI-compatible endpoint (vLLM). Designed to be resumable: it
skips fold points that already have an `M` in the output file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer
from openai import OpenAI

# B1/B2 step splitting, reused so we can recover the raw messages per step.
from stage_b12_fold_points import load_steps  # noqa: E402

# Chat-template kwargs to disable the summarizer's <think> trace (Qwen3.6 is a
# reasoning model; we want the clean four-section block, not a CoT, and we don't
# want to burn output tokens on discarded reasoning). Pass as extra_body.
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}

SUMMARIZER_SYSTEM = """You are the long-term memory writer for a software-engineering ReAct agent.

Your job: compress a soon-to-be-removed slice of the agent's own trajectory into a
structured memory block the agent will rely on for ALL future reasoning. This memory
REPLACES the raw history. If you omit a critical path, line number, symbol, command,
error, test result, patch detail, failed attempt, or constraint, the future agent may
repeat mistakes or break the task.

Output ONLY the final memory block. No analysis, no chain-of-thought, no meta
commentary, no <think> or </think>, no "Let me analyze".

Write EVERYTHING in English, including the section headers. Use exactly these four
headers verbatim:
[Goal]                 the goal / subtask this slice was pursuing
[Attempts & Results]   strategies tried and their outcomes; ALSO put root-cause
                       conclusions, current hypotheses, and disproven hypotheses here
[Environment Feedback] salient environment feedback: errors, paths, configs, results
[Active Constraints]   constraints / invariants / contracts that still bind later steps

LENGTH: Faithfulness OUTRANKS brevity. Keep roughly 30% of the slice (MORE for dense
technical slices; less ONLY when the slice is mostly duplicated logs or listings).
NEVER drop a critical technical fact to hit a length target — exceed the budget instead.
Do not collapse the slice into a vague high-level summary.

FAITHFULNESS: Include ONLY facts that actually appear in the slice. NEVER guess or infer
paths, symbols, line numbers, or results that are not present, and never label anything
as "probably" / "speculation" / "推测". If something is not in the slice, omit it.

COMPRESSION POLICY — this is your core job, decide it fact by fact.

GOVERNING TEST. Keep a fact IF AND ONLY IF a future agent that can no longer see this
raw history would, without it, do one of: (a) redo work already finished, (b) repeat an
approach already shown to fail, (c) lose a confirmed result, root cause, or decision, or
(d) violate a constraint. If a fact fails this test — it is reconstructible by simply
looking again, or it never changed any conclusion — compress or drop it. Apply this test
yourself, step by step; you decide what survives.

IMPORTANCE IS NOT LENGTH. Judge each step by its causal role, NEVER by its size. A short
step is often the pivotal fact — a one-line edit that is the actual fix, a single command
that revealed the root cause, one decisive error line — and must be kept in full. A long
step (a big file or directory dump, verbose logs) is often almost entirely redundant and
should be compressed hard. Never drop a small step because it is short; never keep a long
dump because it is long.

ALWAYS PRESERVE (decision-critical; cannot be re-derived from nothing):
- Exact identifiers: file/dir paths, line numbers, traceback locations, hunk headers,
  function/class/method/variable names, config keys, CLI flags, test/fixture names,
  branch/commit names.
- Commands that were run, especially reproduction and test commands.
- Concrete results: which tests passed/failed (with test IDs + assertion text),
  exception class + first meaningful error line, numeric/observed outputs.
- Code changes actually made: file + function changed, old behavior -> new behavior,
  the minimal snippet or diff intent.
- Root-cause conclusions and the current working hypothesis.
- EVERY failed attempt AND why it failed (this is what stops the agent repeating it).
- Constraints / invariants / API contracts / user requirements still in force.

SAFE TO COMPRESS OR DROP (reconstructible on demand, or redundant):
- Full file contents from a read-only view/cat -> keep only the relevant lines,
  signatures, and the conclusion drawn; the file can be re-read if needed later.
- Directory trees / listings -> keep only the paths that matter.
- Repeated or unchanged observations, progress bars, install logs, tool boilerplate.
- Navigation narration ("let me look at X", "now check Y") -> keep WHAT it found, not
  the act of looking.
- The agent's verbose musing that did not lead to a decision.

WHEN UNSURE, KEEP IT. A slightly longer memory is fine; a lost path, failed attempt,
result, or constraint is not.

THIS DATASET (OpenHands SWE trajectories) — rules verified against all 1.46M steps:
- Almost every bash observation ends with four metadata lines (each ~748k occurrences).
  DROP them: "[Command finished with exit code N]", "[The command completed with exit
  code N.]", "[Python interpreter: ...]", "[Current working directory: ...]".
  EXCEPTIONS you MUST keep: a NON-ZERO exit code (1, 2, 127=command-not-found,
  124=timeout, 139=segfault, ...) is a real failure — keep it and what failed; and
  "... CTRL+C was sent" means the command hung / was interrupted — keep that too.
- DROP pure-noise lines: "[ N%]" progress bars, "[GCC x.y.z]" banner fragments.
- The `think` tool observation is ALWAYS exactly "Your thought has been logged."
  (verified 58,333/58,333) — DROP it; the real content is the Thought in the action.
- `task_tracker`: DROP the "Task list has been updated with N items." confirmation, but
  a `task_tracker view` returns the actual checklist with ✅ done / ⏳ pending markers —
  KEEP which subtasks are done vs pending (that is real progress state).
- `str_replace_editor view` (370k) and directory dumps ("files and directories up to N
  levels deep", 58k) are re-readable later — keep only the relevant paths / lines /
  signatures and the conclusion drawn.
- `str_replace_editor` create / str_replace EDIT confirmations (106k, "The file X has
  been edited. Here's the result of running `cat -n` ... <numbered snippet>") are NOT
  filler — they ARE the code change. KEEP the file path, the function/lines edited, and
  what the new code does; trim only the unchanged context lines around the change.
- pytest / test output: DROP the deprecation-warning preamble and session banners; KEEP
  the pass/fail summary, failing test IDs, assertion messages, and tracebacks (file:line).
- Throwaway scripts the agent CREATED to investigate (reproduce_issue.py, debug_*.py,
  test_*.py, verify_*.py, explore_*.py — not files that exist in the repo): you may drop
  the script's filename and body, but KEEP what it established — whether it reproduced
  the bug and the key output/error/result it produced.
- "[... Observation truncated due to length ...]" (24,806x) and "[Previous command
  outputs are truncated. Showing the last N lines ...]" mean output was cut — note it;
  never imply you saw the whole thing.

TEMPORAL CONSISTENCY:
- If this slice corrects or disproves an earlier assumption, write "OVERTURNED: ..." or
  "CORRECTED TO: ...".
- Distinguish final constraints from temporary buggy behavior.
- Never present a failed intermediate patch as the final solution."""

SUMMARIZER_TEMPLATE = """# Task
{task}

# Fold location
Compress trajectory steps {lo} through {hi}. This raw history is removed after compression.

# Raw trajectory slice
{slice_text}

# Length target
About {budget} tokens. For dense technical evidence, 0.25-0.5 of the slice is acceptable;
shorter is fine if the slice is mostly duplicated logs or listings. If the budget is too
small to keep all critical facts, exceed it rather than dropping them.

# Output
Return exactly the four-section memory block and nothing else."""


def render_step(step: dict) -> str:
    thought = (step.get("thought") or "").strip()
    actions = []
    for m in step["raw"]:
        if m["role"] == "assistant":
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                args_s = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
                actions.append(f"{fn.get('name')}({args_s})")
    obs = (step.get("obs_text") or "").strip()
    parts = []
    if thought:
        parts.append(f"Thought: {thought}")
    if actions:
        parts.append("Action: " + " ; ".join(actions))
    if obs:
        parts.append(f"Observation: {obs}")
    return "\n".join(parts)


def task_text(head: list[dict]) -> str:
    for m in head:
        if m["role"] == "user":
            return (m.get("content") or "")[:4000]
    return ""


def count_tokens(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def summarize(client, model, sys_prompt, user_prompt, max_tokens, temperature,
              extra_body=None):
    """Generate M; self-heal output truncation. If the model stops because it hit the
    token ceiling (finish_reason == 'length'), retry once with double the room so M is
    never cut off mid-section."""
    messages = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}]
    content = ""
    for _ in range(2):
        resp = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature, extra_body=extra_body,
        )
        choice = resp.choices[0]
        content = (choice.message.content or "").strip()
        if getattr(choice, "finish_reason", None) != "length":
            break
        max_tokens = min(max_tokens * 2, 8192)  # truncated -> more room, retry once
    return content


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-plans", default="data/fold_plans/fold_plans.jsonl")
    ap.add_argument("--pool", default="data/raw/tbase_pool_min50.jsonl")
    ap.add_argument("--out", default="data/fold_plans/fold_plans_with_M.jsonl")
    ap.add_argument("--tokenizer", default="/data/models/Qwen3.6-27B")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="Qwen3.6-27B")
    ap.add_argument("--target-ratio", type=float, default=0.30,
                    help="target M tokens / compressible tokens")
    ap.add_argument("--max-ratio", type=float, default=0.60,
                    help="regenerate stricter only if M exceeds this ratio "
                         "(raised from 0.45 so the preserve-first prompt isn't undone)")
    ap.add_argument("--min-ratio", type=float, default=0.18,
                    help="regenerate asking for more specifics if M falls below this "
                         "ratio (catches over-compressed dense segments)")
    ap.add_argument("--max-m-tokens", type=int, default=6000,
                    help="absolute cap on the per-fold M length budget")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=0, help="process only first N trajectories")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    # index the trajectory pool by id (stream once, keep only needed fields)
    print(f"[load] indexing pool {args.pool}")
    pool: dict[str, list[dict]] = {}
    with open(args.pool) as f:
        for line in f:
            rec = json.loads(line)
            pool[rec["trajectory_id"]] = rec["trajectory"]

    # resume: collect already-done trajectory_ids
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["trajectory_id"])
                except Exception:
                    pass
        print(f"[resume] {len(done)} trajectories already done")

    n_traj = n_folds = 0
    ratios = []
    t0 = time.time()
    with open(args.fold_plans) as f, out_path.open("a") as g:
        for line in f:
            if args.limit and n_traj >= args.limit:
                break
            plan = json.loads(line)
            tid = plan["trajectory_id"]
            if tid in done:
                continue
            traj = pool.get(tid)
            if traj is None:
                continue
            head, steps = load_steps(traj)
            task = task_text(head)

            out_folds = []
            for fp in plan["fold_points"]:
                cs = fp.get("compressible_steps")
                if not cs:
                    continue
                lo, hi = cs
                slice_text = "\n\n".join(render_step(steps[i]) for i in range(lo, hi + 1))
                seg_tokens = fp["compressible_tokens"]
                budget = max(128, min(int(seg_tokens * args.target_ratio), args.max_m_tokens))
                # Hard ceiling DECOUPLED from the soft budget: a generous floor so the
                # model can always finish all four sections (the prompt, not max_tokens,
                # controls target length). Coupling out_cap to budget truncated small
                # segments mid-section.
                out_cap = min(args.max_m_tokens + 512, max(1024, budget * 2))
                user_prompt = SUMMARIZER_TEMPLATE.format(
                    task=task, lo=lo, hi=hi, slice_text=slice_text, budget=budget)
                m = summarize(client, args.model, SUMMARIZER_SYSTEM, user_prompt,
                              max_tokens=out_cap, temperature=args.temperature,
                              extra_body=NO_THINK)
                m_tok = count_tokens(tok, m)
                ratio = m_tok / max(1, seg_tokens)
                if ratio > args.max_ratio:
                    strict = user_prompt + (
                        f"\n\nYour previous summary was too long. Rewrite it in under "
                        f"{budget} tokens. Drop only filler (logs, dir listings, file "
                        f"dumps, navigation) — keep every path, symbol, error, command, "
                        f"code edit, failed attempt, and constraint.")
                    m = summarize(client, args.model, SUMMARIZER_SYSTEM, strict,
                                  max_tokens=out_cap, temperature=args.temperature,
                                  extra_body=NO_THINK)
                    m_tok = count_tokens(tok, m)
                    ratio = m_tok / max(1, seg_tokens)
                elif ratio < args.min_ratio:
                    expand = user_prompt + (
                        f"\n\nYour previous summary was too short and likely dropped "
                        f"important detail. Rewrite it adding back the specific paths, "
                        f"line numbers, commands, code edits, test results, and failed "
                        f"attempts from the slice. Stay short ONLY if the slice is "
                        f"genuinely mostly duplicated logs or directory listings.")
                    m = summarize(client, args.model, SUMMARIZER_SYSTEM, expand,
                                  max_tokens=out_cap, temperature=args.temperature,
                                  extra_body=NO_THINK)
                    m_tok = count_tokens(tok, m)
                    ratio = m_tok / max(1, seg_tokens)
                out_folds.append({
                    **fp,
                    "M": m,
                    "M_tokens": m_tok,
                    "compression_ratio": round(ratio, 3),
                })
                ratios.append(ratio)
                n_folds += 1

            g.write(json.dumps({
                "trajectory_id": tid,
                "instance_id": plan.get("instance_id"),
                "num_steps": plan["num_steps"],
                "fold_points": out_folds,
            }, ensure_ascii=False) + "\n")
            g.flush()
            n_traj += 1
            if n_traj % 50 == 0:
                avg = sum(ratios) / max(1, len(ratios))
                rate = n_traj / (time.time() - t0)
                print(f"  ... traj={n_traj} folds={n_folds} avg_ratio={avg:.3f} "
                      f"({rate:.2f} traj/s)", flush=True)

    avg = sum(ratios) / max(1, len(ratios))
    print(f"\n===== Stage B3 summary =====")
    print(f"trajectories : {n_traj}")
    print(f"folds (M gen): {n_folds}")
    print(f"avg compression ratio: {avg:.3f}  (target {args.target_ratio}, paper ~0.30)")
    print(f"out -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
