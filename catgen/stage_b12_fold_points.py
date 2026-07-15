#!/usr/bin/env python3
"""Stage B1+B2 — fold-point selection (rule-based) and context segmentation.

This is the CAT-GENERATOR Phase II steps (1) and (2). It does NOT call any LLM:
the paper selects condenser positions via "heuristic triggers", and the summarizer
LLM is only used later (B3) to write the memory block M. Here we deterministically
detect the three signal families on each base trajectory and emit a "fold plan".

Signals (manual §5-B1):
  (a) expansion       : cumulative context tokens since the last fold exceed T_fold
  (b) structural      : task_tracker(plan) call | test-pass marker in observation |
                        first file write (str_replace_editor create/str_replace)
                        after a run of read-only exploration
  (c) error-recovery  : >= N consecutive steps with error/traceback observations,
                        then a step whose observation is error-free

A step is a fold *candidate* if it matches any signal. Candidates are then thinned
by --min-gap, and we never fold inside the last --recent-k steps (those must stay
verbatim as I^(k)). For each accepted fold a_i we also record the B2 segmentation:
    Q                = system msg + first user msg (task spec)
    I^(k)            = the recent-k steps ending at a_i
    compressible     = steps from (previous fold end | start) up to the recent-k window

Output:
  data/fold_plans/fold_plans.jsonl   one record per trajectory with its fold points
  reports/stage_b1_stats.json        fold-density stats vs the paper's ~4.22/87 anchor
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

# ----- signal lexicons (deterministic, substring-based to avoid regex backtracking) -----
# Observations can contain test-runner separator lines full of '=' which made
# regex alternations backtrack catastrophically; plain lowercased substring
# membership is O(n) and ~1000x faster here.
# Domain-general CLI/agent error lexicon (not just Python) — so error_recovery fires
# on any terminal/agent domain, not only code tracebacks.
ERROR_MARKERS = (
    "traceback (most recent call last)", "error", "exception",
    "command not found", "no such file", "failed", "failure",
    "assertionerror", "syntaxerror", "modulenotfounderror", "non-zero exit",
    "permission denied", "fatal:", "cannot ", "unable to", "not permitted",
    "timed out", "timeout", "connection refused", "segmentation fault", "killed",
)
# "a check / verification succeeded" — kept conservative (meaningful milestones, NOT every
# exit-0) so it doesn't become a firehose. Generalized beyond tests.
VERIFY_MARKERS = (
    "passed", "0 failed", "all tests pass", "tests passed", "=== passed",
    "succeeded", "verification passed", "validation passed", "check passed",
    "no errors", "build succeeded",
)
TESTPASS_MARKERS = VERIFY_MARKERS  # backward-compat alias


def has_error(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in ERROR_MARKERS)


def has_verified(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in VERIFY_MARKERS)


def has_testpass(text: str) -> bool:  # backward-compat alias
    return has_verified(text)


READ_ONLY_BASH = re.compile(r"^\s*(cat|ls|grep|find|head|tail|pwd|cd|echo|which|wc|tree|git (status|log|diff|show))\b")


def load_steps(traj: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split a deserialized OpenHands trajectory into (head, steps).

    head  = [system, first-user]  (the Q anchor segment)
    steps = list of {actions:[tool names], thought, obs_text, edits, plans, raw:[msgs]}
    """
    head = []
    i = 0
    # system
    while i < len(traj) and traj[i]["role"] == "system":
        head.append(traj[i]); i += 1
    # first user (task spec)
    if i < len(traj) and traj[i]["role"] == "user":
        head.append(traj[i]); i += 1

    steps: list[dict] = []
    cur = None
    for msg in traj[i:]:
        role = msg["role"]
        if role == "assistant":
            if cur is not None:
                steps.append(cur)
            actions, plan_call, file_edit, readonly = [], False, False, True
            for tc in (msg.get("tool_calls") or []):
                fn = tc.get("function") or {}
                name = fn.get("name")
                args = fn.get("arguments") or {}
                actions.append(name)
                if name == "task_tracker" and isinstance(args, dict) and args.get("command") == "plan":
                    plan_call = True
                if name == "str_replace_editor" and isinstance(args, dict) and args.get("command") in ("create", "str_replace", "insert"):
                    file_edit = True
                    readonly = False
                if name == "str_replace_editor" and isinstance(args, dict) and args.get("command") == "view":
                    pass  # read-only
                if name == "execute_bash" and isinstance(args, dict):
                    cmd = (args.get("command") or "").strip()
                    if cmd and not READ_ONLY_BASH.match(cmd):
                        readonly = False
            cur = {
                "thought": msg.get("content") or "",
                "actions": actions,
                "plan_call": plan_call,
                "file_edit": file_edit,
                "readonly": readonly,
                "obs_text": "",
                "raw": [msg],
            }
        else:  # tool / user observation belongs to current step
            if cur is None:
                # stray observation before any assistant; attach to head
                head.append(msg)
                continue
            cur["obs_text"] += "\n" + (msg.get("content") or "")
            cur["raw"].append(msg)
    if cur is not None:
        steps.append(cur)
    return head, steps


def text_of(msg: dict) -> str:
    parts = [msg.get("content") or ""]
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        parts.append(fn.get("name") or "")
        args = fn.get("arguments")
        parts.append(args if isinstance(args, str) else json.dumps(args, ensure_ascii=False))
    return "\n".join(p for p in parts if p)


def batch_token_lens(tok, texts: list[str]) -> list[int]:
    """Tokenize a whole trajectory's messages in one batched call (fast)."""
    if not texts:
        return []
    enc = tok(texts, add_special_tokens=False)["input_ids"]
    return [len(ids) for ids in enc]


def detect_folds(head, steps, step_tokens, args) -> list[dict]:
    """Return accepted fold points with B2 segmentation metadata."""
    n = len(steps)
    head_tok = sum(step_tokens["head"])
    step_tok = step_tokens["steps"]

    # error run tracking
    err_flags = [has_error(s["obs_text"]) for s in steps]

    # Single forward pass: decide accept inline so we can RESET the expansion
    # accumulator at each accepted fold (context returns to Q+M+recent, not the
    # full append-only history). min-gap + the last-recent-k guard are enforced
    # here too.
    accepted = []
    last_fold = -10**9
    tokens_since_fold = head_tok  # Q is always present after a fold (so does M, approx by head)
    consec_err = 0
    seen_action = False
    max_fold_idx = n - args.recent_k - 1  # need recent_k steps to remain after fold

    for idx in range(n):
        tokens_since_fold += step_tok[idx]
        signal = None
        state_change = not steps[idx]["readonly"]  # tool-agnostic: any mutating/state-changing step

        # (b) structural — DOMAIN-GENERAL semantics (not SWE tool names):
        if steps[idx]["plan_call"]:
            signal = "structural:plan"                 # explicit plan step (fires only if such a tool exists)
        elif has_verified(steps[idx]["obs_text"]) and not err_flags[idx]:
            signal = "structural:verified"             # a check/verification succeeded (was testpass)
        elif state_change and not seen_action:
            signal = "structural:first_action"         # exploration -> first state-changing action (was first_write)

        # (c) error-recovery: previous steps had a run of errors, this one clean
        if signal is None and consec_err >= args.err_run and not err_flags[idx]:
            signal = "error_recovery"

        # (a) expansion (since the last accepted fold) — fully domain-general budget trigger
        if signal is None and tokens_since_fold >= args.t_fold:
            signal = "expansion"

        # update running state regardless of acceptance
        if state_change:
            seen_action = True
        consec_err = consec_err + 1 if err_flags[idx] else 0

        # accept?
        if (signal is not None
                and idx <= max_fold_idx
                and idx - last_fold >= args.min_gap):
            accepted.append({"step_idx": idx, "signal": signal})
            last_fold = idx
            tokens_since_fold = head_tok  # reset: Q (+M) persist, raw history dropped

    # B2 segmentation per accepted fold
    plans = []
    prev_end = -1  # exclusive end of previous compressible region (= prev fold idx)
    for f in accepted:
        ai = f["step_idx"]
        recent_lo = max(prev_end + 1, ai - args.recent_k + 1)
        comp_lo = prev_end + 1
        comp_hi = recent_lo - 1  # inclusive
        comp_tokens = sum(step_tok[comp_lo:comp_hi + 1]) if comp_hi >= comp_lo else 0
        plans.append({
            "step_idx": ai,
            "signal": f["signal"],
            "q_msgs": len(head),
            "compressible_steps": [comp_lo, comp_hi] if comp_hi >= comp_lo else None,
            "compressible_tokens": comp_tokens,
            "recent_steps": [recent_lo, ai],
        })
        prev_end = ai
    return plans


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="/data/liqingyang/research/cat-teminal/project/data/raw/tbase_pool_min50.jsonl")
    ap.add_argument("--out", default="/data/liqingyang/research/cat-teminal/project/data/fold_plans/fold_plans.jsonl")
    ap.add_argument("--reports-dir", default="/data/liqingyang/research/cat-teminal/project/reports")
    ap.add_argument("--tokenizer", default="/data/liqingyang/models/Qwen3.6-27B")
    ap.add_argument("--t-fold", type=int, default=16000, help="expansion threshold (tokens since last fold)")
    ap.add_argument("--min-gap", type=int, default=8)
    ap.add_argument("--recent-k", type=int, default=6, help="recent steps kept verbatim (I^k)")
    ap.add_argument("--err-run", type=int, default=2, help="consecutive error steps to arm error-recovery")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    print(f"[tokenizer] loading {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.reports_dir).mkdir(parents=True, exist_ok=True)
    out_f = None if args.no_write else out_path.open("w")

    n_traj = 0
    fold_counts: list[int] = []
    signal_counter: Counter[str] = Counter()
    comp_ratio_samples: list[float] = []  # compressible_tokens / total for context

    with open(args.pool) as f:
        for line in f:
            if args.limit and n_traj >= args.limit:
                break
            rec = json.loads(line)
            traj = rec["trajectory"]
            head, steps = load_steps(traj)
            if not steps:
                continue
            # one batched tokenize call per trajectory
            texts = [text_of(m) for m in head]
            step_spans = []  # (lo, hi) into the flat lens list, per step
            for s in steps:
                lo = len(texts)
                texts.extend(text_of(m) for m in s["raw"])
                step_spans.append((lo, len(texts)))
            lens = batch_token_lens(tok, texts)
            head_tok = lens[:len(head)]
            step_tok = [sum(lens[lo:hi]) for lo, hi in step_spans]
            step_tokens = {"head": head_tok, "steps": step_tok}
            plans = detect_folds(head, steps, step_tokens, args)

            n_traj += 1
            fold_counts.append(len(plans))
            for p in plans:
                signal_counter[p["signal"]] += 1
                if p["compressible_tokens"]:
                    comp_ratio_samples.append(p["compressible_tokens"])

            if out_f is not None:
                out_f.write(json.dumps({
                    "trajectory_id": rec["trajectory_id"],
                    "instance_id": rec["instance_id"],
                    "num_steps": len(steps),
                    "head_tokens": sum(head_tok),
                    "total_step_tokens": sum(step_tok),
                    "fold_points": plans,
                }, ensure_ascii=False) + "\n")

            if n_traj % 200 == 0:
                avg = sum(fold_counts) / len(fold_counts)
                print(f"  ... traj={n_traj} avg_folds={avg:.2f}", flush=True)

    if out_f is not None:
        out_f.close()

    avg_folds = sum(fold_counts) / max(1, len(fold_counts))
    dist = Counter(fold_counts)
    stats = {
        "trajectories": n_traj,
        "avg_folds_per_traj": round(avg_folds, 3),
        "paper_anchor_folds": 4.22,
        "fold_count_distribution": dict(sorted(dist.items())),
        "signal_breakdown": dict(signal_counter.most_common()),
        "params": {
            "t_fold": args.t_fold, "min_gap": args.min_gap,
            "recent_k": args.recent_k, "err_run": args.err_run,
        },
        "compressible_tokens_avg": round(sum(comp_ratio_samples) / max(1, len(comp_ratio_samples)), 1),
    }
    stats_path = Path(args.reports_dir) / "stage_b1_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))

    print("\n===== Stage B1/B2 summary =====")
    print(f"trajectories       : {n_traj}")
    print(f"avg folds / traj   : {avg_folds:.2f}   (paper anchor ~4.22, scaled by traj length)")
    print(f"fold-count dist    : {dict(sorted(dist.items()))}")
    print(f"signal breakdown   : {dict(signal_counter.most_common())}")
    print(f"avg compressible tokens/fold : {stats['compressible_tokens_avg']}")
    print(f"\nstats  -> {stats_path}")
    if out_f is not None:
        print(f"plans  -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
