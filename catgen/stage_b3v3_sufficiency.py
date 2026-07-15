#!/usr/bin/env python3
"""Stage B3-v3 — M-SUFFICIENCY rejection sampling (idea #1).

Counterfactual test: does M actually carry enough to CONTINUE, or did the model just
learn to write pretty-but-useless summaries? For each fold compressing steps [lo,hi],
we compare the STUDENT model's negative log-likelihood of the SAME gold continuation
C (the next few decision steps) under two contexts that share an identical prefix:

    RAW    = (prefix, raw steps[lo..hi], C)          # full history
    FOLDED = (prefix, context-call(M), ack, C)       # M replaces [lo..hi]

    ΔNLL_per_tok = ( NLL(C | FOLDED) − NLL(C | RAW) ) / (assistant tokens in C)

ΔNLL ≈ 0  → folding lost no predictive info → M is a SUFFICIENT statistic → KEEP.
ΔNLL ≫ 0  → M dropped something the continuation needs → INSUFFICIENT → drop (or, with
            --repair, ask the summarizer to add back exactly what was missing, re-score).

This is a Monte-Carlo estimate of the conditional-MI the fold destroys: I(C;raw)−I(C;M).
Scored with the SAME base you will fine-tune (point --base-url at it) so sufficiency is
calibrated to THAT model's needs, not a stronger teacher's.

Runs on 79 (needs the student model + its tokenizer + prompt_logprobs from vLLM).
Loss mask (assistant-only) reused from train/cat_dataset; folded rep reused from
stage_b4 so RAW/FOLDED/C are rendered exactly as in training/inference.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from transformers import AutoTokenizer

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent / "train"))
from stage_b12_fold_points import load_steps            # noqa: E402
from stage_b4_stitch_cat_instruct import context_messages, CONTEXT_TOOL  # noqa: E402
from cat_dataset import make_masker                      # noqa: E402
from openai import OpenAI                                # noqa: E402
import stage_b3_generate_memory as b3                    # noqa: E402
import stage_b3v2_forward_memory as v2                   # noqa: E402


def render_ids(tok, msgs, tools):
    # return_dict=True + ["input_ids"] is REQUIRED: without it, apply_chat_template(tokenize=True)
    # returns a BatchEncoding (len==2), not a flat id list — silently breaking all scoring.
    return tok.apply_chat_template(
        msgs, tools=tools, tokenize=True, return_dict=True,
        add_generation_prompt=False, enable_thinking=False)["input_ids"]


def common_prefix_len(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def score_nll(session, url, model, full_ids, c_start, mask_labels, timeout):
    """Sum NLL over the ASSISTANT tokens of C (positions >= c_start). Returns
    (nll, n_tokens) or (None, 0) if unscorable."""
    labels = mask_labels(full_ids)
    idxs = [i for i in range(c_start, len(full_ids)) if i < len(labels) and labels[i] != -100]
    if not idxs:
        return None, 0
    body = {"model": model, "prompt": full_ids, "max_tokens": 1, "temperature": 0,
            "prompt_logprobs": 0}
    r = session.post(f"{url}/completions", json=body, timeout=timeout)
    r.raise_for_status()
    pls = r.json()["choices"][0]["prompt_logprobs"]  # list, one dict per prompt token
    nll = 0.0
    for i in idxs:
        entry = pls[i] if i < len(pls) else None
        if not entry:
            return None, 0
        tid = str(full_ids[i])
        lp = entry.get(tid, {}).get("logprob") if isinstance(entry, dict) else None
        if lp is None:
            # vLLM keys prompt_logprobs by the actual token id; if missing, bail
            return None, 0
        nll -= lp
    return nll, len(idxs)


def build_contexts(tok, tools, head, steps, fp, horizon):
    """Return (raw_full_ids, raw_c_start, folded_full_ids, folded_c_start) or None."""
    lo, hi = fp["compressible_steps"]
    if hi + 1 >= len(steps):
        return None  # nothing to continue into -> can't test sufficiency
    prefix = list(head)
    for i in range(0, lo):
        prefix.extend(steps[i]["raw"])
    slice_msgs = []
    for i in range(lo, hi + 1):
        slice_msgs.extend(steps[i]["raw"])
    fold_msgs = context_messages(fp["M"], fp.get("signal", ""))
    cont_msgs = []
    for i in range(hi + 1, min(len(steps), hi + 1 + horizon)):
        cont_msgs.extend(steps[i]["raw"])
    raw_ctx = prefix + slice_msgs
    folded_ctx = prefix + fold_msgs
    raw_full = render_ids(tok, raw_ctx + cont_msgs, tools)
    raw_only = render_ids(tok, raw_ctx, tools)
    fold_full = render_ids(tok, folded_ctx + cont_msgs, tools)
    fold_only = render_ids(tok, folded_ctx, tools)
    return (raw_full, common_prefix_len(raw_only, raw_full),
            fold_full, common_prefix_len(fold_only, fold_full))


# ---------------------------------------------------------------------------
# Repair closed loop (idea #1's second half): when a fold is insufficient, reveal
# the gold continuation C to the STRONG teacher and ask it to REWRITE M adding back
# exactly the facts C needs — then re-score. Steers CONTENT, not size (budget alone
# doesn't help: the teacher self-limits M at ~0.2). Anti-cheat: M must not copy C's
# action strings verbatim (that would trivially lower NLL but destroy the compression).
# ---------------------------------------------------------------------------
def fold_segments(head, steps, fp, horizon, prefix_steps=-1):
    """(prefix, slice_msgs, cont_msgs) message segments, or None if nothing to continue into.

    prefix_steps: keep only Q + the last `prefix_steps` steps before the slice (drop the far
    history). Full history (prefix_steps<0) makes prompt_logprobs ~1-concurrency / days on long
    contexts; bounding it is BOTH tractable AND more faithful to recursive inference (where the
    far history is itself folded away), isolating the SLICE's contribution to the future."""
    lo, hi = fp["compressible_steps"]
    if hi + 1 >= len(steps):
        return None
    prefix = list(head)
    p0 = 0 if prefix_steps < 0 else max(0, lo - prefix_steps)
    for i in range(p0, lo):
        prefix.extend(steps[i]["raw"])
    slice_msgs = []
    for i in range(lo, hi + 1):
        slice_msgs.extend(steps[i]["raw"])
    cont_msgs = []
    for i in range(hi + 1, min(len(steps), hi + 1 + horizon)):
        cont_msgs.extend(steps[i]["raw"])
    return prefix, slice_msgs, cont_msgs


REPAIR_TEMPLATE = """# Task
{task}

# Fold location
You are rewriting the long-term memory that compresses trajectory steps {lo}-{hi}
(this raw history is REMOVED after compression).

# Raw trajectory slice (steps {lo}-{hi}) — the history being compressed
{slice_text}

# What the agent does NEXT (the gold continuation your memory MUST be able to support)
{cont_text}

# Why you are rewriting
Your PREVIOUS memory was INSUFFICIENT: a base model could NOT reproduce the next steps
from it, so it dropped facts those steps depend on. Rewrite the seven-section memory and
ADD BACK the specific facts FROM THE SLICE that the next steps rely on — exact paths, line
numbers, error text, commands already run and their results, and confirmed root-cause
conclusions.

Hard rules:
- Do NOT copy the next steps' concrete commands, edits, diffs, or code verbatim (especially
  into [Next]); [Next] stays PLAN-level. The agent must still choose the actions itself.
- Add only facts grounded in the slice. Keep all seven headers verbatim.

# Output
Return exactly the seven-section memory block and nothing else."""


def regen_M(client, args, task, cs, slice_text, cont_text):
    lo, hi = cs
    prompt = REPAIR_TEMPLATE.format(task=task[:8000], lo=lo, hi=hi,
                                    slice_text=slice_text[:48000], cont_text=cont_text)
    out_cap = args.max_m_tokens + 768
    m = b3.summarize(client, args.teacher_model, v2.SUMMARIZER_SYSTEM_V2, prompt,
                     max_tokens=out_cap, temperature=args.repair_temperature, extra_body=b3.NO_THINK)
    if not all(h in m for h in v2._REQUIRED_HEADERS):
        m = b3.summarize(client, args.teacher_model, v2.SUMMARIZER_SYSTEM_V2,
                         prompt + "\n\nInclude ALL seven headers verbatim; keep [Next] plan-level.",
                         max_tokens=out_cap, temperature=args.repair_temperature, extra_body=b3.NO_THINK)
    return m


def leaks_C(M, cont_msgs):
    """True if M verbatim-copies a substantial action string from the continuation
    (command / edit / new_str values) — anti-cheat against trivially lowering NLL."""
    cands = []
    for m in cont_msgs:
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            a = (tc.get("function") or {}).get("arguments", "")
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:  # noqa: BLE001
                    cands.append(a); continue
            if isinstance(a, dict):
                cands.extend(v for v in a.values() if isinstance(v, str))
            else:
                cands.append(str(a))
    for c in cands:
        for line in c.splitlines():
            s = line.strip()
            if len(s) >= 25 and s in M:
                return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-m", default="data/fold_plans/fold_plans_with_M_v2_a3b.jsonl",
                    help="folds with M (v1 or v2); this is what we score")
    ap.add_argument("--pool", default="data/raw/tbase_pool_min50.jsonl")
    ap.add_argument("--out", default="data/fold_plans/fold_plans_scored.jsonl")
    ap.add_argument("--tools", default="data/cat_instruct/tools_a3b.json")
    ap.add_argument("--tokenizer", default="/data/liqingyang/models/Qwen3.6-27B",
                    help="MUST be the STUDENT model's tokenizer (the one at --base-url)")
    ap.add_argument("--base-url", default="http://localhost:8000/v1",
                    help="STUDENT model endpoint (vLLM, prompt_logprobs) — the model you fine-tune")
    ap.add_argument("--model", default="cat")
    ap.add_argument("--tau", type=float, default=0.15,
                    help="accept fold if ΔNLL/token <= tau (calibrate on a held-out slice)")
    ap.add_argument("--horizon", type=int, default=4, help="continuation steps to score")
    ap.add_argument("--prefix-steps", type=int, default=4,
                    help="keep Q + last N steps before the slice (drop far history); -1 = full history. "
                         "Bounding is required for tractable long-context prompt_logprobs.")
    ap.add_argument("--max-ctx", type=int, default=60000,
                    help="skip folds whose raw context exceeds this many tokens")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--drop-insufficient", action="store_true",
                    help="write only sufficient folds (default: keep all, annotate)")
    # --- repair closed loop (idea #1 second half) ---
    ap.add_argument("--repair", type=int, default=0,
                    help="max targeted-regeneration rounds on insufficient folds (0=off)")
    ap.add_argument("--teacher-url", default="",
                    help="STRONG summarizer endpoint for --repair (empty=off, just score)")
    ap.add_argument("--teacher-model", default="qwen27b")
    ap.add_argument("--teacher-key", default="EMPTY")
    ap.add_argument("--max-m-tokens", type=int, default=4000, help="repair M output cap")
    ap.add_argument("--cont-obs-cap", type=int, default=800,
                    help="per-step obs char cap when showing C to the repair teacher")
    ap.add_argument("--repair-temperature", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    tools = json.load(open(args.tools))
    if not any((t.get("function", t).get("name")) == "context" for t in tools):
        tools = tools + [CONTEXT_TOOL]
    mask_labels, _, _ = make_masker(tok)

    teacher_client = None
    if args.repair > 0 and args.teacher_url:
        teacher_client = OpenAI(base_url=args.teacher_url, api_key=args.teacher_key,
                                max_retries=5, timeout=600)
        print(f"[repair] ON: up to {args.repair} rounds via {args.teacher_model} @ {args.teacher_url}",
              flush=True)

    print(f"[load] indexing pool {args.pool}", flush=True)
    pool = {}
    with open(args.pool) as f:
        for line in f:
            r = json.loads(line); pool[r["trajectory_id"]] = r["trajectory"]
    print(f"[load] pool indexed: {len(pool)}", flush=True)

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.open():
            try: done.add(json.loads(line)["trajectory_id"])
            except Exception: pass
        print(f"[resume] {len(done)} done", flush=True)

    session = httpx.Client(trust_env=False)
    write_lock = threading.Lock(); stat_lock = threading.Lock()
    g = out_path.open("a")
    stat = {"folds": 0, "suff": 0, "insuff": 0, "skipped": 0, "repaired": 0,
            "recovered": 0, "dnll": []}
    t0 = time.time()

    def _folded_nll(prefix, cont_msgs, M, signal, tr_ref):
        """NLL of C under FOLDED (prefix + ctx(M) + C), or None if unscorable / C-token drift."""
        fold_msgs = context_messages(M, signal)
        fold_full = render_ids(tok, prefix + fold_msgs + cont_msgs, tools)
        fold_only = render_ids(tok, prefix + fold_msgs, tools)
        fold_cs = common_prefix_len(fold_only, fold_full)
        nf, tf = score_nll(session, args.base_url, args.model, fold_full, fold_cs, mask_labels, 300)
        if nf is None or tf == 0 or tf != tr_ref:
            return None
        return nf

    def score_fold(head, steps, fp, task):
        seg = fold_segments(head, steps, fp, args.horizon, args.prefix_steps)
        if seg is None:
            return None
        prefix, slice_msgs, cont_msgs = seg
        raw_full = render_ids(tok, prefix + slice_msgs + cont_msgs, tools)
        if len(raw_full) > args.max_ctx:
            return "skip"
        raw_only = render_ids(tok, prefix + slice_msgs, tools)
        raw_cs = common_prefix_len(raw_only, raw_full)
        nr, tr = score_nll(session, args.base_url, args.model, raw_full, raw_cs, mask_labels, 300)
        if nr is None or tr == 0:
            return "skip"
        signal = fp.get("signal", "")
        M = fp["M"]
        nf = _folded_nll(prefix, cont_msgs, M, signal, tr)
        if nf is None:
            return "skip"  # C-token drift between raw/folded rendering
        dnll = (nf - nr) / tr

        # --- repair closed loop: reveal C to the teacher, re-write M, re-score ---
        repaired = 0
        if dnll > args.tau and teacher_client is not None:
            lo, hi = fp["compressible_steps"]
            slice_text = "\n\n".join(b3.render_step(steps[i]) for i in range(lo, hi + 1))
            cont_text = v2.render_continuation(steps, hi, args.horizon, args.cont_obs_cap)
            while dnll > args.tau and repaired < args.repair:
                try:
                    newM = regen_M(teacher_client, args, task, (lo, hi), slice_text, cont_text)
                except Exception:  # noqa: BLE001 — teacher hiccup: stop repairing this fold
                    break
                if not newM or leaks_C(newM, cont_msgs):
                    break
                nf2 = _folded_nll(prefix, cont_msgs, newM, signal, tr)
                if nf2 is None:
                    break
                repaired += 1
                M, nf, dnll = newM, nf2, (nf2 - nr) / tr
                if dnll <= args.tau:
                    break

        out = {"delta_nll_per_tok": round(dnll, 4), "nll_raw": round(nr, 2),
               "nll_folded": round(nf, 2), "c_tokens": tr, "repaired": repaired,
               "sufficient": bool(dnll <= args.tau)}
        if repaired:
            out["M"] = M  # repaired memory supersedes the original in the output fold
        return out

    def process_traj(plan):
        tid = plan["trajectory_id"]
        traj = pool.get(tid)
        if traj is None:
            return
        head, steps = load_steps(traj)
        task = b3.task_text(head)
        out_folds = []
        for fp in plan["fold_points"]:
            if not fp.get("compressible_steps") or not fp.get("M"):
                continue
            try:
                res = score_fold(head, steps, fp, task)
            except Exception as exc:  # noqa: BLE001
                res = None
            with stat_lock:
                stat["folds"] += 1
                if res == "skip" or res is None:
                    stat["skipped"] += 1
                else:
                    stat["dnll"].append(res["delta_nll_per_tok"])
                    if res.get("repaired"):
                        stat["repaired"] += 1
                        if res["sufficient"]:
                            stat["recovered"] += 1
                    stat["suff" if res["sufficient"] else "insuff"] += 1
            if isinstance(res, dict):
                fp = {**fp, **res}
                if args.drop_insufficient and not res["sufficient"]:
                    continue
            out_folds.append(fp)
        with write_lock:
            g.write(json.dumps({"trajectory_id": tid,
                                "instance_id": plan.get("instance_id"),
                                "num_steps": plan.get("num_steps"),
                                "fold_points": out_folds}, ensure_ascii=False) + "\n")
            g.flush()
        with stat_lock:
            n = stat["suff"] + stat["insuff"]
            if n and n % 200 < 2:
                rate = n / max(1e-6, time.time() - t0)
                med = sorted(stat["dnll"])[len(stat["dnll"]) // 2] if stat["dnll"] else 0
                print(f"  scored={n} suff={stat['suff']} insuff={stat['insuff']} "
                      f"skip={stat['skipped']} median_dNLL={med:.3f} ({rate:.1f} folds/s)",
                      flush=True)

    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        for line in open(args.with_m):
            if args.limit and n >= args.limit:
                break
            plan = json.loads(line)
            if plan["trajectory_id"] in done:
                continue
            n += 1
            futs.append(ex.submit(process_traj, plan))
        for fu in futs:
            fu.result()
    g.close()
    tot = stat["suff"] + stat["insuff"]
    print("\n===== Stage B3-v3 (M-sufficiency) summary =====")
    print(f"folds scored : {tot} (skipped {stat['skipped']})")
    if tot:
        print(f"SUFFICIENT   : {stat['suff']}/{tot} = {100*stat['suff']/tot:.1f}%  (tau={args.tau})")
        d = sorted(stat["dnll"])
        print(f"ΔNLL/token   : median={d[len(d)//2]:.3f}  p90={d[int(len(d)*0.9)]:.3f}")
    if args.repair:
        print(f"REPAIR       : attempted on {stat['repaired']} folds, "
              f"recovered {stat['recovered']} to sufficient")
    print(f"out -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
