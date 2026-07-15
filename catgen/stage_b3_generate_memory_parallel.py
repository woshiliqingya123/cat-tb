#!/usr/bin/env python3
"""Stage B3 (parallel, hardened) — same faithful M-generation as
stage_b3_generate_memory.py, fanned out to saturate a multi-GPU vLLM endpoint.

Reuses (imported, NOT reimplemented) from stage_b3_generate_memory:
    render_step / task_text / SUMMARIZER_SYSTEM / SUMMARIZER_TEMPLATE /
    summarize / count_tokens
so the memory blocks M are byte-for-byte the same prompts as the reference impl.

Hardening over the first parallel draft:
  1. Bounded in-flight via a semaphore (cap = workers * INFLIGHT_FACTOR) so we do
     NOT submit ~89k futures (each carrying a rendered prompt) at once -> no OOM.
  2. Exception-safe workers + callbacks: a failed fold never raises inside the
     callback; the semaphore is always released and the per-trajectory counter
     always advances. A trajectory is written ONLY if all its folds succeed;
     otherwise its id goes to <out>.failed and it is left for a resume run.
  3. Single shared OpenAI client (httpx is thread-safe; sharing reuses the
     connection pool, which is the recommended pattern).
  4. Absolute cap on the summary budget (--max-m-tokens) so oversize tail
     segments don't produce multi-thousand-token "summaries".
  5. Oversize-slice truncation (head 60% + tail 40%) to fit --max-model-len.
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

INFLIGHT_FACTOR = 2  # max in-flight requests = workers * this


def truncate_slice(tok, slice_text: str, max_in: int) -> tuple[str, bool]:
    ids = tok(slice_text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_in:
        return slice_text, False
    head_n = int(max_in * 0.60)
    tail_n = max_in - head_n
    head = tok.decode(ids[:head_n])
    tail = tok.decode(ids[-tail_n:])
    return head + "\n\n[... trajectory middle elided for length ...]\n\n" + tail, True


def build_fold_job(tok, steps, task, fp, args):
    lo, hi = fp["compressible_steps"]
    slice_text = "\n\n".join(b3.render_step(steps[i]) for i in range(lo, hi + 1))
    seg_tokens = fp["compressible_tokens"]
    # ratio-based budget, but capped absolutely so huge segments still yield a
    # compact M (paper M avg ~4676 tokens).
    budget = max(128, min(int(seg_tokens * args.target_ratio), args.max_m_tokens))
    # Hard ceiling decoupled from the soft budget: generous floor so the model can
    # always finish all four sections; the prompt (not max_tokens) controls length.
    out_cap = min(args.max_m_tokens + 512, max(1024, budget * 2))
    max_in = args.max_model_len - out_cap - 256
    slice_text, truncated = truncate_slice(tok, slice_text, max_in)
    user_prompt = b3.SUMMARIZER_TEMPLATE.format(
        task=task, lo=lo, hi=hi, slice_text=slice_text, budget=budget)
    return {"fp": fp, "user_prompt": user_prompt, "budget": budget,
            "out_cap": out_cap, "seg_tokens": seg_tokens, "truncated": truncated}


def run_fold(client, tok, args, job):
    """Never raises. Returns (out_dict, ratio, ok)."""
    try:
        fp, user_prompt = job["fp"], job["user_prompt"]
        budget, seg_tokens = job["budget"], job["seg_tokens"]
        m = b3.summarize(client, args.model, b3.SUMMARIZER_SYSTEM, user_prompt,
                         max_tokens=job["out_cap"], temperature=args.temperature,
                         extra_body=b3.NO_THINK)
        m_tok = b3.count_tokens(tok, m)
        ratio = m_tok / max(1, seg_tokens)
        if ratio > args.max_ratio:
            strict = user_prompt + (
                f"\n\nYour previous summary was too long. Rewrite it in under "
                f"{budget} tokens. Drop only filler (logs, dir listings, file dumps, "
                f"navigation) — keep every path, symbol, error, command, code edit, "
                f"failed attempt, and constraint.")
            m = b3.summarize(client, args.model, b3.SUMMARIZER_SYSTEM, strict,
                             max_tokens=job["out_cap"], temperature=args.temperature,
                             extra_body=b3.NO_THINK)
            m_tok = b3.count_tokens(tok, m)
            ratio = m_tok / max(1, seg_tokens)
        elif ratio < args.min_ratio and not job["truncated"]:
            # too short: a non-truncated slice compressed below the floor likely
            # dropped detail. Ask for more specifics (skip when truncated, since a
            # truncated slice legitimately compresses low).
            expand = user_prompt + (
                f"\n\nYour previous summary was too short and likely dropped important "
                f"detail. Rewrite it adding back the specific paths, line numbers, "
                f"commands, code edits, test results, and failed attempts from the slice. "
                f"Stay short ONLY if the slice is genuinely mostly duplicated logs or "
                f"directory listings.")
            m = b3.summarize(client, args.model, b3.SUMMARIZER_SYSTEM, expand,
                             max_tokens=job["out_cap"], temperature=args.temperature,
                             extra_body=b3.NO_THINK)
            m_tok = b3.count_tokens(tok, m)
            ratio = m_tok / max(1, seg_tokens)
        out = {**fp, "M": m, "M_tokens": m_tok, "compression_ratio": round(ratio, 3)}
        if job["truncated"]:
            out["slice_truncated"] = True
        return out, ratio, True
    except Exception as exc:  # noqa: BLE001 - must not propagate into callback
        return {"_error": f"{type(exc).__name__}: {exc}"}, None, False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-plans", default="data/fold_plans/fold_plans.jsonl")
    ap.add_argument("--pool", default="data/raw/tbase_pool_min50.jsonl")
    ap.add_argument("--out", default="data/fold_plans/fold_plans_with_M.jsonl")
    ap.add_argument("--tokenizer", default="/data/liqingyang/models/Qwen3.6-27B")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="Qwen3.6-27B")
    ap.add_argument("--target-ratio", type=float, default=0.30)
    ap.add_argument("--max-ratio", type=float, default=0.60,
                    help="regenerate stricter only above this ratio (raised from 0.45 "
                         "so the preserve-first prompt isn't undone)")
    ap.add_argument("--min-ratio", type=float, default=0.18,
                    help="regenerate asking for more specifics below this ratio "
                         "(catches over-compressed dense segments; skipped if truncated)")
    ap.add_argument("--max-m-tokens", type=int, default=6000, help="absolute cap on M budget")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-model-len", type=int, default=32768)
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
    g = out_path.open("a")
    gf = failed_path.open("a")
    ratios: list[float] = []
    n_traj = [0]
    n_folds = [0]
    n_failed_folds = [0]
    t0 = time.time()

    def flush_traj(plan, results: dict, n_jobs: int, any_fail: bool):
        ordered = [results[i] for i in sorted(results) if results[i] is not None]
        if any_fail or len(ordered) < n_jobs:
            with write_lock:
                gf.write(json.dumps({"trajectory_id": plan["trajectory_id"],
                                     "instance_id": plan.get("instance_id")},
                                    ensure_ascii=False) + "\n")
                gf.flush()
            return
        with write_lock:
            g.write(json.dumps({
                "trajectory_id": plan["trajectory_id"],
                "instance_id": plan.get("instance_id"),
                "num_steps": plan["num_steps"],
                "fold_points": ordered,
            }, ensure_ascii=False) + "\n")
            g.flush()
        with stat_lock:
            n_traj[0] += 1
            if n_traj[0] % 25 == 0:
                avg = sum(ratios) / max(1, len(ratios))
                rate_t = n_traj[0] / (time.time() - t0)
                rate_f = n_folds[0] / (time.time() - t0)
                print(f"  ... traj={n_traj[0]} folds={n_folds[0]} "
                      f"failed_folds={n_failed_folds[0]} avg_ratio={avg:.3f} "
                      f"({rate_t:.2f} traj/s, {rate_f:.2f} folds/s)", flush=True)

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
                    flush_traj(plan, {}, 0, any_fail=False)
                    continue

                state = {"remaining": len(jobs), "any_fail": False}
                results: dict[int, dict] = {}
                rlock = threading.Lock()

                def make_cb(idx, plan_ref, state_ref, results_ref, rlock_ref, n_jobs):
                    def _cb(fut):
                        try:
                            out, ratio, ok = fut.result()
                        except Exception as exc:  # belt-and-suspenders
                            out, ratio, ok = {"_error": str(exc)}, None, False
                        finally:
                            sem.release()
                        with stat_lock:
                            n_folds[0] += 1
                            if ok:
                                ratios.append(ratio)
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
                    sem.acquire()  # backpressure: bound in-flight requests
                    fut = ex.submit(run_fold, client, tok, args, job)
                    fut.add_done_callback(
                        make_cb(idx, plan, state, results, rlock, len(jobs)))

    g.close()
    gf.close()
    avg = sum(ratios) / max(1, len(ratios))
    print("\n===== Stage B3 (parallel) summary =====")
    print(f"trajectories written : {n_traj[0]}")
    print(f"folds done           : {n_folds[0]} (failed folds: {n_failed_folds[0]})")
    print(f"avg compression ratio: {avg:.3f}  (target {args.target_ratio}, paper ~0.30)")
    print(f"out     -> {out_path}")
    print(f"failed  -> {failed_path} (rerun to retry these trajectories)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
