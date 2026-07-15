#!/usr/bin/env python3
"""Recall check for M — two complementary views, both reported.

(1) ACTION-STATE recall — history that mechanically drives the next action:
      * EDITED FILES (str_replace_editor create/str_replace/insert) — forgetting one
        => agent re-does or conflicts with an edit. Most critical.
      * RUN COMMANDS (non-readonly execute_bash) — esp. test/reproduce commands.
      * ERROR signs (Exception/Error tokens in observations being addressed).

(2) THOUGHT/ACTION IDENTIFIER recall — every identifier the agent itself wrote in
    its Thought or Action (paths, `backtick symbols`, dotted names, errors), split:
      * decision ids   = appear in Thought/Action  -> SHOULD be kept in M
      * observation-only = appear only in tool output -> ok to drop (filler)

Low recall on (1) or on decision ids = real loss. Low observation-only recall is
correct (that is the dump we compress away). Dropped decision items are printed.
"""
from __future__ import annotations
import argparse, json, re, random, sys
sys.path.insert(0, "catgen")
from stage_b12_fold_points import load_steps
import stage_b3_generate_memory as b3

PATH = re.compile(r'[A-Za-z0-9_./-]+\.(?:py|cfg|yml|yaml|json|toml|cpp|cc|hpp|js|ts|go|rs|java|sh|rst|txt|ini|html)(?::\d+(?:-\d+)?)?')
BTICK = re.compile(r'`([A-Za-z_][A-Za-z0-9_./]{2,60})`')
DOTTED = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]+(?:\.[A-Za-z_][A-Za-z0-9_]+)+)\b')
ERRTOK = re.compile(r'\b([A-Z][A-Za-z0-9_]*(?:Error|Exception))\b')
READONLY_CMD = re.compile(r'^\s*(cat|ls|cd|grep|find|head|tail|pwd|echo|which|wc|tree|less|more|sed -n)\b')

def ids_of(text):
    out = set()
    for m in PATH.finditer(text): out.add(m.group(0).split(':')[0])
    for m in BTICK.finditer(text): out.add(m.group(1).split('(')[0])
    for m in DOTTED.finditer(text): out.add(m.group(1))
    for m in ERRTOK.finditer(text): out.add(m.group(1))
    return {x for x in out if len(x) >= 3}

def _args(tc):
    fn = tc.get("function") or {}
    a = fn.get("arguments")
    if isinstance(a, str):
        try: a = json.loads(a)
        except Exception: a = {}
    return fn.get("name"), (a or {})

def step_views(steps, lo, hi):
    ta, obs = [], []
    edited, cmds, errs = set(), set(), set()
    for i in range(lo, hi + 1):
        st = steps[i]
        if st.get("thought"): ta.append(st["thought"])
        for m in st["raw"]:
            if m.get("role") == "assistant":
                for tc in (m.get("tool_calls") or []):
                    name, a = _args(tc)
                    ta.append(name + "(" + json.dumps(a, ensure_ascii=False) + ")")
                    if name == "str_replace_editor" and a.get("command") in ("create","str_replace","insert"):
                        if a.get("path"): edited.add(str(a["path"]))
                    elif name == "execute_bash":
                        c = (a.get("command") or "").strip()
                        if c and not READONLY_CMD.match(c): cmds.add(c[:60])
        o = st.get("obs_text") or ""
        obs.append(o)
        for e in ERRTOK.findall(o): errs.add(e)
    return "\n".join(ta), "\n".join(obs), edited, cmds, errs

def hit(item, M, Ml):
    base = item.split("/")[-1]
    return (item in M) or (base and (base in M or base.lower() in Ml))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default="data/raw/tbase_pool_min50.jsonl")
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args()

    print("[load] pool ...", flush=True)
    pool = {}
    for l in open(args.pool):
        r = json.loads(l); pool[r["trajectory_id"]] = r["trajectory"]
    rows = [json.loads(l) for l in open(args.out)]
    random.seed(0); random.shuffle(rows)

    edit_rec, cmd_rec, err_rec = [], [], []
    dec_rec, obs_rec = [], []
    n = 0; drop_edit = []; drop_dec = []
    for r in rows:
        if n >= args.sample: break
        traj = pool.get(r["trajectory_id"])
        if traj is None: continue
        head, steps = load_steps(traj)
        for f in r["fold_points"]:
            if n >= args.sample: break
            cs = f.get("compressible_steps")
            if not cs: continue
            lo, hi = cs
            try:
                ta_t, obs_t, edited, cmds, errs = step_views(steps, lo, hi)
            except Exception:
                continue
            M = f["M"]; Ml = M.lower(); n += 1
            if edited:
                miss = [e for e in edited if not hit(e, M, Ml)]
                edit_rec.append(1 - len(miss)/len(edited))
                if miss and len(drop_edit) < args.show:
                    drop_edit.append((f.get("signal"), sorted(edited), sorted(miss)))
            if cmds: cmd_rec.append(sum(1 for c in cmds if hit(c, M, Ml))/len(cmds))
            if errs: err_rec.append(sum(1 for e in errs if e in M)/len(errs))
            dec_ids = ids_of(ta_t)
            obs_only = ids_of(obs_t) - dec_ids
            if dec_ids:
                miss = sorted(i for i in dec_ids if not hit(i, M, Ml))
                dec_rec.append(1 - len(miss)/len(dec_ids))
                if miss and len(drop_dec) < args.show:
                    drop_dec.append((round(1-len(miss)/len(dec_ids),2), f.get("signal"), miss[:10]))
            if obs_only:
                obs_rec.append(sum(1 for i in obs_only if hit(i, M, Ml))/len(obs_only))

    import statistics as st
    def line(name, arr):
        if not arr: return name + ": (none)"
        return (name + ": mean=%.2f median=%.2f | fully-kept=%d%% | n=%d" %
                (st.mean(arr), st.median(arr), 100*sum(1 for x in arr if x>=0.999)/len(arr), len(arr)))
    print("\n=== (1) ACTION-STATE recall (sample %d folds) ===" % n)
    print("  " + line("EDITED FILES ", edit_rec))
    print("  " + line("RUN COMMANDS ", cmd_rec))
    print("  " + line("ERROR signs  ", err_rec))
    print("\n=== (2) THOUGHT/ACTION IDENTIFIER recall ===")
    print("  " + line("decision ids (should keep)        ", dec_rec))
    print("  " + line("observation-only ids (ok to drop) ", obs_rec))
    print("\n--- folds dropping an EDITED file ---")
    for sig, ed, miss in drop_edit:
        print("  sig=%s dropped=%s (of %s)" % (sig, miss, ed))
    print("\n--- folds with lowest decision-id recall ---")
    for rec, sig, miss in sorted(drop_dec)[:args.show]:
        print("  recall=%.2f sig=%s dropped=%s" % (rec, sig, miss))

if __name__ == "__main__":
    main()
