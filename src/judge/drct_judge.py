# -*- coding: utf-8 -*-
# drct_judge.py -- DRCT text-arm judging (Lean ground truth + e0a N1 + N2; same gate for all; protocol §4)
# Input: attempts JSONL from drct_player.py; output: records JSONL + summary JSON.
# A byte-identical proof within one arm+problem is judged once (dedup, value reuse).
# File names are unique per problem, so this is parallel-safe.
# Usage:
#   python scratch/drct/drct_judge.py --attempts scratch/drct/runs/smollm_cot_attempts_p8.jsonl \
#       --tag smollm_cot_p8 --workdir scratch/drct/lean/smollm_cot_p8 \
#       --out scratch/drct/runs/smollm_cot_p8 [--jobs 48]
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
os.chdir(RES)
sys.path.insert(0, os.path.join(RES, "scratch", "drct"))
import drct_common as DC  # noqa: E402

JUDGE_SCHEMA = "judge"
DEFAULT_JUDGE_SUFFIX = "judge_summary"   # gen11/Option A: judge-specific suffix (legacy name = empty string)


def judge_summary_path(out, suffix):
    """Judge summary target path: <out>_<suffix>.json; suffix="" => legacy name <out>_summary.json (rollback)."""
    return out + ("_%s.json" % suffix if suffix else "_summary.json")


def guarded_write_json(path, doc):
    """fail-closed: target exists and is not judge schema (no track_full) => refuse to write + record md5 of the file it would have overwritten."""
    if os.path.exists(path):
        old = None
        try:
            old = json.load(open(path, encoding="utf-8"))
        except Exception:
            old = None
        if old is not None and "track_full" not in old:
            h = hashlib.md5(open(path, "rb").read()).hexdigest()
            print("[judge] ABORT 目标已存在且非判卷 schema: %s md5=%s (拒写; 防覆盖 runner/装配件)"
                  % (path, h), flush=True)
            raise SystemExit(3)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)


_G = {}


def _init(attempts_path, workdir, tag, timeout):
    probs = DC.load_problems()
    _G["probs"] = {p["name"]: p for p in probs}
    _G["strata"] = {p["name"]: p["stratum"] for p in probs}
    _G["workdir"], _G["tag"], _G["timeout"] = workdir, tag, timeout
    _G["by_prob"] = {}
    for line in open(attempts_path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            _G["by_prob"].setdefault(r["problem"], []).append(r)


def _judge_problem(name):
    p = _G["probs"][name]
    rec = {"problem": name, "stratum": _G["strata"].get(name), "attempts": {}, "n_dedup": 0}
    seen, calls = {}, 0
    for r in sorted(_G["by_prob"][name], key=lambda x: x["attempt"]):
        k = str(r["attempt"])
        proof = (r.get("proof") or "").strip("\n")
        h = hashlib.md5(proof.encode("utf-8")).hexdigest()
        if h in seen:
            k0, v = seen[h]
            rec["attempts"][k] = dict(v)
            rec["attempts"][k]["dedup_of"] = k0
            rec["n_dedup"] += 1
            continue
        j = DC.judge_attempt(name, p["head"], p["concl"], proof, _G["workdir"],
                             "%s_a%s" % (_G["tag"], k), _G["timeout"])
        calls += 1 + (2 if j["lean"] and j["lean"]["accepted"] else 0)
        rec["attempts"][k] = j
        if proof:
            seen[h] = (k, j)
    rec["lean_calls"] = calls
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--judge-suffix", default=DEFAULT_JUDGE_SUFFIX,
                    help="判卷 summary 后缀 (缺省 judge_summary; 置空串 '' = 旧名, 供回滚)")
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    _init(a.attempts, a.workdir, a.tag, a.timeout)
    names = [n for n in _G["probs"] if n in _G["by_prob"]]
    if a.limit:
        names = names[:a.limit]
    records = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(_judge_problem, n): n for n in names}
        done = 0
        for f in as_completed(futs):
            records.append(f.result())
            done += 1
            if done % 20 == 0:
                print("[drct-judge] %d/%d" % (done, len(names)), flush=True)
    order = {n: i for i, n in enumerate(_G["probs"])}
    records.sort(key=lambda r: order.get(r["problem"], 1 << 30))
    with open(a.out + "_records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ = DC.summarize(records)
    summ["tag"] = a.tag
    summ["lean_calls_total"] = sum(r["lean_calls"] for r in records)
    summ["schema"] = JUDGE_SCHEMA
    summ["wrote_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    sp = judge_summary_path(a.out, a.judge_suffix)
    guarded_write_json(sp, summ)
    print("[drct-judge] summary -> %s" % sp, flush=True)
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
