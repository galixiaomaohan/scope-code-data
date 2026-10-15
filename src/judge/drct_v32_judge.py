# -*- coding: utf-8 -*-
# drct_v32_judge.py -- DRCT V32-arm judging (chain -> witness three gates -> Lean ground truth + e0a N1 + N2)
# Mirrors the per-attempt render-and-judge flow of scratch/xcmp/xcmp_v32_judge.py (problem source switched to the DRCT test set).
# FileProposer is built from the chains jsonl produced by lc30.build_plan (same form as the judging module's FileProposer).
# Usage:
#   python scratch/drct/drct_v32_judge.py --chains scratch/drct/runs/v32_off_chains.jsonl \
#       --tag v32_off --workdir scratch/drct/lean/v32_off --out scratch/drct/runs/v32_off \
#       [--jobs 32]
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


class P:
    def __init__(self, d):
        self.split = "DRCT"
        self.name = d["name"]
        self.doc = ""
        self.head = d["head"]
        self.binders = d["head"][len("theorem %s " % d["name"]):-1].strip()
        self.concl = d["concl"]
        self.raw = d["stmt"] + "\n  sorry\n"
        self.md5 = d["md5"]


def _init(chains, workdir, tag, timeout):
    _G["witness"] = DC.LE._load_witness()
    probs = DC.load_problems()
    _G["probsByName"] = {p["name"]: p for p in probs}
    _G["strata"] = {p["name"]: p["stratum"] for p in probs}
    _G["P"] = {p["name"]: P(p) for p in probs}
    _G["proposer"] = DC.LE.FileProposer(chains)
    _G["workdir"], _G["tag"], _G["timeout"] = workdir, tag, timeout


def _judge_problem_by_name(name):
    p = _G["P"][name]
    plans = _G["proposer"].attempts(p)
    rec = {"problem": name, "stratum": _G["strata"].get(name), "attempts": {},
           "out_of_domain": not plans}
    calls = 0
    for k, item in enumerate(plans):
        ks = str(k + 1)
        plan = item.get("plan") if isinstance(item, dict) else item
        if not plan:
            rec["attempts"][ks] = {"certified": False, "reasons": ["no_plan"]}
            continue
        res = _G["witness"].compile_plan(plan, _G["workdir"], tier=plan.get("tier") or "mathlib",
                                         emit=True, bridge=None, run_bypass=False)
        if not (res["compile_ok"] and res["gate1"]["pass"] and res["gate3"]["pass"]):
            rec["attempts"][ks] = {"certified": False, "reasons": ["witness_gate_fail"],
                                   "witness": {"compile_ok": res["compile_ok"],
                                               "gate1": res["gate1"]["pass"],
                                               "gate3": res["gate3"]["pass"]}}
            continue
        body = res["proof_body"]
        j = DC.judge_attempt(name, p.head, p.concl, "\n".join(body), _G["workdir"],
                             "%s_a%s" % (_G["tag"], ks), _G["timeout"])
        j["n_forward"] = res.get("n_steps")
        calls += 1 + (2 if j["lean"] and j["lean"].get("accepted") else 0)
        rec["attempts"][ks] = j
    rec["lean_calls"] = calls
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--judge-suffix", default=DEFAULT_JUDGE_SUFFIX,
                    help="判卷 summary 后缀 (缺省 judge_summary; 置空串 '' = 旧名, 供回滚)")
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    _init(a.chains, a.workdir, a.tag, a.timeout)
    names = list(_G["probsByName"])
    records = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(_judge_problem_by_name, n): n for n in names}
        done = 0
        for f in as_completed(futs):
            records.append(f.result())
            done += 1
            if done % 20 == 0:
                print("[drct-v32-judge] %d/%d" % (done, len(names)), flush=True)
    order = {n: i for i, n in enumerate(_G["probsByName"])}
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
    print("[drct-v32-judge] summary -> %s" % sp, flush=True)
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
