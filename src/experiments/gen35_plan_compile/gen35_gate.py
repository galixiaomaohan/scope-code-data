# -*- coding: utf-8 -*-
# gen35_gate.py -- node [1] **proof-compiler correctness gate** (V35_DESIGN §4)
#   corpus 2399 rows x gold chain x compiler -> proof text -> frozen judging (batch convention)
#   Gate: certified == 2399/2399 (100%); auxiliary reading: **byte-exact agreement rate** with the
#   corpus reference proof (report the real number)
#   Extra: (a) device-argument self-consistency check (R_in/goal_dom/rho recomputed vs in-row record)
#          (b) wrong-chain diagnosis: perturb each row by 1 step -> expect a reject (judging non-trivial proof)
# Writes only under scratch/ccp/gen35/**. nice -n 19; zero GPU.
import argparse
import collections
import json
import os
import random
import sys
import threading
import time

import gen35_common as C
import gen35_compile as CP

OUT = C.G35 + "/out"
LOGS = C.G35 + "/logs"
LEAN = C.G35 + "/lean"


def _serial_batch(items, wd, chunk, tag):
    """Batch judging (frozen-artifact semantics; serial lock + nice -n 19 comes from run_lean_full)."""
    import drct_common as DC
    import gen23_prescreen_batch as PBATCH
    jm = C.md5f(DC.__file__)
    if jm != C.JUDGE_MD5_EXPECT:
        raise SystemExit("FATAL 判卷件非冻结版 %s" % jm)
    lock = threading.Lock()
    orig = PBATCH.run_lean_full

    def ser(*a, **k):
        with lock:
            return orig(*a, **k)
    PBATCH.run_lean_full = ser
    try:
        out = PBATCH.judge_items_batch(items, wd, chunk=chunk, tag=tag, judge_md5=jm)
    finally:
        PBATCH.run_lean_full = orig
    return out, jm


def compile_corpus(rows, logp=None, neg=0, seed=20261005):
    items, stat = [], {"n": 0, "byte_eq": 0, "md5_eq": 0, "n_lines_same": 0,
                       "fam": {}, "err_step_hist": collections.Counter(),
                       "devkw": collections.Counter(), "devkw_match": collections.Counter(),
                       "device_ok": 0, "n_step": 0, "empty_proof": 0}
    negs = []
    rnd = random.Random(seed)
    for i, r in enumerate(rows):
        pv = CP.pv_of_corpus(r)
        res = CP.compile_chain(pv, list(pv["ops"]))
        stat["n"] += 1
        ref = r["proof_text"]
        if res["proof_text"] == ref:
            stat["byte_eq"] += 1
        if res["proof_md5"] == C.md5s(ref):
            stat["md5_eq"] += 1
        if res["n_lines"] == r["n_lines"]:
            stat["n_lines_same"] += 1
        if not res["proof_text"].strip():
            stat["empty_proof"] += 1
        f = stat["fam"].setdefault(r["family"], {"n": 0, "byte_eq": 0, "cert": None})
        f["n"] += 1
        f["byte_eq"] += 1 if res["proof_text"] == ref else 0
        for k, rec in enumerate(res["steps"]):
            stat["n_step"] += 1
            if rec.get("err"):
                stat["err_step_hist"][rec["err"]] += 1
            if rec.get("ok"):
                stat["device_ok"] += 1
            for key in ("R_in", "goal_dom", "rho"):
                mk = "dev_match_" + key
                if mk in rec:
                    stat["devkw"][key] += 1
                    stat["devkw_match"][key] += 1 if rec[mk] else 0
        items.append({"name": r["name"], "head": r["head"], "concl": r["concl"],
                      "proof_text": res["proof_text"], "family": r["family"],
                      "byte_eq": bool(res["proof_text"] == ref),
                      "n_lines": res["n_lines"], "ref_lines": r["n_lines"]})
        if neg and i % max(1, (len(rows) // neg)) == 0 and len(negs) < neg:
            ch = list(pv["ops"])
            if ch:
                j = rnd.randrange(len(ch))
                alt = [o for o in ("gen32_nest_solve", "gen32_nest_close", "gen32_sos_block",
                                   "gen32_sos_cross", "gen32_sos_tail_zero",
                                   "gen32_sos_tail_square", "gen32_sos_tail_other",
                                   "gen32_nest_fold", "gen32_nest_delta") if o != ch[j]]
                ch[j] = rnd.choice(alt)
                r2 = CP.compile_chain(pv, ch)
                negs.append({"name": r["name"], "family": r["family"], "k": j,
                             "op_wrong": ch[j], "op_gold": pv["ops"][j],
                             "head": r["head"], "concl": r["concl"],
                             "proof_text": r2["proof_text"], "n_lines": r2["n_lines"],
                             "byte_eq": bool(r2["proof_text"] == ref)})
        if (i + 1) % 400 == 0:
            C.log("COMPILE %d/%d byte_eq=%d" % (i + 1, len(rows), stat["byte_eq"]), logp)
    stat["byte_eq_rate"] = round(stat["byte_eq"] / max(1, stat["n"]), 6)
    stat["device_ok_rate"] = round(stat["device_ok"] / max(1, stat["n_step"]), 6)
    stat["err_step_hist"] = dict(stat["err_step_hist"])
    stat["devkw"] = dict(stat["devkw"])
    stat["devkw_match"] = dict(stat["devkw_match"])
    for f in stat["fam"].values():
        f["byte_eq_rate"] = round(f["byte_eq"] / max(1, f["n"]), 6)
    return items, stat, negs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=("compile", "judge", "all"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--neg", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--out", default=OUT + "/GEN35_COMPILE_GATE.json")
    ap.add_argument("--compiled", default=OUT + "/GEN35_COMPILED_CORPUS.jsonl")
    ap.add_argument("--log", default=LOGS + "/gen35_gate.log")
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    t0 = time.time()
    C.log("== gen35 gate mode=%s limit=%d judge=%s =="
          % (a.mode, a.limit, C.JUDGE_MD5_EXPECT), a.log)
    led0 = C.parent_ledger()
    rep = {"schema": "gen35-compile-gate/1",
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "judge_md5": C.judge_gate(),
           "definition": ("compiler: 金标链 × 装置供参 -> 算子模板证明文本 -> 冻结判卷; "
                          "门 = certified/2399 = 100%")}
    items = []
    if a.mode in ("compile", "all"):
        rows = C.jlload(C.DEEP_CORPUS)
        if a.limit:
            rows = rows[:a.limit]
        items, stat, negs = compile_corpus(rows, a.log, neg=a.neg)
        rep["compile"] = stat
        C.jlsave([{k: v for k, v in it.items()} for it in items], a.compiled)
        rep["compiled_md5"] = C.md5f(a.compiled)
        rep["n_compiled"] = len(items)
        C.log("COMPILE DONE n=%d byte_eq=%d (%.6f) device_ok=%s"
              % (stat["n"], stat["byte_eq"], stat["byte_eq_rate"], stat["device_ok_rate"]),
              a.log)
        if negs:
            rep["neg_examples"] = negs
    if a.mode in ("judge", "all"):
        if not items:
            items = C.jlload(a.compiled)
        C.lean_gate(5, a.log)
        wd = os.path.join(LEAN, "batch_gate")
        os.makedirs(wd, exist_ok=True)
        jt = [{"name": x["name"], "head": x["head"], "concl": x["concl"],
               "proof_text": x["proof_text"]} for x in items]
        jr, jm = _serial_batch(jt, wd, a.chunk, "gen35_gate")
        cert = sum(1 for x in jr if x.get("certified"))
        rep["judge"] = {"n": len(jr), "certified": cert,
                        "rate": round(cert / max(1, len(jr)), 6),
                        "reason_hist": dict(collections.Counter(
                            r for x in jr if not x.get("certified")
                            for r in (x.get("reasons") or ["?"]))),
                        "judge_md5": jm}
        by = {x["name"]: x for x in jr}
        if "compile" in rep:
            for it in items:
                it["certified"] = bool((by.get(it["name"]) or {}).get("certified"))
            for fam, f in rep["compile"]["fam"].items():
                rs = [it for it in items if it.get("family") == fam]
                f["cert"] = sum(1 for it in rs if it.get("certified"))
            newitems = [{"name": it["name"], "head": it["head"], "concl": it["concl"],
                         "proof_text": it["proof_text"], "family": it["family"],
                         "byte_eq": it.get("byte_eq"), "certified": it.get("certified")}
                        for it in items]
            C.jlsave(newitems, a.compiled)
            rep["compiled_md5"] = C.md5f(a.compiled)
            rep["gate_pass"] = bool(cert == len(items) and len(items) == 2399)
        # wrong-chain diagnosis (judging is non-trivial, self-evidenced)
        negs = rep.get("neg_examples") or []
        if negs:
            nwd = os.path.join(LEAN, "batch_gate_neg")
            os.makedirs(nwd, exist_ok=True)
            jt2 = [{"name": x["name"] + "_neg", "head": x["head"], "concl": x["concl"],
                    "proof_text": x["proof_text"]} for x in negs]
            jr2, _ = _serial_batch(jt2, nwd, max(1, len(jt2)), "gen35_gate_neg")
            nc = sum(1 for x in jr2 if x.get("certified"))
            rep["neg_diagnosis"] = {"n": len(jr2), "certified": nc,
                                    "expected": 0, "pass": bool(nc == 0)}
        C.log("JUDGE DONE certified=%d/%d pass=%s"
              % (cert, len(items), rep.get("gate_pass")), a.log)
    led1 = C.parent_ledger()
    rep["parent_ledger_unchanged"] = bool(C.canon(led0) == C.canon(led1))
    rep["wall_s"] = round(time.time() - t0, 1)
    rep["out_md5"] = C.save_json(a.out, rep)
    C.log("GATE REPORT %s" % json.dumps({k: rep[k] for k in rep
                                         if k in ("compile", "judge", "gate_pass",
                                                  "neg_diagnosis")},
                                        ensure_ascii=False)[:600], a.log)
    print(json.dumps({"out": a.out, "md5": rep["out_md5"],
                      "certified": (rep.get("judge") or {}).get("certified"),
                      "n": (rep.get("judge") or {}).get("n"),
                      "byte_eq_rate": (rep.get("compile") or {}).get("byte_eq_rate"),
                      "gate_pass": rep.get("gate_pass")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
