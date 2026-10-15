# -*- coding: utf-8 -*-
# gen35_chainread.py -- node [2] stock reading (zero training): gen34 A-arm 218 chain walks x compiler -> frozen judging
#   One file produces:
#     [2a] stock chain (gen34 A-arm walked) x compile -> certified (expected magnitude ~= chain accuracy)
#     [2b] gold chain x compile -> certified (compiler ceiling = 100% re-verification)
#     [4-anchor] gen34 A-arm end-to-end certified (legacy-route anchor, read directly from gen34 PRESCREEN_TRACE)
#     AUD-1 zero-parameter chain mode baseline (same exam set)
# Writes only under scratch/ccp/gen35/**. nice -n 19.
import argparse
import collections
import json
import os
import time

import gen35_common as C
import gen35_armrun as AR

OUT = C.G35 + "/out"
LOGS = C.G35 + "/logs"


def gen34_endpoint_certified(arm="slotA"):
    """Legacy-route anchor: gen34 A-arm end-to-end certified (read directly from the official judging trace, read-only).

    Measured note (this batch): the gen34 fork's judging artifact kept the parent name
    `GEN33_PRESCREEN_TRACE_<arm>.jsonl` (V34 diff list [5] did not rename outputs) => candidate
    paths are probed in order; the first version probed only GEN34_* names => misread 0, fixed per
    measurement here (erratum E-1, see report §6).
    """
    cands = [C.G34 + "/out/GEN33_PRESCREEN_TRACE_%s.jsonl" % arm,
             C.G34 + "/out/GEN34_PRESCREEN_TRACE_%s.jsonl" % arm]
    for p in cands:
        if os.path.exists(p):
            rows = C.jlload(p)
            return {"n": len(rows),
                    "certified": sum(1 for r in rows if r.get("certified")),
                    "certified_names": [r["name"] for r in rows if r.get("certified")],
                    "path": p, "md5": C.md5f(p),
                    "note": "gen34 官方端到端读数 (渲染=LLM), 本批只读不回改"}
    return {"n": 0, "certified": None, "note": "trace 缺失", "candidates": cands}


def anchor_gen34_all():
    """Read-only anchors: gen34 three-arm end-to-end certified (base/slotA/slotB)."""
    out = {}
    for arm in ("base", "slotA", "slotB"):
        out[arm] = gen34_endpoint_certified(arm)
    return out



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="slotA", help="source arm of the gen34 chain walk")
    ap.add_argument("--judge", type=int, default=1)
    ap.add_argument("--fixup", type=int, default=0,
                    help="recompute only the legacy-route anchors (endpoint_gen34/all) and write back "
                         "in place (erratum E-1; judging section untouched)")
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--out", default=OUT + "/GEN35_CHAINREAD.json")
    ap.add_argument("--rows-out", default=OUT + "/GEN35_CHAINREAD_ROWS.jsonl")
    ap.add_argument("--log", default=LOGS + "/gen35_chainread.log")
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    if a.fixup:
        rep = json.load(open(a.out, encoding="utf-8"))
        old = rep.get("endpoint_gen34")
        rep["endpoint_gen34"] = gen34_endpoint_certified(a.arm)
        rep["endpoint_gen34_anchor_all"] = anchor_gen34_all()
        rep["corrections"] = [{"id": "E-1", "field": "endpoint_gen34", "before": old,
                               "after": rep["endpoint_gen34"],
                               "why": "gen34 fork 判卷产物沿父件命名 (GEN33_PRESCREEN_TRACE_*), "
                                      "首版路径探测只含 GEN34_* 名 => 误读 0; 按实测路径修"}]
        rep["out_md5"] = C.save_json(a.out, rep)
        print(json.dumps({"out": a.out, "md5": rep["out_md5"],
                          "endpoint": rep["endpoint_gen34"], "anchor_all": {
                              k: (v or {}).get("certified")
                              for k, v in rep["endpoint_gen34_anchor_all"].items()}},
                         ensure_ascii=False))
        return
    t0 = time.time()
    C.log("== gen35 chainread arm=%s judge=%d ==" % (a.arm, a.judge), a.log)
    led0 = C.parent_ledger()
    rep = {"schema": "gen35-chainread/1",
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "judge_md5": C.judge_gate(), "arm": a.arm,
           "note": "存量链 = gen34 A 臂 hold-out 外的考卷走链 (零训练, 只读)"}
    exam = AR.load_exam()
    rep["n_exam"] = len(exam)
    chains = AR.load_gen34_chains(a.arm)
    rep["n_chains_loaded"] = len(chains)
    rep["chain_exact_hit_infile"] = sum(1 for v in chains.values() if v["exact_hit"])
    rep["chain_first_hit_infile"] = sum(1 for v in chains.values() if v["first_hit"])
    # [2b] gold chain x compile
    rows_gold, stat_gold = AR.compile_arm(exam, {int(r["ri"]): {"walked": list(r["ops"] or []),
                                                                "gold": list(r["ops"] or []),
                                                                "exact_hit": True,
                                                                "first_hit": True,
                                                                "arch_src": "gold"}
                                               for r in exam}, tag="gold_chain")
    # [2a] stock chain x compile
    rows_w, stat_w = AR.compile_arm(exam, chains, tag="gen34_%s" % a.arm)
    if a.judge:
        C.lean_gate(5, a.log)
        rep["judge_gold"] = AR.judge_rows(rows_gold, "gold_chain", a.chunk, a.log)
        rep["judge_stored"] = AR.judge_rows(rows_w, "stored_%s" % a.arm, a.chunk, a.log)
    rep["compile_gold"] = stat_gold
    rep["compile_stored"] = stat_w
    rep["endpoint_gen34"] = gen34_endpoint_certified(a.arm)
    rep["aud1"] = {k: v for k, v in AR.aud1_zero_param(exam).items() if k != "rows"}
    if a.judge:
        # coupling breakdown of chain correctness vs judge pass (same chain, same compile)
        buckets = collections.Counter()
        for r in rows_w:
            buckets["walk_exact=%s|walk_first=%s|cert=%s"
                    % (bool(r["walk_exact_hit"]), bool(r["walk_first_hit"]),
                       bool(r["certified"]))] += 1
        rep["decode_buckets_stored"] = dict(buckets)
        # gold-chain compile: byte-exact vs corpus reference (lookup: exam rows are absent from the corpus, so report judging only)
        rep["gate_upper_bound_pass"] = bool(rep["judge_gold"]["certified"] == len(exam))
    ARR = []
    for tag, rows in (("gold_chain", rows_gold), ("stored_chain", rows_w)):
        for r in rows:
            rr = dict(r)
            rr["arm_tag"] = tag
            ARR.append(rr)
    C.jlsave(ARR, a.rows_out)
    rep["rows_md5"] = C.md5f(a.rows_out)
    led1 = C.parent_ledger()
    rep["parent_ledger_unchanged"] = bool(C.canon(led0) == C.canon(led1))
    rep["wall_s"] = round(time.time() - t0, 1)
    rep["out_md5"] = C.save_json(a.out, rep)
    print(json.dumps({"out": a.out, "md5": rep["out_md5"],
                      "gold_certified": (rep.get("judge_gold") or {}).get("certified"),
                      "stored_certified": (rep.get("judge_stored") or {}).get("certified"),
                      "n": len(exam),
                      "stored_chain_exact": rep["compile_stored"]["n_exact"],
                      "gen34_endpoint": rep["endpoint_gen34"]["certified"],
                      "aud1_exact": rep["aud1"]["exact_rate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
