# -*- coding: utf-8 -*-
# gen35_tables.py -- node [4]/[5] **four-arm comparison table + M1_F/M2_F/M3_F + dual-convention summary**
#   Reads: GEN35_COMPILE_GATE.json / GEN35_COMPILED_CORPUS.jsonl / GEN35_CHAINREAD{,_ROWS}
#       / GEN35_EVAL_PLAN{,_ROWS} / GEN35_PAEQ.json / GEN35_JUDGE_*.json / gen34 anchors
#   Writes: GEN35_TABLES.json
import argparse
import collections
import json
import os
import time

import gen35_common as C
import gen35_armrun as AR

OUT = C.G35 + "/out"
LOGS = C.G35 + "/logs"
GEN34_EXACT_HIT = 121 / 218.0        # gen34 A-arm same-exam full-chain exact (pre-registered reference value)
GEN33_BASE_END = 0                    # gen33 baseline end-to-end certified
GEN34_SLOTA_END = 1                   # gen34 A-arm end-to-end certified (legacy-route anchor)


def _load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def nz(d, k, default=None):
    return (d or {}).get(k, default)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT + "/GEN35_TABLES.json")
    ap.add_argument("--log", default=LOGS + "/gen35_tables.log")
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    t0 = time.time()
    gate = _load(OUT + "/GEN35_COMPILE_GATE.json")
    cr = _load(OUT + "/GEN35_CHAINREAD.json")
    ev = _load(OUT + "/GEN35_EVAL_PLAN.json")
    paeq = _load(OUT + "/GEN35_PAEQ.json")
    g34t = _load(C.G34 + "/out/GEN34_TABLES.json")
    cr_rows = C.jlload(OUT + "/GEN35_CHAINREAD_ROWS.jsonl") if os.path.exists(
        OUT + "/GEN35_CHAINREAD_ROWS.jsonl") else []
    ev_rows = C.jlload(OUT + "/GEN35_EVAL_PLAN_ROWS.jsonl") if os.path.exists(
        OUT + "/GEN35_EVAL_PLAN_ROWS.jsonl") else []
    exam = AR.load_exam()
    rep = {"schema": "gen35-tables/1", "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "judge_md5": C.md5f(C.DRCT + "/drct_common.py"),
           "n_exam": len(exam), "prereg": {"M1_F": "pure-plan certified/n",
                                           "M2_F": "M1_F - stored-chain certified/n",
                                           "M3_F": "chain first/exact hit",
                                           "gate": "compiler gold-chain certified = 100%",
                                           "gen34_exact_hit_ref": GEN34_EXACT_HIT}}
    # ---- node [1] compiler gate ----
    compiled = C.jlload(OUT + "/GEN35_COMPILED_CORPUS.jsonl") if os.path.exists(
        OUT + "/GEN35_COMPILED_CORPUS.jsonl") else []
    corpus = {r["name"]: r for r in C.jlload(C.DEEP_CORPUS)}
    fam = collections.defaultdict(lambda: {"n": 0, "byte_eq": 0, "certified": 0})
    byte_eq = 0
    for r in compiled:
        ref = (corpus.get(r["name"]) or {}).get("proof_text")
        f = fam[(corpus.get(r["name"]) or {}).get("family") or "?"]
        f["n"] += 1
        if ref is not None and r["proof_text"] == ref:
            byte_eq += 1
            f["byte_eq"] += 1
        if r.get("certified"):
            f["certified"] += 1
    cert_n = sum(1 for r in compiled if r.get("certified"))
    rep["gate"] = {"n": len(compiled), "certified": cert_n,
                   "rate": round(cert_n / max(1, len(compiled)), 6),
                   "byte_eq_ref": byte_eq,
                   "byte_eq_rate": round(byte_eq / max(1, len(compiled)), 6),
                   "by_family": {k: dict(v) for k, v in fam.items()},
                   "judge": nz(gate, "judge"), "neg_diagnosis": nz(gate, "neg_diagnosis"),
                   "pass": bool(len(compiled) == 2399 and cert_n == 2399 and byte_eq == 2399)}
    # ---- node [2] stock chain ----
    rep["stored_chain"] = {"compile": nz(cr, "compile_stored"),
                           "judge": nz(cr, "judge_stored"),
                           "decode_buckets": nz(cr, "decode_buckets_stored"),
                           "chain_exact_infile": nz(cr, "chain_exact_hit_infile"),
                           "chain_first_infile": nz(cr, "chain_first_hit_infile")}
    rep["gold_chain_upper"] = {"compile": nz(cr, "compile_gold"),
                               "judge": nz(cr, "judge_gold"),
                               "pass": nz(cr, "gate_upper_bound_pass")}
    rep["endpoint_gen34_slotA"] = nz(cr, "endpoint_gen34")
    rep["aud1_zero_param"] = nz(cr, "aud1")
    # ---- node [4] pure-planning arm ----
    rep["pure_plan"] = {"walk": nz(ev, "walk_table"), "compile": nz(ev, "compile"),
                        "judge": nz(ev, "judge"), "ckpt": nz(ev, "ckpt"),
                        "ckpt_md5": nz(ev, "ckpt_md5"), "flag": nz(ev, "flag")}
    rep["paeq"] = {k: nz(paeq, k) for k in ("pass", "state_dict", "logits_first_step",
                                            "grad_cut", "text_term_dead_at_lam0",
                                            "text_term_alive_at_lam1", "slot_term_alive",
                                            "plan_md5", "plan_file_md5",
                                            "plan_md5_same_as_gen34", "out_md5")}
    # ---- four-arm comparison table ----
    n = len(exam)
    pj = nz(rep["pure_plan"], "judge") or {}
    sj = nz(rep["stored_chain"], "judge") or {}
    gj = nz(rep["gold_chain_upper"], "judge") or {}
    pc = nz(rep["pure_plan"], "compile") or {}
    rep["four_way"] = {
        "1_pure_plan_llm_x_compiler": {"certified": pj.get("certified"),
                                       "n": n,
                                       "rate": round((pj.get("certified") or 0) / n, 6)},
        "2_gold_chain_x_compiler": {"certified": gj.get("certified"), "n": n,
                                    "rate": round((gj.get("certified") or 0) / n, 6)},
        "3_gen34A_chain_x_compiler": {"certified": sj.get("certified"), "n": n,
                                      "rate": round((sj.get("certified") or 0) / n, 6)},
        "4_gen34A_end_to_end": {"certified": (rep["endpoint_gen34_slotA"] or {})
                                .get("certified"), "n": n,
                                "note": "旧路线锚点 (LLM 渲染)"}}
    m1 = (pj.get("certified") or 0) / n
    m3 = (sj.get("certified") or 0) / n
    rep["metrics"] = {
        "M1_F": round(m1, 6),
        "M1_F_int": "%d/%d" % ((pj.get("certified") or 0), n),
        "M2_F": round(m1 - m3, 6),
        "M2_F_int": "%d/%d - %d/%d" % ((pj.get("certified") or 0), n,
                                       (sj.get("certified") or 0), n),
        "M3_F_first_hit": round((pc.get("n_first") or 0) / n, 6),
        "M3_F_exact_hit": round((pc.get("n_exact") or 0) / n, 6),
        "M3_F_exact_ref_gen34": round(GEN34_EXACT_HIT, 6),
        "M3_F_gate_lb": round(C.wilson_lb(GEN34_EXACT_HIT * 218, 218), 6),
        "M3_F_pass": bool(((pc.get("n_exact") or 0) / n) >= C.wilson_lb(
            GEN34_EXACT_HIT * 218, 218)),
        "compiler_gate": rep["gate"]["pass"],
        "M1_F_vs_gen33_base_0": round(m1 - GEN33_BASE_END, 6),
        "M1_F_vs_gen34_slotA_1": round(m1 - GEN34_SLOTA_END / n, 6)}
    # ---- chain-correct vs judge-pass coupling breakdown (pure-planning arm) ----
    if ev_rows:
        b = collections.Counter()
        for r in ev_rows:
            b["exact=%s|first=%s|cert=%s" % (bool(r["exact_hit"]), bool(r["first_hit"]),
                                             bool(r.get("certified")))] += 1
        rep["decode_buckets_pure_plan"] = dict(b)
        fam2 = collections.defaultdict(lambda: {"n": 0, "cert": 0, "exact": 0, "first": 0})
        for r in ev_rows:
            f = fam2[r["family"]]
            f["n"] += 1
            f["cert"] += 1 if r.get("certified") else 0
            f["exact"] += 1 if r["exact_hit"] else 0
            f["first"] += 1 if r["first_hit"] else 0
        rep["pure_plan_by_family"] = {k: dict(v) for k, v in fam2.items()}
        rep["pure_plan_step_err_hist"] = dict(collections.Counter(
            k for r in ev_rows for k in (r.get("err_hist") or {})))
    if cr_rows:
        fam3 = collections.defaultdict(lambda: {"n": 0, "cert": 0, "exact": 0})
        for r in cr_rows:
            if r.get("arm_tag") != "stored_chain":
                continue
            f = fam3[r["family"]]
            f["n"] += 1
            f["cert"] += 1 if r.get("certified") else 0
            f["exact"] += 1 if r["exact_hit"] else 0
        rep["stored_chain_by_family"] = {k: dict(v) for k, v in fam3.items()}
    # ---- dual convention ----
    rep["dual_criteria"] = {}
    for f in sorted(os.listdir(OUT)):
        if f.startswith("GEN35_JUDGE_") and f.endswith(".json"):
            d = json.load(open(os.path.join(OUT, f), encoding="utf-8"))
            rep["dual_criteria"][f] = {k: v for k, v in d.items()
                                       if k in ("src_tag", "src_n", "pool", "pool_n",
                                                "literal", "reverse", "judge_md5_before",
                                                "judge_md5_after", "judge_unchanged",
                                                "literal_pool_note", "literal_pool_n")}
            if isinstance(rep["dual_criteria"][f].get("literal"), dict):
                rep["dual_criteria"][f]["literal"] = {
                    k: v for k, v in rep["dual_criteria"][f]["literal"].items()
                    if k != "records"}
            if isinstance(rep["dual_criteria"][f].get("reverse"), dict):
                rep["dual_criteria"][f]["reverse"] = {
                    k: v for k, v in rep["dual_criteria"][f]["reverse"].items()
                    if k != "records"}
    rep["gen34_tables_ref"] = {k: nz(g34t, k) for k in ("M1_E", "M2_E", "M3_E", "M4_E")} \
        if g34t else None
    # ---- training curve summary ----
    summ = _load(OUT + "/DEEP_PLAN_SUMMARY.json")
    rep["train"] = {}
    if summ:
        rep["train"] = {"tag": nz(summ, "tag"),
                        "plan": nz(summ, "plan"),
                        "steps": nz(summ, "steps"),
                        "flags": {k: nz(summ, k) for k in ("lam_bind", "lam_txt",
                                                           "slot_cond", "slot_dev")}}
    cur = OUT + "/DEEP_PLAN_CURVE.jsonl"
    if os.path.exists(cur):
        rows = C.jlload(cur)
        if rows:
            rep["train"]["curve_first"] = {k: rows[0].get(k) for k in
                                           ("step", "loss", "chain", "slot", "text")}
            rep["train"]["curve_last"] = {k: rows[-1].get(k) for k in
                                          ("step", "loss", "chain", "slot", "text")}
            rep["train"]["n_curve_rows"] = len(rows)
            ev_c = [r for r in rows if r.get("eval")]
            if ev_c:
                rep["train"]["eval_first"] = ev_c[0]["eval"]
                rep["train"]["eval_last"] = ev_c[-1]["eval"]
                rep["train"]["eval_n"] = len(ev_c)
    rep["wall_s"] = round(time.time() - t0, 1)
    rep["out_md5"] = C.save_json(a.out, rep)
    C.log("TABLES M1_F=%s M2_F=%s M3_F=%s/%s gate=%s -> %s"
          % (rep["metrics"]["M1_F_int"], rep["metrics"]["M2_F"],
             rep["metrics"]["M3_F_first_hit"], rep["metrics"]["M3_F_exact_hit"],
             rep["metrics"]["compiler_gate"], a.out), a.log)
    print(json.dumps({"out": a.out, "md5": rep["out_md5"], "metrics": rep["metrics"],
                      "four_way": rep["four_way"], "gate": {
                          k: rep["gate"][k] for k in ("n", "certified", "byte_eq_ref",
                                                      "byte_eq_rate", "pass")}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
