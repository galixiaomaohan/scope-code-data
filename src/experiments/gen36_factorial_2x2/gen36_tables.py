# -*- coding: utf-8 -*-
# gen36_tables.py -- node [3]/[4] summary table: M1_G / M2_G / M3_G + four-arm comparison + honest-negative registry.
#   Input: each arm (WALK json, EVAL_PLAN json (judging), ROWS jsonl) + the same three artifacts for the gen35 anchor
#   Output: out/GEN36_TABLES.json
# Reads upstream artifacts only; writes only under scratch/ccp/gen36/**.
import argparse
import json
import os
import sys

import gen36_common as C

G36, G35 = C.G36, C.G35
OUT = G36 + "/out"

# anchors (gen35 close-out): read-only references
ANCHOR = {
    "pure_plan_gen35": {"walk": G35 + "/out/GEN35_EVAL_PLAN_WALK.json",
                        "plan": G35 + "/out/GEN35_EVAL_PLAN.json",
                        "rows": G35 + "/out/GEN35_EVAL_PLAN_ROWS.jsonl",
                        "certified": 117, "n": 218},
    "gold_chain_gen35": {"rows": G35 + "/out/GEN35_CHAINREAD_ROWS.jsonl",
                         "certified": 218, "n": 218,
                         "note": "金标链 × 编译 = 编译器上限"},
    "stock_chain_gen34A": {"rows": G35 + "/out/GEN35_CHAINREAD_ROWS.jsonl",
                           "certified": 121, "n": 218,
                           "note": "gen34 A 臂存量链 × 编译"},
    "end2end_gen34A": {"certified": 1, "n": 218, "note": "旧路线端到端 (LLM 渲染)"},
}


def read_plan(p):
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def judge_count(plan):
    if not plan:
        return None
    j = plan.get("judge") or {}
    return {"certified": j.get("certified"), "n": j.get("n"),
            "reasons": (j.get("reasons") or j.get("reason_hist") or {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[],
                    help="label=WALKPATH,PLANPATH,ROWSPATH (repeatable)")
    ap.add_argument("--migration", default=OUT + "/GEN36_MIGRATION.json")
    ap.add_argument("--out", default=OUT + "/GEN36_TABLES.json")
    a = ap.parse_args()
    arms = {}
    for spec in a.arm:
        lb, rest = spec.split("=", 1)
        parts = (rest.split(",") + ["", "", ""])[:3]
        arms[lb] = {"walk": parts[0], "plan": parts[1], "rows": parts[2]}
    tab = {"schema": "gen36-tables/1", "anchor": ANCHOR, "arms": {}}
    for lb, p in sorted(arms.items()):
        w = json.load(open(p["walk"], encoding="utf-8")) if os.path.exists(p["walk"]) else None
        pl = read_plan(p["plan"])
        rows = None
        if p["rows"] and os.path.exists(p["rows"]):
            rows = C.jlload(p["rows"])
        cert = judge_count(pl)
        rec = {"walk_file": p["walk"], "plan_file": p["plan"], "rows_file": p["rows"],
               "walk_md5": (C.md5f(p["walk"]) if os.path.exists(p["walk"]) else None),
               "plan_md5": (C.md5f(p["plan"]) if os.path.exists(p["plan"]) else None),
               "rows_md5": (C.md5f(p["rows"]) if os.path.exists(p["rows"]) else None),
               "walk_table": (w or {}).get("table"),
               "n": (w or {}).get("n"),
               "compile": (pl or {}).get("compile"), "judge": cert,
               "flag": (pl or {}).get("flag"),
               "judge_md5": (pl or {}).get("judge_md5"),
               "ckpt_md5": (w or {}).get("ckpt_md5")}
        if rows is not None:
            rec["rows_n"] = len(rows)
        tab["arms"][lb] = rec
    mig = json.load(open(a.migration, encoding="utf-8")) if os.path.exists(a.migration) else None
    tab["migration_md5"] = (mig or {}).get("md5")
    # ---- criteria ----
    crit = {"M1_G_primary_arm": None, "M2_G_delta_vs_gen35_0.5367": None,
            "M3_G_second_arm": None, "n_exam": 218}
    n = 218
    for lb, rec in tab["arms"].items():
        jc = (rec.get("judge") or {}).get("certified")
        if jc is None:
            continue
        if lb.startswith("gw") or lb.startswith("w_"):
            crit["M1_G_primary_arm"] = {"label": lb, "certified": jc, "n": n,
                                        "rate": round(jc / n, 6),
                                        "wilson_lb": round(C.wilson_lb(jc, n), 6),
                                        "delta_vs_gen35": round(jc / n - 117 / n, 6)}
            crit["M2_G_delta_vs_gen35_0.5367"] = crit["M1_G_primary_arm"]["delta_vs_gen35"]
        if lb.startswith("gsw") or lb.startswith("sw_"):
            crit["M3_G_second_arm"] = {"label": lb, "certified": jc, "n": n,
                                       "rate": round(jc / n, 6),
                                       "wilson_lb": round(C.wilson_lb(jc, n), 6),
                                       "delta_vs_gen35": round(jc / n - 117 / n, 6)}
    tab["criteria"] = crit
    tab["honesty"] = {
        "note": ("锚点 = gen35 纯规划 117/218 (0.5367); 每 +1 题 = +0.4587pt; "
                 "判卷冻结件 md5 断言由 C.judge_gate() 现场复核"),
        "declared": ("K17 未执行≠判零; 缺件一律 None (不默认 0); 读件路径逐条登记")}
    tab["judge_md5_now"] = C.judge_gate()
    tab["md5"] = C.save_json(a.out, tab)
    print(json.dumps({"out": a.out, "md5": tab["md5"], "criteria": crit,
                      "arms": {k: {"walk_table": v["walk_table"], "judge": v["judge"]}
                               for k, v in tab["arms"].items()}},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
