# -*- coding: utf-8 -*-
# gen36_migration.py -- error center-of-mass migration matrix (V36_DESIGN §5 criterion): anchor vs each arm, cell by cell.
#   Input: several (label, WALK json) pairs (gen35 anchor + gen36 arms); output out/GEN36_MIGRATION.json
# Reads upstream artifacts only; writes only under scratch/ccp/gen36/**.
import argparse
import collections
import json
import os
import sys

import gen36_common as C

G36, G35 = C.G36, C.G35
OUT = G36 + "/out"
TAIL_KINDS = ("square", "other", "zero")


def tail_kind(op):
    if not op:
        return None
    for k in TAIL_KINDS:
        if op.endswith("tail_" + k):
            return k
    return None


def first_div(walked, gold):
    n = min(len(walked), len(gold))
    for i in range(n):
        if walked[i] != gold[i]:
            return i
    return len(walked) if len(walked) < len(gold) else None


def analyze(path):
    d = json.load(open(path, encoding="utf-8"))
    rows = d["rows"]
    n = len(rows)
    exact = sum(1 for r in rows if r["chain"]["exact_hit"])
    first = sum(1 for r in rows if r["chain"]["first_hit"])
    fam = collections.defaultdict(lambda: {"n": 0, "exact": 0})
    pos_hist, fromend_hist = collections.Counter(), collections.Counter()
    conf = collections.Counter()
    tail_tot, tail_hit = collections.Counter(), collections.Counter()
    tail_conf = collections.Counter()
    pre_pmax, conf_wrong = [], 0
    for r in rows:
        ch = r["chain"]
        walked, gold = list(ch["walked"]), list(ch["gold"])
        f = r.get("family")
        fam[f]["n"] += 1
        fam[f]["exact"] += int(ch["exact_hit"])
        # per tail kind (gold tail kind = row ground truth)
        gk = tail_kind(gold[-1] if gold else None)
        if gk:
            tail_tot[gk] += 1
            if ch["exact_hit"]:
                tail_hit[gk] += 1
        if ch["exact_hit"]:
            continue
        fd = first_div(walked, gold)
        pos_hist[fd] += 1
        fromend_hist[len(gold) - 1 - fd] += 1
        gp = (gold[fd] if fd < len(gold) else None)
        pp = (walked[fd] if fd < len(walked) else None)
        conf["%s -> %s" % (gp, pp)] += 1
        if gk:
            tail_conf["%s -> %s" % (gk, tail_kind(pp))] += 1
        tr = [t for t in (ch.get("trace") or []) if t.get("k") == fd]
        if tr:
            pm = tr[0].get("p_max")
            if pm is not None:
                pre_pmax.append(float(pm))
                conf_wrong += int(float(pm) >= 0.9)
    return {
        "path": path, "n": n, "exact": exact, "first": first,
        "exact_rate": round(exact / max(n, 1), 6),
        "n_bad": n - exact,
        "family": {k: dict(v, rate=round(v["exact"] / max(1, v["n"]), 6))
                   for k, v in sorted(fam.items())},
        "first_div_pos_hist": {str(k): v for k, v in sorted(pos_hist.items())},
        "first_div_from_end_hist": {str(k): v for k, v in sorted(fromend_hist.items())},
        "confusion_gold_to_pred": dict(conf.most_common(30)),
        "tail_by_gold_kind": {k: {"n": tail_tot[k], "exact": tail_hit[k],
                                  "recall": round(tail_hit[k] / max(1, tail_tot[k]), 6)}
                              for k in sorted(tail_tot)},
        "tail_confusion": dict(tail_conf.most_common()),
        "err_p_max_mean": (round(sum(pre_pmax) / len(pre_pmax), 6) if pre_pmax else None),
        "n_confident_wrong_p90": conf_wrong,
        "walk_table": d.get("table"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[],
                    help="label=path (repeatable)")
    ap.add_argument("--anchor", default="gen35=" + G35 + "/out/GEN35_EVAL_PLAN_WALK.json")
    ap.add_argument("--out", default=OUT + "/GEN36_MIGRATION.json")
    a = ap.parse_args()
    arms = [a.anchor] + list(a.arm)
    res, order = {}, []
    for spec in arms:
        lb, path = spec.split("=", 1)
        if not os.path.exists(path):
            print("SKIP (artifact missing): %s -> %s" % (lb, path))
            continue
        res[lb] = analyze(path)
        order.append(lb)
    mig = {}
    if order and order[0] in res:
        base = res[order[0]]
        for lb in order[1:]:
            cur = res[lb]
            d_exact = cur["exact"] - base["exact"]
            cells = {}
            for k in set(list(base["confusion_gold_to_pred"]) +
                         list(cur["confusion_gold_to_pred"])):
                cells[k] = {"anchor": base["confusion_gold_to_pred"].get(k, 0),
                            "arm": cur["confusion_gold_to_pred"].get(k, 0)}
            tk = {}
            for k in set(list(base["tail_by_gold_kind"]) + list(cur["tail_by_gold_kind"])):
                b = (base["tail_by_gold_kind"].get(k) or {}).get("exact", 0)
                c = (cur["tail_by_gold_kind"].get(k) or {}).get("exact", 0)
                bn = (base["tail_by_gold_kind"].get(k) or {}).get("n", 0)
                cn = (cur["tail_by_gold_kind"].get(k) or {}).get("n", 0)
                tk[k] = {"anchor_exact": b, "arm_exact": c, "delta": c - b,
                         "anchor_n": bn, "arm_n": cn}
            mig[lb] = {"vs": order[0], "delta_exact": d_exact,
                       "delta_exact_rate": round(d_exact / max(base["n"], 1), 6),
                       "confusion_cells": cells, "tail_kind_delta": tk,
                       "first_div_from_end": {"anchor": base["first_div_from_end_hist"],
                                              "arm": cur["first_div_from_end_hist"]},
                       "n_confident_wrong_p90": {"anchor": base["n_confident_wrong_p90"],
                                                 "arm": cur["n_confident_wrong_p90"]}}
    rep = {"schema": "gen36-migration/1", "order": order, "arms": res, "migration": mig,
           "judge_md5": C.judge_gate()}
    rep["md5"] = C.save_json(a.out, rep)
    print(json.dumps({"out": a.out, "md5": rep["md5"],
                      "exact": {k: "%d/%d" % (v["exact"], v["n"]) for k, v in res.items()},
                      "migration": {k: {"d_exact": v["delta_exact"],
                                        "tail": {kk: vv["delta"] for kk, vv in
                                                 v["tail_kind_delta"].items()}}
                                    for k, v in mig.items()}},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
