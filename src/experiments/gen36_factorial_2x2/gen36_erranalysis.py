# -*- coding: utf-8 -*-
# gen36_erranalysis.py -- node [0] routing-error analysis (CPU, before design)
#   Input: gen35 WALK artifact (218-exam chain walks vs gold, per problem per step) + gen32 deep-exam corpus (decision-point structure)
#   Output: out/GEN36_ERRANALYSIS.json -- error-step positions / operator confusion / family distribution / terminal-step decoy confusion matrix
# Imports and reads only; write scope is scratch/ccp/gen36/** only. No sudo / no /tmp.
import argparse
import collections
import json
import os
import sys

sys.dont_write_bytecode = True

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = RES + "/scratch/ccp"
G36 = CCP + "/gen36"
G35 = CCP + "/gen35"
G32 = CCP + "/gen32"

TAIL_KINDS = ("square", "other", "zero")


def tail_kind(op):
    """Terminal-step kind: gen32_sos_tail_square / tail_other / tail_zero -> kind."""
    if not op:
        return None
    for k in TAIL_KINDS:
        if op.endswith("_tail_" + k) or op.endswith("tail_" + k):
            return k
    if "tail" in op:
        return "?" + op.split("tail")[-1]
    return None


def first_div(walked, gold):
    """First divergence position (None = fully equal or full prefix correct)."""
    n = min(len(walked), len(gold))
    for i in range(n):
        if walked[i] != gold[i]:
            return i
    if len(walked) < len(gold):
        return len(walked)          # early stop
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", default=G35 + "/out/GEN35_EVAL_PLAN_WALK.json")
    ap.add_argument("--corpus", default=G32 + "/DEEP_CORPUS_V1.jsonl")
    ap.add_argument("--exam-side", default=CCP + "/gen33/out/DEEP_EXAM_SIDECAR.jsonl")
    ap.add_argument("--out", default=G36 + "/out/GEN36_ERRANALYSIS.json")
    ap.add_argument("--rows-out", default=G36 + "/out/GEN36_ERRANALYSIS_ROWS.jsonl")
    a = ap.parse_args()

    W = json.load(open(a.walk, encoding="utf-8"))
    rows = W["rows"]
    n_all = len(rows)

    # ---- corpus side: row name -> meta (cls / tail kind / chain length / blocks) ----
    corp = {}
    for line in open(a.corpus, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        meta = d.get("meta") or {}
        corp[d["name"]] = {
            "family": d.get("family"), "chain_ops": d.get("chain_ops"),
            "n_steps": d.get("n_steps"), "cls": meta.get("cls") or d.get("cls"),
            "cls_sq": meta.get("cls_sq"), "n_blk": meta.get("n_blk"),
            "nv": meta.get("nv"), "D": meta.get("D"), "rho": str(meta.get("rho")),
            "decision_points": d.get("decision_points"),
            "resid_nonzero": d.get("resid_nonzero"),
            "resid_nonzero_rate": d.get("resid_nonzero_rate"),
        }

    # ---- per-problem error localization ----
    per_row = []
    for r in rows:
        ch = r["chain"]
        walked, gold = list(ch["walked"]), list(ch["gold"])
        fd = first_div(walked, gold)
        nm = r["name"]
        ci = corp.get(nm) or {}
        # gold terminal step (last step of the chain) and its kind
        gold_tail = gold[-1] if gold else None
        pred_at_fd = walked[fd] if (fd is not None and fd < len(walked)) else None
        gold_at_fd = gold[fd] if (fd is not None and fd < len(gold)) else None
        rec = {
            "ri": int(r["ri"]), "name": nm, "family": r.get("family"),
            "cls": r.get("cls"),
            "exact": bool(ch["exact_hit"]), "first": bool(ch["first_hit"]),
            "prefix_hit": bool(ch["prefix_hit"]),
            "n_walked": len(walked), "n_gold": len(gold),
            "over_run": bool(ch["over_run"]), "terminated_end": bool(ch["terminated_end"]),
            "walked": walked, "gold": gold,
            "first_div": fd,
            "gold_at_fd": gold_at_fd, "pred_at_fd": pred_at_fd,
            "gold_tail": gold_tail, "gold_tail_kind": tail_kind(gold_tail),
            "is_tail_pos": bool(fd is not None and gold and fd == len(gold) - 1),
            "n_mismatch": sum(1 for i in range(min(len(walked), len(gold)))
                              if walked[i] != gold[i]),
            "corp": {k: ci.get(k) for k in ("cls", "cls_sq", "n_blk", "nv", "D",
                                            "rho", "resid_nonzero")},
        }
        # error-step confidence (trace at the pred position)
        tr = ch.get("trace") or []
        if fd is not None:
            hit_tr = [t for t in tr if t.get("k") == fd]
            if hit_tr:
                rec["err_p_max"] = hit_tr[0].get("p_max")
                rec["err_margin"] = hit_tr[0].get("margin")
                rec["err_pred_idx"] = hit_tr[0].get("pred_idx")
                rec["err_gold_in_space"] = hit_tr[0].get("gold_in_space")
        # control: median margin of correct steps
        ok_m = [t.get("margin") for t in tr if t.get("hit")]
        if ok_m:
            ok_m = sorted(ok_m)
            rec["ok_margin_med"] = ok_m[len(ok_m) // 2]
        per_row.append(rec)

    bad = [x for x in per_row if not x["exact"]]
    good = [x for x in per_row if x["exact"]]

    # ---- aggregation ----
    # 1) family distribution
    fam = collections.defaultdict(lambda: {"n": 0, "exact": 0, "bad": 0})
    for x in per_row:
        f = x["family"]
        fam[f]["n"] += 1
        fam[f]["exact"] += int(x["exact"])
        fam[f]["bad"] += int(not x["exact"])
    # 2) error-step position histogram (absolute k / distance from the terminal step)
    pos_hist = collections.Counter()
    from_end_hist = collections.Counter()
    for x in bad:
        fd = x["first_div"]
        pos_hist[fd] += 1
        if x["n_gold"]:
            from_end_hist[x["n_gold"] - 1 - fd] += 1
    # 3) operator confusion (gold_at_fd -> pred_at_fd)
    conf = collections.Counter()
    for x in bad:
        conf["%s -> %s" % (x["gold_at_fd"], x["pred_at_fd"])] += 1
    conf_by_fam = collections.defaultdict(collections.Counter)
    for x in bad:
        conf_by_fam[x["family"]]["%s -> %s" % (x["gold_at_fd"], x["pred_at_fd"])] += 1
    # 4) terminal-step decoy confusion matrix (only problems whose first error is at the terminal position)
    tail_conf = collections.Counter()
    tail_pos_rows = [x for x in bad if x["is_tail_pos"]]
    for x in tail_pos_rows:
        gk = x["gold_tail_kind"] or tail_kind(x["gold_at_fd"])
        pk = tail_kind(x["pred_at_fd"])
        tail_conf["%s -> %s" % (gk, pk)] += 1
    # 5) full-chain prefix correct (error only at the last step) vs mid-chain errors
    only_tail = [x for x in bad if x["n_mismatch"] == 1 and x["is_tail_pos"]]
    mid_err = [x for x in bad if not x["is_tail_pos"]]
    multi_err = [x for x in bad if x["n_mismatch"] > 1]
    # 6) early stop / overrun
    early_stop = [x for x in bad if x["n_walked"] < x["n_gold"]]
    over_run = [x for x in bad if x["over_run"]]
    # 7) confidence contrast (error-step p_max/margin vs correct-step median)
    err_pmax = [x["err_p_max"] for x in bad if x.get("err_p_max") is not None]
    err_marg = [x["err_margin"] for x in bad if x.get("err_margin") is not None]
    ok_med = [x["ok_margin_med"] for x in per_row if x.get("ok_margin_med") is not None]
    confident_wrong = [x for x in bad if (x.get("err_p_max") or 0) >= 0.9]

    def mean(v):
        return round(sum(v) / len(v), 6) if v else None

    # 8) tail-kind distribution (gold tail kind x judge pass) -- grouped by exam-set ground truth
    tail_by_kind = collections.defaultdict(lambda: {"n": 0, "exact": 0})
    for x in per_row:
        k = x["gold_tail_kind"]
        tail_by_kind[str(k)]["n"] += 1
        tail_by_kind[str(k)]["exact"] += int(x["exact"])

    # 9) corpus side: per-family chain length x tail-kind composition (the oversampling surface of the weighted design)
    corp_stats = collections.defaultdict(lambda: {"n": 0, "tail_kind": collections.Counter(),
                                                  "chain_len": collections.Counter()})
    for nm, ci in corp.items():
        f = ci["family"]
        corp_stats[f]["n"] += 1
        corp_stats[f]["tail_kind"][str(tail_kind((ci["chain_ops"] or [None])[-1]))] += 1
        corp_stats[f]["chain_len"][ci["n_steps"]] += 1
    corp_out = {f: {"n": v["n"], "tail_kind": dict(v["tail_kind"]),
                    "chain_len": {str(k): c for k, c in sorted(v["chain_len"].items())}}
                for f, v in corp_stats.items()}

    # 10) terminal-step decision point: association of the gold tail kind with problem features (cls_sq / rho / resid_nonzero)
    feat_tail = collections.defaultdict(collections.Counter)
    for x in per_row:
        f = x["family"]
        c = x["corp"]
        key = "%s|cls_sq=%s|rho=%s" % (f, c.get("cls_sq"), (c.get("rho") or "?")[:1])
        feat_tail[key][str(x["gold_tail_kind"])] += 1

    rep = {
        "schema": "gen36-erranalysis/1",
        "src_walk": a.walk,
        "src_walk_table": W.get("table"),
        "n": n_all, "n_exact": len(good), "n_bad": len(bad),
        "exact_rate": round(len(good) / max(n_all, 1), 6),
        "A_family": {f: v for f, v in sorted(fam.items())},
        "B_first_div_pos_hist": {str(k): c for k, c in sorted(pos_hist.items())},
        "B_first_div_from_end_hist": {str(k): c for k, c in sorted(from_end_hist.items())},
        "C_confusion_gold_to_pred": dict(conf.most_common(40)),
        "C_confusion_by_family": {f: dict(c.most_common(20)) for f, c in conf_by_fam.items()},
        "D_tail_decoy_confusion": dict(tail_conf.most_common()),
        "D_tail_pos_rows": len(tail_pos_rows),
        "E_err_class": {"only_tail_single": len(only_tail), "mid_err": len(mid_err),
                        "multi_mismatch": len(multi_err),
                        "early_stop": len(early_stop), "over_run": len(over_run)},
        "F_confidence": {"err_p_max_mean": mean(err_pmax),
                         "err_p_max_min": (min(err_pmax) if err_pmax else None),
                         "err_p_max_max": (max(err_pmax) if err_pmax else None),
                         "err_margin_mean": mean(err_marg),
                         "ok_margin_med_mean": mean(ok_med),
                         "n_confident_wrong(p>=0.9)": len(confident_wrong),
                         "n_err": len(bad)},
        "G_tail_kind_by_row_truth": {k: v for k, v in sorted(tail_by_kind.items())},
        "H_corpus_family_stats": corp_out,
        "I_feat_tail": {k: dict(v) for k, v in sorted(feat_tail.items())},
        "rows_md5_src": None,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    with open(a.rows_out, "w", encoding="utf-8") as f:
        for x in per_row:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(json.dumps({"out": a.out, "n": n_all, "n_bad": len(bad),
                      "first_div_pos_hist": rep["B_first_div_pos_hist"],
                      "from_end_hist": rep["B_first_div_from_end_hist"],
                      "err_class": rep["E_err_class"],
                      "tail_conf": rep["D_tail_decoy_confusion"],
                      "conf_top": dict(list(conf.most_common(8)))},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
