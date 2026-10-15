# -*- coding: utf-8 -*-
# gen33_ctrl2.py -- two supplementary readings (closure required for honesty):
#   [C1] **M4_D**: naked (zero-training base) 30 renders -> batch judging by the frozen judge module (expected ≈ 0)
#   [C2] **reverse check 12**: model certified pool = 0 ⇒ a reverse check built on model outputs is **infeasible**
#        (registered as such); instead run the same reverse protocol on a **gold-proof** substrate
#        ((R1) terminal truncation, (R2) cross-problem proof swap) -- goal unchanged: show the
#        judge module is non-trivial (it can reject), and that the substrate itself is certified=True
#        under the frozen judging convention.
# Writes only scratch/ccp/gen33/**; judge module drct_common imported unchanged.
import argparse
import json
import os
import random
import time

import g33_common as C
import gen33_judge as J

G33 = C.G33
OUT = os.path.join(G33, "out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reverse", type=int, default=12)
    ap.add_argument("--out", default=OUT + "/GEN33_CTRL2.json")
    ap.add_argument("--log", default=G33 + "/logs/gen33.log")
    a = ap.parse_args()
    C.lean_gate(5, a.log)
    jm0 = C.md5f(os.path.join(C.DRCT, "drct_common.py"))
    assert jm0 == C.JUDGE_MD5_EXPECT, jm0
    rep = {"schema": "gen33-ctrl2/1",
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "judge_md5_before": jm0}

    # ---------------- C1: naked judging ----------------
    nk = OUT + "/GEN33_EVAL_naked_exam_deep_n30.json"
    if os.path.exists(nk):
        d = json.load(open(nk, encoding="utf-8"))
        side = {int(json.loads(l)["ri"]): json.loads(l)
                for l in open(OUT + "/DEEP_EXAM_SIDECAR.jsonl", encoding="utf-8")
                if l.strip()}
        items = []
        for r in d["rows"]:
            s = side[r["ri"]]
            items.append({"name": r["name"], "head": s["head"], "concl": s["concl"],
                          "proof_text": (r.get("render") or {}).get("body") or "",
                          "cls": r["cls"], "arm_tag": "naked"})
        jr = J.batch_judge(items, "naked_n30", chunk=30, logp=a.log)
        tp = OUT + "/GEN33_PRESCREEN_TRACE_naked_n30.jsonl"
        C.jlsave(jr["raw"], tp)
        n_nonempty = sum(1 for x in items if x["proof_text"].strip())
        rep["M4_D_naked"] = {"n": len(jr["raw"]), "certified": jr["certified"],
                             "certified_rate": round(jr["certified"] / max(1, len(jr["raw"])),
                                                     6),
                             "n_proof_nonempty": n_nonempty,
                             "table": d["table"], "trace": tp,
                             "trace_md5": C.md5f(tp), "judge_md5": jr["judge_md5"],
                             "note": "零训练底座 (SmolLM2-135M) 直出; 预期 ≈ 0"}
        C.log("M4_D naked certified=%d/%d (nonempty=%d)"
              % (jr["certified"], len(jr["raw"]), n_nonempty), a.log)
    else:
        rep["M4_D_naked"] = {"n": None, "note": "naked EVAL 件缺失"}

    # ---------------- C2: reverse check (gold substrate) ----------------
    rows = C.jlload(C.DEEP_CORPUS)
    by_name = {r["name"]: r for r in rows}
    held = [r for r in rows
            if int(__import__("hashlib").md5(r["name"].encode("utf-8")).hexdigest()[:8], 16)
            % 10 == 7]
    rnd = random.Random(20260923)
    picks = rnd.sample(sorted(held, key=lambda x: x["name"]),
                       min(a.n_reverse, len(held)))
    import drct_common as DC
    wd = os.path.join(G33, "lean", "reverse_gold")
    os.makedirs(wd, exist_ok=True)

    def judge(nm, head, concl, proof, tag):
        try:
            j = DC.judge_attempt(nm, head, concl, proof, wd, tag)
            return bool(j.get("certified")), j.get("reasons")
        except Exception as e:                                        # noqa: BLE001
            return None, ["EXC:%s" % e]

    recs = []
    for i, x in enumerate(picks):
        lines = [l for l in x["proof_text"].split("\n") if l.strip()]
        keep = max(1, int(len(lines) * 0.6))
        trunc = "\n".join(lines[:keep])
        c0, r0 = judge(x["name"], x["head"], x["concl"], x["proof_text"], "rg_base")
        c1, r1 = judge(x["name"], x["head"], x["concl"], trunc, "rg_trunc")
        y = picks[(i + 1) % len(picks)]
        c2, r2 = judge(x["name"], x["head"], x["concl"], y["proof_text"], "rg_swap")
        recs.append({"name": x["name"], "n_lines": len(lines), "n_keep": keep,
                     "gold_certified": c0, "R1_trunc_certified": c1, "R1_reasons": r1,
                     "R2_swap_from": y["name"], "R2_swap_certified": c2, "R2_reasons": r2})
        C.log("REV_GOLD %s gold=%s R1_trunc=%s R2_swap=%s (src=%s)"
              % (x["name"], c0, c1, c2, y["name"]), a.log)
    rep["reverse_gold"] = {
        "schema": "gen33-reverse-gold/1", "n": len(recs),
        "gold_certified": sum(1 for r in recs if r["gold_certified"] is True),
        "R1_trunc_rejected": sum(1 for r in recs if r["R1_trunc_certified"] is False),
        "R2_swap_rejected": sum(1 for r in recs if r["R2_swap_certified"] is False),
        "R1_unexpectedly_certified": [r["name"] for r in recs
                                      if r["R1_trunc_certified"] is not False],
        "R2_unexpectedly_certified": [r["name"] for r in recs
                                      if r["R2_swap_certified"] is not False],
        "substrate": "金标语料证明 (留出段抽样 12 题, 自身 certified=True)",
        "why": ("模型 certified 池 = 0/218 (两臂) ⇒ 按『模型产物』反向核验不可行 (如实登记); "
                "改以金标底物同式反向, 目的不变 = 证判卷器非平凡"),
        "records": recs}

    # ---------------- feasibility record for the model-side reverse ----------------
    n_cert = 0
    for tg in ("base", "attn"):
        tp = OUT + "/GEN33_PRESCREEN_TRACE_%s.jsonl" % tg
        if os.path.exists(tp):
            n_cert += sum(1 for l in open(tp, encoding="utf-8")
                          if l.strip() and json.loads(l).get("certified"))
    rep["reverse_model_side"] = {"n_certified_pool": n_cert, "n": 0,
                                 "note": "certified 池为 0 ⇒ 反向核验不可做 (如实登记)"}
    rep["judge_md5_after"] = C.md5f(os.path.join(C.DRCT, "drct_common.py"))
    rep["judge_unchanged"] = bool(rep["judge_md5_after"] == jm0 == C.JUDGE_MD5_EXPECT)
    rep["md5"] = C.save_json(a.out, rep)
    C.log("CTRL2 DONE md5=%s naked=%s/%s rev_gold R1_rej=%s R2_rej=%s judge_unchanged=%s"
          % (rep["md5"], (rep["M4_D_naked"] or {}).get("certified"),
             (rep["M4_D_naked"] or {}).get("n"), rep["reverse_gold"]["R1_trunc_rejected"],
             rep["reverse_gold"]["R2_swap_rejected"], rep["judge_unchanged"]), a.log)
    print("GEN33_CTRL2_RC=0 -> %s" % a.out)
    return rep


if __name__ == "__main__":
    main()
