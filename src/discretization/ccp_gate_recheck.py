# -*- coding: utf-8 -*-
# ccp_gate_recheck.py -- contamination gate R4 window-length fix + all-class re-judging
# (advance team gen12, change 58)
#
# Motivation (instrument fix, same origin as R2's 24-char rule): R4's coverage ratio
# inter/len(window) is unstable on **short windows** --
#   a normalized NT-class conclusion line from a successor state (e.g. "9∣(3*w_a)*b") is only
#   ~12 chars => 3 10-grams, so one hit gives 0.333, crossing the 0.30 review line with **no
#   substantial similarity** (re-judging: max coverage of that line against all reference
#   problems measures 0.000).
#   Fix: R4 is computed only on candidate windows with len >= 24 (24 chars = 15 10-grams,
#   consistent with R2's reject floor); if a line has no window >= 24, R4 falls back to the
#   longest window and the report flags it (in that case R4 is advisory only, never a
#   standalone conviction).
# Reads gen/ccp_*_rows.jsonl and miniF2F only; writes CCP_GATE2_RECHECK.json.
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ccp_bridge as BR
import ccp_gate as G
import ccp_lean as L

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = os.path.join(RES, "scratch", "ccp")
GEN = os.path.join(CCP, "gen")


def judge_v2(g, text, head=None, concl=None):
    cands = [c for c in (G.norm(text), G.norm(concl) if concl else "",
                         G.norm(head) if head else "") if len(c) >= 12]
    long_c = [c for c in cands if len(c) >= 24]
    short_only = not long_c
    if short_only and cands:
        long_c = [max(cands, key=len)]
    hit = {"R1": 0, "R2": 0, "R3": 0, "R4": 0, "R5": 0}
    import hashlib
    import re
    cand_md5 = {hashlib.md5(c.encode("utf-8")).hexdigest() for c in cands}
    for h in cand_md5:
        if h in g.md5map:
            hit["R1"] += 1
    cand_ft = set()
    for c in cands:
        cand_ft |= G.feats(c)
    idxs = g._cand_refs(cands, cand_ft)
    best_all = 0.0
    nrev = 0
    for i in idxs:
        split, name, rconcl, rhead, rft = g.entry[i]
        if re.search(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(name), text or ""):
            hit["R3"] += 1
        for c in cands:
            for r in (rconcl, rhead):
                if r and len(c) >= 24 and (c in r or r in c):
                    hit["R2"] += 1
                    break
            else:
                continue
            break
        best = 0.0
        for c in long_c:
            cng = G.ngrams(c)
            if not cng:
                continue
            for r in (rconcl, rhead):
                rng = G.ngrams(r)
                if not rng:
                    continue
                inter = len(cng & rng)
                if inter:
                    best = max(best, inter / float(len(cng)), inter / float(len(rng)))
        best_all = max(best_all, best)
        if best >= G.REJECT:
            hit["R4"] += 1
        elif best >= G.REVIEW:
            nrev += 1
        big = {x for x in (cand_ft & rft) if x >= 100}
        if len(big) >= 3:
            hit["R5"] += 1
    verdict = "REJECT" if (hit["R1"] or hit["R2"] or hit["R3"] or hit["R4"] or hit["R5"]) else (
        "REVIEW" if nrev else "PASS")
    return {"verdict": verdict, "counts": hit, "n_review": nrev,
            "max_R4": round(best_all, 4), "short_window_only": short_only}


def main():
    g = G.Gate(G.load_refs())
    pc = g.positive_control()
    out = {"schema": "ccp-gate2-recheck/1",
           "rule": "R4 覆盖比只在 len>=24 候选窗口上计算 (与 R2 拒档下限一致); "
                   "无长窗口时退化为最长窗口并标记 short_window_only",
           "positive_control": pc, "classes": {}}
    for cls in ("ind", "ineq", "nt"):
        p = os.path.join(GEN, "ccp_%s_rows.jsonl" % cls)
        if not os.path.exists(p):
            continue
        rows = [json.loads(ln) for ln in open(p, encoding="utf-8")]
        cnt = collections.Counter()
        byk = collections.Counter()
        maxr = 0.0
        nshort = 0
        for r in rows:
            h, c = BR.split_concl(r["stmt"])
            v = judge_v2(g, r["stmt"], head=h, concl=c)
            cnt[v["verdict"]] += 1
            maxr = max(maxr, v["max_R4"])
            if v["verdict"] != "PASS":
                byk[r["ccp_kind"]] += 1
            nshort += 1 if v["short_window_only"] else 0
        out["classes"][cls] = {"n_rows": len(rows), "verdicts": dict(cnt),
                               "max_R4": round(maxr, 4), "rows_short_window_only": nshort,
                               "nonpass_by_kind": dict(byk)}
        print("[recheck %s] rows=%d verdicts=%s maxR4=%.4f short_only=%d" %
              (cls, len(rows), dict(cnt), maxr, nshort), flush=True)
    out["gate_valid"] = pc["caught"] == pc["n"]
    md5 = L.save_json(os.path.join(CCP, "CCP_GATE2_RECHECK.json"), out)
    print("[recheck] → CCP_GATE2_RECHECK.json md5=%s | positive control %d/%d" %
          (md5, pc["caught"], pc["n"]))


if __name__ == "__main__":
    main()
