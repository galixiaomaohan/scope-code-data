# -*- coding: utf-8 -*-
# gen34_tables.py -- gen34 reading summary (three-arm five-column table + M1_E-int strata
#                    + p^k conjugate decomposition)
#
# Position: an after-the-fact **summary artifact** (not a pre-registered criterion).
# Three arms on the same exam under the same convention:
#   M1_E     certified (batch full run; judging artifact GEN34_PRESCREEN_TRACE_<tag>.jsonl)
#   M1_E-int single-integer accuracy (naive + content conventions; split by cls/family/
#            chain-length bucket and slot class)
#   M3_E     chain two columns: first_hit / exact_hit
#   M4_E     same-distribution control a_held n=20
#   **conjugate decomposition**: per-problem content-integer count k, single-integer
#   accuracy p, p^k expected vs measured certified
# Reads gen34/out + gen33/out; writes only scratch/ccp/gen34/**.
import argparse
import collections
import glob
import json
import os
import re

import gen34_common as C

G34, G33 = C.G34, C.G33
OUT = os.path.join(G34, "out")

_RE_NUM = re.compile(r"[0-9]+")


def lits(text):
    t = text or ""
    naive = [m.group(0) for m in _RE_NUM.finditer(t)]
    cont = []
    for m in _RE_NUM.finditer(t):
        a, b = m.start(), m.end()
        if a > 0 and (t[a - 1].isalnum() or t[a - 1] == "_"):
            continue
        if b < len(t) and (t[b].isalnum() or t[b] == "_"):
            continue
        cont.append(m.group(0))
    return naive, cont


def load_arm(tag):
    rows = []
    for f in sorted(glob.glob(OUT + "/GEN34_EVAL_%s_exam_deep_c*.json" % tag)):
        rows += json.load(open(f, encoding="utf-8"))["rows"]
    return rows


def load_trace(tag):
    p = os.path.join(OUT, "GEN34_PRESCREEN_TRACE_%s.jsonl" % tag)
    if not os.path.exists(p):
        return {}
    return {json.loads(l)["name"]: json.loads(l)
            for l in open(p, encoding="utf-8") if l.strip()}


def rate(a, b):
    return round(a / b, 6) if b else None


def table(rows, trace, key_of=None):
    n = len(rows)
    cert = 0
    fh = eh = ph = 0
    term = over = 0
    cons = 0
    nonempty = 0
    toks = 0
    n_ia = 0
    ia = {"naive_hit": 0, "naive_tot": 0, "content_hit": 0, "content_tot": 0}
    for r in rows:
        t = trace.get(r["name"]) or {}
        if t.get("certified"):
            cert += 1
        ch = r.get("chain") or {}
        if ch:
            fh += 1 if ch.get("first_hit") else 0
            eh += 1 if ch.get("exact_hit") else 0
            ph += 1 if ch.get("prefix_hit") else 0
            term += 1 if ch.get("terminated_end") else 0
            over += 1 if ch.get("overrun") else 0
        body = ((r.get("render") or {}).get("body") or "")
        if body.strip():
            nonempty += 1
        toks += int(((r.get("render") or {}).get("n_tok") or 0))
        cons += 1 if r.get("consistency") else 0
        if r.get("int_acc"):
            n_ia += 1
            for k in ia:
                ia[k] += int(r["int_acc"][k])
    return {"n": n, "certified": cert, "certified_rate": rate(cert, n),
            "first_hit": rate(fh, n), "exact_hit": rate(eh, n), "prefix_hit": rate(ph, n),
            "terminated_end": term, "overrun": over, "render_tok_mean": rate(toks, n),
            "nonempty_proof": nonempty, "consistency": rate(cons, n),
            "n_int_acc_rows": n_ia,
            "int_acc": dict(ia, naive_rate=rate(ia["naive_hit"], ia["naive_tot"]),
                            content_rate=rate(ia["content_hit"], ia["content_tot"]))}


def by_key(rows, trace, keyf):
    d = collections.defaultdict(list)
    for r in rows:
        d[keyf(r)].append(r)
    return {k: table(v, trace) for k, v in sorted(d.items(), key=lambda kv: str(kv[0]))}


def conj(rows):
    """Conjugate decomposition: per-problem content-integer count k and single-integer
    accuracy p (content convention), plus p^k."""
    ks, hit = [], 0
    tot = 0
    per = []
    for r in rows:
        a = r.get("int_acc")
        if not a:
            continue
        ks.append(a["content_tot"])
        hit += a["content_hit"]
        tot += a["content_tot"]
        per.append({"name": r["name"], "k": a["content_tot"],
                    "hit": a["content_hit"], "rate": a["content_rate"]})
    p = (hit / tot) if tot else None
    kmean = (sum(ks) / len(ks)) if ks else None
    return {"n_rows": len(ks), "content_hit": hit, "content_tot": tot,
            "p_content": None if p is None else round(p, 6),
            "k_mean": None if kmean is None else round(kmean, 3),
            "k_hist": dict(collections.Counter(ks).most_common()),
            "p_pow_k_expected": (None if (p is None or kmean is None)
                                 else round(p ** kmean, 6)),
            "all_k_correct_rows": sum(1 for x in per if x["k"] and x["hit"] == x["k"]),
            "example_worst": sorted(per, key=lambda x: x["rate"] or 0)[:3],
            "example_best": sorted(per, key=lambda x: -(x["rate"] or 0))[:3]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="slotA,slotB,base")
    ap.add_argument("--out", default=OUT + "/GEN34_TABLES.json")
    ap.add_argument("--log", default=G34 + "/logs/gen34.log")
    a = ap.parse_args()
    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    rep = {"schema": "gen34-tables/1", "tags": tags, "arms": {}, "ctrl_a_held": {},
           "note": ("after-the-fact summary artifact (not a pre-registered criterion); "
                    "M1_E/M1_E-int/M3_E/M4_E on the same exam under the same convention. "
                    "certified comes from the judging trace; token acc must not be cited "
                    "standalone (pitfall log N1).")}
    for tg in tags:
        rows = load_arm(tg)
        if not rows:
            continue
        tr = load_trace(tg)
        d = {"table": table(rows, tr),
             "by_cls": by_key(rows, tr, lambda r: r.get("cls")),
             "by_family": by_key(rows, tr, lambda r: r["name"].rsplit("_", 1)[0]),
             "by_chain_len": by_key(rows, tr, lambda r: len((r.get("chain") or {})
                                                           .get("gold") or []) or 0),
             "conjugate": conj(rows)}
        p = os.path.join(OUT, "GEN34_EVAL_%s_a_held_n20.json" % tg)
        if os.path.exists(p):
            cr = json.load(open(p, encoding="utf-8"))["rows"]
            d["ctrl_a_held_n20"] = table(cr, tr)
            d["ctrl_a_held_n20"]["conjugate"] = conj(cr)
            rep["ctrl_a_held"][tg] = d["ctrl_a_held_n20"]
        rep["arms"][tg] = d
    # paired difference M2_E
    if "slotA" in rep["arms"] and "slotB" in rep["arms"]:
        A, B = rep["arms"]["slotA"]["table"], rep["arms"]["slotB"]["table"]
        rep["M2_E_A_minus_B"] = {"certified": (A["certified"] - B["certified"]),
                                 "certified_rate": (None if A["certified_rate"] is None
                                                    or B["certified_rate"] is None else
                                                    round(A["certified_rate"]
                                                          - B["certified_rate"], 6)),
                                 "content_int_rate": (None if not A["int_acc"]["content_rate"]
                                                      or not B["int_acc"]["content_rate"] else
                                                      round(A["int_acc"]["content_rate"]
                                                            - B["int_acc"]["content_rate"], 6))}
    for base in ("base",):
        if "slotA" in rep["arms"] and base in rep["arms"]:
            A = rep["arms"]["slotA"]["table"]
            Z = rep["arms"][base]["table"]
            rep["M2_E_A_minus_C"] = {"certified": A["certified"] - Z["certified"],
                                     "certified_rate": (None if A["certified_rate"] is None
                                                        or Z["certified_rate"] is None else
                                                        round(A["certified_rate"]
                                                              - Z["certified_rate"], 6)),
                                     "content_int_rate": (None if not A["int_acc"]["content_rate"]
                                                          or not Z["int_acc"]["content_rate"]
                                                          else round(A["int_acc"]["content_rate"]
                                                                     - Z["int_acc"]["content_rate"],
                                                                     6))}
    rep["md5"] = C.save_json(a.out, rep)
    for tg in tags:
        if tg in rep["arms"]:
            t = rep["arms"][tg]["table"]
            ia = t["int_acc"]
            C.log("TABLE %s n=%d certified=%d/%d (%.4f) first=%s exact=%s | int naive=%.4f "
                  "content=%.4f (p=%s k=%s p^k=%s)"
                  % (tg, t["n"], t["certified"], t["n"], t["certified_rate"] or 0,
                     t["first_hit"], t["exact_hit"], ia["naive_rate"] or 0,
                     ia["content_rate"] or 0,
                     rep["arms"][tg]["conjugate"]["p_content"],
                     rep["arms"][tg]["conjugate"]["k_mean"],
                     rep["arms"][tg]["conjugate"]["p_pow_k_expected"]), a.log)
    print("GEN34_TABLES_RC=0 -> %s" % a.out)
    return rep


if __name__ == "__main__":
    main()
