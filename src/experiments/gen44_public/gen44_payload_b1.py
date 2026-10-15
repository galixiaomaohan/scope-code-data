# -*- coding: utf-8 -*-
# gen44_payload_b1.py -- family-B1 payload file: **two** compile/render sets (exam single
#   rendering = gen44_render_exam_v3 semantics; gold enumeration = bounded envelope) plus
#   diversity readings and the payload-ness gate (single-slot chain damage).
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import gen44_bcore as B                                                  # noqa: E402
import gen44_bgate as G                                                  # noqa: E402

FAM = "b1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", default=FAM)
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    up = a.fam.upper()
    rows = [json.loads(l) for l in open(os.path.join(B.OUT, "GEN44_%s_CORPUS.jsonl" % up),
                                        encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    comp, hist_e, hist_g, ng = [], collections.Counter(), collections.Counter(), []
    for r in rows:
        d = B.parse_stmt_ext(B.P(r["name"], r["head"], r["concl"], None, None))
        if not d["ok"]:
            continue
        te = B.render_tree(d["tree"], "exam")
        tg = B.render_tree_multi(d["tree"], "gold", cap=a.cap)
        se = "\n".join(te) if te else None
        if se:
            hist_e[se] += 1
        ng.append(len(tg))
        for t in tg:
            hist_g[t] += 1
        comp.append({"name": r["name"], "exam_md5": B.md5s(se or ""), "n_gold": len(tg),
                     "gold_md5": [B.md5s(t) for t in tg], "chain_ops": r["chain_ops"]})
    with open(os.path.join(B.OUT, "GEN44_%s_COMPILED.jsonl" % up), "w",
              encoding="utf-8") as f:
        for c in comp:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    load = G.payload_tests(a.fam, rows)
    rep = {"schema": "gen44-b-payload/1", "fam": a.fam, "n": len(comp),
           "diversity": {"n_distinct_exam": len(hist_e),
                         "n_distinct_gold": len(hist_g),
                         # NOTE: Chinese report payload values kept verbatim (data, not log lines).
                         "gate": ">= 6 种" if len(hist_e) >= 6 else "< 6 (登记)",
                         "exam_hist": [[k[:60], v] for k, v in hist_e.most_common(12)],
                         "mean_gold_per_row": round(sum(ng) / max(1, len(ng)), 2)},
           "load_axis": load,
           "renderer_md5": {"exam_v3": B.md5f(os.path.join(HERE, "gen44_render_exam_v3.py")),
                            "gold": B.md5f(os.path.join(HERE, "gen44_render_gold.py"))},
           "exam_single_distinct_shortfall": bool(len(hist_e) < 6),
           "verdict": ("PASS" if (load["verdict"] == "PASS"
                                  and (len(hist_e) >= 6 or len(hist_g) >= 6)) else "FAIL")}
    B.jdump(rep, os.path.join(B.OUT, "GEN44_%s_PAYLOAD.json" % up))
    print(json.dumps(rep, ensure_ascii=False, indent=1)[:1200])


if __name__ == "__main__":
    main()
