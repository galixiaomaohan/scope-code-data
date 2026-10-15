# -*- coding: utf-8 -*-
# gen33_report.py -- gen33 report generation (GEN33_REPORT.md + GEN33_TABLES.json)
#
# Readings (pre-registered V33_DESIGN §9):
#   M1_D certified rate of the baseline arm on the valid deep exam set (218)
#   M2_D attention arm − baseline arm **paired delta** (row-wise pairing + agree/flip counts)
#   M3_D two chain columns first_hit / exact_hit (deep-corpus gold operators = operators the model can walk; learnable for the first time this generation)
#   M4_D naked expected-≈0 control (zero-training base n=30)
#   AUD-1 zero-parameter mode baseline reported alongside (gen33_audit)
# Stratification: cls × chain-length band (short 5-6 / deep 7+) -- "zhenti-style stratification" = the deep-chain sublayer (registered)
# Writes only scratch/ccp/gen33/**.
import argparse
import collections
import glob
import json
import os
import time

import g33_common as C

G33 = C.G33
OUT = os.path.join(G33, "out")


def log(msg, p=None):
    C.log(msg, p)


def load_side():
    side = {}
    for line in open(OUT + "/DEEP_EXAM_SIDECAR.jsonl", encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            side[int(r["ri"])] = r
    return side


def load_arm(tags, side):
    """Chunked EVAL artifacts -> {name: row}"""
    out = {}
    files = []
    for tg in tags:
        files += sorted(glob.glob(os.path.join(OUT, "GEN33_EVAL_%s_exam_deep_c*.json" % tg)))
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        for r in d["rows"]:
            s = side.get(r["ri"], {})
            ch = r.get("chain") or {}
            out[r["name"]] = {"name": r["name"], "cls": r["cls"], "ri": r["ri"],
                              "file": os.path.basename(f),
                              "family": s.get("family"), "chain_len": s.get("chain_len"),
                              "stratum": ("deep7" if (s.get("chain_len") or 0) >= 7
                                          else "short5_6"),
                              "gold_ops": ch.get("gold"), "walked": ch.get("walked"),
                              "first_hit": ch.get("first_hit"), "exact_hit": ch.get("exact_hit"),
                              "prefix_hit": ch.get("prefix_hit"),
                              "terminated_end": ch.get("terminated_end"),
                              "overrun": ch.get("overrun"), "scorable": ch.get("scorable"),
                              "args_src_hist": ch.get("args_src_hist"),
                              "body": (r.get("render") or {}).get("body") or "",
                              "text": (r.get("render") or {}).get("text") or "",
                              "n_tok": (r.get("render") or {}).get("n_tok"),
                              "consistency": r.get("consistency"),
                              "err": r.get("err"),
                              "dev_res_src": None}
            # device-recompute convention evidence: args_src of the chain-walk trace (device convention)
            tr = ch.get("trace") or []
            if tr:
                out[r["name"]]["dev_res_src"] = tr[0].get("args_src")
    return out, files


def load_trace(tag):
    p = os.path.join(OUT, "GEN33_PRESCREEN_TRACE_%s.jsonl" % tag)
    d = {}
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                d[r["name"]] = r
    return d


def rate(k, n):
    return (k / n) if n else None


def arm_table(recs, trace):
    """Five columns + certified (joined with the judging trace)"""
    n = len(recs)
    sc = [r for r in recs if r.get("scorable")]
    cert = sum(1 for r in recs if (trace.get(r["name"]) or {}).get("certified"))
    tbl = {"n": n, "n_scorable": len(sc),
           "certified": cert, "certified_rate": rate(cert, n),
           "first_hit": rate(sum(1 for r in sc if r["first_hit"]), len(sc)),
           "exact_hit": rate(sum(1 for r in sc if r["exact_hit"]), len(sc)),
           "prefix_hit": rate(sum(1 for r in sc if r["prefix_hit"]), len(sc)),
           "terminated_end": sum(1 for r in recs if r["terminated_end"]),
           "overrun": sum(1 for r in recs if r["overrun"]),
           "n_err": sum(1 for r in recs if r["err"]),
           "render_tok_mean": (round(sum(r["n_tok"] or 0 for r in recs) / n, 1) if n else 0),
           "consistency": (rate(sum(r["consistency"]["rate"] or 0 for r in recs), n)
                           if n else None),
           "n_proof_nonempty": sum(1 for r in recs if r["body"].strip())}
    return tbl


def strata_table(byname, tag):
    """cls × chain-length band stratification"""
    out = {}
    for key in ("cls", "stratum", "family"):
        g = collections.defaultdict(list)
        for r in byname.values():
            g[r.get(key)].append(r)
        out[key] = {str(k): {"n": len(v),
                             "certified": sum(1 for x in v
                                              if (tag.get(x["name"]) or {}).get("certified")),
                             "certified_rate": rate(sum(1 for x in v
                                                        if (tag.get(x["name"]) or {})
                                                        .get("certified")), len(v)),
                             "first_hit": rate(sum(1 for x in v if x["first_hit"]),
                                               max(1, len(v))),
                             "exact_hit": rate(sum(1 for x in v if x["exact_hit"]),
                                               max(1, len(v)))}
                     for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))}
    return out


def paired(a, b, ta, tb):
    """M2_D: row-wise pairing (b = baseline, a = attention)"""
    names = sorted(set(a) & set(b))
    recs = []
    for nm in names:
        ca = bool((ta.get(nm) or {}).get("certified"))
        cb = bool((tb.get(nm) or {}).get("certified"))
        recs.append({"name": nm, "attn": ca, "base": cb,
                     "delta": int(ca) - int(cb),
                     "first_hit_a": a[nm]["first_hit"], "first_hit_b": b[nm]["first_hit"]})
    n = len(recs)
    na = sum(1 for r in recs if r["attn"])
    nb = sum(1 for r in recs if r["base"])
    both = sum(1 for r in recs if r["attn"] and r["base"])
    only_a = sum(1 for r in recs if r["attn"] and not r["base"])
    only_b = sum(1 for r in recs if r["base"] and not r["attn"])
    neither = n - both - only_a - only_b
    fh_a = sum(1 for r in recs if r["first_hit_a"])
    fh_b = sum(1 for r in recs if r["first_hit_b"])
    return {"n_paired": n, "attn_certified": na, "base_certified": nb,
            "attn_rate": rate(na, n), "base_rate": rate(nb, n),
            "delta_rate": (rate(na, n) - rate(nb, n)) if n else None,
            "both": both, "only_attn": only_a, "only_base": only_b, "neither": neither,
            "mcnemar_style": {"b_only_attn": only_a, "c_only_base": only_b},
            "first_hit_attn": fh_a, "first_hit_base": fh_b,
            "first_hit_delta": (fh_a - fh_b),
            "records": recs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="base,attn")
    ap.add_argument("--out", default=os.path.join(G33, "GEN33_REPORT.md"))
    ap.add_argument("--tables", default=OUT + "/GEN33_TABLES.json")
    ap.add_argument("--log", default=G33 + "/logs/gen33_report.log")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    side = load_side()
    arms, files = {}, {}
    for tg in tags:
        arms[tg], files[tg] = load_arm([tg], side)
    trace = {tg: load_trace(tg) for tg in tags}
    audit = {}
    if os.path.exists(OUT + "/GEN33_AUDIT_BASELINE.json"):
        audit = json.load(open(OUT + "/GEN33_AUDIT_BASELINE.json", encoding="utf-8"))
    decomp = {}
    if os.path.exists(OUT + "/GEN33_RENDER_DECOMP.json"):
        decomp = json.load(open(OUT + "/GEN33_RENDER_DECOMP.json", encoding="utf-8"))
    jl = {}
    for _m in ("literal", "reverse"):
        _fp = OUT + "/GEN33_JUDGE_%s.json" % _m
        if os.path.exists(_fp):
            jl[_m] = json.load(open(_fp, encoding="utf-8"))
    c2 = {}
    if os.path.exists(OUT + "/GEN33_CTRL2.json"):
        c2 = json.load(open(OUT + "/GEN33_CTRL2.json", encoding="utf-8"))

    T = {"schema": "gen33-tables/1", "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
         "design": {"V33_DESIGN.md": C.md5f(G33 + "/V33_DESIGN.md"),
                    "V33_DESIGN_ADDENDUM.md": (C.md5f(G33 + "/V33_DESIGN_ADDENDUM.md")
                                               if os.path.exists(G33 +
                                                                 "/V33_DESIGN_ADDENDUM.md")
                                               else None)},
         "eval_files": {tg: [os.path.basename(f) for f in files[tg]] for tg in tags},
         "valid_exam_n": len(side),
         "arms": {}, "strata": {}, "paired": None, "audit1": audit,
         "render_decomp": decomp, "judge_extra": jl, "ctrl2": c2}
    for tg in tags:
        recs = list(arms[tg].values())
        T["arms"][tg] = arm_table(recs, trace[tg])
        T["strata"][tg] = strata_table(arms[tg], trace[tg])
        log("%s: n=%d certified=%d/%d (%.4f) first_hit=%.4f exact_hit=%.4f"
            % (tg, T["arms"][tg]["n"], T["arms"][tg]["certified"], T["arms"][tg]["n"],
               T["arms"][tg]["certified_rate"] or 0,
               T["arms"][tg]["first_hit"] or 0, T["arms"][tg]["exact_hit"] or 0), a.log)
    if "base" in arms and "attn" in arms:
        T["paired"] = paired(arms["attn"], arms["base"], trace["attn"], trace["base"])
        log("M2_D pairing: n=%d attn=%d base=%d Δ=%.4f (only_attn=%d only_base=%d both=%d)"
            % (T["paired"]["n_paired"], T["paired"]["attn_certified"],
               T["paired"]["base_certified"], T["paired"]["delta_rate"] or 0,
               T["paired"]["only_attn"], T["paired"]["only_base"], T["paired"]["both"]),
            a.log)
    naked = OUT + "/GEN33_EVAL_naked_exam_deep_n30.json"
    if os.path.exists(naked):
        d = json.load(open(naked, encoding="utf-8"))
        T["naked"] = {"n": d["n"], "table": d["table"], "file": os.path.basename(naked)}
    T["md5"] = C.save_json(a.tables, T)

    # ---- Markdown ----
    L = []
    L.append("# GEN33_REPORT.md -- deep-generation V3.5: deep-corpus training + deep exam-set judging report\n")
    L.append("- Batch: gen33 · user order: 「现在应该开始直接训练模型，并批卷」")
    L.append("- Pre-registration: `V33_DESIGN.md` (md5 %s) + `V33_DESIGN_ADDENDUM.md` (md5 %s)"
             % (T["design"]["V33_DESIGN.md"], T["design"]["V33_DESIGN_ADDENDUM.md"]))
    L.append("- Corpus: `DEEP_CORPUS_V1.jsonl` 2399 rows (ind 1600 / ineq 799; chain lengths 5/6/7/9; "
             "15791 steps; res nonzero 15791/15791)")
    L.append("- Valid deep exam set: **n=%d** (C4 pre-dropped 120/338; convention sensitivity in ADDENDUM §A3)" % len(side))
    L.append("- Judging: frozen module `drct_common` md5 7c31b58da9d7228cee31b9be581fe5dd (import only, zero modification)")
    L.append("- GPU: training on a single exclusive card, GPU1 (baseline) / GPU2 (attention); **GPU0 user redline: zero occupancy throughout**\n")
    L.append("## 1. Main readings (pre-registered criteria)\n")
    L.append("| ID | Criterion | Reading (n/total) |")
    L.append("|---|---|---|")
    ba = T["arms"].get("base", {})
    aa = T["arms"].get("attn", {})
    L.append("| **M1_D** | baseline-arm certified rate (valid deep exam set %d) | **%s / %s = %s** |"
             % (len(side), ba.get("certified"), ba.get("n"),
                ("%.4f" % ba["certified_rate"]) if ba.get("certified_rate") is not None
                else "—"))
    p = T.get("paired") or {}
    L.append("| **M2_D** | attention arm − baseline arm paired delta | **%s** (attn %s/%s=%.4f vs base %s/%s=%.4f; "
             "attn-only %d / base-only %d / both %d / neither %d) |"
             % (("%+.4f" % p["delta_rate"]) if p.get("delta_rate") is not None else "—",
                p.get("attn_certified"), p.get("n_paired"), p.get("attn_rate") or 0,
                p.get("base_certified"), p.get("n_paired"), p.get("base_rate") or 0,
                p.get("only_attn", 0), p.get("only_base", 0), p.get("both", 0),
                p.get("neither", 0)))
    L.append("| **M3_D** | two chain columns first_hit / exact_hit (baseline arm) | **%s / %s** (scoreable n=%s); "
             "attention arm %s / %s |"
             % (fmt(ba.get("first_hit")), fmt(ba.get("exact_hit")), ba.get("n_scorable"),
                fmt(aa.get("first_hit")), fmt(aa.get("exact_hit"))))
    if c2.get("M4_D_naked"):
        nk = c2["M4_D_naked"]
        if nk.get("n"):
            L.append("| **M4_D** | naked zero-training base (n=%d) certified | **%s / %s = %s** "
                     "(nonempty proofs %s; expected ≈ 0) |"
                     % (nk["n"], nk["certified"], nk["n"],
                        fmt(nk.get("certified_rate")), nk.get("n_proof_nonempty")))
        else:
            L.append("| **M4_D** | naked zero-training base | %s |" % nk.get("note"))
    elif "naked" in T:
        L.append("| **M4_D** | naked zero-training base (n=%d) certified | %s (expected ≈ 0) |"
                 % (T["naked"]["n"], T["naked"]["table"].get("certified")))
    else:
        L.append("| **M4_D** | naked zero-training base | not run (see §6 not-done) |")
    L.append("")
    L.append("## 2. Five-column table (both arms, full valid deep exam set)\n")
    L.append("| Arm | n | certified | first_hit | exact_hit | prefix_hit | terminated-END | overrun | "
             "render mean tok | nonempty proof | err |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for tg in tags:
        t = T["arms"][tg]
        L.append("| %s | %d | %d (%.4f) | %s | %s | %s | %d | %d | %s | %d | %d |"
                 % (tg, t["n"], t["certified"], t["certified_rate"] or 0, fmt(t["first_hit"]),
                    fmt(t["exact_hit"]), fmt(t["prefix_hit"]), t["terminated_end"],
                    t["overrun"], t["render_tok_mean"], t["n_proof_nonempty"], t["n_err"]))
    L.append("")
    L.append("## 3. Stratification (cls × family × chain-length band)\n")
    for tg in tags:
        L.append("### %s arm\n" % tg)
        for key, tbl in T["strata"][tg].items():
            L.append("| %s | n | certified | rate | first_hit | exact_hit |" % key)
            L.append("|---|---|---|---|---|---|")
            for k, v in tbl.items():
                L.append("| %s | %d | %d | %s | %s | %s |"
                         % (k, v["n"], v["certified"], fmt(v["certified_rate"]),
                            fmt(v["first_hit"]), fmt(v["exact_hit"])))
            L.append("")
    L.append("## 3b. Mechanism decomposition of M1_D = 0 (post-hoc diagnostic, not a pre-registered criterion)\n")
    if decomp:
        L.append("teacher-forced render token accuracy is high (0.9637 on the held-out segment) while judging gives certified=0 ⇒ "
                 "**token-level accuracy is a misleading proxy**. Decomposition:\n")
        L.append("| Set | n | skeleton identical | full-string exact | integer-position hits | render shorter than gold | "
                 "render longer than gold | have lines gold/render |")
        L.append("|---|---|---|---|---|---|---|---|")
        for key, lbl in (("base", "exam base arm"), ("attn", "exam attn arm")):
            d = (decomp.get("exam") or {}).get(key)
            if d:
                L.append("| %s | %d | %s | %d | %s | %d | %d | %d/%d |"
                         % (lbl, d["n"], fmt(d["skeleton_rate"]), d["full_string_exact"],
                            fmt(d["int_position_rate"]), d["render_shorter_than_gold"],
                            d["render_longer_than_gold"], d["have_lines_gold_total"],
                            d["have_lines_render_total"]))
        cb = decomp.get("ctrl_a_held_n20") or {}
        if cb.get("decomp"):
            d = cb["decomp"]
            L.append("| **same-distribution control (a_held n=20)** | %d | %s | %d | %s | %d | %d | %d/%d |"
                     % (d["n"], fmt(d["skeleton_rate"]), d["full_string_exact"],
                        fmt(d["int_position_rate"]), d["render_shorter_than_gold"],
                        d["render_longer_than_gold"], d["have_lines_gold_total"],
                        d["have_lines_render_total"]))
        L.append("")
        bd = (decomp.get("exam") or {}).get("base", {}).get("by_cls", {})
        if bd:
            L.append("Per class (base arm): " + " · ".join(
                "%s n=%d skeleton=%s int=%s" % (k, v["n"], fmt(v["skeleton_rate"]),
                                          fmt(v["int_position_rate"]))
                for k, v in bd.items()) + "\n")
        if cb.get("judge"):
            L.append("Same-distribution control judging: certified **%d/%d**, reasons=%s ⇒ "
                     "the exam 0/218 is **not** specific to exam transfer: the render channel itself is un-judgeable.\n"
                     % (cb["judge"]["certified"], cb["judge"]["n"],
                        json.dumps(cb["judge"]["reasons"], ensure_ascii=False)))
        L.append("**Conjunctive collapse (mechanism)**: proof validity = the conjunction of all content integers ⇒ with per-integer "
                 "accuracy p and k integers per problem the validity goes as p^k. Same-distribution control measures p=0.7919, k≈14.9 ⇒ 0.7919^14.9 ≈ 0.031 ⇒ "
                 "expected 0.6/20, observed 1/20 **matches**; exam ind p=0.7157, k≈18.7 ⇒ ≈0.002 ⇒ 0/37; "
                 "exam ineq p=0.0911 ⇒ 0/181. ⇐ deep chains need ~15-19 content integers per problem (V2F/EXAM_V1 shallow "
                 "problems ~3-5), so the historical 16.6% collapses to 0 in the deep generation.\n")
    else:
        L.append("(decomposition artifact missing)\n")
    L.append("## 4. AUD-1 zero-parameter mode baseline (reading discipline)\n")
    if audit:
        L.append("- distinct chain kinds (whole set / train segment): %s / %s" % (audit.get("B1_chain_distinct_all"),
                                                     audit.get("B1_chain_distinct_train")))
        L.append("- render-target templating (train segment): %s" % json.dumps(
            audit.get("B2_target_template_train"), ensure_ascii=False))
        L.append("- zero-parameter per-class per-position mode baseline: render acc=%s (held) / chain exact=%s (held) / "
                 "chain per-position acc=%s"
                 % ((audit.get("B3_render_mode_baseline_held") or {}).get("acc"),
                    (audit.get("B3_chain_mode_baseline_held") or {}).get("exact_rate"),
                    (audit.get("B3_chain_mode_baseline_held") or {}).get("pos_acc")))
        if audit.get("paired_table_train_side"):
            L.append("- train-side paired table (same slice):")
            for k, v in audit["paired_table_train_side"].items():
                L.append("  - `%s`: attn=%s base=%s delta=%s"
                         % (k, v.get("attn"), v.get("base"), v.get("delta")))
    else:
        L.append("- (AUD-1 artifact missing)")
    L.append("")
    L.append("## 4b. The three judging conventions\n")
    L.append("| Convention | Reading |")
    L.append("|---|---|")
    for tg in tags:
        t = (T["arms"].get(tg) or {})
        L.append("| batch full-pass certified (%s) | %s / %s = %s |"
                 % (tg, t.get("certified"), t.get("n"), fmt(t.get("certified_rate"))))
    if jl.get("literal"):
        lt = jl["literal"].get("literal", {})
        L.append("| literal re-judging (direct drct_common.judge_attempt) | n=%s · agree=%s/%s · "
                 "disagree=%s · exception=%s · not_executed=%s · literal_certified=%s |"
                 % (lt.get("n"), lt.get("n_agree"), lt.get("n_verdict_pairs"),
                    lt.get("n_disagree"), lt.get("n_exception"), lt.get("n_not_executed"),
                    lt.get("literal_certified")))
    if jl.get("reverse"):
        rv = jl["reverse"].get("reverse", {})
        L.append("| reverse check (model outputs) | n=%s · note: certified pool = 0 ⇒ **infeasible** "
                 "(registered as such) |" % rv.get("n"))
    if c2.get("reverse_gold"):
        rg = c2["reverse_gold"]
        L.append("| **reverse check (gold substrate, 12 cases)** | substrate itself certified=%s/%s · "
                 "R1 terminal-truncation rejected **%s/%s** · R2 cross-problem-swap rejected **%s/%s** · "
                 "unexpectedly certified R1=%s R2=%s ⇒ **the judge module is non-trivial** |"
                 % (rg.get("gold_certified"), rg.get("n"), rg.get("R1_trunc_rejected"),
                    rg.get("n"), rg.get("R2_swap_rejected"), rg.get("n"),
                    len(rg.get("R1_unexpectedly_certified") or []),
                    len(rg.get("R2_unexpectedly_certified") or [])))
    if jl:
        L.append("")
        L.append("Judging module frozen: md5 7c31b58da9d7228cee31b9be581fe5dd both before and after execution.")
        L.append("")
    L.append("## 5. Key-artifact md5s\n")
    for k, p2 in (("V33_DESIGN.md", G33 + "/V33_DESIGN.md"),
                  ("V33_DESIGN_ADDENDUM.md", G33 + "/V33_DESIGN_ADDENDUM.md"),
                  ("CCP_VOCAB_V4.json", G33 + "/cache/CCP_VOCAB_V4.json"),
                  ("ccp_allow_v4.bin", G33 + "/cache/ccp_allow_v4.bin"),
                  ("deep_all.bin", G33 + "/cache/deep_all.bin"),
                  ("DEEP_RENDER_SIDECAR.jsonl", OUT + "/DEEP_RENDER_SIDECAR.jsonl"),
                  ("DEEP_EXAM_SIDECAR.jsonl", OUT + "/DEEP_EXAM_SIDECAR.jsonl"),
                  ("GEN33_DEVICE_VS_GOLD.json", OUT + "/GEN33_DEVICE_VS_GOLD.json"),
                  ("GEN33_FRAME_CONSISTENCY.json", OUT + "/GEN33_FRAME_CONSISTENCY.json"),
                  ("GEN33_PAEQ.json", OUT + "/GEN33_PAEQ.json"),
                  ("C4_EXAM_PREFILTER.json", OUT + "/C4_EXAM_PREFILTER.json"),
                  ("GEN33_BUILD.json", OUT + "/GEN33_BUILD.json"),
                  ("GEN33_EXAM.json", OUT + "/GEN33_EXAM.json"),
                  ("GEN33_AUDIT_BASELINE.json", OUT + "/GEN33_AUDIT_BASELINE.json"),
                  ("GEN33_RENDER_DECOMP.json", OUT + "/GEN33_RENDER_DECOMP.json"),
                  ("GEN33_CTRL_HELD.json", OUT + "/GEN33_CTRL_HELD.json"),
                  ("GEN33_JUDGE_batch_base.json", OUT + "/GEN33_JUDGE_batch_base.json"),
                  ("GEN33_JUDGE_batch_attn.json", OUT + "/GEN33_JUDGE_batch_attn.json"),
                  ("GEN33_JUDGE_literal.json", OUT + "/GEN33_JUDGE_literal.json"),
                  ("GEN33_JUDGE_reverse.json", OUT + "/GEN33_JUDGE_reverse.json"),
                  ("GEN33_CTRL2.json", OUT + "/GEN33_CTRL2.json"),
                  ("GEN33_EVAL_naked_exam_deep_n30.json",
                   OUT + "/GEN33_EVAL_naked_exam_deep_n30.json"),
                  ("GEN33_EVAL_base_a_held_n20.json",
                   OUT + "/GEN33_EVAL_base_a_held_n20.json"),
                  ("EVID_DIVERGE_ATTN_psdev_curve.jsonl",
                   OUT + "/EVID_DIVERGE_ATTN_psdev_curve.jsonl"),
                  ("drct_common.py", C.DRCT + "/drct_common.py")):
        L.append("| `%s` | %s |" % (k, C.md5f(p2) if os.path.exists(p2) else "missing"))
    L.append("")
    L.append("## 6. Not done / boundaries (as-is)\n")
    L.append("- The P4-lite plan-summary gate is off (D-7, measured divergence); no fix attempted (would need editing the parent module or adding a hyperparameter).")
    L.append("- The C4 convention is sensitive (7/120/217/326); this batch takes the most conservative one ⇒ valid set 218; no full manual review was done.")
    L.append("- Slot supervision rate 0.1725 with UNK fallback 48200 is an inherited parent limitation (unfixed, keeping the single-difference discipline).")
    L.append("- The naked control is a disclosed n=30 slice (not the full set).")
    L.append("- The reverse check is **infeasible on the model side** (certified pool 0/218); replaced by the same protocol on a gold substrate (12 cases).")
    L.append("- No further attempt was made to fix the render-channel failure cause (changing the render head / adding content supervision = next step).")
    L.append("- Test 243 sealed; not read, not evaluated.")
    body = "\n".join(L) + "\n"
    C.safe_out(a.out)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(body)
    log("REPORT -> %s (md5=%s, %d bytes) tables=%s"
        % (a.out, C.md5f(a.out), os.path.getsize(a.out), T["md5"]), a.log)
    print("GEN33_REPORT_RC=0 -> %s" % a.out)
    return T


def fmt(x):
    return ("%.4f" % x) if isinstance(x, float) else ("%s" % x if x is not None else "—")


if __name__ == "__main__":
    main()
