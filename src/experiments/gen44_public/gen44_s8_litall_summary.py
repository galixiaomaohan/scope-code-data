# -*- coding: utf-8 -*-
# gen44_s8_litall_summary.py -- [S8F] summary of the full literal-convention re-judging (LITALL, five artifacts)
#
# Input: out/GEN44_S8_REJUDGE_LITALL_s0..s4.json (per row, lit_certified = the frozen judging module
#   judging each single theorem row)
# Output: out/GEN44_S8_LITALL_SUMMARY.json
#   1) overall / per-volume / per-family **literal-convention certified** + Wilson 95%
#   2) **two-way decomposition** against the first-pass batch judging: false accepts corrected
#      (batch cert -> lit rej) / false rejects promoted (batch rej -> lit cert) / net delta
#   3) reject-reason histogram (literal convention) + agreement statistics
# Convention: same judging module (drct_common.py 7c31b58d...), same rendered text (proof_md5
#   agreement with the manifest must be 100%)
import collections
import glob
import hashlib
import json
import os
import time

# Experiment-tree resolution (same convention as gen44_common): the inputs
# and outputs live under the gen44 tree of SCOPE_RESEARCH_ROOT on the
# experiment host; the shipped summary this script produced is kept at
# data/raw/server_only/gen44/ in this package.
RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
HERE = os.path.join(RES, "scratch", "ccp", "gen44")
OUT = os.path.join(HERE, "out")


def wilson(k, n, z=1.959964):
    if n <= 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def md5f(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main():
    files = sorted(glob.glob(os.path.join(OUT, "GEN44_S8_REJUDGE_LITALL_s*.json")))
    rows, prov = [], []
    for p in files:
        d = json.load(open(p, encoding="utf-8"))
        rows += d["rows"]
        prov.append({"file": os.path.basename(p), "md5": md5f(p), "n": d["n_picked"],
                     "n_agree": d["n_agree"], "wall_s": d["wall_s"],
                     "judge_ok": d["judge"]["ok"], "created": d.get("created")})
    n = len(rows)
    ok = [r for r in rows if r.get("lit_certified") is not None]
    lit = sum(1 for r in ok if r["lit_certified"])
    bat = sum(1 for r in ok if r["batch_certified"])
    fa = sum(1 for r in ok if r["batch_certified"] and not r["lit_certified"])   # false accepts corrected
    fr = sum(1 for r in ok if not r["batch_certified"] and r["lit_certified"])   # false rejects promoted
    agree = sum(1 for r in ok if r["agree"])
    md5_same = sum(1 for r in ok if r.get("proof_md5_same_as_manifest"))

    def bucket(r):
        v = r.get("vol") or "?"
        return v if v.startswith("A1") else (r.get("fam") or "?")

    by = collections.defaultdict(lambda: {"n": 0, "lit": 0, "bat": 0, "fa": 0, "fr": 0,
                                          "n_bn1": 0, "fa_bn1": 0, "fr_bn1": 0})
    for r in ok:
        b = by[bucket(r)]
        b["n"] += 1
        b["lit"] += 1 if r["lit_certified"] else 0
        b["bat"] += 1 if r["batch_certified"] else 0
        b["fa"] += 1 if (r["batch_certified"] and not r["lit_certified"]) else 0
        b["fr"] += 1 if (not r["batch_certified"] and r["lit_certified"]) else 0
        if r.get("batch_main_batch_n") == 1:
            b["n_bn1"] += 1
            b["fa_bn1"] += 1 if (r["batch_certified"] and not r["lit_certified"]) else 0
            b["fr_bn1"] += 1 if (not r["batch_certified"] and r["lit_certified"]) else 0
    table = {}
    for k, v in sorted(by.items()):
        table[k] = dict(v, lit_rate=round(v["lit"] / max(1, v["n"]), 4),
                        lit_wilson95=wilson(v["lit"], v["n"]),
                        bat_rate=round(v["bat"] / max(1, v["n"]), 4))
    rej_reasons = collections.Counter()
    for r in ok:
        if not r["lit_certified"]:
            for x in (r.get("lit_reasons") or ["?"]):
                rej_reasons[x] += 1
    # Share of literal reject reasons that come from rows the batch pass had accepted by mistake.
    fa_reasons = collections.Counter()
    for r in ok:
        if r["batch_certified"] and not r["lit_certified"]:
            for x in (r.get("lit_reasons") or ["?"]):
                fa_reasons[x] += 1
    rep = {"schema": "gen44-s8-litall-summary/1",
           "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "provenance": prov, "n_rows": n, "n_scored": len(ok),
           "caliper": {"n_judged": len(ok), "lit_certified": lit,
                       "rate": round(lit / max(1, len(ok)), 4),
                       "wilson95": wilson(lit, len(ok)),
                       "batch_certified_first_pass": bat,
                       "batch_rate": round(bat / max(1, len(ok)), 4),
                       "net_delta_lit_minus_batch": lit - bat},
           "two_way": {"n_agree": agree,
                       "n_disagree": len(ok) - agree,
                       "false_accept_corrected": fa,          # batch cert -> literal rej (false accept corrected)
                       "false_reject_promoted": fr,           # batch rej -> literal cert (false reject promoted)
                       "net": lit - bat, "identity_check": fr - fa == lit - bat,
                       "share_of_disagree": {"fa": round(fa / max(1, len(ok) - agree), 4),
                                             "fr": round(fr / max(1, len(ok) - agree), 4)}},
           "proof_md5_same_as_manifest": "%d/%d" % (md5_same, len(ok)),
           "by_vol_fam_literal": table,
           "lit_reject_reasons": dict(rej_reasons.most_common()),
           "lit_reject_reasons_on_corrected_false_accepts": dict(fa_reasons.most_common()),
           # NOTE: Chinese report payload value kept verbatim (data, not a log line).
           "note": ("字面口径 = 冻结判卷件 judge_attempt 逐行单 theorem 判读 (drct_common.py "
                    "7c31b58da9d7228cee31b9be581fe5dd, 判前判后双验); 与批判同渲染文本 "
                    "(proof_md5 == manifest, 一致率须 100%)。第一遍批判受 D-S8F-1 污染, 已降级为过程记录。")}
    jp = os.path.join(OUT, "GEN44_S8_LITALL_SUMMARY.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    m = md5f(jp)                       # md5 taken after a single write; the ledger goes to a **side file** (to avoid changing the hash by self-reference)
    with open(os.path.join(OUT, "GEN44_S8_LITALL_SUMMARY_MD5.json"), "w",
              encoding="utf-8") as f:
        json.dump({"file": os.path.basename(jp), "md5": m, "bytes": os.path.getsize(jp),
                   "created": rep["created"]}, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: rep[k] for k in ("n_rows", "caliper", "two_way",
                                          "proof_md5_same_as_manifest",
                                          "by_vol_fam_literal", "lit_reject_reasons")},
                     ensure_ascii=False, indent=1))
    print("out -> %s md5 %s (ledger: ..._MD5.json)" % (jp, m))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
