# -*- coding: utf-8 -*-
# gen44_c6_excluded.py -- [S8 gap C6] **Deterministic recomputation** of the ill-formed (statement
# layer) exclusion list
#
# Background (GEN44_S8_HANDOFF §4.2(b)(1)): the S8 full-corpus denominator documented as 132,375 =
#   139,826 − **C6=7451**, but 7451 was obtained by hand-summing the category descriptions in
#   LWB_FRONTIER.md §C6 and **cannot be reproduced exactly from the on-disk artifacts** (summing
#   category by category across the eight slots gives 7445-7448, depending on whether
#   comma/arrow/iff/and include the H side). This file recomputes the §C6 categories **key by
#   key**, producing a deterministic list plus a ledger so the denominator is reproducible and
#   auditable:
#     denominator := 139,826 − |rows hit by C6_KEYS|      (measured here = 132,378; 3 rows above the
#     documented 132,375, registered as-is)
#
# §C6 category -> census V3 reject_hist key mapping (both sides are taken where the document lists
# pairs; ambiguous groups are registered twice):
#   freevar_undefined 5157 + H_unparsed:freevar 276 · greek_type 704 + H_unparsed:greek_type 25 ·
#   type_tok 99 · nonarith_token 226 + H_unparsed:nonarith_token 7 · lambda 39 ·
#   reldup 16 + H_unparsed:reldup 246 · dot_tok 403 + H_unparsed:dot_tok 49 · H_unparsed:quant 138 ·
#   ceil 2 + lex 1 · comma/arrow/iff/and: C side 51 + H side 9 (the document says 57 -- neither side
#   is 57 => this file takes **both sides** (consistent with the upper bound 132,378 = 139,826 −
#   7448) and registers the C-side-only variant as well)
#
# Reads: out/GEN44_CENSUS_V3.json · out/GEN44_CENSUS_ROWS_V3.jsonl
# Writes: out/GEN44_C6_EXCLUDED.jsonl (deterministic list) + out/GEN44_C6_RECOMPUTE.json (ledger).
# Overwrites no existing file.
import argparse
import collections
import hashlib
import json
import os
import time

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
OUT = os.path.join(RES, "scratch", "ccp", "gen44", "out")

C6_KEYS = [
    "C_unparsed:freevar_undefined", "H_unparsed:freevar",
    "C_unparsed:greek_type", "H_unparsed:greek_type",
    "C_unparsed:type_tok",
    "C_unparsed:nonarith_token", "H_unparsed:nonarith_token",
    "C_unparsed:lambda",
    "C_unparsed:reldup", "H_unparsed:reldup",
    "C_unparsed:dot_tok", "H_unparsed:dot_tok",
    "H_unparsed:quant",
    "C_unparsed:ceil", "C_unparsed:lex",
]
AMBIG_C = ["C_unparsed:comma", "C_unparsed:arrow", "C_unparsed:iff", "C_unparsed:and"]
AMBIG_H = ["H_unparsed:comma", "H_unparsed:iff", "H_unparsed:arrow"]
DOC_TOTAL = 7451
DOC_DENOM = 132375


def md5f(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default=os.path.join(OUT, "GEN44_CENSUS_V3.json"))
    ap.add_argument("--rows", default=os.path.join(OUT, "GEN44_CENSUS_ROWS_V3.jsonl"))
    ap.add_argument("--out", default=os.path.join(OUT, "GEN44_C6_EXCLUDED.jsonl"))
    ap.add_argument("--report", default=os.path.join(OUT, "GEN44_C6_RECOMPUTE.json"))
    a = ap.parse_args()

    cen = json.load(open(a.census, encoding="utf-8"))
    rh = cen["reject_hist"]
    N = int(cen["n"])
    keys_all = C6_KEYS + AMBIG_C + AMBIG_H
    missing = [k for k in keys_all if k not in rh]
    assert not missing, ("C6 key missing from census reject_hist", missing)
    by_key = {k: int(rh[k]) for k in keys_all}
    tot_core = sum(by_key[k] for k in C6_KEYS)
    tot_ambig_c = sum(by_key[k] for k in AMBIG_C)
    tot_ambig_h = sum(by_key[k] for k in AMBIG_H)
    tot_all = tot_core + tot_ambig_c + tot_ambig_h
    tot_c_only = tot_core + tot_ambig_c

    sel = []
    rs_hist = collections.Counter()
    key_set = set(keys_all)
    for line in open(a.rows, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("rs") in key_set:
            rs_hist[r["rs"]] += 1
            sel.append({"name": r.get("name"), "idx": r.get("idx"), "rs": r.get("rs"),
                        "sub": r.get("sub"), "kind": r.get("kind"),
                        "C": r.get("C"), "H": r.get("H"), "O": r.get("O")})
    # Cross assertion: row-level hit counts == reject_hist per-key counts (the recomputation must
    # reconcile).
    bad = {k: [int(rs_hist.get(k, 0)), by_key[k]] for k in keys_all
           if int(rs_hist.get(k, 0)) != by_key[k]}
    assert not bad, ("row-level counts disagree with reject_hist", bad)
    assert len(sel) == tot_all, (len(sel), tot_all)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in sel:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    md = md5f(a.out)
    denom = N - tot_all
    rep = {"schema": "gen44-c6-recompute/1", "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "source": {"census": a.census, "census_md5": md5f(a.census),
                      "rows": a.rows, "rows_md5": md5f(a.rows), "N_total": N},
           "C6_keys": keys_all,
           "by_key": by_key,
           "totals": {"core": tot_core, "ambig_C_side": tot_ambig_c, "ambig_H_side": tot_ambig_h,
                      "all_two_sides": tot_all, "C_side_only_variant": tot_c_only},
           "denominator": {"value": denom, "formula": "%d - %d" % (N, tot_all),
                           "variant_C_only": N - tot_c_only,
                           "doc_value": DOC_DENOM, "doc_C6": DOC_TOTAL,
                           "delta_vs_doc": denom - DOC_DENOM,
                           # NOTE: Chinese report payload value kept verbatim (data, not a log line).
                           "note": ("文书 7451/132375 由 §C6 类目手工相加, 不可精确复现; 本件复算 = %d "
                                    "(+歧义组两侧 9 = %d); 与文书差 %d 行, 登记为勘误, S8 分母取本件确定值"
                                    % (tot_core + tot_ambig_c, tot_all, denom - DOC_DENOM))},
           "excluded_list": {"path": a.out, "md5": md, "rows": len(sel),
                             "bytes": os.path.getsize(a.out)},
           # NOTE: Chinese report payload value kept verbatim (data, not a log line).
           "cross_check": "行级命中数 == reject_hist 逐键计数 (全键通过)"}
    json.dump(rep, open(a.report, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(rep["totals"], ensure_ascii=False))
    print(json.dumps(rep["denominator"], ensure_ascii=False))
    print(json.dumps(rep["excluded_list"], ensure_ascii=False))
    print("C6_RECOMPUTE_DONE")


if __name__ == "__main__":
    main()
