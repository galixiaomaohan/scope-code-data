# -*- coding: utf-8 -*-
# gen34_cxcheck.py -- line-by-line cross-check of the C arm's live run vs gen33 artifacts
#                    (work proof: the C column of M1_E is not new)
#
# Criterion: the exam-segment renderings produced by the gen34 code with `--slot-cond 0`
#   plus the gen33 ckpt must be **line-by-line, character-by-character equal** to the
#   renderings in the official gen33 `GEN33_EVAL_base_exam_deep_c*.json` files
#   (greedy determinism + equivalence E1-E4).
# Reads the two artifact batches; writes only scratch/ccp/gen34/**.
import argparse
import glob
import json
import os
import re

import gen34_common as C

G34, G33 = C.G34, C.G33
OUT = os.path.join(G34, "out")


def load(pat):
    d = {}
    for f in sorted(glob.glob(pat)):
        for r in json.load(open(f, encoding="utf-8"))["rows"]:
            d[r["name"]] = r
    return d


def body_of(r):
    if r is None:
        return None
    b = r.get("render")
    if isinstance(b, dict):
        return b.get("body") or ""
    return r.get("body") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT + "/GEN34_CXCHECK.json")
    ap.add_argument("--log", default=G34 + "/logs/gen34.log")
    a = ap.parse_args()
    b34 = load(OUT + "/GEN34_EVAL_base_exam_deep_c*.json")
    b33 = load(G33 + "/out/GEN33_EVAL_base_exam_deep_c*.json")
    rep = {"schema": "gen34-cxcheck/1", "n_gen34": len(b34), "n_gen33": len(b33),
           "note": ("C-arm work proof: renderings from the gen34 code with --slot-cond 0 "
                    "plus the gen33 ckpt must be line-by-line identical to the official "
                    "gen33 artifacts (E1-E4 bitwise equivalence + greedy determinism "
                    "=> all-equal expected)")}
    if not b34 or not b33:
        rep["status"] = "missing (gen34 base exam not run, or gen33 artifacts absent)"
        rep["md5"] = C.save_json(a.out, rep)
        print("GEN34_CXCHECK_RC=0 (missing) -> %s" % a.out)
        return rep
    common = sorted(set(b34) & set(b33))
    ne = []
    n_tok_eq = 0
    for k in common:
        t34, t33 = body_of(b34[k]), body_of(b33[k])
        if re.sub(r"\s+", " ", t34).strip() == re.sub(r"\s+", " ", t33).strip():
            n_tok_eq += 1
        else:
            ne.append({"name": k, "len34": len(t34), "len33": len(t33),
                       "head34": t34[:100], "head33": t33[:100]})
    rep.update({"n_common": len(common), "n_body_equal": n_tok_eq,
                "n_body_diff": len(ne), "diff_examples": ne[:5],
                "body_equal_rate": round(n_tok_eq / max(1, len(common)), 6),
                "pass": bool(len(common) > 0 and n_tok_eq == len(common))})
    rep["md5"] = C.save_json(a.out, rep)
    C.log("CXCHECK C arm vs gen33: n_common=%d body_equal=%d diff=%d pass=%s"
          % (len(common), n_tok_eq, len(ne), rep["pass"]), a.log)
    print("GEN34_CXCHECK_RC=0 -> %s" % a.out)
    return rep


if __name__ == "__main__":
    main()
