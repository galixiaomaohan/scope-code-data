# -*- coding: utf-8 -*-
# ccp_cache_verify.py -- CCP cache 4th roundtrip leg (ids cross-check) + build-time statistics
# recomputation (gen15, disposition F-13)
#
# Background (CCP_GEN12_AUDIT.md F-13, verdict: undecidable): the ccp_cache.py smoke run checked
#   3 of the 4 legs (row count / labels==feat_pos+1 / max(labels)<=nops), but **the 4th leg
#   (cached ids equivalent to build time)** and the build-time statistics slot_sup_rate /
#   max_len_labels / tok_trunc were never independently re-checked -- they need the tokenizer and
#   a build_records rerun (CPU + vocabulary, sharing the cache/ target directory with the training
#   batch cache).
#
# This script = the **independent executor** of that re-check, standalone and read-only:
#   leg 4 : rerun the pipeline's own tokenizer + ccp_cache.build_records over gen/ccp_<cls>_rows.jsonl,
#           then cross-check ids line by line against cache/ccp_<cls>_rows.bin (plus the deterministic
#           fields desc/res/pos_task_end/end_pos/readout/readout_only/labels/feat_pos).
#   stats : recompute slot_sup_rate / max_len_labels / tok_trunc / slot_vocab_sizes and cross-check
#           them against the registered values in cache/CCP_CACHE_SMOKE_<CLS>.json.
#   invariants: labels == feat_pos + 1; max(labels) <= nops (the END slot is the last of nops);
#           header n_rows/nops/head consistent with the installed operator space.
#
# Read-only guarantee: this script **writes no file** (no persistence branch; PYTHONDONTWRITEBYTECODE=1
#   is also set in-process); importing ccp_cache only does exist_ok mkdirs (cache/ and lean/ already
#   exist, so no write occurs).
#
# When to run (important): **during a training-batch idle window** -- this script reruns tokenization
#   (CPU) and reads cache/ and gen/ shared files; while the gen13 training batch is in flight those
#   caches/indexes may be under consumption, so do not run inside a batch; this is delivered as an
#   executor only.
# Usage:
#   cd "$SCOPE_RESEARCH_ROOT" && PYTHONPATH="${SCOPE_SUPEROPERATOR:-../device}" \
#     python scratch/ccp/ccp_cache_verify.py [--cls all|ind|ineq|nt]
# Exit code: 0 = all PASS; 1 = some FAIL (read stdout details).
#
# Known non-cross-checked item (by design, not a defect): `bsl` (bound-slot hash) uses Python's
#   builtin hash(str), which PYTHONHASHSEED randomizes => not reproducible across processes
#   (in-process consistency is all it needs), so it is excluded from the cross-check.
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ccp_cache as CC            # noqa: E402  (reuses build_records/read_cache, same convention as the smoke run)

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = os.path.join(RES, "scratch", "ccp")
GEN = os.path.join(CCP, "gen")
CACHE = os.path.join(CCP, "cache")
CLASSES = ("ind", "ineq", "nt")


def _pipeline():
    """Operator-space construction isomorphic to ccp_cache.main (V32 160 + CCP 7, appended at the tail)."""
    import importlib.machinery
    import importlib.util
    pc = CC._load_pipe()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(pc.st.BASE)
    ld = importlib.machinery.SourceFileLoader("vp_vfy", os.path.join(RES, "cppcuda/tools/v32b_prep.py"))
    sp = importlib.util.spec_from_loader("vp_vfy", ld)
    vp = importlib.util.module_from_spec(sp)
    ld.exec_module(vp)
    opidx = {n: i for i, n in enumerate(list(vp.NAMES32))}
    for n in CC.CCP_OPS:
        opidx[n] = len(opidx)
    return tok, opidx, len(opidx), vp.HEAD_V32 + len(CC.CCP_OPS)


def _close(a, b, tol=1e-4):
    if a is None or b is None:
        return a is None and b is None
    if len(a) != len(b):
        return False
    return all(abs(x - y) <= tol * max(1.0, abs(x), abs(y)) for x, y in zip(a, b))


def verify(cls, tok, opidx, nops, head):
    rows = [json.loads(ln) for ln in open(os.path.join(GEN, "ccp_%s_rows.jsonl" % cls),
                                          encoding="utf-8")]
    recs, t4set, stats, vals = CC.build_records(rows, tok, opidx)
    back, hdr = CC.read_cache(os.path.join(CACHE, "ccp_%s_rows.bin" % cls))
    smoke = json.load(open(os.path.join(CACHE, "CCP_CACHE_SMOKE_%s.json" % cls.upper()),
                           encoding="utf-8"))
    f = {}
    f["header_nops_head"] = (hdr["nops"] == nops and hdr["head"] == head and hdr["n_rows"] == len(recs))
    f["leg4_ids"] = all(b2["ids"] == r["ids"] for b2, r in zip(back, recs)) and len(back) == len(recs)
    f["labels_eq_featpos_plus1"] = all(len(b2["labels"]) == len(b2["feat_pos"]) + 1 for b2 in back)
    f["labels_le_nops"] = all(max(b2["labels"] + [0]) <= nops for b2 in back)
    f["desc_res_pos"] = all(b2["desc"] == r["desc"] and b2["pos_task_end"] == r["pos_task_end"]
                            and b2["end_pos"] == r["end_pos"] and b2["t4"] == r["t4"]
                            and b2["readout"] == r["readout"]
                            and b2["readout_only"] == r["readout_only"]
                            and b2["labels"] == r["labels"] and b2["feat_pos"] == r["feat_pos"]
                            and all(_close(x, y) for x, y in zip(b2["res"], r["res"]))
                            for b2, r in zip(back, recs))
    sup_rate = round(stats["slots_sup"] / max(1, stats["slots_all"]), 4)
    f["slot_sup_rate_match"] = (sup_rate == smoke["slot_sup_rate"])
    f["max_len_labels_match"] = (stats["L"] == smoke["max_len_labels"])
    f["tok_trunc_match"] = (stats["trunc"] == smoke["tok_trunc"])
    f["slot_vocab_match"] = ({"%s[%d]" % k: v for k, v in sorted(vals.items())}
                             == smoke["slot_vocab_sizes"])
    f["t4_match"] = ([list(x) for x in t4set] == smoke["t4"])
    ok = all(f.values())
    print("[verify %s] %s | rows %d | slot_sup_rate recomputed %.4f (registered %.4f) "
          "| max_len %d (registered %d) | trunc %d (registered %d)"
          % (cls, "PASS" if ok else "FAIL", len(recs), sup_rate,
             smoke["slot_sup_rate"], stats["L"], smoke["max_len_labels"],
             stats["trunc"], smoke["tok_trunc"]))
    for k, v in f.items():
        if not v:
            print("   FAIL: %s" % k)
    return ok, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", default="all")
    a = ap.parse_args()
    tok, opidx, nops, head = _pipeline()
    allok = True
    for cls in (list(CLASSES) if a.cls == "all" else [a.cls]):
        ok, _ = verify(cls, tok, opidx, nops, head)
        allok = allok and ok
    print("[verify] overall %s (read-only; no file written)" % ("PASS" if allok else "FAIL"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
