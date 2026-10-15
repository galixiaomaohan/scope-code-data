# -*- coding: utf-8 -*-
# gen44_vocab7.py -- [S3] vocabulary V6 -> V7 merge
#   Hard assertions: a7[:207].tobytes() == a6.tobytes() (the old 207 rows are byte-for-byte
#   unchanged) · sum(old segment) == 200605
#   New operator rows use the fixed index order 207..219 (per the pre-registration in
#   out/GEN44_OPS_V7_PLAN.json; if P2/P3 introduce new names, append 220..)
#   Artifacts: cache/CCP_VOCAB_V7.json · cache/ccp_allow_v7.bin · out/GEN44_VOCAB7_GATE.json
#   load_ctx7(): reused by gen44_cachebuild_v7 / gen44_eval (head_v7 / opmap / names_head / per slots)
# Reads: gen42/cache/{CCP_VOCAB_V6.json,ccp_allow_v6.bin}; the judging module is not involved.
# Write side: only scratch/ccp/gen44/**. Append-only semantics: only V7 artifacts are added, the V6
#   artifacts are never touched.
import argparse
import hashlib
import json
import os
import sys
import time

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = os.path.join(RES, "scratch", "ccp")
G42 = os.path.join(CCP, "gen42")
# the gen44 tree on the experiment host (cache/out live there)
G44 = os.path.join(CCP, "gen44")
CACHE = os.path.join(G44, "cache")
OUT = os.path.join(G44, "out")
for _p in (_PKG_DIR, G42):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                      # noqa: E402

MS = 24
NVALS = 352
HEAD_V6 = 207
V7_BASE = 207
OLD_SUM = 200605
V6_VOCAB = os.path.join(G42, "cache", "CCP_VOCAB_V6.json")
V6_ALLOW = os.path.join(G42, "cache", "ccp_allow_v6.bin")
PLAN = os.path.join(OUT, "GEN44_OPS_V7_PLAN.json")
V7_VOCAB = os.path.join(CACHE, "CCP_VOCAB_V7.json")
V7_ALLOW = os.path.join(CACHE, "ccp_allow_v7.bin")

# V7 new-operator slot specification (sfa paths; aligned with the step fields of the gen44 family
# corpora/trees)
#   kind = str|int ; path = the key of that step dict (the value becomes str(...) and enters the
#   digest vocabulary)
#   an empty list = the operator has no slottable arguments (fully determined by the chain
#   structure) => no slot row is generated
SLOT_V7 = {
    "g44_goal_split": [],
    "g44_intro_binder": [{"kind": "str", "path": "intros"}],
    "g44_witness_solve": [{"kind": "str", "path": "witness"}],
    "g44_hyp_pred_ref": [{"kind": "str", "path": "hyp"}],
    "g44_hyp_pred_expand": [{"kind": "str", "path": "hyp"}, {"kind": "str", "path": "pred"}],
    "g44_sqrt_iso": [{"kind": "str", "path": "rel"}, {"kind": "str", "path": "side"}],
    "g44_sqrt_sq": [{"kind": "str", "path": "rel"}, {"kind": "str", "path": "side"}],
    "g44_sqrt_close": [{"kind": "str", "path": "rel"}, {"kind": "str", "path": "side"}],
    "g44_pow_atom": [{"kind": "str", "path": "exp"}],
    "g44_pow_cap": [{"kind": "int", "path": "cap"}],
    "g44_fact_atom": [{"kind": "str", "path": "arg"}],
    "g44_lib_atom": [{"kind": "str", "path": "lib"}],
    "g44_relchain": [{"kind": "int", "path": "nrel"}],
}


def md5f(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def load_plan(path=None):
    d = json.load(open(path or PLAN, encoding="utf-8"))
    oi = {str(k): int(v) for k, v in (d.get("op_index") or {}).items()}
    idxs = sorted(oi.values())
    order = [n for n, _i in sorted(oi.items(), key=lambda kv: kv[1])]
    return d, order, oi, idxs


def build_vocab7(out_vocab=V7_VOCAB, out_allow=V7_ALLOW, extra_ops=()):
    """V6 -> V7. extra_ops = appended new operator names (indices continue after 219, in the
    given order)."""
    os.makedirs(CACHE, exist_ok=True)
    _d, order, oi, idxs = load_plan()
    base_order = list(order)
    base_idx = list(idxs)
    # Fixed-order assertion: the pre-registered segment must be exactly 207..207+n-1, contiguous,
    # without duplicates and with unique names.
    assert base_idx == list(range(V7_BASE, V7_BASE + len(base_idx))), \
        "pre-registered index segment is not contiguous: %s" % base_idx
    assert len(set(base_order)) == len(base_order), "duplicate pre-registered operator name"
    append_ops = [o for o in extra_ops if o not in oi]
    nxt = (base_idx[-1] + 1) if base_idx else V7_BASE
    for o in append_ops:
        oi[o] = nxt
        nxt += 1
    ops_order = [n for n, _i in sorted(oi.items(), key=lambda kv: kv[1])]
    assert len(set(oi.values())) == len(oi), "duplicate index"
    assert sorted(oi.values()) == list(range(V7_BASE, V7_BASE + len(oi))), \
        "V7 indices are not contiguous 207..: %s" % sorted(oi.values())
    head_v7 = V7_BASE + len(oi)

    v6 = json.load(open(V6_VOCAB, encoding="utf-8"))
    a6 = np.fromfile(V6_ALLOW, dtype=np.uint8).reshape(HEAD_V6, MS, NVALS)
    old_sum = int(a6.sum())
    assert old_sum == OLD_SUM, "V6 old-segment sum != %d: %d" % (OLD_SUM, old_sum)
    add = np.ones((len(oi), MS, NVALS), dtype=np.uint8)
    a7 = np.concatenate([a6, add], axis=0)
    a7.tofile(out_allow)

    v7 = dict(v6)
    v7["head_v7"] = head_v7
    v7["head_v7_base"] = V7_BASE
    v7["ops_g44"] = {k: int(v) for k, v in sorted(oi.items(), key=lambda kv: kv[1])}
    # P4 parser entry point: the preferred key of gen44_p4_parsers.load_vocab7 (first choice of the
    # tolerant reader)
    v7["op_index"] = {k: int(v) for k, v in sorted(oi.items(), key=lambda kv: kv[1])}
    v7["ops_g44_appended"] = list(append_ops)
    v7["slot_schema_v7_new"] = {k: v for k, v in SLOT_V7.items()}
    v7["allow_shape_v7"] = [head_v7, MS, NVALS]
    v7["allow_bin_v7"] = os.path.basename(out_allow)
    v7["allow_sum_by_row_v7"] = {"base_0_206": old_sum, "total": int(a7.sum()),
                                 "n_rows": int(a7.shape[0])}
    v7["vocab6_md5"] = md5f(V6_VOCAB)
    v7["allow_v6_md5"] = md5f(V6_ALLOW)
    # NOTE: Chinese report payload value kept verbatim (data, not a log line).
    v7["upgrade_note_v7"] = (
        "旧 207 行/列逐字节保留 (a7[:207] == a6 断言, sum=%d); 新行 207..%d 全 1 (%dx%d); "
        "算子索引固定序见 ops_g44" % (old_sum, head_v7 - 1, MS, NVALS))
    json.dump(v7, open(out_vocab, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- post-write re-verification (read back from disk and assert again) ----
    a7b = np.fromfile(out_allow, dtype=np.uint8).reshape(head_v7, MS, NVALS)
    old_byte_ok = bool(a7b[:HEAD_V6].tobytes() == a6.tobytes())
    new_all_one = bool(int(a7b[HEAD_V6:].sum()) == len(oi) * MS * NVALS)
    assert old_byte_ok, "the old 207 rows were modified (byte-for-byte)"
    assert new_all_one, "the new rows are not all ones"
    v7b = json.load(open(out_vocab, encoding="utf-8"))
    rep = {
        "schema": "gen44-vocab7/1", "created": now_iso(),
        "v6": {"vocab_md5": md5f(V6_VOCAB), "allow_md5": md5f(V6_ALLOW),
               "head": HEAD_V6, "old_sum": old_sum},
        "v7": {"vocab_path": out_vocab, "vocab_md5": md5f(out_vocab),
               "allow_path": out_allow, "allow_md5": md5f(out_allow),
               "head_v7": head_v7, "total_sum": int(a7b.sum()),
               "bytes": int(os.path.getsize(out_allow))},
        "old_rows_byte_identical": old_byte_ok,
        "old_sum_match": bool(old_sum == OLD_SUM),
        "new_rows_all_one": new_all_one,
        "n_new_rows": len(oi),
        "op_index": {k: int(v) for k, v in sorted(oi.items(), key=lambda kv: kv[1])},
        "op_order_index_sorted": ops_order,
        "index_contiguous": True,
        "index_unique": True,
        "name_unique": len(set(ops_order)) == len(ops_order),
        "appended_ops": append_ops,
        "slot_schema_v7_new": SLOT_V7,
        "plan_md5": md5f(PLAN),
        "head_v7_json_match": bool(int(v7b.get("head_v7", -1)) == head_v7),
        "gates_all_pass": bool(old_byte_ok and new_all_one and old_sum == OLD_SUM
                               and int(v7b.get("head_v7", -1)) == head_v7),
    }
    return rep


def load_ctx7(allow_bin=None, vocab_path=None):
    """On top of the V6 ctx (gen42_cachebuild.load_ctx6), append the V7 operator names/slots."""
    import gen42_cachebuild as CB42
    ctx = CB42.load_ctx6()
    v7 = json.load(open(vocab_path or V7_VOCAB, encoding="utf-8"))
    oi = {str(k): int(v) for k, v in (v7.get("ops_g44") or {}).items()}
    if not oi:                                          # degenerate case: read the pre-registration plan
        _d, order, oi, _i = load_plan()
    head_v7 = int(v7.get("head_v7") or (V7_BASE + len(oi)))
    names_head = list(ctx["names_head"]) + [None] * (head_v7 - len(ctx["names_head"]))
    opmap = dict(ctx["opmap"])
    for n, i in oi.items():
        names_head[int(i)] = n
        opmap[n] = int(i)
    assert len(names_head) == head_v7, (len(names_head), head_v7)
    miss = [i for i in range(161, head_v7) if not names_head[i]]
    assert not miss, "names_head has holes: %s" % miss
    toks_ext = ctx["toks_ext"]
    idx_ext = ctx["idx_ext"]
    dig_ids = list(range(toks_ext.index(("digest", "d00")), len(toks_ext)))
    unk = idx_ext[("any", "__UNK__")]
    per = dict(ctx["per"])
    for op, rows in SLOT_V7.items():
        if op not in oi:
            continue
        if not rows:                                    # slotless operator: register an explicit empty row (parent semantics: no slot generated)
            per[op] = []
            continue
        out_rows = []
        for r in rows:
            allow = sorted({unk} | set(dig_ids))
            out_rows.append((r["path"], {"kind": r["kind"]}, list(allow)))
        per[op] = out_rows
    vocab = dict(ctx["vocab"])
    vocab["head"] = head_v7
    ctx.update({"vocab": vocab, "names_head": names_head, "opmap": opmap, "per": per,
                "head_v7": head_v7, "ops_g44": oi, "vocab7_md5": md5f(vocab_path or V7_VOCAB),
                "allow_v7_md5": md5f(allow_bin or V7_ALLOW)})
    assert int(ctx["vocab"]["head"]) == head_v7
    # Reverse check: every V7 operator resolves to an index (uniqueness was asserted at build time).
    for n, i in oi.items():
        assert opmap[n] == i
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-out", default=V7_VOCAB)
    ap.add_argument("--allow-out", default=V7_ALLOW)
    ap.add_argument("--append-ops", default="",
                    help="comma-separated new operator names (indices continue after 219)")
    ap.add_argument("--gate-out", default=os.path.join(OUT, "GEN44_VOCAB7_GATE.json"))
    ap.add_argument("--verify-ctx", type=int, default=1)
    a = ap.parse_args()
    extra = [s for s in (a.append_ops or "").split(",") if s]
    rep = build_vocab7(a.vocab_out, a.allow_out, extra_ops=extra)
    if a.verify_ctx:
        ctx = load_ctx7()
        rep["ctx_check"] = {"head_v7": ctx["head_v7"], "n_ops": len(ctx["ops_g44"]),
                            "names_head_ok": bool(all(ctx["names_head"][i]
                                                      for i in range(161, ctx["head_v7"]))),
                            "per_has_new_ops": sorted(k for k in ctx["per"]
                                                      if k.startswith("g44_"))}
    json.dump(rep, open(a.gate_out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: rep[k] for k in ("v7", "old_rows_byte_identical", "old_sum_match",
                                          "new_rows_all_one", "n_new_rows", "index_contiguous",
                                          "index_unique", "name_unique", "gates_all_pass")},
                     ensure_ascii=False))
    print("gate -> %s md5 %s" % (a.gate_out, md5f(a.gate_out)))
    assert rep["gates_all_pass"], "V7 gate FAIL"


if __name__ == "__main__":
    main()
