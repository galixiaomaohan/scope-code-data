# -*- coding: utf-8 -*-
# ccp_vocab.py -- CCP vocabulary extension registry (gen13 task A step 1; lands the "one extension" left over from constitution change 58)
#
# Convention (registered entries, all mechanically recomputable):
#   * operator space = V32 160 names (indices bit-for-bit unchanged) + CCP 7 (appended at 160..166) + END=167 => head width 168.
#   * value vocabulary = V32 151 values (first 151 entries bit-for-bit unchanged) + **extra values exported by scanning the train split of this course batch**;
#     an extra value = a (kind, value) that occurs in a ccp_* operator slot but not in the V32 value vocabulary -- the derivation rule is fixed
#     (operators in registry order -> slots in schema order -> values sorted), not hand-written, no randomness.
#   * allow table = (168, 24, NVALS): rows 0..159 are a **bit-for-bit reuse** of v32lib.build_allow() (v31 spec assertions included);
#     rows 160..166 are exported mechanically from the ccp_* slot schema + the extended value domain; row 167 (END) is all zero = no schema.
#   * start-point transfer table (init_map): new rows initialize from the semantically nearest row (registered; handed to the trainer for per-row copy_).
#
# Reads (read-only): cppcuda/tools, probes/, skel32/, superoperator/ and scratch/ccp/gen/*.jsonl;
# writes: scratch/ccp/gen13/** only.
import argparse
import collections
import importlib.machinery
import importlib.util
import json
import os
import sys
from fractions import Fraction

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = os.path.join(RES, "scratch", "ccp")
G13 = os.path.join(CCP, "gen13")
os.makedirs(G13, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # local ccp_* modules in the package layout
sys.path.insert(0, os.path.join(RES, "cppcuda", "tools"))
sys.path.insert(0, os.path.join(RES, "skel32"))
sys.path.insert(0, os.environ.get("SCOPE_SUPEROPERATOR",
                                  os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "device")))

CCP_OPS = ["ccp_ind_recurrence", "ccp_ring_identity", "ccp_ineq_sos", "ccp_ineq_close",
           "ccp_nt_dvd_witness", "ccp_nt_dvd_combine", "ccp_nt_dvd_prod_combine"]
# Slot schema, **gen13 measured correction** (registered: the gen12 ccp_cache.SLOT_SCHEMA was a smoke-level
#   declaration and does **not** match the course rows' sf_args -- measured sf_args keys per operator:
#     ccp_ind_recurrence{form,v0,step,family,ty,sample_n} · ccp_ring_identity{nvars,atom,lhs,rhs}
#     ccp_ineq_sos{vars,dom} · ccp_ineq_close{vars,squares,residual}
#     ccp_nt_dvd_witness{m,hyp,var,text} · ccp_nt_dvd_combine{m_h,M,terms,const,hypvars}
#   => this table registers only the **supervisable scalar slots**; structural items (vars/dom/squares/terms/hypvars/lhs/rhs/text)
#   go through derived scalars: nvars=len(vars), dom_c0/dom_c2 = the constant/quadratic coefficients of dom.
CCP_SLOT_SCHEMA = {
    "ccp_ind_recurrence": [("form", "str"), ("v0", "int"), ("step", "int"),
                           ("sample_n", "int"), ("family", "str")],
    "ccp_ring_identity": [("nvars", "int"), ("atom", "str")],
    "ccp_ineq_sos": [("nvars", "int"), ("dom_c0", "int"), ("dom_c2", "int")],
    "ccp_ineq_close": [("nvars", "int"), ("residual", "num")],
    "ccp_nt_dvd_witness": [("m", "int")],
    "ccp_nt_dvd_combine": [("m_h", "int"), ("M", "int"), ("const", "int")],
    "ccp_nt_dvd_prod_combine": [("m_h", "int")],
}
DERIVED_SLOTS = {"nvars", "dom_c0", "dom_c2"}
SCHEMA_GEN12_DELTA = {
    "note": "gen12 ccp_cache.SLOT_SCHEMA → gen13 修正 (逐项): "
            "ccp_ind_recurrence 增 step/family, 删 c1/c0 (在 value 不在 args); "
            "ccp_ineq_sos 删 residual (该键在 ccp_ineq_close), 增 dom_c0/dom_c2 (派生); "
            "ccp_ineq_close 增 nvars (派生)",
    "gen12": {"ccp_ind_recurrence": ["form", "v0", "sample_n", "c1", "c0"],
              "ccp_ring_identity": ["nvars", "atom"],
              "ccp_ineq_sos": ["nvars", "residual"],
              "ccp_ineq_close": ["residual"],
              "ccp_nt_dvd_witness": ["m"],
              "ccp_nt_dvd_combine": ["m_h", "M", "const"],
              "ccp_nt_dvd_prod_combine": ["m_h"]},
}


def extract_slot(path, args):
    """Slot scalar extraction (registered convention; structural items go through derived scalars)."""
    if args is None or not isinstance(args, dict):
        return None
    if path == "nvars":
        vs = args.get("vars")
        if isinstance(vs, list):
            return len(vs)
        v = args.get("nvars")                     # ccp_ring_identity carries the nvars scalar directly
        return int(v) if isinstance(v, int) else None
    if path == "dom_c0":
        d = args.get("dom") or {}
        return int(d.get("0", 0)) if isinstance(d, dict) else None
    if path == "dom_c2":
        d = args.get("dom") or {}
        return int(d.get("2", 0)) if isinstance(d, dict) else None
    v = args.get(path)
    return None if isinstance(v, (list, dict)) else v
# new-row start-point transfer table (registered; semantically nearest row = skeleton family 155..160)
INIT_MAP = {
    "ccp_ind_recurrence": "skel_induction_scheme",
    "ccp_ring_identity": "skel_leaf_close",
    "ccp_ineq_sos": "skel_subgoal_focus",
    "ccp_ineq_close": "skel_leaf_close",
    "ccp_nt_dvd_witness": "skel_case_split",
    "ccp_nt_dvd_combine": "skel_qed_check",
    "ccp_nt_dvd_prod_combine": "skel_qed_check",
}
# ⚠ Index-space convention (measured, registered): the V32 action space has **161** cells (0..160): 0..153 legacy, 154 =
#   v31_end_marker (V32's END label), 155..160 skeletons. CCP therefore appends at **161..167**, head width 168;
#   the END label keeps the V32 semantics (= 154, existing id semantics unchanged).
CCP_OP_BASE = 161
CCP_END = 154
CCP_END_NAME = "v31_end_marker"


def _load(name, path):
    ld = importlib.machinery.SourceFileLoader(name, path)
    sp = importlib.util.spec_from_loader(name, ld)
    m = importlib.util.module_from_spec(sp)
    ld.exec_module(m)
    return m


def load_base(verbose=True):
    """-> (tr, L32, O32, NAMES32, toks_base, per_base)"""
    tr = _load("ccpv_tr31", os.path.join(RES, "probes", "sla_train_v31.py.txt"))
    L32 = _load("ccpv_lib32", os.path.join(RES, "skel32", "v32lib.py"))
    from svokernel_v32 import V32REGISTRY, v32_op_index_map       # noqa: E402
    O32 = v32_op_index_map()
    NAMES32 = list(V32REGISTRY.names())
    toks, per, mx, ntr = tr.build_tables(V32REGISTRY)
    assert len(NAMES32) == 160 and len(O32) == 161, (len(NAMES32), len(O32))
    # ⚠ measured (registered): V32REGISTRY.names() **insertion order** and v32_op_index_map's **index order** disagree on the
    #   6 skeleton rows (names inserts the 6 skeleton operators at 154..159, while the index space is v31_end_marker=154 +
    #   skeletons 155..160). The index space (O32) is authoritative for operator ids (the stepping head has 161 rows = that
    #   space) => this vocabulary is built on O32.
    inv = {v: k for k, v in O32.items()}
    NAMES_IDX = [inv[i] for i in range(len(O32))]
    assert set(NAMES_IDX) == (set(NAMES32) | {"v31_end_marker"}) and len(NAMES_IDX) == 161, \
        (len(NAMES_IDX), len(NAMES32))
    names_order_mismatch = [(i, n, O32[n]) for i, n in enumerate(NAMES32) if O32[n] != i]
    if verbose:
        print("[vocab] names() order == index order? %s (%d mismatching rows: %s)"
              % (not names_order_mismatch, len(names_order_mismatch),
                 [x[1] for x in names_order_mismatch]))
    return tr, L32, O32, NAMES_IDX, toks, per


def _kind_of(v):
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    return "str"


def scan_extra_values(rows_by_cls, toks):
    """Scan the train split -> the list of (kind, value) to append (deterministic order).

    kind convention aligned with the slot schema: str -> ('str', value); int -> ('int', value); num -> ('Fraction', value)
    (Fraction / numeric strings verbatim; paired with the Fraction->int fallback chain of prep_cache_v31.slots_of).
    """
    have = {(k, str(v)) for (k, v) in toks}
    extra, seen = [], set()
    for cls in ("ind", "ineq", "nt"):
        for r in rows_by_cls[cls]:
            if r.get("ccp_held"):
                continue                       # the value domain is built from the train split only (held-out rows play no part)
            sfa = r.get("sf_args") or []
            prog = r.get("program") or []
            for op, args in zip(prog, sfa):
                if op not in CCP_SLOT_SCHEMA or not isinstance(args, dict):
                    continue
                for path, kind0 in CCP_SLOT_SCHEMA[op]:
                    v = extract_slot(path, args)
                    if v is None:
                        continue
                    if kind0 == "str":
                        k, val, typed = "str", str(v), str(v)
                    elif kind0 == "int":
                        k, val, typed = "int", str(int(v)), int(v)
                    else:                          # num -> Fraction vocabulary slot (string form, same as the base vocabulary)
                        s = (str(v) if not isinstance(v, float)
                             else str(Fraction(v).limit_denominator(1000)))
                        k, val, typed = "Fraction", s, s
                    key = (k, val)
                    if key in have or key in seen:
                        continue
                    seen.add(key)
                    extra.append((op, path, k, typed, val))
    # deterministic sort: operator registry order -> slot order -> value
    ordop = {op: i for i, op in enumerate(CCP_OPS)}
    def _key(it):
        op, path, k, _typed, sval = it
        slot = [p for p, _ in CCP_SLOT_SCHEMA[op]].index(path)
        return (ordop[op], slot, k, sval if k == "str" else "%040s" % sval)
    extra.sort(key=_key)
    return extra


def build_vocab(verbose=True):
    tr, L32, O32, NAMES32, toks, per = load_base(verbose=verbose)
    rows_by_cls = {}
    for cls in ("ind", "ineq", "nt"):
        p = os.path.join(CCP, "gen", "ccp_%s_rows.jsonl" % cls)
        rows = [json.loads(ln) for ln in open(p, encoding="utf-8") if ln.strip()]
        for r in rows:
            r["ccp_held"] = (int(__import__("hashlib").md5(
                r["ccp_name"].encode("utf-8")).hexdigest()[:8], 16) % 10 == 7)
        rows_by_cls[cls] = rows
    extra = scan_extra_values(rows_by_cls, toks)
    toks_ext = list(toks) + [(k, typed) for (_op, _p, k, typed, _sv) in extra]
    idx_ext = {(k, str(v)): i for i, (k, v) in enumerate(toks_ext)}
    assert len(idx_ext) == len(toks_ext), "扩展值词表出现重复键"
    # --- operator table (index order 0..160 = V32 space; CCP appended at 161..167; END = 154 semantics unchanged) ---
    opmap = {n: i for i, n in enumerate(NAMES32)}
    assert len(opmap) == 161 and opmap[CCP_END_NAME] == CCP_END
    for j, n in enumerate(CCP_OPS):
        opmap[n] = CCP_OP_BASE + j
    HEAD = CCP_OP_BASE + len(CCP_OPS)          # 168
    END_IDX = CCP_END
    # --- allow (168, 24, NVALS) ---
    import numpy as np
    allow = np.zeros((HEAD, 24, len(toks_ext)), dtype=np.uint8)
    base_allow = L32.build_allow()
    ba = base_allow.numpy() if hasattr(base_allow, "numpy") else base_allow
    assert ba.shape[0] == 161 and ba.shape[1] == 24
    allow[:161, :, :len(toks)] = ba          # base value ids 0..150 bit-for-bit unchanged => prefix columns copied over directly
    unk = idx_ext[("any", "__UNK__")]
    per_slot_extra = collections.defaultdict(set)
    for (op, path, k, _typed, sv) in extra:
        per_slot_extra[(op, path)].add(idx_ext[(k, sv)])
    ccp_slots = {}
    for j, op in enumerate(CCP_OPS):
        ri = CCP_OP_BASE + j
        slots = []
        for si, (path, kind) in enumerate(CCP_SLOT_SCHEMA[op]):
            if kind == "str":
                allowed = {unk} | set(per_slot_extra[(op, path)])
                sub = {"kind": "str", "enum": sorted({sv for (_o, p2, k2, _t, sv) in extra
                                                      if _o == op and p2 == path
                                                      and k2 == "str"})}
            elif kind == "int":
                sub = {"kind": "int"}
                allowed = set(tr.slot_allowed(sub, toks_ext, idx_ext))
            else:                                  # num -> Fraction semantics (full int+Fraction grid)
                sub = {"kind": "Fraction"}
                allowed = set(tr.slot_allowed(sub, toks_ext, idx_ext))
            allowed = sorted(allowed)
            for a in allowed:
                allow[ri, si, a] = 1
            slots.append({"path": path, "kind": kind, "n_allowed": len(allowed),
                          "allowed": allowed if len(allowed) <= 64 else None})
        ccp_slots[op] = slots
    # the END row (=154) is already inside base allow (the v31_end_marker row stays zero = no schema, same as V32)
    # --- start-point transfer table ---
    init_map = {}
    for nm, parent in INIT_MAP.items():
        idx = opmap[nm]
        init_map[str(idx)] = {"name": nm, "parent": parent, "parent_idx": O32[parent]}
    rep = {"schema": "ccp-vocab/2", "head": HEAD, "n_ops_base": 161, "n_ops_ccp": len(CCP_OPS),
           "ccp_op_base": CCP_OP_BASE, "end_idx": END_IDX, "end_name": CCP_END_NAME,
           "index_space_note": "V32 动作空间 = 161 格 (0..160, 154 = END); CCP 追加 161..167 → 头 168",
           "n_vals_base": len(toks), "n_vals_ext": len(toks_ext),
           "ops_ccp": {n: opmap[n] for n in CCP_OPS},
           "op_index_map_ccp": {n: opmap[n] for n in CCP_OPS},
           "n_extra_values": len(extra),
           "extra_values": [{"op": o, "slot": p, "kind": k, "value": sv, "idx": idx_ext[(k, sv)],
                             "typed_value": (typed if k != "Fraction" else sv)}
                            for (o, p, k, typed, sv) in extra],
           "extra_value_source": "scratch/ccp/gen/ccp_{ind,ineq,nt}_rows.jsonl 的 train 段 "
                                 "(ccp_held=False) 扫描; 派生规则 = 算子登记序 → 槽序 → 值排序",
           "slot_schema_ccp": ccp_slots,
           "slot_schema_gen12_delta": SCHEMA_GEN12_DELTA,
           "derived_slots": sorted(DERIVED_SLOTS),
           "init_map": init_map,
           "base_names_md5": __import__("hashlib").md5(
               "\n".join(NAMES32).encode("utf-8")).hexdigest(),
           "base_names_source": "v32_op_index_map 的索引序 (0..160); V32REGISTRY.names() 仅 "
                                "提供集合 (插入序在 6 骨架行上与索引序不一致, 已登记)",
           "base_values_md5": __import__("hashlib").md5(
               json.dumps([[k, str(v)] for k, v in toks], ensure_ascii=False).encode(
                   "utf-8")).hexdigest(),
           "allow_note": "行 0..159 逐位 = v32lib.build_allow(); 行 160..166 机械导出; 行 167 全零",
           "value_kinds": dict(collections.Counter(k for k, _ in toks_ext))}
    return rep, toks_ext, allow, rows_by_cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(G13, "CCP_VOCAB.json"))
    ap.add_argument("--allow-out", default=os.path.join(G13, "ccp_allow_ext.bin"))
    a = ap.parse_args()
    rep, toks_ext, allow, rows_by_cls = build_vocab()
    import numpy as np
    allow.astype(np.uint8).tofile(a.allow_out)
    rep["allow_bin"] = a.allow_out
    rep["allow_bin_md5"] = _md5(a.allow_out)
    rep["allow_shape"] = list(allow.shape)
    rep["allow_sum_by_row"] = {"%d" % i: int(allow[i].sum()) for i in range(161, 168)}
    rep["allow_sum_base_rows"] = int(allow[:161].sum())
    rep["row_counts"] = {c: len(rows_by_cls[c]) for c in rows_by_cls}
    rep["held_counts"] = {c: sum(1 for r in rows_by_cls[c] if r["ccp_held"]) for c in rows_by_cls}
    md5 = _save_json(a.out, rep)
    print("[vocab] head=%d n_vals=%d (base 151 + %d appended) | ops 160+7+END=%d"
          % (rep["head"], rep["n_vals_ext"], rep["n_extra_values"], rep["end_idx"]))
    print("[vocab] allow %s sum(base)=%d sum(new rows)=%s -> %s md5=%s"
          % (rep["allow_shape"], rep["allow_sum_base_rows"],
             sum(rep["allow_sum_by_row"].values()), a.allow_out, rep["allow_bin_md5"]))
    print("[vocab] extra-value samples: %s"
          % json.dumps(rep["extra_values"][:12], ensure_ascii=False))
    print("[vocab] report -> %s md5=%s" % (a.out, md5))
    print("CCP_VOCAB_RC=0")


def _md5(path):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
    return _md5(path)


if __name__ == "__main__":
    main()
