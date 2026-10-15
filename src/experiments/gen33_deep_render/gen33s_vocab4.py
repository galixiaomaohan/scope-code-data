# -*- coding: utf-8 -*-
# gen33s_vocab4.py -- gen33 vocabulary V4 upgrade (V3 -> V4): append the nine gen32 deep operators (head 185 -> 194)
#
# Discipline (follows gen26s_vocab3's upgrade discipline item by item): **upgrade; never edit v2/v3 in place**
#   -- v3 artifacts (CCP_VOCAB_V3.json / ccp_allow_v3.bin) untouched; this module produces new v4 artifacts
#   (cache/CCP_VOCAB_V4.json · cache/ccp_allow_v4.bin).
#   * operator space: head 185 -> 194 (append 185..193); **old 0..184 untouched byte for byte** (asserted)
#   * value space: n_vals_ext 352 **unchanged** (no new values this batch: the nine new operators' slot kinds all fall inside existing enum/digest buckets)
#   * init_map: the existing 24 rows copied as-is; the nine new rows copied from parent rows (neighbouring operators)
#   * allow: (185,24,352) -> (194,24,352); old rows asserted equal via np.array_equal
#
# Read-only: gen26/out/CCP_VOCAB_V3.json · gen26/cache/ccp_allow_v3.bin · gen13/ccp_vocab.py ·
#       gen13/ccp_cachefull.py (load_ctx, reusing tr/idx_ext/tok)
# Writes only: scratch/ccp/gen33/cache/**
import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import sys

import g33_common as C

sys.path.insert(0, C.G13)
import numpy as np                                                     # noqa: E402

MS = 24
N_VALS_V3 = 352
HEAD_V3 = 185
END = 154
NBUCKET = 64
NUM_KINDS = ("int", "nat", "bool", "Fraction", "Int", "Nat")

#: new-operator registration order (head 185..193; semantic order of DEEP_REPORT §7)
NEW_ORDER = ["gen32_nest_solve", "gen32_nest_fold", "gen32_nest_delta", "gen32_nest_close",
             "gen32_sos_block", "gen32_sos_cross",
             "gen32_sos_tail_zero", "gen32_sos_tail_square", "gen32_sos_tail_other"]
#: init parent rows (neighbouring operators; name + idx both listed, matching the DEEP_REPORT §7 relation column)
PARENT = {
    "gen32_nest_solve": ("bh_ind_sum_d1", 168),
    "gen32_nest_fold": ("bh_ind_sum_d3", 170),
    "gen32_nest_delta": ("bh_ind_geo", 172),
    "gen32_nest_close": ("bh_ind_close_sq4", 171),
    "gen32_sos_block": ("bh_ineq_block", 174),
    "gen32_sos_cross": ("bh_ineq_cross", 175),
    "gen32_sos_tail_zero": ("bh_ineq_tail_zero", 176),
    "gen32_sos_tail_square": ("bh_ineq_tail_square", 177),
    "gen32_sos_tail_other": ("bh_ineq_tail_other", 178),
}
#: slot table (design decision D-2; mirrors the bh same-family convention: load-bearing fields only)
SLOTS = {
    "gen32_nest_solve": [("level", "str"), ("nv", "int"), ("v0", "int"), ("deg", "int"),
                         ("S", "str")],
    "gen32_nest_fold": [("level", "str"), ("nv", "int"), ("scale", "int"), ("lower", "str")],
    "gen32_nest_delta": [("level", "str"), ("nv", "int"), ("upper", "str"), ("inc", "str")],
    "gen32_nest_close": [("level", "str"), ("nv", "int"), ("P", "str"), ("T", "str"),
                         ("cls", "str"), ("sq", "list")],
    "gen32_sos_block": [("ell", "str"), ("mult", "int"), ("vars", "list"), ("R_in", "str"),
                        ("hint_idx", "int")],
    "gen32_sos_cross": [("ell", "str"), ("mult", "int"), ("vars", "list"), ("R_in", "str"),
                        ("hint_idx", "int")],
    "gen32_sos_tail_zero": [("vars", "list"), ("rho", "int"), ("hints", "list"),
                            ("goal_dom", "str"), ("hints_poly", "list")],
    "gen32_sos_tail_square": [("vars", "list"), ("rho", "int"), ("hints", "list"),
                              ("goal_dom", "str"), ("hints_poly", "list"), ("sq", "int")],
    "gen32_sos_tail_other": [("vars", "list"), ("rho", "int"), ("hints", "list"),
                             ("goal_dom", "str"), ("hints_poly", "list")],
}


def md5f(p):
    return C.md5f(p)


def canon(v):
    return json.dumps(v, sort_keys=True, ensure_ascii=False, default=str)


def digest_of(v):
    """Text/structural slot digest (registered: 64 buckets, md5(canon-json)[:8] mod 64). The device side must use the same formula."""
    return int(hashlib.md5(canon(v).encode("utf-8")).hexdigest()[:8], 16) % NBUCKET


def _load(name, path):
    ld = importlib.machinery.SourceFileLoader(name, path)
    m = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, ld))
    ld.exec_module(m)
    return m


def _toks_ext_v3(ctx0, voc):
    """Extend ccp_cachefull.load_ctx's 238-value table to the v3/v4 352-value table (convention = gen26s_vocab3.load_ctx3).

    The whole v2->v3 value-space extension is registered in v3's extra_values (idx 151..351); this module reuses it
    bit for bit and **creates no new values** (zero additions this batch). Existing artifacts untouched: pad the table, then overwrite.
    """
    te = list(ctx0["toks_ext"])
    while len(te) < N_VALS_V3:
        te.append(("any", "__UNK__"))
    for ev in voc["extra_values"]:
        i = int(ev["idx"])
        while len(te) <= i:
            te.append(("any", "__UNK__"))
        te[i] = (ev["kind"], ev.get("typed_value", ev["value"]))
    assert len(te) == N_VALS_V3, (len(te), N_VALS_V3)
    assert te[ctx0["UNK"]] == ("any", "__UNK__"), te[ctx0["UNK"]]
    return te


def build_v4(vocab_v3_path, allow_v3_path, head_out, allow_out, logp=None):
    v3 = json.load(open(vocab_v3_path, encoding="utf-8"))
    assert v3["schema"] == "ccp-vocab/3", v3["schema"]
    assert int(v3["head"]) == HEAD_V3 and int(v3["n_vals_ext"]) == N_VALS_V3
    assert int(v3["end_idx"]) == END
    rep = {"schema": "ccp-vocab/4",
           "law": {"用户令": "现在应该开始直接训练模型，并批卷",
                   "设计": "scratch/ccp/gen32/DEEP_REPORT.md §7 (集成设计注)",
                   "预注册": "scratch/ccp/gen33/V33_DESIGN.md §2"},
           "v3": {"vocab": vocab_v3_path, "md5": md5f(vocab_v3_path),
                  "allow": allow_v3_path, "allow_md5": md5f(allow_v3_path)},
           "policy": {"upgrade": "升版 (禁就地改 v2/v3); 既有 0..184 逐字节不动",
                      "value_space": "**不变** (n_vals_ext 352; 本批零新增值)",
                      "no_hot_start": "step_head/allow/value_head/lat_tab 四处变宽 ⇒ 体热启+头全新建",
                      "init": "新算子行权重由 init_map 父行复制 (load_init_ccp 语义)"}}

    opmap = {n: int(i) for n, i in v3["ops_ccp"].items()}
    for k, n in enumerate(NEW_ORDER):
        opmap[n] = HEAD_V3 + k
    head = HEAD_V3 + len(NEW_ORDER)
    assert head == 194 and len(set(opmap.values())) == len(opmap)
    legal = sorted(list(v3["ops_legal_idx"]) + [opmap[n] for n in NEW_ORDER])
    rep["ops"] = {"head": head, "n_new": len(NEW_ORDER),
                  "new": {n: opmap[n] for n in NEW_ORDER},
                  "legal_idx_sorted": legal, "n_legal": len(legal), "end_idx": END}
    assert len(legal) == len(set(legal)), "合法集有重"

    init_map = {k: dict(v) for k, v in v3["init_map"].items()}
    for n in NEW_ORDER:
        pn, pi = PARENT[n]
        assert v3["ops_bh"].get(pn) == pi, (pn, pi, v3["ops_bh"].get(pn))
        init_map[str(opmap[n])] = {"name": n, "parent": pn, "parent_idx": pi}
    missing = [k for k in range(head) if k >= 161 and str(k) not in init_map]
    assert not missing, ("init_map 缺口", missing)
    rep["init_map"] = {k: init_map[k] for k in sorted(init_map, key=int)}
    rep["init_map_new"] = {str(opmap[n]): init_map[str(opmap[n])] for n in NEW_ORDER}

    # ---- value space: follow v3 bit for bit (zero additions) ----
    ctx0 = _load("g33_v4_ccf", os.path.join(C.G13, "ccp_cachefull.py")).load_ctx()
    toks_ext = _toks_ext_v3(ctx0, v3)
    idx_ext = {(k, str(v)): i for i, (k, v) in enumerate(toks_ext)}
    dig_ids = list(range(toks_ext.index(("digest", "d00")), len(toks_ext)))
    assert len(dig_ids) == NBUCKET, len(dig_ids)
    for n in NEW_ORDER:                         # frozen classes are not in the old operator set, so no conflict
        assert n not in ctx0["per"], n
    rep["value_space"] = {"n_vals_ext": N_VALS_V3, "n_digest": NBUCKET,
                          "digest_idx_range": [dig_ids[0], dig_ids[-1]],
                          "new_values_added": 0,
                          "lossy_note": "文本槽摘要有损 (64 桶); 桶内碰撞允许且登记 (v3 同式)"}

    # ---- per-op slot table (nine new operators) ----
    per_new = {}
    for op in NEW_ORDER:
        rows = []
        for path, kind in SLOTS[op]:
            if kind in NUM_KINDS:
                allow = sorted({idx_ext[("any", "__UNK__")]} |
                               {i for i, (k2, _v2) in enumerate(toks_ext) if k2 in NUM_KINDS})
            else:
                allow = sorted({idx_ext[("any", "__UNK__")]} | set(dig_ids))
            rows.append({"path": path, "kind": kind, "allowed": allow})
        per_new[op] = rows
    rep["slot_schema_v4_new"] = {op: [{"path": r["path"], "kind": r["kind"],
                                       "n_allowed": len(r["allowed"])}
                                      for r in rows] for op, rows in per_new.items()}
    rep["n_slots_new_ops"] = {op: len(rows) for op, rows in per_new.items()}

    # ---- allow (194, 24, 352): existing rows carried over from v3 byte for byte ----
    a3 = np.fromfile(allow_v3_path, dtype=np.uint8).reshape(HEAD_V3, MS, N_VALS_V3)
    allow = np.zeros((head, MS, N_VALS_V3), dtype=np.uint8)
    allow[:HEAD_V3, :, :] = a3
    for op, rows in per_new.items():
        i = opmap[op]
        for si, r in enumerate(rows):
            for a in r["allowed"]:
                allow[i, si, a] = 1
    eq = bool(np.array_equal(allow[:HEAD_V3, :, :], a3))
    rep["allow"] = {"path": allow_out, "shape": list(allow.shape),
                    "rows_0_184_bitwise_equal_v3": eq,
                    "sum_new_rows": {op: int(allow[opmap[op]].sum()) for op in per_new},
                    "sum_all": int(allow.sum()), "sum_v3": int(a3.sum())}
    assert eq, "既有 allow 行被改动 (违升版纪律)"
    allow.tofile(allow_out)

    # ---- persist vocab v4 (consumed by the model / evaluation / build) ----
    out = dict(v3)
    out.update({"schema": "ccp-vocab/4", "head": head,
                "ops_gen32": {n: opmap[n] for n in NEW_ORDER},
                "n_ops_gen32": len(NEW_ORDER),
                "ops_legal_idx": legal,
                "n_ops_v3": 17, "n_ops_v3_max_idx": HEAD_V3 - 1,
                "n_vals_ext": N_VALS_V3,
                "init_map": init_map,
                "slot_schema_v4_new": rep["slot_schema_v4_new"],
                "allow_bin": allow_out, "allow_bin_md5": md5f(allow_out),
                "allow_shape": list(allow.shape),
                "upgrade_note": ("v4 = v3 + 9 gen32 深算子 (head 185→194); 既有 0..184 行/列"
                                 "逐位不动; 值空间不变 (零新增值)"),
                "v3_source": {"vocab_md5": md5f(vocab_v3_path),
                              "allow_md5": md5f(allow_v3_path)}})
    json.dump(out, open(head_out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    rep["vocab_v4"] = {"path": head_out, "md5": md5f(head_out), "head": head,
                       "n_vals_ext": N_VALS_V3}
    if logp:
        C.log("VOCAB_V4 head=%d n_vals=%d legs=%d allow_rows_ok=%s allow_sum=%d(v3 %d)"
              % (head, N_VALS_V3, len(legal), eq, rep["allow"]["sum_all"],
                 rep["allow"]["sum_v3"]), logp)
    return rep


# ---------------------------------------------------------------------------
# ctx4: for gen13 ccp_cachefull.build reuse (same key names, operator space extended to 194)
# ---------------------------------------------------------------------------

def _names_from_ops(voc):
    nm = {}
    for key in ("ops_ccp", "ops_bh", "ops_g24", "ops_gen32"):
        for k, v in (voc.get(key) or {}).items():
            nm[int(v)] = k
    return nm


def load_ctx4(vocab4_path, allow4_path=None, logp=None):
    """ctx = the v4 version of ccp_cachefull.load_ctx (same key names; mslt/vocab/opmap/per/names_head extended to 194)."""
    ccf = _load("g33_v4_ccf_ctx", os.path.join(C.G13, "ccp_cachefull.py"))
    ctx = ccf.load_ctx()
    voc = json.load(open(vocab4_path, encoding="utf-8"))
    assert voc["schema"] == "ccp-vocab/4" and int(voc["head"]) == 194, voc["schema"]
    toks_ext = _toks_ext_v3(ctx, voc)
    idx_ext = {(k, str(v)): i for i, (k, v) in enumerate(toks_ext)}
    names_head = list(ctx["names_head"])[:HEAD_V3]
    while len(names_head) < 194:
        names_head.append(None)
    for i, n in _names_from_ops(voc).items():
        if i < 194:
            names_head[i] = n
    assert len(names_head) == 194
    assert all(names_head[i] for i in range(161, 194)), \
        ("names_head 缺口", [i for i in range(161, 194) if not names_head[i]])
    opmap = dict(ctx["opmap"])
    for i, n in _names_from_ops(voc).items():
        opmap[n] = i
    per = dict(ctx["per"])
    dig_ids = list(range(toks_ext.index(("digest", "d00")), len(toks_ext)))
    for op, rows in voc["slot_schema_v4_new"].items():
        out_rows = []
        for r in rows:
            kind = r["kind"]
            if kind in NUM_KINDS:
                allow = sorted({idx_ext[("any", "__UNK__")]} |
                               {i for i, (k2, _v2) in enumerate(toks_ext) if k2 in NUM_KINDS})
            else:
                allow = sorted({idx_ext[("any", "__UNK__")]} | set(dig_ids))
            out_rows.append((r["path"], {"kind": kind}, list(allow)))
        per[op] = out_rows
    ctx.update({"vocab": voc, "toks_ext": toks_ext, "idx_ext": idx_ext,
                "names_head": names_head, "opmap": opmap, "per": per,
                "NVALUES": len(toks_ext), "digest_ids": dig_ids, "MS": MS,
                "END": voc["end_idx"]})
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-v3", default=C.VOCAB_V3)
    ap.add_argument("--allow-v3", default=C.ALLOW_V3)
    ap.add_argument("--vocab-out", default=C.G33 + "/cache/CCP_VOCAB_V4.json")
    ap.add_argument("--allow-out", default=C.G33 + "/cache/ccp_allow_v4.bin")
    ap.add_argument("--report", default=C.G33 + "/out/GEN33_VOCAB4.json")
    ap.add_argument("--log", default=C.G33 + "/logs/gen33.log")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.vocab_out), exist_ok=True)
    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    C.log("== gen33 vocabulary V4 upgrade (head 185→194; old 185 untouched byte for byte) ==", a.log)
    rep = build_v4(a.vocab_v3, a.allow_v3, a.vocab_out, a.allow_out, logp=a.log)
    # read-back self-check
    back = json.load(open(a.vocab_out, encoding="utf-8"))
    v3raw = json.load(open(a.vocab_v3, encoding="utf-8"))
    a4 = np.fromfile(a.allow_out, dtype=np.uint8).reshape(194, MS, N_VALS_V3)
    a3 = np.fromfile(a.allow_v3, dtype=np.uint8).reshape(HEAD_V3, MS, N_VALS_V3)
    chk = {"head": int(back["head"]) == 194,
           "n_vals": int(back["n_vals_ext"]) == N_VALS_V3,
           "end": int(back["end_idx"]) == END,
           "old_allow_bitwise": bool(np.array_equal(a4[:HEAD_V3], a3)),
           "init_map_194": all(str(k) in back["init_map"] for k in range(161, 194)),
           "legal_has_9": all(int(back["ops_gen32"][n]) in back["ops_legal_idx"]
                              for n in NEW_ORDER),
           "old_ops_unchanged": all(int(back[key][k]) == int(v)
                                    for key in ("ops_ccp", "ops_bh", "ops_g24")
                                    for k, v in v3raw[key].items()),
           "old_extra_values_unchanged": (
               json.dumps(back["extra_values"], sort_keys=True)
               == json.dumps(v3raw["extra_values"], sort_keys=True)),
           "old_slot_schema_unchanged": (
               json.dumps(back["slot_schema_v3_new"], sort_keys=True)
               == json.dumps(v3raw["slot_schema_v3_new"], sort_keys=True))}
    rep["readback"] = chk
    rep["readback_ok"] = bool(all(chk.values()))
    ctx4 = load_ctx4(a.vocab_out, logp=a.log)
    rep["ctx4"] = {"names_head_len": len(ctx4["names_head"]),
                   "per_has_new": all(n in ctx4["per"] for n in NEW_ORDER),
                   "NVALUES": ctx4["NVALUES"],
                   "n_legal_names": sum(1 for i in range(161, 194)
                                        if ctx4["names_head"][i])}
    assert rep["readback_ok"], chk
    rep["md5"] = C.save_json(a.report, rep)
    C.log("VOCAB_V4 DONE md5=%s readback=%s" % (rep["md5"], rep["readback_ok"]), a.log)
    print("GEN33_VOCAB_RC=0")
    return rep


if __name__ == "__main__":
    main()
