# -*- coding: utf-8 -*-
# gen44_barm_model.py -- supplementary run, arm B: the **model** walks the chain independently on
#     family-B certifiable rows (1,535)
#     -> family single rendering (leaf-level chain substitution from the model walk) -> frozen
#     judging. (G44-BARM, 2026-09-27)
#
# Measurement convention (registered item by item):
#   * row set = state=="certifiable" in out/GEN44_B{1..9}_CENSUS.jsonl (1,535 across the nine
#     families; the per-family file is authoritative row by row)
#   * sidecar row encoding = gen44_cachebuild_v7.Fam/tree_steps/leaf_steps/san_res/stmt_of
#     (import-only, zero modification) => isomorphic to the training cache (V7):
#     stmt / steps(op,res,args) / sf_args / frames
#   * chain walk = gen44_eval.phase_walk (md5 18505cf1... on the host before the
#     package comment translation, re-frozen to c80c7f8a... below; import-only,
#     zero modification), with an md5 assertion on the final-test ckpt; the only runtime override is
#     gen34_eval._payload_of pointing at the in-row frames **inside this file** (same form as
#     S8 VAM/VAD)
#   * rendering = family single rendering (bcore.render_tree / b345_ext.render_tree_ext /
#     fam_b789.render_tree, mode="exam"; import-only, zero modification) -- the model chain is
#     substituted **only at chain-sensitive spots (chain-leaf nodes)**:
#       the structural skeleton (intro/constructor/left|right) and branch choice still come from
#       the device tree (registered: not a full-chain substitution);
#       chain-leaf test = kind in that family's chain-leaf set; per row we assert that the
#       concatenated chain-leaf chains == the family ops_leaf;
#       slicing = consume model ops along the device skeleton (after the structural segment
#       n_top, consume in order; each leaf takes that leaf's device chain length, and a shortfall
#       stays empty => the renderer fail-closes with skip); tail ops beyond the end (over_run)
#       are dropped and registered.
#   * structural mismatch (walked[:n_top] != tops) => state=struct_dev, **not rendered**
#     (reported in its own column, not scored as zero); the lenient text is stored only for the
#     sensitivity analysis and is excluded from the main reading.
#   * judging = gen44_batchconfirm.py (unique shard tag, lesson of D-S8F-1); the judging module
#     drct_common is import-only, zero modification.
# Write side: only new files with new names under G44/** (this file overwrites no existing file).
# Zero /tmp, zero sudo.
import argparse
import collections
import copy
import hashlib
import json
import os
import sys
import time

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = os.path.join(RES, "scratch", "ccp")
G42 = os.path.join(CCP, "gen42")
G34 = os.path.join(CCP, "gen34")
# HERE is the gen44 tree on the experiment host (out/ and ckpt/ live there,
# not beside this file in the package layout).
HERE = os.path.join(CCP, "gen44")
OUT = os.path.join(HERE, "out")
for _p in (_PKG_DIR, G42, G34):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gen44_bcore as B                                                    # noqa: E402
import gen44_b345_ext as E345                                              # noqa: E402
import gen44_fam_b789 as M789                                              # noqa: E402
import gen44_cachebuild_v7 as CB                                           # noqa: E402

FAMS = ["b%d" % i for i in range(1, 10)]
SEG = "exam_barm_vb"
CKPT = os.path.join(HERE, "ckpt", "g44_v7_formal_s1_a2wt1_6000.pt")
CKPT_MD5 = "8d7c81f0d69322cce1e4eda2b4660fdf"
EVAL_MD5 = "c80c7f8ac66a5e261257d65c6c18ba4b"
RENDER_MD5 = "908535e4d60e483be4238cc3f842d245"
ITEMS = os.path.join(OUT, "GEN44_BARM_ITEMS.jsonl")
ITEMS_META = os.path.join(OUT, "GEN44_BARM_ITEMS_META.json")
WALK = os.path.join(OUT, "GEN44_BARM_WALK.json")
MAN = os.path.join(OUT, "GEN44_BARM_MANIFEST.jsonl")
MAN_META = os.path.join(OUT, "GEN44_BARM_MANIFEST_META.json")
LITALL = os.path.join(OUT, "GEN44_S8_LITALL_SUMMARY.json")

# chain-leaf node kinds per family (one-to-one with the family tree_leaf_ops implementations;
# verified by a per-row assertion)
CL_KINDS = {"b1": ("leaf", "sqrt_red"), "b2": ("leaf", "sqrt_red"), "b6": ("leaf", "sqrt_red"),
            "b3": ("leaf", "sqrt_red"), "b4": ("leaf", "sqrt_red"), "b5": ("leaf", "sqrt_red"),
            "b7": ("leaf",), "b8": ("leaf",), "b9": ("leaf",)}


def md5f(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def jlsave(rows, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return md5f(p)


def jldump(o, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(o, f, ensure_ascii=False, indent=1)
    return md5f(p)


def jlload(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def wilson(k, n, z=1.959964):
    if n <= 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


class P(object):
    __slots__ = ("name", "head", "concl", "split", "idx")

    def __init__(self, name, head, concl, split=None, idx=None):
        self.name, self.head, self.concl, self.split, self.idx = name, head, concl, split, idx


_LWB = None


def load_lwb():
    global _LWB
    if _LWB is None:
        d = {}
        for l in open(os.path.join(HERE, "data", "LWB_ALL.jsonl"), encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                d[int(r["idx"])] = r
        _LWB = d
    return _LWB


def fam_cert(fam):
    p = os.path.join(OUT, "GEN44_%s_CENSUS.jsonl" % fam.upper())
    return [r for r in jlload(p) if r.get("state") == "certifiable"]


def c6set():
    s = set()
    p = os.path.join(OUT, "GEN44_C6_EXCLUDED.jsonl")
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                k = d.get("idx", d.get("key"))
                if k is not None:
                    s.add(int(k))
    return s


def fam_mod(fam):
    return B if fam in ("b1", "b2", "b6") else (E345 if fam in ("b3", "b4", "b5") else M789)


def fam_parse(fam):
    if fam in ("b1", "b2", "b6"):
        return B.parse_stmt_ext
    if fam in ("b3", "b4", "b5"):
        return E345.parse_stmt_b345
    return lambda pr: M789.with_timeout(20.0, M789.parse_stmt_ext789, pr, fam)


def fam_render(fam, tr):
    if fam in ("b1", "b2", "b6"):
        return B.render_tree(tr, "exam")
    if fam in ("b3", "b4", "b5"):
        return E345.render_tree_ext(tr, "exam")
    return M789.render_tree(tr, "exam")


def fam_ops(fam, tr):
    m = fam_mod(fam)
    if fam in ("b1", "b2", "b6"):
        return list(B.tree_ops(tr)), list(B.tree_leaf_ops(tr))
    if fam in ("b3", "b4", "b5"):
        return list(E345.tree_ops_ext(tr)), list(E345.tree_leaf_ops_ext(tr))
    return list(M789.tree_ops(tr)), list(M789.tree_leaf_ops(tr))


def fam_frames(fam, tr):
    m = fam_mod(fam)
    if fam in ("b1", "b2", "b6"):
        return [list(x) if x is not None else None for x in list(B.tree_frames(tr))]
    if fam in ("b3", "b4", "b5"):
        return [list(x) if x is not None else None for x in list(E345.tree_frames_ext(tr))]
    return [list(x) if x is not None else None for x in list(M789.tree_frames(tr))]


def chain_leaves(node, kinds, out=None):
    """Collect chain-leaf nodes in preorder (same traversal order as the family
    tree_leaf_ops)."""
    out = [] if out is None else out
    if node.get("kind") in kinds:
        out.append(node)
    for c in node.get("children") or []:
        chain_leaves(c, kinds, out)
    return out


# ================================================================ [1] sidecar (walk items)
def stage_items(a):
    lwb = load_lwb()
    c6 = c6set()
    rows, why = [], collections.Counter()
    for fam in FAMS:
        cert = fam_cert(fam)
        cert = cert[a.shard::a.nshard] if a.nshard > 1 else cert
        for cr in cert:
            idx = int(cr["idx"])
            rec0 = lwb.get(idx)
            if rec0 is None:
                why["missing_lwb"] += 1
                continue
            head, concl, name = rec0["head"], rec0["concl"], rec0["name"]
            try:
                d = fam_parse(fam)(P(name, head, concl, rec0.get("subset"), idx))
            except Exception as e:                                          # noqa: BLE001
                why["parse_exc"] += 1
                rows.append({"ri": idx, "name": name, "fam": fam, "state": "parse_exc",
                             "reason": repr(e)[:60]})
                continue
            if not d or not d.get("ok"):
                why["parse_reject"] += 1
                rows.append({"ri": idx, "name": name, "fam": fam, "state": "parse_reject",
                             "reason": str((d or {}).get("reason"))[:60]})
                continue
            tr = d["tree"]
            tops, lops = fam_ops(fam, tr)
            prog = tops + lops
            # D-BARM-1 (newly registered for this arm): the family census "ops" is in the **old
            #  convention** for 58 rows (b1 19/b2 3/b3 26/b5 10): b3/b5 record only node ops and
            #  lack the leaf chains; b1/b2 double-count the three sqrt operators in tops/lops.
            #  The training-isomorphic encoding (cachebuild_v7, the convention actually trained on)
            #  is the gold standard => recompute wins, and each difference is registered per row.
            ops_diff = prog != list(cr.get("ops") or [])
            if ops_diff:
                why["census_ops_diff(kept)"] += 1
            else:
                why["census_ops_same"] += 1
            fo = CB.Fam(fam)
            tfr = fam_frames(fam, tr)
            fb = list(tfr[0]) if tfr and tfr[0] else list(CB.ZF)
            steps = CB.tree_steps(fo, tr, fb) + CB.leaf_steps(fo, tr)
            if [op for (op, _f, _ar) in steps] != prog:
                why["steps_order_mismatch"] += 1
                rows.append({"ri": idx, "name": name, "fam": fam, "state": "steps_order_mismatch"})
                continue
            sfa = [{str(k): ("" if v is None else str(v)) for k, v in (ar or {}).items()}
                   for (_o, _f, ar) in steps]
            res = [CB.san_res(f) for (_o, f, _a) in steps]
            rows.append({"ri": idx, "name": name, "head": head, "concl": concl,
                         "stmt": CB.stmt_of({"head": head, "concl": concl}),
                         "type4": list(CB.TYPE4), "seg": SEG, "family": fam,
                         "cls": "g44_%s_%s" % (fam, str(cr.get("kind"))),
                         "ops": prog, "sfa": sfa, "frames": res,
                         "ops_census": list(cr.get("ops") or []), "census_ops_diff": ops_diff,
                         "n_ops": len(prog), "n_top": len(tops), "n_leaf": len(lops),
                         "kind_struct": cr.get("kind"), "rs": cr.get("rs"),
                         "state": "ok", "c6_excluded": idx in c6})
            why["ok"] += 1
    md = jlsave(rows, ITEMS)
    ok = [r for r in rows if r.get("state") == "ok"]
    rep = {"schema": "gen44-barm-items/1", "created": now(), "n": len(rows), "why": dict(why),
           "n_ok": len(ok), "by_fam": dict(collections.Counter(r["family"] for r in ok)),
           "n_census_ops_diff": sum(1 for r in ok if r.get("census_ops_diff")),
           "census_ops_diff_by_fam": dict(collections.Counter(
               r["family"] for r in ok if r.get("census_ops_diff"))),
           "chainlen_hist": dict(collections.Counter(r["n_ops"] for r in ok)),
           "n_top_hist": dict(collections.Counter(r["n_top"] for r in ok)),
           "n_c6": sum(1 for r in rows if r.get("c6_excluded")),
           "items_md5": md, "items_path": ITEMS,
           # NOTE: Chinese report payload value kept verbatim (data, not a log line).
           "note": "行集 = 族件 certifiable 层; 编码 = cachebuild_v7 同构 (零改 import)"}
    jldump(rep, ITEMS_META)
    print(json.dumps(rep, ensure_ascii=False)[:1500], flush=True)
    return 0


# ================================================================ [2] GPU chain walk
def stage_walk(a):
    import gen44_eval as E44
    import gen34_eval as E34
    assert md5f(CKPT) == CKPT_MD5, "final-test ckpt md5 mismatch"
    assert md5f(os.path.join(_PKG_DIR, "gen44_eval.py")) == EVAL_MD5, "gen44_eval does not match the pinned (re-frozen) version"
    assert md5f(os.path.join(_PKG_DIR, "gen44_render_exam_v3.py")) == RENDER_MD5, "renderer v3 is not the unchanged version"

    def _payload_barm(side_rec):
        if side_rec.get("seg") == SEG:
            fr = side_rec.get("frames") or []
            return {"res": [list(x) if x is not None else None for x in fr],
                    "sfa": list(side_rec.get("sfa") or []), "dev_errs": [],
                    "res_src": "sidecar_device_frames(BARM; 与训练缓存 steps[k].res 同口径)"}
        return E44._ORIG_E34_PAYLOAD_OF(side_rec)

    E34._payload_of = _payload_barm
    ns = argparse.Namespace()
    ns.vocab = os.path.join(HERE, "cache", "CCP_VOCAB_V7.json")
    ns.cache = os.path.join(HERE, "cache", "GEN44_V7_ALL.bin")
    ns.side = os.path.join(HERE, "cache", "GEN44_V7_SIDECAR.jsonl")
    ns.exam_side = ITEMS
    ns.allow = os.path.join(HERE, "cache", "ccp_allow_v7.bin")
    ns.lat31 = "cppcuda/cache/lat_tab_v31.bin"
    ns.init = "ckpt/sla_v31f.pt"
    ns.ckpt = CKPT
    ns.seg = SEG
    ns.n = a.n if a.n > 0 else 10 ** 9
    ns.seed = 20260927
    ns.slot_side = os.path.join(HERE, "cache", "GEN44_V7_SLOT_SIDECAR.jsonl")
    ns.dev = a.dev
    ns.log = a.log
    ns.walk_out = WALK
    ns.state_shift = 0
    E44.phase_walk(ns, a.log)
    return 0


# ================================================================ [3] rendering -> manifest
def subsume(tr_copy, kinds, walked, n_top):
    """Leaf-level chain substitution: consume model ops along the device skeleton in order;
    returns (ok, why, used, n_leaf_used)."""
    lc = chain_leaves(tr_copy, kinds)
    used, cursor = [], int(n_top)
    for nd in lc:
        g = nd.get("got")
        if g is None:
            # Leaf has no device chain (the device itself cannot render it) => keep None and the
            # renderer fail-closes with None (row -> render_skip).
            used.append(None)
            continue
        gold = list(g.get("chain_ops") or [])
        take = len(gold)
        seg = list(walked[cursor:cursor + take])
        cursor += take
        g2 = dict(g)
        g2["chain_ops"] = seg
        nd["got"] = g2
        used.append(seg)
    return cursor, used


def stage_manifest(a):
    items = {int(r["ri"]): r for r in jlload(ITEMS) if r.get("state") == "ok"}
    w = json.load(open(WALK, encoding="utf-8"))
    lwb = load_lwb()
    rows, st = [], collections.Counter()
    n_ident_chk = n_ident_bad = 0
    for r in w["rows"]:
        idx = int(r["ri"])
        it = items.get(idx)
        ch = r.get("chain") or {}
        walked = [str(x) for x in (ch.get("walked") or [])]
        ent = {"key": str(idx), "name": r["name"], "idx": idx, "vol": (it or {}).get("family", "?").upper(),
               "fam": (it or {}).get("family"), "struct_state": "certifiable",
               "kind": (it or {}).get("kind_struct"), "rs": (it or {}).get("rs"),
               "ops_walked": walked, "ops_gold": list(ch.get("gold") or []),
               "walk_exact_hit": bool(ch.get("exact_hit")), "walk_first_hit": bool(ch.get("first_hit")),
               "walk_over_run": bool(ch.get("over_run")),
               "walk_terminated_end": bool(ch.get("terminated_end")),
               "n_walked": len(walked), "n_gold": len(ch.get("gold") or [])}
        if it is None:
            ent["state"] = "item_missing"
            rows.append(ent)
            st["item_missing"] += 1
            continue
        fam = it["family"]
        kinds = CL_KINDS[fam]
        try:
            d = fam_parse(fam)(P(r["name"], it["head"], it["concl"], None, idx))
        except Exception as e:                                              # noqa: BLE001
            d = None
            ent["reason"] = repr(e)[:70]
        if not d or not d.get("ok"):
            ent["state"] = "reparse_fail"
            rows.append(ent)
            st["reparse_fail"] += 1
            continue
        tr = d["tree"]
        tops, lops = fam_ops(fam, tr)
        # Per-row assertion: concatenated chain-leaf chains == family ops_leaf (order-sensitive).
        flat = []
        for nd in chain_leaves(tr, kinds):
            flat += list(((nd.get("got") or {}).get("chain_ops")) or [])
        ent["leafseg_ok"] = bool(flat == lops)
        if flat != lops:
            ent["state"] = "leafseg_fail"
            rows.append(ent)
            st["leafseg_fail"] += 1
            continue
        n_top = len(tops)
        ent["n_top"] = n_top
        dev_lines = fam_render(fam, tr)
        dev_txt = "\n".join(dev_lines) if dev_lines else None
        ent["proof_md5_device"] = (hashlib.md5(dev_txt.encode()).hexdigest()
                                   if dev_txt else None)
        if not walked:
            ent["state"] = "render_skip"
            ent["render_skip_why"] = "no_walked"
            rows.append(ent)
            st["render_skip"] += 1
            continue
        ent["walk_struct_hit"] = bool(walked[:n_top] == tops)
        if not ent["walk_struct_hit"]:
            # Sensitivity arm (registered): lenient text = device skeleton + model ops substituted
            # in order (excluded from the main reading when the structure does not match).
            tr_l = copy.deepcopy(tr)
            subsume(tr_l, kinds, walked, n_top)
            try:
                l_l = fam_render(fam, tr_l)
                ent["proofs_lenient"] = (["\n".join(l_l)] if l_l else None)
            except Exception:                                               # noqa: BLE001
                ent["proofs_lenient"] = None
            ent["state"] = "struct_dev"
            rows.append(ent)
            st["struct_dev"] += 1
            continue
        tr2 = copy.deepcopy(tr)
        cursor, used = subsume(tr2, kinds, walked, n_top)
        ent["ops_used"] = [op for seg in used if seg for op in seg]
        ent["n_leaf_dev_none"] = sum(1 for s in used if s is None)
        ent["n_ops_dropped_tail"] = max(0, len(walked) - cursor)
        try:
            lines = fam_render(fam, tr2)
        except Exception as e:                                              # noqa: BLE001
            ent["state"] = "render_exc"
            ent["reason"] = repr(e)[:70]
            rows.append(ent)
            st["render_exc"] += 1
            continue
        if not lines or not [x for x in lines if str(x).strip()]:
            ent["state"] = "render_skip"
            ent["render_skip_why"] = "v3_skip(模型链不可渲染)"
            rows.append(ent)
            st["render_skip"] += 1
            continue
        txt = "\n".join(lines)
        ent.update({"state": "rendered", "head": it["head"], "concl": it["concl"],
                    "proofs": [txt], "proof_md5": hashlib.md5(txt.encode()).hexdigest(),
                    "n_lines": len(lines), "renderer": "family_exam_single_leafchain_modelops"})
        rows.append(ent)
        st["rendered"] += 1
        if ent["walk_exact_hit"]:
            n_ident_chk += 1
            if dev_txt is not None and ent["proof_md5"] == ent["proof_md5_device"]:
                pass
            else:
                n_ident_bad += 1
                ent["ident_fail"] = True
    md = jlsave(rows, MAN)
    jrows = [r for r in rows if r["state"] == "rendered"]
    jpath = MAN.replace(".jsonl", "_JUDGE.jsonl")
    jmd5 = jlsave(jrows, jpath)
    byfam = {}
    for fam in FAMS:
        fr = [r for r in rows if r.get("fam") == fam]
        byfam[fam] = {"n_cert": len(fr),
                      "exact_hit": sum(1 for r in fr if r.get("walk_exact_hit")),
                      "struct_hit": sum(1 for r in fr if r.get("walk_struct_hit")),
                      "struct_dev": sum(1 for r in fr if r["state"] == "struct_dev"),
                      "rendered": sum(1 for r in fr if r["state"] == "rendered"),
                      "render_skip": sum(1 for r in fr if r["state"] == "render_skip"),
                      "other": sum(1 for r in fr if r["state"] not in
                                   ("rendered", "render_skip", "struct_dev")),
                      "over_run": sum(1 for r in fr if r.get("walk_over_run"))}
    rep = {"schema": "gen44-barm-manifest/1", "created": now(), "n": len(rows),
           "state_hist": dict(st), "by_fam": byfam, "manifest_md5": md,
           "manifest_path": MAN, "judge_manifest": {"path": jpath, "md5": jmd5, "n": len(jrows)},
           "walk_table": w.get("table"), "ckpt_md5": w.get("ckpt_md5"),
           "G_IDENT": {"n_exact_hit_checked": n_ident_chk, "n_bad": n_ident_bad,
                       "criterion": "exact_hit 行的代入渲染文本 == 装置渲染文本 (逐行 md5)"},
           "renderer_md5": md5f(os.path.join(_PKG_DIR, "gen44_render_exam_v3.py")),
           "policy": ("结构段 (walks[:n_top]) 必须 == 装置 tops 才渲染; 叶级链按装置骨架顺序代入模型 ops; "
                      "结构不符 => struct_dev 单列 (lenient 文本仅供敏感性, 不入主读数)")}
    jldump(rep, MAN_META)
    print(json.dumps({k: rep[k] for k in ("n", "state_hist", "G_IDENT")},
                     ensure_ascii=False), flush=True)
    return 0


# ================================================================ [4] aggregation (judging shards -> readings)
def stage_summary(a):
    import glob
    man = {int(r["idx"]): r for r in jlload(MAN)}
    shards = sorted(glob.glob(os.path.join(OUT, "GEN44_BARM_JUDGE_s*.json")))
    assert shards, "no judging shards (GEN44_BARM_JUDGE_s*.json)"
    per = {}
    prov = []
    n_cert_all = 0
    for p in shards:
        d = json.load(open(p, encoding="utf-8"))
        prov.append({"file": os.path.basename(p), "md5": md5f(p), "n_rows": d.get("n_rows"),
                     "n_certified": d.get("n_certified"), "wall_s": d.get("wall_s"),
                     "judge_ok": (d.get("judge") or {}).get("ok"),
                     "shard": d.get("shard")})
        for r in d["rows"]:
            k = int(r["key"])
            assert k not in per, ("duplicate key (D-S8F-1 class risk)", k)
            per[k] = r
        n_cert_all += int(d.get("n_certified") or 0)
    assert len(per) == sum(len(json.load(open(p, encoding="utf-8"))["rows"]) for p in shards)
    dev = {}
    if os.path.exists(LITALL):
        dd = json.load(open(LITALL, encoding="utf-8"))
        for k, v in (dd.get("by_vol_fam_literal") or {}).items():
            if k in FAMS:
                dev[k] = {"n_judged": v["n"], "certified": v["lit"], "rate": v["lit_rate"],
                          "wilson95": v["lit_wilson95"], "src": "GEN44_S8_LITALL_SUMMARY.json"}
    byfam = {}
    tot_j = tot_c = 0
    for fam in FAMS:
        fr = [r for r in man.values() if r.get("fam") == fam]
        rend = [r for r in fr if r["state"] == "rendered"]
        judged = [per[int(r["idx"])] for r in rend if int(r["idx"]) in per]
        cert = sum(1 for j in judged if j.get("certified"))
        rej = [j for j in judged if not j.get("certified")]
        reasons = collections.Counter()
        for j in rej:
            for x in (j.get("reasons") or ["?"]):
                reasons[str(x)] += 1
        n = len(judged)
        tot_j += n
        tot_c += cert
        byfam[fam] = {
            "n_cert_rows": len(fr),
            "exact_hit": sum(1 for r in fr if r.get("walk_exact_hit")),
            "exact_hit_rate": round(sum(1 for r in fr if r.get("walk_exact_hit")) / max(len(fr), 1), 4),
            "struct_hit": sum(1 for r in fr if r.get("walk_struct_hit")),
            "struct_dev": sum(1 for r in fr if r["state"] == "struct_dev"),
            "render_skip": sum(1 for r in fr if r["state"] == "render_skip"),
            "over_run": sum(1 for r in fr if r.get("walk_over_run")),
            "n_rendered": len(rend), "n_judged": n, "certified": cert,
            "rate": round(cert / n, 4) if n else None, "wilson95": wilson(cert, n),
            "reject_reasons": dict(reasons),
            "device_arm": dev.get(fam)}
    tbl = {"n_cert_total": sum(v["n_cert_rows"] for v in byfam.values()),
           "n_rendered": sum(v["n_rendered"] for v in byfam.values()),
           "n_judged": tot_j, "certified": tot_c,
           "rate": round(tot_c / tot_j, 4) if tot_j else None, "wilson95": wilson(tot_c, tot_j),
           "exact_hit": sum(v["exact_hit"] for v in byfam.values()),
           "struct_dev": sum(v["struct_dev"] for v in byfam.values()),
           "render_skip": sum(v["render_skip"] for v in byfam.values()),
           "over_run": sum(v["over_run"] for v in byfam.values())}
    rep = {"schema": "gen44-barm-summary/1", "created": now(), "ckpt": CKPT,
           "ckpt_md5": CKPT_MD5, "judge_md5": "7c31b58da9d7228cee31b9be581fe5dd",
           "n_shard_files": len(shards), "provenance": prov, "by_fam": byfam, "total": tbl,
           "device_arm_b_total": {"n_judged": 1525, "certified": 672, "rate": 0.4407,
                                  "wilson95": [0.4160, 0.4659],
                                  "src": "GEN44_FINAL_REPORT.md §2.2 (VB 族树臂)"},
           "note": ("主读数 = 模型走链 (结构相符) x 族单渲染 (叶级链代入) x 冻结判卷; "
                    "struct_dev / render_skip 单列不作 0; 未执行 != 判零")}
    md = jldump(rep, os.path.join(OUT, "GEN44_BARM_SUMMARY.json"))
    print(json.dumps({"total": tbl, "md5": md}, ensure_ascii=False), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("items", "walk", "manifest", "summary"))
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--log", default=os.path.join(HERE, "logs", "gen44_barm.log"))
    a = ap.parse_args()
    return {"items": stage_items, "walk": stage_walk,
            "manifest": stage_manifest, "summary": stage_summary}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
