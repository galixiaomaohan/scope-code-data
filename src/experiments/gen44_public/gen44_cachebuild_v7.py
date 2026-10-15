# -*- coding: utf-8 -*-
# gen44_cachebuild_v7.py -- [S4b] tree -> V7 cache record encoding (route A convention of
# ERRATUM6 §A) plus five gates
#
# Measurement convention (V44_DESIGN_ERRATUM6 §A; frozen ruling artifact,
# md5 c3110884874f38f89481e47a6bebf978):
#   1) The tree is expanded in **preorder**; each tree-level operator node yields one step
#      (operator ∈ V7 indices 207..219).
#   2) The step frame res = residual frame of that node's goal computed on the fly by
#      frame_of_ext; a tree-level structural operator with no residual definition takes the
#      parent goal frame; the root with no parent takes this row's first diagnostic frame (the
#      whole-problem residual frame in the device convention). Frame of a structural step = the
#      parent goal frame is an **explicitly registered** convention.
#      Sub-rule for the three sqrt_red operators (registered): iso = original goal residual
#      frame · sq = reduced goal residual frame · close = exhausted frame 0.
#   3) Leaf chains keep the parent convention (F.chain_of computes the frame per op) and are
#      concatenated after the tree-level steps.
#   4) sf_args follow the V7 artifact slot_schema_v7_new; leaf operators keep the parent payload
#      convention (SLOT_V6 path).
#   5) The row schema is exactly isomorphic to gen42_cachebuild.to_src_row_g42
#      (stmt/type4/program/steps/sf_args/accepted).
#   6) stmt = same form as the parent/exam sidecar: "<binders> ⊢ <concl>" (both the deep corpus
#      and the gen34 exam sidecar use this form).
#   7) Pool = deep corpus a2 (2399, kept bitwise: gen41 cache a2 blocks [0,2162)+[3556,3793))
#      ∪ the nine-family corpus (7200).
#      WARNING registered D-S6B-1: the parent gen42_cachebuild's `deep_rows[:2399]` under the
#      gen41 row order actually spans [2162,2399) gen41 rows (not the deep a2 held block), so 237
#      cached rows in the parent pool were misaligned with the sidecar. This file takes the true
#      deep blocks per the **registered** convention.
#
# Gates (all must pass before S5): G-RT roundtrip_bad=0 · G-BW deep_bitwise_bad=0 · G-HD header
#   assertions · G-OPS ops_unknown=[] · G-E2E stratified sample >=60 rows through exam single
#   rendering v3 + frozen judging module with 100% certified agreement
#   (G-E2E runs sharded via --stage e2e; --stage merge assembles the final gate artifact)
#
# Reads: out/GEN44_S4_CORPUS_MERGED.jsonl · gen41/cache/{g41_all.bin,g41_sidecar.jsonl} ·
#        gen36/cache/deep_all_a2.bin.segidx.json · gen42/cache/{CCP_VOCAB_V6.json,ccp_allow_v6.bin} ·
#        judging module
# Writes: scratch/ccp/gen44/** (cache/GEN44_V7_* · out/GEN44_S4B_* · lean/s4b_e2e/**)
import argparse
import collections
import hashlib
import json
import os
import re
import struct
import sys
import time

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = os.path.join(RES, "scratch", "ccp")
G41 = os.path.join(CCP, "gen41")
G42 = os.path.join(CCP, "gen42")
# the gen44 tree on the experiment host (cache/out/lean live there)
G44 = os.path.join(CCP, "gen44")
CACHE = os.path.join(G44, "cache")
OUT = os.path.join(G44, "out")
LEAN = os.path.join(G44, "lean")
for _p in (_PKG_DIR, G42, G41):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gen44_bcore as B                                                  # noqa: E402
import gen44_b345_ext as E                                               # noqa: E402
import gen44_fam_b789 as M                                               # noqa: E402
import gen41_common as C41                                               # noqa: E402
import gen41_cachebuild as CB41                                          # noqa: E402
import gen42_families as F                                               # noqa: E402

MS = 24
NVALS = 352
HEAD_V7 = 220
TYPE4 = [1, 0, 0, 0]
K = F.K_GRID
ZF = [float(x) for x in F.frame_scalar(0)]
JUDGE_MD5 = "7c31b58da9d7228cee31b9be581fe5dd"
JUDGE_PATH = os.path.join(RES, "scratch", "drct", "drct_common.py")
EXAM_MD5 = "908535e4d60e483be4238cc3f842d245"
CORPUS = os.path.join(OUT, "GEN44_S4_CORPUS_MERGED.jsonl")
FAMS = ["b%d" % i for i in range(1, 10)]
B_BASE = ("b1", "b2", "b6")
B_EXT = ("b3", "b4", "b5")
A2SEG = os.path.join(CCP, "gen36", "cache", "deep_all_a2.bin.segidx.json")
DEEP_TR = (0, 2162)            # deep a2 train block in the gen41 cache row order (a2 segidx a_train)
DEEP_HD = (3556, 3793)         # deep a2 held block in the gen41 cache row order (a2 segidx a_held)
TYPE4_WITNESS = "gen44-v7-cache/1"


def md5f(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


F32MAX = 3.4028234663852886e38


def f32(x):
    try:
        return struct.unpack("<f", struct.pack("<f", float(x)))[0]
    except OverflowError:
        return float("inf") if float(x) > 0 else float("-inf")


def san_res(fr):
    """Frame -> cache-representable domain (finite f32); any unrepresentable value makes the
    whole frame None (no truncation, no fabrication).
    Registered D-S6B-3: b7's pow/fact atoms overflow at the grid points (the corpus frames
    already contain inf)."""
    out = []
    for v in fr:
        try:
            x = float(v)
        except Exception:                                               # noqa: BLE001
            return None
        if x != x or x in (float("inf"), float("-inf")) or abs(x) > F32MAX:
            return None
        out.append(x)
    return out


def fr_eq(a, b, use32=False):
    if a is None or b is None or len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if use32:
            if f32(x) != f32(y):
                return False
        elif float(x) != float(y):
            return False
    return True


class Fam(object):
    """Per-family tree API binding (calls existing modules only, unchanged)."""

    def __init__(self, fam):
        self.fam = fam
        if fam in B_BASE:
            self.parse = B.parse_stmt_ext
            self.tops, self.lops, self.tframes = B.tree_ops, B.tree_leaf_ops, B.tree_frames
            self.render = lambda tr: B.render_tree(tr, "exam")
        elif fam in B_EXT:
            self.parse = E.parse_stmt_b345
            self.tops, self.lops = E.tree_ops_ext, E.tree_leaf_ops_ext
            self.tframes = E.tree_frames_ext
            self.render = lambda tr: E.render_tree_ext(tr, "exam")
        else:
            self.parse = lambda p: M.with_timeout(20.0, M.parse_stmt_ext789, p, fam)
            self.tops, self.lops, self.tframes = M.tree_ops, M.tree_leaf_ops, M.tree_frames
            self.render = lambda tr: M.render_tree(tr, "exam")

    def ops_tree(self, tr):
        return list(self.tops(tr))

    def ops_leaf(self, tr):
        return list(self.lops(tr))

    def frames(self, tr):
        return [list(x) if x is not None else None for x in list(self.tframes(tr))]


def fam_of(name):
    p = str(name).split("_")
    return p[1] if len(p) > 1 and p[1] in FAMS else None


# ================================================================ §1 tree -> steps (ERRATUM6 §A)
def resid_frame(rec):
    """Node goal residual frame (computed on the fly by frame_of_ext; no residual / exception ->
    None)."""
    try:
        c = (rec or {}).get("concl") or {}
        e = c["lhs"] - c["rhs"]
        free = sorted(e.free_symbols, key=str)
        return B.frame_of_ext(e, free, len(free))
    except Exception:                                                   # noqa: BLE001
        return None


def own_frame(n):
    """Residual frame of the node's own goal; a structural operator has no residual definition ->
    None (= take the parent goal frame)."""
    k = n.get("kind")
    if k == "leaf":
        fr = list((n.get("got") or {}).get("frames") or [])
        return fr[0] if fr else None
    if k == "sqrt_red":
        return resid_frame(n.get("rec"))
    if k in ("exists_w", "pred_ref", "pred_expand"):
        fr = list(n.get("frames") or [])
        return fr[0] if fr and fr[0] is not None else None
    return None


def ext_args(op, n):
    if op == "g44_witness_solve":
        return {"witness": " ".join(str(x) for x in (n.get("w_txt") or []))}
    if op == "g44_hyp_pred_ref":
        return {"hyp": str(n.get("hypname") or "")}
    if op == "g44_hyp_pred_expand":
        pr = n.get("pred") or {}
        return {"hyp": str(n.get("hypname") or ""),
                "pred": str(pr.get("kind") or pr.get("name") or "")}
    return {}


def args789(op, n):
    subs = ((n.get("rec") or {}).get("concl") or {}).get("subs") or {}
    if op == "g44_pow_atom":
        for d in subs.values():
            if d.get("kind") == "pow":
                return {"exp": str(d.get("exp"))}
        return {}
    if op == "g44_pow_cap":
        for d in subs.values():
            if d.get("kind") == "pow" and int(d.get("exp_int") or 0) > 256:
                return {"cap": int(d["exp_int"])}
        return {}
    if op == "g44_fact_atom":
        for d in subs.values():
            if d.get("kind") == "fact":
                return {"arg": str(d.get("arg"))}
        return {}
    if op == "g44_lib_atom":
        for d in subs.values():
            if d.get("kind") == "lib":
                return {"lib": str(d.get("name") or "")}
        return {}
    if op == "g44_relchain":
        return {"nrel": int(n.get("n_rel") or len(n.get("children") or []))}
    return {}


def tree_steps(fo, tr, fb):
    """Tree-level steps (preorder) -> [(op, frame, args)]."""
    out = []

    def walk(n, pfr):
        own = own_frame(n)
        eff = own if own is not None else (pfr if pfr is not None else fb)
        k = n.get("kind")
        if fo.fam in B_BASE:
            if k == "forall":
                out.append(("g44_intro_binder", eff,
                            {"intros": " ".join(n.get("intros") or [])}))
            elif k in ("iff", "and", "or"):
                out.append(("g44_goal_split", eff, {}))
            elif k == "sqrt_red":
                rel, side = str(n.get("rel2")), str(n.get("side_b"))
                f2 = resid_frame(n.get("rec2"))
                out.append(("g44_sqrt_iso", eff, {"rel": rel, "side": side}))
                out.append(("g44_sqrt_sq", f2 if f2 is not None else eff,
                            {"rel": rel, "side": side}))
                out.append(("g44_sqrt_close", list(ZF), {"rel": rel, "side": side}))
        elif fo.fam in B_EXT:
            if k in E.NEW_KINDS:
                for op in (n.get("ops") or []):
                    out.append((op, eff, ext_args(op, n)))
            elif k == "forall":
                out.append(("g44_intro_binder", eff,
                            {"intros": " ".join(n.get("intros") or [])}))
            elif k in ("iff", "and", "or"):
                out.append(("g44_goal_split", eff, {}))
        else:
            if k == "leaf":
                for op in (n.get("g_ops") or []):
                    out.append((op, eff, args789(op, n)))
            elif k == "and" and n.get("src") == "relchain":
                out.append(("g44_relchain", eff, args789("g44_relchain", n)))
        for c in n.get("children") or []:
            walk(c, eff)

    walk(tr, None)
    return out


def leaf_args(op, i, got):
    got = got or {}
    if op == "g42_s_reduce":
        return {"var": str(got.get("reduce_var") or "nn1")}
    steps = got.get("steps") or []
    if op == "g42_sb_step" and i < len(steps) and isinstance(steps[i], dict):
        st = steps[i]
        return {k: str(st.get(k, "")) for k in ("var", "hyp")}
    if op == "g42_sb_close":
        pl = got.get("route_payload") or {}
        return {"value": str(pl.get("value") or got.get("claimed") or "")}
    pl = got.get("route_payload") or {}
    return {str(k): str(v) for k, v in pl.items() if k != "frame"}


def leaf_steps(fo, tr):
    """Leaf-chain steps (preorder; one step per op, frame computed on the fly by the parent
    F.chain_of)."""
    out = []

    def walk(n):
        if n.get("kind") in ("leaf", "sqrt_red"):
            got = n.get("got") or {}
            ops = list(got.get("chain_ops") or [])
            fr = list(got.get("frames") or [])
            for i, op in enumerate(ops):
                f = fr[i] if i < len(fr) else (fr[-1] if fr else None)
                out.append((op, f, leaf_args(op, i, got)))
        for c in n.get("children") or []:
            walk(c)

    walk(tr)
    return out


def stmt_of(row):
    head = str(row.get("head") or "")
    concl = str(row.get("concl") or "")
    h = head.rstrip()
    if h.endswith(":"):
        h = h[:-1]
    h = re.sub(r"^theorem\s+", "", h.strip()).strip()
    m = re.match(r"^([A-Za-z0-9_.']+)\s*(.*)$", h, re.S)
    b = (m.group(2) if m else "").strip()
    return ("%s \u22a2 %s" % (b, concl)) if b else ("\u22a2 %s" % concl)


def encode_row(fo, row):
    """Corpus row -> (src_row, diag); on failure -> (None, diag)."""
    dg = {}
    try:
        d = fo.parse(B.P(row["name"], row["head"], row["concl"], None, None))
    except Exception as e:                                              # noqa: BLE001
        return None, {"why": "reparse_exc:" + type(e).__name__}
    if not isinstance(d, dict) or not d.get("ok"):
        return None, {"why": "reparse_fail:" + str((d or {}).get("reason"))}
    tr = d["tree"]
    tops, lops = fo.ops_tree(tr), fo.ops_leaf(tr)
    prog = list(tops) + list(lops)
    tfr = fo.frames(tr)
    fb = list(tfr[0]) if tfr and tfr[0] else list(ZF)
    ts = tree_steps(fo, tr, fb)
    ls = leaf_steps(fo, tr)
    steps = ts + ls
    # Gate 1: chain roundtrip.
    dg["program_ok"] = bool(prog == list(row["chain_ops"]))
    dg["n_step_eq"] = bool(len(steps) == len(prog))
    dg["n_tree_ops"] = len(tops)
    dg["n_leaf_ops"] = len(lops)
    # Gate 1b: exam single-rendering roundtrip (tree -> text versus the corpus proof, verbatim;
    # **diagnostic only**).
    #   Registered D-S6B-2: the D-B7 fix in gen44_bcore.py (sqrt leaf closing script, mtime 04:27)
    #   postdates B6 corpus generation (03:26), so the corpus proof field is **stale** for sqrt
    #   leaf rows; the encoding itself depends only on tree/chain (the G-RT main criterion), not
    #   on the proof text. The training sidecar target text is re-rendered with the **current**
    #   renderer (consistent with the exam phase); differences are registered per family.
    rtxt = None
    try:
        rl = fo.render(tr)
        rtxt = "\n".join(rl) if rl is not None else None
    except Exception:                                                   # noqa: BLE001
        rtxt = None
    dg["render_ok"] = bool(rtxt is not None)
    dg["render"] = rtxt
    dg["proof_ok"] = bool(rtxt is not None and rtxt == str(row.get("proof")))
    # Gate 2: bitwise comparison of the first 2 diagnostic frames.
    #   The source of the corpus frames differs by family (per family tree_frames): b1/b2/b6 and
    #   b7/b8/b9 use leaf-chain frames; b3/b4/b5 use NEW_KINDS tree-level node frames => two
    #   separate criteria, with the main criterion = recomputed diagnostic frames vs corpus frames.
    lfr = [f for (_o, f, _a) in ls if f is not None]
    nfr = [f for (o, f, _a) in ts
           if o in ("g44_witness_solve", "g44_hyp_pred_ref", "g44_hyp_pred_expand")
           and f is not None]
    cfr = [list(x) for x in (row.get("frames") or [])]
    src_leaf = fo.fam not in B_EXT
    n2 = min(2, len(cfr))
    ok_t = all(i < len(tfr) and tfr[i] is not None and fr_eq(tfr[i], cfr[i])
               for i in range(n2)) if n2 else False
    ok_l = (all(i < len(lfr) and fr_eq(lfr[i], cfr[i]) for i in range(n2))
            if (n2 and src_leaf) else True)
    ok_l32 = (all(i < len(lfr) and fr_eq(lfr[i], cfr[i], True) for i in range(n2))
              if (n2 and src_leaf) else True)
    ok_n = ((len(nfr) >= 1 and fr_eq(nfr[0], cfr[0])) if (n2 and not src_leaf) else True)
    ok_n2nd = ((len(cfr) > 1 and fr_eq(cfr[1], ZF)) if (n2 > 1 and not src_leaf) else True)
    dg["bw_main_bad"] = not ok_t
    dg["bw_leaf_bad"] = not ok_l
    dg["bw_leaf32_bad"] = not ok_l32
    dg["bw_node_bad"] = not ok_n
    dg["bw_node_second_bad"] = not ok_n2nd
    dg["bw_src"] = "leaf" if src_leaf else "node"
    dg["bw_exact_bad"] = bool(dg["bw_main_bad"] or dg["bw_leaf_bad"] or dg["bw_node_bad"])
    dg["n_frame_missing"] = sum(1 for (_o, f, _a) in steps if f is None)
    # Row.
    st, sfa = [], []
    n_res_none = 0
    for (op, fr, ar) in steps:
        f = san_res(fr) if fr is not None else None
        if fr is not None and f is None:
            n_res_none += 1
        st.append({"op": op, "ok": True, "err": None, "res": f, "state": "ok"})
        sfa.append({str(k): ("" if v is None else str(v)) for k, v in (ar or {}).items()})
    dg["n_res_none"] = n_res_none
    src = {"stmt": stmt_of(row), "type4": list(TYPE4), "program": prog, "steps": st,
           "sf_args": sfa, "episode": None, "accepted": True, "readout_only": False,
           "name": row["name"], "cls": "g44_%s_%s" % (fo.fam, str(row.get("kind")))}
    return src, dg


# ================================================================ §2 corpus / deep blocks / order
def load_corpus(path, limit=0):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if limit:
        rows = rows[:limit]
    return rows


def order_split(rows):
    """Same form as the parent: md5(name)[:8] % 10 == 7 => held."""
    tr = [r for r in rows if int(C41.md5s(r["name"])[:8], 16) % 10 != 7]
    hd = [r for r in rows if int(C41.md5s(r["name"])[:8], 16) % 10 == 7]
    return tr, hd


def deep_blocks():
    """The two deep a2 blocks (2399 rows) bitwise plus the corresponding gen41 sidecar blocks
    (same row order) plus assertions."""
    rows, hdr, _b = CB41.read_raw(C41.CACHE_G41)
    a2 = json.load(open(A2SEG, encoding="utf-8"))
    seg = a2["segments"]
    assert DEEP_TR[1] - DEEP_TR[0] == seg["a_train"][1] - seg["a_train"][0], seg
    assert DEEP_HD[1] - DEEP_HD[0] == seg["a_held"][1] - seg["a_held"][0], seg
    assert len(rows) >= DEEP_HD[1], len(rows)
    side41 = [json.loads(l) for l in open(C41.RENDER_SIDE_G41, encoding="utf-8") if l.strip()]
    assert len(side41) == len(rows), (len(side41), len(rows))
    d_tr = rows[DEEP_TR[0]:DEEP_TR[1]]
    d_hd = rows[DEEP_HD[0]:DEEP_HD[1]]
    s_tr = side41[DEEP_TR[0]:DEEP_TR[1]]
    s_hd = side41[DEEP_HD[0]:DEEP_HD[1]]
    assert all(x.get("seg") == "a_train" for x in s_tr), "deep train block sidecar seg mismatch"
    assert all(x.get("seg") == "a_held" for x in s_hd), "deep held block sidecar seg mismatch"
    return (d_tr, s_tr, d_hd, s_hd)


# ================================================================ §3 stage build
def stage_build(a):
    t0 = time.time()
    import gen44_vocab7 as V7
    ctx = V7.load_ctx7()
    assert int(ctx["vocab"]["head"]) == HEAD_V7, ctx["vocab"]["head"]
    jm0 = md5f(JUDGE_PATH)
    assert jm0 == JUDGE_MD5, ("judging module md5 mismatch before judging", jm0)
    assert md5f(os.path.join(_PKG_DIR, "gen44_render_exam_v3.py")) == EXAM_MD5
    rows = load_corpus(a.corpus, a.limit)
    tr_rows, hd_rows = order_split(rows)
    order = tr_rows + hd_rows
    ntr = len(tr_rows)
    print("[S4b] corpus %d rows (train %d / held %d)" % (len(order), ntr, len(hd_rows)),
          flush=True)
    srcs, diags, fam_hist = [], [], collections.Counter()
    n_parse_bad, n_prog_bad, n_proof_bad, n_fr_missing, n_render_bad = 0, 0, 0, 0, 0
    n_bw_main = n_bw_leaf = n_bw_leaf32 = n_bw_node = n_bw_node2 = 0
    n_res_none = 0
    res_none_by_fam = collections.Counter()
    bw_by_fam = collections.Counter()
    proof_diff_by_fam = collections.Counter()
    proof_dg = []
    render_out = open(a.render_dump, "w", encoding="utf-8")
    for r in order:
        fam = fam_of(r["name"])
        fam_hist[fam] += 1
        src, dg = encode_row(Fam(fam), r)
        if src is None:
            n_parse_bad += 1
            diags.append({"name": r["name"], "fam": fam, "why": dg.get("why")})
            continue
        srcs.append(src)
        if not dg["program_ok"] or not dg["n_step_eq"]:
            n_prog_bad += 1
            diags.append({"name": r["name"], "fam": fam, "why": "program"})
        if not dg["render_ok"]:
            n_render_bad += 1
        elif not dg["proof_ok"]:
            n_proof_bad += 1
            proof_diff_by_fam[fam] += 1
            if len(proof_dg) < 5:
                proof_dg.append({"name": r["name"], "fam": fam})
        render_out.write(json.dumps({"name": r["name"], "fam": fam,
                                     "md5": hashlib.md5((dg.get("render") or "").encode()
                                                        ).hexdigest(),
                                     "text": dg.get("render")}, ensure_ascii=False) + "\n")
        n_bw_main += int(dg["bw_main_bad"])
        n_bw_leaf += int(dg["bw_leaf_bad"])
        n_bw_leaf32 += int(dg["bw_leaf32_bad"])
        n_bw_node += int(dg["bw_node_bad"])
        n_bw_node2 += int(dg["bw_node_second_bad"])
        if dg["bw_exact_bad"]:
            bw_by_fam[fam] += 1
        n_fr_missing += dg["n_frame_missing"]
        n_res_none += dg.get("n_res_none", 0)
        if dg.get("n_res_none"):
            res_none_by_fam[fam] += dg["n_res_none"]
    render_out.close()
    n_bw_bad = n_bw_main + n_bw_leaf + n_bw_node + n_bw_node2
    print("[S4b] encoded %d · parse_bad %d · program_bad %d · render_bad %d · proof_diff %d "
          "· bw(main %d leaf %d leaf32 %d node %d node2 %d) · frame_missing %d · %.1fs"
          % (len(srcs), n_parse_bad, n_prog_bad, n_render_bad, n_proof_bad, n_bw_main,
             n_bw_leaf, n_bw_leaf32, n_bw_node, n_bw_node2, n_fr_missing,
             time.time() - t0), flush=True)
    assert not n_parse_bad and not n_prog_bad, "reparse/chain roundtrip failed => stop condition"
    # ---- cache ----
    recs_new, t4set, stats = CB41.CCF.build(ctx, srcs)
    assert len(recs_new) == len(srcs) == len(order), (len(recs_new), len(srcs))
    d_tr, s_tr, d_hd, s_hd = deep_blocks()
    n_tr, n_hd = len(d_tr), len(d_hd)
    all_recs = list(d_tr) + recs_new[:ntr] + list(d_hd) + recs_new[ntr:]
    n_train = n_tr + ntr
    segs = {"a_train": [0, n_train], "a_held": [n_train, len(all_recs)]}
    segidx = {"cache": a.out_cache, "schema_rev": "ccpcd5/1-gen44v7",
              "ops": {"base": 161, "end": 154, "head": HEAD_V7, "n_vals": NVALS,
                      "g44": {k: int(v) for k, v in sorted(ctx["ops_g44"].items())}},
              "segments": segs,
              "held_rows": list(range(n_train, len(all_recs))),
              # NOTE: Chinese data value kept verbatim (written into the cache segidx artifact).
              "held_rule": "md5(name)[:8] % 10 == 7 (九族语料); 深语料按 a2 segidx 块",
              "n_held": len(all_recs) - n_train,
              "vocab_md5": md5f(os.path.join(CACHE, "CCP_VOCAB_V7.json")),
              "deep_src": {"path": C41.CACHE_G41, "blocks": [list(DEEP_TR), list(DEEP_HD)],
                           "rows": n_tr + n_hd, "a2_kept_bitwise": True,
                           # NOTE: Chinese report payload value kept verbatim (data, not a log line).
                           "note": "深 a2 2399 行按 a2 segidx 两块逐位保留 (D-S6B-1 登记: "
                                   "父件 gen42 在 [2162,2399) 取到 gen41 行)"},
              "corpus44": {"path": a.corpus, "md5": md5f(a.corpus), "rows": len(order)},
              "encoder": {"file": os.path.basename(__file__), "md5": md5f(__file__),
                          "rule": "V44_DESIGN_ERRATUM6 §A"}}
    os.makedirs(CACHE, exist_ok=True)
    md5 = CB41.write_cache(a.out_cache, all_recs, ctx, segidx)
    # ---- gates ----
    back, bhdr, _b2 = CB41.read_raw(a.out_cache)
    rt_bad = [i for i, (x, y) in enumerate(zip(back, all_recs)) if CB41.row_equal(x, y)]
    deep_bad = []
    for i in range(n_tr):
        if CB41.row_equal(back[i], d_tr[i]):
            deep_bad.append(i)
    for j in range(n_hd):
        if CB41.row_equal(back[n_train + j], d_hd[j]):
            deep_bad.append(n_train + j)
    hdr_chk = {"n_rows": bhdr[0] == len(all_recs), "head": bhdr[12] == HEAD_V7,
               "MS": bhdr[13] == MS, "nvals": bhdr[14] == NVALS,
               "base": bhdr[11] == 161, "end": bhdr[10] == 154,
               "tmax": bhdr[9] == max(len(r["labels"]) for r in all_recs)}
    ops_unknown = sorted({o for r in srcs for o in r["program"] if o not in ctx["opmap"]})
    empty_slot = sorted({o for r in srcs for o in r["program"]
                         if o.startswith("g44_") and not ctx["per"].get(o)
                         and V7.SLOT_V7.get(o)})
    jm1 = md5f(JUDGE_PATH)
    rep = {"schema": TYPE4_WITNESS, "created": now_iso(),
           "corpus": {"path": a.corpus, "md5": md5f(a.corpus), "rows": len(order),
                      "fam_hist": dict(fam_hist), "n_train": ntr, "n_held": len(hd_rows)},
           "cache": {"path": a.out_cache, "md5": md5, "bytes": os.path.getsize(a.out_cache),
                     "n_rows": len(all_recs), "n_deep": n_tr + n_hd, "n_new": len(recs_new),
                     "segments": segs},
           "gates": {
               "G-RT": {"roundtrip_bad": len(rt_bad), "program_bad": n_prog_bad,
                        "reparse_bad": n_parse_bad, "render_bad": n_render_bad,
                        "proof_text_diff_vs_corpus": n_proof_bad,
                        "proof_diff_by_fam": dict(proof_diff_by_fam),
                        # NOTE: Chinese report payload value kept verbatim (data, not a log line).
                        "note": ("G-RT 主判据 = 重建树/链 == 语料 chain_ops (ERRATUM6 §A); "
                                 "proof 文本差异为 D-S6B-2 登记的语料陈旧 (D-B7 修复后重渲), "
                                 "不作 FAIL 判据, 但侧车目标文本取现渲染器重渲"),
                        "verdict": "PASS" if (not rt_bad and not n_prog_bad
                                              and not n_parse_bad
                                              and not n_render_bad) else "FAIL"},
               "G-BW": {"deep_bitwise_bad": n_bw_main, "leaf_step_bad": n_bw_leaf,
                        "leaf_step_f32_bad": n_bw_leaf32, "node_step_bad": n_bw_node,
                        "node_second_frame_bad": n_bw_node2, "by_fam": dict(bw_by_fam),
                        "frame_missing": n_fr_missing,
                        "res_none_unrepresentable": n_res_none,
                        "res_none_by_fam": dict(res_none_by_fam),
                        # NOTE: Chinese report payload value kept verbatim (data, not a log line).
                        "criterion": ("语料 frames = 族件 tree_frames 前 2 诊断帧 (b1/b2/b6+b7/b8/b9 "
                                      "为叶链帧, b3/b4/b5 为 NEW_KINDS 节点帧); 主判 = 重算诊断帧 "
                                      "bitwise 相等, 并出编码步帧一致性; D-S6B-3: 超 f32 域帧值 "
                                      "整帧记 None (不截断)"),
                        "verdict": "PASS" if (n_bw_bad == 0 and n_fr_missing == 0
                                              and n_bw_leaf32 == 0) else "FAIL"},
               "G-HD": {"checks": hdr_chk, "verdict": "PASS" if all(hdr_chk.values())
                        else "FAIL"},
               "G-OPS": {"ops_unknown": ops_unknown,
                         "ops_g44_declared_but_no_slot": empty_slot,
                         "n_distinct_ops": len({o for r in srcs for o in r["program"]}),
                         "verdict": "PASS" if not ops_unknown else "FAIL"},
               # NOTE: Chinese report payload value kept verbatim (data, not a log line).
               "G-E2E": {"status": "PENDING", "note": "--stage e2e 分片跑后 --stage merge"}},
           "slot_stats": {k: int(v) for k, v in sorted(stats.items()) if k != "L"},
           "judge": {"path": JUDGE_PATH, "md5_pre": jm0, "md5_post": jm1,
                     "ok": jm0 == jm1 == JUDGE_MD5},
           "encoder": {"file": os.path.basename(__file__), "md5": md5f(__file__)},
           "vocab": {"path": os.path.join(CACHE, "CCP_VOCAB_V7.json"),
                     "md5": md5f(os.path.join(CACHE, "CCP_VOCAB_V7.json")),
                     "head_v7": HEAD_V7},
           "dirty_rows": diags[:20], "n_dirty": len(diags),
           "proof_diff_rows": proof_dg,
           "render_dump": {"path": a.render_dump, "md5": md5f(a.render_dump)},
           "wall_s": round(time.time() - t0, 1)}
    B.jdump(rep, a.partial)
    print(json.dumps({k: rep["gates"][k].get("verdict", rep["gates"][k].get("status"))
                      for k in ("G-RT", "G-BW", "G-HD", "G-OPS", "G-E2E")},
                     ensure_ascii=False), flush=True)
    print("partial -> %s (md5 %s) · cache md5 %s" % (a.partial, md5f(a.partial), md5),
          flush=True)
    return rep


# ================================================================ §4 stage side (training sidecar)
def stage_side(a):
    """Sidecar in training format (row order aligned with the cache row by row; deep blocks
    copied bitwise from the two gen41 blocks; new families follow the parent form)."""
    import gen44_vocab7 as V7
    ctx = V7.load_ctx7()
    tok = ctx["tok"]
    import v31_model as vm
    M_TASK = int(vm.st.M_TASK)
    rows = load_corpus(a.corpus, a.limit)
    tr_rows, hd_rows = order_split(rows)
    order = tr_rows + hd_rows
    ntr = len(tr_rows)
    d_tr, s_tr, d_hd, s_hd = deep_blocks()
    n_tr, n_hd = len(d_tr), len(d_hd)
    n_train = n_tr + ntr
    rmap, n_re_used, n_corpus_used = {}, 0, 0
    if a.render_dump and os.path.exists(a.render_dump):
        for l in open(a.render_dump, encoding="utf-8"):
            if l.strip():
                x = json.loads(l)
                rmap[x["name"]] = x.get("text")
    side = []
    for i, r in enumerate(order):
        st = stmt_of(r)
        tg = rmap.get(r["name"])
        if tg is None:
            tg = str(r["proof"])
            n_corpus_used += 1
        else:
            n_re_used += 1
        assert tg and "\n" in tg or tg, ("empty target text", r["name"])
        p = tok(CB41.PROMPT_TMPL % st, add_special_tokens=False)["input_ids"]
        t = tok("```lean4\n" + tg + "\n```", add_special_tokens=False)["input_ids"]
        assert t, ("empty target tokens", r["name"])
        ids = [M_TASK] + [int(x) for x in p] + [int(x) for x in t]
        side.append({"ri": 0, "name": r["name"], "cls": "g44_%s_%s" % (fam_of(r["name"]),
                                                                       str(r.get("kind"))),
                     "seg": "a_train" if i < ntr else "a_held", "np": len(p) + 1,
                     "n": len(ids), "ids": ids, "stmt": st, "head": r["head"],
                     "concl": r["concl"], "type4": list(TYPE4),
                     "ops": list(r["chain_ops"]),
                     "sfa": [{"op": o} for o in r["chain_ops"]],
                     "vars": [], "dom_dump": None, "family": "g44_%s" % fam_of(r["name"]),
                     "chain_len": len(r["chain_ops"]), "res": None,
                     # NOTE: Chinese report payload value kept verbatim (data, not a log line).
                     "res_note": "侧车不载金标 res; 状态由装置现算"})
    out = []
    for j, x in enumerate(s_tr):
        y = dict(x)
        y["ri"] = len(out)
        y["seg"] = "a_train"
        out.append(y)
    for j, x in enumerate(side[:ntr]):
        y = dict(x)
        y["ri"] = len(out)
        out.append(y)
    for j, x in enumerate(s_hd):
        y = dict(x)
        y["ri"] = len(out)
        y["seg"] = "a_held"
        out.append(y)
    for j, x in enumerate(side[ntr:]):
        y = dict(x)
        y["ri"] = len(out)
        out.append(y)
    with open(a.side_out, "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    chk = {"n_rows": len(out) == n_tr + len(order) + n_hd,
           "ri_seq": all(x["ri"] == i for i, x in enumerate(out)),
           "deep_tr_ids_bitwise": all(out[i]["ids"] == s_tr[i]["ids"] for i in range(n_tr)),
           "deep_hd_ids_bitwise": all(out[n_train + j]["ids"] == s_hd[j]["ids"]
                                      for j in range(n_hd)),
           "np_range": all(0 < x["np"] < x["n"] for x in out),
           "seg_consistent": all((x["seg"] == "a_train") == (x["ri"] < n_train)
                                 for x in out),
           "no_gold_res": all(x["res"] is None for x in out),
           "stmt_kind": all(("\u22a2" in x["stmt"]) for x in out)}
    rep = {"schema": "gen44-v7-sidecar/1", "created": now_iso(),
           "side": {"path": a.side_out, "md5": md5f(a.side_out), "bytes":
                    os.path.getsize(a.side_out)},
           "n": len(out), "n_train": n_train,
           "segment": "a_train %d / a_held %d" % (n_train, len(out) - n_train),
           "deep_blocks": {"tr": list(DEEP_TR), "hd": list(DEEP_HD),
                           "bitwise_from_g41_sidecar": True},
           "prompt_tmpl_md5": C41.md5s(CB41.PROMPT_TMPL), "M_TASK": M_TASK,
           "target_src": {"re_render": n_re_used, "corpus_proof": n_corpus_used,
                          "render_dump": a.render_dump,
                          # NOTE: Chinese report payload value kept verbatim (data, not a log line).
                          "note": "D-S6B-2: 目标文本取现渲染器重渲 (与考试期一致)"},
           "checks": chk, "all_pass": all(chk.values())}
    B.jdump(rep, a.side_gate)
    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)
    assert all(chk.values()), chk
    return rep


# ================================================================ §5 stage e2e (G-E2E)
def e2e_select(rows, n_per_fam):
    byfam = collections.OrderedDict((f, []) for f in FAMS)
    for r in sorted(rows, key=lambda x: x["name"]):
        f = fam_of(r["name"])
        if f in byfam:
            byfam[f].append(r)
    sel = []
    for f in FAMS:
        rs = byfam[f]
        if not rs:
            continue
        step = max(1, len(rs) // n_per_fam)
        sel += rs[::step][:n_per_fam]
    return sel


def head_judge(h, norm=True):
    """Judging-convention head (LWB convention): for `<head> <concl> := by` the head must carry a
    trailing colon.
    Registered D-S6B-4: the 2400 synthetic corpus rows for b3/b4/b5 have heads without the
    trailing colon (bgate.gen_corpus normalizes, the b345_ext route does not), so judging them
    directly yields a Lean syntax error (unexpected token); this file normalizes to the **same
    LWB convention** and reports both readings (before and after normalization), without
    loosening the gate."""
    h = str(h or "")
    if norm:
        t = h.rstrip()
        if not t.endswith(":"):
            t += " :"
        return t
    return h


def stage_e2e(a):
    jm0 = md5f(JUDGE_PATH)
    assert jm0 == JUDGE_MD5, ("judging module md5 mismatch before judging", jm0)
    assert md5f(os.path.join(_PKG_DIR, "gen44_render_exam_v3.py")) == EXAM_MD5
    rows = load_corpus(a.corpus, a.limit)
    sel = e2e_select(rows, a.e2e_n)
    sel = sel[a.shard::a.nshard]
    wd = os.path.join(LEAN, "s4b_e2e")
    os.makedirs(wd, exist_ok=True)
    out = []
    t0 = time.time()
    for i, r in enumerate(sel):
        fam = fam_of(r["name"])
        ent = {"name": r["name"], "fam": fam, "idx": i, "stratum": fam}
        try:
            d = Fam(fam).parse(B.P(r["name"], r["head"], r["concl"], None, None))
            tr = d["tree"]
            txt = None if not d.get("ok") else Fam(fam).render(tr)
        except Exception as e:                                          # noqa: BLE001
            tr, txt = None, None
            ent["err"] = type(e).__name__
        if not txt:
            ent["state"] = "render_skip"
            out.append(ent)
            continue
        proof = "\n".join(txt)
        ent["proof_md5"] = B.md5s(proof)
        ent["proof_eq_corpus"] = bool(proof == str(r.get("proof")))
        hj = head_judge(r["head"], bool(a.head_normalize))
        ent["head_normalized"] = bool(hj != str(r["head"]))
        jr = B.judge_nice(r["name"], hj, r["concl"], proof, wd,
                          "s4b%s" % fam, timeout=a.timeout, maxj=a.max_lean_jobs,
                          gate_wait_s=a.gate_wait_s)
        ent["certified"] = bool(jr["certified"])
        ent["reasons"] = jr["reasons"]
        ent["lean"] = jr["lean"]
        ent["n1n2"] = jr["n1n2"]
        ent["state"] = "certified" if ent["certified"] else "judged_reject"
        out.append(ent)
        print("[e2e s%d %d/%d] %s %s cert=%d %.0fs" % (
            a.shard, i + 1, len(sel), fam, ent["state"],
            sum(1 for x in out if x.get("certified")), time.time() - t0), flush=True)
    jm1 = md5f(JUDGE_PATH)
    judged = [x for x in out if x.get("state") in ("certified", "judged_reject")]
    nc = sum(1 for x in judged if x["certified"])
    rep = {"schema": "gen44-s4b-e2e/1", "created": now_iso(), "shard": a.shard,
           "nshard": a.nshard, "seed_policy": "per-fam 等距抽样 (deterministic)",
           "n_sample": len(out), "n_judged": len(judged), "n_certified": nc,
           "rate": round(nc / max(1, len(judged)), 4),
           "wilson95": B.wilson(nc, len(judged)),
           "by_fam": {f: {"n": sum(1 for x in out if x["fam"] == f),
                          "judged": sum(1 for x in judged if x["fam"] == f),
                          "certified": sum(1 for x in judged if x["fam"] == f
                                           and x["certified"])} for f in FAMS},
           "state_hist": dict(collections.Counter(x.get("state") for x in out)),
           "reasons_hist": dict(collections.Counter(
               (x.get("reasons") or ["?"])[0] for x in judged if not x["certified"])),
           "proof_eq_corpus_bad": sum(1 for x in out if x.get("proof_eq_corpus") is False),
           "n_head_normalized": sum(1 for x in out if x.get("head_normalized")),
           "head_normalize": bool(a.head_normalize),
           # NOTE: Chinese report payload value kept verbatim (data, not a log line).
           "head_note": ("D-S6B-4: b3/b4/b5 语料 head 缺 LWB 尾冒号 => 按同口径归一后判卷"),
           "renderer": "exam_single_v3", "renderer_md5": EXAM_MD5,
           "judge": {"md5_pre": jm0, "md5_post": jm1, "ok": jm0 == jm1 == JUDGE_MD5},
           "wall_s": round(time.time() - t0, 1), "rows": out}
    p = a.e2e_out if a.nshard == 1 else a.e2e_out + ".s%d" % a.shard
    B.jdump(rep, p)
    print("e2e -> %s certified %d/%d (judged) wall %.0fs" % (p, nc, len(judged),
                                                             time.time() - t0), flush=True)
    print(json.dumps({k: rep[k] for k in ("n_sample", "n_judged", "n_certified", "rate",
                                          "state_hist", "reasons_hist")},
                     ensure_ascii=False), flush=True)
    return rep


# ================================================================ §6 stage merge
def stage_merge(a):
    part = json.load(open(a.partial, encoding="utf-8"))
    shards = []
    for s in range(a.nshard):
        p = a.e2e_out + (".s%d" % s if a.nshard > 1 else "")
        assert os.path.exists(p), p
        shards.append(json.load(open(p, encoding="utf-8")))
    n_s = sum(x["n_sample"] for x in shards)
    n_j = sum(x["n_judged"] for x in shards)
    n_c = sum(x["n_certified"] for x in shards)
    by_fam = {}
    for f in FAMS:
        by_fam[f] = {"n": sum(x["by_fam"][f]["n"] for x in shards),
                     "judged": sum(x["by_fam"][f]["judged"] for x in shards),
                     "certified": sum(x["by_fam"][f]["certified"] for x in shards)}
    rs = collections.Counter()
    for x in shards:
        rs.update(x["reasons_hist"])
    peq = sum(x["proof_eq_corpus_bad"] for x in shards)
    jm_ok = all(x["judge"]["ok"] for x in shards) and \
        all(x["judge"]["md5_post"] == JUDGE_MD5 for x in shards)
    agree100 = bool(n_j == n_s and n_c == n_j and n_s >= 60)
    g = {"status": "DONE", "n_sample": n_s, "n_judged": n_j, "n_certified": n_c,
         "certified_rate_of_judged": round(n_c / max(1, n_j), 4),
         "wilson95": B.wilson(n_c, n_j), "by_fam": by_fam,
         "reasons_hist": dict(rs), "proof_eq_corpus_bad": peq,
         "renderer": "exam_single_v3", "renderer_md5": EXAM_MD5,
         "judge_md5_ok": jm_ok, "agree_100": agree100,
         "verdict": "PASS" if (agree100 and jm_ok and peq == 0) else "FAIL",
         "shards": [x["shard"] for x in shards]}
    part["gates"]["G-E2E"] = g
    vs = {k: part["gates"][k].get("verdict") for k in ("G-RT", "G-BW", "G-HD", "G-OPS",
                                                       "G-E2E")}
    part["all_gates_pass"] = all(v == "PASS" for v in vs.values())
    part["verdicts"] = vs
    part["merged_at"] = now_iso()
    B.jdump(part, a.gate)
    print(json.dumps({"verdicts": vs, "g_e2e": {k: g[k] for k in
                                                ("n_sample", "n_judged", "n_certified",
                                                 "certified_rate_of_judged", "agree_100",
                                                 "verdict")}}, ensure_ascii=False, indent=1),
          flush=True)
    print("GATE -> %s md5 %s" % (a.gate, md5f(a.gate)), flush=True)
    return part


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="build",
                    choices=("build", "side", "e2e", "merge"))
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-cache", default=os.path.join(CACHE, "GEN44_V7_ALL.bin"))
    ap.add_argument("--side-out", default=os.path.join(CACHE, "GEN44_V7_SIDECAR.jsonl"))
    ap.add_argument("--side-gate", default=os.path.join(OUT, "GEN44_S4B_SIDE.json"))
    ap.add_argument("--render-dump", default=os.path.join(OUT, "GEN44_S4B_RENDER.jsonl"))
    ap.add_argument("--partial", default=os.path.join(OUT, "GEN44_S4B_GATE.partial.json"))
    ap.add_argument("--gate", default=os.path.join(OUT, "GEN44_S4B_GATE.json"))
    ap.add_argument("--e2e-out", default=os.path.join(OUT, "GEN44_S4B_E2E.json"))
    ap.add_argument("--e2e-n", type=int, default=8, help="rows sampled per family (9 families x n)")
    ap.add_argument("--head-normalize", type=int, default=1,
                    help="normalize the judging head to the LWB trailing colon (D-S6B-4)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--max-lean-jobs", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--gate-wait-s", type=float, default=900.0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    if a.stage == "build":
        stage_build(a)
    elif a.stage == "side":
        stage_side(a)
    elif a.stage == "e2e":
        stage_e2e(a)
    else:
        stage_merge(a)


if __name__ == "__main__":
    main()
