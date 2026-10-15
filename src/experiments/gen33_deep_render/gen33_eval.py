# -*- coding: utf-8 -*-

# ---- gen33 fork registration (deep-generation V3.5 processing arm) ----
# This file = a **line-by-line fork copy** of gen30_eval.py (md5 e5b66c319fe646dfbb16ff25860323da); existing artifacts untouched (fork-copy regime).
# Inherits three parent fixes (**as-is, zero change**): ① LN_q deviation; ② CRIT-30A query gating fix;
#   ③ CRIT-30B vocabulary convention fix.
# Batch-unique differences (see scratch/ccp/gen33/V33_DESIGN.md pre-registration) patch list:
#   [1] paths: added G33; OUT/log/judge dirs point at gen33
#   [2] vocabulary convention: v4 loading (load_ctx4) + legal set includes the nine gen32 operators; allow/vocab default to v4
#   [3] **device-recompute convention** (change96): args_mode gains "device" -- each step's 8-dim state is
#       recomputed by gen33_device (gold chain state never read); chain walk and exam share one payload source
#   [4] WALK_BUDGET 7 -> 12 (longest deep chain 9 steps + 3; pre-registered constant, see design addendum)
#   [5] render-consistency keyword table gains the nine new operators (diagnostic column)
#   [6] imports gen33_hybrid / gen33_planattn (forks) instead of gen30_*
#   [7] default --dev cuda:1 (GPU0 user redline)
# Write surface: only scratch/ccp/gen33/** (gen30/gen26/gen13 artifacts read-only as inputs). Redlines as in the parent.


# ---- gen30 fork registration (processing arm) ----
# This file = a **line-by-line fork copy** of gen26_eval.py (md5 101d73f3a022c9897e367f9da460b5a7); existing artifacts untouched (fork-copy regime).
# The only difference = mounting gen30 plan-sequence self-attention (P1 multi-head execution-aware step-level attention +
#   P2 dynamic operator embedding + P4-lite plan summary), see scratch/ccp/gen30/PLAN_ATTN_DESIGN.md (pre-registered). Patch list:
#   [1] paths: added G30; OUT/log/judge dirs point at gen30
#   [2] import gen30_hybrid (fork) + AttnModel
#   [3] load_model: plan-attention aware (private seed/arch restored from ckpt args; plan_attn.* keys loaded strictly)
#   [4] render_greedy: allow parameter + P4-lite plan summary merged into the render-head input (same convention as training)
#   [5] walk_chain: per-head step-level attention export (evidence >=20 samples)
#   [6] CLI: --pa-enable/--pa-seed/--plan-summary/--attn-export/--attn-out
# Write surface: only scratch/ccp/gen30/** (gen26 artifacts read-only as inputs). Redlines as in the parent.

# gen26_eval.py -- gen26 (latent-space hybrid main arm) T2: exam/evaluation interface (chain trace + render + attribution + judging + three arms)
#
# Legal sources: constitution change 74 T1③ + change 49 (model routes only, device computes) + change 75 (redline R0)
#       + post-gen27-review correction order (HIGH-2/HIGH-4/HIGH-5/MED-7/MED-8 landed item by item in this file).
#
# Two-stage interface:
#   Stage 1 chain of thought (free chain walk): the model picks operators (step_head argmax, legal set = CCP 7 ∪ END); each step's
#       payload (res/pst frames) is supplied device-side; every frame is traced; **no judging**.
#       --args-mode bridge = exam convention (change-49 device argument-derivation loop; **default for formal evaluation**)
#       --args-mode gold   = diagnostic convention (gold payload; disclosed as a diagnostic column only, never a formal reading)
#   Stage 2 render: chain context (latent, not externalized) + render prompt -> LM head greedy render -> frozen judging chain.
#   --arm hybrid|naked|rule · --seg a_held|a_train|exam_v1|exam_zhenti
#   => same-exam same-gate comparison table, five columns: certified | first-operator hit | whole-chain hit | slot column | render consistency
#
# Reuse (imports-only): gen13 ccp_cachefull · gen13 v32ccp_train · ccp_bridge (device argument derivation) ·
#   gen23 bh_rule/bh_ops (rule arm) · frozen judge module drct_common + gen23_prescreen_batch (import only).
# Write surface: only scratch/ccp/gen26/**.
import argparse
import collections
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import sys
import time

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = os.path.join(RES, "scratch", "ccp")
G26 = os.path.join(CCP, "gen26")            # read-only input source
G13 = os.path.join(CCP, "gen13")
G30 = os.path.join(CCP, "gen30")            # parent (read-only; fork ledger)
G33 = os.path.join(CCP, "gen33")            # gen33 fork write surface
OUT = os.path.join(G33, "out")
os.makedirs(OUT, exist_ok=True)
for _p in (RES, CCP, G13, os.path.join(RES, "cppcuda", "tools"), os.path.join(RES, "skel32"),
           G26, G30, G33, os.path.join(CCP, "gen23"), os.path.join(RES, "scratch", "drct"),
           os.path.join(CCP, "bh", "gen"), os.path.join(CCP, "bh"),
           os.path.join(CCP, "gen24"), os.path.join(CCP, "gen32"),
           os.environ.get("SCOPE_SUPEROPERATOR", "../device")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                          # noqa: E402
import torch                                                                # noqa: E402

import gen33_hybrid as HY                                                   # noqa: E402
import gen33_planattn as PA33                                               # noqa: E402
import gen33_device as DEV                                                  # noqa: E402
AttnModel = PA33.make_attn_model(HY.HybridModel)
CC = HY.HC
L32 = HY.L32
NSLOT, END31 = HY.NSLOT, HY.END31
_ld = importlib.machinery.SourceFileLoader("g26_cachefull_eval",
                                           os.path.join(G13, "ccp_cachefull.py"))
CCF = importlib.util.module_from_spec(importlib.util.spec_from_loader(
    "g26_cachefull_eval", _ld))
_ld.exec_module(CCF)
VOC = json.load(open(os.environ.get("GEN33_VOCAB",
                                    os.path.join(G33, "cache", "CCP_VOCAB_V4.json")),
                     encoding="utf-8"))
# MED-7: value-space truth = CCP_VOCAB.n_vals_ext (238); the historical 239 in code is a dead value/comment leftover
NVALS = int(VOC["n_vals_ext"])
NHEAD = int(VOC["head"])


def _set_vocab(path):
    """A3 (official arm A): ccp-vocab/3 upgrade hook -- head 185 / n_vals 352 / 17 legal operators."""
    global VOC, NVALS, NHEAD
    VOC = json.load(open(path, encoding="utf-8"))
    NVALS = int(VOC["n_vals_ext"])
    NHEAD = int(VOC["head"])
    return VOC
def _load_ctx_eval(a):
    """gen30 [7]: pick the frame-construction context by vocab schema -- ccp-vocab/3 ⇒ gen26s_vocab3.load_ctx3(v3)
    (opmap includes bh_/g24_, value space 352); otherwise fall back to ccp_cachefull.load_ctx() (v2).
    Reason: the v2 ctx opmap lacks bh_/g24_ names ⇒ walking to those operators raises KeyError in ccp_cachefull.build,
    and slot values would be misaligned against the 238 space. Existing artifacts untouched (this fork only picks the
    loader; gen26s_vocab3 imported read-only)."""
    vp = getattr(a, "vocab", "") or ""
    try:
        _sch = json.load(open(vp, encoding="utf-8")).get("schema") if vp else ""
    except Exception:                                                # noqa: BLE001
        _sch = ""
    if _sch == "ccp-vocab/4":
        import importlib.machinery as _im4
        import importlib.util as _iu4
        _ld4 = _im4.SourceFileLoader("g33_vocab4", os.path.join(G33, "gen33s_vocab4.py"))
        _v4 = _iu4.module_from_spec(_iu4.spec_from_loader("g33_vocab4", _ld4))
        _ld4.exec_module(_v4)
        return _v4.load_ctx4(vp)
    need3 = (_sch == "ccp-vocab/3")
    if need3:
        import importlib.machinery as _im
        import importlib.util as _iu
        _ld = _im.SourceFileLoader("g30_vocab3", os.path.join(G26, "gen26s_vocab3.py"))
        _v3 = _iu.module_from_spec(_iu.spec_from_loader("g30_vocab3", _ld))
        _ld.exec_module(_v3)
        return _v3.load_ctx3(vp)
    return CCF.load_ctx()


K_EMB, K_WIN, K_OP, K_RES, K_PST, K_PSZ = 0, 4, 1, 2, 3, 5
# MED-8: free chain-walk bound = **pre-registered constant** (not per-row gold): 2×longest gold chain (3) + 1 = 7 decisions
# gen33 [4]: longest deep-chain gold = 9 steps ⇒ pre-registered constant = 9 + 3 = 12 (see V33_DESIGN_ADDENDUM §A1)
WALK_BUDGET = 12
CCP_OPS = CCF.V.CCP_OPS

CONSISTENCY_KEYWORDS = {
    "ccp_ind_recurrence": ["induction", "induct", "hstep", "ih"],
    "ccp_ring_identity": ["ring", "push_cast", "norm_num", "linarith", "omega"],
    "ccp_ineq_sos": ["nlinarith", "positivity", "sq_nonneg", "mul_self_nonneg", "sos",
                     "mul_nonneg"],
    "ccp_ineq_close": ["nlinarith", "linarith", "positivity", "sq_nonneg"],
    "ccp_nt_dvd_witness": ["dvd", "\u27e8", "obtain", "use ", "witness"],
    "ccp_nt_dvd_combine": ["dvd_trans", "mul_dvd", "dvd_mul", "dvd"],
    "ccp_nt_dvd_prod_combine": ["dvd_trans", "mul_dvd", "dvd_mul", "dvd"],
    # --- A3/v3: 17 new operators (bh 11 + g24 6); keywords registered per family semantics; hit rate is a diagnostic column only ---
    "bh_ind_sum_d1": ["induction", "hstep", "ih", "sum", "Finset"],
    "bh_ind_sum_d3": ["induction", "hstep", "ih", "sum", "Finset", "ring"],
    "bh_ind_geo": ["induction", "hstep", "ih", "geom", "pow", "mul_comm"],
    "bh_ind_close_sq2": ["ring", "push_cast", "linarith", "norm_num", "nlinarith"],
    "bh_ind_close_sq4": ["ring", "push_cast", "linarith", "norm_num", "nlinarith"],
    "bh_ind_close_const": ["ring", "norm_num", "linarith", "push_cast"],
    "bh_ineq_block": ["nlinarith", "positivity", "sq_nonneg", "mul_nonneg", "linarith"],
    "bh_ineq_cross": ["nlinarith", "positivity", "sq_nonneg", "mul_nonneg"],
    "bh_ineq_tail_zero": ["nlinarith", "positivity", "sq_nonneg", "linarith"],
    "bh_ineq_tail_square": ["nlinarith", "positivity", "sq_nonneg", "pow_nonneg"],
    "bh_ineq_tail_other": ["nlinarith", "positivity", "sq_nonneg", "linarith"],
    "g24_ind_ratio": ["induction", "hstep", "ih", "field_simp", "ring"],
    "g24_ind_ratio_close": ["field_simp", "ring", "push_cast", "linarith"],
    "g24_ind_pf": ["induction", "hstep", "ih", "field_simp", "ring"],
    "g24_ind_pf_close": ["field_simp", "ring", "push_cast", "linarith"],
    "g24_ineq_ht": ["nlinarith", "positivity", "sq_nonneg", "mul_nonneg"],
    "g24_ineq_ht2": ["nlinarith", "positivity", "sq_nonneg", "mul_nonneg"],
    # --- gen33/v4: nine gen32 deep operators (keywords per family semantics; hit rate is a diagnostic column only) ---
    "gen32_nest_solve": ["induction", "hstep", "ih", "ring", "norm_num", "solve"],
    "gen32_nest_fold": ["fold", "rw", "ring", "mul"],
    "gen32_nest_delta": ["delta", "shift", "ring", "telescop"],
    "gen32_nest_close": ["ring", "push_cast", "norm_num", "linarith", "close"],
    "gen32_sos_block": ["positivity", "sq_nonneg", "nlinarith", "pow_two", "mul_self"],
    "gen32_sos_cross": ["positivity", "sq_nonneg", "nlinarith", "mul_nonneg", "pow_two"],
    "gen32_sos_tail_zero": ["nlinarith", "positivity", "sq_nonneg", "linarith"],
    "gen32_sos_tail_square": ["nlinarith", "positivity", "sq_nonneg", "sq_nonneg", "pow"],
    "gen32_sos_tail_other": ["nlinarith", "positivity", "sq_nonneg", "linarith", "norm_num"],
}


def log(msg, path=None):
    line = "[%s] %s" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), msg)
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Mini cache (single-row prefix; frame construction still runs through gen13 ccp_cachefull.build, no reimplementation)
# ---------------------------------------------------------------------------

class MiniCache:
    def __init__(self, rec, t4set):
        R = HY.HC.RowC()
        R.ids = list(rec["ids"])
        R.t4 = int(rec["t4"])
        R.readout_only = int(rec["readout_only"])
        R.pos_task_end = int(rec["pos_task_end"])
        R.end_pos = int(rec["end_pos"])
        R.readout = float(rec["readout"])
        R.desc = [(int(k), int(v)) for k, v in rec["desc"]]
        R.resHas = [1 if x is not None else 0 for x in rec["res"]]
        R.res = rec["res"]
        R.labels = list(rec["labels"])
        R.feat_pos = list(rec["feat_pos"])
        R.bsl = list(rec["bsl"])
        R.state_pos = list(rec["state_pos"])
        R.src = "walk"
        self.rows = [R]
        self.t4 = [tuple(float(x) for x in t) for t in t4set]
        self.nT4 = len(self.t4)
        self.n_rows = 1
        self.HEAD = NHEAD
        self.NVALS = NVALS
        self.ENDIDX = END31
        self.NSLOT = NSLOT
        self.segidx = {"segments": {"walk": [0, 1]}, "held_rows": [], "n_held": 0}
        self.held = set()

    def seg_of(self, i):
        return "walk"


def _dev_row_of(side_rec):
    """Sidecar -> synthetic row for device recomputation (feeds only args/ops plus the vars/dom_dump family-C recompute needs)."""
    ops = list(side_rec.get("ops") or [])
    sfa = list(side_rec.get("sfa") or [])
    steps = [{"op": op, "args": (dict(sfa[k]) if k < len(sfa) and sfa[k] else {})}
             for k, op in enumerate(ops)]
    meta = {}
    if side_rec.get("vars"):
        meta["vars"] = list(side_rec["vars"])
    if side_rec.get("dom_dump") is not None:
        meta["dom_dump"] = side_rec["dom_dump"]
    return {"name": side_rec.get("name"), "steps": steps, "meta": meta}


def _payload_of(side_rec):
    """gen33 [3] **device-recompute convention** (change96): each step's 8-dim state is recomputed by gen33_device and
    in-row/sidecar gold res is **never read** (gold res serves only the change-97 gate checks, never the exam path).
    Even when the sidecar carries a non-empty res and recompute was not explicitly requested (diagnostic convention),
    recompute still wins -- single state source."""
    row = _dev_row_of(side_rec)
    res, errs = [], []
    for k in range(len(row["steps"])):
        r8, _ex, _sp, _op, err = DEV.dev_resid(row, k)
        if err:
            res.append(None)
            errs.append({"k": k, "op": row["steps"][k]["op"], "err": err})
        else:
            res.append(r8)
    return {"res": res, "sfa": list(side_rec.get("sfa") or []),
            "dev_errs": errs,
            "res_src": "device_live(live_state)"}


def _mk_src_row(side_rec, ops, payload):
    steps, sfa = [], []
    for k, op in enumerate(ops):
        res = payload["res"][k] if k < len(payload["res"]) else None
        a = payload["sfa"][k] if k < len(payload["sfa"]) else None
        steps.append({"op": op, "ok": True, "err": None, "res": res, "state": "ok"})
        sfa.append(dict(a) if isinstance(a, dict) else {})
    return {"stmt": side_rec["stmt"], "type4": list(side_rec.get("type4") or [1, 0, 0, 0]),
            "program": list(ops), "steps": steps, "sf_args": sfa, "episode": None,
            "accepted": True, "readout_only": False}


def _prefix(ctx, side_rec, ops, payload):
    recs, t4set, _st = CCF.build(ctx, [_mk_src_row(side_rec, ops, payload)])
    return MiniCache(recs[0], t4set)


def _name_space():
    """Registered operator name space (ccp 7 + bh 11 + g24 6 + END); the scorable criterion."""
    return set(_legal_ops()[1].values()) | {"v31_end_marker"}


def _legal_ops():
    names = {int(v): k for k, v in VOC["ops_ccp"].items()}
    for key in ("ops_bh", "ops_g24", "ops_gen32"):   # A3 (v3) 17 + gen33 (v4) 9
        for k, v in (VOC.get(key) or {}).items():
            names[int(v)] = k
    return sorted(names), names


def _bridge_args(cls, stmt_state):
    """Device argument derivation (change-49 exam convention): ccp_bridge first, then bh_bridge on failure (design §4 table); else None.

    Registered (HIGH-2/MED-8/A3): the bridges solve only **statement-derivable** operators; later-operator slots come from the
    **device state**, and this batch uses gold args as the device-side proxy (proxy source recorded per frame in trace.args_src).
    bh branch form criteria per design §4: contains "f (k + 1) = f k + (" -> bridge_ind_sum; contains "* f k" -> bridge_ind_geo;
    contains ≥/≤ -> bridge_ineq. The g24 branch is **not wired this batch** (it needs params/objs reverse lookup via
    g24_families.make_problem; the exam main criterion set is all bh, so this is not a blocker) -- registered explicitly, not silently.
    """
    import re
    # --- ccp bridge (existing v2 behavior) ---
    try:
        import ccp_bridge as BR
        stmt = "theorem _g26_ " + stmt_state.replace("\u22a2", ":")
        if cls == "ind":
            return BR.bridge_ind(stmt)
        if cls == "nt":
            return BR.bridge_nt(stmt)
        if cls == "ineq":
            header = stmt_state.split("\u22a2")[0]
            grp = re.findall(r"\(([^()]*?)\s*:\s*\u2115\)", header)
            vars_ = grp[-1].split() if grp else []
            if vars_:
                return BR.bridge_ineq(stmt, vars_)
    except Exception as e:                                        # noqa: BLE001
        _BRIDGE_ERR.append("ccp:" + repr(e))
    # --- bh bridge (added in A3; design §4) ---
    try:
        import bh_bridge as BB
        stmt = "theorem _g26_ " + stmt_state.replace("\u22a2", ":")
        if "f (k + 1) = f k + (" in stmt_state or "f (k+1) = f k + (" in stmt_state:
            return BB.bridge_ind_sum(stmt)
        if "* f k" in stmt_state:
            return BB.bridge_ind_geo(stmt)
        if ("\u2265" in stmt_state) or ("\u2264" in stmt_state):
            header = stmt_state.split("\u22a2")[0]
            grp = re.findall(r"\(([^()]*?)\s*:\s*[\u2115\u2124\u211a]\)", header)
            vars_ = grp[-1].split() if grp else []
            if vars_:
                return BB.bridge_ineq(stmt, vars_)
    except Exception as e:                                        # noqa: BLE001
        _BRIDGE_ERR.append("bh:" + repr(e))
    return None


_BRIDGE_ERR = []            # device argument-derivation failures (first 5; disclosed in the report)


# ---------------------------------------------------------------------------
# Stage 1: free chain walk
# ---------------------------------------------------------------------------

@torch.no_grad()
def walk_chain(model, ctx, side_rec, allow, dev, args_mode="bridge", budget=WALK_BUDGET,
               logp=None, attn_log=None):
    model.eval()
    gold_ops = list(side_rec.get("ops") or [])
    payload = _payload_of(side_rec)
    ops, trace, bridge_fb, n_src = [], [], 0, collections.Counter()
    legal_idx, names = _legal_ops()
    for k in range(budget):
        cache = _prefix(ctx, side_rec, ops, payload)
        bc = L32.BatchCpu32().build(cache, [0], 0, 0, 0)
        bc.rowIds = [0]
        cm, cosc, sinc = model.consts(bc.L, dev)
        mk = model.mask_of(cm, bc.B, bc.L, bc.lenB, dev)
        seq = model.assemble(bc.B, bc.L, bc, cache, dev)
        H = model.forward_body(seq, mk, cosc, sinc)
        lg, _zb, _tgt = model.heads(H, bc, cache, dev, allow)
        if attn_log is not None:                 # gen30 [5]: step-level attention evidence (per head)
            try:
                _snap = model.attn_snapshot(bc, cache, op_names=names)
                if _snap:
                    _snap[0]["k"] = k
                    attn_log.append(_snap[0])
            except Exception as _e:                                  # noqa: BLE001
                attn_log.append({"k": k, "err": repr(_e)})
        T = bc.Talloc
        z = lg.view(1, T, lg.shape[-1])[0][min(k, T - 1)].clone()
        mask = torch.full_like(z, float("-inf"))
        mask[legal_idx] = 0.0
        mask[END31] = 0.0
        z = z + mask
        pred = int(z.argmax())
        p = torch.softmax(z, -1)
        top2 = torch.topk(p, 2)
        gold_k = gold_ops[k] if k < len(gold_ops) else None
        rec = {"k": k, "pred_idx": pred,
               "pred": "v31_end_marker" if pred == END31 else names.get(pred, "?"),
               "gold": gold_k, "gold_in_space": bool(gold_k in set(CCP_OPS)),
               "hit": bool(gold_k is not None and pred != END31
                           and names.get(pred) == gold_k),
               "p_max": round(float(top2.values[0]), 5),
               "margin": round(float(top2.values[0] - top2.values[1]), 5)}
        if pred == END31:
            rec["stop"] = "END"
            trace.append(rec)
            break
        op_name = names.get(pred)
        asrc = "device_state_gold_proxy"
        if args_mode == "bridge":
            ba = _bridge_args(side_rec["cls"], side_rec["stmt"]) if k == 0 else None
            if ba is not None and k < len(payload["sfa"]):
                payload["sfa"][k] = ba
                asrc = "bridge"
            elif k == 0:
                bridge_fb += 1
        rec["args_src"] = asrc
        n_src[asrc] += 1
        ops.append(op_name)
        trace.append(rec)
        del H, seq, lg, z
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    model.train()
    cmp_n = min(len(ops), len(gold_ops))
    return {"trace": trace, "ops_walked": ops, "gold_ops": gold_ops,
            "first_hit": bool(trace and trace[0].get("hit")),
            "exact_hit": bool(ops == gold_ops),
            "prefix_hit": bool(cmp_n and ops[:cmp_n] == gold_ops[:cmp_n]),
            "overrun": bool(len(ops) > len(gold_ops)),
            "terminated_end": bool(trace and trace[-1].get("stop") == "END"),
            "bridge_fallback": bridge_fb, "args_src_hist": dict(n_src),
            "scorable": bool(gold_ops) and all(o in _name_space() for o in gold_ops),
            "budget": budget}


# ---------------------------------------------------------------------------
# Stage 2: rendering (hybrid / naked) and the rule arm
# ---------------------------------------------------------------------------

@torch.no_grad()
def render_greedy(model, ctx, side_rec, ops, payload, dev, max_new=160, stop_seq="```",
                  allow=None):
    import v31_model as vm
    M_TASK = vm.st.M_TASK
    model.eval()
    prompt_ids = side_rec["ids"][1:side_rec["np"]]
    cache = _prefix(ctx, side_rec, ops, payload)
    base = [tuple(x) for x in cache.rows[0].desc] + [(K_EMB, int(M_TASK))]
    base += [(K_EMB, int(t)) for t in prompt_ids]
    gen = []
    for _ in range(max_new):
        R = cache.rows[0]
        R.desc = base + [(K_EMB, int(t)) for t in gen]
        R.readout_only = 1
        R.pos_task_end = int(min(R.pos_task_end, len(R.desc) - 1))
        bc = L32.BatchCpu32().build(cache, [0], 0, 0, 0)
        bc.rowIds = [0]
        cm, cosc, sinc = model.consts(bc.L, dev)
        mk = model.mask_of(cm, bc.B, bc.L, bc.lenB, dev)
        seq = model.assemble(bc.B, bc.L, bc, cache, dev)
        H = model.forward_body(seq, mk, cosc, sinc)
        hin = H[:, -1, :]
        if getattr(model, "plan_summary", False) and hasattr(model, "plan_summary_from_H"):
            _summ = model.plan_summary_from_H(H[:, :bc.L, :].contiguous(), bc, cache, dev)
            if _summ is not None:                # gen30 [4]: P4-lite merged into the render head (same convention as training)
                hin = hin + _summ.to(hin.dtype)
        z = model.lm.lm_head(hin)
        nxt = int(z.argmax(-1))
        gen.append(nxt)
        txt = _ids_to_text(ctx["tok"], gen)
        if stop_seq and txt.rstrip().endswith(stop_seq) and len(gen) > 8:
            break
        if ctx["tok"].eos_token_id is not None and nxt == ctx["tok"].eos_token_id:
            break
    model.train()
    return _ids_to_text(ctx["tok"], gen), len(gen)


@torch.no_grad()
def render_naked(tok, side_rec, dev, max_new=200, model_name="135M"):
    """Naked base arm: zero-training base direct generation (same prompt, greedy; change-75 control convention)."""
    from transformers import AutoModelForCausalLM
    path = os.path.join(RES, "models", "SmolLM2-%s" % model_name)
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(dev)
    m.eval()
    p = side_rec["ids"][1:side_rec["np"]]
    ids = torch.tensor([p], dtype=torch.long, device=dev)
    out = m.generate(ids, max_new_tokens=max_new, do_sample=False,
                     pad_token_id=tok.eos_token_id)
    txt = tok.decode(out[0, len(p):], skip_special_tokens=True)
    del m
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return txt, int(out.shape[1] - len(p))


def rule_arm_text(side_rec, logp=None):
    """Rule arm (device auto-solving; zero training): gen23 bh_rule program -> device execution returns the tactic line."""
    try:
        import gen23_exam as EX
        chains = EX.load_chains(side_rec["cls"])
        c = chains.get(side_rec["name"])
        if not c:
            return "", "no_chain"
        pr = c.get("pr") or c.get("problem") or c
        ok, why, prog, txt, n = EX.arm_text(side_rec["cls"], pr, c.get("arm") or "RBplay")
        return txt, ("ok" if ok else "rule_fail:%s" % why)
    except Exception as e:                                       # noqa: BLE001
        return "", "rule_unavailable:%r" % (e,)


def _ids_to_text(tok, ids):
    try:
        return tok.decode(ids, skip_special_tokens=True)
    except Exception:                                            # noqa: BLE001
        return ""


def _extract_body(text):
    if "```" not in (text or ""):
        return (text or "").strip()
    parts = text.split("```")
    body = parts[1] if len(parts) > 1 else parts[0]
    lines = body.split("\n")
    if lines and lines[0].strip().lower() in ("lean4", "lean"):
        lines = lines[1:]
    return "\n".join(lines).strip()


def consistency_row(ops, text):
    """Consistency column (diagnostic): only operators with **registered keywords** count in the denominator; unregistered operators are tallied separately as n_noreg (no false zeros)."""
    lo = (text or "").lower()
    hits, nreg = [], 0
    for op in ops:
        kws = CONSISTENCY_KEYWORDS.get(op)
        if not kws:
            nreg += 1
            continue
        hits.append({"op": op, "hit": bool(any(k.lower() in lo for k in kws)),
                     "keywords": kws})
    n = len(hits)
    return {"n": n, "n_noreg": nreg, "hit": sum(1 for x in hits if x["hit"]),
            "rate": (sum(1 for x in hits if x["hit"]) / n) if n else None,
            "per_op": hits}


# ---------------------------------------------------------------------------
# Judging (frozen judging chain; import only, zero modification; off by default)
# ---------------------------------------------------------------------------

def judge_texts(items, wd, chunk=60, tag="gen26"):
    # Concurrency discipline: Lean under nice -n 19 ∧ concurrency ≤5 ⇒ this module **serializes** all Lean calls
    # (1 job), and the caller process already runs under nice -n 19 (children inherit it). Frozen artifacts are
    # untouchable: the lock is wrapped temporarily inside this function only.
    import threading
    import drct_common as DC
    import gen23_prescreen_batch as PBATCH
    jm = md5f(DC.__file__)
    _lock = threading.Lock()
    _orig = PBATCH.run_lean_full

    def _serialized(*a2, **k2):
        with _lock:
            return _orig(*a2, **k2)
    PBATCH.run_lean_full = _serialized
    try:
        out = PBATCH.judge_items_batch(items, wd, chunk=chunk, tag=tag, judge_md5=jm)
    finally:
        PBATCH.run_lean_full = _orig
    return {"judge_md5": jm, "n": len(items), "serialized": True,
            "certified": sum(1 for x in out if x.get("certified")), "raw": out}


# ---------------------------------------------------------------------------
# Unified evaluation: three arms × five columns
# ---------------------------------------------------------------------------

def run_eval(model, side, allow, items, dev, ctx, a, logp):
    rows = []
    attn_recs = []
    for ri, s in items:
        payload = _payload_of(s)
        ops, txt, ntok, w, err = [], "", 0, None, None
        if a.arm == "hybrid":
            _n_exp = int(getattr(a, "attn_export", 0) or 0)
            _alog = [] if (_n_exp and len(rows) < _n_exp) else None
            w = walk_chain(model, ctx, s, allow, dev, args_mode=a.args_mode, logp=logp,
                           attn_log=_alog)
            if _alog:
                attn_recs.append({"ri": int(ri), "name": s["name"], "cls": s["cls"],
                                  "gold_ops": (s.get("ops") or []), "steps": _alog})
            ops = w["ops_walked"]
            try:
                txt, ntok = render_greedy(model, ctx, s, ops, payload, dev,
                                          max_new=a.max_new, allow=allow)
            except Exception as e:                               # noqa: BLE001
                err = repr(e)
        elif a.arm == "naked":
            try:
                txt, ntok = render_naked(ctx["tok"], s, dev, max_new=a.max_new,
                                         model_name=a.naked_base)
            except Exception as e:                               # noqa: BLE001
                err = repr(e)
        elif a.arm == "rule":
            txt, why = rule_arm_text(s, logp)
            ntok = len(txt)
            err = None if why == "ok" else why
        cons = consistency_row(ops, txt)
        rows.append({"ri": int(ri), "name": s["name"], "cls": s["cls"], "seg": s["seg"],
                     "arm": a.arm, "args_mode": a.args_mode,
                     "chain": None if w is None else {
                         "walked": w["ops_walked"], "gold": w["gold_ops"],
                         "first_hit": w["first_hit"], "exact_hit": w["exact_hit"],
                         "prefix_hit": w["prefix_hit"], "overrun": w["overrun"],
                         "terminated_end": w["terminated_end"],
                         "scorable": w["scorable"], "budget": w["budget"],
                         "args_src_hist": w["args_src_hist"],
                         "bridge_fallback": w["bridge_fallback"], "trace": w["trace"]},
                     "render": {"text": txt, "n_tok": ntok, "body": _extract_body(txt)},
                     "consistency": cons, "err": err})
        log("[%s/%s/%s] %s chain=%s gold=%s first_hit=%s exact=%s render=%d cons=%s"
            % (a.arm, a.seg, a.args_mode, s["name"], ops, (s.get("ops") or []),
               None if w is None else w["first_hit"], None if w is None else w["exact_hit"],
               ntok, cons["rate"]), logp)
    if attn_recs:
        _ap = getattr(a, "attn_out", "") or os.path.join(
            OUT, "GEN33_ATTN_SAMPLES%s.jsonl" % ("" if a.seg == "exam_v1" else "_%s" % a.seg))
        os.makedirs(os.path.dirname(_ap), exist_ok=True)
        with open(_ap, "w", encoding="utf-8") as f:
            f.write(json.dumps({"schema": "ccp-a-planseq-attn/1", "arm": a.arm,
                                "seg": a.seg, "args_mode": a.args_mode,
                                "n_rows": len(attn_recs),
                                "note": "逐头步级注意力权重 (步骤关系图: 每步回看哪些前步、"
                                        "权重多少); w_prefix_head0=状态依赖头, "
                                        "w_prefix_head1=算子组合相容性头"}) + "\n")
            for r in attn_recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log("gen30 attention evidence file: %s (%d samples)" % (_ap, len(attn_recs)), logp)
    tbl = table_of(rows)
    if a.judge:
        jt = [{"name": r["name"], "head": side[r["ri"]]["head"],
               "concl": side[r["ri"]]["concl"], "proof_text": r["render"]["body"]}
              for r in rows]
        jr = judge_texts(jt, os.path.join(G33, "lean", "judge_%s_%s" % (a.arm, a.seg)),
                         tag="gen33_%s_%s" % (a.arm, a.seg))
        by = {x["name"]: x for x in jr["raw"]}
        for r in rows:
            x = by.get(r["name"]) or {}
            r["judge"] = {"certified": bool(x.get("certified")),
                          "reasons": x.get("reasons")}
        tbl["certified"] = sum(1 for r in rows if (r.get("judge") or {}).get("certified"))
        tbl["judge_md5"] = jr["judge_md5"]
    return rows, tbl


def table_of(rows):
    """Five columns: certified | first-operator hit | whole-chain hit | slot column | render consistency."""
    n = len(rows)
    ch = [r for r in rows if r.get("chain")]
    sc = [r for r in ch if r["chain"].get("scorable")]
    return {"n": n, "n_chain_rows": len(ch), "n_chain_scored": len(sc), "certified": None,
            "first_hit": (sum(1 for r in sc if r["chain"]["first_hit"]) / len(sc))
                          if sc else None,
            "exact_hit": (sum(1 for r in sc if r["chain"]["exact_hit"]) / len(sc))
                          if sc else None,
            "slot_col": _slot_col(sc),
            "consistency": (sum(r["consistency"]["rate"] or 0 for r in rows) / n)
                            if n else None,
            "op_out_of_space": sum(1 for r in ch if not r["chain"].get("scorable")),
            "terminated_end": sum(1 for r in ch if r["chain"].get("terminated_end")),
            "render_tok_mean": round((sum(r["render"]["n_tok"] for r in rows) / n), 1)
                                if n else 0,
            "n_err": sum(1 for r in rows if r.get("err"))}


def _slot_col(sc):
    """Slot column (proxy convention, registered): share of args_src=bridge frames among frames whose walked operator matches gold.
    True slot accuracy (slot top-1 vs gold bsl) comes from eval_hybrid on the train/held-out segments (gen13 convention);
    on the exam segment the operator spaces are disjoint -> the slot column is unscoreable; returns None and the report discloses it (see op_space_coverage)."""
    tot = sum(sum(r["chain"].get("args_src_hist", {}).values()) for r in sc)
    br = sum(r["chain"].get("args_src_hist", {}).get("bridge", 0) for r in sc)
    return (br / tot) if tot else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_model(a, parents, dev):
    """gen30 fork: plan-attention-aware loading. Private seed/architecture are restored from the ckpt args (CLI values if absent);
    plan_attn.*/plan_summary* keys load **strictly** (missing keys are reported, never silent)."""
    sd = None
    if a.ckpt and str(a.ckpt).lower() not in ("", "none", "-"):
        sd = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    cargs = dict((sd or {}).get("args") or {})
    pa_on = int(getattr(a, "pa_enable", 1)) and (sd is None or bool(cargs.get("pa_enable", 1)))
    mcls = AttnModel if pa_on else HY.HybridModel
    model = mcls(lat_tab_path=a.lat31, head=NHEAD, nvals=NVALS, parents=parents)
    info = {"path": a.ckpt or None}
    if sd is not None:
        info.update({"md5": md5f(a.ckpt), "step": sd.get("step")})
    if sd is None:
        model.load_init_ccp(a.init)
        info.update({"init": a.init,
                     "note": "无 ckpt: V32 起点 (未训练基线, 供对照披露)"})
    if pa_on:
        seed = int(cargs.get("pa_seed", -1))
        seed = seed if seed >= 0 else int(getattr(a, "pa_seed", 20260923))
        arch = {"use_state": bool(cargs.get("pa_state", getattr(a, "pa_state", 1))),
                "n_heads": int(cargs.get("pa_heads", getattr(a, "pa_heads", 2))),
                "d_head": int(cargs.get("pa_dhead", getattr(a, "pa_dhead", 64)))}
        ps = int(cargs.get("plan_summary", getattr(a, "plan_summary", 1)))
        if int(getattr(a, "plan_summary", -1)) in (0, 1):
            ps = int(a.plan_summary)                      # gen30 [8]: -1 = keep ckpt, 0/1 = force
        model.attach_plan_attn(seed=seed, capture=bool(getattr(a, "pa_capture", 0)),
                               enable=True, arch=arch, plan_summary=True)
        model.set_plan_summary(bool(ps))                  # two states: same parameter set, only the gate differs
        info["plan_attn"] = model.pa_param_report()
        info["plan_summary_effective"] = bool(ps)
    if sd is not None:
        missing, unexpected = model.load_state_dict(sd["model"], strict=False)
        pa_missing = [k for k in missing if k.startswith(("plan_attn", "plan_summary"))]
        info.update({"missing": len(missing), "unexpected": len(unexpected),
                     "pa_missing": pa_missing[:8]})
        if pa_on:
            assert not pa_missing and not unexpected, (pa_missing[:8], list(unexpected)[:8])
    return model.to(dev), info


def load_items(a):
    off = max(0, int(getattr(a, "chunk_offset", 0)))
    if a.seg in ("a_held", "a_train"):
        C = HY.CacheCCP(a.cache)
        side = HY.load_sidecar(a.side)
        lo, hi = C.segidx["segments"][a.seg]
        ii = list(range(lo, hi))[off:off + a.n]
        return [(i, side[i]) for i in ii]
    ex = HY.load_sidecar(a.exam_side)
    want = ("exam_main" if a.seg == "exam_v1"
            else ("exam_deep" if a.seg == "exam_deep" else "exam_zhenti"))
    ii = [i for i in sorted(ex) if ex[i]["seg"] == want][off:off + a.n]
    return [(i, ex[i]) for i in ii]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="hybrid", choices=["hybrid", "naked", "rule"])
    ap.add_argument("--seg", default="a_held",
                    choices=["a_held", "a_train", "exam_v1", "exam_zhenti", "exam_deep"])
    ap.add_argument("--args-mode", default="device",
                    choices=["device", "bridge", "gold"],
                    help="gen33 [3]: device = device-recomputed state (official convention); bridge/gold are diagnostic controls")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--init", default="ckpt/sla_v31f.pt")
    ap.add_argument("--cache", default=os.path.join(G33, "cache", "deep_all.bin"))
    ap.add_argument("--side", default=os.path.join(OUT, "DEEP_RENDER_SIDECAR.jsonl"))
    ap.add_argument("--exam-side", default=os.path.join(OUT, "DEEP_EXAM_SIDECAR.jsonl"))
    ap.add_argument("--allow", default=os.path.join(G33, "cache", "ccp_allow_v4.bin"))
    ap.add_argument("--lat31", default="cppcuda/cache/lat_tab_v31.bin")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--chunk-offset", type=int, default=0)
    ap.add_argument("--vocab", default=os.path.join(G33, "cache", "CCP_VOCAB_V4.json"),
                    help="gen33: ccp-vocab/4 (head 194 / value space 352)")
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--judge", type=int, default=0)
    ap.add_argument("--naked-base", default="135M")
    ap.add_argument("--dev", default="cuda:1")
    ap.add_argument("--out", default="")
    ap.add_argument("--log", default=os.path.join(G33, "logs", "gen33_eval.log"))
    ap.add_argument("--pa-enable", type=int, default=1)
    ap.add_argument("--pa-seed", type=int, default=20260923)
    ap.add_argument("--pa-state", type=int, default=1)
    ap.add_argument("--pa-heads", type=int, default=2)
    ap.add_argument("--pa-dhead", type=int, default=64)
    ap.add_argument("--plan-summary", type=int, default=-1,
                    help="gen30: -1 = keep ckpt args; 0/1 = force this run (two-state reading)")
    ap.add_argument("--pa-capture", type=int, default=1)
    ap.add_argument("--attn-export", type=int, default=0,
                    help="gen30: export the first N step-level attention samples (>=20 = paper evidence)")
    ap.add_argument("--attn-out", default="")
    a = ap.parse_args()
    if str(a.dev).startswith("cuda"):
        HY.gpu_gate(a.dev, a.log)
    dev = torch.device(a.dev)
    if a.vocab:
        _set_vocab(a.vocab)
        log("A3 vocab: %s head=%d n_vals=%d legal_ops=%d" % (a.vocab, NHEAD, NVALS,
                                                             len(_legal_ops()[0])), a.log)
    parents = {int(k): v["parent_idx"] for k, v in VOC["init_map"].items()}
    allow = torch.from_numpy(np.fromfile(a.allow, dtype=np.uint8).reshape(
        NHEAD, NSLOT, NVALS).copy())
    items = load_items(a)
    log("== eval arm=%s seg=%s args_mode=%s n=%d judge=%s =="
        % (a.arm, a.seg, a.args_mode, len(items), a.judge), a.log)
    t0 = time.time()
    if a.arm == "hybrid":
        ctx = _load_ctx_eval(a)
        model, minfo = load_model(a, parents, dev)
    else:
        ctx = _load_ctx_eval(a)
        model, minfo = None, {"note": "非 hybrid 臂 (naked=零训练底座 / rule=装置规则臂)"}
    rows, tbl = run_eval(model, {i: s for i, s in items}, allow, items, dev, ctx, a, a.log)
    res = {"schema": "ccp-a-hybrid-eval/2",
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "arm": a.arm, "seg": a.seg, "args_mode": a.args_mode, "n": len(rows),
           "table": tbl, "model": minfo,
           "columns": ["certified", "first_hit", "exact_hit", "slot_col", "consistency"],
           "inputs": {"cache": a.cache, "side": a.side, "exam_side": a.exam_side,
                      "ckpt": a.ckpt, "judge": bool(a.judge)},
           "bridge_errors": _BRIDGE_ERR[:5],
           "rows": rows, "wall_s": round(time.time() - t0, 2)}
    outp = a.out or os.path.join(OUT, "GEN33_EVAL_%s_%s%s.json"
                                 % (a.arm, a.seg, "" if a.args_mode == "device"
                                    else "_%s" % a.args_mode))
    json.dump(res, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("TABLE %s -> %s" % (json.dumps(tbl, ensure_ascii=False), outp), a.log)
    print("GEN26_EVAL_RC=0")


if __name__ == "__main__":
    main()
