# -*- coding: utf-8 -*-

# ---- gen33 fork registration (deep generation V3.5 treatment arm) ----
# This file is a **line-by-line fork copy** of gen30_planattn.py
# (md5 0da0a2dc9d5fd29cab6d587b196bdd4f); pre-existing files untouched (fork-copy rule).
# All three parent fixes are inherited (**as-is, zero edits**): (1) LN_q deviation;
#   (2) CRIT-30A query gating fix; (3) CRIT-30B vocab convention fix.
# This batch's difference = **none** (pure copy + registration header): this file holds no
# path constants and no vocab constants, and is logically identical to the parent.
# Write surface: scratch/ccp/gen33/**.

# gen30_planattn.py -- plan-sequence self-attention module (P1 execution-aware step-level
#                     multi-head attention + P2 dynamic op embedding + P4-lite plan summary)
#
# Design doc: scratch/ccp/gen30/PLAN_ATTN_DESIGN.md (pre-registered, before any training
#   reading from this file)
# Position: the treatment arm's only differing module. Control arm = gen26 formal-A
#   hybrid_formal_f935e143.
# North star (user's framing): "the low-dimensional latent loop = a vectorized operator RNN,
#   and the solution-step sequence is that RNN's task objective; gen30 upgrades it to a
#   **vectorized operator Transformer** (adding the relation axis of step-level
#   self-attention on the loop skeleton)". Upgrade = adding an axis, not replacing the body:
#   the discrete op skeleton / allow gate / WALK_BUDGET / device grounding / auditability /
#   chain-supervision objective all stay unchanged.
#
# Insertion point (registered position by position): inside HybridModel.heads(), after
#   x = [H[idxH], H[idxS]] and before step_head, one line injects x <- x + delta; with the
#   switch pa_enabled=False the baseline path runs unchanged (not a delta=0 approximation).
#   Fusion point = residual: x_fused = x + W_out.LN_A(o)  (a post-fusion LN would break
#   bitwise equivalence of the delta=0 start => the normalization slots moved to query-side
#   LN_q + readout-side LN_A; registered deviation, see design doc §2)
#   - query    : LN_q(x_t) -> W_q^h                          (2*HID=1152 -> 2 heads x 64)
#   - key/value: previous-step triples (op, args, device_state) embedded;
#       **head split (>= 2 heads)**:
#       * head 0 = **state-dependence head**: full triple e=[op~ ; args ; dev] + device-only
#         projections (W_kd/W_vd)
#       * head 1 = **op-composition compatibility head**: op~ only (op x op compatibility,
#         decoupled from state needs)
#       * P2: op~_j = lat_tab[op_j] * (1 + tanh(W_g.LN_q(x_j)))   (g=0 at the start =>
#         bitwise equal to the static row)
#       * args: R.bsl[j*24:(j+1)*24] -> E_arg(352+1) -> valid-slot mean pooling -> W_ap (32)
#       * device: R.res[res_idx_j] (8-d genuinely computed device res frame) -> MLP -> 64-d
#   - step-position bias (an RNN->Transformer standard item): 6 buckets of **relative step
#     distance** per head (b=0..3 for d=1..4, b=4 covers d in [5,7], b=5 covers d >= 8);
#     **zero-initialized** => the start equals no bias, and the recency prior is learned
#     (post-training per-bucket values enter the report = an interpretable answer to
#     "recency-weighted?")
#   - causality: strictly j < t; block-diagonal across rows; a query without a legal key
#     outputs an all-zero row
#   - output: W_out (zero-initialized) projects back to 2*HID; delta is exactly 0 at the
#     start (bitwise), see design doc §3
#   - P4-lite (switch --plan-summary, on by default): last-step readout o_T -> W_ps -> render
#     head input (zero-parameter start = 0)
#
# Hard constraints: the chain stays an auditable symbolic op sequence (attention only changes
#         logits, never the sequence representation; weights can be exported as a step
#         relation graph, per head) · no retrieval memory (the only external input = lat_tab
#         parameters + the current sequence frames) · zero exam-side contact surface
# In-suite ablation (pre-registered): use_state=True (P1-full) vs False (P1-nostate, the
#   state-dependence head drops the device channel)
# Corpus prior fact (measured in gen30, registered): the 16,583 device res steps of the V2F
#   cache are **all zero** => that channel carries no step-varying information in this corpus;
#   the module keeps the full implementation as instructed, and the ablation is expected to
#   read "P1-full ≈ P1-nostate", reported as is (see design doc §0.1)
# RNG isolation: all new parameters use a private torch.Generator, with the global RNG state
#   saved/restored around construction => baseline non-new parameters stay bitwise unchanged
#   under the same seed (PA-EQ1)
# Load order: the PA module **must** be attached after model.load_init_ccp() (v32lib.load_init
#   asserts on unknown state_dict keys; see attach_plan_attn)
import sys

import torch
from torch import nn
import torch.nn.functional as F

try:                                                     # frame kind constants (skel32/v32lib)
    import v31_model as vm
    K_OP, K_RES, K_PSZ = vm.K_OP, vm.K_RES, vm.K_PSZ
    NSLOT_DEF = int(vm.NSLOT)
    NEG_BIG = float(vm.NEG_BIG)
except Exception:                                        # noqa: BLE001 (CPU unit tests may degrade)
    K_OP, K_RES, K_PSZ = 1, 2, 3
    NSLOT_DEF, NEG_BIG = 24, float(torch.finfo(torch.float32).min)

END31_DEF = 154
NEG_INF = float("-inf")
ARG_UNK_ROW = 0                                          # arg-embedding row 0 = no-args placeholder
N_BUCKETS = 6                                            # relative-distance buckets: 1..4 / 5..7 / >=8
PA_SCHEMA = "ccp-a-planseq-attn/3"


def count_params(mod):
    return sum(p.numel() for p in mod.parameters()) if mod is not None else 0


def dist_bucket(d):
    """Relative step distance d (= t-j >= 1) -> bucket number (0..5)"""
    if d <= 4:
        return d - 1
    return 4 if d <= 7 else 5


class PlanStepAttention(nn.Module):
    """P1+P2: step-level multi-head execution-aware attention (2 heads x 64 dims, strictly
    causal, step-position bias, zero-initialized output)."""

    def __init__(self, d2=1152, head=185, nvals=352, nslot=NSLOT_DEF, n_heads=2, d_head=64,
                 d_g=64, d_arg=32, d_dev=64, use_state=True, seed=20260923):
        super().__init__()
        self.d2, self.head, self.nvals, self.nslot = d2, head, nvals, nslot
        self.n_heads, self.d_head, self.d_a = int(n_heads), int(d_head), int(n_heads * d_head)
        self.d_g, self.d_arg, self.d_dev = d_g, d_arg, d_dev
        self.use_state = bool(use_state)
        self.seed = int(seed)
        self.scale = float(d_head) ** -0.5
        self.n_buckets = N_BUCKETS
        self.ln_q = nn.LayerNorm(d2)
        self.wq = nn.ModuleList([nn.Linear(d2, d_head) for _ in range(self.n_heads)])
        self.wg = nn.Linear(d2, d_g)                              # P2 gate
        self.arg_emb = nn.Embedding(nvals + 1, d_arg)
        self.w_ap = nn.Linear(d_arg, d_arg)
        self.e_dim = d_g + d_arg + (d_dev if self.use_state else 0)
        self.ln_e = nn.LayerNorm(self.e_dim)
        # head 0 = state-dependence head (full triple + device-only projections)
        self.wk_0 = nn.Linear(self.e_dim, d_head)
        self.wv_0 = nn.Linear(self.e_dim, d_head)
        # head 1 = op-composition compatibility head (op~ only)
        self.wk_1 = nn.Linear(d_g, d_head)
        self.wv_1 = nn.Linear(d_g, d_head)
        if self.use_state:
            self.dev1 = nn.Linear(8, d_dev)
            self.dev2 = nn.Linear(d_dev, d_dev)
            self.w_kd = nn.Linear(d_dev, d_head)
            self.w_vd = nn.Linear(d_dev, d_head)
        else:
            self.dev1 = self.dev2 = self.w_kd = self.w_vd = None
        self.pos_bias = nn.Parameter(torch.zeros(self.n_heads, self.n_buckets))
        self.ln_a = nn.LayerNorm(self.d_a)
        self.wout = nn.Linear(self.d_a, d2)
        self.reset_pa_parameters()
        self.capture = False
        self.last = {}
        self.stats = {"calls": 0, "delta_max": 0.0, "attn_entropy_sum": 0.0, "attn_rows": 0,
                      "o_norm_mean": 0.0}

    # ---- init: zero-init output/gate/step bias + private generator (no global RNG use) ----
    def reset_pa_parameters(self):
        g = torch.Generator(device="cpu").manual_seed(self.seed * 1000003 + 30)
        mods = list(self.wq) + [m for m in (self.wk_0, self.wv_0, self.wk_1, self.wv_1,
                                            self.w_ap, self.dev1, self.dev2, self.w_kd,
                                            self.w_vd) if m is not None]
        with torch.no_grad():
            for m in mods:
                nn.init.xavier_uniform_(m.weight, gain=1.0, generator=g)
                m.bias.zero_()
            self.arg_emb.weight.normal_(0.0, 0.02, generator=g)
            self.arg_emb.weight[ARG_UNK_ROW].zero_()
            for m in (self.ln_q, self.ln_e, self.ln_a):
                m.weight.fill_(1.0)
                m.bias.zero_()
            self.wg.weight.zero_()                        # P2 start = the static lat_tab row
            self.wg.bias.zero_()
            self.wout.weight.zero_()                      # P1 start = bitwise baseline
            self.wout.bias.zero_()
            self.pos_bias.zero_()                         # step bias starts at no bias
        return self

    # ---- forward ----
    def forward(self, x, lat_tab, op_idx, arg_idx, dev_vec, step_ok, attn_ok, dist_idx=None):
        """x:(N,d2) · lat_tab:(head,64) · op_idx:(N,) · arg_idx:(N,S) -1=missing ·
        dev_vec:(N,8) · step_ok:(N,) · attn_ok:(N,N) · dist_idx:(N,N) bucket numbers
        (None => derive from the in-row order of attn_ok)
        -> (delta:(N,d2), attn:(H,N,N), o:(N,d_a))"""
        N = x.shape[0]
        c = self.ln_q(x)
        g = torch.tanh(self.wg(c))                                   # (N,d_g)
        op_row = lat_tab.index_select(0, op_idx.clamp_(0, self.head - 1))
        op_dyn = op_row * (1.0 + g.to(op_row.dtype))                 # P2 (N,d_g)
        av = arg_idx.clamp(min=0)
        ae = self.arg_emb(av)
        m = (arg_idx >= 0).to(ae.dtype).unsqueeze(-1)
        args_e = self.w_ap((ae * m).sum(1) / m.sum(1).clamp(min=1.0))  # (N,d_arg)
        D = None
        if self.use_state:
            D = self.dev2(F.gelu(self.dev1(dev_vec)))                 # (N,d_dev)
            e = self.ln_e(torch.cat([op_dyn, args_e, D], -1))
            k0 = self.wk_0(e) + self.w_kd(D)                          # device: first-class key/value
            v0 = self.wv_0(e) + self.w_vd(D)
        else:
            e = self.ln_e(torch.cat([op_dyn, args_e], -1))
            k0, v0 = self.wk_0(e), self.wv_0(e)
        k1 = self.wk_1(op_dyn)                                        # head 1: op only
        v1 = self.wv_1(op_dyn)
        if dist_idx is None:
            dist_idx = _bucket_matrix(attn_ok)
        dist_idx = dist_idx.clamp(0, self.n_buckets - 1)
        has_key = attn_ok.any(dim=-1, keepdim=True)
        eff = attn_ok | (~has_key)
        attns, outs = [], []
        for h in range(self.n_heads):
            q = self.wq[h](c)
            k, v = (k0, v0) if h == 0 else (k1, v1)
            scores = torch.matmul(q, k.transpose(0, 1)) * self.scale
            scores = scores + self.pos_bias[h].index_select(0, dist_idx.reshape(-1)
                                                            ).reshape(N, N)
            scores = scores.masked_fill(~eff, NEG_INF)
            at = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
            at = at * has_key.to(at.dtype)
            attns.append(at)
            outs.append(torch.matmul(at, v))
        attn = torch.stack(attns, 0)                                  # (H,N,N)
        o = torch.cat(outs, -1)                                       # (N,d_a)
        delta = self.wout(self.ln_a(o))
        if self.capture:
            self.last = {"attn": attn.detach(), "has_key": has_key.squeeze(-1).detach(),
                         "op_idx": op_idx.detach(), "step_ok": step_ok.detach()}
        with torch.no_grad():
            self.stats["calls"] += 1
            self.stats["delta_max"] = max(self.stats["delta_max"],
                                          float(delta.detach().abs().max()))
            rows = int(has_key.sum())
            if rows:
                ent = -(attn.clamp_min(1e-12).log() * attn).sum(-1)
                self.stats["attn_entropy_sum"] += float(ent.masked_select(
                    has_key.squeeze(-1).unsqueeze(0).expand_as(ent)).sum())
                self.stats["attn_rows"] += rows * self.n_heads
            nr = int(step_ok.sum())
            if nr:
                self.stats["o_norm_mean"] = float(o.detach()[step_ok].norm(dim=-1).mean())
        return delta, attn, o

    # ---- P2/P4 audit interfaces ----
    def dynamic_op_rows(self, x, lat_tab, op_idx):
        with torch.no_grad():
            g = torch.tanh(self.wg(self.ln_q(x)))
            row = lat_tab.index_select(0, op_idx.clamp_(0, self.head - 1))
            return row * (1.0 + g)

    def gate_stats(self, x, lat_tab, op_idx):
        with torch.no_grad():
            d = self.dynamic_op_rows(x, lat_tab, op_idx)
            s = lat_tab.index_select(0, op_idx.clamp_(0, self.head - 1))
            return {"gate_abs_max": float(torch.tanh(self.wg(self.ln_q(x))).abs().max()),
                    "row_rel_change_max": float(((d - s).abs().sum(-1)
                                                 / s.abs().sum(-1).clamp(min=1e-6)).max())}

    def bias_report(self):
        b = self.pos_bias.detach().cpu()
        return {"bucket_edges": ["d=1", "d=2", "d=3", "d=4", "d=5..7", "d>=8"],
                "per_head": [[round(float(x), 6) for x in b[h]] for h in range(self.n_heads)],
                "recency_gap_head": [round(float(b[h][0] - b[h][-1]), 6)
                                     for h in range(self.n_heads)],
                "note": "recent-minus-distant bias > 0 => that head learned recency "
                        "weighting; all zero at the start (no prior)"}


def _bucket_matrix(attn_ok):
    """(N,N) bool -> (N,N) long relative-distance bucket numbers (within one sample row only,
    d = t-j >= 1).
    attn_ok is the in-row causal mask => in-row column order is step order (block
    diagonality is guaranteed by build_step_features)."""
    N = int(attn_ok.shape[0])
    idx = torch.arange(N, dtype=torch.long, device=attn_ok.device)
    d = (idx.unsqueeze(1) - idx.unsqueeze(0)).clamp(min=1)
    b = torch.where(d <= 4, d - 1, torch.where(d <= 7, torch.full_like(d, 4),
                                               torch.full_like(d, 5)))
    return b


# ---------------------------------------------------------------------------
# Row-frame shifted fetch (read-only; per-row layout assertion + cache)
# ---------------------------------------------------------------------------

_ROW_META = {}
_ROW_META_CAP = 4096
_LAYOUT_VIOLATIONS = []


def _row_meta(R, cache_key):
    """-> {"op":[..], "res_idx":[..], "bsl":[..], "n_key":int, "n_lab":int, "nslot":int}
    op_j = R.labels[j] (the op supervised at that decision point);
    res_idx_j = the K_RES arg of desc[feat_pos[j]-2] (fail frame = K_PSZ => None)."""
    m = _ROW_META.get(cache_key)
    if m is not None:
        return m
    desc = list(R.desc)
    fp = list(R.feat_pos)
    lab = list(R.labels)
    n_key = min(len(lab), len(fp))
    ops, res_idx = [], []
    for j in range(n_key):
        p_op, p_res = fp[j] - 4, fp[j] - 2
        if not (0 <= p_op < len(desc) and desc[p_op][0] == K_OP):
            _LAYOUT_VIOLATIONS.append({"row": getattr(R, "ri", None), "j": j, "p": p_op,
                                       "why": "op frame not K_OP"})
        ridx = None
        if 0 <= p_res < len(desc) and desc[p_res][0] == K_RES:
            ridx = int(desc[p_res][1])
        elif 0 <= p_res < len(desc) and desc[p_res][0] == K_PSZ:
            ridx = None                                   # fail frame: no device res
        else:
            _LAYOUT_VIOLATIONS.append({"row": getattr(R, "ri", None), "j": j, "p": p_res,
                                       "why": "res/psz frame missing"})
        ops.append(int(lab[j]))
        res_idx.append(ridx)
    nslot = int(getattr(R, "nslot", NSLOT_DEF) or NSLOT_DEF)
    m = {"op": ops, "res_idx": res_idx, "bsl": list(R.bsl), "n_key": n_key,
         "n_lab": len(lab), "nslot": nslot}
    if len(_ROW_META) > _ROW_META_CAP:
        _ROW_META.clear()
    _ROW_META[cache_key] = m
    return m


def _res_vec(R, ridx):
    out = [0.0] * 8
    if ridx is None:
        return out
    res = getattr(R, "res", None)
    if not res or ridx >= len(res) or res[ridx] is None:
        return out
    v = list(res[ridx])
    for i in range(min(8, len(v))):
        try:
            out[i] = float(v[i])
        except (TypeError, ValueError):
            out[i] = 0.0
    return out


def build_step_features(bc, C, dev, dtype=torch.float32):
    """Take (op_idx, arg_idx, dev_vec, step_ok=q-side, key_ok, attn_ok, dist_idx) from the
    batch + read-only cache rows.

    **Key point (CRIT-30A fix, 2026-09-23)**: query-side validity and key-side validity
      **must be separated** --
      * q side (step_ok): the row is a **decision row** (t < len(R.labels)) => includes the
        **terminal decision row / END row** (no op/res frames);
      * key side (key_ok): that step has op/res frames => it can serve as a key (the END row
        cannot).
    During a chain walk the decision row of every step **is always the last row of the prefix
    (= the END row, no frames)**: the old implementation gated the q side by the key-side
    criterion too => attention on the exam segment was identically 0 at every decision step
    (the treatment arm degenerated to the control arm there). After this fix, decision-row
    semantics are the same in training and at exam time (training: row t decides op_t; chain
    walk: row k decides step k+1).
    Shapes: op_idx:(N,) arg_idx:(N,S) dev_vec:(N,8) step_ok:(N,) key_ok:(N,) attn_ok:(N,N)"""
    B, T = bc.B, int(bc.Talloc)
    N = B * T
    ri = list(getattr(bc, "rowIds", range(B)))
    S = NSLOT_DEF
    op_idx = torch.zeros((N,), dtype=torch.long)
    arg_idx = torch.full((N, S), -1, dtype=torch.long)
    dev_vec = torch.zeros((N, 8), dtype=dtype)
    step_ok = torch.zeros((N,), dtype=torch.bool)          # q side: decision rows
    key_ok = torch.zeros((N,), dtype=torch.bool)           # key side: rows with op/res frames
    n_res_nonzero = 0
    for r in range(B):
        R = C.rows[ri[r]]
        meta = _row_meta(R, id(R))
        S = meta["nslot"]
        for t in range(T):
            flat = r * T + t
            if t >= meta["n_lab"]:
                continue
            step_ok[flat] = True                           # decision row (incl. END row)
            op_idx[flat] = meta["op"][t] if t < meta["n_key"] else int(R.labels[t])
            b0 = t * S
            if b0 + S <= len(meta["bsl"]):
                arg_idx[flat] = torch.tensor(meta["bsl"][b0:b0 + S], dtype=torch.long)
            if t < meta["n_key"]:
                rv = _res_vec(R, meta["res_idx"][t])
                dev_vec[flat] = torch.tensor(rv, dtype=dtype)
                key_ok[flat] = True
                if any(x != 0.0 for x in rv):
                    n_res_nonzero += 1
    attn_ok = torch.zeros((N, N), dtype=torch.bool)
    for r in range(B):
        for t in range(T):
            ft = r * T + t
            if not bool(step_ok[ft]):
                continue
            for j in range(t):
                fj = r * T + j
                if bool(key_ok[fj]):
                    attn_ok[ft, fj] = True
    op_idx = op_idx.clamp_(min=0)
    return {"op_idx": op_idx.to(dev), "arg_idx": arg_idx.to(dev),
            "dev_vec": dev_vec.to(dev), "step_ok": step_ok.to(dev), "key_ok": key_ok.to(dev),
            "attn_ok": attn_ok.to(dev),
            "dist_idx": _bucket_matrix(attn_ok).to(dev),
            "arg_missing_frac": float((arg_idx < 0).to(torch.float32).mean()),
            "n_key_steps": int(key_ok.sum()), "n_q_steps": int(step_ok.sum()),
            "n_res_nonzero": n_res_nonzero}


def corpus_device_scan(C, stride=1, max_rows=0):
    """Read-only whole-corpus scan: chain-length distribution + nonzero device res steps
    (corpus prior fact, no GPU use)"""
    import collections as _c
    hist = _c.Counter()
    n_rows = C.n_rows if not max_rows else min(C.n_rows, max_rows)
    steps = nz = 0
    for i in range(0, n_rows, stride):
        R = C.rows[i]
        meta = _row_meta(R, id(R))
        hist[min(meta["n_key"], 8)] += 1
        steps += meta["n_key"]
        for j in range(meta["n_key"]):
            if any(v != 0.0 for v in _res_vec(R, meta["res_idx"][j])):
                nz += 1
    return {"n_rows_scanned": len(range(0, n_rows, stride)), "n_steps": steps,
            "n_device_nonzero_steps": nz,
            "device_nonzero_frac": (nz / steps) if steps else None,
            "chain_len_hist": {str(k): v for k, v in sorted(hist.items())},
            "mean_chain_len": (sum(k * v for k, v in hist.items()) / max(sum(hist.values()), 1))}


def last_valid_index(step_ok, T):
    N = int(step_ok.shape[0])
    B = N // T
    idx = torch.zeros((B,), dtype=torch.long, device=step_ok.device)
    any_ok = torch.zeros((B,), dtype=torch.bool, device=step_ok.device)
    for r in range(B):
        for t in range(T):
            if bool(step_ok[r * T + t]):
                idx[r] = t
                any_ok[r] = True
    return idx, any_ok


def layout_violations():
    return list(_LAYOUT_VIOLATIONS)


# ---------------------------------------------------------------------------
# Model subclass factory: override heads (single-line injection) + forward_joint
# (P4-lite plan summary)
# ---------------------------------------------------------------------------

def make_attn_model(base_cls):
    """base_cls = the gen26/gen30 HybridModel (not modified by one character; read-only
    import)."""
    _bmod = sys.modules.get(base_cls.__module__)
    _END31 = int(getattr(_bmod, "END31", END31_DEF)) if _bmod is not None else END31_DEF
    _NSLOT = int(getattr(_bmod, "NSLOT", NSLOT_DEF)) if _bmod is not None else NSLOT_DEF

    class HybridModelAttn(base_cls):
        """P1+P2(+P4-lite) treatment arm: inside heads() x <- x + delta; render head input
        += W_ps.o_T."""

        def __init__(self, pa_enable=False, pa_seed=None, pa_capture=False, pa_arch=None,
                     plan_summary=True, **kw):
            st = torch.get_rng_state()
            super().__init__(**kw)
            torch.set_rng_state(st)
            self.pa_enabled = False
            self.pa_seed = int(pa_seed if pa_seed is not None else 20260923)
            self.pa_arch = dict(pa_arch or {})
            self.pa_capture = bool(pa_capture)
            self.plan_summary = bool(plan_summary)
            self.pa_delta_max = 0.0
            self.ps_delta_max = 0.0
            self.plan_attn = None
            self.plan_summary_head = None
            self._pa_cache = None
            if pa_enable:
                self.attach_plan_attn(seed=self.pa_seed, capture=pa_capture, enable=True,
                                      arch=self.pa_arch, plan_summary=self.plan_summary)

        def attach_plan_attn(self, seed=None, capture=False, enable=True, arch=None,
                             plan_summary=None):
            st = torch.get_rng_state()
            seed = int(seed if seed is not None else self.pa_seed)
            self.plan_attn = PlanStepAttention(d2=2 * int(self.HID_D), head=int(self.HEAD_C),
                                               nvals=int(self.NVALS_C), nslot=_NSLOT,
                                               seed=seed, **dict(arch or self.pa_arch or {}))
            if plan_summary is None:
                plan_summary = self.plan_summary
            self.plan_summary = bool(plan_summary)
            # The P4-lite head is **always built** (zero-init => a no-op at the start); the
            # switch only gates its **use** -- both readings share one parameter set (ckpt keys
            # never missing; registered as the companion semantics of eval patch [8])
            self.plan_summary_head = nn.Linear(self.plan_attn.d_a, int(self.HID_D))
            with torch.no_grad():
                self.plan_summary_head.weight.zero_()
                self.plan_summary_head.bias.zero_()
            torch.set_rng_state(st)
            self.plan_attn.capture = bool(capture)
            self.pa_seed = seed
            self.pa_enabled = bool(enable)
            self.pa_capture = bool(capture)
            return self

        def set_pa_enabled(self, flag):
            self.pa_enabled = bool(flag)
            return self

        def set_plan_summary(self, flag):
            self.plan_summary = bool(flag)
            return self

        def set_pa_capture(self, flag):
            self.pa_capture = bool(flag)
            if self.plan_attn is not None:
                self.plan_attn.capture = bool(flag)
            return self

        # --- heads: baseline line-by-line copy + one-line injection ---
        def heads(self, H, bc, C, dev, allow_rows):
            if (not self.pa_enabled) or (self.plan_attn is None):
                return base_cls.heads(self, H, bc, C, dev, allow_rows)
            B, L, T = bc.B, bc.L, bc.Talloc
            Hh = H.to(self.step_head.weight.dtype)
            Hf = Hh.reshape(-1, self.HID_D)
            idxH = torch.tensor(bc.idxH, dtype=torch.long, device=dev)
            idxS = torch.tensor(getattr(bc, "idxS", bc.idxC), dtype=torch.long, device=dev)
            x = torch.cat([Hf.index_select(0, idxH), Hf.index_select(0, idxS)], -1)
            feats = build_step_features(bc, C, dev, dtype=x.dtype)
            delta, _attn, o = self.plan_attn(x, self.codebook.lat_tab, feats["op_idx"],
                                             feats["arg_idx"], feats["dev_vec"],
                                             feats["step_ok"], feats["attn_ok"],
                                             feats["dist_idx"])
            self.pa_delta_max = float(delta.detach().abs().max())
            self._pa_cache = {"o": o, "step_ok": feats["step_ok"], "T": int(T), "B": int(B)}
            x = x + delta                                              # <<< P1/P2 sole injection
            step_logits = F.linear(x, self.step_head.weight, self.step_head.bias)
            se = self.slot_emb.weight.unsqueeze(0).expand(x.shape[0], _NSLOT, self.DMODEL_D)
            if self.prompt_cond:
                p = self._prompt_repr(Hh, bc, C, dev)
                if p is None:
                    p = x.new_zeros(x.shape[0] // max(T, 1), self.HID_D)
                pT = p.repeat_interleave(T, dim=0)
                zi = torch.cat([se + x.unsqueeze(1), pT.unsqueeze(1).expand(-1, _NSLOT, -1)],
                               -1)
            else:
                zi = se + x.unsqueeze(1)
            zb = F.linear(zi, self.value_head.weight, self.value_head.bias)
            op_flat = torch.tensor(bc.labels, dtype=torch.long, device=dev)
            op_cl = op_flat.clamp(0, self.HEAD_C - 1)
            allowed = allow_rows.to(dev).index_select(0, op_cl).view(B * T, _NSLOT,
                                                                     self.NVALS_C)
            sv = torch.tensor(bc.slots, dtype=torch.long, device=dev).view(B * T, _NSLOT)
            sm = (sv >= 0) & (op_flat.view(-1, 1) != _END31)
            keep = allowed.bool() & sm.unsqueeze(-1)
            zb = zb.masked_fill(~keep, NEG_BIG)
            tgt = torch.where(sm, sv.clamp(0, self.NVALS_C - 1), torch.full_like(sv, -100))
            return step_logits, zb, tgt

        # --- plan summary vector (P4-lite; carries gradients in training, feeds the render
        #     head) ---
        def plan_summary_vec(self):
            cache = getattr(self, "_pa_cache", None)
            if (not self.plan_summary) or cache is None or self.plan_summary_head is None:
                return None
            idx, any_ok = last_valid_index(cache["step_ok"], cache["T"])
            o3 = cache["o"].view(cache["B"], cache["T"], -1)
            o_last = o3[torch.arange(cache["B"], device=o3.device), idx]
            summ = self.plan_summary_head(o_last)
            return summ * any_ok.to(summ.dtype).unsqueeze(-1)

        # --- plan summary from chain-segment hidden states only (for render-time inference;
        #     produces no step_logits and needs no allow) ---
        def plan_summary_from_H(self, Hc, bc, C, dev):
            if not (self.pa_enabled and self.plan_summary and self.plan_attn is not None):
                return None
            B, L, T = bc.B, bc.L, bc.Talloc
            Hf = Hc.to(self.step_head.weight.dtype).reshape(-1, self.HID_D)
            idxH = torch.tensor(bc.idxH, dtype=torch.long, device=dev)
            idxS = torch.tensor(getattr(bc, "idxS", bc.idxC), dtype=torch.long, device=dev)
            x = torch.cat([Hf.index_select(0, idxH), Hf.index_select(0, idxS)], -1)
            feats = build_step_features(bc, C, dev, dtype=x.dtype)
            delta, _a, o = self.plan_attn(x, self.codebook.lat_tab, feats["op_idx"],
                                          feats["arg_idx"], feats["dev_vec"],
                                          feats["step_ok"], feats["attn_ok"],
                                          feats["dist_idx"])
            self.pa_delta_max = float(delta.detach().abs().max())
            self._pa_cache = {"o": o, "step_ok": feats["step_ok"], "T": int(T), "B": int(B)}
            return self.plan_summary_vec()

        # --- forward_joint: baseline line-by-line copy + P4-lite injection ---
        def forward_joint(self, bc, C, dev, allow, rids, rtgt, lenB_full, grad_ckpt=False):
            Lc = bc.L
            seq_c = self.assemble(bc.B, Lc, bc, C, dev)
            seq_r = self.render_embed(rids)
            seq = torch.cat([seq_c, seq_r], 1)
            L = Lc + int(rids.shape[1])
            cm, cosc, sinc = self.consts(L, dev)
            mk = self.mask_of(cm, bc.B, L, lenB_full, dev)
            if grad_ckpt:
                H = self.forward_body_ckpt(seq, mk, cosc, sinc)
            else:
                H = self.forward_body(seq, mk, cosc, sinc)
            Hc = H[:, :Lc, :].contiguous()
            lg, zb, tgt = self.heads(Hc, bc, C, dev, allow)
            loss, ls, lb, n_unk, nbf = self.loss_f(lg, zb, tgt, bc, lam_bind=self.lam_bind)
            Hr = H[:, Lc:, :]
            summ = self.plan_summary_vec()
            if summ is not None:
                ps = self.plan_summary_head.weight.dtype
                Hr = Hr + summ.to(ps).unsqueeze(1)
                self.ps_delta_max = float(summ.detach().abs().max())
            lgt = self.lm.lm_head(Hr)
            t = rtgt.reshape(-1)
            ntxt = int((t != -100).sum())
            lt = F.cross_entropy(lgt.reshape(-1, lgt.shape[-1]), t,
                                 ignore_index=-100, reduction="sum") / max(ntxt, 1)
            total = loss + self.lam_txt * lt
            nv = int((torch.tensor(bc.labels) != -100).sum())
            return total, ls, lb, lt, nv, nbf, ntxt, L

        # --- parameter groups: the baseline three plus the new module as a fourth ---
        def trainable_groups(self):
            g = dict(super().trainable_groups())
            names = []
            if self.plan_attn is not None:
                names += ["plan_attn." + n for n, _ in self.plan_attn.named_parameters()]
            if self.plan_summary_head is not None:
                names += ["plan_summary_head." + n for n, _ in
                          self.plan_summary_head.named_parameters()]
            if names:
                g["plan_attn"] = names
            return g

        def pa_param_report(self):
            n_pa = count_params(self.plan_attn)
            n_ps = count_params(self.plan_summary_head)
            arch = {k: getattr(self.plan_attn, k) for k in
                    ("n_heads", "d_head", "d_a", "d_g", "d_arg", "d_dev", "d2", "head",
                     "nvals", "use_state", "e_dim", "n_buckets", "seed")} \
                if self.plan_attn is not None else {}
            return {"schema": PA_SCHEMA, "n_pa_params": n_pa, "n_ps_params": n_ps,
                    "n_new_params": n_pa + n_ps,
                    "n_new_params_M": round((n_pa + n_ps) / 1e6, 4), "budget_M": 2.0,
                    "within_budget": bool((n_pa + n_ps) <= 2_000_000),
                    "enabled": bool(self.pa_enabled), "capture": bool(self.pa_capture),
                    "plan_summary": bool(self.plan_summary), "arch": arch,
                    "HID": int(self.HID_D), "head": int(self.HEAD_C),
                    "nvals": int(self.NVALS_C),
                    "bias": self.plan_attn.bias_report() if self.plan_attn is not None else None,
                    "stats": dict(self.plan_attn.stats) if self.plan_attn is not None else {}}

        def pa_stats_reset(self):
            if self.plan_attn is not None:
                self.plan_attn.stats = {"calls": 0, "delta_max": 0.0,
                                        "attn_entropy_sum": 0.0, "attn_rows": 0,
                                        "o_norm_mean": 0.0}
            return self

        # --- attention export (interpretability evidence; per head) ---
        def attn_snapshot(self, bc, C, op_names=None, threshold=0.05):
            last = getattr(self.plan_attn, "last", None) or {}
            if not last:
                return None
            B, T = bc.B, int(bc.Talloc)
            ri = list(getattr(bc, "rowIds", range(B)))
            W = last["attn"].detach().cpu()                       # (H,N,N)
            H = int(W.shape[0])
            out = []
            for r in range(B):
                R = C.rows[ri[r]]
                meta = _row_meta(R, id(R))
                steps, edges = [], []
                for t in range(min(int(bc.Talloc), meta["n_lab"])):
                    ft = r * T + t
                    row = W[:, ft, r * T:(r + 1) * T]             # (H,T)
                    wk = [[round(float(x), 6) for x in row[h, :t]] for h in range(H)]
                    wmean = [round(float(sum(wk[h][j] for h in range(H)) / H), 6)
                             for j in range(t)]
                    if t < meta["n_key"]:
                        oi = int(meta["op"][t])
                        nm = (op_names or {}).get(oi, str(oi))
                    else:
                        nm = "END"
                    steps.append({"t": t, "op": nm,
                                  "res_idx": meta["res_idx"][t] if t < meta["n_key"] else None,
                                  "w_prefix_heads": wk, "w_prefix": wmean,
                                  "w_prefix_head0": wk[0] if H > 0 else [],
                                  "w_prefix_head1": wk[1] if H > 1 else []})
                    for j in range(t):
                        wv = [wk[h][j] for h in range(H)]
                        if max(wv) >= threshold:
                            edges.append([t, j] + [round(x, 6) for x in wv])
                out.append({"row": getattr(R, "ri", ri[r]), "T": len(steps), "steps": steps,
                            "edges": edges, "stepsym": [s["op"] for s in steps],
                            "n_heads": H,
                            "attn_sum": [round(float(W[h, r * T:(r + 1) * T].sum()), 6)
                                         for h in range(H)]})
            return out

    return HybridModelAttn
