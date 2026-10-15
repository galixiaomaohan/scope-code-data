# ---- gen41 fork register (miniF2F domain expansion · changes 98/104/106 continued) ----
# This file = a **minimal-surface patched fork** of gen36_hybrid.py (md5 8e4eabc83141dc750a2ffdf6a65bfffc); parent untouched (fork-and-copy).
# The only variable of this batch = **domain expansion** (V41_DESIGN §1 single-variable principle): vocab V4->V5 (head 194->198) + the training pool
#   expanded to deep corpus (a2 shift, kept bitwise) UNION gen41 new families (3944 rows; 1545 are new gen41 rows).
# Training semantics are **isomorphic** to the G-S arm (a2 state shift + w-tail 1.0 equal weight + slot protocol + seed 20260923 + 6000 steps),
#   parent semantics **unchanged**; the only differences = head growth (old 194 rows carried over with max|Δ|=0 + new 4 rows = component mean) and paths.
# Provenance: pre-registered V41_DESIGN.md / LEAK_PROTOCOL.md (on disk before any training).
# Red lines as in the parent: write surface only scratch/ccp/gen41/**; one job per GPU (target card has no job of ours + free>6000MiB);
#   drct_common 7c31b58d frozen read-only.
# ---- gen36 fork register (terminal-step routing push · change 98 continued) ----
# This file = a **minimal-surface fork copy** of gen35_hybrid.py (md5 791496e5ab934140fd5c9fc2c4db9145); parent untouched (fork-and-copy).
# Differences in this batch (V36_DESIGN §3; the only training-semantics difference is [2]):
#   [1] paths: new G36 (write surface); OUT/artifacts default to gen36; gen35 kept read-only (ledger)
#   [2] **terminal-step decision-position chain CE weighting** `--w-tail W` (labels in {tail_zero, tail_square, tail_other});
#       with W == 1.0 the loss_f **falls back to the parent verbatim** (CCPModel.loss_f, structurally equivalent)
#   [3] registered: curve/summary gain w_tail and cache_md5 (readings carry their provenance)
# Red lines as in the parent: write surface only scratch/ccp/gen36/**; GPU0 user red line / GPU1 reserved region absolutely off-limits; drct_common frozen read-only.
# ---- gen35 fork register (pure planning + mechanized rendering · change 98) ----
# ---- gen35 fork register (pure planning + mechanized rendering · change 98) ----
# This file = a **minimal-surface fork copy** of gen34_hybrid.py (md5 094515124766d60499a81b7db2388ed4); parent untouched (fork-and-copy).
# The only differences of this batch (V35_DESIGN §3 difference list; training semantics **unchanged**):
#   [1] paths: new G35 (write surface); OUT -> gen35/out; G34 kept as a **read-only input** (default slot-sidecar path)
#   [2] marker constants: GEN35_PURE_PLAN=1 / GEN35_FORK_OF / GEN35_PARENT_MD5 (read by PA-EQ and ledger artifacts)
#   [3] line endings: the parent was CRLF and this file reads in text mode => LF normalized (n_crlf_parent=0; the only non-logic difference)
# **The rendering text loss is not changed in this file**: it is disabled by the command line --lam-txt 0 (V35_DESIGN §3 P1), zero code edits.
# Red lines as in the parent: write surface only scratch/ccp/gen35/**; GPU0 user red line absolutely off-limits; drct_common 7c31b58d frozen read-only.
# ---- gen34 fork register (integer rendering push) ----
# -*- coding: utf-8 -*-

# ---- gen34 fork register (integer rendering push · A/B arm treatment file) ----
# This file = a **line-by-line fork copy** of gen33_hybrid.py (md5 f7ebb791eab5cff9757995759c7442d2); existing files untouched (fork-and-copy).
# All parent fixes are inherited (**inherited as-is, zero edits**): the LN_q deviation · CRIT-30A query gating · CRIT-30B vocab convention ·
#   the full gen33 [1]-[7] patch set (v4 vocab / deep-corpus sidecar / 2 classes / fork imports / cuda:1 default).
# The only differences of this batch (pre-registered in scratch/ccp/gen34/V34_DESIGN.md §2), patch list:
#   [0] line endings: the parent is CRLF, this file reads in text mode -> LF normalized (the only non-logic difference; ledger.parent_crlf)
#   [1] paths: new G34; OUT/logs/artifacts point at gen34 (write surface); G33 kept in sys.path (read-only parent)
#   [2] **slotted rendering head** (V34_DESIGN §2): slot embedding slot_render_emb(8,d) ~ N(0,.02^2)
#       + gate g_pos=0 (**naming**: the parent CCPModel/SLANavV32 already uses `slot_emb` for the chain head's 24-slot table
#       (24,dmodel); this batch's slot embedding **must** be renamed slot_render_emb, otherwise it overwrites the chain-head slot table => heads() breaks)
#       + gate g_dev=0 + constant β; with g==0 the increment is exactly +0.0 => bitwise baseline equivalence (an identity, not a float coincidence)
#   [3] `render_batch_slots` + `load_slot_sidecar`: draw slots from the gen34 slot sidecar (DEEP_SLOT_SIDECAR)
#   [4] the rendering loss is **grouped** by slot/non-slot (lam_txt / lam_slot, default 3.0); with `slot_kind is None` the
#       path is **verbatim** the parent's (arm-C equivalence decided bitwise by [3])
#   [5] `--slot-cond` / `--slot-dev` / `--slot-side` / `--slot-beta` / `--lam-slot` CLI
#   [6] `--slot-cond 1` combined with `--pa-enable 1` is **not wired** (planattn brings its own forward_joint that masks the slot
#       branch) => structural SystemExit refusal, no silent degradation. Both A/B arms of this batch use --pa-enable 0
#   [7] import gen34_planattn / gen34_eval (fork)
#   [8] the curve gains slot_g_pos / slot_g_dev / lt_slot (measured gate values = mechanism readings)
# Write surface: only scratch/ccp/gen34/** (gen33/gen32/gen13 artifacts are read-only inputs). Red lines as in the parent.

# ---- gen33 fork register (deep-generation V3.5 treatment arm) ----
# This file = a **line-by-line fork copy** of gen30_hybrid.py (md5 4f630a796190d510c0b17010fc1b1beb); existing files untouched (fork-and-copy).
# The three parent fixes are inherited (**inherited as-is, zero edits**): (1) the LN_q deviation; (2) the CRIT-30A query gating fix;
#   (3) the CRIT-30B vocab convention fix.
# The only differences of this batch (pre-registered in scratch/ccp/gen33/V33_DESIGN.md), patch list:
#   [1] paths: new G33; OUT/logs/artifacts point at gen33 (write surface)
#   [2] vocab convention: default --vocab/--allow point at ccp-vocab/4 (head 194 / value space 352) and ccp_allow_v4.bin
#   [3] cache/sidecar default to the gen33 deep-corpus artifacts (deep_all.bin / DEEP_RENDER_SIDECAR.jsonl)
#   [4] CLASSES 3 classes -> 2 classes (ind/ineq; the deep corpus has no nt class, D-5)
#   [5] import gen33_planattn / gen33_eval (fork) instead of gen30_*
#   [6] default --dev cuda:1 (GPU0 user red line: structurally disabled)
# Write surface: only scratch/ccp/gen33/** (gen30/gen26/gen13 artifacts are read-only inputs). Red lines as in the parent.


# ---- gen30 fork register (treatment arm) ----
# This file = a **line-by-line fork copy** of gen26_hybrid.py (md5 d0602916b81567f27324fedbaed4065e); existing files untouched (fork-and-copy).
# The only difference = attaching the gen30 plan-sequence self-attention (P1 multi-head execution-aware step-level attention + P2 dynamic operator embedding +
#   P4-lite plan summary), see scratch/ccp/gen30/PLAN_ATTN_DESIGN.md (pre-registered). Patch list:
#   [1] paths: new G30; OUT points at gen30/out (write surface)
#   [2] model class: AttnModel = PA30.make_attn_model(HybridModel); used with --pa-enable=1
#   [3] load order: attach_plan_attn after load_init_ccp (v32lib.load_init asserts on unknown keys)
#   [4] the curve gains pa_delta_max/ps_delta_max; the summary gains a plan_attn section
#   [5] the sample block imports gen30_eval (fork) instead of gen26_eval
# Write surface: only scratch/ccp/gen30/** (gen26 artifacts are read-only inputs). Red lines as in the parent.

# gen26_hybrid.py -- gen26 (latent hybrid main arm) T2/T3: joint training harness
#
# Legal basis: constitution change #74 (main arm A = latent hybrid architecture: V32 protocol chain of thought + LM head rendering the full proof)
#       + change #75 (red line R0: the latent thinking mainline must not be deviated from; pure-text direct-generation SFT must not serve as the main arm).
#
# Architecture (registered, one-to-one with design doc HYBRID_DESIGN.md §1/§2):
#   sequence = [chain segment: V32 protocol frames (statement token frame + frame-number/pst/op/res frames, gen13 semantics reused bitwise)]
#          ++ [render segment: (K_EMB, M_TASK) boundary frames + rendering prompt tokens + proof_text tokens]
#   both heads share one SmolLM2 body (single sequence, single forward, causal attention):
#     * chain head (step_head 168 rows / value_head 239 columns / slot_emb 24 slots) reads chain-segment hidden states -> chain loss + slot loss
#     * LM head (self.lm.lm_head) reads render-segment hidden states -> render-segment token CE (loss counted on the target segment only)
#   joint objective (**pre-registered**, change #74 T1(2)):  L = L_step + (lam_bind/NSLOT)·L_slot + lam_txt·L_text
#     the chain-segment loss follows gen13 formula by formula (v32ccp_train.CCPModel.loss_f, not changed at all); L_text = render-segment CE.
#
# Reuse (imports-only, zero edits):
#   * scratch/ccp/gen13/v32ccp_train.py -> CCPModel / CacheCCP (chain head/body/frame assembly, 135M+1.7B dual-track convention)
#   * cppcuda/tools/v32b_train.py       -> L32 / read_plan / micro_split / _a
#   * cppcuda/tools/v31_model.py        -> body/frame kind/mask (indirectly via v32lib)
#   * gen13/CCP_VOCAB.json + ccp_allow_ext.bin -> operator space 168 / value space 239 / allow table
#   * scratch/ccp/gen26/gen26_hybuild.py -> hybrid cache + render sidecar (produced by this batch)
# Interleaved feeding convention = gen23_sft.interleaved_batches semantics (newly written implementation here, provenance registered;
#   reason: change #74 authorizes read-only referencing of in-flight batches, not a hard dependency on gen23's instant state).
#
# Write surface: only scratch/ccp/gen26/**. GPU: exclusive gate (free > 4500MiB) before using a single card.
import os
import sys

# determinism: the CUBLAS workspace must be configured before torch initializes CUDA (same as gen13)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if not ("--det-train" in sys.argv and sys.argv[sys.argv.index("--det-train") + 1:][:1] == ["0"]):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import collections
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import random
import time

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = os.path.join(RES, "scratch", "ccp")
G26 = os.path.join(CCP, "gen26")            # read-only input source (v3 vocab / control-arm artifacts)
G13 = os.path.join(CCP, "gen13")
G30 = os.path.join(CCP, "gen30")            # parent (read-only; fork ledger)
G33 = os.path.join(CCP, "gen33")            # parent (read-only; fork ledger)
G34 = os.path.join(CCP, "gen34")            # parent/input (read-only; default slot-sidecar path)
G36 = os.path.join(CCP, "gen36")            # gen36 fork write surface
G35 = os.path.join(CCP, "gen35")            # read-only previous generation (ledger)
OUT = os.path.join(G36, "out")
G41 = os.path.join(CCP, "gen41")            # gen41 fork write surface (miniF2F domain expansion)
OUT = os.path.join(G41, "out")
LOGS41 = os.path.join(G41, "logs")
GEN41_FORK_OF = "gen36_hybrid.py"
GEN41_PARENT_MD5 = "8e4eabc83141dc750a2ffdf6a65bfffc"
GEN41_HEAD_V5 = 198                         # real head (the V5 artifact's "head"=194 is the legacy V4 anchor, registered D14)
GEN41_N_SLOT_KINDS = 9                      # 0..6 (the gen34 seven) + 7 padding + 8 hyp_ref
# gen41 new-operator row init (head growth): component operators = V4 legal operators **of the same role and the same
# generation (gen32 deep domain)** as the new row
GEN41_COMPONENTS = {
    194: [185, 186, 187, 189, 190],         # g41_subst_bind  : deep-domain device step operators
    195: [188, 191, 192, 193],              # g41_subst_close : deep-domain closing/closure operators
    196: [188, 191, 192, 193],              # g41_eval_arith  : closing
    197: [188, 191, 192, 193],              # g41_eval_mod    : closing
}
GEN41_SLOT_HYPEREF_COMPONENTS = [5, 6]      # new slot class hyp_ref row <- tail_const + template mean

GEN36_TAIL_W = 1
GEN36_FORK_OF = "gen35_hybrid.py"
GEN36_PARENT_MD5 = "791496e5ab934140fd5c9fc2c4db9145"
TAIL_IDX = []                               # gen36 [2]: terminal-step operator label indices
os.makedirs(OUT, exist_ok=True)
for _p in (RES, G13, os.path.join(RES, "cppcuda", "tools"), os.path.join(RES, "skel32"),
           G26, G30, G33, G34, G35, G36, G41,
           os.environ.get("SCOPE_SUPEROPERATOR", os.path.join(RES, "superoperator"))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

def _md5f36(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def tail_op_idx36(vocab_path):
    """Terminal-step operator label indices (read from vocab ops_gen32; the three tail operators)."""
    v = json.load(open(vocab_path, encoding="utf-8"))
    return sorted(int(i) for n, i in (v.get("ops_gen32") or {}).items() if "tail" in n)

import numpy as np                                                          # noqa: E402
import torch                                                                # noqa: E402
from torch import nn                                                        # noqa: E402
import torch.nn.functional as F                                             # noqa: E402

HC = importlib.machinery.SourceFileLoader(
    "g26_v32ccp_train", os.path.join(G13, "v32ccp_train.py"))
HC = importlib.util.module_from_spec(importlib.util.spec_from_loader("g26_v32ccp_train", HC))
HC.__loader__.exec_module(HC)
v32b = HC.v32b
L32 = HC.L32
CacheCCP = HC.CacheCCP
CCPModel = HC.CCPModel
NSLOT = HC.NSLOT
END31 = HC.END31
CLASSES = ("ind", "ineq", "g41")   # gen41 [P4]: deep corpus ind/ineq + this batch's new domain g41
#   (three classes interleaved, per = bs//3 = 2 => bs stays 6; gen41's three subclasses subst_chain/closed_mod/
#    closed_arith merge into **one sampling class** g41; the original subclass labels stay in the sidecar cls field)
HID = L32.HID


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
# Model: hybrid arm = CCP chain head + LM rendering head (one body, single sequence)
# ---------------------------------------------------------------------------

class HybridModel(CCPModel):
    """V32 protocol chain (gen13 inherited bitwise) + LM rendering head, single-sequence joint forward."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.lam_bind = 1.0
        self.lam_txt = 1.0
        self.w_tail = 1.0        # gen36 [2]: terminal-step decision-position CE weight (1.0 = parent verbatim)


    # ---- gen36 [2]: terminal-step decision-position chain CE weighting (V36_DESIGN §3 D1) ----
    def loss_f(self, step_logits, zb, tgt, bc, lam_bind=1.0):
        """Same form as gen13 CCPModel.loss_f; the only addition = CE at decision positions whose label is a **terminal-step operator** multiplied by w_tail.

        w_tail == 1.0 => **fall back to the parent verbatim** (structurally equivalent; no new tensors/parameters => no new module).
        """
        w = float(getattr(self, "w_tail", 1.0) or 1.0)
        if w == 1.0:
            return CCPModel.loss_f(self, step_logits, zb, tgt, bc, lam_bind=lam_bind)
        lab = torch.tensor(bc.labels, dtype=torch.long, device=step_logits.device)
        per = F.cross_entropy(step_logits.reshape(-1, self.HEAD_C), lab.reshape(-1),
                              ignore_index=-100, reduction="none")
        pu = lab.reshape(-1)
        wt = torch.ones_like(per)
        for _ix in TAIL_IDX:
            wt = torch.where(pu == _ix, torch.full_like(wt, w), wt)
        act = (pu != -100)
        ls = (per * wt)[act].sum() / wt[act].sum().clamp_min(1.0)
        t = tgt.clone()
        n_unk = int((t == HC.UNK).sum())
        t[t == HC.UNK] = -100
        nbf = int((t != -100).sum())
        lb = F.cross_entropy(zb.reshape(-1, self.NVALS_C), t.reshape(-1),
                             ignore_index=-100, reduction="sum") / max(nbf, 1)
        return ls + (lam_bind / HC.NSLOT) * lb, ls, lb, n_unk, nbf

    # ---- gen41 [P6]: head-growth loading (old rows copied row by row + new rows = component mean; V32/gen33 precedent) ----
    def load_init_ccp_g41(self, path, parents_ext=None, logp=None):
        """Load from the **gen36 G-S arm snapshot** (head 194, with trained step_head/lat_tab/slot_render_emb).

        Differences from the parent load_init_ccp (registered):
          (a) the init artifact is wrapped as {"model": state_dict, ...} => unwrap first;
          (b) head 194 -> 198: the **old 194 rows of step_head.{weight,bias} / codebook.lat_tab are copied row by
              row** with an assert max|Δ| == 0.0; the new 4 rows = the **mean** of the component-operator rows
              given by GEN41_COMPONENTS;
          (c) slot_render_emb 8 -> 9 rows: old 8 rows copied row by row (asserted) + the new row (hyp_ref) = component-row mean;
          (d) all other key shapes are identical => strictly bitwise load (missing/unexpected must be empty).
        """
        raw = torch.load(path, map_location="cpu", weights_only=False)
        sd = raw["model"] if (isinstance(raw, dict) and "model" in raw) else raw
        meta = {k: v for k, v in raw.items() if k != "model"} if isinstance(raw, dict) else {}
        msd = self.state_dict()
        grow_keys = {"step_head.weight", "step_head.bias", "codebook.lat_tab",
                     "slot_render_emb.weight"}
        dropped, grow_src = [], {}
        for k in list(sd):
            if k in msd and tuple(sd[k].shape) != tuple(msd[k].shape):
                if k in grow_keys:
                    grow_src[k] = sd.pop(k)          # growth key: stash for row-by-row copy (must not be dropped)
                    continue
                dropped.append((k, list(sd[k].shape), list(msd[k].shape)))
                sd.pop(k)
        r = self.load_state_dict(sd, strict=False)
        miss = [k for k in r.missing_keys if k not in grow_keys]
        assert not miss and not list(r.unexpected_keys), (miss, r.unexpected_keys)
        assert {"step_head.weight", "step_head.bias", "codebook.lat_tab"} <= set(grow_src),             sorted(grow_src)
        grew = {}
        with torch.no_grad():
            for key, comp in (("step_head.weight", GEN41_COMPONENTS),
                              ("step_head.bias", GEN41_COMPONENTS),
                              ("codebook.lat_tab", GEN41_COMPONENTS)):
                src = grow_src.get(key)
                if src is None:
                    raise SystemExit("init missing %s (the gen36 G-S snapshot is required)" % key)
                nold = int(src.shape[0])
                assert nold == 194, (key, nold)
                dst = getattr(self, key.split(".")[0])
                dst = getattr(dst, key.split(".")[1])
                dst[:nold].copy_(src)
                d0 = float((dst.detach().cpu()[:nold] - src).abs().max())
                assert d0 == 0.0, ("old-row copy is not bitwise", key, d0)
                for row, comps in sorted(comp.items()):
                    m = torch.stack([dst[c].detach() for c in comps], 0).mean(0)
                    dst[row].copy_(m)
                grew[key] = {"n_old": nold, "n_new": int(dst.shape[0]) - nold,
                             "old_rows_max_abs_diff": d0,
                             "components": {str(k2): v2 for k2, v2 in sorted(comp.items())}}
            sk = "slot_render_emb.weight"
            if sk in grow_src:
                src = grow_src[sk]
                nold = int(src.shape[0])
                assert nold == 8, nold
                dstw = self.slot_render_emb.weight
                dstw[:nold].copy_(src)
                d0s = float((dstw.detach().cpu()[:nold] - src).abs().max())
                assert d0s == 0.0, ("slot-head old-row copy is not bitwise", d0s)
                m = torch.stack([dstw[c].detach()
                                 for c in GEN41_SLOT_HYPEREF_COMPONENTS], 0).mean(0)
                dstw[nold].copy_(m)
                grew[sk] = {"n_old": nold, "n_new": int(dstw.shape[0]) - nold,
                            "old_rows_max_abs_diff": d0s,
                            "components": {"8": GEN41_SLOT_HYPEREF_COMPONENTS}}
            else:
                grew[sk] = {"n_old": 0, "note": "init snapshot has no slot head (nothing copied)"}
        n_new = sum(p.numel() for p in self.parameters())
        info = {"mode": "g41_head_growth_from_parent_snapshot", "ckpt": path,
                "parent_meta": {k2: meta.get(k2) for k2 in ("step", "head", "nvals")},
                "dropped_shape_mismatch": dropped, "growth": grew,
                "n_params": n_new, "head": self.HEAD_C, "nvals": self.NVALS_C,
                "slot_rows_reused": int(grew.get(sk, {}).get("n_old") or 0)}
        if logp:
            log("gen41 head growth: %s" % json.dumps(
                {"parent": {k2: meta.get(k2) for k2 in ("step", "head", "nvals")},
                 "growth_keys": sorted(grew), "step_head": grew.get("step_head.weight"),
                 "slot_head": grew.get(sk), "params_M": round(n_new / 1e6, 3)},
                ensure_ascii=False), logp)
        return info

    # ---- render-segment payload ----
    def render_embed(self, rids):
        """(B, Lr) token ids -> (B, Lr, HID) body embedding (same embedding matrix as the chain-segment token frames)."""
        B, Lr = rids.shape
        e = self.lm.model.embed_tokens.weight.index_select(0, rids.reshape(-1))
        e = e.view(B, Lr, self.HID_D)
        if getattr(self, "body_dtype", None) is not None and self.body_dtype != torch.float32:
            e = e.to(self.body_dtype)
        return e

    def forward_joint(self, bc, C, dev, allow, rids, rtgt, lenB_full, grad_ckpt=False,
                      slot_tok=None, slot_kind=None):                # gen34 [8]
        """Single micro-batch joint forward -> (loss, ls, lb, lt, nv, nbf, ntxt, Lfull)"""
        Lc = bc.L
        seq_c = self.assemble(bc.B, Lc, bc, C, dev)                  # (B, Lc, HID)
        seq_r = self.render_embed(rids)                              # (B, Lr, HID)
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
        # ---- render segment: loss only on the target segment (prompt/boundary frames = -100) ----
        t = rtgt.reshape(-1)
        ntxt = int((t != -100).sum())
        if getattr(self, "slot_cond", False) and slot_kind is not None:
            # gen34 [8]/[4]: slot vs non-slot **grouped** loss (slots use logits after the device-gated fusion)
            ltr, lt_main, _lts, _ns = self.render_slot_forward(
                H[:, Lc:, :], rtgt, slot_tok, slot_kind)
            nv = int((torch.tensor(bc.labels) != -100).sum())
            return loss + ltr, ls, lb, lt_main, nv, nbf, ntxt, L
        lgt = self.lm.lm_head(H[:, Lc:, :])
        lt = F.cross_entropy(lgt.reshape(-1, lgt.shape[-1]), t,
                             ignore_index=-100, reduction="sum") / max(ntxt, 1)
        total = loss + self.lam_txt * lt
        nv = int((torch.tensor(bc.labels) != -100).sum())
        return total, ls, lb, lt, nv, nbf, ntxt, L

    def trainable_groups(self):
        """Three parameter groups (self-checked: the gradient exit of each of the three losses exists)."""
        return {"chain_step": ["step_head.weight", "step_head.bias"],
                "chain_slot": ["value_head.weight", "value_head.bias", "slot_emb.weight"],
                "text": ["lm.lm_head.weight"]}


# ---------------------------------------------------------------------------
# gen30 treatment arm: plan-sequence self-attention (P1+P2) + plan summary (P4-lite)
# ---------------------------------------------------------------------------
import gen34_planattn as PA33                                        # noqa: E402

AttnModel = PA33.make_attn_model(HybridModel)

# ---------------------------------------------------------------------------
# gen34 [8]: integer-slot-conditional rendering head (V34_DESIGN §2)
# ---------------------------------------------------------------------------
n_slot_kinds = 9          # gen41 [P5]: 0..6 (gen34) + 7 padding + 8 hyp_ref


def load_slot_sidecar(path):
    """Slot sidecar (produced by gen34_slots) -> {ri: {"slots": [[abs_id_idx, tokid, kind, src], ...]}}."""
    recs = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r.get("slots") is not None:
                recs[int(r["ri"])] = r
    return recs


def render_batch_slots(side, sslot, idx, dev, pad_id=0, slot_on=0):
    """render_batch (parent verbatim) + slot tensors (label coordinates).

    The slot sidecar gives the **target token absolute id index** a; the parent convention is `rtgt[b, p] = ids[p+1]`
    => that token's label position p = a - 1. Returns slot_tok/slot_kind shaped (B, Lr), non-slot = -1.
    """
    rids, rtgt, ln = render_batch(side, idx, dev, pad_id=pad_id)
    if not slot_on or not sslot:
        return rids, rtgt, ln, None, None
    B, Lr = rids.shape
    stok = torch.full((B, Lr), -1, dtype=torch.long)
    skind = torch.full((B, Lr), -1, dtype=torch.long)
    for b, i in enumerate(idx):
        r = sslot.get(int(i))
        if not r:
            continue
        for (a, tokid, kind, _src) in r["slots"]:
            p = int(a) - 1
            if 0 <= p < Lr:
                stok[b, p] = int(tokid)
                skind[b, p] = int(kind)
    return rids, rtgt, ln, stok.to(dev), skind.to(dev)


class SlotHybridModel(HybridModel):
    """Hybrid main arm + integer-slot-conditional rendering head (V34_DESIGN §2).

    Arm A (slot_dev=1) = slot embedding + slot loss + **device one-hot gated fusion**;
    Arm B (slot_dev=0) = slot embedding + slot loss (LM self-fills, no device query).
    Both arms have the **same parameter count** (E_slot/g_pos/g_dev all present), only the g_dev channel is
    structurally inactive => A−B = the net device contribution.
    """

    def enable_slot(self, dev_flag=True, beta=8.0, lam_slot=3.0, seed=0, logp=None):
        H = self.HID_D
        g = torch.Generator().manual_seed(int(seed) + 340034)
        self.slot_render_emb = nn.Embedding(n_slot_kinds, H)
        with torch.no_grad():
            self.slot_render_emb.weight.normal_(0.0, 0.02, generator=g)
        self.g_pos = nn.Parameter(torch.zeros(1))
        self.g_dev = nn.Parameter(torch.zeros(1))
        self.slot_cond = True
        self.slot_dev = bool(dev_flag)
        self.slot_beta = float(beta)
        self.lam_slot = float(lam_slot)
        self._lt_slot_last = 0.0
        if logp:
            log("gen34 slot conditioning: kinds=%d beta=%g dev=%s lam_slot=%g | E_slot ~N(0,.02^2) "
                "g_pos=g_dev=0.0 (start increment identically +0.0 => the PA-EQ identity)"
                % (n_slot_kinds, self.slot_beta, self.slot_dev, self.lam_slot), logp)
        return self

    def slot_param_report(self):
        return {"slot_cond": bool(getattr(self, "slot_cond", False)),
                "slot_dev": bool(getattr(self, "slot_dev", False)),
                "slot_beta": getattr(self, "slot_beta", None),
                "lam_slot": getattr(self, "lam_slot", None),
                "n_slot_kinds": n_slot_kinds,
                "slot_params": int(self.slot_render_emb.weight.numel() + 2),
                "g_pos": float(self.g_pos.detach()), "g_dev": float(self.g_dev.detach())}

    def render_slot_forward(self, Hr, rtgt, slot_tok, slot_kind):
        """Slot rendering forward -> (L_render, lt_main, lt_slot, n_slot)."""
        m = (slot_kind >= 0)
        Hr2 = Hr
        if bool(m.any()):
            Hr2 = Hr + (self.g_pos * self.slot_render_emb(slot_kind.clamp(min=0))) * \
                m.unsqueeze(-1).to(Hr.dtype)
        lgt = self.lm.lm_head(Hr2)
        t = rtgt.reshape(-1)
        if self.slot_dev and slot_tok is not None:
            dm = m & (slot_tok >= 0)
            if bool(dm.any()):
                oh = F.one_hot(slot_tok.clamp(min=0), lgt.shape[-1]).to(lgt.dtype)
                # oh (B,Lr,V) * dm (B,Lr,1) => (B,Lr,V); keep the mask 3-dim (a previous reshape(-1) broke broadcasting)
                add = ((self.g_dev * self.slot_beta) * oh) * dm.unsqueeze(-1).to(lgt.dtype)
                lgt = lgt + add
        lg = lgt.reshape(-1, lgt.shape[-1])
        mm = m.reshape(-1)
        sel_s = mm & (t != -100)
        sel_m = (~mm) & (t != -100)
        n_slot = int(sel_s.sum())
        lt_main = F.cross_entropy(lg[sel_m], t[sel_m], reduction="sum") \
            / max(int(sel_m.sum()), 1)
        lt_slot = (F.cross_entropy(lg[sel_s], t[sel_s], reduction="sum") / max(n_slot, 1)) \
            if n_slot else torch.zeros((), device=lg.device, dtype=lg.dtype)
        self._lt_slot_last = float(lt_slot.detach())
        return self.lam_txt * lt_main + self.lam_slot * lt_slot, lt_main, lt_slot, n_slot



# ---------------------------------------------------------------------------
# Render sidecar / batch assembly
# ---------------------------------------------------------------------------

def load_sidecar(path):
    recs = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            recs[int(r["ri"])] = r
    return recs


def render_batch(side, idx, dev, pad_id=0):
    """-> (rids, rtgt, ln_real) -- render segment = [M_TASK]+prompt+target; loss only on target."""
    Lr = max(side[i]["n"] for i in idx)
    B = len(idx)
    rids = torch.full((B, Lr), pad_id, dtype=torch.long)
    rtgt = torch.full((B, Lr), -100, dtype=torch.long)
    ln = []
    for b, i in enumerate(idx):
        s = side[i]
        ids = s["ids"]
        rids[b, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        # CRIT-1 fix (gen27 audit): autoregressive misalignment -- the label at position p must be the **next** token
        #   ids[p+1], i.e. target-segment ids[np:] lands in [np-1, len-1). The old form rtgt[np:len]=ids[np:] was a
        #   trivial "copy the input" shortcut (especially with tie_word_embeddings), and at inference the first step
        #   read position np-1, which is always -100.
        np_ = int(s["np"])
        rtgt[b, np_ - 1:len(ids) - 1] = torch.tensor(ids[np_:], dtype=torch.long)
        ln.append(len(ids))
    return rids.to(dev), rtgt.to(dev), ln


def micro_split_hybrid(idx, C, side, cap):
    """Split micro-batches by padding load B*L (L = chain length + render length) (ascending total length within a batch)."""
    if cap <= 0 or len(idx) <= 1:
        return [list(idx)]
    order = sorted(idx, key=lambda i: len(C.rows[i].desc) + side[i]["n"])
    out, cur = [], []
    for i in order:
        cand = cur + [i]
        Lmax = max(len(C.rows[j].desc) + side[j]["n"] for j in cand)
        if cur and len(cand) * Lmax > cap:
            out.append(cur)
            cur = [i]
        else:
            cur = cand
    if cur:
        out.append(cur)
    return out


def interleaved_batches(pool_by_cls, bs=6, seed=0):
    """Interleaved feeding (convention = gen23_sft.interleaved_batches: all three classes appear in every batch, in-class shuffle without replacement).
    provenance: this implementation was newly written in gen26 (registered); semantically equivalent to the same-named function in gen23_sft.py."""
    rng = random.Random(seed)
    buckets = {}
    for c in CLASSES:
        items = list(pool_by_cls[c])
        rng.shuffle(items)
        buckets[c] = items
    per = max(1, bs // len(CLASSES))
    idx = {c: 0 for c in CLASSES}
    out = []
    while all(idx[c] < len(buckets[c]) for c in CLASSES):
        grp = []
        for c in CLASSES:
            grp += buckets[c][idx[c]:idx[c] + per]
            idx[c] += per
        out.append(grp)
    return out


def build_plan(train_idx, C, steps, bs, seed, snap_path=None):
    """Pre-registered batch order: reshuffle every epoch; loop epochs if steps are insufficient (registered). Returns (plan, plan_md5)."""
    pool = collections.defaultdict(list)
    for i in train_idx:
        pool[_cls_of(C, i)].append(i)
    missing = [c for c in CLASSES if not pool.get(c)]
    if missing:
        raise SystemExit("build_plan: class pool empty %s (cls mapping unset/wrong) pool keys=%s"
                         % (missing, sorted(pool)))
    plan, epoch = [], 0
    while len(plan) < steps:
        for grp in interleaved_batches(pool, bs=bs, seed=seed + 1000 * epoch):
            plan.append(list(grp))
            if len(plan) >= steps:
                break
        epoch += 1
    plan = plan[:steps]
    assert len(plan) == steps, ("batch order under-generated", len(plan), steps)
    blob = "\n".join(" ".join(str(x) for x in rr) for rr in plan).encode("utf-8")
    pmd5 = hashlib.md5(blob).hexdigest()
    if snap_path:
        os.makedirs(os.path.dirname(snap_path), exist_ok=True)
        with open(snap_path, "w", encoding="utf-8") as f:
            f.write("%d %d\n" % (len(plan), bs))
            for rr in plan:
                f.write(" ".join(str(x) for x in rr) + "\n")
    return plan, pmd5


_CLS_OF = {}


def _cls_of(C, i):
    return _CLS_OF.get(i, "?")


def set_cls_map(rows_meta, idx_list):
    for i in idx_list:
        _CLS_OF[i] = rows_meta[i]


# ---------------------------------------------------------------------------
# Training main flow
# ---------------------------------------------------------------------------

def gpu_gate(dev, logp):
    """gen41 [P10] launch gate (constitution #104 / NEXT_STEP [A]): target card has **no job of ours** and free > 6000MiB."""
    import gen41_common as _C41
    g = _C41.gpu_gate(dev, logp=logp, require_free=6000,
                      note="gen41 one job per GPU gate (when the other side's residual context is <2%% VRAM, decide by free)")
    return g


def _g41_meta(a, minfo, head, nvals):
    """gen41 snapshot metadata (provenance: md5s of every input behind this batch's readings)."""
    import gen41_common as _C41
    return {"fork_of": GEN41_FORK_OF, "parent_md5": GEN41_PARENT_MD5,
            "head": head, "nvals": nvals, "n_slot_kinds": n_slot_kinds,
            "components": {str(k): v for k, v in sorted(GEN41_COMPONENTS.items())},
            "slot_hyp_ref_components": GEN41_SLOT_HYPEREF_COMPONENTS,
            "init": {"path": a.init, "md5": _md5f36(a.init),
                     "growth": (minfo or {}).get("growth")},
            "inputs": {"cache": {"path": a.cache, "md5": _md5f36(a.cache)},
                       "vocab": {"path": a.vocab, "md5": _md5f36(a.vocab)},
                       "allow": {"path": a.allow, "md5": _md5f36(a.allow)},
                       "side": {"path": a.side, "md5": _md5f36(a.side)},
                       "slot_side": {"path": a.slot_side, "md5": _md5f36(a.slot_side)}},
            "recipe": "G-S (a2 shift + w-tail 1.0 equal weight + slot protocol) · seed %d · steps %d · bs %d"
                      % (a.seed, a.steps, a.bs),
            "judge_md5": _C41.JUDGE_MD5_EXPECT}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(G41, "cache", "g41_all.bin"))
    ap.add_argument("--side", default=os.path.join(G41, "cache",
                                              "DEEP_RENDER_SIDECAR.jsonl"))
    ap.add_argument("--init", default="ckpt/sla_v31f.pt")
    ap.add_argument("--lat31", default="cppcuda/cache/lat_tab_v31.bin")
    ap.add_argument("--allow", default=os.path.join(G41, "cache", "ccp_allow_v5.bin"))
    ap.add_argument("--vocab", default=os.path.join(G41, "cache", "CCP_VOCAB_V5.json"))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--smoke", type=int, default=0, help=">0 = smoke steps (100)")
    ap.add_argument("--run-kind", default="", choices=["", "smoke", "probe", "formal"],
                    help="naming-gate class (empty = inferred from --smoke); probe = L0 reconnaissance version")
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lam-bind", type=float, default=1.0)
    ap.add_argument("--lam-txt", type=float, default=1.0)
    ap.add_argument("--w-tail", type=float, default=1.0,
                    help="gen36 [2]: terminal-step decision-position chain CE weight (1.0 = parent verbatim fallback)")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--grad-ckpt", type=int, default=1)
    ap.add_argument("--micro-token-cap", type=int, default=9000)
    ap.add_argument("--det-train", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--dev", default="cuda:1")
    ap.add_argument("--logevery", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-rows", type=int, default=48, help="MED-9: default 48 (held-out pool 236)")
    ap.add_argument("--eval-rotate", type=int, default=1, help="MED-9: rotate the slice at every eval point")
    ap.add_argument("--snap-every", type=int, default=100)
    ap.add_argument("--stopnan", type=int, default=1)
    ap.add_argument("--args-mode", default="bridge", choices=["bridge", "gold"],
                    help="HIGH-2: samples/eval use bridge (device-derived args); gold for diagnostics only")
    ap.add_argument("--sample-max-new", type=int, default=160)
    ap.add_argument("--body-dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--base", default=None)
    ap.add_argument("--out", default=os.path.join(G41, "ckpt", "g41_smoke.pt"))
    ap.add_argument("--curve", default=os.path.join(OUT, "DEEP_CURVE.jsonl"))
    ap.add_argument("--plan-out", default=os.path.join(OUT, "batch_idx_deep.txt"))
    ap.add_argument("--tag", default="", help="run label (e.g. l0_probe)")
    ap.add_argument("--samples", default=os.path.join(OUT, "DEEP_SAMPLES.json"))
    ap.add_argument("--summary", default=os.path.join(OUT, "DEEP_SUMMARY.json"))
    ap.add_argument("--log", default=os.path.join(G41, "logs", "gen41_hybrid.log"))
    ap.add_argument("--render-only-eval", type=int, default=1)
    ap.add_argument("--pa-enable", type=int, default=1, help="gen30: P1/P2 step-level attention")
    ap.add_argument("--pa-seed", type=int, default=-1, help="gen30: new-module seed (-1=--seed)")
    ap.add_argument("--pa-state", type=int, default=1, help="gen30: device channel (0=nostate)")
    ap.add_argument("--plan-summary", type=int, default=1, help="gen30: P4-lite plan summary")
    ap.add_argument("--pa-capture", type=int, default=0)
    ap.add_argument("--pa-heads", type=int, default=2)
    ap.add_argument("--pa-dhead", type=int, default=64)
    ap.add_argument("--slot-cond", type=int, default=0, help="gen34: slot-conditional rendering head")
    ap.add_argument("--slot-dev", type=int, default=1,
                    help="gen34: 1=device one-hot gated fusion (arm A); 0=LM self-fill (arm B)")
    ap.add_argument("--slot-side", default=os.path.join(G41, "cache",
                                                       "DEEP_SLOT_SIDECAR.jsonl"))
    ap.add_argument("--slot-beta", type=float, default=8.0)
    ap.add_argument("--lam-slot", type=float, default=3.0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    smoke = a.smoke > 0
    steps = a.smoke if smoke else a.steps
    kind = a.run_kind or ("smoke" if smoke else "formal")
    bn = os.path.basename(a.out)
    other = [k for k in ("smoke", "probe", "formal") if k != kind]
    if kind not in bn:
        raise SystemExit("HIGH-5/naming gate: artifact name must contain run-kind '%s': %s" % (kind, bn))
    if any(k in bn for k in other):
        raise SystemExit("HIGH-5/naming gate: artifact name must not contain %s: %s" % (other, bn))
    t0 = time.time()
    dev = torch.device(a.dev)
    ginfo = gpu_gate(a.dev, a.log) if str(a.dev).startswith("cuda") else {}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    if a.det_train:
        torch.use_deterministic_algorithms(True)

    voc = json.load(open(a.vocab, encoding="utf-8"))
    head, nvals = int(voc.get("head_v5", voc["head"])), voc["n_vals_ext"]
    # gen41 [P7]: V5 artifact "head" = 194 (legacy V4 anchor) · real head = "head_v5" = 198 (registered D14)
    assert head == GEN41_HEAD_V5 and nvals == 352, (head, nvals)
    parents = {int(k): v["parent_idx"] for k, v in voc["init_map"].items()}
    allow = torch.from_numpy(np.fromfile(a.allow, dtype=np.uint8).reshape(
        head, NSLOT, nvals).copy())
    assert tuple(allow.shape) == (head, NSLOT, nvals), allow.shape
    C = CacheCCP(a.cache)
    assert (C.HEAD, C.NVALS, C.ENDIDX) == (head, nvals, END31)
    side = load_sidecar(a.side)
    assert len(side) == C.n_rows, (len(side), C.n_rows)
    segs = (C.segidx or {}).get("segments", {})
    train_idx, held_idx = [], []
    for sg, (lo, hi) in segs.items():
        (held_idx if sg == "a_held" else train_idx).extend(range(lo, hi))
    cls_by_row = {}
    for i, s in side.items():
        _c = s["cls"]
        cls_by_row[int(i)] = ("g41" if str(s.get("name", "")).startswith("gen41_")
                              else _c)      # gen41 [P8]
    set_cls_map(cls_by_row, range(C.n_rows))
    slot_on = int(a.slot_cond)            # gen34 [5] placed after train_idx/held
    sslot = load_slot_sidecar(a.slot_side) if slot_on else None
    if slot_on:
        n_ok = sum(1 for i in train_idx if (sslot.get(i) or {}).get("slots"))
        log("gen34 slot sidecar: %s rows=%d (train segment with slots=%d/%d)"
            % (a.slot_side, len(sslot), n_ok, len(train_idx)), a.log)
        assert n_ok == len(train_idx), (n_ok, len(train_idx))
    log("cache %s rows=%d segs=%s train=%d held=%d sidecar=%d" %
        (a.cache, C.n_rows, segs, len(train_idx), len(held_idx), len(side)), a.log)
    plan, pmd5 = build_plan(train_idx, C, steps, a.bs, a.seed, snap_path=a.plan_out)
    log("gen34 batch order md5=%s steps=%d bs=%d seed=%d (arms A/B must match)"
        % (pmd5, len(plan), a.bs, a.seed), a.log)
    n_off = sum(1 for rr in plan for i in rr if i not in set(train_idx))
    assert n_off == 0, "batch order hits non-training rows: %d (held/out-of-range)" % n_off
    log("batch order: %d steps bs=%d md5=%s (interleaved feeding, all 3 classes present; 0 held hits)" % (len(plan), a.bs, pmd5),
        a.log)
    pools = collections.Counter(_cls_of(C, i) for i in train_idx)
    log("MED-6 sampler coverage register: class pool=%s | batches per epoch=%d (limited by the smallest class), bs=%d, in-class per=%d "
        "=> a single epoch covers only %d rows per class (reshuffling across epochs widens coverage; coverage listed in the curve)"
        % (dict(pools), min(pools.values()) // max(1, a.bs // 3), a.bs, max(1, a.bs // 3),
           max(1, a.bs // 3)), a.log)

    torch.manual_seed(a.seed)
    if a.slot_cond and a.pa_enable:
        raise SystemExit("gen34 [6]: --slot-cond 1 combined with --pa-enable 1 is not wired "
                         "(planattn's own forward_joint masks the slot branch); no silent degradation")
    MCLS = (SlotHybridModel if a.slot_cond
            else (AttnModel if a.pa_enable else HybridModel))
    model = MCLS(base=a.base, lat_tab_path=a.lat31, head=head, nvals=nvals,
                 parents=parents)
    if a.body_dtype == "bf16":
        model.cast_body(torch.bfloat16)
    if a.slot_cond:      # gen41 [P9]: the slot head must be **built before loading** (otherwise the parent's trained slot table is re-randomized)
        model.enable_slot(dev_flag=bool(a.slot_dev), beta=a.slot_beta,
                          lam_slot=a.lam_slot, seed=a.seed, logp=a.log)
    minfo = model.load_init_ccp_g41(a.init, parents_ext=GEN41_COMPONENTS, logp=a.log)
    if a.pa_enable:                              # gen30 [3]: load order (after init)
        _paseed = int(a.pa_seed) if a.pa_seed >= 0 else int(a.seed)
        model.attach_plan_attn(seed=_paseed, capture=bool(a.pa_capture), enable=True,
                               arch={"use_state": bool(a.pa_state),
                                     "n_heads": int(a.pa_heads), "d_head": int(a.pa_dhead)},
                               plan_summary=bool(a.plan_summary))
        _parep = model.pa_param_report()
        assert _parep["within_budget"], _parep
        log("gen30 P1/P2/P4-lite: %s" % json.dumps(_parep, ensure_ascii=False), a.log)
    if a.slot_cond:                              # gen41 [P9]: already built before loading (zero edits here)
        log("gen41 slot head: old %d rows copied row by row + new row (hyp_ref) = component mean"
            % int((minfo.get('growth') or {}).get('slot_render_emb.weight', {})
                  .get('n_old') or 0), a.log)
    model.lam_bind, model.lam_txt = a.lam_bind, a.lam_txt
    model.w_tail = float(a.w_tail)                     # gen36 [2]
    TAIL_IDX[:] = tail_op_idx36(a.vocab)
    log("gen36 terminal-step weighting: w_tail=%g TAIL_IDX=%s (bitwise parent fallback at 1.0)"
        % (a.w_tail, TAIL_IDX), a.log)
    model = model.to(dev)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    n_train_p = sum(p.numel() for p in params)
    n_lm = sum(p.numel() for p in model.lm.parameters())
    log("model: total %.1fM / trainable %.1fM (body %.1fM) | head=%d nvals=%d | body=%s dev=%s | "
        "lam_bind=%g lam_txt=%g" % (minfo["n_params"] / 1e6, n_train_p / 1e6, n_lm / 1e6,
                                    head, nvals, a.body_dtype, a.dev, a.lam_bind, a.lam_txt),
        a.log)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, a.warmup)))

    curve, step, micro_cnt, t_last = [], 0, 0, time.time()
    stopped_nan = False
    seen = {c: set() for c in CLASSES}          # MED-6: sampling coverage column
    while step < steps:
        grp = plan[step]
        mbs = micro_split_hybrid(grp, C, side, a.micro_token_cap)
        for i in grp:
            seen[_cls_of(C, i)].add(i)
        bc_cls = collections.Counter(_cls_of(C, i) for i in grp)
        opt.zero_grad(set_to_none=True)
        acc = {"loss": 0.0, "ls": 0.0, "lb": 0.0, "lt": 0.0, "nv": 0, "nbf": 0, "ntxt": 0}
        Lmax = 0
        for mb in mbs:
            bc = L32.BatchCpu32().build(C, list(mb), 0, 0, 0)
            bc.rowIds = list(mb)
            _r, _t, ln, _stok, _skind = render_batch_slots(side, sslot, mb, dev,
                                                          slot_on=slot_on)
            rids, rtgt = _r, _t
            lenB_full = [len(C.rows[i].desc) + ln[b] for b, i in enumerate(mb)]
            out = model.forward_joint(bc, C, dev, allow, rids, rtgt, lenB_full,
                                      grad_ckpt=bool(a.grad_ckpt),
                                      slot_tok=_stok, slot_kind=_skind)
            total, ls, lb, lt, nv, nbf, ntxt, L = out
            (total / len(mbs)).backward()
            fv = [float(x.detach()) for x in (total, ls, lb, lt)]
            acc["loss"] += fv[0] * max(nv, 1)
            acc["ls"] += fv[1] * max(nv, 1)
            acc["lb"] += fv[2] * max(nbf, 1)
            acc["lt"] += fv[3] * max(ntxt, 1)
            acc["nv"] += nv
            acc["nbf"] += nbf
            acc["ntxt"] += ntxt
            Lmax = max(Lmax, L)
            micro_cnt += 1
            del out, total, ls, lb, lt
        gn = torch.nn.utils.clip_grad_norm_(params, a.clip)
        if a.stopnan and not np.isfinite(float(gn)):
            stopped_nan = True
            log("STOP-NaN step=%d |g|=%.4g" % (step, float(gn)), a.log)
            break
        opt.step()
        sched.step()
        step += 1
        rec = {"step": step, "loss": round(acc["loss"] / max(acc["nv"], 1), 5),
               "chain": round(acc["ls"] / max(acc["nv"], 1), 5),
               "slot": round(acc["lb"] / max(acc["nbf"], 1), 5),
               "text": round(acc["lt"] / max(acc["ntxt"], 1), 5),
               "gn": round(float(gn), 4), "lr": round(opt.param_groups[0]["lr"], 8),
               "pa_delta_max": round(float(getattr(model, "pa_delta_max", 0.0)), 6),
               "ps_delta_max": round(float(getattr(model, "ps_delta_max", 0.0)), 6),
               "slot_g_pos": (round(float(getattr(model, "g_pos", torch.zeros(1)).detach()), 8)
                              if a.slot_cond else None),
               "slot_g_dev": (round(float(getattr(model, "g_dev", torch.zeros(1)).detach()), 8)
                              if a.slot_cond else None),
               "lt_slot": round(float(getattr(model, "_lt_slot_last", 0.0)), 5),
               "L": Lmax, "micro": len(mbs), "nv": acc["nv"], "nbf": acc["nbf"],
               "ntxt": acc["ntxt"], "secs": round(time.time() - t0, 2),
               "batch_cls": [_cls_of(C, i) for i in grp],
               "batch_cls_counts": {c: bc_cls.get(c, 0) for c in CLASSES},
               "seen": {c: len(seen[c]) for c in CLASSES}}
        curve.append(rec)
        if step % a.logevery == 0 or step == 1:
            log("[%4d/%d] loss=%.4f chain=%.4f slot=%.4f text=%.4f gn=%.3f L=%d micro=%d "
                "seen=%s t=%.0fs" % (step, steps, rec["loss"], rec["chain"], rec["slot"],
                                     rec["text"], rec["gn"], rec["L"], rec["micro"],
                                     rec["seen"], rec["secs"]), a.log)
        if a.eval_every and (step % a.eval_every == 0 or step == steps):
            pool_ev = held_idx or train_idx
            k = max(1, int(a.eval_rows))
            if a.eval_rotate and len(pool_ev) > k:
                r0 = ((step // max(1, a.eval_every)) * k) % len(pool_ev)
                ev_idx = [pool_ev[(r0 + j) % len(pool_ev)] for j in range(k)]
            else:
                ev_idx = pool_ev[:k]
            ev = eval_hybrid(model, C, side, allow, ev_idx, dev, a)
            rec["eval"] = ev
            rec["eval_idx"] = ev_idx
            log("EVAL step=%d (rotate=%d, n=%d) %s" % (step, int(a.eval_rotate), len(ev_idx),
                                                       json.dumps(ev, ensure_ascii=False)),
                a.log)
        with open(a.curve, "w", encoding="utf-8") as f:
            for c in curve:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        if step % 300 == 0 or step == steps:      # gen33 [7]: dump a curve sample every 300 steps
            with open(os.path.join(OUT, "DEEP_CURVE_300.jsonl"), "a",
                      encoding="utf-8") as _f3:
                _f3.write(json.dumps({"step": step, "loss": rec["loss"],
                                      "chain": rec["chain"], "slot": rec["slot"],
                                      "text": rec["text"], "gn": rec["gn"],
                                      "lr": rec["lr"], "seen": rec["seen"],
                                      "secs": rec["secs"], "nv": rec["nv"],
                                      "nbf": rec["nbf"], "ntxt": rec["ntxt"],
                                      "plan_md5": pmd5, "steps_total": steps,
                                      "tag": a.tag}, ensure_ascii=False) + "\n")
        if a.snap_every and step % a.snap_every == 0:
            torch.save({"model": model.state_dict(), "step": step, "head": head,
                        "nvals": nvals, "args": vars(a),
                        "gen41": _g41_meta(a, minfo, head, nvals)}, a.out)
    if not os.path.exists(a.out):
        torch.save({"model": model.state_dict(), "step": step, "head": head,
                    "nvals": nvals, "args": vars(a),
                    "gen41": _g41_meta(a, minfo, head, nvals)}, a.out)

    # ---- samples and the five-column table (chain trace + rendering + attribution; same audited path as gen26_eval) ----
    samples, sample_tbl = [], {}
    if a.render_only_eval and held_idx:
        try:
            import argparse as _ap
            import gen34_eval as EV                                   # gen34 [7] fork
            ctx = EV._load_ctx_eval(a)                                # gen30 [6] v3 ctx
            na = _ap.Namespace(arm="hybrid", seg="a_held", args_mode=a.args_mode,
                               max_new=a.sample_max_new, judge=0, naked_base="135M")
            its = [(i, side[i]) for i in held_idx[:5]]
            samples, sample_tbl = EV.run_eval(model, side, allow, its, dev, ctx, na, a.log)
        except Exception as e:                                      # noqa: BLE001
            log("sample generation failed (registered; does not affect training readings): %r" % (e,), a.log)
    sm = {"schema": "ccp-a-hybrid-smoke/2", "tag": a.tag, "run_kind": kind,
          "headline": ("L0 reconnaissance version · corpus coverage 11.7% (1097/9366) · reading is qualitative = "
                       "pipeline shaken + trend signal · not used as the M1_A main reading" if kind == "probe" else "smoke run: mechanical-path connectivity evidence"),
          "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
          "law": "change #74 (hybrid main arm) + change #75 (red line R0) + the gen27 correction order (CRIT-1..LOW-16)",
          "arch": "V32 chain head (step 168 / slot 24x238) + LM rendering head, one body, single-sequence joint training",
          "track": {"body": (a.base or os.path.join(RES, "models", "SmolLM2-135M")),
                    "body_dtype": a.body_dtype,
                    "dual_track_disclosure": "MED-11: this batch runs the 135M track only; the 1.7B track code path exists "
                                             "(CCPModel.dyn=True / cast_body bf16) but is **not wired and not run**, "
                                             "contributing no readings",
                    "params_M": round(minfo["n_params"] / 1e6, 1)},
          "cache": {"path": a.cache, "md5": md5f(a.cache), "n_rows": C.n_rows,
                    "segments": segs, "n_train": len(train_idx), "n_held": len(held_idx)},
          "sidecar": {"path": a.side, "md5": md5f(a.side), "n": len(side)},
          "init": {"path": a.init, "info": {k: v for k, v in minfo.items()
                                            if k in ("n_params", "value_head_rows_copied",
                                                     "ccp_rows_parent_copy", "head", "nvals")}},
          "slot_cond": (model.slot_param_report() if a.slot_cond else
                        {"enabled": False, "note": "gen34 slot scheme off (baseline-equivalent path)"}),
          "plan_attn": (model.pa_param_report() if a.pa_enable else
                        {"enabled": False, "note": "gen30 treatment arm off (baseline-equivalent path)"}),
          "objective": {"form": "L = L_step + (lam_bind/NSLOT)*L_slot + lam_txt*L_text",
                        "lam_bind": a.lam_bind, "lam_txt": a.lam_txt, "NSLOT": NSLOT,
                        "w_tail": a.w_tail, "cache": a.cache,
                        "cache_md5": _md5f36(a.cache),
                        "crit1_fix": "render-segment autoregressive misalignment (label at position p = ids[p+1]); see the MISALIGN unit test"},
          "plan": {"steps": len(plan), "bs": a.bs, "md5": pmd5, "held_hits": int(n_off),
                   "interleave": "3 classes x %d problems/step" % max(1, a.bs // 3)},
          "coverage_med6": {"cls_pool": dict(pools),
                            "seen_unique_final": {c: len(seen[c]) for c in CLASSES},
                            "deviation": "interleaved feeding truncates by the smallest class => a single epoch covers only "
                                         "per x batches rows per class; reshuffling across epochs widens coverage (seen listed in the curve every 10 steps)"},
          "micro": {"token_cap": a.micro_token_cap, "total_micro_batches": micro_cnt},
          "eval_med9": {"eval_rows": a.eval_rows, "rotate": int(a.eval_rotate),
                        "held_pool": len(held_idx)},
          "gpu": ginfo, "steps_done": step, "stopped_nan": stopped_nan,
          "curve": {"path": a.curve, "md5": md5f(a.curve),
                    "first": curve[0] if curve else None,
                    "last": curve[-1] if curve else None,
                    "min": {k: min(c[k] for c in curve) for k in ("loss", "chain", "slot",
                                                                 "text")} if curve else None},
          "has_three_curves": bool(curve and all(k in curve[0] for k in
                                                ("chain", "slot", "text"))),
          "ckpt": {"path": a.out, "md5": md5f(a.out)},
          "samples": samples, "sample_table": sample_tbl,
          "wall_s": round(time.time() - t0, 2)}
    json.dump(sm, open(a.summary, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("SMOKE DONE steps=%d loss %.4f->%.4f chain %.4f->%.4f slot %.4f->%.4f "
        "text %.4f->%.4f wall=%.1fs" %
        (step, curve[0]["loss"], curve[-1]["loss"], curve[0]["chain"], curve[-1]["chain"],
         curve[0]["slot"], curve[-1]["slot"], curve[0]["text"], curve[-1]["text"],
         sm["wall_s"]), a.log)
    print("GEN26_HYBRID_RC=0")


@torch.no_grad()
def eval_hybrid(model, C, side, allow, idx, dev, a, bs=8):
    """Held-out readings: chain top-1 (per frame) + slot hits + render-segment token accuracy (same form, same segments)."""
    model.eval()
    n = min(len(idx), a.eval_rows)
    idx = idx[:n]
    tot = {"n_frames": 0, "sel": 0, "slots_tot": 0, "slots_hit": 0,
           "slots_nonunk_tot": 0, "slots_nonunk_hit": 0, "tok_tot": 0, "tok_hit": 0}
    for i0 in range(0, len(idx), bs):
        ii = list(idx[i0:i0 + bs])
        bc = L32.BatchCpu32().build(C, ii, 0, 0, 0)
        bc.rowIds = list(ii)
        rids, rtgt, ln = render_batch(side, ii, dev)
        lenB_full = [len(C.rows[i].desc) + ln[b] for b, i in enumerate(ii)]
        seq_c = model.assemble(bc.B, bc.L, bc, C, dev)
        seq_r = model.render_embed(rids)
        seq = torch.cat([seq_c, seq_r], 1)
        L = bc.L + int(rids.shape[1])
        cm, cosc, sinc = model.consts(L, dev)
        mk = model.mask_of(cm, bc.B, L, lenB_full, dev)
        H = model.forward_body(seq, mk, cosc, sinc)
        lg, zb, tgt = model.heads(H[:, :bc.L, :].contiguous(), bc, C, dev, allow)
        lt = model.lm.lm_head(H[:, bc.L:, :])
        T = bc.Talloc
        lgc = lg.view(bc.B, T, lg.shape[-1]).argmax(-1).cpu()
        zbc = zb.view(bc.B, T, NSLOT, model.NVALS_C).argmax(-1).cpu()
        tpk = lt.argmax(-1).cpu()
        for r, ri in enumerate(ii):
            R = C.rows[ri]
            for t in range(len(R.labels)):
                gold = int(bc.labels[r * T + t])
                if gold < 0:
                    continue
                tot["n_frames"] += 1
                tot["sel"] += int(int(lgc[r, t]) == gold)
                for j in range(NSLOT):
                    k = t * NSLOT + j
                    if k >= len(R.bsl):
                        break
                    g = R.bsl[k]
                    if g < 0:
                        continue
                    tot["slots_tot"] += 1
                    h = int(int(zbc[r, t, j]) == g)
                    tot["slots_hit"] += h
                    if g != HC.UNK:
                        tot["slots_nonunk_tot"] += 1
                        tot["slots_nonunk_hit"] += h
            for p in range(int(rtgt.shape[1])):
                g = int(rtgt[r, p])
                if g < 0:
                    continue
                tot["tok_tot"] += 1
                tot["tok_hit"] += int(int(tpk[r, p]) == g)
        del H, seq, seq_c, seq_r, lg, zb, lt
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    model.train()
    return {"n_rows": n, "sel_top1": tot["sel"] / max(tot["n_frames"], 1),
            "n_frames": tot["n_frames"],
            "slot_top1": (tot["slots_hit"] / tot["slots_tot"]) if tot["slots_tot"] else None,
            "slot_top1_nonunk": ((tot["slots_nonunk_hit"] / tot["slots_nonunk_tot"])
                                 if tot["slots_nonunk_tot"] else None),
            "n_slots": tot["slots_tot"],
            "render_tok_acc": (tot["tok_hit"] / tot["tok_tot"]) if tot["tok_tot"] else None,
            "n_render_tok": tot["tok_tot"]}


if __name__ == "__main__":
    main()
