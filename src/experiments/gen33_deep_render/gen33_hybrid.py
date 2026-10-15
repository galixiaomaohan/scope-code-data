# -*- coding: utf-8 -*-

# ---- gen33 fork registration (deep-generation V3.5 processing arm) ----
# This file = a **line-by-line fork copy** of gen30_hybrid.py (md5 4f630a796190d510c0b17010fc1b1beb); existing artifacts untouched (fork-copy regime).
# Inherits three parent fixes (**as-is, zero change**): ① LN_q deviation; ② CRIT-30A query gating fix;
#   ③ CRIT-30B vocabulary convention fix.
# Batch-unique differences (see scratch/ccp/gen33/V33_DESIGN.md pre-registration) patch list:
#   [1] paths: added G33; OUT/log/artifacts point at gen33 (write surface)
#   [2] vocabulary convention: default --vocab/--allow point at ccp-vocab/4 (head 194 / value space 352) and ccp_allow_v4.bin
#   [3] cache/sidecar defaults point at the gen33 deep-corpus artifacts (deep_all.bin / DEEP_RENDER_SIDECAR.jsonl)
#   [4] CLASSES 3 -> 2 classes (ind/ineq; the deep corpus has no nt class, D-5)
#   [5] imports gen33_planattn / gen33_eval (forks) instead of gen30_*
#   [6] default --dev cuda:1 (GPU0 user redline: structurally disabled)
# Write surface: only scratch/ccp/gen33/** (gen30/gen26/gen13 artifacts read-only as inputs). Redlines as in the parent.


# ---- gen30 fork registration (processing arm) ----
# This file = a **line-by-line fork copy** of gen26_hybrid.py (md5 d0602916b81567f27324fedbaed4065e); existing artifacts untouched (fork-copy regime).
# The only difference = mounting gen30 plan-sequence self-attention (P1 multi-head execution-aware step-level attention +
#   P2 dynamic operator embedding + P4-lite plan summary), see scratch/ccp/gen30/PLAN_ATTN_DESIGN.md (pre-registered). Patch list:
#   [1] paths: added G30; OUT points at gen30/out (write surface)
#   [2] model class: AttnModel = PA30.make_attn_model(HybridModel); used when --pa-enable=1
#   [3] load order: attach_plan_attn after load_init_ccp (v32lib.load_init asserts on unknown keys)
#   [4] curve gains pa_delta_max/ps_delta_max; summary gains a plan_attn section
#   [5] sample block imports gen30_eval (fork) instead of gen26_eval
# Write surface: only scratch/ccp/gen30/** (gen26 artifacts read-only as inputs). Redlines as in the parent.

# gen26_hybrid.py -- gen26 (latent-space hybrid main arm) T2/T3: joint training harness
#
# Legal sources: constitution change 74 (arm A = latent-space hybrid architecture: V32 protocol-head chain of thought + LM-head full-proof rendering)
#       + change 75 (redline R0: the latent-space thought mainline must not drift; plain-text direct SFT is barred as the main arm).
#
# Architecture (registered; item-by-item match with design doc HYBRID_DESIGN.md §1/§2):
#   sequence = [chain segment: V32 protocol frames (statement token frames + frame-no/pst/op/res frames, gen13 semantics bit-for-bit)]
#          ++ [render segment: (K_EMB, M_TASK) boundary frame + render prompt tokens + proof_text tokens]
#   Both heads share one SmolLM2 body (single sequence, one forward pass, causal attention):
#     * chain head (step_head 168 rows / value_head 239 cols / slot_emb 24 slots) reads chain-segment hidden states -> chain loss + slot loss
#     * LM head (self.lm.lm_head) reads render-segment hidden states -> render-segment token CE (target span only)
#   Joint objective (**pre-registered**, change 74 T1②):  L = L_step + (lam_bind/NSLOT)·L_slot + lam_txt·L_text
#     chain-segment loss follows gen13 formula-for-formula (v32ccp_train.CCPModel.loss_f, unchanged); L_text = render-segment CE.
#
# Reuse (imports-only, zero modification):
#   * scratch/ccp/gen13/v32ccp_train.py -> CCPModel / CacheCCP (chain head/body/frame assembly, 135M+1.7B dual-track convention)
#   * cppcuda/tools/v32b_train.py       -> L32 / read_plan / micro_split / _a
#   * cppcuda/tools/v31_model.py        -> body/frame kind/mask (indirectly via v32lib)
#   * gen13/CCP_VOCAB.json + ccp_allow_ext.bin -> operator space 168 / value space 239 / allow table
#   * scratch/ccp/gen26/gen26_hybuild.py -> hybrid cache + render sidecar (produced by this batch)
# Interleaved-feeding convention = gen23_sft.interleaved_batches semantics (reimplemented here, provenance registered;
#   reason: change 74 authorizes only read-only references to in-flight batches, so no hard dependency on the instantaneous state of gen23 files).
#
# Write surface: only scratch/ccp/gen26/**. GPU: single card after the exclusivity gate (free > 4500MiB).
import os
import sys

# Determinism: CUBLAS workspace must be set before torch initializes CUDA (same as gen13)
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
G33 = os.path.join(CCP, "gen33")            # gen33 fork write surface
OUT = os.path.join(G33, "out")
os.makedirs(OUT, exist_ok=True)
for _p in (RES, G13, os.path.join(RES, "cppcuda", "tools"), os.path.join(RES, "skel32"),
           G26, G30, G33, os.environ.get("SCOPE_SUPEROPERATOR", "../device")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
CLASSES = ("ind", "ineq")         # D-5: the deep corpus has only ind/ineq classes (no nt)
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
# Model: hybrid arm = CCP chain head + LM render head (one body, single sequence)
# ---------------------------------------------------------------------------

class HybridModel(CCPModel):
    """V32 protocol chain (inherited bit-for-bit from gen13) + LM render head, single-sequence joint forward."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.lam_bind = 1.0
        self.lam_txt = 1.0

    # ---- render-segment payload ----
    def render_embed(self, rids):
        """(B, Lr) token ids -> (B, Lr, HID) body embeddings (same embedding matrix as chain-segment token frames)."""
        B, Lr = rids.shape
        e = self.lm.model.embed_tokens.weight.index_select(0, rids.reshape(-1))
        e = e.view(B, Lr, self.HID_D)
        if getattr(self, "body_dtype", None) is not None and self.body_dtype != torch.float32:
            e = e.to(self.body_dtype)
        return e

    def forward_joint(self, bc, C, dev, allow, rids, rtgt, lenB_full, grad_ckpt=False):
        """Single-microbatch joint forward -> (loss, ls, lb, lt, nv, nbf, ntxt, Lfull)"""
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
        # ---- render segment: loss on the target span only (prompt/boundary frames = -100) ----
        lgt = self.lm.lm_head(H[:, Lc:, :])
        t = rtgt.reshape(-1)
        ntxt = int((t != -100).sum())
        lt = F.cross_entropy(lgt.reshape(-1, lgt.shape[-1]), t,
                             ignore_index=-100, reduction="sum") / max(ntxt, 1)
        total = loss + self.lam_txt * lt
        nv = int((torch.tensor(bc.labels) != -100).sum())
        return total, ls, lb, lt, nv, nbf, ntxt, L

    def trainable_groups(self):
        """Three parameter groups (self-check: each of the three losses has a gradient outlet)."""
        return {"chain_step": ["step_head.weight", "step_head.bias"],
                "chain_slot": ["value_head.weight", "value_head.bias", "slot_emb.weight"],
                "text": ["lm.lm_head.weight"]}


# ---------------------------------------------------------------------------
# gen30 processing arm: plan-sequence self-attention (P1+P2) + plan summary (P4-lite)
# ---------------------------------------------------------------------------
import gen33_planattn as PA33                                        # noqa: E402

AttnModel = PA33.make_attn_model(HybridModel)


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
    """-> (rids, rtgt, ln_real) -- render segment = [M_TASK]+prompt+target; target-span loss only."""
    Lr = max(side[i]["n"] for i in idx)
    B = len(idx)
    rids = torch.full((B, Lr), pad_id, dtype=torch.long)
    rtgt = torch.full((B, Lr), -100, dtype=torch.long)
    ln = []
    for b, i in enumerate(idx):
        s = side[i]
        ids = s["ids"]
        rids[b, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        # CRIT-1 fix (gen27 review): autoregressive misalignment -- the label at position p must be the **next**
        #   token ids[p+1], i.e. target ids[np:] land in [np-1, len-1). The original rtgt[np:len]=ids[np:] copies the
        #   input (a trivial shortcut, worse under tie_word_embeddings), and leaves position np-1 always -100 at inference.
        np_ = int(s["np"])
        rtgt[b, np_ - 1:len(ids) - 1] = torch.tensor(ids[np_:], dtype=torch.long)
        ln.append(len(ids))
    return rids.to(dev), rtgt.to(dev), ln


def micro_split_hybrid(idx, C, side, cap):
    """Split microbatches by padding load B*L (L = chain length + render length); ascending total length within a batch."""
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
    """Interleaved feeding (convention = gen23_sft.interleaved_batches: all three classes present in every batch, in-class shuffle without replacement).
    provenance: newly written in gen26 (registered); semantically equivalent to the same-named function in gen23_sft.py."""
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
    """Pre-registered batch order: reshuffled every epoch; epochs cycle when steps run short (registered). Returns (plan, plan_md5)."""
    pool = collections.defaultdict(list)
    for i in train_idx:
        pool[_cls_of(C, i)].append(i)
    missing = [c for c in CLASSES if not pool.get(c)]
    if missing:
        raise SystemExit("build_plan: 类池为空 %s (cls 映射未设/错) 池键=%s"
                         % (missing, sorted(pool)))
    plan, epoch = [], 0
    while len(plan) < steps:
        for grp in interleaved_batches(pool, bs=bs, seed=seed + 1000 * epoch):
            plan.append(list(grp))
            if len(plan) >= steps:
                break
        epoch += 1
    plan = plan[:steps]
    assert len(plan) == steps, ("批序生成不足", len(plan), steps)
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
# Main training loop
# ---------------------------------------------------------------------------

def gpu_gate(dev, logp):
    import subprocess
    idx = dev.split(":")[1] if ":" in dev else "0"
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                        "--format=csv,noheader,nounits", "-i", idx],
                       capture_output=True, text=True).stdout.strip()
    used, total = [int(x) for x in q.split(",")[1:3]]
    free = total - used
    log("GPU GATE dev=%s used=%dMiB free=%dMiB (require >4500MiB)" % (dev, used, free), logp)
    if free <= 4500:
        raise SystemExit("GPU 独占门未过: free=%dMiB" % free)
    return {"dev": dev, "used": used, "total": total, "free": free}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(G33, "cache", "deep_all.bin"))
    ap.add_argument("--side", default=os.path.join(OUT, "DEEP_RENDER_SIDECAR.jsonl"))
    ap.add_argument("--init", default="ckpt/sla_v31f.pt")
    ap.add_argument("--lat31", default="cppcuda/cache/lat_tab_v31.bin")
    ap.add_argument("--allow", default=os.path.join(G33, "cache", "ccp_allow_v4.bin"))
    ap.add_argument("--vocab", default=os.path.join(G33, "cache", "CCP_VOCAB_V4.json"))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--smoke", type=int, default=0, help=">0 = smoke-run steps (100)")
    ap.add_argument("--run-kind", default="", choices=["", "smoke", "probe", "formal"],
                    help="naming-gate category (empty = inferred from --smoke); probe = L0 pathfinding version")
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lam-bind", type=float, default=1.0)
    ap.add_argument("--lam-txt", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--grad-ckpt", type=int, default=1)
    ap.add_argument("--micro-token-cap", type=int, default=9000)
    ap.add_argument("--det-train", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--dev", default="cuda:1")
    ap.add_argument("--logevery", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-rows", type=int, default=48, help="MED-9: default 48 (held-out pool 236)")
    ap.add_argument("--eval-rotate", type=int, default=1, help="MED-9: rotate the slice at every evaluation point")
    ap.add_argument("--snap-every", type=int, default=100)
    ap.add_argument("--stopnan", type=int, default=1)
    ap.add_argument("--args-mode", default="bridge", choices=["bridge", "gold"],
                    help="HIGH-2: samples/eval use bridge (device argument derivation); gold is diagnostic only")
    ap.add_argument("--sample-max-new", type=int, default=160)
    ap.add_argument("--body-dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--base", default=None)
    ap.add_argument("--out", default=os.path.join(G33, "ckpt", "deep_smoke100.pt"))
    ap.add_argument("--curve", default=os.path.join(OUT, "DEEP_CURVE.jsonl"))
    ap.add_argument("--plan-out", default=os.path.join(OUT, "batch_idx_deep.txt"))
    ap.add_argument("--tag", default="", help="report tag (e.g. l0_probe)")
    ap.add_argument("--samples", default=os.path.join(OUT, "DEEP_SAMPLES.json"))
    ap.add_argument("--summary", default=os.path.join(OUT, "DEEP_SUMMARY.json"))
    ap.add_argument("--log", default=os.path.join(G33, "logs", "gen33_hybrid.log"))
    ap.add_argument("--render-only-eval", type=int, default=1)
    ap.add_argument("--pa-enable", type=int, default=1, help="gen30: P1/P2 step-level attention")
    ap.add_argument("--pa-seed", type=int, default=-1, help="gen30: new-module seed (-1 = --seed)")
    ap.add_argument("--pa-state", type=int, default=1, help="gen30: device channel (0 = nostate)")
    ap.add_argument("--plan-summary", type=int, default=1, help="gen30: P4-lite plan summary")
    ap.add_argument("--pa-capture", type=int, default=0)
    ap.add_argument("--pa-heads", type=int, default=2)
    ap.add_argument("--pa-dhead", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    smoke = a.smoke > 0
    steps = a.smoke if smoke else a.steps
    kind = a.run_kind or ("smoke" if smoke else "formal")
    bn = os.path.basename(a.out)
    other = [k for k in ("smoke", "probe", "formal") if k != kind]
    if kind not in bn:
        raise SystemExit("HIGH-5/命名闸: 产物名必须含 run-kind '%s': %s" % (kind, bn))
    if any(k in bn for k in other):
        raise SystemExit("HIGH-5/命名闸: 产物名不得含 %s: %s" % (other, bn))
    t0 = time.time()
    dev = torch.device(a.dev)
    ginfo = gpu_gate(a.dev, a.log) if str(a.dev).startswith("cuda") else {}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    if a.det_train:
        torch.use_deterministic_algorithms(True)

    voc = json.load(open(a.vocab, encoding="utf-8"))
    head, nvals = voc["head"], voc["n_vals_ext"]
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
        cls_by_row[int(i)] = s["cls"]
    set_cls_map(cls_by_row, range(C.n_rows))
    log("cache %s rows=%d segs=%s train=%d held=%d sidecar=%d" %
        (a.cache, C.n_rows, segs, len(train_idx), len(held_idx), len(side)), a.log)
    plan, pmd5 = build_plan(train_idx, C, steps, a.bs, a.seed, snap_path=a.plan_out)
    n_off = sum(1 for rr in plan for i in rr if i not in set(train_idx))
    assert n_off == 0, "批序命中非训练行 %d 个 (留出/越界)" % n_off
    log("batch order: %d steps bs=%d md5=%s (interleaved, 3 classes present; 0 held-out hits)" % (len(plan), a.bs, pmd5),
        a.log)
    pools = collections.Counter(_cls_of(C, i) for i in train_idx)
    log("MED-6 sampler-coverage record: class pools=%s | batches per epoch=%d (limited by the smallest class), bs=%d, in-class per=%d "
        "=> one epoch covers only %d rows per class (reshuffling across epochs widens coverage; coverage tracked in the curve)"
        % (dict(pools), min(pools.values()) // max(1, a.bs // 3), a.bs, max(1, a.bs // 3),
           max(1, a.bs // 3)), a.log)

    torch.manual_seed(a.seed)
    MCLS = AttnModel if a.pa_enable else HybridModel
    model = MCLS(base=a.base, lat_tab_path=a.lat31, head=head, nvals=nvals,
                 parents=parents)
    if a.body_dtype == "bf16":
        model.cast_body(torch.bfloat16)
    minfo = model.load_init_ccp(a.init)
    if a.pa_enable:                              # gen30 [3]: load order (after init)
        _paseed = int(a.pa_seed) if a.pa_seed >= 0 else int(a.seed)
        model.attach_plan_attn(seed=_paseed, capture=bool(a.pa_capture), enable=True,
                               arch={"use_state": bool(a.pa_state),
                                     "n_heads": int(a.pa_heads), "d_head": int(a.pa_dhead)},
                               plan_summary=bool(a.plan_summary))
        _parep = model.pa_param_report()
        assert _parep["within_budget"], _parep
        log("gen30 P1/P2/P4-lite: %s" % json.dumps(_parep, ensure_ascii=False), a.log)
    model.lam_bind, model.lam_txt = a.lam_bind, a.lam_txt
    model = model.to(dev)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    n_train_p = sum(p.numel() for p in params)
    n_lm = sum(p.numel() for p in model.lm.parameters())
    log("model: total %.1fM / trainable %.1fM (body %.1fM) | head=%d vals=%d | body=%s dev=%s | "
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
            rids, rtgt, ln = render_batch(side, mb, dev)
            lenB_full = [len(C.rows[i].desc) + ln[b] for b, i in enumerate(mb)]
            out = model.forward_joint(bc, C, dev, allow, rids, rtgt, lenB_full,
                                      grad_ckpt=bool(a.grad_ckpt))
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
        if step % 300 == 0 or step == steps:      # gen33 [7]: persist a curve sample every 300 steps
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
                        "nvals": nvals, "args": vars(a)}, a.out)
    if not os.path.exists(a.out):
        torch.save({"model": model.state_dict(), "step": step, "head": head,
                    "nvals": nvals, "args": vars(a)}, a.out)

    # ---- samples and the five-column table (chain trace + render + attribution; same post-audit path as gen26_eval) ----
    samples, sample_tbl = [], {}
    if a.render_only_eval and held_idx:
        try:
            import argparse as _ap
            import gen33_eval as EV                                   # gen33 [5] fork
            ctx = EV._load_ctx_eval(a)                                # gen30 [6] v3 ctx
            na = _ap.Namespace(arm="hybrid", seg="a_held", args_mode=a.args_mode,
                               max_new=a.sample_max_new, judge=0, naked_base="135M")
            its = [(i, side[i]) for i in held_idx[:5]]
            samples, sample_tbl = EV.run_eval(model, side, allow, its, dev, ctx, na, a.log)
        except Exception as e:                                      # noqa: BLE001
            log("sample generation failed (registered; training readings unaffected): %r" % (e,), a.log)
    sm = {"schema": "ccp-a-hybrid-smoke/2", "tag": a.tag, "run_kind": kind,
          "headline": ("L0 探路版·语料覆盖 11.7% (1097/9366)·读数定性=管线 shaken+趋势信号"
                       "·不作 M1_A 主读数" if kind == "probe" else "冒烟: 机械通路连通证据"),
          "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
          "law": "变更 74 (混合主臂) + 变更 75 (红线 R0) + gen27 修正令 (CRIT-1..LOW-16)",
          "arch": "V32 链头 (step 168 / slot 24x238) + LM 渲染头, 同体单序列联合训练",
          "track": {"body": (a.base or os.path.join(RES, "models", "SmolLM2-135M")),
                    "body_dtype": a.body_dtype,
                    "dual_track_disclosure": "MED-11: 本批只跑 135M 轨; 1.7B 轨代码路径存在 "
                                             "(CCPModel.dyn=True / cast_body bf16) 但**未接未跑**, "
                                             "不作任何读数",
                    "params_M": round(minfo["n_params"] / 1e6, 1)},
          "cache": {"path": a.cache, "md5": md5f(a.cache), "n_rows": C.n_rows,
                    "segments": segs, "n_train": len(train_idx), "n_held": len(held_idx)},
          "sidecar": {"path": a.side, "md5": md5f(a.side), "n": len(side)},
          "init": {"path": a.init, "info": {k: v for k, v in minfo.items()
                                            if k in ("n_params", "value_head_rows_copied",
                                                     "ccp_rows_parent_copy", "head", "nvals")}},
          "plan_attn": (model.pa_param_report() if a.pa_enable else
                        {"enabled": False, "note": "gen30 处理臂未开 (基线等价路径)"}),
          "objective": {"form": "L = L_step + (lam_bind/NSLOT)*L_slot + lam_txt*L_text",
                        "lam_bind": a.lam_bind, "lam_txt": a.lam_txt, "NSLOT": NSLOT,
                        "crit1_fix": "渲染段自回归错位 (位置 p 标签 = ids[p+1]); 见单测 MISALIGN"},
          "plan": {"steps": len(plan), "bs": a.bs, "md5": pmd5, "held_hits": int(n_off),
                   "interleave": "3 类 × %d 题/步" % max(1, a.bs // 3)},
          "coverage_med6": {"cls_pool": dict(pools),
                            "seen_unique_final": {c: len(seen[c]) for c in CLASSES},
                            "deviation": "交错混喂按最小类截断 ⇒ 单 epoch 只覆盖各类 per×批数 行; "
                                         "跨 epoch 重洗扩大覆盖 (seen 列入曲线逐 10 步)"},
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
    """Held-out reading: chain top-1 (per frame) + slot hits + render-segment token accuracy (same formula, same segment)."""
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
