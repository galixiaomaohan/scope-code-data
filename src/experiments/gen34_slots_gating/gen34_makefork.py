# -*- coding: utf-8 -*-
# gen34_makefork.py -- gen34 fork bench: derive the four gen34 files from the four gen33
#                     files by line-by-line forking
#
# Discipline: **exact replacement only, parents untouched** (fork-copy rule; the parent md5
#   ledger lives in GEN34_FORK_LEDGER.json). Every patch must hit exactly once, otherwise
#   the whole build is refused (never silent).
# Usage: python gen34_makefork.py
#
# NOTE: the patch match keys below are byte-for-byte the gen33 parent text as it stood at
#   fork time (their original comments included), except that the host-specific
#   superoperator path was replaced with the SCOPE_SUPEROPERATOR lookup on both sides; the
#   inserted blocks carry the post-cleaning English wording. The bench is a provenance
#   record of the fork operation, not a re-runnable build against this package.
import hashlib
import os

import gen34_common as C

G33, G34 = C.G33, C.G34

HDR_HYB = """
# ---- gen34 fork registration (integer-rendering push · A/B arm processor) ----
# This file is a **line-by-line fork copy** of gen33_hybrid.py (md5 %(md5)s); pre-existing
# files untouched (fork-copy rule).
# All parent fixes are inherited (**as-is, zero edits**): LN_q deviation · CRIT-30A query
#   gating · CRIT-30B vocab convention · gen33 patches [1]-[7] (v4 vocab / deep-corpus
#   sidecar / 2 classes / fork imports / cuda:1 default).
# This batch's only differences (pre-registered in scratch/ccp/gen34/V34_DESIGN.md §2)
# patch list:
#   [0] line endings: the parent is CRLF, this file is read in text mode -> LF-normalized
#       (the only non-logic difference; ledger.parent_crlf)
#   [1] paths: new G34; OUT/log/artifacts point at gen34 (write surface); G33 stays on
#       sys.path (read-only parent)
#   [2] **slot-conditioned render head** (V34_DESIGN §2): slot embedding
#       slot_render_emb(8,d) ~ N(0,.02^2)
#       + gate g_pos=0 (**naming**: the parent CCPModel/SLANavV32 already owns `slot_emb`
#       = the chain-head 24-slot table (24,dmodel); this batch's slot embedding **must** be
#       renamed slot_render_emb, otherwise it overwrites the chain-head slot table =>
#       heads() breaks)
#       + gate g_dev=0 + constant beta; with g==0 the increment is exactly +0.0 => bitwise
#       equivalent to the baseline (an identity, not a floating-point coincidence)
#   [3] `render_batch_slots` + `load_slot_sidecar`: draw slots from the gen34 slot sidecar
#       (DEEP_SLOT_SIDECAR)
#   [4] render loss **grouped** into slot / non-slot (lam_txt / lam_slot, default 3.0);
#       with `slot_kind is None` the path is **verbatim** the parent's (C-arm equivalence
#       decided bitwise by [3])
#   [5] `--slot-cond` / `--slot-dev` / `--slot-side` / `--slot-beta` / `--lam-slot` CLI
#   [6] `--slot-cond 1` **combined with** `--pa-enable 1` is **not wired** (planattn brings
#       its own forward_joint that masks the slot branch) => structurally refused with
#       SystemExit, never silently degraded. All A/B arms in this batch run --pa-enable 0
#   [7] import gen34_planattn / gen34_eval (forks)
#   [8] curve adds slot_g_pos / slot_g_dev / lt_slot (measured gate values = mechanism
#       readings)
# Write surface: scratch/ccp/gen34/** only (gen33/gen32/gen13 artifacts are read-only
# input). Red lines as in the parent.
"""

HDR_EVAL = """
# ---- gen34 fork registration (integer-rendering push · exam artifact) ----
# This file is a **line-by-line fork copy** of gen33_eval.py (md5 %(md5)s); pre-existing
# files untouched (fork-copy rule).
# All parent patches are inherited (v4 vocab / device-live convention / WALK_BUDGET=12 /
# 9 new-op keywords / fork imports).
# This batch's only differences (V34_DESIGN §2.2) patch list:
#   [0] line endings: the parent is CRLF, this file is read in text mode -> LF-normalized
#       (the only non-logic difference; ledger.parent_crlf)
#   [1] paths: new G34; OUT/log point at gen34 (write surface); G33 kept (read-only parent)
#   [2] `render_greedy_slot`: **skeleton-anchored device supply** -- the device re-renders
#       the row's own payload (reads only sfa/vars/dom_dump, never the gold target) -> device
#       text + content-integer token bitmap; once the generated tokens equal the device
#       tokens at **every non-slot position**, inject `g_dev * beta * onehot(device token)`
#       at the slot; on desynchronization inject nothing and register it by counter
#       (the skeleton has already drifted => that problem cannot pass anyway, so injection
#       is pointless)
#   [3] `--slot-cond` / `--slot-dev` / `--slot-beta` / `--lam-slot` CLI + strict slot-param
#       load assertion
#   [4] per-row **content-integer accuracy** M1_E-int (naive + content conventions; gold
#       comes from the exam artifact or the corpus depending on seg)
#   [5] import gen34_hybrid / gen34_planattn (forks) instead of gen33_*
# Write surface: scratch/ccp/gen34/** only. Red lines as in the parent.
"""

HDR_PA = """
# ---- gen34 fork registration (integer-rendering push) ----
# This file is a **pure-logic copy** of gen33_planattn.py (md5 %(md5)s): zero patches
# (character-for-character identical).
#   The parent uses CRLF line endings; this file is read in text mode -> LF-normalized
#   (the only difference = line endings; see ledger.parent_crlf).
#   Both A/B arms in this batch run with `--pa-enable 0` (attention off); this file exists
#   only for fork completeness (closing the import chain).
"""

HDR_JUDGE = """
# ---- gen34 fork registration (integer-rendering push · judging artifact) ----
# This file is a **line-by-line fork copy** of gen33_judge.py (md5 %(md5)s); pre-existing
# files untouched (fork-copy rule).
# Only differences: [1] write surface/artifact/log root G33 -> G34; [2] EVAL artifact prefix
#   GEN33_EVAL_ -> GEN34_EVAL_; [3] default --tags point at the three gen34 arms
#   (slotA,slotB,base).
# The judging kernel (frozen drct_common.judge_attempt) is **untouched**; its frozen-md5
# gate is inherited.
"""

# ===========================================================================
# gen34_hybrid.py patches
# ===========================================================================

SLOT_BLOCK = '''
# ---------------------------------------------------------------------------
# gen34 [8]: integer-slot conditioner on the render head (V34_DESIGN §2)
# ---------------------------------------------------------------------------
n_slot_kinds = 8          # 7 classes (V34_DESIGN §1.3) + 1 padding


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

    The slot sidecar gives the **target token's absolute id index** a; the parent convention
    is `rtgt[b, p] = ids[p+1]` => that token's label position is p = a - 1.
    Returns slot_tok/slot_kind of shape (B, Lr), non-slot = -1.
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
    """Main hybrid arm + integer-slot conditioner on the render head (V34_DESIGN §2).

    A arm (slot_dev=1) = slot embedding + slot loss + **device one-hot gate fusion**;
    B arm (slot_dev=0) = slot embedding + slot loss (LM fills slots itself, no device query).
    Both arms have the **same parameter count** (E_slot/g_pos/g_dev all present); only the
    g_dev channel is structurally inactive => A-B = the device's net contribution.
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
            log("gen34 slot conditioning: kinds=%d beta=%g dev=%s lam_slot=%g | "
                "E_slot ~N(0,.02^2) g_pos=g_dev=0.0 (increment at the start is exactly "
                "+0.0 => PA-EQ identity)"
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
        """Slot render forward -> (L_render, lt_main, lt_slot, n_slot)."""
        m = (slot_kind >= 0)
        Hr2 = Hr
        if bool(m.any()):
            Hr2 = Hr + (self.g_pos * self.slot_render_emb(slot_kind.clamp(min=0))) * \\
                m.unsqueeze(-1).to(Hr.dtype)
        lgt = self.lm.lm_head(Hr2)
        t = rtgt.reshape(-1)
        if self.slot_dev and slot_tok is not None:
            dm = m & (slot_tok >= 0)
            if bool(dm.any()):
                oh = F.one_hot(slot_tok.clamp(min=0), lgt.shape[-1]).to(lgt.dtype)
                # oh (B,Lr,V) * dm (B,Lr,1) => (B,Lr,V); the mask must stay 3-D (a
                # reshape(-1) once broke the broadcast)
                add = ((self.g_dev * self.slot_beta) * oh) * dm.unsqueeze(-1).to(lgt.dtype)
                lgt = lgt + add
        lg = lgt.reshape(-1, lgt.shape[-1])
        mm = m.reshape(-1)
        sel_s = mm & (t != -100)
        sel_m = (~mm) & (t != -100)
        n_slot = int(sel_s.sum())
        lt_main = F.cross_entropy(lg[sel_m], t[sel_m], reduction="sum") \\
            / max(int(sel_m.sum()), 1)
        lt_slot = (F.cross_entropy(lg[sel_s], t[sel_s], reduction="sum") / max(n_slot, 1)) \\
            if n_slot else torch.zeros((), device=lg.device, dtype=lg.dtype)
        self._lt_slot_last = float(lt_slot.detach())
        return self.lam_txt * lt_main + self.lam_slot * lt_slot, lt_main, lt_slot, n_slot
'''

PATCH_HYB = [
    ('''G33 = os.path.join(CCP, "gen33")            # gen33 fork 写面
OUT = os.path.join(G33, "out")
os.makedirs(OUT, exist_ok=True)
for _p in (RES, G13, os.path.join(RES, "cppcuda", "tools"), os.path.join(RES, "skel32"),
           G26, G30, G33, os.environ.get("SCOPE_SUPEROPERATOR",
                                         os.path.join(RES, os.pardir, "superoperator"))):''',
     '''G33 = os.path.join(CCP, "gen33")            # parent (read-only; fork ledger)
G34 = os.path.join(CCP, "gen34")            # gen34 fork write surface
OUT = os.path.join(G34, "out")
os.makedirs(OUT, exist_ok=True)
for _p in (RES, G13, os.path.join(RES, "cppcuda", "tools"), os.path.join(RES, "skel32"),
           G26, G30, G33, G34, os.environ.get("SCOPE_SUPEROPERATOR",
                                              os.path.join(RES, os.pardir,
                                                           "superoperator"))):'''),
    ("import gen33_planattn as PA33                                        # noqa: E402",
     "import gen34_planattn as PA33                                        # noqa: E402"),
    ("AttnModel = PA33.make_attn_model(HybridModel)",
     "AttnModel = PA33.make_attn_model(HybridModel)\n" + SLOT_BLOCK),
    ('''    def forward_joint(self, bc, C, dev, allow, rids, rtgt, lenB_full, grad_ckpt=False):
        """单微批联合前向 -> (loss, ls, lb, lt, nv, nbf, ntxt, Lfull)"""''',
     '''    def forward_joint(self, bc, C, dev, allow, rids, rtgt, lenB_full, grad_ckpt=False,
                      slot_tok=None, slot_kind=None):                # gen34 [8]
        """Single-microbatch joint forward -> (loss, ls, lb, lt, nv, nbf, ntxt, Lfull)"""'''),
    ('''        # ---- 渲染段: 仅 target 段计损 (prompt/边界帧 = -100) ----
        lgt = self.lm.lm_head(H[:, Lc:, :])
        t = rtgt.reshape(-1)
        ntxt = int((t != -100).sum())
        lt = F.cross_entropy(lgt.reshape(-1, lgt.shape[-1]), t,
                             ignore_index=-100, reduction="sum") / max(ntxt, 1)
        total = loss + self.lam_txt * lt''',
     '''        # ---- render segment: only the target part contributes loss (prompt/boundary
        #      frames = -100) ----
        t = rtgt.reshape(-1)
        ntxt = int((t != -100).sum())
        if getattr(self, "slot_cond", False) and slot_kind is not None:
            # gen34 [8]/[4]: slot vs non-slot **grouped** loss (slots go through the
            # device gate-fused logits)
            ltr, lt_main, _lts, _ns = self.render_slot_forward(
                H[:, Lc:, :], rtgt, slot_tok, slot_kind)
            nv = int((torch.tensor(bc.labels) != -100).sum())
            return loss + ltr, ls, lb, lt_main, nv, nbf, ntxt, L
        lgt = self.lm.lm_head(H[:, Lc:, :])
        lt = F.cross_entropy(lgt.reshape(-1, lgt.shape[-1]), t,
                             ignore_index=-100, reduction="sum") / max(ntxt, 1)
        total = loss + self.lam_txt * lt'''),
    ("    set_cls_map(cls_by_row, range(C.n_rows))",
     "    set_cls_map(cls_by_row, range(C.n_rows))\n"
     "    slot_on = int(a.slot_cond)            # gen34 [5] placed after train_idx/held\n"
     "    sslot = load_slot_sidecar(a.slot_side) if slot_on else None\n"
     "    if slot_on:\n"
     "        n_ok = sum(1 for i in train_idx if (sslot.get(i) or {}).get(\"slots\"))\n"
     "        log(\"gen34 slot sidecar: %s rows=%d (training segment with slots=%d/%d)\"\n"
     "            % (a.slot_side, len(sslot), n_ok, len(train_idx)), a.log)\n"
     "        assert n_ok == len(train_idx), (n_ok, len(train_idx))"),
    ("    MCLS = AttnModel if a.pa_enable else HybridModel",
     "    if a.slot_cond and a.pa_enable:\n"
     "        raise SystemExit(\"gen34 [6]: --slot-cond 1 combined with --pa-enable 1 is "
     "not wired \"\n"
     "                         \"(planattn brings its own forward_joint that masks the slot "
     "branch); never silently degraded\")\n"
     "    MCLS = (SlotHybridModel if a.slot_cond\n"
     "            else (AttnModel if a.pa_enable else HybridModel))"),
    ("    model.lam_bind, model.lam_txt = a.lam_bind, a.lam_txt",
     "    if a.slot_cond:                    # gen34 [2] (after init, same order as planattn)\n"
     "        model.enable_slot(dev_flag=bool(a.slot_dev), beta=a.slot_beta,\n"
     "                          lam_slot=a.lam_slot, seed=a.seed, logp=a.log)\n"
     "    model.lam_bind, model.lam_txt = a.lam_bind, a.lam_txt"),
    ("            rids, rtgt, ln = render_batch(side, mb, dev)",
     "            _r, _t, ln, _stok, _skind = render_batch_slots(side, sslot, mb, dev,\n"
     "                                                          slot_on=slot_on)\n"
     "            rids, rtgt = _r, _t"),
    ("            out = model.forward_joint(bc, C, dev, allow, rids, rtgt, lenB_full,\n"
     "                                      grad_ckpt=bool(a.grad_ckpt))",
     "            out = model.forward_joint(bc, C, dev, allow, rids, rtgt, lenB_full,\n"
     "                                      grad_ckpt=bool(a.grad_ckpt),\n"
     "                                      slot_tok=_stok, slot_kind=_skind)"),
    ('    ap.add_argument("--pa-dhead", type=int, default=64)',
     '    ap.add_argument("--pa-dhead", type=int, default=64)\n'
     '    ap.add_argument("--slot-cond", type=int, default=0,\n'
     '                    help="gen34: slot-conditioned render head")\n'
     '    ap.add_argument("--slot-dev", type=int, default=1,\n'
     '                    help="gen34: 1=device one-hot gate fusion (A arm); '
     '0=LM fills slots itself (B arm)")\n'
     '    ap.add_argument("--slot-side", default=os.path.join(G34, "out",\n'
     '                                                       "DEEP_SLOT_SIDECAR.jsonl"))\n'
     '    ap.add_argument("--slot-beta", type=float, default=8.0)\n'
     '    ap.add_argument("--lam-slot", type=float, default=3.0)'),
    ("            import gen33_eval as EV                                   # gen33 [5] fork",
     "            import gen34_eval as EV                                   # gen34 [7] fork"),
    ('          "plan_attn": (model.pa_param_report() if a.pa_enable else',
     '          "slot_cond": (model.slot_param_report() if a.slot_cond else\n'
     '                        {"enabled": False, "note": "gen34 slot scheme off '
     '(baseline-equivalent path)"}),\n'
     '          "plan_attn": (model.pa_param_report() if a.pa_enable else'),
    ('''               "pa_delta_max": round(float(getattr(model, "pa_delta_max", 0.0)), 6),
               "ps_delta_max": round(float(getattr(model, "ps_delta_max", 0.0)), 6),''',
     '''               "pa_delta_max": round(float(getattr(model, "pa_delta_max", 0.0)), 6),
               "ps_delta_max": round(float(getattr(model, "ps_delta_max", 0.0)), 6),
               "slot_g_pos": (round(float(getattr(model, "g_pos", torch.zeros(1)).detach()), 8)
                              if a.slot_cond else None),
               "slot_g_dev": (round(float(getattr(model, "g_dev", torch.zeros(1)).detach()), 8)
                              if a.slot_cond else None),
               "lt_slot": round(float(getattr(model, "_lt_slot_last", 0.0)), 5),'''),
    ('    plan, pmd5 = build_plan(train_idx, C, steps, a.bs, a.seed, snap_path=a.plan_out)',
     '    plan, pmd5 = build_plan(train_idx, C, steps, a.bs, a.seed, snap_path=a.plan_out)\n'
     '    log("gen34 batch-order md5=%s steps=%d bs=%d seed=%d (must match across A/B arms)"\n'
     '        % (pmd5, len(plan), a.bs, a.seed), a.log)'),
]

# ===========================================================================
# gen34_eval.py patches
# ===========================================================================

SLOT_EVAL_BLOCK = '''
# ---------------------------------------------------------------------------
# gen34 [2]: skeleton-anchored device supply (V34_DESIGN §2.2) + content-integer accuracy
#            (M1_E-int)
# ---------------------------------------------------------------------------
import re as _re34        # module-level block with its own import (parent import table
                          # untouched)
_RE_NUM2 = _re34.compile(r"[0-9]+")


def _content_spans(text):
    """Content-integer spans (V34_DESIGN §1.2): [0-9]+ whose neighbouring characters are
    not letters/underscores."""
    out, t = [], text or ""
    for m in _RE_NUM2.finditer(t):
        a, b = m.start(), m.end()
        if a > 0 and (t[a - 1].isalnum() or t[a - 1] == "_"):
            continue
        if b < len(t) and (t[b].isalnum() or t[b] == "_"):
            continue
        out.append((a, b))
    return out


def lits_text(text):
    """-> (naive, content) two lists of literal strings (naive = gen33 §3b convention,
    bitwise comparable)."""
    naive = [m.group(0) for m in _RE_NUM2.finditer(text or "")]
    t = text or ""
    return naive, [t[a:b] for (a, b) in _content_spans(t)]


def _pair_rate(a, b):
    m = min(len(a), len(b))
    h = sum(1 for i in range(m) if a[i] == b[i])
    return h, max(len(a), len(b))


def int_accuracy(gold, gen):
    """M1_E-int: generated integers vs gold integers, compared position by position."""
    gn, gc = lits_text(gold)
    hn, hc = lits_text(gen)
    h1, t1 = _pair_rate(gn, hn)
    h2, t2 = _pair_rate(gc, hc)
    return {"naive_hit": h1, "naive_tot": t1, "naive_rate": round(h1 / max(1, t1), 6),
            "content_hit": h2, "content_tot": t2, "content_rate": round(h2 / max(1, t2), 6)}


_GOLD_CACHE = {}


#: gold source files (**explicit paths**: the parent `C` is not g33_common; the first
#: version misused C.EXAM_CAND and a bare except swallowed it silently => int_acc was all
#: null; this version uses explicit paths + a load assertion, no more silence)
_G32DIR = os.path.join(CCP, "gen32")
_G_CAND = os.path.join(_G32DIR, "EXAM_DEEP_CANDIDATE.json")
_G_CORPUS = os.path.join(_G32DIR, "DEEP_CORPUS_V1.jsonl")


def gold_proof_of(name, side_rec=None):
    """Gold proof text: EXAM_DEEP_CANDIDATE for the exam segment, corpus for the held-out
    segment (same source as judging, read-only)."""
    if not _GOLD_CACHE:
        for it in json.load(open(_G_CAND, encoding="utf-8"))["items"]:
            _GOLD_CACHE[it["name"]] = it.get("proof_text")
        with open(_G_CORPUS, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    _GOLD_CACHE.setdefault(r["name"], r.get("proof_text"))
        assert len(_GOLD_CACHE) >= 2399, len(_GOLD_CACHE)
    return _GOLD_CACHE.get(name)


def device_target_tokens(ctx, side_rec, logp=None):
    """Device re-render of this row's **own payload** -> (dev_text, tids, slot_tok_idx).

    slot_tok_idx = {token index in the device target string: device token id} (content
    integer positions).
    **Never reads the gold target**; only sfa/vars/dom_dump are used (same source and
    convention as the gen33 live state).
    """
    import gen34_slots as S34
    S34.init_dev()
    drow = _dev_row_of(side_rec)
    dev_text, _spans, info = S34.row_proof_instr(drow)
    full = "\u0060\u0060\u0060lean4\\n" + dev_text + "\\n\u0060\u0060\u0060"
    enc = ctx["tok"](full, add_special_tokens=False, return_offsets_mapping=True)
    tids = [int(x) for x in enc["input_ids"]]
    offs = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    pre = len("\u0060\u0060\u0060lean4\\n")
    csp = [(pre + a, pre + b) for (a, b) in _content_spans(dev_text)]
    out = {}
    for j, (a, b) in enumerate(offs):
        if b <= a:
            continue
        if any(a < lb and b > la for (la, lb) in csp):
            out[j] = tids[j]
    return dev_text, tids, out, info


def render_greedy_slot(model, ctx, side_rec, ops, payload, dev, max_new=160,
                       stop_seq="\u0060\u0060\u0060", allow=None, slot_dev=True,
                       slot_beta=None, logp=None):
    """gen34 [2] skeleton-anchored device supply (V34_DESIGN §2.2).

    Alignment criterion (strict, unambiguous): generated tokens gen and device tokens tids
    are equal at **every non-slot position** => the skeleton has not drifted, so inject at
    the slot: nxt = argmax(logits + g_dev*beta*onehot(tids[len(gen)])).
    Any single non-slot mismatch => desynchronized, no further injection in this run
    (registered by counter).
    """
    import v31_model as vm
    M_TASK = vm.st.M_TASK
    model.eval()
    beta = float(getattr(model, "slot_beta", 8.0) if slot_beta is None else slot_beta)
    diag = {"n_boost": 0, "n_desync": 0, "n_slot_pos": 0, "dev_err": None}
    by_tok, tids = {}, []
    try:
        _dt, tids, by_tok, info = device_target_tokens(ctx, side_rec, logp)
        if info.get("dev_err"):
            diag["dev_err"] = info["dev_err"][:2]
    except Exception as e:                                           # noqa: BLE001
        diag["dev_err"] = "SLOT_DEV:%r" % (e,)
    prompt_ids = side_rec["ids"][1:side_rec["np"]]
    cache = _prefix(ctx, side_rec, ops, payload)
    base = [tuple(x) for x in cache.rows[0].desc] + [(K_EMB, int(M_TASK))]
    base += [(K_EMB, int(t)) for t in prompt_ids]
    gen = []
    use = bool(getattr(model, "slot_cond", False)) and bool(slot_dev) and bool(by_tok)
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
            if _summ is not None:
                hin = hin + _summ.to(hin.dtype)
        z = model.lm.lm_head(hin)
        j = len(gen)
        if use and j < len(tids):
            ok = all(gen[k] == tids[k] for k in range(j) if k not in by_tok)
            if not ok:
                diag["n_desync"] += 1
                use = False                       # skeleton drifted: no more injection
                                                  # (one desync locks it)
            elif j in by_tok:
                diag["n_slot_pos"] += 1
                oh = torch.zeros_like(z)
                oh[0, int(by_tok[j])] = 1.0
                z = z + (float(model.g_dev.detach()) * beta) * oh
                diag["n_boost"] += 1
        nxt = int(z.argmax(-1))
        gen.append(nxt)
        txt = _ids_to_text(ctx["tok"], gen)
        if stop_seq and txt.rstrip().endswith(stop_seq) and len(gen) > 8:
            break
        if ctx["tok"].eos_token_id is not None and nxt == ctx["tok"].eos_token_id:
            break
    model.train()
    return _ids_to_text(ctx["tok"], gen), len(gen), diag
'''

PATCH_EVAL = [
    ('''G33 = os.path.join(CCP, "gen33")            # gen33 fork 写面
OUT = os.path.join(G33, "out")''',
     '''G33 = os.path.join(CCP, "gen33")            # parent (read-only; device/vocab source)
G34 = os.path.join(CCP, "gen34")            # gen34 fork write surface
OUT = os.path.join(G34, "out")'''),
    ('''           G26, G30, G33, os.path.join(CCP, "gen23"), os.path.join(RES, "scratch", "drct"),''',
     '''           G26, G30, G33, G34, os.path.join(CCP, "gen23"), os.path.join(RES, "scratch", "drct"),'''),
    ("import gen33_hybrid as HY                                                   # noqa: E402",
     "import gen34_hybrid as HY                                                   # noqa: E402"),
    ("import gen33_planattn as PA33                                               # noqa: E402",
     "import gen34_planattn as PA33                                               # noqa: E402"),
    ("def run_eval(model, side, allow, items, dev, ctx, a, logp):",
     SLOT_EVAL_BLOCK.strip("\n") + "\n\n\ndef run_eval(model, side, allow, items, dev, ctx, a, logp):"),
    ('''            try:
                txt, ntok = render_greedy(model, ctx, s, ops, payload, dev,
                                          max_new=a.max_new, allow=allow)''',
     '''            try:
                if int(getattr(a, "slot_cond", 0)):
                    txt, ntok, _dg = render_greedy_slot(
                        model, ctx, s, ops, payload, dev, max_new=a.max_new,
                        allow=allow, slot_dev=bool(getattr(a, "slot_dev", 1)),
                        slot_beta=getattr(a, "slot_beta", None), logp=logp)
                else:
                    txt, ntok = render_greedy(model, ctx, s, ops, payload, dev,
                                              max_new=a.max_new, allow=allow)'''),
    ("        cons = consistency_row(ops, txt)",
     "        cons = consistency_row(ops, txt)\n"
     "        _gp = gold_proof_of(s[\"name\"], s)\n"
     "        _ia = int_accuracy(_gp, _extract_body(txt)) if _gp else None"),
    ('        rows.append({"ri": int(ri), "name": s["name"], "cls": s["cls"], "seg": s["seg"],',
     '        rows.append({"ri": int(ri), "name": s["name"], "cls": s["cls"], "seg": s["seg"],\n'
     '                     "int_acc": _ia, "slot_diag": (None if a.arm != "hybrid" or\n'
     '                                                   not int(getattr(a, "slot_cond", 0))\n'
     '                                                   else locals().get("_dg")),'),
    ('    model = mcls(lat_tab_path=a.lat31, head=NHEAD, nvals=NVALS, parents=parents)',
     '    slot_on = int(getattr(a, "slot_cond", 0))\n'
     '    if slot_on:\n'
     '        mcls = HY.SlotHybridModel\n'
     '    model = mcls(lat_tab_path=a.lat31, head=NHEAD, nvals=NVALS, parents=parents)\n'
     '    if slot_on:\n'
     '        model.enable_slot(dev_flag=bool(getattr(a, "slot_dev", 1)),\n'
     '                          beta=float(getattr(a, "slot_beta", 8.0) or 8.0),\n'
     '                          lam_slot=float(getattr(a, "lam_slot", 3.0) or 3.0),\n'
     '                          seed=int(cargs.get("seed", 20260923) or 20260923))'),
    ('        if pa_on:\n'
     '            assert not pa_missing and not unexpected, (pa_missing[:8], list(unexpected)[:8])',
     '        if pa_on:\n'
     '            assert not pa_missing and not unexpected, (pa_missing[:8], list(unexpected)[:8])\n'
     '        if slot_on:\n'
     '            miss_s = [k for k in missing if k.startswith(("slot_render_emb", "g_pos", "g_dev"))]\n'
     '            assert not miss_s, ("gen34 slot params missing (ckpt not a slot build?): '
     '%s" % miss_s[:6])\n'
     '            info["slot"] = model.slot_param_report()'),
    ('    ap.add_argument("--attn-export", type=int, default=0,',
     '    ap.add_argument("--slot-cond", type=int, default=0,\n'
     '                    help="gen34: slot-conditioned render head")\n'
     '    ap.add_argument("--slot-dev", type=int, default=1,\n'
     '                    help="gen34: 1=device supply (A arm); '
     '0=LM fills slots itself (B arm)")\n'
     '    ap.add_argument("--slot-side", default=os.path.join(G34, "out",\n'
     '                                                       "DEEP_SLOT_SIDECAR.jsonl"))\n'
     '    ap.add_argument("--slot-beta", type=float, default=8.0)\n'
     '    ap.add_argument("--lam-slot", type=float, default=3.0)\n'
     '    ap.add_argument("--attn-export", type=int, default=0,'),
    ('    outp = a.out or os.path.join(OUT, "GEN33_EVAL_%s_%s%s.json"',
     '    outp = a.out or os.path.join(OUT, "GEN34_EVAL_%s_%s%s.json"'),
    ('    res = {"schema": "ccp-a-hybrid-eval/2",',
     '    res = {"schema": "gen34-eval/1",'),
]

PATCH_JUDGE = [
    ("G33 = C.G33", "G33 = C.G34"),
    ('            if not (f.startswith("GEN33_EVAL_%s_exam_deep_c" % tag) and f.endswith(".json")):',
     '            if not (f.startswith("GEN34_EVAL_%s_exam_deep_c" % tag) and f.endswith(".json")):'),
    ('    ap.add_argument("--tags", default="base,attn")',
     '    ap.add_argument("--tags", default="slotA,slotB,base")'),
]

PATCH_PA = []


def apply_patches(name, src_text, patches):
    out = src_text
    for i, (old, new) in enumerate(patches):
        n = out.count(old)
        if n != 1:
            raise SystemExit("FATAL %s patch[%d] matched %d times (expected 1): %r"
                             % (name, i, n, old[:80]))
        out = out.replace(old, new, 1)
    return out


def build(src, dst, hdr, patches, skip_header=False):
    s = os.path.join(G33, src)
    d = os.path.join(G34, dst)
    txt = open(s, encoding="utf-8").read()
    md5 = hashlib.md5(txt.encode("utf-8")).hexdigest()
    md5_bin = C.md5f(s)          # ledger convention = binary (same source as C.parent_ledger)
    n_crlf = C.crlf_count(s)
    body = apply_patches(dst, txt, patches) if patches else txt
    if skip_header:
        new = body
    else:
        k = body.index("\n\n", body.index("# -*- coding: utf-8 -*-"))
        new = body[:k] + "\n\n" + (hdr % {"md5": md5, "src": src}).strip("\n") + body[k:]
    with open(d, "w", encoding="utf-8") as f:
        f.write(new)
    print("[fork] %s -> %s (parent md5=%s, %d patches)" % (src, dst, md5, len(patches)))
    return {"src": src, "dst": dst, "parent_md5": md5_bin,
            "parent_md5_textmode": md5, "textmode_eq_binary": bool(md5 == md5_bin),
            "parent_crlf": int(n_crlf), "line_endings_normalized": bool(n_crlf > 0),
            "n_patches": len(patches),
            "out_md5": hashlib.md5(new.encode("utf-8")).hexdigest(),
            "out_bytes": len(new.encode("utf-8"))}


def main():
    led = []
    for src, dst, hdr, pat, skip in (
            ("gen33_hybrid.py", "gen34_hybrid.py", HDR_HYB, PATCH_HYB, False),
            ("gen33_eval.py", "gen34_eval.py", HDR_EVAL, PATCH_EVAL, False),
            ("gen33_planattn.py", "gen34_planattn.py", HDR_PA, PATCH_PA, True),
            ("gen33_judge.py", "gen34_judge.py", HDR_JUDGE, PATCH_JUDGE, False)):
        led.append(build(src, dst, hdr, pat, skip_header=skip))
    before = {k: v["md5"] for k, v in C.parent_ledger().items()}
    rep = {"schema": "gen34-fork/1", "forks": led,
           "parent_ledger": before,
           "parent_ledger_after": {k: v["md5"] for k, v in C.parent_ledger().items()},
           "judge_md5": C.md5f(C.DRCT + "/drct_common.py")}
    rep["parent_unchanged"] = bool(rep["parent_ledger"] == rep["parent_ledger_after"])
    rep["md5"] = C.save_json(os.path.join(G34, "GEN34_FORK_LEDGER.json"), rep)
    C.log("FORK: %d files; parent unchanged=%s; ledger md5=%s" % (len(led),
                                                                 rep["parent_unchanged"],
                                                                 rep["md5"]),
          os.path.join(G34, "logs", "gen34.log"))
    print("GEN34_MAKEFORK_RC=0 parent_unchanged=%s" % rep["parent_unchanged"])
    return rep


if __name__ == "__main__":
    main()
