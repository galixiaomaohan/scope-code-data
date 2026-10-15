# -*- coding: utf-8 -*-
# v31_model.py -- V31 reference-side model/data pipeline (new file; existing probes/ and tools/ files untouched)
#
# Convention sources (verifiable item by item):
#   * Body (SmolLM2-135M) + frame assembly/forward: strictly mirrors probes/sla_train_g1.py.txt /
#     sla_train.py.txt and cppcuda/src/model.h (V30C already verified equivalent), 30 layers of eager
#     attention, no fast-math.
#   * Dual heads (step head with 155 classes + binding head emitting per-slot values): structure =
#     DualHead from probes/sla_train_v31.py.txt (Linear(d_model,head_dim) +
#     Embedding(max_slots,d_model) + Linear(d_model,n_values),
#     `binding_logits = value_head(slot_emb + h)` plus a per-slot -1e9 mask), instantiated with
#     d_model=1152 -- 1152 = [h; c_t] (frame-final hidden 576 + context hidden 576), i.e. the engine-side
#     "per-step representation" width, so the step head Linear(1152,155) and the binding head
#     slot_emb(24,1152)/value_head(1152,151) align shape by shape.
#   * Data: **reads the C++-side cache directly**, cppcuda/cache/rows_v31_mixed.bin (CPPCD2) -- both
#     engines consume the same binary, so the batch plan (frames/labels/slots) is not "recomputed
#     separately" but the same fact.
#
# Usage: imported by tools/v31_ref_train.py and tools/prep_v31_init.py; not run standalone.
import importlib.machinery
import importlib.util
import json
import os
import struct
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RESEARCH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if RESEARCH not in sys.path:
    sys.path.insert(0, RESEARCH)
SUP = os.environ.get("SCOPE_SUPEROPERATOR", os.path.join(RESEARCH, "device"))
if SUP not in sys.path:
    sys.path.insert(0, SUP)


def _load(name, path):
    ld = importlib.machinery.SourceFileLoader(name, path)
    sp = importlib.util.spec_from_loader(name, ld)
    m = importlib.util.module_from_spec(sp)
    ld.exec_module(m)
    return m


st = _load('st', os.path.join(RESEARCH, 'probes/sla_train.py.txt'))
tr31 = _load('tr31', os.path.join(RESEARCH, 'probes/sla_train_v31.py.txt'))

# --- Engine constants (mirror the cfg in cppcuda/src/common.h) ---
HID = 576
NLAYER = 30
NHEAD = 9
NKV = 3
HEAD_DIM = 64
FFN = 1536
VOCAB = 49152
MAXFR = 25
RESSTRIDE = 25
RMS_EPS = 1e-5
ROPE_THETA = 100000.0
ATTN_SCALE = 0.125
NEG_BIG = float(torch.finfo(torch.float32).min)
M_TASK, M_TYPE, M_OP, M_RES, M_END, M_TAGFR = 730, 731, 732, 733, 734, 740
END31 = 154
NSLOT = 24
NVALS = 151
HEAD155 = 155
DMODEL = 1152
#: position-descriptor kind (same as plan.h / prep_cache_v31.py)
K_EMB, K_OP, K_RES, K_PST, K_WIN, K_PSZ = 0, 1, 2, 3, 4, 5


# ---------------------------------------------------------------------------
# cache (mirrors plan.h Cache::load)
# ---------------------------------------------------------------------------
class RowPlan:
    __slots__ = ('ids', 't4', 'readout_only', 'pos_task_end', 'end_pos', 'readout',
                 'desc', 'resHas', 'res', 'labels', 'feat_pos', 'bsl')


class Cache:
    def __init__(self, path):
        b = open(path, 'rb').read()
        assert b[:8] == b'CPPCD2\x00\x00', b[:8]
        off = 8

        def ri():
            nonlocal off
            v = struct.unpack_from('<i', b, off)[0]; off += 4; return v

        def rf():
            nonlocal off
            v = struct.unpack_from('<f', b, off)[0]; off += 4; return v

        self.n_rows, self.nT4, self.MTASK, self.MTYPE, self.MOP, self.MRES = \
            [ri() for _ in range(6)]
        self.MEND, self.MTAGFR, self.MTAGFRN, self.TMAX, self.ENDIDX, self.NOPS = \
            [ri() for _ in range(6)]
        self.HEAD, self.NSLOT, self.NVALS = [ri() for _ in range(3)]
        assert (self.HEAD, self.NSLOT, self.NVALS, self.ENDIDX, self.NOPS) == \
               (155, 24, 151, 154, 154)
        self.t4 = [tuple(rf() for _ in range(4)) for _ in range(self.nT4)]
        self.rows = []
        for _ in range(self.n_rows):
            R = RowPlan()
            nid = ri(); R.t4 = ri(); R.readout_only = ri()
            R.ids = [ri() for _ in range(nid)]
            R.pos_task_end = ri(); R.end_pos = ri(); R.readout = rf()
            nd = ri()
            R.desc = [(ri(), ri()) for _ in range(nd)]
            nr = ri()
            R.resHas, R.res = [], []
            for _ in range(nr):
                has = b[off]; off += 1
                R.resHas.append(has)
                R.res.append([rf() for _ in range(8)] if has else None)
            nl = ri()
            R.labels = [ri() for _ in range(nl)]
            nf = ri()
            R.feat_pos = [ri() for _ in range(nf)]
            nb = ri()
            R.bsl = [ri() for _ in range(nb)]
            self.rows.append(R)
        assert off == len(b), (off, len(b))

    def load_spec(self, spec_bin='cppcuda/cache/v31_spec.bin'):
        """Read the dense allowed table from v31_spec.bin -> (154,24,151) bool."""
        d = open(spec_bin, 'rb').read()
        i = d.rfind(b'ALLO')
        assert i > 0, 'v31_spec.bin 缺 ALLO 块'
        a = np.frombuffer(d, dtype=np.uint8, offset=i + 4)
        assert a.size == 154 * 24 * 151, a.size
        return torch.from_numpy(a.reshape(154, 24, 151).copy()).to(torch.bool)


# ---------------------------------------------------------------------------
# batch collate (mirrors plan.h BatchCpu::build, per-row in the same order)
# ---------------------------------------------------------------------------
class BatchCpu:
    def build(self, C, rowsIdx, Lfix=0, Tfix=0, TallocIn=0):
        B = len(rowsIdx)
        Ln = max(len(C.rows[i].desc) for i in rowsIdx)
        Tm = max(len(C.rows[i].labels) for i in rowsIdx)
        self.B, self.Ln, self.Tm = B, Ln, Tm
        L = Lfix if Lfix > 0 else Ln
        assert L >= Ln
        Tmax = Tfix if Tfix > 0 else Tm
        Talloc = TallocIn if TallocIn > 0 else Tmax
        assert Tmax >= Tm
        self.L, self.Tmax, self.Talloc = L, Tmax, Talloc
        self.oE = 0
        self.oW = B * L
        self.oO = self.oW + B
        self.oP = self.oO + B * MAXFR
        self.oR = self.oP + B * MAXFR
        self.oZ = self.oR + B * (MAXFR + 1)
        self.embIdx = [0] * (B * L)
        self.winIdx = [0] * B
        self.opIdx = [0] * (B * MAXFR)
        self.pstIdx = [0] * (B * MAXFR)
        self.resBuf = [0.0] * (B * (MAXFR + 1) * 8)
        self.srcIdx = [self.oZ] * (B * L)
        self.lenB = [0] * B
        self.labels = [-100] * (B * Talloc)
        self.slots = [-1] * (B * Talloc * NSLOT)
        self.idxH = [0] * (B * Talloc)
        self.idxC = [0] * (B * Talloc)
        self.Nvalid = 0
        self.Nbind = 0
        for r in range(B):
            R = C.rows[rowsIdx[r]]
            nreal = len(R.desc)
            self.lenB[r] = nreal
            cEmb = cOp = cPst = 0
            for p in range(L):
                flat = r * L + p
                if p >= nreal:
                    self.srcIdx[flat] = self.oZ
                    continue
                kind, arg = R.desc[p]
                if kind == K_EMB:
                    assert cEmb < L
                    slot = self.oE + r * L + cEmb
                    self.embIdx[slot - self.oE] = arg
                    cEmb += 1
                elif kind == K_WIN:
                    slot = self.oW + r
                    self.winIdx[r] = arg
                elif kind == K_OP:
                    assert cOp < MAXFR
                    slot = self.oO + r * MAXFR + cOp
                    self.opIdx[r * MAXFR + cOp] = arg
                    cOp += 1
                elif kind == K_PST:
                    assert cPst < MAXFR
                    slot = self.oP + r * MAXFR + cPst
                    self.pstIdx[r * MAXFR + cPst] = arg
                    cPst += 1
                elif kind == K_RES:
                    assert arg < MAXFR
                    slot = self.oR + r * (MAXFR + 1) + arg
                    if arg < len(R.res) and R.resHas[arg]:
                        for k in range(8):
                            self.resBuf[(r * (MAXFR + 1) + arg) * 8 + k] = R.res[arg][k]
                elif kind == K_PSZ:
                    slot = self.oR + r * (MAXFR + 1) + MAXFR
                else:
                    raise AssertionError('unknown kind %d' % kind)
                self.srcIdx[flat] = slot
            nl_eff = 0 if R.readout_only else len(R.labels)
            for t in range(nl_eff):
                self.labels[r * Talloc + t] = R.labels[t]
            self.Nvalid += nl_eff
            nb = 0
            for t in range(nl_eff):
                for j in range(NSLOT):
                    v = R.bsl[t * NSLOT + j]
                    self.slots[(r * Talloc + t) * NSLOT + j] = v
                    if v >= 0:
                        nb += 1
            self.Nbind += nb
            for t in range(Talloc):
                self.idxH[r * Talloc + t] = r * L + R.pos_task_end
                cp = R.pos_task_end if (t == 0 or t - 1 >= len(R.feat_pos)) else R.feat_pos[t - 1]
                self.idxC[r * Talloc + t] = r * L + cp
        return self


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SLANavV31(nn.Module):
    """SmolLM2-135M body + V31 dual heads (step head 155 / binding head 24x151)."""

    def __init__(self, base=None, lat_tab_path='cppcuda/cache/lat_tab_v31.bin'):
        super().__init__()
        from transformers import AutoModelForCausalLM
        base = base or st.BASE
        self.lm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32,
                                                       attn_implementation='eager')
        self.W_in = nn.Linear(24, HID)
        with torch.no_grad():                      # FL2-style initialization (identity scaled by the median embedding norm)
            med = self.lm.get_input_embeddings().weight.float().norm(dim=-1).median()
            self.W_in.weight.zero_()
            self.W_in.weight[:24, :] = torch.eye(24) * med
            self.W_in.bias.zero_()
        self.P_op = nn.Linear(64, HID)
        self.P_res = nn.Linear(8, HID)
        self.P_st = nn.Linear(4, HID)
        # --- V31 dual heads (structure = probes/sla_train_v31.py.txt DualHead) ---
        self.step_head = nn.Linear(DMODEL, HEAD155)
        self.slot_emb = nn.Embedding(NSLOT, DMODEL)
        self.value_head = nn.Linear(DMODEL, NVALS)
        self.slot_emb.weight.data.normal_(0, 0.02)
        # --- Codebook latent table (V31 155 rows x 64): trainable parameter slot (enters the optimizer with --joint-cb 1) ---
        self.codebook = nn.Module()
        lat = np.fromfile(lat_tab_path, dtype=np.float32).reshape(HEAD155, 64)
        self.codebook.lat_tab = nn.Parameter(torch.from_numpy(lat.copy()))

    # ---- Body ----
    def _rms(self, x, w):
        v = x.pow(2).mean(-1, True)
        return w * (x * torch.rsqrt(v + RMS_EPS))

    @staticmethod
    def _rope(t, c, s):
        h = HEAD_DIM // 2
        x1, x2 = t[..., :h], t[..., h:]
        rot = torch.cat([-x2, x1], -1)
        return t * c + rot * s

    def forward_body(self, seq, mask, cosc, sinc):
        B, S, _ = seq.shape
        c = cosc.unsqueeze(1); s = sinc.unsqueeze(1)
        x = seq
        for l in self.lm.model.layers:
            res = x
            h = self._rms(x, l.input_layernorm.weight)
            q = l.self_attn.q_proj(h).view(B, S, NHEAD, HEAD_DIM).transpose(1, 2)
            k = l.self_attn.k_proj(h).view(B, S, NKV, HEAD_DIM).transpose(1, 2)
            v = l.self_attn.v_proj(h).view(B, S, NKV, HEAD_DIM).transpose(1, 2)
            q = self._rope(q, c, s); k = self._rope(k, c, s)
            k = k.unsqueeze(2).expand(B, NKV, NHEAD // NKV, S, HEAD_DIM).reshape(B, NHEAD, S, HEAD_DIM)
            v = v.unsqueeze(2).expand(B, NKV, NHEAD // NKV, S, HEAD_DIM).reshape(B, NHEAD, S, HEAD_DIM)
            att = torch.matmul(q, k.transpose(2, 3)) * ATTN_SCALE + mask
            att = torch.softmax(att, -1)
            o = torch.matmul(att, v).transpose(1, 2).contiguous().view(B, S, HID)
            x = res + l.self_attn.o_proj(o)
            res = x
            h = self._rms(x, l.post_attention_layernorm.weight)
            g = F.silu(l.mlp.gate_proj(h)) * l.mlp.up_proj(h)
            x = res + l.mlp.down_proj(g)
        return self._rms(x, self.lm.model.norm.weight)

    def assemble(self, B, L, bc, C, dev):
        """Mirrors model.h Model::assemble (lat_tab participates in the P_op table lookup -> the gradient
        path for joint-training the codebook)."""
        lat_tab = self.codebook.lat_tab
        opTab = F.linear(lat_tab, self.P_op.weight, self.P_op.bias)
        oh4 = torch.eye(4, dtype=torch.float32, device=dev)
        pstTab = F.linear(oh4, self.P_st.weight, self.P_st.bias)
        t4 = torch.tensor(C.t4, dtype=torch.float32, device=dev)
        t4pad = torch.zeros(C.nT4, 24, dtype=torch.float32, device=dev)
        t4pad[:, :4] = t4
        winTab = F.linear(t4pad, self.W_in.weight, self.W_in.bias)
        resBuf = torch.tensor(bc.resBuf, dtype=torch.float32, device=dev).view(bc.B * (MAXFR + 1), 8)
        resTab = F.linear(resBuf, self.P_res.weight, self.P_res.bias)
        zrow = torch.zeros(1, HID, dtype=torch.float32, device=dev)

        def ti(key):
            return torch.tensor(bc.__dict__[key], dtype=torch.long, device=dev)

        embG = self.lm.model.embed_tokens.weight.index_select(0, ti('embIdx'))
        winG = winTab.index_select(0, ti('winIdx'))
        opG = opTab.index_select(0, ti('opIdx'))
        pstG = pstTab.index_select(0, ti('pstIdx'))
        src = torch.cat([embG, winG, opG, pstG, resTab, zrow], 0)
        return src.index_select(0, ti('srcIdx')).view(B, L, HID)

    def consts(self, L, dev):
        cm = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev), 0).unsqueeze(0)
        pos = torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=dev) / HEAD_DIM
        inv = 1.0 / torch.pow(torch.tensor(ROPE_THETA, dtype=torch.float32, device=dev), pos)
        p = torch.arange(L, dtype=torch.float32, device=dev).view(1, 1, L)
        fr = torch.matmul(inv.view(1, -1, 1), p).transpose(1, 2)
        emb = torch.cat([fr, fr], -1)
        return cm, emb.cos().contiguous(), emb.sin().contiguous()

    def mask_of(self, cm, B, L, lenB, dev):
        """Mirrors model.h LConsts::mask: causal and position < row length -> 0, otherwise finfo.min."""
        posAr = torch.arange(L, dtype=torch.long, device=dev).view(1, 1, -1)
        lb = torch.tensor(lenB, dtype=torch.long, device=dev)
        keep = cm & (posAr < lb.view(B, 1, 1))
        z = torch.zeros((), dtype=torch.float32, device=dev)
        m = torch.where(keep, z, torch.full((), NEG_BIG, dtype=torch.float32, device=dev))
        return m.unsqueeze(1)

    def heads(self, H, bc, C, dev, allow_rows):
        B, L, T = bc.B, bc.L, bc.Talloc
        Hf = H.reshape(-1, HID)
        idxH = torch.tensor(bc.idxH, dtype=torch.long, device=dev)
        idxC = torch.tensor(bc.idxC, dtype=torch.long, device=dev)
        gH = Hf.index_select(0, idxH)
        gC = Hf.index_select(0, idxC)
        x = torch.cat([gH, gC], -1)                                   # (B*T, 1152)
        step_logits = F.linear(x, self.step_head.weight, self.step_head.bias)
        se = self.slot_emb.weight.unsqueeze(0).expand(x.shape[0], NSLOT, DMODEL)
        zb = F.linear(se + x.unsqueeze(1), self.value_head.weight, self.value_head.bias)
        op_flat = torch.tensor(bc.labels, dtype=torch.long, device=dev)
        op_cl = op_flat.clamp(0, 153)
        allowed = allow_rows.to(dev).index_select(0, op_cl).view(B * T, NSLOT, NVALS)
        sv = torch.tensor(bc.slots, dtype=torch.long, device=dev).view(B * T, NSLOT)
        sm = (sv >= 0) & (op_flat.view(-1, 1) != END31)
        keep = allowed & sm.unsqueeze(-1)
        zb = zb.masked_fill(~keep, NEG_BIG)
        tgt = torch.where(sm, sv.clamp(0, NVALS - 1), torch.full_like(sv, -100))
        return step_logits, zb, tgt

    def loss(self, step_logits, zb, tgt, bc, lam_bind=1.0):
        lab = torch.tensor(bc.labels, dtype=torch.long, device=step_logits.device)
        nv = int((lab != -100).sum())
        ls = F.cross_entropy(step_logits.reshape(-1, HEAD155), lab.reshape(-1),
                             ignore_index=-100, reduction='sum') / max(nv, 1)
        nbf = int((tgt != -100).sum())
        lb = F.cross_entropy(zb.reshape(-1, NVALS), tgt.reshape(-1),
                             ignore_index=-100, reduction='sum') / max(nbf, 1)
        return ls + (lam_bind / NSLOT) * lb, ls, lb


def apply_ema(lat, lat_init, decay):
    """Joint-training codebook EMA drift guard: lat <- decay*lat + (1-decay)*lat_init (in place, no gradients)."""
    with torch.no_grad():
        lat.mul_(decay).add_(lat_init, alpha=1.0 - decay)


def row_renorm(lat, n0, eps=1e-8):
    """Per-row renorm to the original norms (codebook-collapse guard): lat[i] *= n0[i]/max(||lat[i]||,eps)."""
    with torch.no_grad():
        cur = lat.norm(dim=-1)
        f = n0 / cur.clamp_min(eps)
        lat.mul_(f.unsqueeze(-1))
