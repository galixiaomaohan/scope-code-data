# -*- coding: utf-8 -*-
# dyn_train.py -- the main arena of C3 superiority: context-continuous dynamical-system rollout.
#   Task: the context provides K pairs (x, f(x)) of consecutive examples (f = random linear map,
#         spectral norm 0.9), then queries f^T(x0); the loop extrapolates step by step, the readout
#         head regresses continuous values, supervised by per-step MSE.
#   Value encoding: W_in: R^D -> R^576 single-position superposition (trained); readout W_out: R^576 -> R^D.
#   Arms: latent = feedback injected at full precision; disc8/disc4 = feedback quantized to 8/4 bits per dim first.
#   Criterion: latent final error < disc8 < disc4; latent stays flat along T, discrete accumulates with T.
# Usage: python dyn_train.py train --arm latent|disc8|disc4
import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

D = 8
K_EX = 12
TMAX = 16
BASE = 'models/SmolLM2-135M'
M_IN, M_OUT, M_Q, M_A = 500, 600, 700, 800
BITS = {'latent': 0, 'disc8': 8, 'disc4': 4, 'disc8m': 8, 'disc4m': 4,
        'latq': 5}                                      # v11: feedback snapped to 5 bits (32 levels)
MU = 255.0


def quant(v, bits, mu=False):
    lev = 2 ** bits
    if mu:                                    # μ-law companding: fine grid for small signals = strongest form of the discrete arm (control)
        x = v.clamp(-1, 1)
        y = x.sign() * torch.log1p(MU * x.abs()) / math.log1p(MU)
        q = torch.round((y + 1) / 2 * (lev - 1)) / (lev - 1) * 2 - 1
        return q.sign() * (((1 + MU) ** q.abs() - 1) / MU)
    return torch.round((v.clamp(-1, 1) + 1) / 2 * (lev - 1)) / (lev - 1) * 2 - 1


def dequant_idx(idx, bits):
    """v11 readable readout: bin-center μ-law dequantization (under CE W_out has no gradient =
    frozen random head; the continuous head reads blind)"""
    lev = 2 ** bits
    y = idx.float() / (lev - 1) * 2 - 1
    return y.sign() * (((1 + MU) ** y.abs() - 1) / MU)


def quant_idx(v, bits, mu=True):
    """v11 latq: μ-law bin index (0..lev-1, long) -- the core of the discretization experience:
    under the CE loss the 'emit the mean' degenerate solution does not exist, so the model is
    forced to emit a conditional prediction from the first step."""
    lev = 2 ** bits
    x = v.clamp(-1, 1)
    y = x.sign() * torch.log1p(MU * x.abs()) / math.log1p(MU)
    return torch.round((y + 1) / 2 * (lev - 1)).long().clamp(0, lev - 1)


class Dyn(nn.Module):
    def __init__(self, iface='random', scratch=False, nbins=0, nca=False):
        super().__init__()
        if scratch:                                   # v8: isomorphic scratch twin (differs only in initialization)
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(BASE)
            self.lm = AutoModelForCausalLM.from_config(cfg).to(torch.float32)
        else:
            self.lm = AutoModelForCausalLM.from_pretrained(
                BASE, dtype=torch.float32, attn_implementation='eager')
        self.W_in = nn.Linear(D, 576)
        self.W_out = nn.Linear(576, D)
        if nbins:                                 # v11 latq: per-dimension μ-law bin classification head (CE loss)
            self.W_bins = nn.Linear(576, D * nbins)
        if nca:                                   # v17 NCA: rule head (D-dim bin classification of the convolution kernel)
            self.Khead = nn.Linear(576, D * 32)
        # Injection scale: make ||W_in(unit v)|| match the median token-embedding norm (~3.1;
        # default init ~6.9 is 2.2x too large).
        # v4 iface='aligned': values sit in the first D fixed coordinates (axis-aligned), removing
        # the random-subspace decode cold-start obstacle.
        with torch.no_grad():
            med = (self.lm.get_input_embeddings().weight.float()
                   .norm(dim=-1).median())
            if iface == 'aligned':
                self.W_in.weight.zero_()
                self.W_in.weight[:D, :] = torch.eye(D) * med
                self.W_in.bias.zero_()
            else:
                v = torch.randn(512, D)
                v = v / v.norm(dim=-1, keepdim=True)
                cur = self.W_in(v).norm(dim=-1).mean()
                self.W_in.weight.mul_(med / cur)
                self.W_in.bias.mul_(med / cur)


def gen_episode(B, sg, kex, ident=False, afam='ortho', want_A=False):
    if ident:
        A = torch.eye(D).unsqueeze(0).expand(B, D, D).clone()   # v4 warm start: identity map
    elif afam == 'perm':        # TCA A-structure curriculum (1): permutation matrices = a continuous cloak for the stagekey native operation
        idx = torch.rand(B, D, generator=sg).argsort(dim=1)     # [B,D]
        A = torch.eye(D)[idx]                                   # [B,D,D] row permutation
    elif afam == 'sperm':       # (2): signed permutation (magnitude information enters the curriculum)
        idx = torch.rand(B, D, generator=sg).argsort(dim=1)
        sg2 = (torch.rand(B, D, generator=sg) < 0.5).float() * 2 - 1
        A = torch.eye(D)[idx] * sg2.unsqueeze(1)
    elif afam == 'circ':        # v17 NCA: orthogonal circulant family = circular convolution kernels
        # (Fourier phase parameterization; continuous values + convolution structure = the minimal
        # step from the discrete world toward the continuous one; unit spectral magnitude =>
        # automatically orthogonal, stable chain)
        th = torch.rand(B, D // 2 + 1, generator=sg) * 2 * math.pi
        d = torch.complex(torch.cos(th), torch.sin(th))     # unit-magnitude complex phases
        d = torch.cat([torch.where(th[:, :1] < math.pi,      # DC taken as ±1 (real)
                                    torch.ones_like(th[:, :1]),
                                    -torch.ones_like(th[:, :1]))
                       .to(torch.complex64), d[:, 1:]], dim=1)
        if D % 2 == 0:                                       # Nyquist must also be real
            sgn = torch.where(th[:, -1:] < math.pi,
                              torch.ones_like(th[:, -1:]),
                              -torch.ones_like(th[:, -1:])).to(torch.complex64)
            d = torch.cat([d[:, :-1], sgn], dim=1)
        full = torch.zeros(B, D, dtype=torch.complex64)
        full[:, :D // 2 + 1] = d
        full[:, D // 2 + 1:] = torch.flip(
            torch.conj(d[:, 1:D // 2]), dims=[1])           # conjugate symmetry => real kernel
        c = torch.fft.ifft(full, dim=1).real                # real kernel, row norm = 1
        A = torch.stack([torch.roll(c, i, dims=1) for i in range(D)], dim=1)
    else:
        # v5: QR-orthogonal A (spectral norm = 1, no signal decay). Audit round 3 #1: a decaying
        # A (0.9^t) plus an absolute grid inflates the discrete-arm error by 0.9^-t (an engineering
        # artifact, not the intrinsic cost of discretization), turning the superiority comparison
        # into a straw man
        M = torch.randn(B, D, D, generator=sg)
        Q, R = torch.linalg.qr(M)
        A = Q * torch.diagonal(R, dim1=-2, dim2=-1).sign().unsqueeze(-2)
    X = torch.randn(B, kex + 1, D, generator=sg)
    X = X / X.norm(dim=-1, keepdim=True)
    EX = X[:, :kex]                                      # [B,K,D]
    EY = torch.einsum('bij,bkj->bki', A, EX)
    x0 = X[:, kex]
    chain = []
    v = x0
    for _ in range(TMAX):
        v = torch.einsum('bij,bj->bi', A, v)
        chain.append(v)
    chain = torch.stack(chain, 1)
    # P0 fix (audit-math#5): the old cat+reshape(B,-1) misaligned across the batch -- all batches'
    # EY rows came first and the chain rows after, so after regrouping s came from another episode
    # (measured on perm: EY saturation 14.1%, worst |A/s| = 1.87). Changed to a per-batch amax;
    # the task distribution applies uniformly to all v10-v18.
    s = (torch.cat([EY, chain, A.reshape(B, D, D)], 1).abs()
         .amax(dim=(1, 2)) + 1e-6)[:, None, None]             # [B,1,1]
    if want_A:                     # for the v12A curriculum / A-explicit arm: A scaled identically (off by default; old callers unaffected)
        return (EX / s, EY / s, x0 / s.squeeze(1), chain / s, A / s)
    return (EX / s, EY / s, x0 / s.squeeze(1), chain / s)


def run_chain(m, EX, EY, x0, Ln, bits, kex=K_EX, mu=False, tf_prob=0.0,
              chain=None, collect_h=False, aexplicit=False, A=None,
              fb_bins=False):
    emb_all = m.lm.get_input_embeddings().weight
    B = EX.shape[0]
    dev = EX.device
    parts = []
    if aexplicit:                     # TCA diagnostic arm: A's 8 rows injected directly as tokens (bypasses inference, isolates 'applying A')
        for r in range(D):
            parts += [emb_all[M_A].expand(B, -1), m.W_in(A[:, r, :])]
    else:
        for k in range(kex):
            parts += [emb_all[M_IN].expand(B, -1),
                      m.W_in(EX[:, k]),
                      emb_all[M_OUT].expand(B, -1),
                      m.W_in(EY[:, k])]
    parts += [emb_all[M_Q].expand(B, -1), m.W_in(x0)]
    seq = torch.stack(parts, 1)                          # [B, 4K+2, 576]
    am = torch.ones(B, seq.shape[1], device=dev)
    preds = []
    hs = []
    for i in range(Ln):
        out = m.lm(inputs_embeds=seq, attention_mask=am,
                   use_cache=False, output_hidden_states=True)
        h = out.hidden_states[-1][:, -1]
        if fb_bins:
            # P0 fix (audit-math#15/E1): latq readout and feedback both go through the bin channel
            # (matching deployment). The old W_out had a permanently None gradient under the CE
            # loss, so feedback = a random head and the deep-chain self-feedback was never trained.
            pred = dequant_idx(
                m.W_bins(h.float()).view(B, D, 32).argmax(-1), 5)
        else:
            pred = m.W_out(h.float())
        preds.append(pred)
        if collect_h:
            hs.append(h.float())
        if i + 1 < Ln:
            fb = pred.detach()
            if tf_prob > 0 and chain is not None:      # v6 teacher forcing: feed the ground-truth state with probability p
                gmask = torch.rand(B, 1, device=dev) < tf_prob
                tv = chain[:, i]
                if fb_bins:    # audit A1 short-term item: align the teacher channel with the deployment channel (feed bin-center ground truth)
                    tv = dequant_idx(quant_idx(tv, 5), 5)
                fb = torch.where(gmask, tv, fb)
            if bits and not fb_bins:   # the bin readout is already a grid center, no second quantization
                fb = quant(fb, bits, mu=mu)
            seq = torch.cat([seq, m.W_in(fb).unsqueeze(1)],
                            1)
            am = torch.cat([am, torch.ones(B, 1, device=dev)], 1)
    if collect_h:
        return torch.stack(preds, 1), torch.stack(hs, 1)   # [B,Ln,D],[B,Ln,576]
    return torch.stack(preds, 1)                         # [B,Ln,D]


def run_train(args):
    dev = 'cuda'
    torch.manual_seed(101)
    torch.backends.cuda.matmul.allow_tf32 = False   # round 4 #1: TF32 = 10-bit mantissa, must be disabled
    torch.backends.cudnn.allow_tf32 = False
    m = Dyn(args.iface, args.scratch,
            nbins=32 if args.arm == 'latq' else 0,
            nca=args.nca).to(dev)
    m.train()
    # v6b: keeping all activations for the Ln forward passes of the feedback chain => OOM at T=16
    # (12GB card); gradient checkpointing compresses activations 10-30x (same mechanism as smol)
    m.lm.gradient_checkpointing_enable()
    m.lm.enable_input_require_grads()
    if args.iface == 'aligned':
        with torch.no_grad():
            # audit #14: scaling on unit vectors leaves up to ~2x residual distortion under the real
            # value distribution (norm < 1) -- rescale on the real episode value distribution:
            # mean output norm = median embedding norm
            norms = []
            sg_c = torch.Generator().manual_seed(7)
            for _ in range(8):
                EX, EY, x0, chain = gen_episode(16, sg_c, args.kex,
                                                afam=args.afam)  # audit design#2: calibration must follow the family (the old ortho calibration inflated the perm injection norm by 11%)
                norms += [float(EX.norm(dim=-1).mean()),
                          float(EY.norm(dim=-1).mean()),
                          float(x0.norm(dim=-1).mean()),
                          float(chain.norm(dim=-1).mean())]
            mstar = sum(norms) / len(norms)
            m.W_in.weight.mul_(1.0 / mstar)
            print(f'[iface] aligned rescale: E||v||={mstar:.3f} scale×{1/mstar:.3f}',
                  flush=True)
    bits = BITS[args.arm]
    head_params = list(m.W_in.parameters()) + list(m.W_out.parameters())
    if hasattr(m, 'W_bins'):
        head_params += list(m.W_bins.parameters())
    if hasattr(m, 'Khead'):
        head_params += list(m.Khead.parameters())
    lm_params = [p for p in m.lm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{'params': lm_params, 'lr': args.lrpeak},
         {'params': head_params, 'lr': 3e-3}], betas=(0.9, 0.95))
    sched = None
    if args.scratch:                      # v8 scratch LR schedule: 3e-4 peak + 1000-step warmup + cosine to 0.1x
        warm = 1000

        def _lrlam(step):
            if step < warm:
                return step / max(warm, 1)
            prog = (step - warm) / max(args.steps - warm, 1)
            return 0.1 + 0.45 * (1 + math.cos(math.pi * prog))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lrlam)
    gate_tf = gate_dep = False                     # v9 mastery gates (ported from the dwell2 mechanism)
    if args.gates:
        gs = [g.strip() for g in args.gates.split(',') if g.strip()]
        gate_tf, gate_dep = 'tf' in gs, 'depth' in gs
    tf_held, dep_lvl, last_adv, tf_win = 1.0, 1, 0, []
    tgt = f'ckpt/dyn_{args.arm}{args.tag}'
    sg = torch.Generator().manual_seed(101)
    t0 = time.time()
    print(f"[argv] {vars(args)}", flush=True)   # record the command in the log (anti-archaeology; v17a lesson)
    for it in range(args.steps):
        # v6 pedagogy: (1) reset the curriculum on switches (real problems ramp again from T=1);
        # (2) anneal teacher forcing (over the first 1500 steps the feedback goes linearly from
        # ground-truth states to self-feedback, fixing the state-distribution cold start);
        # (3) identity regularization (10% mixed in, against prior interference).
        # v9: tf annealing and depth advancement move from a clock to mastery gates -- clock
        #     parameters are tuned for pretrained models; a scratch model burns through tf before
        #     learning (self-feedback garbage) and the curriculum outruns the learner.
        eff_it = it if it < args.warmup else it - args.warmup
        if gate_dep:
            tmax = min(TMAX, dep_lvl)
        else:
            tmax = min(TMAX, 1 + eff_it // 250)
        if gate_tf:
            tf_p = tf_held
        else:
            tf_p = (0.0 if it < args.warmup
                    else max(0.0, 1.0 - (it - args.warmup) / 1500.0))
        ident = (args.warmup > 0) and (
            it < args.warmup or float(torch.rand(1, generator=sg)) < 0.10)
        # v10 gradient accumulation: bs_eff = bs*accum (diagnosis: the Garg et al. literature uses
        # bs>=128; our bs6 has too low a gradient SNR, so direction learning deadlocks in the
        # predict-the-mean basin; accumulation does not change memory, it just collects cleaner gradients)
        opt.zero_grad()
        mloss, mce, mkce = 0.0, float('nan'), float('nan')  # audit design#11: print the micro-batch mean
        for _mi in range(max(1, args.accum)):
            Av = None
            if args.aexplicit:
                EX, EY, x0, chain, Av = gen_episode(
                    args.bs, sg, args.kex, ident=ident,
                    afam=args.afam, want_A=True)
            elif args.nca:
                EX, EY, x0, chain, Av = gen_episode(
                    args.bs, sg, args.kex, ident=ident,
                    afam=args.afam, want_A=True)
            else:
                EX, EY, x0, chain = gen_episode(
                    args.bs, sg, args.kex, ident=ident, afam=args.afam)
            EX, EY, x0, chain = (t.to(dev) for t in (EX, EY, x0, chain))
            Av = Av.to(dev) if Av is not None else None
            Ln = int(torch.randint(1, tmax + 1, (1,), generator=sg))
            mufb = args.arm.endswith('m') or args.arm == 'latq'
            if args.arm == 'latq':
                preds, hs = run_chain(m, EX, EY, x0, Ln, bits, args.kex,
                                      mu=mufb, tf_prob=tf_p, chain=chain,
                                      collect_h=True,
                                      aexplicit=args.aexplicit, A=Av,
                                      fb_bins=True)
                NB = 32
                logits = m.W_bins(hs).view(args.bs, Ln, D, NB)
                tb = quant_idx(chain[:, :Ln], 5)          # μ-law 5bit = 32-bin target
                ce_l = F.cross_entropy(logits.reshape(-1, NB),
                                       tb.reshape(-1))
                if args.nca:              # v17: rule-head supervision (kernel = first row of A/s, same μ-law grid)
                    kb = quant_idx(Av[:, 0, :], 5)
                    klg = m.Khead(hs[:, 0]).view(args.bs, D, 32)
                    ke_l = F.cross_entropy(klg.reshape(-1, 32),
                                           kb.reshape(-1))
                    loss = ce_l + ke_l
                    mce, mkce = float(ce_l), float(ke_l)
                else:
                    loss = ce_l
                    mce = float(ce_l)
            else:
                preds = run_chain(m, EX, EY, x0, Ln, bits, args.kex,
                                  mu=mufb, tf_prob=tf_p, chain=chain,
                                  aexplicit=args.aexplicit, A=Av)
                loss = F.mse_loss(preds, chain[:, :Ln])
            (loss / max(1, args.accum)).backward()
            mloss += float(loss)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if sched is not None:
            sched.step()
        if it % 50 == 0:
            with torch.no_grad():
                if args.arm == 'latq':      # readable readout: dequantize the W_bins bin centers
                    lgq = m.W_bins(hs).view(*hs.shape[:-1], D, 32)
                    preds = dequant_idx(lgq.argmax(-1), 5)
                if args.nca:                # v17: rule-readout accuracy (an independently measurable metric)
                    kbq = quant_idx(Av[:, 0, :], 5)
                    klgq = m.Khead(hs[:, 0]).view(args.bs, D, 32)
                    kacc = float((klgq.argmax(-1) == kbq).float().mean())
                    print(f'[nca] it={it} kernel_bin_acc={kacc:.3f}',
                          flush=True)
                rel = ((preds - chain[:, :Ln]).norm(dim=-1)
                       / chain[:, :Ln].norm(dim=-1).clamp_min(1e-6))
                # median convention (audit round 3 #5): the relative error is heavy-tailed; the mean
                # is dominated by small-denominator episodes
                hop1 = rel[:, 0].median()
                deep = rel[:, 1:].median() if Ln > 1 else torch.tensor(0.0)
                fin = rel[:, -1].median()
            _ex = f" ce={mce:.4f} kce={mkce:.4f}" if args.nca else ""
            print(f"[dyn_{args.arm}] it={it} T={Ln} "
                  f"loss={mloss / max(1, args.accum):.5f}{_ex} "
                  f"e1={float(hop1):.3f} edeep={float(deep):.3f} "
                  f"efin={float(fin):.3f} "
                  f"warm={'Y' if it < args.warmup else 'n'} "
                  f"{it/(time.time()-t0):.2f} it/s", flush=True)
        if (gate_tf or gate_dep) and it > 0 and it % 250 == 0:
            with torch.no_grad():              # v9 gate probe: fresh seed, n=24
                sgp = torch.Generator().manual_seed(9000 + it // 250)
                # P0 fix (0919): the probe must use the same convention as training -- the afam
                # family plus explicit A. The old version omitted afam (defaulting to ortho) and
                # omitted aexplicit/A => the v13-v17a gate probe kept measuring ortho x infer
                # (the hardest cell), freezing the gradient gate at T=1 forever.
                if args.aexplicit or args.nca:
                    EXP, EYP, x0P, chP, AvP = gen_episode(
                        24, sgp, args.kex, afam=args.afam, want_A=True)
                else:
                    EXP, EYP, x0P, chP = gen_episode(
                        24, sgp, args.kex, afam=args.afam)
                    AvP = None
                EXP, EYP, x0P, chP = (t.to(dev) for t in
                                      (EXP, EYP, x0P, chP))
                AvP = AvP.to(dev) if AvP is not None else None
                e2tf = edep = float('nan')
                if gate_tf:                    # stepping stone: 2 steps with true feedback (read A + copy feedback + multiply once)
                    if args.arm == 'latq':     # readable readout: bin dequantization
                        p2, h2 = run_chain(m, EXP, EYP, x0P, 2, bits,
                                           args.kex, mu=True,
                                           tf_prob=1.0, chain=chP,
                                           collect_h=True,
                                           aexplicit=args.aexplicit, A=AvP,
                                           fb_bins=True)
                        p2 = dequant_idx(
                            m.W_bins(h2).view(*h2.shape[:-1], D, 32)
                            .argmax(-1), 5)
                    else:
                        p2 = run_chain(m, EXP, EYP, x0P, 2, bits, args.kex,
                                       mu=args.arm.endswith('m'),
                                       tf_prob=1.0, chain=chP,
                                       aexplicit=args.aexplicit, A=AvP)
                    e2tf = float((((p2 - chP[:, :2]).norm(dim=-1)
                                   / chP[:, :2].norm(dim=-1).clamp_min(1e-6))
                                  .amax(1)).median())
                if gate_dep:                   # deployment: self-feedback final-step error at the current depth
                    if args.arm == 'latq':
                        pd, hd = run_chain(m, EXP, EYP, x0P, tmax, bits,
                                           args.kex, mu=True, collect_h=True,
                                           aexplicit=args.aexplicit, A=AvP,
                                           fb_bins=True)
                        pd = dequant_idx(
                            m.W_bins(hd).view(*hd.shape[:-1], D, 32)
                            .argmax(-1), 5)
                    else:
                        pd = run_chain(m, EXP, EYP, x0P, tmax, bits,
                                       args.kex,
                                       mu=args.arm.endswith('m'),
                                       aexplicit=args.aexplicit, A=AvP)
                    edep = float((((pd - chP[:, :tmax]).norm(dim=-1)
                                   / chP[:, :tmax].norm(dim=-1).clamp_min(1e-6))
                                  [:, -1]).median())
            if gate_tf:                        # step down one notch (0.05 per notch) only after two consecutive passes
                tf_win.append(e2tf < 0.30)
                if len(tf_win) > 2:
                    tf_win.pop(0)
                if tf_win == [True, True] and tf_held > 0.0:
                    tf_held = max(0.0, round(tf_held - 0.05, 2))
                    tf_win.clear()
                    print(f"[gate-tf] it={it} e2_tf={e2tf:.3f} "
                          f"tf_p->{tf_held:.2f}", flush=True)
            if gate_dep:                       # advancement thresholds tighten with depth (same shape as dwell2)
                # audit design#4 recalibration: the measured e1 floor of the μ-law 32-bin readout is
                # ~0.102; the old 0.06 notch sat below the floor and the 0.12 notch left only 0.018
                # => the depth gate was permanently blocked; raised above the floor
                thr = (0.50 if tmax <= 4 else 0.30 if tmax <= 8
                       else 0.20 if tmax <= 12 else 0.15)
                if tmax < TMAX and edep < thr:
                    dep_lvl += 1
                    last_adv = it
                    print(f"[gate-dep] it={it} e_fin@{tmax}={edep:.3f}<{thr} "
                          f"tmax->{min(TMAX, dep_lvl)}", flush=True)
                if (tmax < TMAX and it - last_adv >= 3000
                        and (it - last_adv) % 3000 == 0):
                    print(f"[gate-dep] STALL it={it} {it-last_adv} steps since last advance "
                          f"tmax={tmax} e_fin={edep:.3f}", flush=True)
            print(f"[gate-probe] it={it} tmax={tmax} tf_p={tf_held:.2f} "
                  f"e2_tf={e2tf:.3f} e_fin@tmax={edep:.3f}", flush=True)
        if it >= args.warmup and it % 500 == 0:      # round 4 #4: identity-forgetting probe
            with torch.no_grad():
                Avi = None                            # audit A1 call site 5: the identity probe needs aexplicit/A too
                if args.aexplicit:
                    EXi, EYi, x0i, chaini, Avi = gen_episode(
                        args.bs, sg, args.kex, ident=True, want_A=True)
                    Avi = Avi.to(dev)
                else:
                    EXi, EYi, x0i, chaini = gen_episode(
                        args.bs, sg, args.kex, ident=True)
                EXi, EYi, x0i, chaini = (t.to(dev) for t in
                                         (EXi, EYi, x0i, chaini))
                pi = run_chain(m, EXi, EYi, x0i, 4, bits, args.kex,
                               mu=args.arm.endswith('m') or args.arm == 'latq',
                               fb_bins=args.arm == 'latq',
                               aexplicit=args.aexplicit, A=Avi)
                ei = ((pi - chaini[:, :4]).norm(dim=-1)
                      / chaini[:, :4].norm(dim=-1).clamp_min(1e-6))
                print(f'[probe-id] it={it} e_id_med={float(ei.median()):.4f} '
                      f'e_id_t4={float(ei[:, -1].median()):.4f}', flush=True)
        if args.ckevery and (it + 1) % args.ckevery == 0 and it + 1 < args.steps:
            ms = f'{tgt}_m{(it + 1) // 1000}k'   # milestone: add-only, never overwrite (protocol zero)
            m.lm.save_pretrained(ms)
            import json as _json
            _json.dump({'argv': vars(args), 'ts': time.strftime('%Y%m%d_%H%M%S'),
                        'it': it + 1},
                       open(f'{ms}/recipe.json', 'w'), indent=1, default=str)
            exd = {'W_in': m.W_in.state_dict(), 'W_out': m.W_out.state_dict()}
            if hasattr(m, 'W_bins'):
                exd['W_bins'] = m.W_bins.state_dict()
            if hasattr(m, 'Khead'):     # audit math#19: Khead was not saved => the v17 rule head cannot be reproduced from the ckpt
                exd['Khead'] = m.Khead.state_dict()
            torch.save(exd, f'{ms}/extras.pt')
            print(f'[ckpt-m] {ms}', flush=True)
    import os, shutil                               # time is already imported at module level; do not re-import inside the function
    if os.path.exists(tgt):                     # protocol zero: automatic backup before overwriting
        os.makedirs('backups', exist_ok=True)
        bk = f'backups/{os.path.basename(tgt)}_{time.strftime("%m%d_%H%M%S")}'
        shutil.copytree(tgt, bk)
        print(f'[backup-p0] {tgt} -> {bk}', flush=True)
    m.lm.save_pretrained(tgt)
    import json as _json
    _json.dump({'argv': vars(args), 'ts': time.strftime('%Y%m%d_%H%M%S')},
               open(f'{tgt}/recipe.json', 'w'), indent=1, default=str)
    exd = {'W_in': m.W_in.state_dict(), 'W_out': m.W_out.state_dict()}
    if hasattr(m, 'W_bins'):
        exd['W_bins'] = m.W_bins.state_dict()
    if hasattr(m, 'Khead'):             # audit math#19: Khead not saved (same as above)
        exd['Khead'] = m.Khead.state_dict()
    torch.save(exd, f'{tgt}/extras.pt')
    print(f'[ckpt] dyn_{args.arm}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['train'])
    ap.add_argument('--arm', default='latent', choices=list(BITS))
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--gates', default='',
                    help="v9 mastery gates: 'tf','depth' or 'tf,depth'; empty = keep the clock")
    ap.add_argument('--ckevery', type=int, default=0,
                    help='milestone checkpoint interval (steps, 0 = off); turn on for long runs')
    ap.add_argument('--accum', type=int, default=1,
                    help='gradient-accumulation micro-batches (v10: bs_eff=bs*accum)')
    ap.add_argument('--afam', default='ortho',
                    choices=['ortho', 'perm', 'sperm', 'circ'],
                    help='TCA A-structure curriculum: permutation / signed permutation / general orthogonal')
    ap.add_argument('--aexplicit', action='store_true',
                    help="TCA diagnostic arm: A's 8 rows injected directly as tokens (bypasses inference)")
    ap.add_argument('--nca', action='store_true',
                    help='v17: context-conditioned NCA (circ task + rule-head kernel-bin supervision)')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--iface', default='random', choices=['random', 'aligned'])
    ap.add_argument('--warmup', type=int, default=0)     # identity-map warm-start steps
    ap.add_argument('--kex', type=int, default=K_EX)     # number of in-context examples
    ap.add_argument('--scratch', action='store_true')    # v8: scratch twin
    ap.add_argument('--lrpeak', type=float, default=1e-4)
    ap.add_argument('--tag', default='')                 # checkpoint-name suffix to avoid collisions
    args = ap.parse_args()
    if args.nca and args.arm != 'latq':    # A3-R1: invalid combination intercepted immediately (the old version only raised NameError at it=0)
        ap.error('--nca only supports --arm latq')
    {'train': run_train}[args.cmd](args)
