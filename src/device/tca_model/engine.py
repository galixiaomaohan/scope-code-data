"""engine.py -- TCA v4 (clean version, 2026-09-18).

Distilled from the research engine (now at ../research/engine/tca_engine.py, the adjudication
chain over 104 readings): **keep every empirically positive item, drop every empirically negative
or zero-effect item.**

Adopted (on by default):
  - skip identity shortcut (--no-skip turns it off): mqar paired difference −0.22/−0.42 reproduced
    on two seeds, wt2 −0.02 / dose +0.017 (attribution in research/docs/DEVIATIONS.md §17) is
    harmless, P16 +0.09 short-range tax (R2 final review)
  - adapt_a adaptation term (--no-adapt turns it off): P16 −0.26 / P512 −0.12 (wave5 final review)
Adopted (optional switches, off by default):
  - --delta --delta-gamma 0.99: synthetic-binding-only part (old mqar −0.84, zero harm on the
    content side; unstable at 5M on the language side -- use in synthetic domains only)
  - --chunk 256 --tw 256: long-credit switch (P512 −0.35/P256 −0.10; mqar +0.39 domain swap after
    compensation; ~8x slower, mutually exclusive with --cgraph)
  - --tf32 / --cgraph / --adam-fused: throughput switches (+4.9% / +4.5% numerically equivalent
    ΔNLL=0.0003 / optimizer switch)
Dropped (closed lines of the research engine, all with final-review evidence):
  hold/hm/mring/dsc/swm/pll/cam/phase_ring/long_delay/sparse_key/gumbel/
  neuromod/inh/replay/oracle/lb/tau-anneal/aux-phase/rank-reg/alpha-train
  -- evidence chain in ../research/docs/DEVIATIONS.md §7–16 and REPORT_brain_arch_AUDIT.md.

Bit-level verification (against the research engine, mqarV2_K32 50k tokens seed101):
  this engine --no-skip --no-adapt ≡ research engine defaults (10.6165)
  this engine defaults ≡ research engine --skip --adapt-a

Discipline (carried over from the research chain's iron rules):
  all readout contributors enter with unit norm projected by 0.02; evaluation settings must be
  paired with training settings (cross-config eval costs +0.38 empirically); paired-difference
  decisions within the same chain and protocol; equal-step protocols (when comparing different
  chunk/replay arms the optimizer steps must be equalized).

Usage:
    python engine.py --data <corpus_dir> --tokens 1e6 --seed 101 --bs 48 --tag A1
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

C, N, CN, D_B, D_EMB, V = 8, 128, 1024, 4096, 384, 33278
E_NN = 2 * C
MAX_DELAY = 8
CLOCK_PERIOD = {0: 16, 1: 64, 2: 512}
EDGES = [(j, j) for j in range(C)] + [(j, (j + 1) % C) for j in range(C)]
# constants (include/tca.cuh)
LAM, BETA, ALPHA = 0.72, 0.82, 0.55
TAU_S, TAU_C, TAU_SOFT = 0.22, 0.08, 0.15
K_GATE, K_STAGE = 12, 4
GATE_FLOOR, KAPPA = 0.10, 8.0
SAL_W_ERR, SAL_W_NOV, AUX_W = 1.0, 0.9, 0.1
EPS2_P, REC_DENSITY = 0.85, 0.25


def draw_normal(g, *sh):
    return torch.empty(*sh).normal_(0.0, 1.0, generator=g)


def draw_uniform(g, *sh):
    return torch.empty(*sh).uniform_(0.0, 1.0, generator=g)


class TCAP(nn.Module):
    """TCA v4: base + skip (on by default) + adaptation term (on by default) + optional delta
    memory."""

    def __init__(self, seed=101, bus_init_scale=None,
                 delta=False, delta_gamma=0.99,
                 skip=True, adapt_a=True):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        bus_s = bus_init_scale  # None = the frozen version's per-stage scaling
        s_in = 2.0 / math.sqrt(D_EMB)
        s_fb = 2.0 / math.sqrt(D_B)
        s_ro = 1.0 / math.sqrt(N)
        s_rw = 0.5 / math.sqrt(REC_DENSITY * N)
        self.w_in = nn.Parameter(draw_normal(g, D_EMB, CN) * s_in)
        self.w_fb = nn.Parameter(draw_normal(g, D_B, CN) * (s_fb if bus_s is None else bus_s))
        self.r_out = nn.Parameter(draw_normal(g, C, N, D_B) * s_ro)
        self.w_bus2emb = nn.Parameter(draw_normal(g, D_B, D_EMB) * (0.02 if bus_s is None else bus_s))
        self.sal_head = nn.Parameter(draw_normal(g, C, D_B) * (0.02 if bus_s is None else bus_s))
        self.o_v = nn.Parameter(draw_normal(g, 32, D_B) * s_ro)
        self.o_u = nn.Parameter(draw_normal(g, 32, D_EMB) * (1.0 / math.sqrt(D_B)))
        self.rec_w = nn.Parameter(draw_normal(g, E_NN, N, N) * s_rw)
        self.theta_phase = nn.Parameter(draw_uniform(g, C, N) * 2 * math.pi - math.pi)
        self.embed = nn.Embedding(V, D_EMB)
        with torch.no_grad():
            self.embed.weight.copy_(draw_normal(g, V, D_EMB))
        # delta frozen key-value lookup table (orthogonality between keys is guaranteed by
        # construction, immune to embedding anisotropy)
        self.delta = delta
        self.delta_gamma = delta_gamma
        if delta:
            self.register_buffer("fz_emb", self.embed.weight.detach().clone())
        # buffers
        self.register_buffer("theta_v", 0.30 + 0.05 * draw_normal(g, C, N))
        self.register_buffer("rec_mask",
                             (draw_uniform(g, E_NN, N, N) < REC_DENSITY).float())
        self.register_buffer("delay_ids",
                             torch.bitwise_left_shift(
                                 torch.ones(E_NN, N, N, dtype=torch.long),
                                 torch.empty(E_NN, N, N, dtype=torch.long)
                                 .random_(0, 4, generator=g)))
        self.register_buffer("band",
                             torch.empty(C, N, dtype=torch.long).random_(0, 3, generator=g))
        self.register_buffer("edge_src", torch.tensor([s_ for (s_, _) in EDGES]))
        self.register_buffer("edge_dst", torch.tensor([d_ for (_, d_) in EDGES]))
        self.register_buffer("periods", torch.tensor([CLOCK_PERIOD[b_] for b_ in range(3)]))
        # auxiliary phase scaffold parameters: the research engine's unconditional first four g2
        # draws. Kept = bit-level alignment with the research chain (the position of the skip/delta
        # initialization draws in the g2 stream is unchanged); unused in the forward pass, no
        # gradient.
        g2 = torch.Generator().manual_seed(seed + 999)
        self.aux_proj = nn.Parameter(draw_normal(g2, 64, D_B) * (1.0 / math.sqrt(D_B)))
        self.aux_h16 = nn.Parameter(draw_normal(g2, 16, 64) * (1.0 / 8.0))
        self.aux_h64 = nn.Parameter(draw_normal(g2, 64, 64) * (1.0 / 8.0))
        self.aux_h512 = nn.Parameter(draw_normal(g2, 512, 64) * (1.0 / 8.0))
        if delta:
            self.W_dro = nn.Parameter(draw_normal(g2, D_EMB, D_EMB) * 0.02)
        # adopted 1: adaptation term a<-βa+α·s_prev (α=0.55, C++ canonical forward.cu:43 semantics)
        self.adapt_a = adapt_a
        # adopted 2: skip identity shortcut (current token embedding wired straight to the readout;
        # unit norm projected by 0.02)
        self.skip = skip
        if skip:
            self.W_skro = nn.Parameter(draw_normal(g2, D_EMB, D_EMB) * 0.02)
        # gate-temperature runtime scale (the --cgraph capture path requires the attribute to exist)
        self.tau_scale = 1.0

    def bus_params(self):
        return [self.w_fb, self.w_bus2emb, self.sal_head]

    def phases_at(self, t):
        # t may be a python int or a 0-dim tensor (--cgraph capture needs this: a scalar would be
        # baked into the graph)
        tf = t if torch.is_tensor(t) else torch.tensor(float(t),
                                                       device=self.periods.device)
        frac = torch.remainder(tf.float(), self.periods.float()) / self.periods.float()
        return (2.0 * math.pi * frac)[self.band]

    def _rec_contrib(self, slot, w_eff):
        """Block-sparse recurrence contribution: assemble w_eff [E_NN,N,N] into the [C(dst),C(src)]
        block matrix, then einsum."""
        Wd = slot.new_zeros(C, C, N, N)
        Wd[self.edge_dst, self.edge_src] = w_eff
        return torch.einsum("bsn,csnm->bcm", slot.view(-1, C, N), Wd).reshape(slot.shape)

    def init_states(self, B, device):
        st = dict(u=torch.zeros(B, C, N, device=device),
                  a=torch.zeros(B, C, N, device=device),
                  bus=torch.zeros(B, D_B, device=device),
                  bus_ema=torch.zeros(B, D_B, device=device),
                  hist=torch.zeros(B, MAX_DELAY, CN, device=device))
        if self.adapt_a:
            st["s_prev"] = torch.zeros(B, C, N, device=device)
        if self.delta:
            st["Am"] = torch.zeros(B, D_EMB, D_EMB, device=device)  # gradient-free runtime buffer
            st["prev_tok"] = torch.zeros(B, dtype=torch.long, device=device)
        return st

    def step(self, tok, st, noise_on, step_idx):
        B = tok.shape[0]
        emb = self.embed(tok)
        bus = st["bus"]
        syn = emb @ self.w_in + bus @ self.w_fb
        if noise_on:
            # ε2 synaptic ST: each active synapse is kept with probability eps2_p (0/1 sampling);
            # the backward pass passes through naturally
            for d in (1, 2, 4, 8):
                slot = st["hist"][:, d - 1]
                p = self.rec_mask * (self.delay_ids == d)
                mh = torch.bernoulli(p * EPS2_P)
                syn = syn + self._rec_contrib(slot, self.rec_w * mh)
        else:
            # a5 straight-through (same expression as the ref_grads reference)
            for d in (1, 2, 4, 8):
                p = self.rec_mask * (self.delay_ids == d)
                syn = syn + self._rec_contrib(st["hist"][:, d - 1], self.rec_w * p)
        u = LAM * st["u"] + (1 - LAM) * syn.view(B, C, N)
        a = BETA * st["a"] + (ALPHA * st["s_prev"] if self.adapt_a else 0.0)
        drive = u - a - self.theta_v
        rho0 = torch.sigmoid(drive / TAU_S)
        kth = torch.kthvalue(drive, N - K_GATE + 1, dim=-1).values
        win = torch.sigmoid((drive - kth.detach().unsqueeze(-1))
                            / (TAU_SOFT * self.tau_scale))
        rho = (rho0 * (GATE_FLOOR + (1 - GATE_FLOOR) * win)).clamp(1e-6, 1 - 1e-6)
        if noise_on:  # ε1 Bernoulli-ST: binary forward, backward as if s=rho
            hard = torch.bernoulli(rho)
            s = rho + (hard - rho).detach()
        else:
            s = rho
        phi = self.phases_at(step_idx).unsqueeze(0)
        dphi = torch.atan2(torch.sin(phi - self.theta_phase),
                           torch.cos(phi - self.theta_phase))
        gph = torch.exp(KAPPA * (torch.cos(dphi) - 1.0))
        h = s * gph
        rho_c = rho - rho.mean(dim=-1, keepdim=True)
        c = F.normalize(torch.einsum("bcn,cnd->bcd", rho_c, self.r_out), dim=-1)
        yh = F.normalize(torch.einsum("bcd,rd->bcr", c, self.o_v) @ self.o_u, dim=-1)
        nxt = F.normalize(self.embed(tok).detach(), dim=-1)
        err = 1 - (yh * nxt.unsqueeze(1)).sum(-1).clamp_min(0)
        nov = 1 - (c * st["bus_ema"].unsqueeze(1)).sum(-1).clamp_min(0)
        sal = (SAL_W_ERR * err.detach() + SAL_W_NOV * nov
               + 0.1 * torch.einsum("bcd,cd->bc", c, self.sal_head))
        top = sal.topk(K_STAGE, dim=-1).indices
        soft = torch.softmax(sal.gather(-1, top) / (TAU_C * self.tau_scale), -1)
        sel = c.gather(1, top.unsqueeze(-1).expand(-1, -1, D_B))
        bus_sum = (soft.unsqueeze(-1) * sel).sum(1)
        bus_next = F.normalize(bus_sum, dim=-1)
        hist = torch.roll(st["hist"], 1, dims=1).clone()
        hist[:, 0] = h.reshape(B, CN)
        pooled_pre = c.mean(dim=1)
        if self.delta:
            # Delta rule: erase old, write new (frozen-key version): key = previous token,
            # value/query = current token
            with torch.no_grad():
                k_d = F.normalize(self.fz_emb[st["prev_tok"]], dim=-1)
                v_d = F.normalize(self.fz_emb[tok], dim=-1)
                pre = torch.einsum("bde,bd->be", st["Am"], k_d)
                Am2 = self.delta_gamma * (st["Am"] + torch.einsum(
                    "bd,be->bde", v_d - pre, k_d))
                r_delta = torch.einsum("bde,bd->be", Am2, v_d)
                # unit norm (engine readout convention: contributor norm ~1 enters projected by 0.02)
                r_delta = r_delta / (torch.sqrt(
                    r_delta.pow(2).mean(-1, keepdim=True) + 1e-6) * math.sqrt(D_EMB))
        st_out = dict(u=u, a=a, bus=bus_next,
                      bus_ema=0.95 * st["bus_ema"] + 0.05 * bus.detach(),
                      hist=hist)
        if self.adapt_a:
            st_out["s_prev"] = s
        if self.delta:
            st_out["Am"] = Am2
            st_out["prev_tok"] = tok
        return dict(emb=emb, pooled_pre=pooled_pre, err=err, bus_sum=bus_sum,
                    bus_next=bus_next,
                    delta_r=(r_delta if self.delta else None),
                    skip_feat=(F.normalize(emb, dim=-1) if self.skip else None),
                    sal=sal), st_out

    def readout(self, aux):
        """v4 identity = pooled readout + skip identity shortcut + (optional) delta memory
        readout."""
        p_pool = F.normalize(aux["pooled_pre"], dim=-1) @ self.w_bus2emb
        if self.delta and aux.get("delta_r") is not None:
            p_pool = p_pool + aux["delta_r"] @ self.W_dro
        if self.skip and aux.get("skip_feat") is not None:
            p_pool = p_pool + aux["skip_feat"] @ self.W_skro
        return p_pool


def load_corpus(data_dir):
    vocab = json.load(open(os.path.join(data_dir, "vocab.json"), encoding="utf-8"))
    unk = vocab.get("<unk>", 0)
    def enc(fn):
        toks = []
        with open(os.path.join(data_dir, fn), encoding="utf-8") as f:
            for line in f:
                for w in line.split():
                    toks.append(vocab.get(w, unk))
        return torch.tensor(toks, dtype=torch.long)
    return enc("train.txt"), enc("valid.txt")


def make_batches(toks, B, T, seed):
    g = torch.Generator().manual_seed(seed)
    L = len(toks)
    cursors = torch.empty(B, dtype=torch.long).random_(0, max(L - T - 1, 1), generator=g).tolist()
    def nxt():
        x = torch.zeros(B, T, dtype=torch.long, device=toks.device)
        y = torch.zeros(B, T, dtype=torch.long, device=toks.device)
        start = list(cursors)
        for b in range(B):
            idx = [(cursors[b] + i) % L for i in range(T)]
            x[b] = toks[torch.tensor(idx, device=toks.device)]
            y[b] = toks[torch.tensor([(i + 1) % L for i in idx], device=toks.device)]
            cursors[b] = (cursors[b] + T) % max(L - T - 1, 1)
        return x, y, start
    return nxt


@torch.no_grad()
def eval_nll(model, valid, T=256, bs=16):
    """rows = distinct windows (16 rows walk 16 different chunks in parallel; the clock = each
    window's global position s+t)."""
    model.eval()
    starts = list(range(0, max(len(valid) - T - 1, 1), T))
    nlls, ntok = 0.0, 0
    for w0 in range(0, len(starts), bs):
        grp = starts[w0:w0 + bs]
        if len(grp) < 2:
            break
        rows = len(grp)
        x = torch.stack([valid[s:s + T] for s in grp]).to(valid.device)
        st = model.init_states(rows, valid.device)
        for t in range(T):
            aux, st = model.step(x[:, t], st, False, grp[0] + t)
            logits = model.readout(aux) @ model.embed.weight.t()
            y = torch.stack([valid[s + t + 1] for s in grp]).to(valid.device)
            nlls += F.cross_entropy(logits, y, reduction="sum").item()
        ntok += rows * T
    return nlls / max(ntok, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokens", type=float, default=1e6)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--bs", type=int, default=48,
                    help="adjudication-protocol batch size (research scale protocol = 48)")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--wd-bus", type=float, default=None, help="None = same as --wd")
    ap.add_argument("--bus-init-scale", type=float, default=None)
    ap.add_argument("--tw", type=int, default=256, help="training window length (default 256 = canonical)")
    ap.add_argument("--delta", action="store_true",
                    help="Delta-rule memory (synthetic binding only: old mqar −0.84; unstable on the language side, do not use in language domains)")
    ap.add_argument("--delta-gamma", type=float, default=0.99,
                    help="memory decay γ (0.99 = empirical optimum; 0.999 collapses / 0.995 zero)")
    ap.add_argument("--no-skip", action="store_true",
                    help="ablation: turn off the skip identity shortcut (on by default)")
    ap.add_argument("--no-adapt", action="store_true",
                    help="ablation: turn off the adaptation term (on by default; off = research engine v3.1 semantics)")
    ap.add_argument("--tf32", action="store_true",
                    help="speed switch: TF32 (changes numeric precision; compare only against controls that also use --tf32)")
    ap.add_argument("--prof-at", type=int, default=0)
    ap.add_argument("--cgraph", action="store_true",
                    help="speed switch (numerically equivalent ΔNLL=0.0003): CUDA Graph capture/replay; state incompatibility with --delta untested, do not combine")
    ap.add_argument("--adam-fused", action="store_true")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--load")
    ap.add_argument("--save")
    ap.add_argument("--tag", default="P")
    ap.add_argument("--eval-every", type=float, default=500000)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = TCAP(seed=args.seed, bus_init_scale=args.bus_init_scale,
                 delta=args.delta, delta_gamma=args.delta_gamma,
                 skip=not args.no_skip, adapt_a=not args.no_adapt).to(dev)
    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=dev, weights_only=True))
    train, valid = load_corpus(args.data)
    train, valid = train.to(dev), valid.to(dev)
    print(f"[engine] dev={dev} train={len(train)} valid={len(valid)} "
          f"skip={not args.no_skip} adapt={not args.no_adapt} delta={args.delta}",
          flush=True)
    if args.eval_only:
        nll = eval_nll(model, valid)
        print(f"[final] valid NLL = {nll:.4f}  ppl = {math.exp(nll):.2f}")
        return
    bus_ids = [id(p) for p in model.bus_params()]
    groups = [{"params": [p for p in model.parameters() if id(p) not in bus_ids],
               "weight_decay": args.wd},
              {"params": model.bus_params(),
               "weight_decay": args.wd if args.wd_bus is None else args.wd_bus}]
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95),
                            fused=args.adam_fused)
    T = args.tw
    nxt = make_batches(train, args.bs, T, args.seed)
    total_steps = int(args.tokens) // (args.bs * args.chunk)
    warmup = max(total_steps // 40, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else
        0.5 * (1 + math.cos(math.pi * (s - warmup) / max(total_steps - warmup, 1))))
    t0 = time.time(); seen = 0; step_g = 0
    _next_rate = 20.0
    tok_g = 0                     # global token step (the clock advances every tick, matching C++ semantics)
    st = model.init_states(args.bs, dev)
    model.train()
    nchunk = T // args.chunk
    # ---- CUDA Graph capture (--cgraph; numerically equivalent ΔNLL=0.0003: same kernels, same
    # order, replayed) ----
    use_graph = args.cgraph
    if use_graph:
        assert not args.delta, "--cgraph 未与 --delta 状态联测，勿并用"
        xc_s = torch.zeros(args.bs, args.chunk, dtype=torch.long, device=dev)
        yc_s = torch.zeros_like(xc_s)
        tok_s = torch.zeros((), dtype=torch.long, device=dev)
        st_in = {k: v.clone() for k, v in st.items()}
        st_out = {k: v.clone() for k, v in st.items()}

        def _fwd():
            stc = dict(st_in)
            tot = None
            for t in range(args.chunk):
                aux, st2 = model.step(xc_s[:, t], stc, True, tok_s)
                logits = model.readout(aux) @ model.embed.weight.t()
                l = F.cross_entropy(logits, yc_s[:, t])
                tot = l if tot is None else tot + l
                stc = st2
            for k in st_out:
                st_out[k].copy_(stc[k])
            return tot
        _ws = torch.cuda.Stream()
        _ws.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(_ws):
            for _ in range(3):
                _tw = _fwd()
                (_tw / args.chunk).backward()
        torch.cuda.current_stream().wait_stream(_ws)
        opt.zero_grad(set_to_none=False)
        gph_c = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gph_c):
            tot_g = _fwd()
            (tot_g / args.chunk).backward()
        print("[cgraph] captured: chunk fwd+bwd, numerically equivalent switch", flush=True)
    while seen < args.tokens:
        x, y, curs = nxt()
        loss_acc = 0.0
        _prof = (torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA])
            if args.prof_at and step_g >= args.prof_at else None)
        if _prof is not None:
            _prof.start()
        for ci in range(0, T, args.chunk):
            xc, yc = x[:, ci:ci + args.chunk], y[:, ci:ci + args.chunk]
            if use_graph:
                xc_s.copy_(xc); yc_s.copy_(yc); tok_s.fill_(tok_g)
                for k in st_in:
                    st_in[k].copy_(st_out[k])
                opt.zero_grad(set_to_none=False)
                gph_c.replay()
                tot = tot_g
                tok_g += args.chunk
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step(); sched.step(); step_g += 1
                loss_acc += tot.item(); seen += args.bs * args.chunk
                continue
            opt.zero_grad()
            # chunked BPTT: each chunk truncates the fast state, carried across chunks (detach)
            st = {k: v.detach() for k, v in st.items()}
            tot = 0.0
            for t in range(args.chunk):
                aux, st = model.step(xc[:, t], st, True, tok_g)
                logits = model.readout(aux) @ model.embed.weight.t()
                tot = tot + F.cross_entropy(logits, yc[:, t])
                tok_g += 1
            (tot / args.chunk).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step(); sched.step(); step_g += 1
            loss_acc += tot.item(); seen += args.bs * args.chunk
        if (seen % args.eval_every) < args.bs * nchunk:
            nll = eval_nll(model, valid)
            rate = seen / (time.time() - t0)
            print(f"[{args.tag}] {seen/1e6:.2f}M tok  train {loss_acc/(nchunk*args.chunk):.4f}  "
                  f"valid {nll:.4f}  lr {sched.get_last_lr()[0]:.2e}  {rate:.0f} tok/s",
                  flush=True)
        elif args.eval_every > 5e6 and (time.time() - t0) > _next_rate:
            print(f"[{args.tag}] {seen/1e6:.2f}M tok  rate {seen/(time.time()-t0):.0f} tok/s", flush=True)
            _next_rate += 20.0
        if _prof is not None:
            _prof.stop()
            ka = _prof.key_averages()
            print(ka.table(sort_by="self_cuda_time_total", row_limit=25), flush=True)
            print(f"[prof] op_kinds={len(ka)}  chunks_profiled={nchunk}", flush=True)
            break
    nll = eval_nll(model, valid)
    print(f"[final] valid NLL = {nll:.4f}  ppl = {math.exp(nll):.2f}")
    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"[ckpt] {args.save}")


if __name__ == "__main__":
    main()
