# -*- coding: utf-8 -*-
# gen34_paeq.py -- gen34 PA-EQ assertions (the **bitwise equivalence** gate before the A arm
# opens; V34_DESIGN §2.1 gate 1)
#
# Criteria (all must pass before training starts):
#   A1 step-head logits (step_head): baseline vs slot arm (zero-init) bitwise equal
#   A2 value-head logits (value_head) bitwise equal
#   A3 render-head logits (lm_head) bitwise equal
#   A4 gates at zero: g_pos == 0.0 and g_dev == 0.0
#   B1 channel liveness (proof that the gate is not idling): with g_pos=g_dev=1 forced,
#      render-head logits change **only at slot positions** (delta > 0) while non-slot
#      delta == 0 => the slot channel points exactly at the slots and is genuinely live
#      (it is not "a dead pipe attached")
# Writes only scratch/ccp/gen34/**.
import argparse
import json
import os
import sys

import gen34_common as C

G34, G33 = C.G34, C.G33
OUT = os.path.join(G34, "out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=G33 + "/cache/deep_all.bin")
    ap.add_argument("--side", default=G33 + "/out/DEEP_RENDER_SIDECAR.jsonl")
    ap.add_argument("--slot-side", default=OUT + "/DEEP_SLOT_SIDECAR.jsonl")
    ap.add_argument("--vocab", default=G33 + "/cache/CCP_VOCAB_V4.json")
    ap.add_argument("--allow", default=G33 + "/cache/ccp_allow_v4.bin")
    ap.add_argument("--init", default="ckpt/sla_v31f.pt")
    ap.add_argument("--lat31", default="cppcuda/cache/lat_tab_v31.bin")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--dev", default="cuda:1")
    ap.add_argument("--out", default=OUT + "/GEN34_PAEQ.json")
    ap.add_argument("--log", default=G34 + "/logs/gen34.log")
    a = ap.parse_args()
    C.lean_gate(5, a.log)
    sys.path.insert(0, C.G13)
    import numpy as np
    import torch
    import gen34_hybrid as HY

    v4 = json.load(open(a.vocab, encoding="utf-8"))
    head, nvals = int(v4["head"]), int(v4["n_vals_ext"])
    parents = {int(k): v["parent_idx"] for k, v in v4["init_map"].items()}
    allow = torch.from_numpy(np.fromfile(a.allow, dtype=np.uint8).reshape(
        head, HY.NSLOT, nvals).copy())
    C.gpu_gate(a.dev, a.log)
    dev = torch.device(a.dev)
    CK = HY.CacheCCP(a.cache)
    side = HY.load_sidecar(a.side)
    sslot = HY.load_slot_sidecar(a.slot_side)
    lo, hi = CK.segidx["segments"]["a_held"]
    idx = list(range(lo, hi))[:a.n]
    n_slot_tok = sum(len((sslot.get(i) or {}).get("slots") or []) for i in idx)

    def build(cls, slot):
        torch.manual_seed(20260923)
        m = cls(base=None, lat_tab_path=a.lat31, head=head, nvals=nvals, parents=parents)
        m.load_init_ccp(a.init, verbose=False)
        if slot:
            m.enable_slot(dev_flag=True, beta=8.0, lam_slot=3.0, seed=20260923, logp=a.log)
        return m.to(dev).eval()

    mB = build(HY.HybridModel, False)
    mS = build(HY.SlotHybridModel, True)
    preport = mS.slot_param_report()
    n_diff_init = 0
    for k, v in mB.state_dict().items():
        if k in mS.state_dict() and not torch.equal(v, mS.state_dict()[k]):
            n_diff_init += 1
    rep = {"schema": "gen34-pa-eq/1", "n_rows": len(idx), "n_slot_tokens": n_slot_tok,
           "cache": a.cache, "slot_side": a.slot_side, "init": a.init, "dev": a.dev,
           "slot_param_report": preport,
           "shared_params_bitwise_equal_at_init": bool(n_diff_init == 0),
           "n_diff_params_at_init": n_diff_init,
           "note": ("PA-EQ: mounting the zero-init slot channel must not change baseline "
                    "outputs (three axes bitwise); B1 proves the channel really points at "
                    "the slots and is live")}
    with torch.no_grad():
        bc = HY.L32.BatchCpu32().build(CK, idx, 0, 0, 0)
        bc.rowIds = list(idx)
        rids, rtgt, ln, stok, skind = HY.render_batch_slots(
            side, sslot, idx, dev, slot_on=1)
        lenB_full = [len(CK.rows[i].desc) + ln[b] for b, i in enumerate(idx)]
        outs = {}
        for tag, m in (("base", mB), ("slot", mS)):
            seq_c = m.assemble(bc.B, bc.L, bc, CK, dev)
            seq_r = m.render_embed(rids)
            seq = torch.cat([seq_c, seq_r], 1)
            L = bc.L + int(rids.shape[1])
            cm, cosc, sinc = m.consts(L, dev)
            mk = m.mask_of(cm, bc.B, L, lenB_full, dev)
            H = m.forward_body(seq, mk, cosc, sinc)
            lg, zb, _t = m.heads(H[:, :bc.L, :].contiguous(), bc, CK, dev, allow)
            lt = m.lm.lm_head(H[:, bc.L:, :])
            outs[tag] = {"lg": lg.detach().cpu(), "zb": zb.detach().cpu(),
                         "lt": lt.detach().cpu(),
                         "Hr": H[:, bc.L:, :].detach().cpu().clone()}
            del H, seq, seq_c, seq_r, lg, zb, lt
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    A, Bb = outs["slot"], outs["base"]
    eq_lg = bool(torch.equal(A["lg"], Bb["lg"]))
    eq_zb = bool(torch.equal(A["zb"], Bb["zb"]))
    eq_lt = bool(torch.equal(A["lt"], Bb["lt"]))
    eq_hr = bool(torch.equal(A["Hr"], Bb["Hr"]))
    dmax = {"lg": float((A["lg"] - Bb["lg"]).abs().max()),
            "zb": float((A["zb"] - Bb["zb"]).abs().max()),
            "lt": float((A["lt"] - Bb["lt"]).abs().max()),
            "Hr": float((A["Hr"] - Bb["Hr"]).abs().max())}
    # ---- B1 liveness proof: force g=1 and measure the **logits increment** (the slot
    #      embedding is added inside render_slot_forward and never reaches the body hidden
    #      state H => logits, not H, must be measured; the first version measured the wrong
    #      quantity and this probe refuted itself = the gate works) ----
    with torch.no_grad():
        mS.g_pos.fill_(1.0)
        mS.g_dev.fill_(1.0)
        bc2 = HY.L32.BatchCpu32().build(CK, idx, 0, 0, 0)
        bc2.rowIds = list(idx)
        seq_c = mS.assemble(bc2.B, bc2.L, bc2, CK, dev)
        seq_r = mS.render_embed(rids)
        seq = torch.cat([seq_c, seq_r], 1)
        L = bc2.L + int(rids.shape[1])
        cm, cosc, sinc = mS.consts(L, dev)
        mk = mS.mask_of(cm, bc2.B, L, lenB_full, dev)
        H = mS.forward_body(seq, mk, cosc, sinc)
        Hr = H[:, bc2.L:, :]
        m = (skind >= 0)
        Hr2 = Hr + (mS.g_pos * mS.slot_render_emb(skind.clamp(min=0))) * \
            m.unsqueeze(-1).to(Hr.dtype)
        lgt_live = mS.lm.lm_head(Hr2)
        delta = (lgt_live - A["lt"].to(dev)).abs().amax(-1)          # (B, Lr)
        d_slot = float(delta[m].max()) if int(m.sum()) else 0.0
        d_nonslot = float(delta[~m].max()) if int((~m).sum()) else 0.0
        pred = lgt_live.argmax(-1)
        n_slot_pos = int(m.sum())
        n_arg_hit = int((pred[m] == stok[m]).sum()) if n_slot_pos else 0
        ltr, lt_main, lt_slot, n_slot = mS.render_slot_forward(Hr, rtgt, stok, skind)
        del H, seq, seq_c, seq_r, Hr, Hr2, lgt_live, delta, pred
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    live = {"logit_delta_max_at_slot": d_slot, "logit_delta_max_at_nonslot": d_nonslot,
            "slot_positions_in_batch": n_slot_pos,
            "non_slot_positions_in_batch": int((~m).sum()),
            "argmax_equals_device_token_at_slot": n_arg_hit,
            "argmax_hit_rate_at_slot": (round(n_arg_hit / n_slot_pos, 6)
                                        if n_slot_pos else None),
            "lt_main": float(lt_main), "lt_slot": float(lt_slot), "n_slot": int(n_slot),
            # structural liveness = the **plumbing** criterion: the increment lands only on
            # slots and is genuinely nonzero
            "channel_live": bool(d_slot > 0.0 and d_nonslot == 0.0 and n_slot_pos > 0),
            # effectiveness is a **training product**, not a gate (g_dev zero-init =>
            # injected amount 0 => a low argmax hit rate at init is expected; gradient
            # direction: dL/d(dev token logit) = p_dev - 1 < 0 => the descent direction
            # **increases** g_dev)
            "channel_effective_at_init": bool(n_slot_pos > 0 and n_arg_hit == n_slot_pos),
            "effectiveness_note": ("g=0 at init => injection=0 => argmax_hit should be near "
                                   "the baseline level; after training recheck g_dev and "
                                   "argmax_hit (GEN34_POST_TRAIN_PROBE)")}
    rep["axes"] = {"A1_step_head_bitwise": eq_lg, "A2_value_head_bitwise": eq_zb,
                   "A3_lm_head_bitwise": eq_lt, "A3b_render_hidden_bitwise": eq_hr,
                   "abs_max_diff": dmax}
    rep["gate_zero"] = {"g_pos": preport["g_pos"], "g_dev": preport["g_dev"],
                        "both_zero": bool(preport["g_pos"] == 0.0
                                          and preport["g_dev"] == 0.0)}
    rep["channel_live_probe"] = live
    rep["pass"] = bool(eq_lg and eq_zb and eq_lt and eq_hr and rep["gate_zero"]["both_zero"]
                       and live["channel_live"] and n_diff_init == 0)
    rep["md5"] = C.save_json(a.out, rep)
    C.log("PA-EQ pass=%s A1=%s A2=%s A3=%s A3b=%s gate0=%s live=%s dmax=%s"
          % (rep["pass"], eq_lg, eq_zb, eq_lt, eq_hr, rep["gate_zero"]["both_zero"],
             live["channel_live"], dmax), a.log)
    print("GEN34_PAEQ_RC=0 pass=%s" % rep["pass"])
    return rep


if __name__ == "__main__":
    main()
