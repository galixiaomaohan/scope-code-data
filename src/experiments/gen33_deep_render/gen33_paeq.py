# -*- coding: utf-8 -*-
# gen33_paeq.py -- PA-EQ assertion (the **bit-for-bit equivalence** gate before enabling the attention arm)
#
# Pre-registered (V33_DESIGN §5): forward passes of both arms from the same ckpt start must be
#   bit-for-bit equivalent (torch.equal level) under zero-initialized attention (pa_delta_max == 0) --
#   i.e. attaching attention at zero init must not change baseline outputs.
#
# Criteria (three axes; all must pass before the attention arm is enabled):
#   A1 chain-head logits (step_head)   bitwise equal
#   A2 value-head logits (value_head)  bitwise equal
#   A3 render-head logits (lm_head)    bitwise equal
#   ++ pa_delta_max == 0.0 and ps_delta_max == 0.0 (device self-report)
#
# Implementation: build both models (HybridModel / AttnModel) under one batch/seed with identical init (same --init for load_init_ccp),
#   pa_enable=0 / 1 (attach at zero init), single-microbatch forward -> compare.
# Writes only scratch/ccp/gen33/**.
import argparse
import json
import os
import sys

import g33_common as C

G33 = C.G33
OUT = os.path.join(G33, "out")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=G33 + "/cache/deep_all.bin")
    ap.add_argument("--side", default=OUT + "/DEEP_RENDER_SIDECAR.jsonl")
    ap.add_argument("--vocab", default=G33 + "/cache/CCP_VOCAB_V4.json")
    ap.add_argument("--allow", default=G33 + "/cache/ccp_allow_v4.bin")
    ap.add_argument("--init", default="ckpt/sla_v31f.pt")
    ap.add_argument("--lat31", default="cppcuda/cache/lat_tab_v31.bin")
    ap.add_argument("--n", type=int, default=12, help="rows to compare (first n of the held-out segment)")
    ap.add_argument("--dev", default="cuda:2")
    ap.add_argument("--out", default=OUT + "/GEN33_PAEQ.json")
    ap.add_argument("--log", default=G33 + "/logs/gen33.log")
    a = ap.parse_args()
    C.lean_gate(5, a.log)
    sys.path.insert(0, C.G13)
    import numpy as np
    import torch
    import gen33_hybrid as HY
    import gen33_planattn as PA33
    import v31_model as _vm                                                 # noqa: F401,E402

    v4 = json.load(open(a.vocab, encoding="utf-8"))
    head, nvals = int(v4["head"]), int(v4["n_vals_ext"])
    parents = {int(k): v["parent_idx"] for k, v in v4["init_map"].items()}
    allow = torch.from_numpy(np.fromfile(a.allow, dtype=np.uint8).reshape(
        head, HY.NSLOT, nvals).copy())
    C.gpu_gate(a.dev, a.log)
    dev = torch.device(a.dev)
    CK = HY.CacheCCP(a.cache)
    side = HY.load_sidecar(a.side)
    lo, hi = CK.segidx["segments"]["a_held"]
    idx = list(range(lo, hi))[:a.n]

    AttnModel = PA33.make_attn_model(HY.HybridModel)

    def build(pa):
        torch.manual_seed(20260923)
        cls = AttnModel if pa else HY.HybridModel
        m = cls(base=None, lat_tab_path=a.lat31, head=head, nvals=nvals, parents=parents)
        m.load_init_ccp(a.init)
        if pa:
            m.attach_plan_attn(seed=20260923, capture=False, enable=True,
                               arch={"use_state": True, "n_heads": 2, "d_head": 64},
                               plan_summary=True)
        return m.to(dev).eval()

    mB, mA = build(False), build(True)
    rep = {"schema": "gen33-pa-eq/1", "n_rows": len(idx), "cache": a.cache,
           "vocab": a.vocab, "init": a.init, "dev": a.dev,
           "note": "PA-EQ: 零初始化注意力挂载不得改变基线输出 (三轴逐位)"}
    with torch.no_grad():
        bc = HY.L32.BatchCpu32().build(CK, idx, 0, 0, 0)
        bc.rowIds = list(idx)
        rids, rtgt, ln = HY.render_batch(side, idx, dev)
        lenB_full = [len(CK.rows[i].desc) + ln[b] for b, i in enumerate(idx)]
        outs = {}
        for tag, m in (("base", mB), ("attn", mA)):
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
                         "pa_delta_max": float(getattr(m, "pa_delta_max", 0.0)),
                         "ps_delta_max": float(getattr(m, "ps_delta_max", 0.0))}
            del H, seq, seq_c, seq_r, lg, zb, lt
    A = outs["attn"]
    Bb = outs["base"]
    eq_lg = bool(torch.equal(A["lg"], Bb["lg"]))
    eq_zb = bool(torch.equal(A["zb"], Bb["zb"]))
    eq_lt = bool(torch.equal(A["lt"], Bb["lt"]))
    dmax = {"lg": float((A["lg"] - Bb["lg"]).abs().max()),
            "zb": float((A["zb"] - Bb["zb"]).abs().max()),
            "lt": float((A["lt"] - Bb["lt"]).abs().max())}
    rep["axes"] = {"A1_step_head_bitwise": eq_lg, "A2_value_head_bitwise": eq_zb,
                   "A3_lm_head_bitwise": eq_lt, "abs_max_diff": dmax}
    rep["device_self_report"] = {k: {"pa_delta_max": A[k], "ps_delta_max": A[k]}
                                 for k in ("pa_delta_max", "ps_delta_max")}
    rep["pa_delta_max_zero"] = bool(A["pa_delta_max"] == 0.0 and A["ps_delta_max"] == 0.0)
    rep["pass"] = bool(eq_lg and eq_zb and eq_lt and rep["pa_delta_max_zero"])
    rep["md5"] = C.save_json(a.out, rep)
    C.log("PA-EQ pass=%s A1=%s A2=%s A3=%s pa_delta=%.3g ps_delta=%.3g maxdiff=%s"
          % (rep["pass"], eq_lg, eq_zb, eq_lt, A["pa_delta_max"], A["ps_delta_max"],
             dmax), a.log)
    print("GEN33_PAEQ_RC=0 pass=%s" % rep["pass"])
    return rep


if __name__ == "__main__":
    main()
