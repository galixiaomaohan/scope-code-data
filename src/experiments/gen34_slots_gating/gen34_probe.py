# -*- coding: utf-8 -*-
# gen34_probe.py -- post-training **device-channel measurement probe** (mechanism readings;
#                  an after-the-fact artifact, not a pre-registered criterion)
#
# Purpose: PA-EQ only proves at init that "gates at zero => bitwise equivalence + channel
#   structurally live"; after training the following must be measured:
#   P1 final slot-parameter values of the ckpt (g_pos / g_dev) and the curve-tail lt_slot
#   P2 with the gate genuinely open, **can it override the LM's own logit margin** =>
#      does the slot argmax equal the device token (hit rate)
#   P3 log scale: logit rank of the device token (top-1/5/20) and margin distribution
#      (decides whether beta=8 is enough)
#   P4 (control) the same hit rate with the gate forced closed (g_dev=0) => net increment
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default=G33 + "/cache/deep_all.bin")
    ap.add_argument("--side", default=G33 + "/out/DEEP_RENDER_SIDECAR.jsonl")
    ap.add_argument("--slot-side", default=OUT + "/DEEP_SLOT_SIDECAR.jsonl")
    ap.add_argument("--vocab", default=G33 + "/cache/CCP_VOCAB_V4.json")
    ap.add_argument("--allow", default=G33 + "/cache/ccp_allow_v4.bin")
    ap.add_argument("--lat31", default="cppcuda/cache/lat_tab_v31.bin")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--dev", default="cuda:1")
    ap.add_argument("--out", default="")
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
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    cargs = dict(sd.get("args") or {})
    dev_flag = bool(int(cargs.get("slot_dev", 1)))
    m = HY.SlotHybridModel(base=None, lat_tab_path=a.lat31, head=head, nvals=nvals,
                           parents=parents)
    m.enable_slot(dev_flag=dev_flag, beta=float(cargs.get("slot_beta", 8.0) or 8.0),
                  lam_slot=float(cargs.get("lam_slot", 3.0) or 3.0),
                  seed=int(cargs.get("seed", 20260923)))
    mi, un = m.load_state_dict(sd["model"], strict=False)
    assert not [k for k in mi if k.startswith(("slot_render_emb", "g_pos", "g_dev"))], mi
    m = m.to(dev).eval()
    lo, hi = CK.segidx["segments"]["a_train"]
    idx = list(range(lo, hi))[:a.n]
    rep = {"schema": "gen34-probe-slot/1", "ckpt": a.ckpt, "ckpt_md5": C.md5f(a.ckpt),
           "step": sd.get("step"), "slot_dev_flag": dev_flag, "n_rows": len(idx),
           "slot_params": m.slot_param_report()}
    with torch.no_grad():
        bc = HY.L32.BatchCpu32().build(CK, idx, 0, 0, 0)
        bc.rowIds = list(idx)
        rids, rtgt, ln, stok, skind = HY.render_batch_slots(side, sslot, idx, dev,
                                                           slot_on=1)
        lenB_full = [len(CK.rows[i].desc) + ln[b] for b, i in enumerate(idx)]
        seq_c = m.assemble(bc.B, bc.L, bc, CK, dev)
        seq_r = m.render_embed(rids)
        seq = torch.cat([seq_c, seq_r], 1)
        L = bc.L + int(rids.shape[1])
        cm, cosc, sinc = m.consts(L, dev)
        mk = m.mask_of(cm, bc.B, L, lenB_full, dev)
        H = m.forward_body(seq, mk, cosc, sinc)
        Hr = H[:, bc.L:, :]
        msk = (skind >= 0)
        gt = rtgt.clone()
        # (A) gate at its as-trained ckpt value
        lgt = m.lm.lm_head(Hr)
        dm = msk & (stok >= 0)
        oh = torch.nn.functional.one_hot(stok.clamp(min=0), lgt.shape[-1]).to(lgt.dtype)
        lgt_on = lgt + ((m.g_dev * m.slot_beta) * oh) * dm.unsqueeze(-1).to(lgt.dtype)
        # (B) gate forced closed (g_dev := 0)
        lgt_off = lgt

        def stat(lg, tag):
            pred = lg.argmax(-1)
            sel = msk & (gt != -100)
            n = int(sel.sum())
            hit = int((pred[sel] == gt[sel]).sum()) if n else 0
            # device-token logit rank and margin
            rank1 = rank5 = rank20 = 0
            marg = []
            v = lg[dm]
            t = gt[dm]
            if v.numel():
                srt = v.argsort(dim=-1, descending=True)
                pos = (srt == t.unsqueeze(-1)).nonzero()[:, -1]
                rank1 = int((pos == 0).sum())
                rank5 = int((pos < 5).sum())
                rank20 = int((pos < 20).sum())
                top = v.gather(1, srt[:, :1]).squeeze(1)
                tv = v.gather(1, t.unsqueeze(-1)).squeeze(1)
                marg = (top - tv).tolist()
            return {"n_slot_label": n, "argmax_hit": hit,
                    "argmax_hit_rate": round(hit / max(n, 1), 6),
                    "device_tok_rank1": rank1, "device_tok_rank5": rank5,
                    "device_tok_rank20": rank20, "n_dev": int(v.shape[0]),
                    "device_rank1_rate": round(rank1 / max(1, v.shape[0]), 6),
                    "device_rank5_rate": round(rank5 / max(1, v.shape[0]), 6),
                    "margin_top1_minus_dev_median": (round(float(np.median(marg)), 4)
                                                     if marg else None),
                    "margin_top1_minus_dev_p90": (round(float(np.percentile(marg, 90)), 4)
                                                  if marg else None),
                    "tag": tag}
        rep["gate_on"] = stat(lgt_on, "g as-trained")
        rep["gate_off"] = stat(lgt_off, "g_dev:=0 (control)")
        rep["md5_partial"] = C.save_json(
            (a.out or os.path.join(OUT, "GEN34_PROBE_%s.json"
                                   % os.path.basename(a.ckpt).split("_")[1])), rep)
        eff = float(m.g_dev) * m.slot_beta
        rep["effective_boost"] = {"g_dev": float(m.g_dev), "beta": m.slot_beta,
                                  "g_dev_times_beta": eff,
                                  "sufficient_p90_margin": bool(
                                      rep["gate_off"]["margin_top1_minus_dev_p90"] is not None
                                      and eff > rep["gate_off"][
                                          "margin_top1_minus_dev_p90"])}
        for _nm in ("v", "lgt", "lgt_on", "oh", "H", "seq", "seq_c", "seq_r", "Hr"):
            if _nm in dir():
                del _nm
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    rep["delta_argmax_hit"] = (rep["gate_on"]["argmax_hit"] - rep["gate_off"]["argmax_hit"])
    outp = a.out or os.path.join(OUT, "GEN34_PROBE_%s.json"
                                 % os.path.basename(a.ckpt).split("_")[1])
    rep["md5"] = C.save_json(outp, rep)
    C.log("PROBE %s g_dev=%.4g*beta=%g=%.4g | hit g_on=%d/%d (%.4f) g_off=%d/%d (%.4f) "
          "delta=%+d | devtok rank1=%.4f rank5=%.4f | margin p90=%s"
          % (os.path.basename(a.ckpt), float(m.g_dev), m.slot_beta, eff,
             rep["gate_on"]["argmax_hit"], rep["gate_on"]["n_slot_label"],
             rep["gate_on"]["argmax_hit_rate"], rep["gate_off"]["argmax_hit"],
             rep["gate_off"]["n_slot_label"], rep["gate_off"]["argmax_hit_rate"],
             rep["delta_argmax_hit"], rep["gate_on"]["device_rank1_rate"],
             rep["gate_on"]["device_rank5_rate"],
             rep["gate_off"]["margin_top1_minus_dev_p90"]), a.log)
    print("GEN34_PROBE_RC=0 -> %s" % outp)
    return rep


if __name__ == "__main__":
    main()
