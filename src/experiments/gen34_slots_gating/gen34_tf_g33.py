# -*- coding: utf-8 -*-
# gen34_tf_g33.py -- **same-convention teacher-forced slot-token accuracy**, three-arm
# comparison (after-the-fact diagnostic)
#
# Motivation: gen34_probe measured a teacher-forced argmax hit rate ~0.99 on **slot tokens**
#   for the A/B arms, whereas the p=0.79 reported by gen33 comes from a **free-generation**
#   convention => the two are not comparable. This artifact pulls the gen33 baseline ckpt
#   onto the **same teacher-forced slot convention** (same slot sidecar / same batch rows /
#   same mask) to give comparable readings:
#       tf_slot_acc(g33 baseline) vs tf_slot_acc(g34 A/B)
#   and reports "non-slot token accuracy" (skeleton axis) alongside, to decompose the two axes.
# Reads existing ckpts and sidecars; writes only scratch/ccp/gen34/**.
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
    ap.add_argument("--arch", default="g33", choices=["g33", "g34slot"])
    ap.add_argument("--slot-dev", type=int, default=0)
    ap.add_argument("--cache", default=G33 + "/cache/deep_all.bin")
    ap.add_argument("--side", default=G33 + "/out/DEEP_RENDER_SIDECAR.jsonl")
    ap.add_argument("--slot-side", default=OUT + "/DEEP_SLOT_SIDECAR.jsonl")
    ap.add_argument("--vocab", default=G33 + "/cache/CCP_VOCAB_V4.json")
    ap.add_argument("--allow", default=G33 + "/cache/ccp_allow_v4.bin")
    ap.add_argument("--lat31", default="cppcuda/cache/lat_tab_v31.bin")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--dev", default="cuda:2")
    ap.add_argument("--out", default="")
    ap.add_argument("--log", default=G34 + "/logs/gen34.log")
    a = ap.parse_args()
    C.lean_gate(5, a.log)
    sys.path.insert(0, C.G13)
    import numpy as np
    import torch
    import gen33_hybrid as H33
    import gen34_hybrid as H34

    v4 = json.load(open(a.vocab, encoding="utf-8"))
    head, nvals = int(v4["head"]), int(v4["n_vals_ext"])
    parents = {int(k): v["parent_idx"] for k, v in v4["init_map"].items()}
    allow = torch.from_numpy(np.fromfile(a.allow, dtype=np.uint8).reshape(
        head, H34.NSLOT, nvals).copy())
    C.gpu_gate(a.dev, a.log)
    dev = torch.device(a.dev)
    CK = H34.CacheCCP(a.cache)
    side = H34.load_sidecar(a.side)
    sslot = H34.load_slot_sidecar(a.slot_side)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    cargs = dict(sd.get("args") or {})
    if a.arch == "g33":
        m = H33.HybridModel(base=None, lat_tab_path=a.lat31, head=head, nvals=nvals,
                            parents=parents)
        m.load_state_dict(sd["model"], strict=False)
        boost = 0.0
    else:
        m = H34.SlotHybridModel(base=None, lat_tab_path=a.lat31, head=head, nvals=nvals,
                                parents=parents)
        m.enable_slot(dev_flag=bool(int(cargs.get("slot_dev", 1))),
                      beta=float(cargs.get("slot_beta", 8.0) or 8.0),
                      lam_slot=float(cargs.get("lam_slot", 3.0) or 3.0),
                      seed=int(cargs.get("seed", 20260923)))
        m.load_state_dict(sd["model"], strict=False)
        boost = float(m.g_dev) * m.slot_beta if a.slot_dev else 0.0
    m = m.to(dev).eval()
    lo, hi = CK.segidx["segments"]["a_held"]
    idx = list(range(lo, hi))[:a.n]
    with torch.no_grad():
        bc = H34.L32.BatchCpu32().build(CK, idx, 0, 0, 0)
        bc.rowIds = list(idx)
        rids, rtgt, ln = H34.render_batch(side, idx, dev)
        _t, _o, _l, stok, skind = H34.render_batch_slots(side, sslot, idx, dev, slot_on=1)
        lenB_full = [len(CK.rows[i].desc) + ln[b] for b, i in enumerate(idx)]
        seq_c = m.assemble(bc.B, bc.L, bc, CK, dev)
        seq_r = m.render_embed(rids)
        seq = torch.cat([seq_c, seq_r], 1)
        L = bc.L + int(rids.shape[1])
        cm, cosc, sinc = m.consts(L, dev)
        mk = m.mask_of(cm, bc.B, L, lenB_full, dev)
        H = m.forward_body(seq, mk, cosc, sinc)
        Hr = H[:, bc.L:, :]
        lgt = m.lm.lm_head(Hr)
        if boost:
            dm = (skind >= 0) & (stok >= 0)
            oh = torch.nn.functional.one_hot(stok.clamp(min=0), lgt.shape[-1]).to(lgt.dtype)
            lgt = lgt + (boost * oh) * dm.unsqueeze(-1).to(lgt.dtype)
        pred = lgt.argmax(-1)
        gt = rtgt
        msk = (skind >= 0) & (gt != -100)
        nmask = (skind < 0) & (gt != -100)
        n_s = int(msk.sum())
        n_n = int(nmask.sum())
        hs = int((pred[msk] == gt[msk]).sum()) if n_s else 0
        hn = int((pred[nmask] == gt[nmask]).sum()) if n_n else 0
        # free-generation vs teacher-forced: per row, "all slot tokens of that row correct"
        rows_all = int(msk.sum(1).gt(0).sum())
        rows_ok = 0
        for b in range(gt.shape[0]):
            sel = msk[b]
            if int(sel.sum()) and bool((pred[b][sel] == gt[b][sel]).all()):
                rows_ok += 1
        rep = {"schema": "gen34-tf-slot-acc/1", "ckpt": a.ckpt, "ckpt_md5": C.md5f(a.ckpt),
               "arch": a.arch, "slot_dev_flag": a.slot_dev, "boost": boost,
               "n_rows": len(idx), "seg": "a_held",
               "tooltip": ("teacher-forced convention (label = gold next token): "
                           "tf_slot_acc = slot-token hit rate; tf_skel_acc = non-slot token "
                           "hit rate; row_slot_all_correct = rows whose slot tokens are all "
                           "correct"),
               "n_slot_tok": n_s, "slot_hit": hs, "tf_slot_acc": round(hs / max(n_s, 1), 6),
               "n_skel_tok": n_n, "skel_hit": hn,
               "tf_skel_acc": round(hn / max(n_n, 1), 6),
               "row_slot_all_correct": rows_ok, "row_slot_rows": rows_all}
        outp = a.out or os.path.join(OUT, "GEN34_TFACC_%s.json" % a.arch)
        rep["md5"] = C.save_json(outp, rep)
        C.log("TFACC %s boost=%.4g slot %d/%d=%.6f | skel %d/%d=%.6f | row-all %d/%d"
              % (a.arch + "(" + os.path.basename(a.ckpt) + ")", boost, hs, n_s,
                 rep["tf_slot_acc"], hn, n_n, rep["tf_skel_acc"], rows_ok, rows_all), a.log)
    print("GEN34_TFACC_RC=0 -> %s" % outp)
    return rep


if __name__ == "__main__":
    main()
