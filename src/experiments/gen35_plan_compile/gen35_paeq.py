# -*- coding: utf-8 -*-
# gen35_paeq.py -- **PA-EQ starting-point assertion** (V35_DESIGN §3 pre-training gate)
#   (a) parent (gen34_hybrid) and fork (gen35_hybrid) with the same init/seed => per-key fingerprints
#       equal + first-step logits equal (time-sliced cards: only one model resident at a time, comparing
#       per-key fingerprints (shape/sum/absmax))
#   (b) lam_txt: on the same render batch, perturbing a non-slot target makes the lam_txt=1 total loss
#       **change** (positive control; the checker is non-trivial), while lam_txt=0 leaves the total loss
#       **unchanged** (gradient cut); slot target count > 0 (sensing channel preserved)
# Writes only under scratch/ccp/gen35/**. GPU0 redline; single-GPU exclusive gate; nice -n 19.
import argparse
import json
import os
import sys
import time

import gen35_common as C

import numpy as np                                                        # noqa: E402
import torch                                                              # noqa: E402

import gen34_hybrid as H34                                                # noqa: E402
import gen35_hybrid as H35                                                # noqa: E402

OUT = C.G35 + "/out"
LOGS = C.G35 + "/logs"


def build(mod, a, dev):
    voc = json.load(open(a.vocab, encoding="utf-8"))
    head, nvals = voc["head"], voc["n_vals_ext"]
    parents = {int(k): v["parent_idx"] for k, v in voc["init_map"].items()}
    allow = torch.from_numpy(np.fromfile(a.allow, dtype=np.uint8).reshape(
        head, mod.NSLOT, nvals).copy()).to(dev)
    cache = mod.CacheCCP(a.cache)
    assert (cache.HEAD, cache.NVALS, cache.ENDIDX) == (head, nvals, mod.END31)
    torch.manual_seed(a.seed)
    model = mod.SlotHybridModel(base=a.base, lat_tab_path=a.lat31, head=head,
                                nvals=nvals, parents=parents)
    minfo = model.load_init_ccp(a.init)
    model.enable_slot(dev_flag=True, beta=a.slot_beta, lam_slot=a.lam_slot, seed=a.seed)
    model.lam_bind, model.lam_txt = a.lam_bind, 0.0
    model = model.to(dev)
    model.eval()
    return model, cache, allow, minfo


def sd_fp(model):
    """Per-key fingerprint: [shape, sum, absmax, numel] (comparable across processes under time-sliced cards)."""
    out = {}
    for k, v in model.state_dict().items():
        t = v.detach().float().cpu()
        out[k] = [list(t.shape), float(t.sum()), float(t.abs().max()), int(t.numel())]
    return out


def sd_cmp(f1, f2):
    keys = sorted(set(f1) | set(f2))
    n_eq = n_ne = 0
    bad = []
    for k in keys:
        if k in f1 and k in f2 and f1[k] == f2[k]:
            n_eq += 1
        else:
            n_ne += 1
            if len(bad) < 6:
                bad.append(k)
    return {"n_keys": len(keys), "n_equal": n_eq, "n_diff": n_ne, "bad": bad}


def fwd_logits(m, cache, allow, grp, dev):
    with torch.no_grad():
        bc = m.L32 if hasattr(m, "L32") else H35.L32
        bc = H35.L32.BatchCpu32().build(cache, list(grp), 0, 0, 0)
        bc.rowIds = list(grp)
        cm, cosc, sinc = m.consts(bc.L, dev)
        mk = m.mask_of(cm, bc.B, bc.L, bc.lenB, dev)
        seq = m.assemble(bc.B, bc.L, bc, cache, dev)
        H = m.forward_body(seq, mk, cosc, sinc)
        lg = m.heads(H, bc, cache, dev, allow)[0]
        out = lg.detach().float().cpu().clone()
        del bc, cm, cosc, sinc, mk, seq, H, lg
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=C.G33 + "/cache/deep_all.bin")
    ap.add_argument("--vocab", default=C.G33 + "/cache/CCP_VOCAB_V4.json")
    ap.add_argument("--allow", default=C.G33 + "/cache/ccp_allow_v4.bin")
    ap.add_argument("--init", default="ckpt/sla_v31f.pt")
    ap.add_argument("--base", default=None)
    ap.add_argument("--lat31", default="cppcuda/cache/lat_tab_v31.bin")
    ap.add_argument("--side", default=C.G33 + "/out/DEEP_RENDER_SIDECAR.jsonl")
    ap.add_argument("--slot-side", default=C.G34 + "/out/DEEP_SLOT_SIDECAR.jsonl")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--lam-bind", type=float, default=1.0)
    ap.add_argument("--lam-slot", type=float, default=3.0)
    ap.add_argument("--slot-beta", type=float, default=8.0)
    ap.add_argument("--dev", default="cuda:1")
    ap.add_argument("--gcb", type=int, default=2, help="batch size for the gradient-cut check (<= bs)")
    ap.add_argument("--out", default=OUT + "/GEN35_PAEQ.json")
    ap.add_argument("--log", default=LOGS + "/gen35_paeq.log")
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    t0 = time.time()
    rep = {"schema": "gen35-paeq/1", "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    C.gpu_gate(a.dev, a.log)
    led0 = C.parent_ledger()
    side = H35.load_sidecar(a.side)
    plan, pmd5 = H35.build_plan([], None, 0, 0, 0) if False else (None, None)
    # batch order: same source as training (train_idx from cache segments); first two batches only
    cache0 = H35.CacheCCP(a.cache)
    segs = (cache0.segidx or {}).get("segments", {})
    train_idx = []
    for sg, (lo, hi) in segs.items():
        if sg != "a_held":
            train_idx.extend(range(lo, hi))
    H35.set_cls_map({int(i): s["cls"] for i, s in side.items()}, range(cache0.n_rows))
    plan, pmd5 = H35.build_plan(train_idx, cache0, a.steps, a.bs, a.seed,
                                snap_path=os.path.join(C.G35, "in", "paeq_plan.txt"))
    grp = list(plan[0])
    rep["plan_md5"] = pmd5
    rep["plan_n"] = len(plan)
    rep["plan_file_md5"] = C.md5f(os.path.join(C.G35, "in", "paeq_plan.txt"))
    rep["plan_file_md5_gen34"] = C.md5f(C.G34 + "/out/batch_idx_deep_slotA.txt")
    rep["plan_md5_same_as_gen34"] = bool(rep["plan_file_md5"] == rep["plan_file_md5_gen34"])
    C.log("PA-EQ batch order internal_md5=%s file_md5=%s (gen34 A-arm file=%s) same=%s"
          % (pmd5, rep["plan_file_md5"], rep["plan_file_md5_gen34"],
             rep["plan_md5_same_as_gen34"]), a.log)
    del cache0

    m34, cache, allow, i34 = build(H34, a, a.dev)
    f34, lg34 = sd_fp(m34), fwd_logits(m34, cache, allow, grp[:a.gcb], a.dev)
    n34 = i34["n_params"]
    del m34
    torch.cuda.empty_cache()
    C.log("PA-EQ parent built (%.1fM), fingerprint %d keys; fork built after release"
          % (n34 / 1e6, len(f34)), a.log)
    m35, cache2, allow2, i35 = build(H35, a, a.dev)
    f35, lg35 = sd_fp(m35), fwd_logits(m35, cache2, allow2, grp[:a.gcb], a.dev)
    rep["params"] = {"n34": n34, "n35": i35["n_params"]}
    rep["state_dict"] = sd_cmp(f34, f35)
    rep["logits_first_step"] = {"torch_equal": bool(torch.equal(lg34, lg35)),
                                "maxdiff": float((lg34 - lg35).abs().max()),
                                "n_dim": int(lg34.numel()), "shape": list(lg34.shape),
                                "n_rows": int(len(grp[:a.gcb]))}
    del f34, lg34, lg35
    # ---- (b) render-segment text loss: alive/cut contrast (same model, same batch; non-slot target perturbed) ----
    sslot = H35.load_slot_sidecar(a.slot_side)
    mb = list(grp[:a.gcb])
    rids, rtgt, ln, stok, skind = H35.render_batch_slots(side, sslot, mb, a.dev, slot_on=True)
    t = rtgt.reshape(-1)
    m_slot = (skind.reshape(-1) >= 0)
    nonslot = (t != -100) & (~m_slot)
    rtgt2 = rtgt.clone()
    flat = rtgt2.reshape(-1)
    idxs = torch.nonzero(nonslot).reshape(-1)
    if idxs.numel():
        flat[idxs] = (flat[idxs] + 7) % max(2, int(cache2.NVALS))
    gcuts = {}
    with torch.no_grad():
        for tag, lam in (("lam1", 1.0), ("lam0", 0.0)):
            m35.lam_txt = lam
            vals = []
            for rt in (rtgt, rtgt2):
                bc = H35.L32.BatchCpu32().build(cache2, mb, 0, 0, 0)
                bc.rowIds = list(mb)
                lenB_full = [len(cache2.rows[i].desc) + ln[b] for b, i in enumerate(mb)]
                tot, ls, lb, lt, nv, nbf, ntxt, _L = m35.forward_joint(
                    bc, cache2, a.dev, allow2, rids, rt, lenB_full, grad_ckpt=False,
                    slot_tok=stok, slot_kind=skind)
                vals.append((float(tot.detach()), float(lt.detach()), int(ntxt), int(nbf)))
                del bc, tot, ls, lb, lt
            gcuts[tag] = {"total": vals[0][0], "total_perturbed": vals[1][0],
                          "absdiff": abs(vals[0][0] - vals[1][0]),
                          "lt_main": vals[0][1], "ntxt": vals[0][2], "nbf": vals[0][3]}
            torch.cuda.empty_cache()
        m35.lam_txt = 0.0
    # slot alive reading (device gating tensors present + slot target count > 0)
    gcuts["slot_side"] = {"n_slot_target": int(m_slot.sum()),
                          "n_nonslot_target": int(idxs.numel()),
                          "g_dev": float(m35.g_dev.detach()), "g_pos": float(m35.g_pos.detach()),
                          "slot_cond": bool(getattr(m35, "slot_cond", False)),
                          "slot_dev": bool(getattr(m35, "slot_dev", False))}
    rep["grad_cut"] = gcuts
    rep["text_term_dead_at_lam0"] = bool(gcuts["lam0"]["absdiff"] < 1e-3)
    rep["text_term_alive_at_lam1"] = bool(gcuts["lam1"]["absdiff"] > 1e-3)
    rep["slot_term_alive"] = bool(gcuts["slot_side"]["n_slot_target"] > 0
                                  and gcuts["slot_side"]["slot_cond"])
    rep["pass"] = bool(rep["state_dict"]["n_diff"] == 0
                       and rep["logits_first_step"]["torch_equal"]
                       and rep["text_term_dead_at_lam0"] and rep["text_term_alive_at_lam1"]
                       and rep["slot_term_alive"])
    led1 = C.parent_ledger()
    rep["parent_ledger_unchanged"] = bool(C.canon(led0) == C.canon(led1))
    rep["wall_s"] = round(time.time() - t0, 1)
    rep["out_md5"] = C.save_json(a.out, rep)
    C.log("PA-EQ pass=%s sd_diff=%d logits_eq=%s dead@lam0=%s alive@lam1=%s slot_alive=%s "
          "plan_same=%s" % (rep["pass"], rep["state_dict"]["n_diff"],
                            rep["logits_first_step"]["torch_equal"],
                            rep["text_term_dead_at_lam0"], rep["text_term_alive_at_lam1"],
                            rep["slot_term_alive"], rep["plan_md5_same_as_gen34"]), a.log)
    print(json.dumps({"out": a.out, "md5": rep["out_md5"], "pass": rep["pass"],
                      "plan_md5": pmd5, "plan_same": rep["plan_md5_same_as_gen34"],
                      "sd_diff": rep["state_dict"]["n_diff"],
                      "logits_maxdiff": rep["logits_first_step"]["maxdiff"],
                      "dead_at_lam0": rep["text_term_dead_at_lam0"],
                      "alive_at_lam1": rep["text_term_alive_at_lam1"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
