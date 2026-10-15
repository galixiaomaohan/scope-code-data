# -*- coding: utf-8 -*-
# gen34_equiv.py -- C-arm equivalence decision (V34_DESIGN §2.3 gate; task node [3])
#
# Criterion (**bitwise**; all must pass before the C arm may reuse gen33 artifacts without
# retraining):
#   E1 params: both models loading the same ckpt have bitwise-equal parameter tensors
#   E2 axes: on one forward pass, step_head / value_head / lm_head logits are bitwise equal
#   E3 loss: the (total, ls, lb, lt) returned by forward_joint are bitwise equal
#            (slot_kind=None path)
#   E4 generation: greedy 80-step token sequences are **token-by-token equal**
#            (render_greedy on each code path)
#   E5 path: with --slot-cond 0 on the gen34 side and the slot channel disabled the two are
#            **structurally the same path** (code-path assertion)
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
    ap.add_argument("--ckpt", default=G33 + "/ckpt/deep_base_formal_6000.pt")
    ap.add_argument("--cache", default=G33 + "/cache/deep_all.bin")
    ap.add_argument("--side", default=G33 + "/out/DEEP_RENDER_SIDECAR.jsonl")
    ap.add_argument("--exam-side", default=G33 + "/out/DEEP_EXAM_SIDECAR.jsonl")
    ap.add_argument("--vocab", default=G33 + "/cache/CCP_VOCAB_V4.json")
    ap.add_argument("--allow", default=G33 + "/cache/ccp_allow_v4.bin")
    ap.add_argument("--lat31", default="cppcuda/cache/lat_tab_v31.bin")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--dev", default="cuda:1")
    ap.add_argument("--out", default=OUT + "/GEN34_EQUIV.json")
    ap.add_argument("--log", default=G34 + "/logs/gen34.log")
    a = ap.parse_args()
    C.lean_gate(5, a.log)
    sys.path.insert(0, C.G13)
    import numpy as np
    import torch
    import gen33_eval as E33
    import gen33_hybrid as H33
    import gen34_eval as E34
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
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    C.log("EQUIV load gen33 ckpt %s step=%s md5=%s"
          % (a.ckpt, sd.get("step"), C.md5f(a.ckpt)), a.log)

    def build(cls):
        torch.manual_seed(20260923)
        m = cls(base=None, lat_tab_path=a.lat31, head=head, nvals=nvals, parents=parents)
        mi, un = m.load_state_dict(sd["model"], strict=False)
        assert not un and not mi, (list(mi)[:6], list(un)[:6])
        return m.to(dev).eval()

    m33 = build(H33.HybridModel)
    m34 = build(H34.HybridModel)
    # E1 bitwise params
    s33, s34 = m33.state_dict(), m34.state_dict()
    assert set(s33) == set(s34), (len(s33), len(s34))
    diff = [k for k in s33 if not torch.equal(s33[k], s34[k])]
    e1 = not diff
    lo, hi = CK.segidx["segments"]["a_held"]
    idx = list(range(lo, hi))[:a.n]
    rep = {"schema": "gen34-equiv/1", "ckpt": a.ckpt, "ckpt_md5": C.md5f(a.ckpt),
           "step": sd.get("step"), "n_rows": len(idx), "dev": a.dev,
           "note": ("C-arm equivalence: gen33 code path vs gen34 code path on the same "
                    "gen33 ckpt; all-pass => the C arm may reuse gen33 artifacts "
                    "(saves the 100-minute retrain)")}
    rep["E1_params_bitwise"] = bool(e1)
    rep["E1_n_diff_params"] = len(diff)
    with torch.no_grad():
        bc = H34.L32.BatchCpu32().build(CK, idx, 0, 0, 0)
        bc.rowIds = list(idx)
        r33, t33, l33 = H33.render_batch(side, idx, dev)
        r34, t34, l34, _s1, _s2 = H34.render_batch_slots(side, None, idx, dev, slot_on=0)
        assert torch.equal(r33, r34) and torch.equal(t33, t34) and l33 == l34
        lenB_full = [len(CK.rows[i].desc) + l33[b] for b, i in enumerate(idx)]
        gov = {}
        for tag, m, tb in (("g33", m33, t33), ("g34", m34, t34)):
            seq_c = m.assemble(bc.B, bc.L, bc, CK, dev)
            seq_r = m.render_embed(r33)
            seq = torch.cat([seq_c, seq_r], 1)
            L = bc.L + int(r33.shape[1])
            cm, cosc, sinc = m.consts(L, dev)
            mk = m.mask_of(cm, bc.B, L, lenB_full, dev)
            H = m.forward_body(seq, mk, cosc, sinc)
            lg, zb, _t = m.heads(H[:, :bc.L, :].contiguous(), bc, CK, dev, allow)
            lt = m.lm.lm_head(H[:, bc.L:, :])
            gov[tag] = {"lg": lg.detach().cpu(), "zb": zb.detach().cpu(),
                        "lt": lt.detach().cpu()}
            del H, seq, seq_c, seq_r, lg, zb, lt
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        eq = {k: bool(torch.equal(gov["g33"][k], gov["g34"][k])) for k in ("lg", "zb", "lt")}
        dmax = {k: float((gov["g33"][k] - gov["g34"][k]).abs().max()) for k in eq}
        # E3 bitwise loss
        o33 = m33.forward_joint(bc, CK, dev, allow, r33, t33, lenB_full, grad_ckpt=False)
        o34 = m34.forward_joint(bc, CK, dev, allow, r33, t33, lenB_full, grad_ckpt=False,
                                slot_tok=None, slot_kind=None)
        e3 = bool(all(float(x) == float(y) for x, y in zip(o33[:4], o34[:4])))
        # E3b verbatim same path as the parent: the slot branch must not be entered when
        # the slot channel is off
        e3b = bool(not getattr(m34, "slot_cond", False))
    rep["E2_axes_bitwise"] = eq
    rep["E2_abs_max_diff"] = dmax
    rep["E3_loss_bitwise"] = e3
    rep["E3_values"] = {"g33": [float(x) for x in o33[:4]], "g34": [float(x) for x in o34[:4]]}
    rep["E3b_slot_channel_off"] = e3b
    # E4 greedy token sequences equal token by token (includes device-side live _prefix)
    ctx = E34._load_ctx_eval(a)
    exside = {int(r["ri"]): r for r in C.jlload(a.exam_side)}
    ekeys = [i for i in sorted(exside) if exside[i]["seg"] == "exam_deep"][:2]
    seqs = []
    for i in ekeys:
        s = exside[i]
        payload = E34._payload_of(s)
        ops = list(s.get("ops") or [])
        o33 = E33.render_greedy(m33, ctx, s, ops, payload, dev, max_new=a.steps)
        o34 = E34.render_greedy(m34, ctx, s, ops, payload, dev, max_new=a.steps)
        seqs.append({"ri": int(i), "name": s["name"], "n33": o33[1], "n34": o34[1],
                     "text_equal": bool(o33[0] == o34[0]),
                     "token_equal": bool(o33[1] == o34[1] and o33[0] == o34[0])})
    rep["E4_greedy_render"] = seqs
    rep["E4_all_equal"] = bool(all(x["token_equal"] for x in seqs))
    rep["pass"] = bool(e1 and all(eq.values()) and e3 and e3b and rep["E4_all_equal"])
    rep["verdict"] = ("C arm may reuse gen33 artifacts (no retrain)" if rep["pass"] else
                      "not equivalent => the C arm must be retrained with the gen34 code "
                      "and --slot-cond 0")
    rep["md5"] = C.save_json(a.out, rep)
    C.log("EQUIV pass=%s E1=%s E2=%s E3=%s E3b=%s E4=%s dmax=%s | %s"
          % (rep["pass"], e1, eq, e3, e3b, rep["E4_all_equal"], dmax, rep["verdict"]), a.log)
    print("GEN34_EQUIV_RC=0 pass=%s" % rep["pass"])
    return rep


if __name__ == "__main__":
    main()
