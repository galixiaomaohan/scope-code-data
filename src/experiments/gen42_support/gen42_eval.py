# ---- gen42 fork register (miniF2F domain expansion path B) ----
# This file = a patched fork of gen41_eval.py (md5 cc51fa76d51535dff179fc87459c7fc3); **parent untouched**.
# Differences: [1] import gen42_common; [2] OUT/LEAN/LOGS -> gen42 write surface;
#       [3] vocab head 198 -> 207 (V6, new operators 198..206); [4] dispatch the g42_* families.
# Judging discipline: drct_common md5 7c31b58da9d7228cee31b9be581fe5dd imported only, zero edits (verified before and after judging).
# ---- gen41 fork register (miniF2F domain expansion · exam segment) ----
# This file = a patched fork of gen36_eval.py (md5 7dc0281ca9bc5d80092bc0c5fba5f784); parent untouched.
# Differences: [1] import gen41_common/gen41_armrun/gen41_hybrid; [2] OUT/LOGS -> gen41;
#   [3] --state-shift defaults to 1 (deep domain a2 shift UNION new-family native post-step frames => the whole line uses the post-step convention);
#   [4] the gen41-family payload = gen41_payload.payload_of_g41 (device live computation, same convention as the training cache, bit for bit);
#   [5] NHEAD takes the vocab's head_v5 (the V5 artifact's "head" = 194 is the legacy V4 anchor); [6] defaults -> gen41.
# ---- gen36 fork register (terminal-step routing push · exam segment) ----
# This file = a **minimal-surface fork copy** of gen35_eval.py (md5 b9beac840e79e2348e57a9257779918d); parent untouched.
# Differences (V36_DESIGN §3 D3): [1] path defaults OUT/LOGS/ckpt -> gen36;
#   [2] `--state-shift 1` = payload res **shift convention** (res(k) := device live R_in(k+1), tail frame kept),
#       bitwise the same convention as the G-SW arm training cache (gen36/cache/deep_all_a2.bin); default 0 = the parent convention.
#   [3] judging/compiling go through gen36_armrun (Lean workdir gen36/lean).
# ---- gen35 fork register ----
# -*- coding: utf-8 -*-
# gen35_eval.py -- node [4] **pure-planning arm exam segment**: the new LLM chain walk -> compiler rendering -> frozen judging
#   two-phase separation (the chain walk needs a GPU; compile+judge is CPU/Lean):
#     --phase walk    : new ckpt free chain walk (218-exam set) -> out/GEN35_EVAL_PLAN_WALK.json
#     --phase compile : chain walk -> compiler -> judging (batch) -> out/GEN35_EVAL_PLAN.json
#   The model class comes from the gen35 fork (gen35_hybrid); the walk/payload conventions follow gen34_eval line by line (parent imported only, zero edits).
# Write-only scratch/ccp/gen35/**. GPU0 red line; single-card exclusive gate; nice -n 19.
import argparse
import json
import os
import time

import gen42_common as C

import torch                                                              # noqa: E402

import gen34_eval as E34                                                  # noqa: E402 (import only)
import gen41_hybrid as H35                                                # noqa: E402
import gen42_armrun as AR                                                 # noqa: E402
import numpy as np                                                        # noqa: E402

E34.HY = H35            # model class / sidecar loading taken from the gen35 fork (the fork only patches paths, semantics same-origin)

# ---- gen41 [F1-F4]: runtime override of gen34_eval's **module-level** reader artifacts (existing files untouched; replaced only inside this fork) ----
#   F1 ctx: V5 vocab (head_v5 present) => gen41_cachebuild.load_ctx5 (opmap/names_head cover 161..197;
#      otherwise the chain walk hits a KeyError when building frames for the new operators) · F2 action space: add ops_g41 (otherwise the model cannot emit 194..197)
#   F3 the gold_in_space criterion then includes the new operators · F4 load_items: other seg names filter by the same name (miniF2F set = exam_g41)
_E34_ORIG_LOAD_ITEMS = E34.load_items


def _legal_ops_v5():
    VOC = E34.VOC
    names = {int(v): k for k, v in VOC["ops_ccp"].items()}
    for key in ("ops_bh", "ops_g24", "ops_gen32", "ops_g41", "ops_g42"):
        for k, v in (VOC.get(key) or {}).items():
            names[int(v)] = k
    return sorted(names), names


def _load_ctx_eval_v5(a):
    vp = getattr(a, "vocab", "") or ""
    try:
        v = json.load(open(vp, encoding="utf-8")) if vp else {}
    except Exception:                                                # noqa: BLE001
        v = {}
    if "head_v6" in v:                       # gen42: V6 ctx (opmap/names_head cover 198..206)
        import gen42_cachebuild as CB42
        return CB42.load_ctx6()
    if "head_v5" in v:
        import gen41_cachebuild as CB41
        return CB41.load_ctx5()
    return E34._load_ctx_eval(a)


def _load_items_v5(a):
    if a.seg in ("a_held", "a_train", "exam_main", "exam_deep", "exam_zhenti"):
        return _E34_ORIG_LOAD_ITEMS(a)
    off = max(0, int(getattr(a, "chunk_offset", 0)))
    ex = H35.load_sidecar(a.exam_side)
    ii = [i for i in sorted(ex) if ex[i].get("seg") == a.seg][off:off + a.n]
    return [(i, ex[i]) for i in ii]


E34._load_ctx_eval = _load_ctx_eval_v5
E34._legal_ops = _legal_ops_v5
E34.load_items = _load_items_v5
try:
    _v5d = json.load(open(C.VOCAB_V5, encoding="utf-8"))
    _g41_names = [n for n, _i in sorted((_v5d.get("ops_g41") or {}).items(),
                                        key=lambda kv: int(kv[1]))]
    _g42_names = [n for n, _i in sorted((_v5d.get("ops_g42") or {}).items(),
                                        key=lambda kv: int(kv[1]))]
    E34.CCP_OPS = list(E34.CCP_OPS) + _g41_names + _g42_names
except Exception:                                                    # noqa: BLE001
    pass

def nan_guard(model, stats=None):
    """gen42 [D16] chain-walk NaN guard: when op logits contain NaN/Inf, fold them to -1e4 (NaN = no preference),
    forbidding argmax from landing on an illegal index (raw state => names.get(pred)=None => ccp_cachefull KeyError =>
    the whole exam crashes). stats = {"n_calls","n_nan"} counters (None => use the module-level NAN_GUARD_STATS)."""
    import torch as _t
    st = stats if stats is not None else NAN_GUARD_STATS
    orig = model.heads

    def heads(*a, **k):
        out = orig(*a, **k)
        lg = out[0]
        if _t.isnan(lg).any() or _t.isinf(lg).any():
            st["n_nan"] += 1
            return (_t.nan_to_num(lg, nan=-1e4, posinf=1e4, neginf=-1e4),) + tuple(out[1:])
        st["n_calls"] += 1
        return out

    model.heads = heads
    return model


NAN_GUARD_STATS = {"n_calls": 0, "n_nan": 0}


_ORIG_E34_PAYLOAD_OF = E34._payload_of   # 2026-09-25 fix: snapshot the original before patching, preventing _payload_of_shift from recurring into itself (RecursionError at depth 992)

OUT = C.G41 + "/out"
LOGS = C.G41 + "/logs"


_ORIG_SETVOCAB = E34._set_vocab


def _set_vocab_v5(path):
    """gen41 [E-vocab]: the V5 artifact's head=194 is the legacy V4 anchor; the real head = head_v5 = 198 (registered D14)."""
    _ORIG_SETVOCAB(path)
    v = json.load(open(path, encoding="utf-8"))
    if "head_v6" in v:                       # gen42: V6 real head = 207
        E34.NHEAD = int(v["head_v6"])
    elif "head_v5" in v:
        E34.NHEAD = int(v["head_v5"])
    return E34.NHEAD


def _payload_of_g41(side_rec):
    """gen41 [E-payload]: the g41_* families use gen41_payload device live computation (same convention as the training cache, bit for bit)."""
    fam = (side_rec.get("family") or "")
    if fam.startswith("g41_"):
        import gen41_payload as GP
        return GP.payload_of_g41(side_rec)
    return _payload_of_shift(side_rec)


def _payload_of_shift(side_rec):
    """gen36 [D3] payload convention = the gen36/cache/deep_all_a2 training-cache convention (single state source gen36_state).

    Only the cascade family is shifted (bitwise consistent with the cache scope=cascade; other families use the parent convention):
      frame k (k <= n-2) carries res := evals(R_out(k)) = post-step device residual;
      the tail frame (k = n-1) keeps the parent value (dev_resid(row, n-1); only read by the END decision).
    """
    base = _ORIG_E34_PAYLOAD_OF(side_rec)
    if (side_rec.get("family") or "") != "gen32_ineq_cascade":
        return base
    import gen36_state as S36
    sfa = list(side_rec.get("sfa") or [])
    res = list(base["res"])
    errs = list(base.get("dev_errs") or [])
    n = len(sfa)
    for k in range(n):
        if k >= n - 1:
            continue
        try:
            r8, _ro, _nv = S36.post_action_res(sfa[k])
            res[k] = r8
        except Exception as e:                                    # noqa: BLE001
            errs.append({"k": k, "err": "SHIFT:" + repr(e)})
    return {"res": res, "sfa": sfa, "dev_errs": errs,
            "res_src": "device_live(post_action/cascade)"}


def build_args(a):
    """Namespace of the fields required by E34.load_model / load_items / _load_ctx_eval."""
    ns = argparse.Namespace()
    ns.vocab = a.vocab
    ns.cache = a.cache
    ns.side = a.side
    ns.exam_side = a.exam_side
    ns.allow = a.allow
    ns.lat31 = a.lat31
    ns.init = a.init
    ns.ckpt = a.ckpt
    ns.seg = a.seg
    ns.n = a.n
    ns.chunk_offset = 0
    ns.args_mode = "device"
    ns.pa_enable = 0
    ns.pa_seed = a.seed
    ns.pa_state = 1
    ns.pa_heads = 2
    ns.pa_dhead = 64
    ns.plan_summary = -1
    ns.pa_capture = 0
    ns.slot_cond = 1
    ns.slot_dev = 1
    ns.slot_side = a.slot_side
    ns.slot_beta = 8.0
    ns.lam_slot = 3.0
    ns.max_new = 0
    ns.judge = 0
    ns.dev = a.dev
    ns.arm = "hybrid"
    ns.log = a.log
    return ns


def phase_walk(a, logp):
    dev = torch.device(a.dev)
    na = build_args(a)
    _set_vocab_v5(a.vocab)
    if int(getattr(a, "state_shift", 0)):      # gen36 [D3] payload shift convention
        E34._payload_of = _payload_of_g41
        C.log("gen36 payload shift convention: res(k) := dev_resid(k+1), tail frame kept", logp)
    if str(a.dev).startswith("cuda"):
        H35.gpu_gate(a.dev, logp)
    C.log("== gen35 eval WALK ckpt=%s seg=%s dev=%s ==" % (a.ckpt, a.seg, a.dev), logp)
    parents = {int(k): v["parent_idx"] for k, v in E34.VOC["init_map"].items()}
    allow = torch.from_numpy(np.fromfile(a.allow, dtype=np.uint8).reshape(
        E34.NHEAD, E34.NSLOT, E34.NVALS).copy())
    ctx = E34._load_ctx_eval(na)
    model, minfo = E34.load_model(na, parents, dev)
    nan_guard(model)               # gen42 [D16] NaN guard (protection against the whole-exam crash root cause)
    C.log("model load: %s" % json.dumps({k: v for k, v in minfo.items()
                                         if k in ("md5", "step", "missing", "unexpected",
                                                  "slot", "note")}, ensure_ascii=False), logp)
    items = E34.load_items(na)
    C.log("exam rows=%d" % len(items), logp)
    rows = []
    t0 = time.time()
    for ri, s in items:
        w = E34.walk_chain(model, ctx, s, allow, dev, args_mode="device", logp=None)
        rows.append({"ri": int(ri), "name": s["name"], "cls": s.get("cls"),
                     "family": s.get("family"), "seg": s.get("seg"), "arm": "pure_plan",
                     "head": s["head"], "concl": s["concl"],
                     "chain": {"walked": w["ops_walked"], "gold": w["gold_ops"],
                               "first_hit": w["first_hit"], "exact_hit": w["exact_hit"],
                               "prefix_hit": w["prefix_hit"], "over_run": w["overrun"],
                               "terminated_end": w["terminated_end"],
                               "scorable": w["scorable"], "budget": w["budget"],
                               "args_src_hist": w["args_src_hist"],
                               "trace": w["trace"]}})
        C.log("WALK %d/%d %s chain=%s gold=%s first_hit=%s exact=%s"
              % (len(rows), len(items), s["name"], w["ops_walked"], w["gold_ops"],
                 w["first_hit"], w["exact_hit"]), logp)
    tbl = {"n": len(rows), "first_hit": sum(1 for r in rows if r["chain"]["first_hit"]),
           "exact_hit": sum(1 for r in rows if r["chain"]["exact_hit"]),
           "terminated_end": sum(1 for r in rows if r["chain"]["terminated_end"]),
           "over_run": sum(1 for r in rows if r["chain"]["over_run"])}
    tbl["nan_guard"] = dict(NAN_GUARD_STATS)
    res = {"schema": "gen35-eval-walk/1",
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ckpt": a.ckpt,
           "ckpt_md5": (C.md5f(a.ckpt) if a.ckpt and os.path.exists(a.ckpt) else None),
           "seg": a.seg, "n": len(rows), "table": tbl, "model": {
               k: v for k, v in minfo.items() if k in ("md5", "step", "missing",
                                                       "unexpected", "slot", "note")},
           "wall_s": round(time.time() - t0, 2), "rows": rows}
    C.save_json(a.walk_out, res)
    C.log("WALK DONE %s -> %s" % (json.dumps(tbl, ensure_ascii=False), a.walk_out), logp)
    return res


def phase_compile(a, logp):
    d = json.load(open(a.walk_out, encoding="utf-8"))
    exam = {int(r["ri"]): r for r in AR.load_exam()}
    chains = {}
    for r in d["rows"]:
        ch = r["chain"]
        chains[int(r["ri"])] = {"walked": list(ch["walked"]), "gold": list(ch["gold"]),
                                "exact_hit": ch["exact_hit"], "first_hit": ch["first_hit"],
                                "overrun": ch["over_run"],
                                "terminated_end": ch["terminated_end"],
                                "arch_src": "gen35_pure_plan"}
    recs = [exam[k] for k in sorted(exam)]
    rows, stat = AR.compile_arm(recs, chains, tag="pure_plan")
    rep = {"schema": "gen35-eval/1", "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "arm": "pure_plan", "ckpt": a.ckpt,
           "ckpt_md5": (C.md5f(a.ckpt) if a.ckpt and os.path.exists(a.ckpt) else None),
           "walk_table": d["table"], "compile": stat, "judge_md5": C.judge_gate(),
           "flag": {"lam_txt": 0.0, "lam_bind": 1.0, "slot_cond": 1, "slot_dev": 1,
                    "note": "pure-planning arm: the render text loss is removed (V35_DESIGN §3 P1); rendering-head output is not scored"}}
    if a.judge:
        C.lean_gate(5, a.log)
        rep["judge"] = AR.judge_rows(rows, "pure_plan", a.chunk, a.log)
    C.jlsave(rows, a.rows_out)
    rep["rows_md5"] = C.md5f(a.rows_out)
    rep["out_md5"] = C.save_json(a.out, rep)
    print(json.dumps({"out": a.out, "md5": rep["out_md5"],
                      "certified": (rep.get("judge") or {}).get("certified"),
                      "n": stat["n"], "exact": stat["n_exact"],
                      "first": stat["n_first"]}, ensure_ascii=False))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="compile", choices=("walk", "compile"))
    ap.add_argument("--ckpt", default=C.G36 + "/ckpt/deep_plan_wt5_6000.pt")
    ap.add_argument("--init", default="ckpt/sla_v31f.pt")
    ap.add_argument("--seg", default="exam_deep")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--cache", default=C.CACHE_G41)
    ap.add_argument("--side", default=C.RENDER_SIDE_G41)
    ap.add_argument("--exam-side", default=C.EXAM_SIDECAR)
    ap.add_argument("--allow", default=C.ALLOW_V5)
    ap.add_argument("--vocab", default=C.VOCAB_V5)
    ap.add_argument("--lat31", default="cppcuda/cache/lat_tab_v31.bin")
    ap.add_argument("--slot-side", default=C.SLOT_SIDE_G41)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--state-shift", type=int, default=1,
                    help="gen36 [D3]: 1 = payload res shift convention (same convention as the G-SW training cache)")
    ap.add_argument("--dev", default="cuda:2")
    ap.add_argument("--judge", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--walk-out", default=OUT + "/GEN41_WALK.json")
    ap.add_argument("--out", default=OUT + "/GEN41_PLAN.json")
    ap.add_argument("--rows-out", default=OUT + "/GEN41_PLAN_ROWS.jsonl")
    ap.add_argument("--log", default=LOGS + "/gen41_eval.log")
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    if a.phase == "walk":
        phase_walk(a, a.log)
    else:
        phase_compile(a, a.log)


if __name__ == "__main__":
    main()
