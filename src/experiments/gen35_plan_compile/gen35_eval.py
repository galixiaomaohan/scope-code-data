# -*- coding: utf-8 -*-
# gen35_eval.py -- node [4] **pure-planning arm exam section**: new LLM chain selection -> compiler rendering -> frozen judging
#   Two phases split apart (chain walk needs GPU; compile + judging are CPU/Lean):
#     --phase walk    : new ckpt free chain walk (218 exam set) -> out/GEN35_EVAL_PLAN_WALK.json
#     --phase compile : walk -> compiler -> judging (batch) -> out/GEN35_EVAL_PLAN.json
#   Model class comes from the gen35 fork (gen35_hybrid); the chain-walk/payload convention follows
#   gen34_eval line by line (parent imported, zero edits).
# Writes only under scratch/ccp/gen35/**. GPU0 redline; single-GPU exclusive gate; nice -n 19.
import argparse
import json
import os
import time

import gen35_common as C

import torch                                                              # noqa: E402

import gen34_eval as E34                                                  # noqa: E402 (import only)
import gen35_hybrid as H35                                                # noqa: E402
import gen35_armrun as AR                                                 # noqa: E402
import numpy as np                                                        # noqa: E402

E34.HY = H35            # model class / sidecar loading comes from the gen35 fork (fork patches paths only; semantics shared)

OUT = C.G35 + "/out"
LOGS = C.G35 + "/logs"


def build_args(a):
    """Namespace with the fields required by E34.load_model / load_items / _load_ctx_eval."""
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
    E34._set_vocab(a.vocab)
    if str(a.dev).startswith("cuda"):
        H35.gpu_gate(a.dev, logp)
    C.log("== gen35 eval WALK ckpt=%s seg=%s dev=%s ==" % (a.ckpt, a.seg, a.dev), logp)
    parents = {int(k): v["parent_idx"] for k, v in E34.VOC["init_map"].items()}
    allow = torch.from_numpy(np.fromfile(a.allow, dtype=np.uint8).reshape(
        E34.NHEAD, E34.NSLOT, E34.NVALS).copy())
    ctx = E34._load_ctx_eval(na)
    model, minfo = E34.load_model(na, parents, dev)
    C.log("model loaded: %s" % json.dumps({k: v for k, v in minfo.items()
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
        C.log("WALK %d/%d %s chain=%s gold=%s first=%s exact=%s"
              % (len(rows), len(items), s["name"], w["ops_walked"], w["gold_ops"],
                 w["first_hit"], w["exact_hit"]), logp)
    tbl = {"n": len(rows), "first_hit": sum(1 for r in rows if r["chain"]["first_hit"]),
           "exact_hit": sum(1 for r in rows if r["chain"]["exact_hit"]),
           "terminated_end": sum(1 for r in rows if r["chain"]["terminated_end"]),
           "over_run": sum(1 for r in rows if r["chain"]["over_run"])}
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
                    "note": "纯规划臂: 渲染 text 损失移除 (V35_DESIGN §3 P1); 渲染头输出不计分"}}
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
    ap.add_argument("--ckpt", default=C.G35 + "/ckpt/deep_plan_formal_6000.pt")
    ap.add_argument("--init", default="ckpt/sla_v31f.pt")
    ap.add_argument("--seg", default="exam_deep")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--cache", default=C.G33 + "/cache/deep_all.bin")
    ap.add_argument("--side", default=C.G33 + "/out/DEEP_RENDER_SIDECAR.jsonl")
    ap.add_argument("--exam-side", default=C.EXAM_SIDECAR)
    ap.add_argument("--allow", default=C.G33 + "/cache/ccp_allow_v4.bin")
    ap.add_argument("--vocab", default=C.G33 + "/cache/CCP_VOCAB_V4.json")
    ap.add_argument("--lat31", default="cppcuda/cache/lat_tab_v31.bin")
    ap.add_argument("--slot-side", default=C.G34 + "/out/DEEP_SLOT_SIDECAR.jsonl")
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--dev", default="cuda:2")
    ap.add_argument("--judge", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--walk-out", default=OUT + "/GEN35_EVAL_PLAN_WALK.json")
    ap.add_argument("--out", default=OUT + "/GEN35_EVAL_PLAN.json")
    ap.add_argument("--rows-out", default=OUT + "/GEN35_EVAL_PLAN_ROWS.jsonl")
    ap.add_argument("--log", default=LOGS + "/gen35_eval.log")
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    if a.phase == "walk":
        phase_walk(a, a.log)
    else:
        phase_compile(a, a.log)


if __name__ == "__main__":
    main()
