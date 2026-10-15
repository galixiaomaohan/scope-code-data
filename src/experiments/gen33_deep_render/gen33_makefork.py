# -*- coding: utf-8 -*-
# gen33_makefork.py -- gen33 deep-generation V3.5 fork bench: line-by-line fork of the three gen30 files into the three gen33 files
#
# Provenance record: the gen30 parents are not part of this package, so the
# find-side patch patterns below keep placeholders where they originally held
# the host's superoperator path; run against the original tree to re-derive.
# Discipline: **exact replacement only, parent files never modified** (fork-copy regime; parent md5 ledger in V33_DESIGN §0).
#   Every patch must hit exactly once, otherwise the whole artifact is refused (no silent output).
# Usage: run `python gen33_makefork.py` inside the stage dir; input = fork/ (parent copies), output = same dir.
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FORK = os.path.join(HERE, "fork")

HDR_HYB = """
# ---- gen33 fork 登记 (深代 V3.5 处理臂) ----
# 本件 = %(src)s (md5 %(md5)s) 的**逐行 fork 副本**; 既有件零改动 (fork 复制制)。
# 继承父件三条修复 (**原样继承, 零改**): ① LN_q 偏离; ② CRIT-30A query 门控修复;
#   ③ CRIT-30B 词表口径修复。
# 本批唯一差异 (见 scratch/ccp/gen33/V33_DESIGN.md 预注册) 补丁清单:
#   [1] 路径: 新增 G33; OUT/日志/产物指向 gen33 (写面)
#   [2] 词表口径: 默认 --vocab/--allow 指向 ccp-vocab/4 (head 194 / 值空间 352) 与 ccp_allow_v4.bin
#   [3] 缓存/侧车默认指向 gen33 深语料件 (deep_all.bin / DEEP_RENDER_SIDECAR.jsonl)
#   [4] CLASSES 3 类 -> 2 类 (ind/ineq; 深语料无 nt 类, D-5)
#   [5] import gen33_planattn / gen33_eval (fork) 而非 gen30_*
#   [6] 默认 --dev cuda:1 (GPU0 用户红线: 结构性禁用)
# 写面: 仅 scratch/ccp/gen33/** (gen30/gen26/gen13 产物只读作输入)。红线同父件。
"""
HDR_EVAL = """
# ---- gen33 fork 登记 (深代 V3.5 处理臂) ----
# 本件 = %(src)s (md5 %(md5)s) 的**逐行 fork 副本**; 既有件零改动 (fork 复制制)。
# 继承父件三条修复 (**原样继承, 零改**): ① LN_q 偏离; ② CRIT-30A query 门控修复;
#   ③ CRIT-30B 词表口径修复。
# 本批唯一差异 (见 scratch/ccp/gen33/V33_DESIGN.md 预注册) 补丁清单:
#   [1] 路径: 新增 G33; OUT/日志/判卷目录指向 gen33
#   [2] 词表口径: v4 装载 (load_ctx4) + 合法集含 gen32 九算子; allow/vocab 默认指向 v4
#   [3] **装置现算口径** (变更96): args_mode 增 "device" —— 每步 8 维状态由 gen33_device
#       装置现算 (禁止读金标链状态), 走链/考试的载荷同源
#   [4] WALK_BUDGET 7 -> 12 (深链最长 9 步 + 3; 预注册常量, 见设计书补充件)
#   [5] 渲染一致性关键词表增 9 新算子 (诊断列)
#   [6] import gen33_hybrid / gen33_planattn (fork) 而非 gen30_*
#   [7] 默认 --dev cuda:1 (GPU0 用户红线)
# 写面: 仅 scratch/ccp/gen33/** (gen30/gen26/gen13 产物只读作输入)。红线同父件。
"""
HDR_PA = """
# ---- gen33 fork 登记 (深代 V3.5 处理臂) ----
# 本件 = %(src)s (md5 %(md5)s) 的**逐行 fork 副本**; 既有件零改动 (fork 复制制)。
# 继承父件三条修复 (**原样继承, 零改**): ① LN_q 偏离; ② CRIT-30A query 门控修复;
#   ③ CRIT-30B 词表口径修复。
# 本批差异 = **零** (纯副本 + 登记头): 本件不含任何路径常量/词表常量, 与父件逐字节同逻辑。
# 写面: 仅 scratch/ccp/gen33/**。
"""

PATCH_HYB = [
    # [1] path block
    ('''G26 = os.path.join(CCP, "gen26")            # gen30: 对照臂产物 (只读输入源)
G13 = os.path.join(CCP, "gen13")
G30 = os.path.join(CCP, "gen30")            # gen30 fork 写面
OUT = os.path.join(G30, "out")''',
     '''G26 = os.path.join(CCP, "gen26")            # 只读输入源 (v3 词表/对照臂产物)
G13 = os.path.join(CCP, "gen13")
G30 = os.path.join(CCP, "gen30")            # 父件 (只读; fork 台账)
G33 = os.path.join(CCP, "gen33")            # gen33 fork 写面
OUT = os.path.join(G33, "out")'''),
    ('''for _p in (RES, G13, os.path.join(RES, "cppcuda", "tools"), os.path.join(RES, "skel32"),
           G26, G30, "<superoperator path on the original host>"):''',
     '''for _p in (RES, G13, os.path.join(RES, "cppcuda", "tools"), os.path.join(RES, "skel32"),
           G26, G30, G33, os.environ.get("SCOPE_SUPEROPERATOR", "../device")):'''),
    # [4] two classes
    ('CLASSES = ("ind", "ineq", "nt")',
     'CLASSES = ("ind", "ineq")         # D-5: 深语料只有 ind/ineq 两类 (无 nt)'),
    # [5] fork import
    ('import gen30_planattn as PA30                                        # noqa: E402\n\nAttnModel = PA30.make_attn_model(HybridModel)',
     'import gen33_planattn as PA33                                        # noqa: E402\n\nAttnModel = PA33.make_attn_model(HybridModel)'),
    ('            import gen30_eval as EV                                   # gen30 [5] fork',
     '            import gen33_eval as EV                                   # gen33 [5] fork'),
    # [2][3][6] CLI defaults
    ('    ap.add_argument("--cache", default=os.path.join(G26, "cache", "hyb_a_all.bin"))',
     '    ap.add_argument("--cache", default=os.path.join(G33, "cache", "deep_all.bin"))'),
    ('    ap.add_argument("--side", default=os.path.join(OUT, "A_RENDER_SIDECAR.jsonl"))',
     '    ap.add_argument("--side", default=os.path.join(OUT, "DEEP_RENDER_SIDECAR.jsonl"))'),
    ('    ap.add_argument("--allow", default=os.path.join(G13, "ccp_allow_ext.bin"))',
     '    ap.add_argument("--allow", default=os.path.join(G33, "cache", "ccp_allow_v4.bin"))'),
    ('    ap.add_argument("--vocab", default=os.path.join(G13, "CCP_VOCAB.json"))',
     '    ap.add_argument("--vocab", default=os.path.join(G33, "cache", "CCP_VOCAB_V4.json"))'),
    ('    ap.add_argument("--dev", default="cuda:0")',
     '    ap.add_argument("--dev", default="cuda:1")'),
    ('    ap.add_argument("--out", default=os.path.join(G26, "ckpt", "hybrid_smoke100.pt"))',
     '    ap.add_argument("--out", default=os.path.join(G33, "ckpt", "deep_smoke100.pt"))'),
    ('    ap.add_argument("--curve", default=os.path.join(OUT, "HYBRID_CURVE.jsonl"))',
     '    ap.add_argument("--curve", default=os.path.join(OUT, "DEEP_CURVE.jsonl"))'),
    ('    ap.add_argument("--plan-out", default=os.path.join(OUT, "batch_idx_hybrid.txt"))',
     '    ap.add_argument("--plan-out", default=os.path.join(OUT, "batch_idx_deep.txt"))'),
    ('    ap.add_argument("--samples", default=os.path.join(OUT, "HYBRID_SAMPLES.json"))',
     '    ap.add_argument("--samples", default=os.path.join(OUT, "DEEP_SAMPLES.json"))'),
    ('    ap.add_argument("--summary", default=os.path.join(OUT, "HYBRID_SMOKE.json"))',
     '    ap.add_argument("--summary", default=os.path.join(OUT, "DEEP_SUMMARY.json"))'),
    ('    ap.add_argument("--log", default=os.path.join(G26, "logs", "gen26_hybrid.log"))',
     '    ap.add_argument("--log", default=os.path.join(G33, "logs", "gen33_hybrid.log"))'),
    # curve sample every 300 steps (V33_DESIGN §6)
    ('''        if a.snap_every and step % a.snap_every == 0:''',
     '''        if step % 300 == 0 or step == steps:      # gen33 [7]: 每 300 步曲线抽点落盘
            with open(os.path.join(OUT, "DEEP_CURVE_300.jsonl"), "a",
                      encoding="utf-8") as _f3:
                _f3.write(json.dumps({"step": step, "loss": rec["loss"],
                                      "chain": rec["chain"], "slot": rec["slot"],
                                      "text": rec["text"], "gn": rec["gn"],
                                      "lr": rec["lr"], "seen": rec["seen"],
                                      "secs": rec["secs"], "nv": rec["nv"],
                                      "nbf": rec["nbf"], "ntxt": rec["ntxt"],
                                      "plan_md5": pmd5, "steps_total": steps,
                                      "tag": a.tag}, ensure_ascii=False) + "\\n")
        if a.snap_every and step % a.snap_every == 0:'''),
]

PATCH_EVAL = [
    # [1] path block
    ('''G26 = os.path.join(CCP, "gen26")            # gen30: 对照臂产物 (只读输入源)
G13 = os.path.join(CCP, "gen13")
G30 = os.path.join(CCP, "gen30")            # gen30 fork 写面
OUT = os.path.join(G30, "out")''',
     '''G26 = os.path.join(CCP, "gen26")            # 只读输入源
G13 = os.path.join(CCP, "gen13")
G30 = os.path.join(CCP, "gen30")            # 父件 (只读; fork 台账)
G33 = os.path.join(CCP, "gen33")            # gen33 fork 写面
OUT = os.path.join(G33, "out")'''),
    ('''for _p in (RES, CCP, G13, os.path.join(RES, "cppcuda", "tools"), os.path.join(RES, "skel32"),
           G26, os.path.join(CCP, "gen23"), os.path.join(RES, "scratch", "drct"),
           os.path.join(CCP, "bh", "gen"), os.path.join(CCP, "bh"),
           os.path.join(CCP, "gen24"), "<superoperator path on the original host>"):''',
     '''for _p in (RES, CCP, G13, os.path.join(RES, "cppcuda", "tools"), os.path.join(RES, "skel32"),
           G26, G30, G33, os.path.join(CCP, "gen23"), os.path.join(RES, "scratch", "drct"),
           os.path.join(CCP, "bh", "gen"), os.path.join(CCP, "bh"),
           os.path.join(CCP, "gen24"), os.path.join(CCP, "gen32"),
           os.environ.get("SCOPE_SUPEROPERATOR", "../device")):'''),
    # [6] fork import
    ('''import gen30_hybrid as HY                                                   # noqa: E402
import gen30_planattn as PA30                                               # noqa: E402
AttnModel = PA30.make_attn_model(HY.HybridModel)''',
     '''import gen33_hybrid as HY                                                   # noqa: E402
import gen33_planattn as PA33                                               # noqa: E402
import gen33_device as DEV                                                  # noqa: E402
AttnModel = PA33.make_attn_model(HY.HybridModel)'''),
    # [2] default vocab/allow (v4)
    ('''VOC = json.load(open(os.environ.get("GEN26_VOCAB", os.path.join(G13, "CCP_VOCAB.json")),
                     encoding="utf-8"))''',
     '''VOC = json.load(open(os.environ.get("GEN33_VOCAB",
                                    os.path.join(G33, "cache", "CCP_VOCAB_V4.json")),
                     encoding="utf-8"))'''),
    # [2] v4 loading branch
    ('''    vp = getattr(a, "vocab", "") or ""
    try:
        need3 = (bool(vp)
                 and json.load(open(vp, encoding="utf-8")).get("schema") == "ccp-vocab/3")
    except Exception:                                                # noqa: BLE001
        need3 = False
    if need3:''',
     '''    vp = getattr(a, "vocab", "") or ""
    try:
        _sch = json.load(open(vp, encoding="utf-8")).get("schema") if vp else ""
    except Exception:                                                # noqa: BLE001
        _sch = ""
    if _sch == "ccp-vocab/4":
        import importlib.machinery as _im4
        import importlib.util as _iu4
        _ld4 = _im4.SourceFileLoader("g33_vocab4", os.path.join(G33, "gen33s_vocab4.py"))
        _v4 = _iu4.module_from_spec(_iu4.spec_from_loader("g33_vocab4", _ld4))
        _ld4.exec_module(_v4)
        return _v4.load_ctx4(vp)
    need3 = (_sch == "ccp-vocab/3")
    if need3:'''),
    # [2] legal set includes the nine gen32 operators
    ('''    names = {int(v): k for k, v in VOC["ops_ccp"].items()}
    for key in ("ops_bh", "ops_g24"):          # A3 (v3): 17 算子入合法集
        for k, v in (VOC.get(key) or {}).items():
            names[int(v)] = k
    return sorted(names), names''',
     '''    names = {int(v): k for k, v in VOC["ops_ccp"].items()}
    for key in ("ops_bh", "ops_g24", "ops_gen32"):   # A3 (v3) 17 + gen33 (v4) 9
        for k, v in (VOC.get(key) or {}).items():
            names[int(v)] = k
    return sorted(names), names'''),
    # [4] chain-walk budget
    ('''WALK_BUDGET = 7''',
     '''# gen33 [4]: 深链最长金标链 = 9 步 ⇒ 预注册常量 = 9 + 3 = 12 (见 V33_DESIGN_ADDENDUM §A1)
WALK_BUDGET = 12'''),
    # [3] device-recompute payload
    ('''def _payload_of(side_rec):
    return {"res": list(side_rec.get("res") or []), "sfa": list(side_rec.get("sfa") or [])}''',
     '''def _dev_row_of(side_rec):
    """侧车 -> 装置现算用的合成行 (只喂 args/ops 与族 C 现算所需的 vars/dom_dump)。"""
    ops = list(side_rec.get("ops") or [])
    sfa = list(side_rec.get("sfa") or [])
    steps = [{"op": op, "args": (dict(sfa[k]) if k < len(sfa) and sfa[k] else {})}
             for k, op in enumerate(ops)]
    meta = {}
    if side_rec.get("vars"):
        meta["vars"] = list(side_rec["vars"])
    if side_rec.get("dom_dump") is not None:
        meta["dom_dump"] = side_rec["dom_dump"]
    return {"name": side_rec.get("name"), "steps": steps, "meta": meta}


def _payload_of(side_rec):
    """gen33 [3] **装置现算口径** (变更96): 每步 8 维状态由 gen33_device 现算,
    **不读**行内/侧车金标 res (金标 res 仅用于变更97 门的校验件, 不进考试路径)。
    侧车若**自带**非空 res 且未显式要求现算 (诊断口径), 仍以现算为准 —— 单一状态源。"""
    row = _dev_row_of(side_rec)
    res, errs = [], []
    for k in range(len(row["steps"])):
        r8, _ex, _sp, _op, err = DEV.dev_resid(row, k)
        if err:
            res.append(None)
            errs.append({"k": k, "op": row["steps"][k]["op"], "err": err})
        else:
            res.append(r8)
    return {"res": res, "sfa": list(side_rec.get("sfa") or []),
            "dev_errs": errs,
            "res_src": "device_live(live_state)"}'''),
    # [5] consistency keyword table (nine new operators)
    ('''    "g24_ineq_ht2": ["nlinarith", "positivity", "sq_nonneg", "mul_nonneg"],
}''',
     '''    "g24_ineq_ht2": ["nlinarith", "positivity", "sq_nonneg", "mul_nonneg"],
    # --- gen33/v4: 9 个 gen32 深算子 (关键词按族语义登记; 命中率仅作诊断列) ---
    "gen32_nest_solve": ["induction", "hstep", "ih", "ring", "norm_num", "solve"],
    "gen32_nest_fold": ["fold", "rw", "ring", "mul"],
    "gen32_nest_delta": ["delta", "shift", "ring", "telescop"],
    "gen32_nest_close": ["ring", "push_cast", "norm_num", "linarith", "close"],
    "gen32_sos_block": ["positivity", "sq_nonneg", "nlinarith", "pow_two", "mul_self"],
    "gen32_sos_cross": ["positivity", "sq_nonneg", "nlinarith", "mul_nonneg", "pow_two"],
    "gen32_sos_tail_zero": ["nlinarith", "positivity", "sq_nonneg", "linarith"],
    "gen32_sos_tail_square": ["nlinarith", "positivity", "sq_nonneg", "sq_nonneg", "pow"],
    "gen32_sos_tail_other": ["nlinarith", "positivity", "sq_nonneg", "linarith", "norm_num"],
}'''),
    # [7]/CLI defaults
    ('''    ap.add_argument("--cache", default=os.path.join(G26, "cache", "hyb_a_all.bin"))
    ap.add_argument("--side", default=os.path.join(OUT, "A_RENDER_SIDECAR.jsonl"))
    ap.add_argument("--exam-side", default=os.path.join(OUT, "A_EXAM_SIDECAR.jsonl"))
    ap.add_argument("--allow", default=os.path.join(G13, "ccp_allow_ext.bin"))''',
     '''    ap.add_argument("--cache", default=os.path.join(G33, "cache", "deep_all.bin"))
    ap.add_argument("--side", default=os.path.join(OUT, "DEEP_RENDER_SIDECAR.jsonl"))
    ap.add_argument("--exam-side", default=os.path.join(OUT, "DEEP_EXAM_SIDECAR.jsonl"))
    ap.add_argument("--allow", default=os.path.join(G33, "cache", "ccp_allow_v4.bin"))'''),
    ('    ap.add_argument("--vocab", default="", help="A3: ccp-vocab/3 路径 (空=gen13 v2)")',
     '    ap.add_argument("--vocab", default=os.path.join(G33, "cache", "CCP_VOCAB_V4.json"),\n                    help="gen33: ccp-vocab/4 (head 194 / 值空间 352)")'),
    ('    ap.add_argument("--dev", default="cuda:0")',
     '    ap.add_argument("--dev", default="cuda:1")'),
    ('    ap.add_argument("--log", default=os.path.join(G30, "logs", "gen30_eval.log"))',
     '    ap.add_argument("--log", default=os.path.join(G33, "logs", "gen33_eval.log"))'),
    ('''    ap.add_argument("--seg", default="a_held",
                    choices=["a_held", "a_train", "exam_v1", "exam_zhenti"])''',
     '''    ap.add_argument("--seg", default="a_held",
                    choices=["a_held", "a_train", "exam_v1", "exam_zhenti", "exam_deep"])'''),
    ('''    ap.add_argument("--args-mode", default="bridge", choices=["bridge", "gold"],
                    help="HIGH-2: 正式评估锁 bridge (装置导参闭环); gold 仅诊断披露")''',
     '''    ap.add_argument("--args-mode", default="device",
                    choices=["device", "bridge", "gold"],
                    help="gen33 [3]: device = 装置现算状态 (正式口径); bridge/gold 诊断对照")'''),
    # segment selection: deep exam set
    ('''    ex = HY.load_sidecar(a.exam_side)
    want = "exam_main" if a.seg == "exam_v1" else "exam_zhenti"''',
     '''    ex = HY.load_sidecar(a.exam_side)
    want = ("exam_main" if a.seg == "exam_v1"
            else ("exam_deep" if a.seg == "exam_deep" else "exam_zhenti"))'''),
    # artifact names
    ('''    outp = a.out or os.path.join(OUT, "GEN30_EVAL_%s_%s%s.json"
                                 % (a.arm, a.seg, "" if a.args_mode == "bridge"
                                    else "_gold"))''',
     '''    outp = a.out or os.path.join(OUT, "GEN33_EVAL_%s_%s%s.json"
                                 % (a.arm, a.seg, "" if a.args_mode == "device"
                                    else "_%s" % a.args_mode))'''),
    ('''        _ap = getattr(a, "attn_out", "") or os.path.join(
            OUT, "GEN30_ATTN_SAMPLES%s.jsonl" % ("" if a.seg == "exam_v1" else "_%s" % a.seg))''',
     '''        _ap = getattr(a, "attn_out", "") or os.path.join(
            OUT, "GEN33_ATTN_SAMPLES%s.jsonl" % ("" if a.seg == "exam_v1" else "_%s" % a.seg))'''),
    ('''        jr = judge_texts(jt, os.path.join(G30, "lean", "judge_%s_%s" % (a.arm, a.seg)),
                         tag="gen30_%s_%s" % (a.arm, a.seg))''',
     '''        jr = judge_texts(jt, os.path.join(G33, "lean", "judge_%s_%s" % (a.arm, a.seg)),
                         tag="gen33_%s_%s" % (a.arm, a.seg))'''),
]


def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def patch(txt, patches, name):
    ok = True
    for i, (old, new) in enumerate(patches):
        n = txt.count(old)
        if n != 1:
            print("[FAIL] %s patch#%d hits %d times (required 1): %r" % (name, i + 1, n, old[:70]))
            ok = False
            continue
        txt = txt.replace(old, new)
    print("[%s] %s: %d patches" % ("OK" if ok else "BAD", name, len(patches)))
    return txt, ok


def head_insert(txt, src, note):
    lines = txt.split("\n")
    md5 = md5f(os.path.join(FORK, src))
    block = note % {"src": src, "md5": md5}
    return "\n".join([lines[0], block] + lines[1:]), md5


def main():
    srcs = [("gen30_hybrid.py", "gen33_hybrid.py", PATCH_HYB, HDR_HYB),
            ("gen30_eval.py", "gen33_eval.py", PATCH_EVAL, HDR_EVAL),
            ("gen30_planattn.py", "gen33_planattn.py", [], HDR_PA)]
    out = {}
    allok = True
    for s, d, patches, note in srcs:
        sp = os.path.join(FORK, s)
        txt = open(sp, encoding="utf-8").read()
        txt, md5 = head_insert(txt, s, note)
        txt, ok = patch(txt, patches, d)
        allok = allok and ok
        op = os.path.join(HERE, d)
        with open(op, "w", encoding="utf-8") as f:
            f.write(txt)
        out[d] = {"src": s, "src_md5": md5, "out": op, "out_md5": md5f(op),
                  "patches": len(patches), "ok": bool(ok),
                  "bytes": os.path.getsize(op)}
        print("[wrote] %s <- %s (%d bytes, md5 %s)" % (d, s, out[d]["bytes"],
                                                       out[d]["out_md5"]))
    import json
    with open(os.path.join(HERE, "FORK_LEDGER.json"), "w", encoding="utf-8") as f:
        json.dump({"schema": "gen33-fork-ledger/1", "here": HERE, "files": out}, f,
                  ensure_ascii=False, indent=1)
    print("FORK_OK=%s" % allok)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
