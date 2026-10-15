# -*- coding: utf-8 -*-
# drct_common.py -- DRCT 判卷公共件 (D-RCT 专组, 2026-09-22)
#
# 口径:
#   * Lean 真判 = lake env lean (cwd=miniF2F 根, PATH 前置 elan, timeout 600s):
#     rc==0 ∧ 非 not-run ∧ 无 sorryAx = accepted (xcmp_common.run_lean 同口径 + nice 19)。
#   * N1 剥参装饰判废 = **e0a 版语义** (scratch/v34/e0a_n1n2.py, 对拍胜出,
#     RECONCILIATION_N1.md): 只剥良构参数段保持语法合法 —— simp only [..]→simp、
#     simpa only [..] using h→simpa、(field_simp|norm_num|ring_nf|linarith|nlinarith|
#     simp_all|simp_arith|simp)[..]→裸头、(simpa|linarith|nlinarith) using X→裸头;
#     **不剥 exact/rw/apply/use 等必带参战术**。剥后仍闭合 ⇒ 装饰 ⇒ 判废。
#     ⚠️ xcmp_common.certify 的 N1 (剥全部实参) 系对拍判废的旧版, 本文件不得使用其语义。
#   * N2 = 原文 + set_option autoImplicit false + relaxedAutoImplicit false 复编译,
#     失败 ⇒ 判废。
#   * 认证解 = Lean 接受 ∧ N1 非装饰 ∧ N2 过 (V32 臂另需 witness 三闸, 见 drct_v32_judge)。
#
# 纪律: 只写 scratch/drct/**; probes/ 只读 import; 渲染头 = XCMP E3 官方头。
import importlib.machinery
import importlib.util
import json
import os
import re
import subprocess
import time

RES = "/home/data/ZHC/tcaboost/research"
MINIF2F = "/home/data/ZHC/tcaboost/lean/miniF2F-main"
DRCT = os.path.join(RES, "scratch", "drct")
NOT_RUN = ("does not exist", "no such file or directory", "LEAN_BIN_MISSING",
           "no default toolchain configured", "cannot execute")
_ELAN_BIN = os.path.expanduser("~/.elan/bin")


def safe_path(p, root=RES):
    rp = os.path.realpath(p)
    if not (rp == root or rp.startswith(root + os.sep)):
        raise ValueError("path escapes sandbox: %r" % (p,))
    return rp


def _load(name, path):
    ld = importlib.machinery.SourceFileLoader(name, path)
    sp = importlib.util.spec_from_loader(name, ld)
    m = importlib.util.module_from_spec(sp)
    ld.exec_module(m)
    return m


LE = _load("le_exam_drc", os.path.join(RES, "probes", "lean_exam.py.txt"))

# ---------------------------------------------------------------- 渲染 (E3 官方头)
OFFICIAL_HEADER = ["import MiniF2F.ProblemImports",
                   "open scoped Real Nat Topology Polynomial"]


def _normalize_indent(proof):
    lines = (proof or "").split("\n")
    if not any(ln.strip() for ln in lines):
        return ""
    first = next(ln for ln in lines if ln.strip())
    ind = len(first) - len(first.lstrip(" "))
    delta = 2 - ind
    out = []
    for ln in lines:
        if not ln.strip():
            out.append("")
        else:
            body = ln.lstrip(" ")
            out.append(" " * max(0, len(ln) - len(body) + delta) + body)
    return "\n".join(out).rstrip()


def render_lean(name, head, concl, proof, strict=False):
    L = list(OFFICIAL_HEADER)
    if strict:
        L.append("set_option autoImplicit false")
        L.append("set_option relaxedAutoImplicit false")
    L.append("")
    L.append("%s %s := by" % (head, concl))
    body = _normalize_indent(proof)
    if body:
        L.append(body)
    L.append("")
    L.append("#print axioms %s" % name)
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- Lean 真判
def run_lean(path, timeout=600.0, nice=False):
    """判卷真判 (变更 42 规则 R4: D-RCT 判卷全项目最高优先, XCMP 让位 → 默认不 nice;
    nice=True 保留给让位场景)。"""
    path = os.path.abspath(path)
    argv = (["nice", "-n", "19"] if nice else []) + ["lake", "env", "lean", path]
    env = dict(os.environ)
    env["PATH"] = _ELAN_BIN + os.pathsep + env.get("PATH", "")
    t0 = time.time()
    try:
        cp = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                            cwd=MINIF2F, env=env)
        out = (cp.stdout or "") + (cp.stderr or "")
        rc = cp.returncode
    except subprocess.TimeoutExpired:
        return {"rc": 124, "accepted": False, "sorry": False,
                "secs": round(time.time() - t0, 2), "out_head": "LEAN_TIMEOUT"}
    except FileNotFoundError as e:
        return {"rc": 127, "accepted": False, "sorry": False,
                "secs": round(time.time() - t0, 2), "out_head": "LEAN_BIN_MISSING: %s" % e}
    ok = (rc == 0) and (not any(k in out for k in NOT_RUN))
    sorry = ("sorryAx" in out) or ("declaration uses 'sorry'" in out)
    return {"rc": rc, "accepted": bool(ok and not sorry), "sorry": bool(sorry),
            "secs": round(time.time() - t0, 2), "out_head": out[:1200]}


# ---------------------------------------------------------------- N1 (e0a 语义)
_RE_SIMPA_ONLY = re.compile(r"\bsimpa\b\s+only\s*\[[^\]]*\]")
_RE_SIMP_ONLY = re.compile(r"\bsimp\b\s+only\s*\[[^\]]*\]")
_RE_BRACKET = re.compile(
    r"\b(field_simp|norm_num|ring_nf|linarith|nlinarith|simp_all|simp_arith|simp)\s*\[[^\]]*\]")
_RE_USING = re.compile(
    r"\b(simpa|linarith|nlinarith)\s+using\s+[A-Za-z_0-9.\[\]]+(?:\s+[A-Za-z_0-9.\[\]]+)*")


def strip_tactic_args(line):
    """e0a_n1n2.strip_tactic_args 逐字 (只剥良构参数段, 骨架保持语法合法)。"""
    s = line
    s = _RE_SIMPA_ONLY.sub("simpa", s)
    s = _RE_SIMP_ONLY.sub("simp", s)
    s = _RE_BRACKET.sub(r"\1", s)
    s = _RE_USING.sub(r"\1", s)
    return s


def certify(name, head, concl, proof, workdir, tag, timeout=600.0):
    """N1+N2 复判 (每个 raw accept 必跑)。返回 (certified, reasons, 详情)。"""
    os.makedirs(safe_path(workdir), exist_ok=True)
    base = re.sub(r"[^A-Za-z0-9_.-]", "_", "%s_%s" % (tag, name))
    # N1: e0a 剥参骨架重判 —— 剥后仍闭合 ⇒ 实参是装饰 ⇒ 判废
    sk = "\n".join(strip_tactic_args(ln) for ln in proof.split("\n"))
    # 空真守卫 (审核 S1 修复, 主代令 FIX-2): 剥离器未改动任何一行 (sk == proof) 时,
    # "剥后仍闭合" 为恒真推论, 不构成装饰证据 ⇒ 跳过 N1 判废并标记 n1_vacuous。
    n1_vacuous = (sk.strip() == proof.strip())
    n1_path = safe_path(os.path.join(workdir, base + "_skel.lean"))
    open(n1_path, "w", encoding="utf-8").write(render_lean(name, head, concl, sk))
    n1 = run_lean(n1_path, timeout)
    # N2: 双选项严格 (autoImplicit + relaxedAutoImplicit) —— 失败 ⇒ 判废
    n2_path = safe_path(os.path.join(workdir, base + "_strict.lean"))
    open(n2_path, "w", encoding="utf-8").write(render_lean(name, head, concl, proof, strict=True))
    n2 = run_lean(n2_path, timeout)
    reasons = []
    if n1["accepted"] and not n1_vacuous:
        reasons.append("bare_closure_after_strip")
    if not n2["accepted"]:
        reasons.append("autoimplicit_dependent")
    return (not reasons), reasons, {"n1": n1, "n2": n2, "stripped": sk,
                                    "n1_vacuous": bool(n1_vacuous),
                                    "n1_note": ("N1 不适用(无实参可剥: 剥参恒等)"
                                                if n1_vacuous else "N1 已判")}


# ---------------------------------------------------------------- 文本提取
_FENCE = re.compile(r"```lean4\n(.*?)\n```", re.DOTALL)


def extract_proof(gen_text):
    """协议 §8 同式: 围栏优先; 停在下一起始 theorem; 剥 import/open 前缀与孤立围栏行。"""
    if not gen_text:
        return ""
    m = _FENCE.search(gen_text)
    t = m.group(1) if m else gen_text
    mb = re.search(r"(?m)^theorem\s", t)
    if mb and mb.start() > 0:
        t = t[:mb.start()]
    lines, started = [], False
    for ln in t.split("\n"):
        s = ln.strip()
        if re.fullmatch(r"`{3,}[a-zA-Z0-9_-]*", s):
            continue
        if not started and (s.startswith("import ") or s.startswith("open ")):
            continue
        if s.startswith("import "):
            continue
        if s and not s.startswith("--"):
            started = True
        lines.append(ln)
    return "\n".join(lines).lstrip("\n").rstrip()


_FENCE_LINE = re.compile(r"\s*`{3,}[a-zA-Z0-9_-]*\s*")


def judge_attempt(name, head, concl, proof, workdir, tag, timeout=600.0):
    """单尝试全判: 渲染 → Lean 真判 → (accept 则) N1+N2。"""
    rec = {"certified": False, "reasons": [], "lean": None, "n1n2": None}
    proof = (proof or "").strip("\n")
    proof = "\n".join(ln for ln in proof.split("\n") if not _FENCE_LINE.fullmatch(ln))
    if not proof:
        rec["reasons"] = ["empty_proof"]
        return rec
    f = safe_path(os.path.join(workdir, re.sub(r"[^A-Za-z0-9_.-]", "_",
                                               "%s_%s.lean" % (tag, name))))
    open(f, "w", encoding="utf-8").write(render_lean(name, head, concl, proof))
    r = run_lean(f, timeout)
    rec["lean"] = {k: r[k] for k in ("rc", "accepted", "sorry", "secs")}
    if not r["accepted"]:
        rec["reasons"] = ["lean_reject" + ("_sorry" if r["sorry"] else "")]
        return rec
    ok, reasons, detail = certify(name, head, concl, proof, workdir, tag, timeout)
    rec["certified"] = ok
    rec["reasons"] = reasons
    rec["n1n2"] = {"n1_accepted": detail["n1"]["accepted"],
                   "n2_accepted": detail["n2"]["accepted"],
                   "n1_secs": detail["n1"]["secs"], "n2_secs": detail["n2"]["secs"],
                   "n1_vacuous": detail.get("n1_vacuous"),
                   "n1_note": detail.get("n1_note")}
    return rec


# ---------------------------------------------------------------- 题集装载
def load_problems(path=None):
    """DRCT 测试集 → [dict] (name/head/concl/stmt/stratum/...)。"""
    path = path or os.path.join(DRCT, "drct_problems.json")
    d = json.load(open(path, encoding="utf-8"))
    return d["problems"]


def _measured(r):
    """实测记录判定 (gen6; 变更 57 / R3 §4.4-1 处置)。

    判卷器 (drct_v2_judge) 对**链文件未覆盖**的题也产记录 (attempts 补零: no_plan 且
    ok_ops/why 皆无) —— 那些题不是"实测"。分母必须收敛到实测题数, 否则部分跑会产出
    "3/210" 假象 (R3 §4.2 实证件 _trackc_selftest_judge_summary.json)。
    优先读 judge 侧显式标记 r["measured"]; 缺标记 (旧记录) 时按内容推断:
    任一尝试带 ok_ops(非 None)/why(非空)/非 no_plan 原因/Lean 结果 ⇒ 实测。
    """
    if "measured" in r:
        return bool(r["measured"])
    att = r.get("attempts") or {}
    if not att:
        return False
    for a in att.values():
        a = a or {}
        if a.get("ok_ops") is not None or a.get("why") or a.get("lean") or a.get("n1n2"):
            return True
        if a.get("reasons") != ["no_plan"]:
            return True
    return False


def summarize(records):
    """records: [{problem, stratum, attempts: {k: rec}}] → 全卷+分档 pass@1/@8。

    gen6 (变更 57; R3 §4.4-1 分母收敛): 分母 = **实测记录数** (链文件覆盖过的题),
    不再把判卷器补零的未测题计入分母; 另出 n_measured / n_records 双值备查。
    全卷跑 (本卷全部臂) n_measured == n_records ⇒ 既有读数逐位不变, 无需重判。
    """
    def score(sub):
        p1 = sum(1 for r in sub if (r["attempts"].get("0") or r["attempts"].get("1")
                                    or {}).get("certified"))
        p8 = sum(1 for r in sub if any((a or {}).get("certified")
                                       for k, a in r["attempts"].items() if k != "0"))
        raw8 = sum(1 for r in sub if any((a or {}).get("lean", {}) and a["lean"].get("accepted")
                                         for k, a in r["attempts"].items() if k != "0"))
        n = len(sub)
        return {"n": n, "pass@1": p1, "pass@1_rate": round(p1 / max(1, n), 4),
                "pass@8": p8, "pass@8_rate": round(p8 / max(1, n), 4),
                "raw_lean_accept@8": raw8}
    meas = [r for r in records if _measured(r)]
    out = {"track_full": score(meas), "n_measured": len(meas), "n_records": len(records),
           "denominator": "n_measured (实测记录数; 变更 57 分母收敛)"}
    for L in ("L1", "L2", "L3"):
        out["track_" + L] = score([r for r in meas if r.get("stratum") == L])
    return out
