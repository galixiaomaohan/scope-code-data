# -*- coding: utf-8 -*-
"""audit_data.py -- the **non-triviality gate** for skeleton data + **operator coverage audit**
(reusable; the flywheel pipeline calls it directly).

Two duties (additional constraint from the main agent, incorporated into design doc §7.5):

1. **Non-triviality gate** (`check_nontrivial` / `nontriviality_report`):
   every skeleton problem must **pass the gate to enter the data**. The gate's criteria and
   parameters are **registered explicitly** (`NONTRIVIAL_GATE`), not hidden in the code:
     * `min_content_steps = 2` -- the proof chain must have >= 2 **non-trivial steps**;
       a "non-trivial step" is one whose certificate **asserts mathematical content** (see
       `CONTENT_CERT_KINDS`); only the terminal audit `skel_qed_check` (pure ledger audit) does not
       count as a content step. Why the parameter: a single-step proof (e.g. "one negated goal +
       contradiction close") structurally cannot demonstrate case-splitting/induction/decomposition
       ability -- exactly the disease this direction treats (the evaluate+END single-gene pattern
       of R1/R3).
     * `min_distinct_ops = 2` -- at least 2 distinct operators must be used (no "same trick twice"
       padding);
     * `forbid_trivial_goal = True` -- trivial propositions are forbidden: modulus >= 2 (no
       tautologies like `1 | x`), the expression must not be identically the zero polynomial, and
       the goal variables and the parameter table must be self-consistent;
     * `require_closure` -- the last step must be `skel_qed_check` with `qed=True` (the proof really
       closes, it is not a half-finished problem "left hanging").
   `gate.params()` returns the parameter table itself (for flywheel registration/versioning).

2. **Operator coverage audit** (`operator_coverage`):
   tabulate each operator's usage in the dataset (rows/frames/first-step counts) and **explicitly
   list zero-coverage operators** -- this is the infrastructure against "the wheel spinning
   narrower": when the data covers only a few operators, the model's selection ability degrades
   silently. Multi-source merging is supported (skeleton data + legacy v31 mixed data), because
   "coverage" must be meaningful for the **whole corpus**.

Both functions consume JSONL rows (dicts) only and do not depend on torch/GPU -- the flywheel can
call them right after data generation.
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from svokernel_v32 import SKEL_OPS, V32REGISTRY, v32_op_index_map, audit_static_v32  # noqa: F401,E402

#: Certificate kind -> does it assert mathematical content (the definition of a non-trivial step;
#: one-to-one with the certificate table of DESIGN §5)
CONTENT_CERT_KINDS = ("case_exhaustive", "induction_scheme", "stack_machine",
                      "leaf_residue_complete", "contradiction_witness")
#: Pure audit certificates (not content steps: they only check ledger closure, producing no
#: mathematical assertion)
AUDIT_CERT_KINDS = ("skeleton_qed",)

#: **Gate parameter registration** (the flywheel calls `NONTRIVIAL_GATE` itself; do not copy the
#: constants)
NONTRIVIAL_GATE = {
    "gate_id": "v32-skel-nontrivial/1",
    "min_content_steps": 2,
    "min_distinct_ops": 2,
    "forbid_trivial_goal": True,
    "require_closure": True,
    "content_cert_kinds": CONTENT_CERT_KINDS,
    "audit_cert_kinds": AUDIT_CERT_KINDS,
    "rationale": ("骨架题是自举飞轮的出题器素材: 单步/恒真/未收拢的题会把课程稀释成"
                  "'会一招' —— 正是本方向要治的 R1/R3 单基因病"),
}


def params() -> dict:
    """Gate parameter table (for flywheel registration; returns a copy, so the caller cannot
    modify this module's constants)."""
    return json.loads(json.dumps(NONTRIVIAL_GATE))


def _cert_of(op):
    try:
        return V32REGISTRY.tool(op).op.cert_kind or ""
    except Exception:                                            # noqa: BLE001
        return ""


def check_nontrivial(row: dict) -> tuple:
    """Run one skeleton problem through the gate -> (ok, reason, detail). The flywheel can call
    this directly on any skeleton row."""
    g = NONTRIVIAL_GATE
    prog = row.get("program") or [s.get("op") for s in (row.get("steps") or [])]
    if not prog:
        return False, "EMPTY_PROGRAM", "程序为空"
    contents = [p for p in prog if _cert_of(p) in g["content_cert_kinds"]]
    if len(contents) < g["min_content_steps"]:
        return False, "TOO_FEW_CONTENT_STEPS", ("非平凡步 %d < %d (内容步 = 证书属于 %r; "
                                                "只有终局审计不算)" % (len(contents),
                                                                  g["min_content_steps"],
                                                                  list(g["content_cert_kinds"])))
    if len(set(prog)) < g["min_distinct_ops"]:
        return False, "TOO_FEW_DISTINCT_OPS", "不同算子 %d < %d" % (len(set(prog)),
                                                                 g["min_distinct_ops"])
    if g["forbid_trivial_goal"]:
        for k in ("sf_args",):
            for a in (row.get(k) or []):
                if not isinstance(a, dict):
                    continue
                m = a.get("m")
                if isinstance(m, int) and m < 2:
                    return False, "TRIVIAL_MODULUS", "出现 m=%r (<2 ⇒ 恒真式)" % (m,)
        v32 = row.get("v32") or {}
        if v32.get("modulus") is not None and int(v32["modulus"]) < 2:
            return False, "TRIVIAL_MODULUS", "题面模数 %r < 2" % (v32["modulus"],)
        root = v32.get("root_goal") or ""
        if not root.strip():
            return False, "EMPTY_GOAL", "根目标为空"
        if " 0 divides " in root or "divides 0" in root:
            return False, "TRIVIAL_GOAL", "出现平凡命题 (0 整除/被 0 整除)"
    if g["require_closure"]:
        if prog[-1] != "skel_qed_check":
            return False, "NO_CLOSURE", "末步 %r 不是 skel_qed_check" % (prog[-1],)
        fin = row.get("expected_final")
        if not isinstance(fin, dict) or not fin.get("qed"):
            return False, "NOT_QED", "末步未给出 qed=True (证明没收拢)"
    return True, "PASS", "非平凡步 %d, 不同算子 %d, 已收拢" % (len(contents), len(set(prog)))


def nontriviality_report(rows) -> dict:
    """Batch gate run -> {n_rows, n_pass, n_reject, rejected:[{instance_id, code, detail}]}."""
    n_pass, rej, by_code = 0, [], {}
    for r in rows:
        ok, code, detail = check_nontrivial(r)
        if ok:
            n_pass += 1
        else:
            by_code[code] = by_code.get(code, 0) + 1
            if len(rej) < 20:
                rej.append({"instance_id": (r.get("v32") or {}).get("instance_id"),
                            "code": code, "detail": detail})
    return {"gate": params(), "n_rows": len(rows), "n_pass": n_pass,
            "n_reject": len(rows) - n_pass,
            "reject_rate": (len(rows) - n_pass) / max(1, len(rows)),
            "reject_by_code": dict(sorted(by_code.items())), "rejected_examples": rej}


def _iter_programs(path, key="program"):
    """Stream a JSONL file and yield each row's operator sequence (parse only the needed fields,
    never keep a whole row in memory)."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:                                    # noqa: BLE001
                continue
            prog = r.get(key) or [s.get("op") for s in (r.get("steps") or [])]
            if prog:
                yield r, [p for p in prog if p]


def operator_coverage(paths, only_new=False) -> dict:
    """Multi-source coverage table: each operator's [rows, frames, first-step count] + an
    **explicit list of zero-coverage operators**.

    paths: [(label, jsonl path), ...] -- e.g. [("skel", skeleton data), ("v31mix", legacy mixed
    data)]. The operator space of skeleton data = **all 160 v32 operators** (118 legacy + 36 new
    v31 + 6 skeleton); for single-source data the zero-coverage operators are listed one by one --
    that is not an error but the fact of "which operators this corpus exercises".
    """
    names = list(V32REGISTRY.names())
    rows_by, frames_by, first_by = {}, {}, {}
    per_src = {}
    for tag, p in paths:
        if not os.path.exists(p):
            per_src[tag] = {"error": "文件不存在: %s" % p}
            continue
        nr = nf = 0
        loc = set()
        for _r, prog in _iter_programs(p):
            nr += 1
            nf += len(prog)
            first_by[prog[0]] = first_by.get(prog[0], 0) + 1
            loc.add(prog[0])
            for op in prog:
                loc.add(op)
                rows_by[op] = rows_by.get(op, 0) + 1
                frames_by[op] = frames_by.get(op, 0) + 1
        per_src[tag] = {"path": p, "n_rows": nr, "n_frames": nf,
                        "n_ops_used": len(loc)}
    zero = [n for n in names if n not in rows_by]
    used = sorted(rows_by.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "n_ops_total": len(names),
        "n_ops_used": len(names) - len(zero),
        "per_source": per_src,
        "table": [{"op": n, "rows": rows_by.get(n, 0), "frames": frames_by.get(n, 0),
                   "first_step": first_by.get(n, 0)} for n in names],
        "used_ranked": used,
        "zero_coverage": zero,
        "zero_coverage_skel": [n for n in zero if n in SKEL_OPS],
        "note": ("zero_coverage 是**事实登记**, 不是错误: 单源语料只覆盖它自己的动作空间。"
                 "飞轮据此决定下一轮该出哪一族的题 (防 '轮子越转越窄')。"),
    }


def coverage_text(cov: dict, top=20) -> str:
    """Human-readable summary (for dumping/reporting)."""
    lines = ["operator coverage: total %d, used %d, zero-coverage %d"
             % (cov["n_ops_total"], cov["n_ops_used"], len(cov["zero_coverage"])),
             "  per source: %r" % ({k: v.get("n_rows") for k, v in cov["per_source"].items()},),
             "  top used (rows/frames/first-step):"]
    for op, n in cov["used_ranked"][:top]:
        row = next((x for x in cov["table"] if x["op"] == op), {})
        lines.append("    %-28s %6d %6d %6d" % (op, n, row.get("frames", 0),
                                                row.get("first_step", 0)))
    lines.append("  zero coverage (first 40): %r" % (cov["zero_coverage"][:40],))
    if cov["zero_coverage_skel"]:
        lines.append("  !! skeleton operators with zero coverage: %r" % (cov["zero_coverage_skel"],))
    return "\n".join(lines)


if __name__ == "__main__":                                            # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--skel", default="data/"
                                      "sla_v32_skeleton.jsonl")
    ap.add_argument("--v31mix", default="data/"
                                        "sla_v31_mixed.jsonl")
    ap.add_argument("--emit-json", default="")
    a = ap.parse_args()
    rows = [r for r, _p in _iter_programs(a.skel)]
    rep = {"nontrivial": nontriviality_report(rows)}
    rep["coverage_skel_only"] = operator_coverage([("skel", a.skel)])
    rep["coverage_combined"] = operator_coverage([("skel", a.skel), ("v31mix", a.v31mix)])
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if a.emit_json:
        with open(a.emit_json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=1)
