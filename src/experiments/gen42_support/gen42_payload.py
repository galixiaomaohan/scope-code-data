# -*- coding: utf-8 -*-
# gen42_payload.py -- gen42 compiler (chain -> proof text; operator-name keyed, chain-sensitive)
# [new in this batch, not a fork]
#
# Convention (V42_DESIGN section 1): terminal operators always use **argument-free tactics** (N1
#   strip is the identity => the vacuity guard passes); chain sensitivity is guaranteed by the
#   "operator -> rendered line" map; family A's terminal step is the exception (norm_num [hyps], whose
#   argument = the chain's binder order).
#   E2 erratum: g42_s_mod does not produce corpus chains (kept in the vocabulary only); if it shows up
#   at compile time it is rendered per the table.
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import gen42_families as F                                              # noqa: E402

TACTIC_LINE = {
    "g42_s_reduce": None,        # decided by the variant (see reduce_line)
    "g42_s_poly": "ring",
    "g42_s_eval": "norm_num",
    "g42_s_mod": None,           # two lines
    "g42_s_lin": "linarith",
    "g42_s_omega": "omega",
    "g42_s_pos": "positivity",
    "g42_sb_step": None,         # pure state step
    "g42_sb_close": None,        # family A: norm_num [hyps]
}
REDUCE_LINE = {"push": "push_cast", "fs": "field_simp", "nn1": "norm_num1"}


def pv_of_exam(rec):
    """Exam-set row -> compile view (same field surface as gen41_payload)."""
    return {"ri": int(rec["ri"]), "name": rec.get("name"), "family": rec.get("family"),
            "cls": rec.get("cls"), "route": rec.get("route"),
            "head": rec.get("head"), "concl": rec.get("concl"),
            "ops": list(rec.get("ops") or []), "sfa": [dict(x) for x in (rec.get("sfa") or [])],
            "hyp_names": rec.get("hyp_names") or [],
            "stmt_state": rec.get("stmt_state"), "gold_proof": rec.get("proof_text")}


def compile_chain_g42(pv, walked_ops, keep_recs=True):
    """Chain -> proof text (name-keyed; each line is determined by the operator => a different chain
    necessarily gives different text, and swapping a single terminal operator necessarily changes the
    text versus its sibling operator)."""
    fam = pv.get("family") or ""
    sfa = pv.get("sfa") or []
    route = pv.get("route")
    mut = {"err": None}
    lines = []
    if not walked_ops:
        mut["err"] = "EMPTY_CHAIN"
        return _ret(pv, walked_ops, "skip", mut)
    if fam == F.FAM_SUBST:
        # steps = [sb_step]*k + [sb_close]; the terminal argument = the hyp of each step on the chain (positional order)
        if walked_ops[-1] != "g42_sb_close":
            mut["err"] = "NO_SB_CLOSE_AT_END"
            return _ret(pv, walked_ops, "skip", mut)
        if any(o != "g42_sb_step" for o in walked_ops[:-1]):
            mut["err"] = "NON_SB_STEP_INTERIOR"
        hyps = [s.get("hyp") for s in sfa[:len(walked_ops) - 1]]
        hyps = [h for h in hyps if h]
        n_gold = len(pv["ops"]) - 1
        if len(walked_ops) - 1 != n_gold:
            mut["err"] = mut["err"] or ("BIND_COUNT_%d_vs_%d" % (len(walked_ops) - 1, n_gold))
        if not hyps:
            return _ret(pv, walked_ops, "skip", mut)
        lines = ["norm_num [%s]" % ", ".join(hyps)]
    else:
        for k, op in enumerate(walked_ops):
            if op == "g42_s_reduce":
                var = (sfa[k] if k < len(sfa) else {}).get("var") or "nn1"
                lines.append(REDUCE_LINE.get(var, "norm_num1"))
            elif op == "g42_s_mod":
                lines.append("rw [Nat.pow_mod]")
                lines.append("norm_num")
            elif op in ("g42_s_poly", "g42_s_eval", "g42_s_lin", "g42_s_omega",
                        "g42_s_pos"):
                lines.append(TACTIC_LINE[op])
            else:
                mut["err"] = "OP_UNKNOWN_%s" % op
                return _ret(pv, walked_ops, "skip", mut)
        if not lines:
            mut["err"] = mut["err"] or "NO_LINES"
            return _ret(pv, walked_ops, "skip", mut)
    return _ret(pv, walked_ops, "\n".join(lines), mut)


def _ret(pv, walked, text, mut):
    gold = list(pv.get("ops") or [])
    return {"proof_text": text,
            "n_lines": (0 if not text.strip() else text.count("\n") + 1),
            "proof_md5": hashlib.md5(text.encode("utf-8")).hexdigest(),
            "first_hit": bool(walked and gold and walked[0] == gold[0]),
            "exact_hit": bool(walked == gold),
            "over_run": len(walked) > len(gold),
            "n_emit": len(walked),
            "n_mismatch": sum(1 for a, b in zip(walked, gold) if a != b),
            "err_hist": ({mut["err"]: 1} if mut.get("err") else {}),
            "steps": [{"k": k, "op": o,
                       "err": (None if (k < len(gold) and gold[k] == o) else "OP_MISMATCH")}
                      for k, o in enumerate(walked)],
            "fam": pv.get("family"), "cls": pv.get("cls"), "tpl_src": "gen42_payload/1"}


def negatives(walked):
    """Chain-sensitivity negative cases: multi-step = truncate the last step / drop an interior step /
    keep only the terminal step; single-step = swap in the sibling operator."""
    out = {}
    if len(walked) >= 2:
        out["trunc_drop_close"] = list(walked[:-1])
        if len(walked) >= 3:
            out["drop_interior"] = [walked[0]] + list(walked[2:])
        out["only_close"] = [walked[-1]]
    elif len(walked) == 1:
        sib = {"g42_s_eval": "g42_s_poly", "g42_s_poly": "g42_s_eval",
               "g42_s_lin": "g42_s_omega", "g42_s_omega": "g42_s_lin",
               "g42_s_pos": "g42_s_eval", "g42_sb_close": "g42_s_eval"}
        out["sibling_op"] = [sib.get(walked[0], "g42_s_eval")]
    return out


def selftest(cases):
    """cases: [(pv, walked_gold)] -> whether every negative case changes the compiled text, per case."""
    rows, n_bad = [], 0
    for pv, walked in cases:
        g = compile_chain_g42(pv, walked)
        item = {"name": pv.get("name"), "family": pv.get("family"),
                "route": pv.get("route"), "gold_text": g["proof_text"],
                "neg": {}}
        for k, w in negatives(walked).items():
            r = compile_chain_g42(pv, w)
            item["neg"][k] = {"text": r["proof_text"],
                              "changed": bool(r["proof_md5"] != g["proof_md5"])}
        ch = [v["changed"] for v in item["neg"].values()]
        item["all_neg_changed"] = bool(ch and all(ch))
        if not item["all_neg_changed"]:
            n_bad += 1
        rows.append(item)
    return {"schema": "gen42-chain-sensitivity/1", "n": len(cases), "n_bad": n_bad,
            "pass": bool(n_bad == 0 and cases), "cases": rows}
