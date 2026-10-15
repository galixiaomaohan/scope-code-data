# -*- coding: utf-8 -*-
# gen41_payload.py -- gen41 device payload + the proof compiler (new-family chain -> Lean proof text)  **audit-fixed version (G-A)**
#
# Fix register (adversarial audit CRIT-1 / HIGH-6; fix plan = audit §5 G-A):
#   [R1] **operator convention unified on operator names** (the production path = gen34_eval.walk_chain emits op_name; the old
#        version wrongly compared integers 194..197 => family A always fell back to norm_num, family B always skipped, and the
#        compiled text was **byte-wise unrelated** to the emitted chain).
#   [R2] the compiled text must **vary with the emitted chain**: per-record compiled-text md5, plus a chain-sensitivity
#        self-test (drop a step / reorder / keep only close as three negatives => the text must change).
#   [R3] template registration + N1 non-triviality: the frozen judge's N1 rule (still closed after stripping tactic
#        arguments => rejected) systematically rejects "decorative" templates. Template choice = **measured**
#        (gen41_health.py tests each template with judge_attempt, results land in out/GEN41_TEMPLATE_GATE.json); this
#        module only takes templates from the registry; no template attached => bare norm_num + disclosure.
#   [R4] target operators are name keys; unknown name => empty/reject (fail-closed).
#
# Device payload (duty 1) unchanged: compute step state frames (8-dim) live from the in-row sfa, same convention
# as the training cache, slot for slot.
import collections
import hashlib

import gen41_common as C                                              # noqa: E402

G41 = C.G41
K_GRID = 8

# ---- operator name keys (G-A R1/R4) ----
A_BIND, A_CLOSE = "g41_subst_bind", "g41_subst_close"
B_ARITH, B_MOD = "g41_eval_arith", "g41_eval_mod"
FAM_A = "g41_alg_subst"
FAM_B = "g41_nt_eval"

# ---- template registry (G-A R3): backfilled after gen41_health.py measures; default = bare tactic (disclose chain-insensitivity) ----
TEMPLATE_REGISTRY = {
    # op name -> (template, note)
    B_ARITH: ("norm_num", "bare norm_num; the arith subclass is **chain-insensitive** (any chain gives the same text) -- the report must state this"),
    B_MOD: ("norm_num", "bare norm_num (N1 non-triviality allows it); decorated device templates are rejected by the frozen judge's N1 (HIGH-6)"),
}
TEMPLATE_SRC = "out/GEN41_TEMPLATE_GATE.json (measured artifact) or the default registry"


def set_templates(reg):
    """External (health-check module) backfill of measured templates (G-A R3)."""
    TEMPLATE_REGISTRY.update(reg)


def load_template_gate(path=None):
    """Read the measured template artifact and backfill (if present); on failure => keep the default registry (no silence)."""
    import json as _j
    p = path or (G41 + "/out/GEN41_TEMPLATE_GATE.json")
    try:
        d = _j.load(open(p, encoding="utf-8"))
    except Exception:                                                 # noqa: BLE001
        return {}
    rec = d.get("recommend") or {}
    reg = {}
    if rec.get("B_MOD"):
        reg[B_MOD] = (rec["B_MOD"], "measured GEN41_TEMPLATE_GATE.json md5=%s" % d.get("md5"))
    if rec.get("B_ARITH"):
        reg[B_ARITH] = (rec["B_ARITH"], "measured GEN41_TEMPLATE_GATE.json md5=%s" % d.get("md5"))
    if reg:
        set_templates(reg)
    return reg


load_template_gate()


def _load(name, path):
    import importlib.machinery as im
    import importlib.util as iu
    ld = im.SourceFileLoader(name, path)
    m = iu.module_from_spec(iu.spec_from_loader(name, ld))
    ld.exec_module(m)
    return m


CHK = _load("g41_chk_payload", G41 + "/gen41_checks.py")


def route_fixed(base, exp, m):
    """Powering via the cycle route (pre-period/cycle, rho detection; implemented differently from pow/binary exponentiation)."""
    if m <= 1:
        raise ValueError("bad modulus")
    if exp == 0:
        return 1 % m
    seq, seen, cur = [], {}, base % m
    while cur not in seen:
        seen[cur] = len(seq)
        seq.append(cur)
        cur = (cur * base) % m
    pre, cyc = seen[cur], len(seq) - seen[cur]
    if exp - 1 < pre:
        return seq[exp - 1]
    return seq[pre + (exp - 1 - pre) % cyc]


def resid_chain_of(close_args):
    """Family A: rebuild the residual state frame chain from the close step's arguments (independent checker path)."""
    chain, err = CHK.rebuild_resid_chain(
        CHK._parse(close_args["orig_target_expr"]),
        [{"var": b["var"], "expr": CHK._parse(b["expr"])} for b in close_args["binds"]],
        list(close_args["vars"]), int(close_args["nv"]))
    if err:
        raise RuntimeError("rebuild_resid_chain: %s" % err)
    return chain


def frames_of(sfa, family):
    """Compute step state frames live from sfa (8-dim float; same convention as the training cache, bit for bit)."""
    if family == FAM_A:
        chain = resid_chain_of(sfa[-1])
        return [[float(x) for x in chain[min(k, len(chain) - 1)]] for k in range(len(sfa))]
    a = sfa[0]
    if a.get("kind") in ("mod_eq", "mod_neq", "dvd", "ndvd"):
        r = route_fixed(int(a["base"]), int(a["exp"]), int(a["m"]))
        return [[float(r % (j + 2)) for j in range(K_GRID)]]
    return [[0.0] * K_GRID]


def payload_of_g41(side_rec):
    """gen41 replacement for gen34_eval._payload_of -> {"res": [...], "sfa": [...], "dev_errs": []}"""
    sfa = list(side_rec.get("sfa") or [])
    fam = side_rec.get("family")
    try:
        res = frames_of(sfa, fam)
        errs = []
    except Exception as e:                                            # noqa: BLE001
        res, errs = [None] * len(sfa), [{"err": repr(e)[:200]}]
    return {"res": res, "sfa": sfa, "dev_errs": errs,
            "res_src": "device_live(gen41_payload/frames_of)"}


# ================================================================ the proof compiler (G-A fix version)

def pv_of_exam(rec):
    """Exam row -> compile view (same usage as gen35_compile.pv_of_exam)."""
    return {"ri": int(rec["ri"]), "name": rec["name"], "family": rec.get("family"),
            "cls": rec.get("cls"), "head": rec.get("head"), "concl": rec.get("concl"),
            "ops": list(rec.get("ops") or []), "sfa": [dict(x) for x in (rec.get("sfa") or [])],
            "hyp_names": rec.get("hyp_names") or [],
            "stmt_state": rec.get("stmt_state"),
            "gold_proof": rec.get("proof_text"),
            "binders": rec.get("binders"), "concl_text": rec.get("concl_text")}


def _a_proof(pv, walked_ops, mut):
    """Family A: take hyps by emitted **position order**; the structure must be [bind]*n + [close] with the same length as gold (chain-sensitive)."""
    gold = pv["ops"]
    hyps = [s.get("hyp") for s in pv["sfa"][:-1]]
    n_gold_bind = len(gold) - 1
    binds = [i for i, o in enumerate(walked_ops) if o == A_BIND]
    closes = [i for i, o in enumerate(walked_ops) if o == A_CLOSE]
    if not walked_ops:
        mut["err"] = "EMPTY_CHAIN"
        return "skip"
    if len(closes) != 1 or closes[-1] != len(walked_ops) - 1:
        mut["err"] = "NO_CLOSE_AT_END"
        return "skip"
    if any(o != A_BIND for o in walked_ops[:-1]):
        mut["err"] = "NON_BIND_INTERIOR"
        return "skip"
    if len(binds) != n_gold_bind:
        mut["err"] = "BIND_COUNT_%d_vs_%d" % (len(binds), n_gold_bind)
        use = hyps[:len(binds)]
    else:
        use = hyps
    if any(h is None for h in use):
        mut["err"] = "HYP_MISSING"
        use = [h for h in use if h]
    if not use:
        return "norm_num"
    return "norm_num [%s]" % ", ".join(use)


def _b_proof(pv, walked_ops, mut):
    """Family B: single-step closed form; the first emitted operator picks the template (template = measured registry value)."""
    if not walked_ops:
        mut["err"] = "EMPTY_CHAIN"
        return "skip"
    if len(walked_ops) != 1:
        mut["err"] = "B_MULTI_STEP"
        return "skip"
    op = walked_ops[0]
    t = TEMPLATE_REGISTRY.get(op)
    if t is None:
        mut["err"] = "OP_UNKNOWN_%s" % op
        return "skip"
    return t[0]


def compile_chain_g41(pv, walked_ops, keep_recs=True):
    """Chain -> proof text + per-step trace (interface as gen35_compile.compile_chain; **name keys**)."""
    gold = list(pv["ops"])
    fam = pv.get("family")
    mut = {"err": None}
    if fam == FAM_A:
        text = _a_proof(pv, walked_ops, mut)
    elif fam == FAM_B:
        text = _b_proof(pv, walked_ops, mut)
    else:
        mut["err"] = "FAMILY_UNSUPPORTED"
        text = "skip"
    n_emit, n_mismatch = len(walked_ops), sum(
        1 for a, b in zip(walked_ops, gold) if a != b)
    exact = bool(walked_ops == gold)
    first = bool(walked_ops and gold and walked_ops[0] == gold[0])
    over = len(walked_ops) > len(gold)
    err_hist = collections.Counter()
    if mut["err"]:
        err_hist[mut["err"]] += 1
    steps = [{"k": k, "op": o,
              "err": (None if (k < len(gold) and gold[k] == o) else "OP_MISMATCH")}
             for k, o in enumerate(walked_ops)]
    return {"proof_text": text, "n_lines": (0 if not text.strip() else text.count("\n") + 1),
            "proof_md5": hashlib.md5(text.encode("utf-8")).hexdigest(),
            "first_hit": first, "exact_hit": exact, "over_run": bool(over),
            "n_emit": n_emit, "n_mismatch": n_mismatch, "err_hist": dict(err_hist),
            "steps": steps, "fam": fam, "cls": pv.get("cls"),
            "tpl_src": TEMPLATE_SRC}


# ================================================================ chain-sensitivity self-test (G-A R2)

def _negatives(walked):
    """Negatives: multi-step chain = drop the last step / reorder / keep only close; single-step chain (family B) = wrong operator (the other op name)."""
    out = {}
    if len(walked) >= 2:
        out["trunc_drop_close"] = list(walked[:-1])          # last-step truncation (shorter chain)
        if len(walked) >= 3:
            out["drop_interior"] = [walked[0]] + list(walked[2:])   # drop an interior bind (count changes)
        out["only_close"] = [walked[-1]]                     # keep only the closing step
    elif len(walked) == 1:
        other = B_ARITH if walked[0] == B_MOD else B_MOD
        out["wrong_op"] = [other]                            # single-step chain: wrong operator
    return out


def selftest_chain_sensitivity(cases, logp=None):
    """cases: [(pv, walked_gold)] -> register: per-case compiled-text md5 + whether the negatives change the text.

    Gate: for each case, all three negative compiled texts must differ from the gold text (otherwise = chain-insensitive, counted in bad).
    """
    rows, n_bad = [], 0
    for pv, walked in cases:
        g = compile_chain_g41(pv, walked)
        item = {"name": pv.get("name"), "family": pv.get("family"),
                "gold_text": g["proof_text"], "gold_md5": g["proof_md5"],
                "neg": {}}
        for k, w in _negatives(walked).items():
            r = compile_chain_g41(pv, w)
            item["neg"][k] = {"text": r["proof_text"], "md5": r["proof_md5"],
                              "changed": bool(r["proof_md5"] != g["proof_md5"]),
                              "err": r["err_hist"]}
        changed = [v["changed"] for v in item["neg"].values()]
        item["all_neg_changed"] = bool(changed and all(changed))
        if not item["all_neg_changed"]:
            n_bad += 1
            item["insensitive_reason"] = ("single-step-chain template has no argument segment (family B bare norm_num) => switching operators gives the same text, "
                                          "a HIGH-6 disclosed convention limit" if len(walked) <= 1
                                          else "negatives did not change the compiled text")
        rows.append(item)
    fam_stat = {}
    for r in rows:
        d = fam_stat.setdefault(r["family"], {"n": 0, "sens": 0})
        d["n"] += 1
        d["sens"] += 1 if r["all_neg_changed"] else 0
    rep = {"schema": "gen41-chain-sensitivity/1", "n_cases": len(cases), "n_bad": n_bad,
           "pass": bool(n_bad == 0 and len(cases) > 0), "by_family": fam_stat,
           "rule": ("multi-step-chain negatives = drop an interior step / truncate the last step / keep only the closing step (compiled-text md5 must change); "
                    "single-step-chain negative = wrong operator (with a parameter-free template the text is necessarily identical => record insensitive and disclose)"),
           "templates": {k: list(v) for k, v in sorted(TEMPLATE_REGISTRY.items())},
           "cases": rows}
    if logp:
        C.log("G-A chain-sensitivity self-test: n=%d bad=%d pass=%s" % (len(cases), n_bad, rep["pass"]),
              logp)
    return rep
