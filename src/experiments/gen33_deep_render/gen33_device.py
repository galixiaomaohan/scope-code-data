# -*- coding: utf-8 -*-
# gen33_device.py -- deep-generation V3.5 **_prefix device recomputer** (change96): per-step state recomputation for the nine new gen32 operators
#
# Positioning (pre-registered V33_DESIGN §3 / D-1): the 8-dim state of exam rows and chain-walk prefixes is
#   **recomputed by the device from the row's own data**; reading gold chain state is **forbidden**
#   (in-row steps[].res is used for [verification] only, never for [supply]).
#
# State semantics (DEEP_REPORT §3): resid = exact values of the main operation polynomial at the 8 fixed
#   integer-lattice points P_j = (j+2, j+3, …, j+2+nv−1), j=0..7 (length always 8).
#
# Recompute rules (derived empirically; full 15791/15791 steps recomputed bit-identically = this module's self-check reading):
#   * gen32_nest_{solve,fold,delta,close} / gen32_sos_tail_{zero,square,other}:
#       args carry the full payload ⇒ call gen32_ops.call(op, args) directly.
#   * gen32_sos_{block,cross}: the args recorded in corpus/exam rows **lack** the three device-injected fields
#       (vars / R_in / hint_idx). Rebuilt on the fly (using only the row's own data):
#         vars     := meta.vars
#         hint_idx := (count of sos blocks before this step) + 1   (device injects by chain position)
#         R_in     := goal_dom − Σ_{i<k} mult_i · ℓ_i²   (ℓ_i rebuilt from step i's args.ell via _ell_poly)
#     where goal_dom := meta.dom_dump (family-C problem-domain polynomial).
#
# Dependencies: gen32_ops / gen32_poly (import only, zero modification) · gen32_families' _ell_poly convention
#   (kept local here, avoiding an import of the whole generator family). Existing artifacts untouched.
import sys
from fractions import Fraction

import g33_common as C

if C.G32 not in sys.path:
    sys.path.insert(0, C.G32)
if C.BH not in sys.path:
    sys.path.insert(0, C.BH)

import gen32_ops as OPS                                               # noqa: E402
import gen32_poly as P                                               # noqa: E402

RES_DIM = 8
NEW_OPS = ("gen32_nest_solve", "gen32_nest_fold", "gen32_nest_delta", "gen32_nest_close",
           "gen32_sos_block", "gen32_sos_cross",
           "gen32_sos_tail_zero", "gen32_sos_tail_square", "gen32_sos_tail_other")
SOS_BLOCK_OPS = ("gen32_sos_block", "gen32_sos_cross")
#: Observation axis: args fields a corpus row must carry (any missing -> reject, fail-closed)
REQUIRED_ARGS = {
    "gen32_nest_solve": ("level", "nv", "v0", "deg", "S"),
    "gen32_nest_fold": ("level", "nv", "scale", "lower"),
    "gen32_nest_delta": ("level", "nv", "upper", "inc"),
    "gen32_nest_close": ("level", "nv", "P", "T_text", "cls"),
    "gen32_sos_block": ("ell", "mult"),
    "gen32_sos_cross": ("ell", "mult"),
    "gen32_sos_tail_zero": ("vars", "rho", "goal_dom", "hints", "hints_poly"),
    "gen32_sos_tail_square": ("vars", "rho", "goal_dom", "hints", "hints_poly", "sq"),
    "gen32_sos_tail_other": ("vars", "rho", "goal_dom", "hints", "hints_poly"),
}


def ell_poly(args):
    """Local ℓ-polynomial rebuild (formula-identical to gen32_ops._ell_poly)."""
    vars_ = list(args["vars"])
    nv = len(vars_)
    idx = {v: i for i, v in enumerate(vars_)}
    out = {}
    for v, cf in args["ell"]["coefs"].items():
        if Fraction(cf) == 0:
            continue
        m = [0] * nv
        m[idx[v]] = 1
        out[tuple(m)] = Fraction(cf)
    sh = Fraction(args["ell"].get("shift", 0))
    if sh != 0:
        out[(0,) * nv] = out.get((0,) * nv, Fraction(0)) - sh
    return out


def _dom_dump(row):
    """goal_dom payload source: meta.dom_dump on family-C rows; exam rows use the same field (same generator version)."""
    meta = row.get("meta") or {}
    d = meta.get("dom_dump")
    if d is None:
        # Fallback: terminal-step args.goal_dom (only tail operators carry it, but equivalent to dom)
        for s in reversed(row.get("steps") or []):
            if s.get("args", {}).get("goal_dom") is not None:
                return s["args"]["goal_dom"]
        raise KeyError("no dom_dump (meta.dom_dump / steps[].args.goal_dom 皆缺)")
    return d


def dev_args(row, k):
    """Device-execution args for step k (payload rebuilt on the fly; in-row res never read)."""
    steps = row["steps"]
    s = steps[k]
    op = s["op"]
    a = dict(s.get("args") or {})
    if op in SOS_BLOCK_OPS:
        vars_ = list(a.get("vars") or (row.get("meta") or {}).get("vars") or [])
        if not vars_:
            raise KeyError("sos 块缺 vars (args.vars / meta.vars 皆无)")
        R = dict(P.load(_dom_dump(row)))
        nb = 0
        for j in range(k):
            sj = steps[j]
            if sj["op"] in SOS_BLOCK_OPS:
                aa = dict(sj["args"])
                aa["vars"] = vars_
                ell = ell_poly(aa)
                R = P.psub(R, P.pscale(P.pmul(ell, ell), Fraction(sj["args"]["mult"])))
                nb += 1
        a.update({"vars": vars_, "R_in": P.dump(R), "hint_idx": nb + 1})
    return a


def dev_resid(row, k):
    """Recompute the 8-dim state for step k -> (res8 float list, resid_exact list, spec, op, err)."""
    s = row["steps"][k]
    op = s["op"]
    miss = [x for x in REQUIRED_ARGS.get(op, ()) if x not in (s.get("args") or {})]
    if miss:
        return None, None, None, op, "MISSING_ARGS:%s" % ",".join(miss)
    try:
        r = OPS.call(op, dev_args(row, k))
    except Exception as e:                                           # noqa: BLE001
        return None, None, None, op, "EXC:%s:%s" % (type(e).__name__, e)
    if r.err:
        return None, None, None, op, "DEV_ERR:%s" % r.err
    res = [float(x) for x in r.resid]
    return res, list(r.resid_exact), r.resid_spec, op, None


def dev_resid_all(row):
    """Recompute a whole row -> list[(res8, exact, spec, op, err)]."""
    return [dev_resid(row, k) for k in range(len(row["steps"]))]


def state_source_report(rows, tag="", logp=None):
    """State-source health check (change-96 pre-exam gate): recompute row by row, step by step; tally nonzero rate, errors, all-zero rows."""
    n_step = n_nonzero = n_err = 0
    err_hist, spec_hist, op_hist = {}, {}, {}
    bad = []
    for r in rows:
        for k, (res, _ex, spec, op, err) in enumerate(dev_resid_all(r)):
            n_step += 1
            op_hist[op] = op_hist.get(op, 0) + 1
            if err:
                n_err += 1
                err_hist[err] = err_hist.get(err, 0) + 1
                if len(bad) < 8:
                    bad.append({"name": r.get("name"), "k": k, "op": op, "err": err})
                continue
            spec_hist[spec] = spec_hist.get(spec, 0) + 1
            if any(x != 0.0 for x in res):
                n_nonzero += 1
            else:
                if len(bad) < 8:
                    bad.append({"name": r.get("name"), "k": k, "op": op, "err": "ALL_ZERO"})
    rep = {"schema": "gen33-state-source/1", "tag": tag, "n_rows": len(rows),
           "n_step": n_step, "n_nonzero": n_nonzero,
           "nonzero_rate": round(n_nonzero / max(1, n_step), 6),
           "n_err": n_err, "err_hist": err_hist, "spec_hist": spec_hist,
           "op_hist": op_hist, "bad_examples": bad, "ok": bool(n_err == 0 and n_step > 0)}
    if logp:
        C.log("STATE_SOURCE %s rows=%d steps=%d nonzero=%d/%d rate=%.6f err=%d ok=%s"
              % (tag, len(rows), n_step, n_nonzero, n_step, rep["nonzero_rate"], n_err,
                 rep["ok"]), logp)
    return rep


def verify_against_gold(rows, logp=None, limit=None):
    """**Change-97 gate 1**: device-recomputed res vs in-row gold res, bit-for-bit (exact float equality)."""
    n = n_eq = n_ne = n_err = 0
    bad = []
    for r in rows[:limit] if limit else rows:
        for k, (res, _ex, _sp, op, err) in enumerate(dev_resid_all(r)):
            n += 1
            if err:
                n_err += 1
                if len(bad) < 8:
                    bad.append({"name": r["name"], "k": k, "op": op, "err": err})
                continue
            gold = [float(x) for x in r["steps"][k]["res"]]
            if res == gold:
                n_eq += 1
            else:
                n_ne += 1
                if len(bad) < 8:
                    bad.append({"name": r["name"], "k": k, "op": op,
                                "got": res[:3], "gold": gold[:3]})
    rep = {"schema": "gen33-device-vs-gold/1", "n_step": n, "n_eq": n_eq, "n_ne": n_ne,
           "n_err": n_err, "eq_rate": round(n_eq / max(1, n), 6),
           "bitwise_equal": bool(n_eq == n and n > 0), "bad_examples": bad}
    if logp:
        C.log("DEVICE_VS_GOLD n=%d eq=%d ne=%d err=%d rate=%.6f bitwise=%s"
              % (n, n_eq, n_ne, n_err, rep["eq_rate"], rep["bitwise_equal"]), logp)
    return rep
