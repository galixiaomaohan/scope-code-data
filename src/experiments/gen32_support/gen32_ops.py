# -*- coding: utf-8 -*-
# gen32_ops.py -- gen32 long-chain deep operators (with certificates; every step returns the **real numeric state** resid, not a placeholder)
#
# Key difference from bh_ops (the substantive change of this batch, see DEEP_FAMILIES_DESIGN.md §2):
#   bh_ops.OpResult.resid was hard-coded to [0.0] (placeholder) => device res frames were identically zero (gen30 FINDING);
#   every operator in this module returns the **8-dim exact value of that step's primary operation object** (resid_vals);
#   and _finish is **fail-closed** on an all-zero state (RESID_ZERO_STATE) => the corpus res nonzero rate is a verifiable reading.
#
# No self-certification: all re-verification is done by gen32_check (new criteria) and bh_check (frozen criteria, for
#   dual verification); the register_checker of both modules structurally rejects same-module functions.
import os
import sys
from fractions import Fraction

import bh_check as BC                                                   # frozen criteria (read-only reuse)
import g32_common as C
import gen32_check as CK
import gen32_poly as P

PROOF_OPS = ("gen32_nest_solve", "gen32_nest_fold", "gen32_nest_delta", "gen32_nest_close",
             "gen32_sos_block", "gen32_sos_cross",
             "gen32_sos_tail_zero", "gen32_sos_tail_square", "gen32_sos_tail_other")
BANNED_OPS = ("math_evaluate_expr",)


class OpResult(object):
    __slots__ = ("ok", "err", "value", "cert", "cert_ok", "cert_detail", "tactic",
                 "resid", "resid_exact", "resid_spec")

    def __init__(self, ok, err=None, value=None, cert=None, cert_ok=None,
                 cert_detail="", tactic="", resid=None, resid_exact=None, resid_spec=""):
        self.ok, self.err, self.value = ok, err, value
        self.cert, self.cert_ok, self.cert_detail = cert, cert_ok, cert_detail
        self.tactic = tactic
        self.resid = resid or [0] * P.RES_DIM
        self.resid_exact = resid_exact or [str(x) for x in self.resid]
        self.resid_spec = resid_spec

    def to_dict(self):
        return {"ok": self.ok, "err": self.err, "value": self.value, "cert": self.cert,
                "cert_ok": self.cert_ok, "cert_detail": self.cert_detail,
                "tactic": self.tactic, "resid": self.resid,
                "resid_exact": self.resid_exact, "resid_spec": self.resid_spec}


def check_cert(kind, payload):
    """Criterion routing: gen32_* -> new criterion module; bh_* -> frozen criterion module. Returns (ok, resid, detail, mod)."""
    if kind.startswith("gen32_"):
        ok, r, d = CK.check(kind, payload)
        return ok, r, d, CK
    ok, r, d = BC.check(kind, payload)
    return ok, r, d, BC


def _resid_of(poly, nv, spec):
    """Build (resid numeric, exact strings, spec): 8-dim exact values (DEEP_FAMILIES_DESIGN §2)."""
    vals = P.resid_vals(poly, nv)
    num = []
    for v in vals:
        v = Fraction(v)
        num.append(int(v) if v.denominator == 1 else float(v))
    return num, [str(v) for v in vals], spec


def _finish(ok, err, value, cert, tactic, poly=None, nv=1, spec="", detail=""):
    """Unified close-out: independent certificate re-verification + **nonzero-state fail-closed** + resid packing."""
    if not ok:
        return OpResult(False, err=err, value=value, tactic=tactic)
    resid, resid_exact, rspec = _resid_of(poly, nv, spec)
    if not any(x != 0 for x in resid):
        return OpResult(False, err="RESID_ZERO_STATE",
                        value={"note": "state object is the zero polynomial => the whole problem is void", "spec": rspec},
                        tactic=tactic)
    kind = cert["kind"]
    cok, cr, cdet, mod = check_cert(kind, cert["payload"])
    ind = mod.__name__ not in mod.independence(kind)
    msg = (cdet if not detail else detail + " | " + cdet) + " | independent=%s" % ind
    if not (cok and ind):
        return OpResult(False, err="CERT_REJECT", value=value, cert=cert, cert_ok=False,
                        cert_detail=msg, tactic=tactic, resid=resid,
                        resid_exact=resid_exact, resid_spec=rspec)
    return OpResult(True, None, value, cert, True, msg, tactic=tactic, resid=resid,
                    resid_exact=resid_exact, resid_spec=rspec)


# ================================================================ family A/C: first-order difference equation
def solve_block(sym, closed, h0, hstep, ref=None, intro=False):
    """One-layer induction Lean 4 block (shared by families A/C; cites the lower lemma ref to form a lemma chain)."""
    lines = []
    if intro:
        lines.append("intro n")
    lines.append("have h%s : \u2200 n, %s n = %s := by" % (sym, sym, closed))
    lines.append("  intro n")
    lines.append("  induction n with")
    lines.append("  | zero => norm_num [%s]" % h0)
    lines.append("  | succ n ih =>")
    lines.append("    rw [%s, ih%s]" % (hstep, "" if not ref else ", %s n" % ref))
    lines.append("    push_cast")
    lines.append("    ring")
    return "\n".join(lines)


def op_nest_solve(args):
    """args {level, sym, h0, hstep, ref, intro, v0, deg, S}: solve ΔP = S, P(0)=v0 => this level's closed form P.
    Class assertion = the declared degree deg must == the **exact** degree of S (wrong class = real failure)."""
    nv = int(args.get("nv", 1))
    v0 = Fraction(args["v0"])
    deg = int(args["deg"])
    S = P.load(args["S"])
    real = max([m[0] for m in S] or [0])
    if real != deg:
        return OpResult(False, err="DEG_CLASS_VIOLATION",
                        value={"real_deg": real, "op_deg": deg}, tactic="")
    Pd = P.solve_difference(S, v0)
    payload = {"nv": nv, "deg": deg, "v0": str(v0), "S": P.dump(S), "P": P.dump(Pd)}
    cert = {"kind": "gen32_solve_cert", "payload": payload}
    # dual verification (for deg<=3 the frozen criterion bc_check.chk_ind_sum must also pass; reading registered)
    dual = "dual_na(deg=%d>3)" % deg
    if deg in (0, 1, 2, 3):
        ok2, _r2, d2 = BC.chk_ind_sum(payload)
        dual = "dual_bh_ind_sum_cert=%s" % ("PASS" if ok2 else "FAIL:%s" % d2[:40])
        if not ok2:
            return OpResult(False, err="DUAL_CHECK_FAIL", value={"detail": d2}, tactic="")
    closed = P.render_poly(Pd, ["n"])
    blk = solve_block(args["sym"], closed, args["h0"], args["hstep"],
                      args.get("ref"), args.get("intro", False))
    return _finish(True, None, {"closed": closed, "P": P.dump(Pd), "deg": deg,
                                "level": args.get("level")},
                   cert, tactic=blk, poly=S, nv=nv,
                   spec="S_%s" % args.get("level", "?"),
                   detail="difference_equation_solve|" + dual)


# ================================================================ family A/C: folding
def op_nest_fold(args):
    """args {level, scale, lower}: upper increment := scale · lower closed form (device construction + certificate verification)."""
    nv = int(args.get("nv", 1))
    scale = Fraction(args["scale"])
    lo = P.load(args["lower"])
    up = P.pscale(lo, scale)
    payload = {"nv": nv, "scale": str(scale), "lower": P.dump(lo), "upper": P.dump(up)}
    cert = {"kind": "gen32_fold_cert", "payload": payload}
    return _finish(True, None, {"upper": P.dump(up), "upper_text": P.render_poly(up, ["n"]),
                                "scale": str(scale), "level": args.get("level")},
                   cert, tactic="fold_scale_%s" % scale, poly=lo, nv=nv,
                   spec="G_%s" % args.get("level", "?"), detail="inner_closed_folded")


# ================================================================ family A/C: telescoping verification
def op_nest_delta(args):
    """args {level, upper, inc}: verify Δ(closed form) ≡ this level's increment (self-consistent closed chain)."""
    nv = int(args.get("nv", 1))
    up = P.load(args["upper"])
    inc = P.load(args["inc"])
    d = P.shift_sub(up)
    if P.dump(d) != P.dump(inc):
        return OpResult(False, err="DELTA_MISMATCH",
                        value={"delta": P.dump(d), "claimed": P.dump(inc)}, tactic="")
    payload = {"nv": nv, "upper": P.dump(up), "inc": P.dump(inc)}
    cert = {"kind": "gen32_delta_cert", "payload": payload}
    return _finish(True, None, {"delta": P.dump(d), "level": args.get("level")},
                   cert, tactic="delta_check", poly=up, nv=nv,
                   spec="F_%s" % args.get("level", "?"), detail="telescoping_closure")


# ================================================================ family A/C: closed-form equivalence + class exclusivity
def op_nest_close(args):
    """args {level, P, T_text (statement-declared as-is), cls, sq?}: equivalence + class exclusivity."""
    nv = int(args.get("nv", 1))
    Pd = P.load(args["P"])
    if "T" in args:
        Td = P.load(args["T"])
    else:
        try:
            Td = P.parse_poly(args["T_text"], ["n"])
        except Exception:                                           # noqa: BLE001
            return OpResult(False, err="T_NOT_POLYNOMIAL", value={"cls": args.get("cls")},
                            tactic="")
    payload = {"nv": nv, "P": P.dump(Pd), "T": P.dump(Td), "cls": args["cls"]}
    if args["cls"] in ("sq2", "sq4"):
        if "sq" not in args:
            return OpResult(False, err="MISSING_SQ_ROOT", tactic="")
        payload["sq"] = [str(x) for x in args["sq"]]
    cert = {"kind": "gen32_close_cert", "payload": payload}
    lines = ["rw [%s]" % args.get("close_lemma", "hclose")]
    if args.get("need_ring", True):
        lines.append("ring")
    return _finish(True, None, {"cls": args["cls"], "T": P.dump(Td),
                                "level": args.get("level")},
                   cert, tactic="\n".join(lines), poly=Td, nv=nv,
                   spec="T_decl_%s" % args.get("level", "?"), detail="closed_form_equiv")


# ================================================================ family B: square-block absorption
def _ell_poly(vars_, ell):
    nv = len(vars_)
    idx = {v: i for i, v in enumerate(vars_)}
    out = {}
    for v, cf in ell["coefs"].items():
        if Fraction(cf) == 0:
            continue
        m = [0] * nv
        m[idx[v]] = 1
        out[tuple(m)] = Fraction(cf)
    sh = Fraction(ell.get("shift", 0))
    if sh != 0:
        out[(0,) * nv] = out.get((0,) * nv, Fraction(0)) - sh
    return out


def ell_text(ell):
    parts = []
    for v, cf in ell["coefs"].items():
        cf = Fraction(cf)
        if cf == 0:
            continue
        s = v if abs(cf) == 1 else "%d * %s" % (int(abs(cf)), v)
        parts.append(("+" if cf > 0 else "-", s))
    sh = Fraction(ell.get("shift", 0))
    if sh != 0 or not parts:
        parts.append(("-" if sh > 0 else "+", str(int(abs(sh)))))
    out = ""
    for i, (sg, b) in enumerate(parts):
        if i == 0:
            out = ("- " if sg == "-" else "") + b
        else:
            out += (" - " if sg == "-" else " + ") + b
    return "(%s)" % out


def _sos_block(args, name):
    """args {vars, R_in(dump), ell:{coefs,shift}, mult, hint_idx}. Device constructs R_out = R_in − mult·ℓ²."""
    vars_ = list(args["vars"])
    nv = len(vars_)
    R_in = P.load(args["R_in"])
    ell = args["ell"]
    supp = [v for v, cf in ell["coefs"].items() if Fraction(cf) != 0]
    if name == "gen32_sos_cross" and len(supp) < 2:
        return OpResult(False, err="CROSS_NEEDS_TWO_VARS", value={"supp": supp}, tactic="")
    if name == "gen32_sos_block" and len(supp) != 1:
        return OpResult(False, err="BLOCK_NEEDS_ONE_VAR", value={"supp": supp}, tactic="")
    lp = _ell_poly(vars_, ell)
    mult = Fraction(args["mult"])
    R_out = P.psub(R_in, P.pscale(P.pmul(lp, lp), mult))
    payload = {"vars": vars_, "R_in": P.dump(R_in), "R_out": P.dump(R_out),
               "ell": P.dump(lp), "mult": str(mult)}
    cert = {"kind": "gen32_block_cert", "payload": payload}
    hint = "%d * %s ^ 2" % (int(mult), ell_text(ell))
    tact = "have h%d : (0:\u211a) \u2264 %s := by positivity" % (int(args["hint_idx"]), hint)
    valo = {"R_out": P.dump(R_out), "R_out_text": P.render_poly(R_out, vars_),
            "ell": ell, "ell_text": ell_text(ell), "mult": str(mult), "hint": hint,
            "absorbed": P.dump(P.pscale(P.pmul(lp, lp), mult))}
    return _finish(True, None, valo, cert, tactic=tact, poly=R_in, nv=nv,
                   spec="R_in_round%d" % int(args["hint_idx"]), detail="sos_block_absorb")


def op_sos_block(args):
    return _sos_block(args, "gen32_sos_block")


def op_sos_cross(args):
    return _sos_block(args, "gen32_sos_cross")


# ================================================================ family B: closing semantic class
def _sos_tail(args, cls):
    """args {vars, rho, hints, goal_dom(dump), hints_poly, sq?, hint_idx}: closed chain + class exclusivity."""
    vars_ = list(args["vars"])
    nv = len(vars_)
    rho = Fraction(args["rho"])
    payload = {"rho": str(rho), "cls": cls, "vars": vars_,
               "goal_dom": dict(args["goal_dom"]),
               "hints_poly": [dict(h) for h in args["hints_poly"]]}
    if cls == "square":
        payload["sq"] = int(args["sq"])
    cert = {"kind": "gen32_tail_cert", "payload": payload}
    hs = list(args["hints"])
    if cls == "zero":
        body = ["nlinarith only [%s]" % ", ".join(hs)]
    elif cls == "square":
        s = int(args["sq"])
        body = ["have hrho : (0:\u211a) \u2264 (%d : \u211a) ^ 2 := sq_nonneg (%d : \u211a)"
                % (s, s), "nlinarith only [%s, hrho]" % ", ".join(hs)]
    else:
        body = ["have hrho : (0:\u211a) \u2264 (%s : \u211a) := by norm_num" % rho,
                "nlinarith only [%s, hrho]" % ", ".join(hs)]
    return _finish(True, None, {"rho": str(rho), "cls": cls}, cert, tactic="\n".join(body),
                   poly=P.load(args["goal_dom"]), nv=nv, spec="goal_dom",
                   detail="tail_class_%s" % cls)


def op_sos_tail_zero(args):
    return _sos_tail(args, "zero")


def op_sos_tail_square(args):
    return _sos_tail(args, "square")


def op_sos_tail_other(args):
    return _sos_tail(args, "other")


OPS = {"gen32_nest_solve": op_nest_solve, "gen32_nest_fold": op_nest_fold,
       "gen32_nest_delta": op_nest_delta, "gen32_nest_close": op_nest_close,
       "gen32_sos_block": op_sos_block, "gen32_sos_cross": op_sos_cross,
       "gen32_sos_tail_zero": op_sos_tail_zero,
       "gen32_sos_tail_square": op_sos_tail_square,
       "gen32_sos_tail_other": op_sos_tail_other}


def call(op, args):
    fn = OPS.get(op)
    if fn is None:
        return OpResult(False, err="OP_NOT_REGISTERED", tactic="")
    if op in BANNED_OPS:
        return OpResult(False, err="BANNED_OP", tactic="")
    return fn(args)


def whitelist_check(ops_used):
    bad = [o for o in ops_used if o not in PROOF_OPS]
    banned = [o for o in ops_used if o in BANNED_OPS]
    return (not bad) and (not banned), {"non_whitelist": bad, "banned": banned}


if __name__ == "__main__":
    import json
    st = CK.selftest()
    n_neg = [k for k in st if k.endswith(("_neg", "perturbed", "wrongdeg", "wrongroot",
                                          "on_square", "on_quartic", "zero_scale",
                                          "missing_block"))]
    n_pos = [k for k in st if k not in n_neg]
    print(json.dumps({"selftest": st,
                      "pos_pass": sum(1 for k in n_pos if st[k]),
                      "n_pos": len(n_pos),
                      "neg_all_false": all(not st[k] for k in n_neg),
                      "n_neg": len(n_neg)}, ensure_ascii=False, indent=1))
