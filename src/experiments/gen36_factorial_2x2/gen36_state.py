# -*- coding: utf-8 -*-
# gen36_state.py -- frame-state definition (V36_DESIGN §3 D2/D3): **post-step device residual**
#
# Definition (single state source; the training cache and the exam-section payload share this file so both
#   sides use the same convention bitwise):
#   frame k (k <= n-2) carries res := evals(R_out(k))   -- R_out = R_in - mult*l^2, arithmetic **verbatim**
#                                                    from gen32_ops._sos_block (device-recomputed, not copied)
#   terminal frame (k = n-1) has no successor step => keeps the parent value (that frame's res is only read
#   by the END decision; registered as a limitation)
# Rationale (node [0] probe): the terminal-step decision needs the residual after all block actions (= constant
#   rho); the parent frame carries R_in(k) (pre-step), so the decision position only sees up to R_in(n-2)
#   => one step late.
# Imports only, zero edits (gen32_poly / gen32_ops); writes only under scratch/ccp/gen36/**.
from fractions import Fraction

import gen36_common as C                                              # noqa: F401

import gen32_ops as OPS                                               # noqa: E402
import gen32_poly as P                                                # noqa: E402

TAIL_OK_KINDS = ("gen32_sos_tail_zero", "gen32_sos_tail_square", "gen32_sos_tail_other")


def r_out_of_step_args(args):
    """Device arithmetic: R_out = R_in - mult*l^2 (verbatim from gen32_ops._sos_block; returns (dump, nv, R_out))."""
    vars_ = list(args["vars"])
    nv = len(vars_)
    R_in = P.load(args["R_in"])
    lp = OPS._ell_poly(vars_, args["ell"])
    mult = Fraction(args["mult"])
    R_out = P.psub(R_in, P.pscale(P.pmul(lp, lp), mult))
    return P.dump(R_out), nv, R_out


def res_of_poly(poly, nv):
    """Exact values on the 8-D integer grid (floated; same convention as gen32_poly.resid_pack)."""
    return [float(x) for x in P.resid_vals(poly, nv)]


def post_action_res(args):
    """Frame-state candidate: the 8-D device residual **after** that step."""
    _dump, nv, R_out = r_out_of_step_args(args)
    return res_of_poly(R_out, nv), R_out, nv


def is_const(poly, nv):
    """Constant-polynomial test; constant -> Fraction, otherwise None."""
    if not poly:
        return Fraction(0)
    keys = [k for k, v in poly.items() if Fraction(v) != 0]
    if len(keys) == 0:
        return Fraction(0)
    z = (0,) * nv
    if all(k == z for k in keys):
        return Fraction(poly.get(z, 0))
    return None


def shifted_res_frames(args_list, tail_keep):
    """Whole-row frame state: [k <= n-2 -> evals(R_out(k)); k = n-1 -> tail_keep].

    -> (res_list, diag); diag registers: n_const_last (whether the post-last-block value is constant), rho_dev, errs.
    """
    n = len(args_list)
    out, diag = [], {"n": n, "last_const": None, "rho_dev": None, "errs": []}
    for k in range(n):
        if k >= n - 1:
            out.append(tail_keep)
            continue
        try:
            r8, R_out, nv = post_action_res(args_list[k])
            out.append(r8)
            if k == n - 2:
                cc = is_const(R_out, nv)
                diag["last_const"] = bool(cc is not None)
                diag["rho_dev"] = (None if cc is None else str(cc))
        except Exception as e:                                        # noqa: BLE001
            diag["errs"].append({"k": k, "err": repr(e)})
            out.append(None)
    return out, diag
