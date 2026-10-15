# -*- coding: utf-8 -*-
# bh_tpl.py -- bridge-hard batch "same value, different form" declaration templates (rendering spec; the
# **rendering convention** shared by generator and checker)
#
# Note: this module carries **writing conventions only** (which equivalent shape the statement uses for
# the same mathematical object); it contains no mathematical decisions.
# All mathematical decisions (whether c1/c0/r are consistent with the statement, whether classes are
# exclusive) are **independently recomputed** inside bh_check.
TPL_IDS = ("const_pow_cancel", "const_pow_cancel_lin", "const_lin_obf", "const_neg_obf",
           "const_noise", "pow_masked", "pow_masked_reordered", "pow_plain",
           "pow_reordered")
# Main forms for bridge-hard (the two declared shapes are **lexically indistinguishable**: both contain
# '^ n' and '((n : ℤ) + 1) - (n : ℤ)')
HARD_TPL = {"const": ("const_pow_cancel", "const_pow_cancel_lin"),
            "pow": ("pow_masked", "pow_masked_reordered")}


def cancel_base(r):
    """Cancellation-term base (deterministic: derived from r; the checker can recompute it). The term
    has the form (b^n - b^n) ≡ 0."""
    a = abs(int(r))
    return a if a >= 2 else 2


def sign_of(x):
    return 1 if int(x) >= 0 else -1


def noise_base(r):
    """Noise-term base (deterministic: derived from r; the checker can recompute it)."""
    return abs(int(r)) + 1 if int(r) > 0 else 3


def render(tpl, c1, c0, r):
    c1, c0, r = int(c1), int(c0), int(r)
    pw = "(%d) ^ n" % r if r < 0 else "%d ^ n" % r
    if tpl in ("const_lin_obf", "const_neg_obf", "const_noise", "const_pow_cancel",
               "const_pow_cancel_lin"):
        if c1 != 0:
            raise ValueError("const 模板要求 c1=0 (c1=%s)" % c1)
        if tpl == "const_lin_obf":
            if c0 < 0:
                raise ValueError("const_lin_obf 要求 c0≥0")
            return "%d * ((n : \u2124) + 1) - %d * (n : \u2124)" % (c0, c0)
        if tpl == "const_neg_obf":
            if c0 > 0:
                raise ValueError("const_neg_obf 要求 c0≤0")
            return "%d * (n : \u2124) - %d * ((n : \u2124) + 1)" % (-c0, -c0)
        if tpl == "const_pow_cancel":
            b = cancel_base(r)
            return "%d + (%d ^ n - %d ^ n)" % (c0, b, b)
        if tpl == "const_pow_cancel_lin":
            b = cancel_base(r)
            return ("%d * ((n : ℤ) + 1) - %d * (n : ℤ) + (%d ^ n - %d ^ n)"
                    % (c0, c0, b, b))
        if c0 < 0:
            raise ValueError("const_noise 要求 c0>=0")
        if tpl == "const_noise":
            q = noise_base(r)
            return ("%d * ((n : \u2124) + 1) - %d * (n : \u2124) + (%d ^ 2 - %d * %d)"
                    % (c0, c0, q, q, q))
        b = cancel_base(r)
        return ("%d * ((n : \u2124) + 1) - %d * (n : \u2124) + (%d ^ n - %d ^ n)"
                % (c0, c0, b, b))
    if abs(c1) == 1:
        mag = pw
        signed = ("-(%s)" % pw) if c1 < 0 else pw
    else:
        mag = "%d * %s" % (abs(c1), pw)
        signed = ("-%d * %s" % (abs(c1), pw)) if c1 < 0 else mag
    body = mag
    FAC = "(((n : \u2124) + 1) - (n : \u2124))"
    if tpl in ("pow_masked", "pow_masked_reordered"):
        if abs(c1) == 1:
            term = ("%s * %s" % (pw, FAC)) if c1 > 0 else ("-%s * %s" % (pw, FAC))
        else:
            term = ("%d * %s * %s" % (c1, pw, FAC)) if c1 > 0 else \
                   ("-%d * %s * %s" % (-c1, pw, FAC))
        if tpl == "pow_masked":
            return term if c0 == 0 else "%d + %s" % (c0, term)
        return term if c0 == 0 else "%s + %d" % (term, c0)
    if tpl == "pow_plain":
        return signed if c0 == 0 else "%s %s %d" % (signed, "+" if c0 > 0 else "-",
                                                    abs(c0))
    if tpl == "pow_reordered":
        if c0 == 0:
            return signed
        return ("%d - %s" % (c0, body)) if c1 < 0 else ("%d + %s" % (c0, body))
    raise ValueError("未知模板 %r" % tpl)


def pick(cls, c0, r, rng):
    """Template pick for bridge-hard: both classes draw from HARD_TPL (lexically indistinguishable)."""
    return rng.choice(HARD_TPL["const" if cls == "const" else "pow"])


def pick_easy(cls, c0, r, rng):
    """Template pick for the rule-easy reference layer (lexically separable; for difficulty contrast)."""
    if cls == "const":
        return rng.choice(["const_lin_obf", "const_noise"])
    return rng.choice(["pow_plain", "pow_reordered"])
