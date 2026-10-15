# -*- coding: utf-8 -*-
# gen44_fam_b1.py -- family B1: goal-shape extension `g44_goal_split`
# (and/iff/imp/or decomposition plus hypothesis absorption).
#
# Semantics (device computes, no model arithmetic): the conclusion is split at
# depth 0 into a subgoal tree; every leaf joins the existing routes
# (g42_std / g42_subst); the left side of an implication is absorbed into the
# hypothesis set (rendered as intro h44); iff splits into both directions;
# or receives a decision witness from the device (deterministic: left before
# right).
# Rendering (single-render exam convention): `try constructor` /
# `try intro h44` / `try left|right` plus the leaf chain (one try per line).
# State frames: leaf-chain frames computed live (no corpus field reads);
# mechanical assertions live in bgate.payload_tests.
# Read-only dependency: gen44_bcore (shared family kernel).
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import gen44_bcore as B                                                  # noqa: E402

FAM = "b1"
SPEC = {"fam": "b1", "desc": "goal-shape: and/iff/imp/or (C_and/_iff/_imp/_or_post)",
        "ops": ["g44_goal_split"],
        "semantics": "depth-0 connective split into a subgoal tree; imp left side "
                     "absorbed into hypotheses; or decided by device witness",
        "render_exam": "try constructor / try intro h44 / try left|right + leaf chain",
        "render_gold": "same chain, leaf candidates enumerated (C1..C5)",
        "frame": "leaf-chain state frames (computed live)",
        "claim": "tree contains and/iff/or, or (imp without forall)"}

VARS = ["x", "y", "z", "a", "b", "c", "m", "n", "p", "q", "u", "v", "w", "t", "s", "r"]
REL = ["≤", "<", "="]


def claim(tree):
    kinds = set()

    def walk(n):
        kinds.add(n.get("kind"))
        for c in n.get("children") or []:
            walk(c)
    walk(tree)
    return bool(kinds & {"and", "iff", "or"}) or ("imp" in kinds and "forall" not in kinds)


def damage_ops(ops):
    """Single-slot chain damage: replace the terminal leaf operator with a
    sibling (the rendered text must change)."""
    return B.damage_leaf_op(ops)


def perturb_digit(row):
    """B1 chains are structural (tree plus operator order); digits never enter
    the compiled text, so the digit axis is not applicable (registered)."""
    return None


def _var(rng, k=1):
    out = []
    for _ in range(k):
        nm = rng.choice(VARS)
        if rng.random() < 0.35:
            nm = "%s%d" % (nm, rng.randint(2, 9))
        out.append(nm)
    return out


def corpus_row(rng, k):
    """Synthesize one row (random variable names and coefficients; contains no
    Lean-Workbook statement or title)."""
    shape = rng.choice(["and", "imp", "iff", "or", "nested"])
    name = "gen44_b1_%06d" % (700000 + k)
    x, y = _var(rng, 2)
    if x == y:
        y = y + "2"
    c1, c2, c3 = (rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]) for _ in range(3))
    if rng.random() < 0.4:
        c1 = -c1
    head = "theorem %s (%s %s : ℝ) (h : %s ≤ %s)" % (name, x, y, x, y)
    if shape == "and":
        concl = "(%s + %d ≤ %s + %d) ∧ (%d * %s ≤ %d * %s)" % (x, c1, y, c1, c2, x, c2, y)
    elif shape == "imp":
        concl = "(%s ≤ %s) → (%s + %d ≤ %s + %d)" % (x, y, x, c3, y, c3)
    elif shape == "iff":
        concl = "(%s ≤ %s) ↔ (%s - %s ≤ %d)" % (x, y, x, y, 0)
    elif shape == "or":
        concl = "(%s + %d ≤ %s + %d) ∨ (2 + 2 = 4)" % (x, c1, y, c1)
    else:
        concl = "(%s + %d ≤ %s + %d) ∧ ((%s ≤ %s) ∨ (3 = 3))" % (
            x, c1, y, c1, x, y)
    return {"name": name, "head": head, "concl": concl}


def selftest():
    st = []
    t = B.build_tree("(x + 1 ≤ y + 1) ∧ (2 * x ≤ 2 * y)", {"x": "ℝ", "y": "ℝ"},
                     [{"name": "h", "rel": "≤", "neg": False, "lhs": B.sp.Symbol("x"),
                       "rhs": B.sp.Symbol("y"), "txt": "x ≤ y", "meta": {}}], False)
    assert t["kind"] == "and", t
    assert claim(t)
    st.append("and_claim")
    t2 = B.build_tree("(x ≤ y) → (x + 3 ≤ y + 3)", {"x": "ℝ", "y": "ℝ"}, [], False)
    assert t2["kind"] == "imp" and claim(t2)
    st.append("imp_claim")
    t3 = B.build_tree("(x ≤ y) ∨ (3 = 3)", {"x": "ℝ", "y": "ℝ"}, [], False)
    assert t3["kind"] == "or" and B.certifiable(t3)
    st.append("or_witness")
    return st


if __name__ == "__main__":
    print({"fam": FAM, "selftest": selftest(), "spec": SPEC})
