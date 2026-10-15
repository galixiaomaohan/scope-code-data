# -*- coding: utf-8 -*-
"""gen_skeleton.py -- V32 skeleton-flavor frame-level data generator (deterministic; K2
deliverable).

Usage (the script resolves paths itself):
    P="${SCOPE_PYTHON:-python3}"
    PYTHONPATH="${SCOPE_SUPEROPERATOR:-../device}" $P -m svokernel_v32.gen_skeleton \\
        --seed 20260922 --out data/sla_v32_skeleton.jsonl
    ... $P -m svokernel_v32.gen_skeleton --audit-vals          # value-reachability spot check only (no write)
    ... $P -m svokernel_v32.gen_skeleton --digest-check a.jsonl b.jsonl

Three hard properties:
  1. **Really executed**: every step of every row is a **real call** to a kernel operator (all six
     gates pass: schema -> precond -> budget -> invoke -> output schema -> independent verifier ->
     independent certificate re-check); any step not ok fails the whole row -- there are no
     "imagined calls" in the data;
  2. **Deterministic**: no unseeded randomness / no wall clock / no hash iteration order anywhere in
     the flow; two runs must produce the same digest (--digest-check);
  3. **Value-reachable** (design doc §2.1, lesson from v31f's 11.15%): the gold value of every
     **state slot** must appear literally in the problem-text tokens (or be derivable from the
     fresh list declared in the text); the spot-check reading = literal coverage of state slots,
     required >= 0.90.
     Only the **action-choice slot** (`act`: apply/split/rewrite) is exempt -- it **is the decision
     to be predicted**, so by definition it cannot appear in the text beforehand; the two categories
     are counted separately, never mixed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                      # superoperator/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
RESEARCH = os.path.join(os.path.dirname(_ROOT), "research")

from svokernel_v32 import V32REGISTRY, V32_SCHEMA, v32_op_index_map   # noqa: E402
from .audit_data import _iter_programs, nontriviality_report, operator_coverage  # noqa: E402
from svokernel_v32 import skelmath as sm                              # noqa: E402

from svokernel.audit import AuditTrace, BudgetLedger                  # noqa: E402
from svokernel.tool import CallCtl                                     # noqa: E402

DEFAULT_OUT = os.path.join(RESEARCH, "data", "sla_v32_skeleton.jsonl")

#: Action-choice slot (the decision to be predicted; excluded from the "value-reachable"
#: denominator, see the module docstring)
CHOICE_SLOTS = {"act"}

#: Named-rendering aliases: slot -> field name in the problem text (strict check f"{alias}={value}";
#: fresh slots are special-cased)
SLOT_ALIAS = {"goal": "goal", "m": "m", "vid": "vid", "gid": "gid", "n0": "n0",
              "stk": "stk", "tk": "goal", "ledger": "ledger"}

#: Node kind -> the operator code that creates it (0=root/implicit, 1=case_split, 2=induction,
#: 3=subgoal_focus, 4=leaf_close, 5=contradiction_close); used by the K_GSTK channel (see
#: v32lib.CREATOR_CODE)
NODE_CODE = {"root": 0, "claim": 3, "leaf": 1, "neg": 0, "ind-base": 2, "ind-step": 2}

#: Three problem-text styles (deterministic; only the shell changes, never the mathematical content)
STYLE_HEAD = ("V32SKEL", "PROOF-STATE", "SKEL")

#: Presentation variants: factored ("n*(n+1)") vs expanded ("n^2 + n") -- parsed to the same
#: polynomial
PRES = ("fac", "exp")


# ---------------------------------------------------------------------------
# catalog enumeration (deterministic): instance = (fid, m, vid, pres, style, witness)
# ---------------------------------------------------------------------------

def negation_catalog(max_m=11, ds=(2, 3, 4)):
    """Systematically enumerate congruence **unsolvable** goals: for n^d - c (mod m), c not a d-th
    residue => no integer solution.

    Every entry is really verified at import time with the complete-residue decision (unsolvable =>
    no 0 among the residues).
    """
    out = []
    for d in ds:
        for m in range(3, max_m + 1):
            powers = {pow(r, d, m) for r in range(m)}
            for c in range(1, m):
                if c in powers:
                    continue
                c_ok = not any((pow(r, d, m) - c) % m == 0 for r in range(m))
                if c_ok:
                    out.append((d, m, c))
    return tuple(out)


def _poly_text(coeffs, var, pres):
    """Render the expression per presentation variant (fac: the factored form comes from gen; exp:
    canonical expansion)."""
    return sm.poly_str(coeffs, var) if pres == "exp" else None


def enumerate_instances(seed: int, vars_n=5, styles=(0, 1, 2), witnesses=(1, 3, 5, 7),
                        max_rows=6000):
    """Deterministic instance table: every instance = one complete skeleton proof (one data
    row)."""
    inst = []
    styles = tuple(styles)

    def add(fid, m, kind, factor_f, var, vid, pres, style, w):
        inst.append({"fid": fid, "m": m, "kind": kind, "expr": factor_f, "var": var,
                     "vid": vid, "pres": pres, "style": style, "w": w})

    for fam in sm.FAMILY_CATALOG:
        if fam.kind in ("conj", "neg"):
            continue                      # the conj/neg families are assembled **in pairs** below
            #                                 (the non-triviality gate requires it)
        for m in sm.VALID_MODULI[fam.fid]:
            if fam.kind == "case" and m > 6:
                continue                 # branch count = program length; m<=6 => at most 13 steps (both the 25-frame budget and the L budget bind)
            c = sm.parse_poly(fam.factored, "n")
            for vi in range(vars_n):
                var = sm.VAR_NAMES[vi]
                for pres in PRES:
                    # the family catalog fixes its expression under the variable "n"; the instance
                    # variable name may vary => rename through the same tokenizer as the parser
                    expr = (sm.rename_text(fam.factored, "n", var) if pres == "fac"
                            else sm.poly_str(c, var))
                    for st in styles:
                        ws = witnesses if fam.kind != "neg" else (0,)
                        for w in ws:
                            add(fam.fid, m, fam.kind, expr, var, vi, pres, st, w)
    # Negated family (the non-triviality gate requires >=2 non-trivial steps) => assemble as the
    # conjunction of two non-existence claims:
    #   pairing = combinations of negated goals from the same family catalog (or with the
    #   systematically enumerated n^d-c), so both claims hold individually.
    neg_items = []                                   # (fid, m, expr_in_n, tag)
    for fam in sm.FAMILY_CATALOG:
        if fam.kind != "neg":
            continue
        for mm in sm.VALID_MODULI[fam.fid]:
            neg_items.append((fam.fid, mm, fam.factored))
    cat = negation_catalog()
    pick = cat[::max(1, len(cat) // 12)][:12]
    for idx, (dd, mm, cc) in enumerate(pick):
        neg_items.append((100 + idx, mm, "n^%d - %d" % (dd, cc)))
    pairs = [(neg_items[i], neg_items[j])
             for i in range(len(neg_items)) for j in range(i + 1, len(neg_items))
             if neg_items[i][1] != neg_items[j][1] or neg_items[i][2] != neg_items[j][2]]
    pairs = pairs[::4]           # distribution control: take only a deterministic subset of the neg family so subgoal_focus does not dominate the table's first steps
    for (fa, ma, ea), (fb, mb, eb) in pairs:
        for vi in range(vars_n):
            var = sm.VAR_NAMES[vi]
            for pres in PRES:
                if pres == "fac":
                    e1, e2 = sm.rename_text(ea, "n", var), sm.rename_text(eb, "n", var)
                else:
                    e1 = sm.poly_str(sm.parse_poly(ea, "n"), var)
                    e2 = sm.poly_str(sm.parse_poly(eb, "n"), var)
                for st in styles:
                    inst.append({"fid": fa, "m": ma, "kind": "neg", "expr": e1,
                                 "var": var, "vid": vi, "pres": pres, "style": st, "w": 0,
                                 "pair": (e2, mb)})
    # Conjunction family: pair up case families (m <= 3, to control program length); each subgoal is
    # a case goal
    pairs = [(fid, m) for fid in (0, 1, 2, 3) for m in sm.VALID_MODULI[fid] if m <= 2]
    pairs = sorted(set(pairs))
    conj_pairs = [(a, b) for i, a in enumerate(pairs) for b in pairs[i + 1:]]
    for (fa, ma), (fb, mb) in conj_pairs:
        for vi in range(vars_n):
            var = sm.VAR_NAMES[vi]
            for pres in PRES:
                for st in styles:
                    for w in (witnesses[0], witnesses[2], witnesses[3]):
                        inst.append({"fid": -1, "m": ma, "kind": "conj", "expr": "",
                                     "var": var, "vid": vi, "pres": pres, "style": st, "w": w,
                                     "pair": (fa, ma, fb, mb)})
    # deterministic reshuffle (fixed seed) + truncation -- two runs see the same order
    order = list(range(len(inst)))
    random.Random(seed).shuffle(order)
    inst = [inst[i] for i in order]
    return inst[:max_rows]


def _instance_id(x: dict) -> str:
    if x["kind"] == "conj":
        fa, ma, fb, mb = x["pair"]
        tag = "conj_%d/%d_%d/%d" % (fa, ma, fb, mb)
    elif x["kind"] == "neg":              # negated conjunction: two non-existence claims
        tag = "negconj_m%d_%s__m%d_%s" % (x["m"], x["expr"].replace("*", "x").replace("^", "p"),
                                          x["pair"][1],
                                          x["pair"][0].replace("*", "x").replace("^", "p"))
    else:
        tag = "%s_m%d" % (sm.family_of(x["fid"]).name, x["m"])
    return "%s_v%d_%s_s%d_w%d" % (tag, x["vid"], x["pres"], x["style"], x["w"])


# ---------------------------------------------------------------------------
# proof driver: goal stack + real calls (every step passes the six gates)
# ---------------------------------------------------------------------------

class Driver:
    def __init__(self, iid: str, style: int):
        self.iid, self.style = iid, style
        self.nodes: list = []
        self.focus = -1
        self.depth = 0
        self.nxt = 0
        self.steps: list = []          # [(op, args, value)]
        self.ctx: list = []            # problem-text state string of each step
        self.ctx_args: list = []       # (K_GSTK, K_BPTR) payload of each step (dumped explicitly for cache loading)

    # -- stack maintenance --------------------------------------------------
    def by(self, g):
        for n in self.nodes:
            if n["gid"] == g:
                return n
        raise KeyError(g)

    def alloc(self, kind, goal, parent=-1):
        g = self.nxt
        self.nxt += 1
        self.nodes.append({"gid": g, "kind": kind, "status": "open", "parent": parent,
                           "goal": goal})
        return g

    def close(self, g):
        self.by(g)["status"] = "closed"

    def n_open(self):
        return sum(1 for n in self.nodes if n["status"] == "open")

    def stk(self):
        return sm.stack_str(self.nodes, self.focus, self.depth)

    def ledger(self):
        return sm.ledger_str([(n["gid"], n["kind"], n["status"]) for n in self.nodes])

    # -- problem-text state string ------------------------------------------
    def state_text(self, goal, m, vid, gid, n0=0, with_fresh=2, ledger=False, with_stk=False,
                   with_iid=False):
        """Problem-text state string (the concrete form of value reachability).

        `with_stk` is enabled only when the **next operator really consumes the stk argument** --
        the state string enters the token stream frame by frame, and a redundant machine stack
        string would burn the context (L budget). The switch is set by the caller per operator
        signature, and cross-checked by `audit_values` (if it should be there and is not, the spot
        check reports a miss immediately).
        """
        head = STYLE_HEAD[self.style % len(STYLE_HEAD)]
        fresh = ",".join(str(self.nxt + i) for i in range(with_fresh))
        gtxt = goal if goal else "(none: all goals closed)"
        s = ("%s%s | d%d o%d f%d nxt=%d m=%d vid=%d gid=%d n0=%d fresh=%s | goal: %s"
             % (head, (" " + self.iid) if with_iid else "", self.depth, self.n_open(),
                self.focus, self.nxt, m, vid, gid, n0, fresh, gtxt))
        if with_stk:
            s += " | stk: %s" % self.stk()
        if ledger:
            s += " | ledger: %s" % self.ledger()
        return s

    # -- real calls ---------------------------------------------------------
    def call(self, op, args, state_text):
        ctl = CallCtl(trace=AuditTrace("skel:%s" % op), budget=BudgetLedger(total_units=1e9),
                      timeout_s=60.0)
        r = V32REGISTRY.tool(op).call(dict(args), ctl)
        if not r.ok:
            raise AssertionError("骨架步骤失败: %s args=%r err=%s notes=%s"
                                 % (op, args, r.err, r.notes))
        self.steps.append((op, dict(args), r.value))
        self.ctx.append(state_text)
        code = 0
        if self.focus >= 0:
            for n in self.nodes:
                if n["gid"] == self.focus:
                    code = NODE_CODE[n["kind"]]
        self.ctx_args.append([((code & 7) << 8) | (self.depth & 0xFF),
                              ((min(self.n_open(), 7) & 7) << 8) | (max(self.focus, 0) & 0xFF)])
        return r.value


# ---------------------------------------------------------------------------
# complete proof programs for the four families
# ---------------------------------------------------------------------------

def _conj_texts(fa, ma, fb, mb, var, pres="fac"):
    """Conjunction goal text + the two claim texts (the same source as
    skelmath.conjunction_goal_text).

    pres="fac" uses factored forms, pres="exp" canonical expansions -- both parse to the same
    polynomial, only the surface differs.
    """
    if pres == "exp":
        ea = sm.poly_str(sm.parse_poly(sm.family_of(fa).factored, "n"), var)
        eb = sm.poly_str(sm.parse_poly(sm.family_of(fb).factored, "n"), var)
    else:
        ea = sm.rename_text(sm.family_of(fa).factored, "n", var)
        eb = sm.rename_text(sm.family_of(fb).factored, "n", var)
    return (sm.conjunction_goal_text(var, [(ma, ea), (mb, eb)]),
            sm.universal_goal_text(ma, ea, var), sm.universal_goal_text(mb, eb, var))


def prove_case(d: Driver, fid, m, expr, var, vid, w):
    """Case split: case_split -> (leaf_close -> apply -> leaf_close)* -> qed."""
    c = sm.parse_poly(expr, var)
    root = sm.universal_goal_text(m, expr, var)
    d.alloc("root", root)
    d.focus = 0                          # the root node is the initial focus (consistent with focus = current goal to prove)
    t = d.state_text(root, m, vid, 0, with_iid=True)
    v = d.call("skel_case_split", {"goal": root, "m": m, "vid": vid, "gid": 0, "c0": d.nxt}, t)
    d.close(0)
    kids = []
    for i in range(v["ncase"]):
        kids.append(d.alloc("leaf", v["cases_goal"][i], parent=0))
    d.depth, d.focus = 1, kids[0]
    vk = sm.VAR_ID.get(sm.parse_goal(v["cases_goal"][0]).var, 2)
    for i, g in enumerate(kids):
        if i > 0:
            t = d.state_text(v["cases_goal"][i], m, vk, g, with_fresh=0, with_stk=True)
            d.call("skel_subgoal_focus", {"act": "apply", "gid": g, "vid": vk, "tk":
                                          v["cases_goal"][i], "stk": d.stk(), "c1": 0, "c2": 0}, t)
            d.focus = g
        t = d.state_text(v["cases_goal"][i], m, vk, g, with_fresh=0, with_stk=True)
        d.call("skel_leaf_close", {"goal": v["cases_goal"][i], "m": m, "vid": vk, "gid": g}, t)
        d.close(g)
    d.focus = -1
    t = d.state_text("", 0, 0, 0, with_fresh=0, ledger=True, with_stk=True)
    return d.call("skel_qed_check", {"stk": d.stk(), "ledger": d.ledger(), "gid": 0}, t), root


def prove_induction(d: Driver, fid, m, expr, var, vid, w):
    """Induction: induction_scheme (base + step) -> close the base -> apply the step -> close the
    step -> qed."""
    root = sm.universal_goal_text(m, expr, var)
    d.alloc("root", root)
    d.focus = 0                          # the root node is the initial focus (consistent with focus = current goal to prove)
    n0 = 0 if w % 3 else 1
    t = d.state_text(root, m, vid, 0, n0=n0, with_iid=True)
    v = d.call("skel_induction_scheme", {"goal": root, "m": m, "vid": vid, "gid": 0, "n0": n0,
                                        "c0": d.nxt}, t)
    d.close(0)
    gb = d.alloc("ind-base", v["base_goal"], parent=0)
    gs = d.alloc("ind-step", v["step_goal"], parent=0)
    d.depth, d.focus = 1, gb
    vjb = sm.VAR_ID.get(sm.parse_goal(v["base_goal"]).var, 2)
    t = d.state_text(v["base_goal"], m, vjb, gb, with_fresh=0, with_stk=True)
    d.call("skel_leaf_close", {"goal": v["base_goal"], "m": m, "vid": vjb, "gid": gb}, t)
    d.close(gb)
    vks = sm.VAR_ID.get(sm.parse_goal(v["step_goal"]).var, 2)
    t = d.state_text(v["step_goal"], m, vks, gs, with_fresh=0, with_stk=True)
    d.call("skel_subgoal_focus", {"act": "apply", "gid": gs, "vid": vks, "tk": v["step_goal"],
                                  "stk": d.stk(), "c1": 0, "c2": 0}, t)
    d.focus = gs
    t = d.state_text(v["step_goal"], m, vks, gs, with_fresh=0, with_stk=True)
    d.call("skel_leaf_close", {"goal": v["step_goal"], "m": m, "vid": vks, "gid": gs}, t)
    d.close(gs)
    d.focus = -1
    t = d.state_text("", 0, 0, 0, with_fresh=0, ledger=True, with_stk=True)
    return d.call("skel_qed_check", {"stk": d.stk(), "ledger": d.ledger(), "gid": 0}, t), root


def prove_negation(d: Driver, fid, m, expr, var, vid, w, pair=None):
    """Proof by contradiction (negated-conjunction form): subgoal_focus(split) -> the two
    non-existence claims are each closed by contradiction_close -> qed.

    Why not a single negated goal: the **non-triviality gate** (audit_data.NONTRIVIAL_GATE)
    requires a proof chain of >= 2 non-trivial steps; the natural proof of a single negated goal has
    only 1 step and is rejected by the gate (the generator records it in the gate's rejected list).
    This form is mathematically just as real: the conjunction of two independent **unsolvable
    congruence** propositions, each refuted with complete residues.
    """
    e2, m2 = pair
    ra, rb = sm.negation_goal_text(m, expr, var), sm.negation_goal_text(m2, e2, var)
    root = sm.conjunction_neg_goal_text(var, [(m, expr), (m2, e2)])
    d.alloc("claim", root)
    d.focus = 0
    t = d.state_text(root, m, vid, 0, with_stk=True, with_iid=True)
    ga, gb = d.nxt, d.nxt + 1
    d.call("skel_subgoal_focus", {"act": "split", "gid": 0, "vid": vid, "tk": root,
                                  "stk": d.stk(), "c1": ga, "c2": gb}, t)
    d.close(0)
    d.alloc("neg", ra, parent=0)
    d.alloc("neg", rb, parent=0)
    d.depth, d.focus = 1, ga
    for gid, gg, mm in ((ga, ra, m), (gb, rb, m2)):
        if gid == gb:
            t = d.state_text(gg, mm, vid, gid, with_fresh=0, with_stk=True)
            d.call("skel_subgoal_focus", {"act": "apply", "gid": gid, "vid": vid, "tk": gg,
                                          "stk": d.stk(), "c1": 0, "c2": 0}, t)
            d.focus = gid
        t = d.state_text(gg, mm, vid, gid, with_fresh=0)
        d.call("skel_contradiction_close", {"goal": gg, "m": mm, "vid": vid, "gid": gid}, t)
        d.close(gid)
    d.focus = -1
    t = d.state_text("", 0, 0, 0, with_fresh=0, ledger=True, with_stk=True)
    return d.call("skel_qed_check", {"stk": d.stk(), "ledger": d.ledger(), "gid": 0}, t), root


def prove_conj(d: Driver, fa, ma, fb, mb, var, vid, w, pres="fac"):
    """Conjunction: subgoal_focus(split) -> two case subtrees -> qed."""
    conj, ca, cb = _conj_texts(fa, ma, fb, mb, var, pres)
    d.alloc("claim", conj)
    d.focus = 0                          # split's precond requires the focus to be on the conjunction node
    t = d.state_text(conj, ma, vid, 0, with_stk=True, with_iid=True)
    d.call("skel_subgoal_focus", {"act": "split", "gid": 0, "vid": vid, "tk": conj,
                                 "stk": d.stk(), "c1": d.nxt, "c2": d.nxt + 1}, t)
    ga, gb = d.alloc("claim", ca, parent=0), d.alloc("claim", cb, parent=0)
    d.close(0)
    d.depth, d.focus = 1, ga
    for gid, ggoal, mm in ((ga, ca, ma), (gb, cb, mb)):
        if gid == gb:
            t = d.state_text(ggoal, mm, vid, gid, with_fresh=0, with_stk=True)
            d.call("skel_subgoal_focus", {"act": "apply", "gid": gid, "vid": vid, "tk": ggoal,
                                          "stk": d.stk(), "c1": 0, "c2": 0}, t)
            d.focus = gid
        t = d.state_text(ggoal, mm, vid, gid)
        v = d.call("skel_case_split", {"goal": ggoal, "m": mm, "vid": vid, "gid": gid,
                                      "c0": d.nxt}, t)
        d.close(gid)
        kids = [d.alloc("leaf", v["cases_goal"][i], parent=gid) for i in range(v["ncase"])]
        d.depth, d.focus = d.depth + 1, kids[0]
        vk = sm.VAR_ID.get(sm.parse_goal(v["cases_goal"][0]).var, 2)
        for i, g in enumerate(kids):
            if i > 0:
                t = d.state_text(v["cases_goal"][i], mm, vk, g, with_fresh=0, with_stk=True)
                d.call("skel_subgoal_focus", {"act": "apply", "gid": g, "vid": vk,
                                              "tk": v["cases_goal"][i], "stk": d.stk(),
                                              "c1": 0, "c2": 0}, t)
                d.focus = g
            t = d.state_text(v["cases_goal"][i], mm, vk, g, with_fresh=0)
            d.call("skel_leaf_close", {"goal": v["cases_goal"][i], "m": mm, "vid": vk,
                                       "gid": g}, t)
            d.close(g)
        d.depth = 1
    d.focus = -1
    t = d.state_text("", 0, 0, 0, with_fresh=0, ledger=True, with_stk=True)
    return d.call("skel_qed_check", {"stk": d.stk(), "ledger": d.ledger(), "gid": 0}, t), conj


# ---------------------------------------------------------------------------
# value-reachability spot check (the core reading of K2)
# ---------------------------------------------------------------------------

def audit_values(steps, ctxs):
    """Returns {n_state, n_lit_state, n_choice, n_lit_choice, misses:[...]}.

    State slots: the gold value of every slot must appear literally in **this step's problem text**
    (fresh slots are recognized via the fresh list declared in the text); the choice slot (act) is
    counted separately and excluded from the reachability denominator (it is the decision to be
    predicted).
    """
    n_s, k_s, n_c, k_c, miss = 0, 0, 0, 0, []
    for (op, args, _val), txt in zip(steps, ctxs):
        sch = V32REGISTRY.tool(op).op.in_schema
        req = sch.get("required", [])
        m = sm.parse_goal(txt.split("| goal: ")[1].split(" | stk:")[0]) \
            if False else None
        fresh = []
        if "fresh=" in txt:
            seg = txt.split("fresh=")[1].split(" |")[0]
            fresh = [int(x) for x in seg.split(",") if x.strip().lstrip("-").isdigit()]
        for slot in req:
            v = args.get(slot)
            if slot in CHOICE_SLOTS:
                n_c += 1
                k_c += 1 if str(v) in txt else 0
                continue
            if slot in ("c1", "c2") and args.get("act") != "split":
                continue                          # unused slot (placeholder of apply/rewrite) is not counted
            n_s += 1
            lit = False
            if slot in ("c0", "c1", "c2"):
                lit = int(v) in fresh             # child ids must be in the fresh list declared in the text
            elif slot in SLOT_ALIAS:
                a = SLOT_ALIAS[slot]
                lit = str(v) in txt and (("%s=%s" % (a, v)) in txt
                                         or ("%s: %s" % (a, v)) in txt or a == "goal")
            else:
                lit = str(v) in txt
            k_s += 1 if lit else 0
            if not lit:
                miss.append({"op": op, "slot": slot, "value": str(v), "txt": txt[:200]})
    return {"n_state": n_s, "n_lit_state": k_s, "n_choice": n_c, "n_lit_choice": k_c,
            "misses": miss[:12]}


# ---------------------------------------------------------------------------
# row construction (v31 frame compatible + v32 versioned extension block)
# ---------------------------------------------------------------------------

def _held_split(iid: str, fid, m, seed: int) -> str:
    """Three-way split: held_family (the whole (family, modulus) template stays out of training) /
    held / train -- bucketed by fingerprint."""
    if (fid, m) in HELD_TEMPLATES:
        return "held_family"
    h = int(hashlib.sha256(("%d:%s" % (seed, iid)).encode("utf-8")).hexdigest()[:8], 16)
    return "held" if h % 20 == 7 else "train"


#: **Family-level held-out**: these (fid, modulus) templates stay entirely out of training (tests
#: strategy selection on unseen goal shapes)
HELD_TEMPLATES = ((2, 12), (6, 7), (8, 4))


def build_row(x: dict, seed: int, sd) -> dict:
    iid = _instance_id(x)
    d = Driver(iid, x["style"])
    if x["kind"] == "case":
        val, root = prove_case(d, x["fid"], x["m"], x["expr"], x["var"], x["vid"], x["w"])
    elif x["kind"] == "ind":
        val, root = prove_induction(d, x["fid"], x["m"], x["expr"], x["var"], x["vid"], x["w"])
    elif x["kind"] == "neg":
        val, root = prove_negation(d, x["fid"], x["m"], x["expr"], x["var"], x["vid"], x["w"],
                                   pair=x["pair"])
    else:
        fa, ma, fb, mb = x["pair"]
        val, root = prove_conj(d, fa, ma, fb, mb, x["var"], x["vid"], x["w"], x["pres"])
    steps = [{"op": op, "ok": True, "err": None, "res": _res_of(v), "state": "ok"}
             for (op, _a, v) in d.steps]
    sf_args = [a for (_op, a, _v) in d.steps]
    aud = audit_values(d.steps, d.ctx)
    inject = None
    wrong = {"case": "skel_induction_scheme", "ind": "skel_case_split",
             "neg": "skel_case_split", "conj": "skel_case_split"}[x["kind"]]
    h = int(hashlib.sha256(iid.encode("utf-8")).hexdigest()[:8], 16)
    if h % 8 == 0:                       # ~12.5% failure injection (C8-stage material)
        inject = {"fail_op": wrong, "after": 1, "kind": "v32_fail_inject",
                  "fail_state_text": d.ctx[0]}
    row = {
        "family": "skel",
        "feats": sd._feat("synth", [x["m"], x["vid"], len(steps), d.nxt, max(0, x["fid"]), x["w"],
                                    1.0 if x["kind"] == "conj" else 0.0, 1.0 if inject else 0.0]),
        "type4": sd._type4("synth"),
        "program": [s["op"] for s in steps],
        "steps": steps,
        "expected_final": val,
        "accepted": True,
        "sf_key": None,
        "episode": inject,
        "stmt": d.ctx[0],
        "bridge_key": steps[0]["op"],
        "sf_args": sf_args,
        "v32": {
            "schema": V32_SCHEMA,
            "instance_id": iid,
            "family_kind": x["kind"],
            "fid": x["fid"],
            "modulus": x["m"],
            "var": x["var"],
            "vid": x["vid"],
            "presentation": x["pres"],
            "style": x["style"],
            "witness": x["w"],
            "n_steps": len(steps),
            "n_nodes": d.nxt,
            "root_goal": root,
            "stack_final": d.stk(),
            "ledger_final": d.ledger(),
            "state_texts": d.ctx,
            "ctx_args": d.ctx_args,
            "ctx_pts": [[s[1].get("gid", 0), s[2].get("n_open", 0) if isinstance(s[2], dict) else 0]
                        for s in d.steps],
            "value_access": {"n_state": aud["n_state"], "n_literal_state": aud["n_lit_state"],
                             "cover": (aud["n_lit_state"] / aud["n_state"]) if aud["n_state"] else None,
                             "n_choice": aud["n_choice"], "n_literal_choice": aud["n_lit_choice"]},
            "heldout_split": _held_split(iid, x["fid"], x["m"], seed),
        },
    }
    return row


def _res_of(v):
    """Value -> the 8-dim residual domain (same format as v31 data; the skeleton step's "numeric
    residual" is the 0/1 closure indicator)."""
    r = [0.0] * 8
    if isinstance(v, dict):
        flag = v.get("closed", v.get("qed", v.get("exhaustive", v.get("diff_ok"))))
        r[0] = 0.0 if flag else 1.0
        for i, k in enumerate(("ncase", "n_open", "n_nodes", "focus", "depth", "m", "gid", "n0")):
            if i + 1 >= 8:
                break
            x = v.get(k)
            r[i + 1] = float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else 0.0
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--max-rows", type=int, default=6000)
    ap.add_argument("--audit-vals", action="store_true", help="value-reachability spot check only; no write")
    ap.add_argument("--digest-check", nargs=2, default=None, metavar=("A", "B"))
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    if a.digest_check:
        dig = []
        for p in a.digest_check:
            h = hashlib.sha256()
            n = 0
            for line in open(p, "rb"):
                h.update(line)
                n += 1
            dig.append({"path": p, "rows": n, "digest": h.hexdigest()})
        same = dig[0]["digest"] == dig[1]["digest"] and dig[0]["rows"] == dig[1]["rows"]
        print(json.dumps({"same": bool(same), "a": dig[0], "b": dig[1]}, ensure_ascii=False,
                         indent=1))
        return 0 if same else 1

    sd = _load_probe("sd_gen", os.path.join(RESEARCH, "probes", "sla_data.py.txt"))
    t0 = time.time()
    inst = enumerate_instances(a.seed, max_rows=a.max_rows)
    rows, bad = [], []
    global MISS_TAIL
    MISS_TAIL = []
    for x in inst:
        try:
            row = build_row(x, a.seed, sd)
            rows.append(row)
        except AssertionError as e:
            bad.append({"iid": _instance_id(x), "err": str(e)[:300]})
    # value reachability: a real per-slot spot check (not a sampling estimate) -- misses are dumped
    # for localization
    auc = {"n_state": 0, "n_lit_state": 0, "n_choice": 0, "n_lit_choice": 0, "misses": []}
    for r in rows:
        steps = [(s["op"], a, None) for s, a in zip(r["steps"], r["sf_args"])]
        aa = audit_values(steps, r["v32"]["state_texts"])
        for k in ("n_state", "n_lit_state", "n_choice", "n_lit_choice"):
            auc[k] += aa[k]
        if aa["misses"] and len(auc["misses"]) < 20:
            auc["misses"].append(aa["misses"][0])
    MISS_TAIL = auc["misses"]
    n_state = sum(r["v32"]["value_access"]["n_state"] for r in rows)
    n_lit = sum(r["v32"]["value_access"]["n_literal_state"] for r in rows)
    n_choice = sum(r["v32"]["value_access"]["n_choice"] for r in rows)
    cover = (n_lit / n_state) if n_state else None
    by_first, by_kind, by_split, by_len = {}, {}, {}, {}
    for r in rows:
        by_first[r["bridge_key"]] = by_first.get(r["bridge_key"], 0) + 1
        by_kind[r["v32"]["family_kind"]] = by_kind.get(r["v32"]["family_kind"], 0) + 1
        by_split[r["v32"]["heldout_split"]] = by_split.get(r["v32"]["heldout_split"], 0) + 1
        by_len[r["v32"]["n_steps"]] = by_len.get(r["v32"]["n_steps"], 0) + 1
    digest = hashlib.sha256()
    for r in rows:
        digest.update((json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    summary = {
        "schema": V32_SCHEMA, "seed": a.seed, "rows": len(rows), "failed": len(bad),
        "failed_examples": bad[:5],
        "value_access": {"n_state_slots": n_state, "n_literal_state": n_lit, "cover": cover,
                         "n_choice_slots": n_choice, "criterion": ">= 0.90 (状态槽)",
                         "misses": MISS_TAIL[:20]},
        "by_first_op": dict(sorted(by_first.items())),
        "by_kind": dict(sorted(by_kind.items())),
        "by_split": dict(sorted(by_split.items())),
        "by_n_steps": dict(sorted(by_len.items())),
        "digest_sha256": digest.hexdigest(),
        "secs": round(time.time() - t0, 1),
    }
    if a.audit_vals:
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    md5 = hashlib.md5(open(a.out, "rb").read()).hexdigest()
    summary["out"] = a.out
    summary["md5"] = md5
    summary["bytes"] = os.path.getsize(a.out)
    with open(a.out + ".md5", "w", encoding="utf-8") as fh:
        fh.write("%s  %s\n" % (md5, os.path.basename(a.out)))
    with open(a.out + ".digest.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    if not a.audit_vals:
        try:
            from .audit_data import nontriviality_report, operator_coverage
            gate = nontriviality_report([r for r, _ in _iter_programs(a.out)])
            cov = operator_coverage([("skel", a.out)])
            cov_all = operator_coverage([("skel", a.out),
                                         ("v31mix", os.path.join(RESEARCH, "data",
                                                                  "sla_v31_mixed.jsonl"))])
            summary["nontrivial_gate"] = gate
            summary["coverage"] = {"skel_only": {"n_ops_total": cov["n_ops_total"],
                                                "n_ops_used": cov["n_ops_used"],
                                                "zero_coverage": cov["zero_coverage"],
                                                "used_ranked": cov["used_ranked"],
                                                "table_used": [x for x in cov["table"]
                                                                if x["rows"] > 0]},
                                    "combined": {"n_ops_total": cov_all["n_ops_total"],
                                                  "n_ops_used": cov_all["n_ops_used"],
                                                  "zero_coverage": cov_all["zero_coverage"],
                                                  "zero_coverage_skel":
                                                      cov_all["zero_coverage_skel"],
                                                  "per_source": cov_all["per_source"]}}
            with open(a.out + ".digest.json", "w", encoding="utf-8") as fh:
                json.dump(summary, fh, ensure_ascii=False, indent=1)
        except Exception as e:
            summary["gate_error"] = repr(e)
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0 if (len(rows) >= 3000 and (cover or 0) >= 0.90 and not bad) else 1


def _load_probe(name, path):
    ld = importlib.machinery.SourceFileLoader(name, path)
    sp = importlib.util.spec_from_loader(name, ld)
    m = importlib.util.module_from_spec(sp)
    ld.exec_module(m)
    return m


if __name__ == "__main__":
    sys.exit(main())
