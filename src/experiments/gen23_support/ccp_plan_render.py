# -*- coding: utf-8 -*-
# ccp_plan_render.py -- B device v3 core: plan-consuming renderer + plan builders + perturbation family (gen17)
#
# Basis: constitution docs/EXPERIMENT_PLAN_P1P2.md §6 change 64 (user ruling: B extends the renderer, effective at once);
#        design = scratch/ccp/gen17/B_DESIGN.md (section by section).
# Semantics:
#   * render_plan(cls, name, head, concl, stmt, plan) consumes the plan step by step (operator order + slot bindings):
#       each operator -> OPS.call (independent re-verification by the certificate machine) -> that operator's tactic/template -> Lean lines;
#       any step failure stops at once and yields an **incomplete proof** (lines already emitted are kept, the why field records refused).
#   * Failure paths of bad plans (wrong operator / order / slot) per family are documented in B_DESIGN §4 (certificate-machine reject / order gate / statement-coupling gate / Lean reject).
#   * The old rule-arm path (RA.ARM) lives elsewhere; kept as the comparison column (two-way read of the readout files).
# Read-only deps: ccp_bridge / ccp_ops / ccp_rulearm / ccp_rollout (constants) / ccp_vocab (slot schema) / drct_common.
# Write surface: scratch/ccp/gen17/** only.
import collections
import copy
import hashlib
import json
import os
import sys
from fractions import Fraction

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = os.path.join(RES, "scratch", "ccp")
G13 = os.path.join(CCP, "gen13")
G16 = os.path.join(CCP, "gen16")
G17 = os.path.join(CCP, "gen17")
OUT = os.path.join(G17, "out")
LEANWD = os.path.join(G17, "lean")
os.makedirs(OUT, exist_ok=True)
os.makedirs(LEANWD, exist_ok=True)
for _p in (CCP, G13, G16, os.path.join(RES, "scratch", "drct"),
           os.path.join(RES, "cppcuda", "tools"), os.path.join(RES, "skel32"),
           os.environ.get("SCOPE_SUPEROPERATOR", os.path.join(RES, os.pardir, "device"))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ccp_bridge as BR                                                # noqa: E402
import ccp_ops as OPS                                                  # noqa: E402
import ccp_rulearm as RA                                               # noqa: E402
import ccp_vocab as V                                                  # noqa: E402 (single source of the slot schema)
import drct_common as DC                                               # noqa: E402

PLAN_OPS = set(OPS.PROOF_OPS)
SCHEMA = V.CCP_SLOT_SCHEMA
CONST_FIRST = {"ind": "ccp_ind_recurrence", "ineq": "ccp_ineq_sos",
               "nt": "ccp_nt_dvd_combine"}
CLOSING_OPS = ("ccp_ring_identity", "ccp_ineq_close", "ccp_nt_dvd_combine",
               "ccp_nt_dvd_prod_combine")


def stmt_theorem(head, concl):
    """Render-side statement (theorem form): the bridge regex requires ':' straight after the binder ')' (same measurement convention as arm_ind/arm_nt)."""
    return "%s\n    %s" % (head, concl)


def stmt_surface(item):
    """Decode-side statement (state form = training surface): same per-line convention as v1 eval_item / v2 load_sets."""
    s = item.get("stmt", "")
    if s.count("⊢") != 1:
        return "%s ⊢ %s" % (RA.binders_of(item["head"]), item["concl"])
    return s

# Slot overlay table (B_DESIGN §3.2): op -> [(slot path, arg path, expected kind)]
OVERLAY = {
    "ccp_ind_recurrence": [("form", "form", "str"), ("v0", "v0", "int"),
                           ("step", "step", "int"), ("family", "family", "str")],
    "ccp_ring_identity": [("nvars", "nvars", "int"), ("atom", "atom", "str")],
    "ccp_ineq_sos": [],
    "ccp_ineq_close": [("residual", "residual", "num")],
    "ccp_nt_dvd_witness": [("m", "m", "int")],
    "ccp_nt_dvd_combine": [("m_h", "m_h", "int"), ("M", "M", "int"), ("const", "const", "int")],
    "ccp_nt_dvd_prod_combine": [("m_h", "m_h", "int")],
}
READOUT_OPS = {"ccp_ineq_sos": ["nvars", "dom_c0", "dom_c2"]}


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- slot-value helpers
def slot_of(id2val, vid):
    """Value id -> (kind, typed_value); UNK/out-of-range -> None."""
    try:
        vid = int(vid)
    except Exception:                                                  # noqa: BLE001
        return None
    if vid <= 0 or vid >= len(id2val):
        return None
    tv = id2val[vid]
    if not tv:
        return None
    kind, val = tv[0], tv[1]
    if kind == "any":
        return None
    return kind, val


def _coerce(ekind, kind, val):
    """Model/label slot value -> argument value (type check; on mismatch returns (False, None))."""
    if ekind == "int":
        if kind != "int" or isinstance(val, bool):
            return False, None
        return True, int(val)
    if ekind == "str":
        if kind != "str":
            return False, None
        return True, str(val)
    if ekind == "num":
        if kind not in ("int", "Fraction"):
            return False, None
        return True, str(val)
    return False, None


def slot_index(op, path):
    return [p for (p, _k) in SCHEMA[op]].index(path)


def overlay_slots(op, args_dev, sv, id2val):
    """Device argument skeleton + model/label slots -> (args, slot_src, slot_info). Does not mutate args_dev (deepcopy).

    Fallback rule (B_DESIGN §3.2): when the slot value is UNK / type-mismatched / the device-derived
    value is a list, keep the device value, set slot_src='device_fallback', and record a per-slot trace.
    """
    args = copy.deepcopy(args_dev)
    slot_src, slot_info = {}, {}
    for path, akey, ekind in OVERLAY.get(op, []):
        si = slot_index(op, path)
        vid = sv[si] if si < len(sv) else -1
        svv = slot_of(id2val, vid)
        if svv is None:
            slot_src[path] = "device_fallback"
            slot_info[path] = {"vid": int(vid), "src": "device_fallback", "reason": "unk_or_oob"}
            continue
        ok, conv = _coerce(ekind, svv[0], svv[1])
        if not ok:
            slot_src[path] = "device_fallback"
            slot_info[path] = {"vid": int(vid), "src": "device_fallback",
                               "reason": "kind_%s_not_%s" % (svv[0], ekind)}
            continue
        if isinstance(args.get(akey), list) and not isinstance(conv, list):
            slot_src[path] = "device_fallback"
            slot_info[path] = {"vid": int(vid), "src": "device_fallback",
                               "reason": "device_value_is_list"}
            continue
        args[akey] = conv
        slot_src[path] = "model"
        slot_info[path] = {"vid": int(vid), "value": conv, "src": "model"}
    return args, slot_src, slot_info


def derive_ring_args(form, step, c1, c0, nvars=1, side=4):
    """Device derivation of the ind ring's structural terms (same recipe as the generator, B_DESIGN §3.3)."""
    c1, c0 = Fraction(c1), Fraction(c0)
    if form == "lin":
        d = int(step)
        lhs, rhs, atom = {(0,): c0 + d, (1,): c1}, {(0,): c0 + c1, (1,): c1}, "n"
    else:
        r, c = int(step[0]), int(step[1])
        lhs, rhs, atom = {(0,): r * c0 + c, (1,): r * c1}, {(0,): c0, (1,): c1 * r}, "%d^n" % r
    return {"nvars": int(nvars), "atom": atom, "side": int(side),
            "lhs": {",".join(map(str, k)): str(v) for k, v in lhs.items() if v != 0},
            "rhs": {",".join(map(str, k)): str(v) for k, v in rhs.items() if v != 0}}


# ---------------------------------------------------------------- statement-coupling gate (ind)
def ind_statement_coupling(stmt, args):
    try:
        br = BR.bridge_ind(stmt)
    except Exception as e:                                             # noqa: BLE001
        return "bridge_fail:%s" % type(e).__name__

    def _n(x):
        return (x[0], None if x[1] is None else int(x[1]),
                list(map(int, x[2])) if isinstance(x[2], list) else int(x[2]))
    got = (args.get("form"), args.get("v0"), args.get("step"))
    exp = (br["form"], br["v0"], br["step"])
    try:
        g, e = _n(got), _n(exp)
    except Exception:                                                  # noqa: BLE001
        return "slot_shape_bad:%s" % (got,)
    if g != e:
        return "slot_vs_bridge_mismatch:%s!=%s" % (g, e)
    return None


# ---------------------------------------------------------------- renderer (core)
def render_plan(cls, name, head, concl, stmt, plan):
    """plan (operator order + slot bindings) -> {lines, refused, step_log}. Instantiates proof lines step by step."""
    lines, log, state, refused = [], [], {}, None
    steps = plan.get("steps") or []
    for i, st in enumerate(steps):
        op = st.get("op")
        args = copy.deepcopy(st.get("args") or {})
        entry = {"i": i, "op": op, "slot_src": st.get("slot_src")}
        log.append(entry)
        if op not in PLAN_OPS:
            entry["err"] = "op_not_whitelisted"
            refused = "step%d_op_not_whitelisted:%s" % (i, op)
            break
        try:
            r = OPS.call(op, args)
        except Exception as e:                                         # noqa: BLE001
            entry["err"] = "op_exc:%s" % type(e).__name__
            refused = "step%d_op_exc:%s" % (i, type(e).__name__)
            break
        entry.update({"ok": bool(r.ok), "cert_ok": bool(r.cert_ok), "err": r.err,
                      "cert_detail": (r.cert_detail or "")[:200]})
        if not (r.ok and r.cert_ok):
            refused = "step%d_%s:%s" % (i, op, r.err or "cert_reject")
            break
        if op == "ccp_ind_recurrence":
            coup = ind_statement_coupling(stmt, args)
            if coup:
                entry["coupling"] = coup
                refused = "step%d_ind_%s" % (i, coup)
                break
            bind = RA.binders_of(head)
            h0 = RA.hyp_name(bind, r"f\s+0\s*=")
            hstep = RA.hyp_name(bind, r"∀\s*k")
            if not h0 or not hstep:
                refused = "step%d_ind_hypnames_missing" % i
                break
            try:
                decl = BR.bridge_ind_declared(stmt)
            except Exception as e:                                     # noqa: BLE001
                refused = "step%d_ind_declared:%s" % (i, type(e).__name__)
                break
            rw = "rw [%s, ih%s]" % (hstep, ", pow_succ" if "^" in decl else "")
            state["ind"] = {"form": args.get("form"), "step": args.get("step"),
                            "c1": r.value["c1"], "c0": r.value["c0"]}
            lines += ["intro n", "induction n with", "| zero => simp [%s]" % h0,
                      "| succ n ih =>", "  " + rw]
            entry["lines_emitted"] = 5
        elif op == "ccp_ring_identity":
            if "ind" not in state:
                refused = "step%d_ring_before_recurrence" % i
                break
            lines += (["  push_cast", "  ring"] if state["ind"]["form"] == "lin"
                      else ["  ring"])
            entry["lines_emitted"] = 2 if state["ind"]["form"] == "lin" else 1
        elif op == "ccp_ineq_sos":
            state["sos"] = True
            entry["hints"] = list(r.value.get("hints") or [])
            entry["lines_emitted"] = 0
        elif op == "ccp_ineq_close":
            if not state.get("sos"):
                refused = "step%d_close_before_sos" % i
                break
            try:
                hints = [OPS.lin_text(sq["coefs"], args["vars"], sq["shift"])
                         for sq in args["squares"]]
            except Exception as e:                                     # noqa: BLE001
                refused = "step%d_close_hints:%s" % (i, type(e).__name__)
                break
            lines.append("nlinarith [%s]" % ", ".join("sq_nonneg (%s)" % h for h in hints))
            entry["hints"] = hints
            entry["lines_emitted"] = 1
        elif op == "ccp_nt_dvd_witness":
            state["wit"] = int(state.get("wit", 0)) + 1
            lines.append(r.tactic.strip())
            entry["lines_emitted"] = 1
        elif op in ("ccp_nt_dvd_combine", "ccp_nt_dvd_prod_combine"):
            if not state.get("wit"):
                refused = "step%d_combine_before_witness" % i
                break
            lines.append(r.tactic.strip())
            entry["lines_emitted"] = 1
    # Completeness gate (B_DESIGN §2): the plan must contain one closing operator; otherwise it is an "incomplete proof" even with no mid-way error.
    if refused is None:
        closed = [e.get("op") for e in log if e.get("ok") and e.get("cert_ok")
                  and e.get("op") in CLOSING_OPS]
        if not closed:
            refused = "plan_incomplete:no_closing_op"
    return {"lines": lines, "refused": refused, "step_log": log}


def judge_plan(name, head, concl, res, wd, tag):
    """Render result -> judging record (Lean∧N1∧N2; when refused, still judge the incomplete proof to get Lean-level evidence)."""
    lines, refused = res["lines"], res["refused"]
    rec = {"refused": refused, "n_lines": len(lines), "lines": lines,
           "step_log": res["step_log"]}
    if refused and not lines:
        rec.update({"certified": False, "lean": None, "reasons": [refused]})
        return rec
    os.makedirs(wd, exist_ok=True)
    j = DC.judge_attempt(name, head, concl, "\n".join(lines), wd, tag)
    rec["lean"] = ({k: j["lean"].get(k) for k in ("rc", "accepted", "sorry", "secs")}
                   if j.get("lean") else None)
    rec["reasons"] = list(j.get("reasons") or []) + ([refused] if refused else [])
    rec["certified"] = bool(j.get("certified")) and not refused
    return rec


# ---------------------------------------------------------------- problem source (synthetic held-out v3)
CANONICAL_SHAPE = {"ind": ["ccp_ind_recurrence", "ccp_ring_identity"],
                   "ineq": ["ccp_ineq_sos", "ccp_ineq_close"]}


def label_shape_ok(cls, row):
    """Whether the course label chain has the **full canonical shape** (B_DESIGN §4: the main criterion uses only complete labels; truncated ones form a separate supplementary family)."""
    ops = list(row.get("chain_ops") or [])
    if cls in CANONICAL_SHAPE:
        return ops == CANONICAL_SHAPE[cls]
    if cls == "nt":
        return (len(ops) >= 2 and ops[-1] in ("ccp_nt_dvd_combine", "ccp_nt_dvd_prod_combine")
                and set(ops[:-1]) == {"ccp_nt_dvd_witness"})
    return False


def _held_rows(cls):
    """Held-out rule: accepted ∧ ¬readout_only ∧ md5(name)%10==7 ∧ item-level dedup; name ascending."""
    p = os.path.join(CCP, "gen", "ccp_%s_rows.jsonl" % cls)
    seen, out = set(), []
    for ln in open(p, encoding="utf-8"):
        if not ln.strip():
            continue
        r = json.loads(ln)
        if not r.get("accepted") or r.get("readout_only"):
            continue
        if int(hashlib.md5(r["ccp_name"].encode("utf-8")).hexdigest()[:8], 16) % 10 != 7:
            continue
        if r["ccp_name"] in seen:
            continue
        seen.add(r["ccp_name"])
        out.append(r)
    out.sort(key=lambda r: r["ccp_name"])
    return out


def item_of_row(cls, r):
    h0txt, concl = BR.split_concl(r["stmt"])
    return {"cls": cls, "name": r["ccp_name"], "head": "theorem %s %s :" % (r["ccp_name"], h0txt),
            "concl": concl, "stmt": r["stmt"], "family": r.get("ccp_family"),
            "rowkind": r.get("ccp_rowkind"), "head_note": "state→theorem 归一",
            "gold_ops": r.get("chain_ops")}


def synth_held_items_v3(cls, per_class=None, require_complete_label=True):
    """Synthesize course held-out rows (complete label chains; the main-criterion problem source)."""
    if require_complete_label:
        out = [item_of_row(cls, r) for r in _held_rows(cls) if label_shape_ok(cls, r)]
    else:
        out = [item_of_row(cls, r) for r in _held_rows(cls)]
    return out[:per_class] if per_class else out


def synth_held_truncated(cls):
    """Held-out items whose label chains are **truncated** (supplementary family; excluded from the main-criterion denominator)."""
    return [{"cls": cls, "name": r["ccp_name"],
             "head": "theorem %s %s :" % (r["ccp_name"], BR.split_concl(r["stmt"])[0]),
             "concl": BR.split_concl(r["stmt"])[1], "stmt": r["stmt"],
             "rowkind": r.get("ccp_rowkind"), "gold_ops": r.get("chain_ops"),
             "n_sf": len(r.get("sf_args") or [])} for r in _held_rows(cls)
            if not label_shape_ok(cls, r)]


def rows_by_name(cls):
    p = os.path.join(CCP, "gen", "ccp_%s_rows.jsonl" % cls)
    out = {}
    for ln in open(p, encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln)
            out.setdefault(r["ccp_name"], r)
    return out


# ---------------------------------------------------------------- GT plan (course labels)
def gt_plan(cls, row, concl):
    """Course label -> plan (operator order = chain_ops; slots = sf_args; structural terms completed per B_DESIGN §3.3)."""
    ops = list(row.get("chain_ops") or [])
    sf = list(row.get("sf_args") or [])
    if len(ops) != len(sf) or not ops:
        return None, "gt_label_shape:ops=%d args=%d" % (len(ops), len(sf))
    if not label_shape_ok(cls, row):
        return None, "gt_label_truncated:ops=%s" % (ops,)
    steps = []
    if cls == "ind":
        a0 = copy.deepcopy(sf[0])
        r1 = OPS.call("ccp_ind_recurrence", a0)
        if not (r1.ok and r1.cert_ok):
            return None, "gt_step0_%s" % (r1.err or "cert_reject")
        ring = derive_ring_args(a0["form"], a0["step"], r1.value["c1"], r1.value["c0"],
                                nvars=(sf[1].get("nvars") if isinstance(sf[1], dict) else 1))
        steps.append({"op": ops[0], "args": a0,
                      "slots": {p: V.extract_slot(p, a0) for (p, _k) in SCHEMA[ops[0]]},
                      "slot_src": {p: "gt" for (p, _k) in SCHEMA[ops[0]]
                                   if V.extract_slot(p, a0) is not None},
                      "struct_src": ["sample_n", "lhs", "rhs"]})
        steps.append({"op": ops[1], "args": ring,
                      "slots": {"nvars": ring["nvars"], "atom": ring["atom"]},
                      "slot_src": {"nvars": "gt", "atom": "gt"},
                      "struct_src": ["lhs", "rhs", "side"]})
    elif cls == "ineq":
        a0 = copy.deepcopy(sf[0])
        a1 = copy.deepcopy(sf[1])
        close = {"vars": a1["vars"], "squares": a1["squares"],
                 "residual": a1["residual"], "text": concl}
        steps.append({"op": ops[0], "args": a0,
                      "slots": {p: V.extract_slot(p, a0) for (p, _k) in SCHEMA[ops[0]]},
                      "slot_src": {p: "gt" for (p, _k) in SCHEMA[ops[0]]},
                      "struct_src": ["dom"]})
        steps.append({"op": ops[1], "args": close,
                      "slots": {p: V.extract_slot(p, close) for (p, _k) in SCHEMA[ops[1]]},
                      "slot_src": {p: "gt" for (p, _k) in SCHEMA[ops[1]]
                                   if V.extract_slot(p, close) is not None},
                      "struct_src": ["squares", "text"]})
    elif cls == "nt":
        for op, a in zip(ops, sf):
            a = copy.deepcopy(a)
            steps.append({"op": op, "args": a,
                          "slots": {p: V.extract_slot(p, a) for (p, _k) in SCHEMA[op]},
                          "slot_src": {p: "gt" for (p, _k) in SCHEMA[op]
                                       if V.extract_slot(p, a) is not None},
                          "struct_src": ["hyp", "var", "text", "terms", "hypvars"]})
    else:
        return None, "gt_unknown_cls:%s" % cls
    if [s["op"] for s in steps] != ops:
        return None, "gt_op_mismatch"
    return {"cls": cls, "src": "gt_course_label", "steps": steps}, None


def gt_plan_truncated(cls, row, concl):
    """Truncated label chain -> plan (for the supplementary family; the course-side half-way / endfail labels, expected not to close)."""
    ops = list(row.get("chain_ops") or [])
    sf = list(row.get("sf_args") or [])
    steps, prev_ind = [], None
    for op, a in zip(ops, sf):
        a = copy.deepcopy(a)
        if op == "ccp_ring_identity" and prev_ind is not None:
            a = derive_ring_args(prev_ind["form"], prev_ind["step"], prev_ind["c1"],
                                 prev_ind["c0"], nvars=(a.get("nvars") or 1))
        if op == "ccp_ind_recurrence":
            r = OPS.call(op, a)
            prev_ind = ({"form": a.get("form"), "step": a.get("step"),
                         "c1": r.value["c1"], "c0": r.value["c0"]}
                        if (r.ok and r.cert_ok and not isinstance(a.get("step"), list)) else None)
        steps.append({"op": op, "args": a, "slots": {}, "slot_src": {},
                      "struct_src": ["label_truncated"]})
    return {"cls": cls, "src": "gt_label_truncated", "steps": steps}, None


# ---------------------------------------------------------------- model plan (135M decode output)
def model_plan(cls, steps_dec, id2val, dev_slots=False):
    """Decoded steps (op/sv/args_dev) -> plan. dev_slots=True means all-device slots (diagnostic column)."""
    out, ind_state = [], None
    for st in steps_dec:
        op, args_dev, sv = st["op"], st["args_dev"], st["sv"]
        if op == "ccp_ring_identity" and ind_state is not None:
            args_dev = derive_ring_args(ind_state["form"], ind_state["step"],
                                        ind_state["c1"], ind_state["c0"],
                                        nvars=(args_dev.get("nvars") or 1))
        if dev_slots:
            args = copy.deepcopy(args_dev)
            src = {p: "device" for (p, _k) in SCHEMA.get(op, [])}
            info = {p: {"src": "device"} for (p, _k) in SCHEMA.get(op, [])}
        else:
            args, src, info = overlay_slots(op, args_dev, sv, id2val)
        out.append({"op": op, "args": args, "slots": {}, "slot_src": src,
                    "slot_info": info, "struct_src": ["device_bridge"]})
        if op == "ccp_ind_recurrence":
            r = OPS.call(op, args)
            if r.ok and r.cert_ok and not isinstance(args.get("step"), list):
                ind_state = {"form": args.get("form"), "step": args.get("step"),
                             "c1": r.value["c1"], "c0": r.value["c0"]}
            else:
                ind_state = None
    return {"cls": cls, "src": "model_v3_devslots" if dev_slots else "model_v3",
            "steps": out}


# ---------------------------------------------------------------- bad-plan family (B_DESIGN §4)
def pert_drop_last(plan):
    p = copy.deepcopy(plan)
    p["src"] = "perturb:drop_last"
    p["steps"] = p["steps"][:-1] if len(p["steps"]) > 1 else []
    return p


def pert_swap(plan):
    p = copy.deepcopy(plan)
    p["src"] = "perturb:swap"
    p["steps"] = list(reversed(p["steps"]))
    return p


def pert_slot_scalar(plan, cls):
    p = copy.deepcopy(plan)
    p["src"] = "perturb:slot_scalar"
    try:
        if cls == "ind":
            st = p["steps"][0]
            st["args"]["v0"] = int(st["args"]["v0"]) + 1
            st["slot_src"]["v0"] = "perturb"
        elif cls == "ineq":
            st = p["steps"][1]
            st["args"]["residual"] = str(Fraction(st["args"]["residual"]) + 1)
            st["slot_src"]["residual"] = "perturb"
        elif cls == "nt":
            st = p["steps"][0]
            st["args"]["m"] = int(st["args"]["m"]) + 1
            st["slot_src"]["m"] = "perturb"
    except Exception:                                                  # noqa: BLE001
        return None
    return p


def pert_slot_struct(plan, cls):
    p = copy.deepcopy(plan)
    p["src"] = "perturb:slot_struct"
    try:
        if cls == "ind":
            st = p["steps"][1]
            lh = dict(st["args"]["lhs"])
            k = sorted(lh)[0]
            lh[k] = str(Fraction(lh[k]) + 1)
            st["args"]["lhs"] = lh
        elif cls == "ineq":
            st = p["steps"][1]
            sq = [dict(s) for s in st["args"]["squares"]]
            sq[0]["shift"] = str(Fraction(str(sq[0]["shift"])) + 1)
            st["args"]["squares"] = sq
        elif cls == "nt":
            st = p["steps"][-1]
            tm = [dict(t) for t in st["args"]["terms"]]
            tm[0]["coef"] = int(tm[0]["coef"]) + 1
            st["args"]["terms"] = tm
    except Exception:                                                  # noqa: BLE001
        return None
    return p


PERTURBERS = collections.OrderedDict([
    ("drop_last", lambda plan, cls: pert_drop_last(plan)),
    ("swap", lambda plan, cls: pert_swap(plan)),
    ("slot_scalar", pert_slot_scalar),
    ("slot_struct", pert_slot_struct),
])


# ---------------------------------------------------------------- rule column (two-way comparison; read from record files, never re-run)
def load_rule_column():
    """-> ({name: (certified, reason)}, source md5). RULEARM_both.json + ANCHOR_V2.json."""
    col, src = {}, {}
    p1 = os.path.join(G13, "out", "RULEARM_both.json")
    if os.path.exists(p1):
        d = json.load(open(p1, encoding="utf-8"))
        for r in d["sets"]["synth_held"]["records"]:
            col[r["name"]] = (bool(r["certified"]), r["reason"])
        src[p1] = _md5(p1)
    p2 = os.path.join(CCP, "ANCHOR_V2.json")
    if os.path.exists(p2):
        d = json.load(open(p2, encoding="utf-8"))
        for cls in ("ind", "ineq", "nt"):
            for it in d["classes"].get(cls, []):
                col[it["name"]] = (bool(it["rulearm"]["certified"]), it["rulearm"]["reason"])
            for it in (d.get("parse_pool") or {}).get(cls, []):
                col.setdefault(it["name"], ())
                col[it["name"]] = (bool(it["rulearm"]["certified"]), it["rulearm"]["reason"])
        src[p2] = _md5(p2)
    return col, src


def plan_pattern_of(cls, head):
    """Plan pattern (single source = ccp_rollout.plan_pattern / n_hyps_of; lazy import so the CPU path does not pull in torch)."""
    global _RO
    if _RO is None:
        import ccp_rollout as _m
        _RO = _m
    return _RO.plan_pattern(cls, _RO.n_hyps_of(head))


_RO = None


if __name__ == "__main__":
    print("[plan-render] self-check: PLAN_OPS=%s" % sorted(PLAN_OPS))
    print("[plan-render] schema ops = %s" % sorted(SCHEMA))
    assert set(SCHEMA) == PLAN_OPS, (set(SCHEMA) ^ PLAN_OPS)
    print("[plan-render] OK")
