# -*- coding: utf-8 -*-
# gen44_census_v3.py -- node [2] Lean-Workbook three-state census v3
#   (= fork of v2; **only the C-layer label decision is fixed, plus R_none subdivision and
#    freevar attribution**; the device call path (parse_stmt/chain_of) is byte-for-byte unchanged,
#    so the A1 counts must be bitwise identical to v1/v2, and we assert that)
#
# The three v3 fixes over v2 (E6 erratum, they affect **labels only**, never any decision):
#   (1) C layer: first check the **leading quantifier** (∀/∃), then scan **depth-0 connectives**
#       (∧∨↔→), and only then the relation symbol; and separate "single relation" / "chained
#       multiple relations" / "relation + connective composition". (v2 took the leftmost relation
#       symbol only, so `x ≥ 1 ∧ y ≥ 2` was mislabelled C_rel:≥ while the device actually fails
#       with chr:∧.)
#   (2) C/H layer unparsed sub-kinds: attributed from the device PErr text as-is, adding
#       exp_big/fact/nonarith_token; `fvar:<dotted library name>` => constructor sub-kind,
#       `fvar:<plain name>` => freevar class (further split into pred_binder/undefined).
#   (3) R_none:has_vars subdivided by relation symbol x hypothesis shape x whether the right side
#       is closed (for the B-list tiers).
#
# Reads: data/LWB_ALL.jsonl · gen42/gen42_stmt.py · gen42/gen42_families.py · gen42/cache/CCP_VOCAB_V6.json
# Writes: scratch/ccp/gen44/**
import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = os.path.join(RES, "scratch", "ccp")
G42 = os.path.join(CCP, "gen42")
G44 = os.path.join(CCP, "gen44")
sys.path.insert(0, G42)
sys.path.insert(0, _PKG_DIR)

import gen42_stmt as S                                                  # noqa: E402
import gen42_families as F                                              # noqa: E402

SCHEMA = "gen44-census/3"
V6_PATH = os.path.join(G42, "cache", "CCP_VOCAB_V6.json")

RELS = ("≠", "≤", "≥", "=", "<", ">", "∣")
CONNS = ("∧", "∨", "↔", "→")

TYPE_HEADS = {
    "Polynomial", "Matrix", "Set", "Finset", "List", "Multiset", "Vector", "Fin", "Type",
    "Sort", "NNReal", "ENNReal", "Subtype", "Quot", "Equiv", "Function", "Array", "Option",
    "Sum", "Prod", "Sigma", "ULift", "PLift", "ZMod", "Nat", "Int", "Rat", "Real", "Complex",
    "Interval", "Filter", "Submodule", "Ideal", "Module", "Basis", "Symmetric",
    "Finsupp", "Fintype", "Bool", "Prop", "Unit", "Empty", "PUnit", "PEmpty", "Icc", "Ioo",
    "Ico", "Ioc", "Sym", "UInt8", "UInt16", "UInt32", "UInt64", "Float", "String", "Char",
    "EuclideanSpace", "MeasureTheory", "Topology", "OrderIso", "RelIso", "PNat",
}
PRED_HEADS = {
    "Prime", "Odd", "Even", "Coprime", "Irrational", "Rational", "Integer", "Continuous",
    "Differentiable", "Monotone", "StrictMono", "Antitone", "StrictAnti", "Injective",
    "Surjective", "Bijective", "Symmetric", "Reflexive", "Transitive", "Integral", "Positive",
    "Nonneg", "Nonzero", "True", "False", "Fermat", "Lipschitz", "Convex", "Concave",
    "Bounded", "Convergent", "Cauchy", "Measurable", "Periodic", "Monotonic",
}
SET_HEADS = {"Set", "Finset", "List", "Multiset", "Interval", "ZMod", "Finsupp", "Subtype",
             "TopologicalSpace", "MeasurableSpace", "Filter", "Ideal", "Submodule"}
SYM_FEATURE = (
    (("Real.sqrt", "NNReal.sqrt"), "sqrt"),
    (("Real.log", "Real.exp", "Real.rpow", "Real.logb"), "exp_log"),
    (("Real.sin", "Real.cos", "Real.tan", "Real.arctan", "Real.arcsin", "Real.arccos",
      "Real.sinh", "Real.cosh", "Real.tanh"), "trig"),
    (("Real.pi", "Complex.I", "Real.e"), "const_special"),
    (("Nat.fib", "Int.fib", "Nat.sqrt", "Nat.log", "Nat.choose", "Nat.factorial",
      "Nat.gcd", "Nat.lcm", "Int.gcd", "Nat.Prime", "Nat.Coprime"), "funcapp"),
)


def _norm_dev(d):
    """Device PErr text as-is -> (kind, token)"""
    if not d:
        return (None, None)
    d = re.sub(r"^(concl|hyp)_unparsed:", "", d)
    d = re.sub(r"^h[^:]*:", "", d) if d.startswith("h") and ":" in d[:24] else d
    m = re.match(r"fvar:([A-Za-z0-9_.'\u00c0-\u024f]+)", d)
    if m:
        return ("fvar", m.group(1))
    m = re.match(r"chr '(.{0,3})'", d)
    if m:
        return ("chr", m.group(1))
    m = re.match(r"atom:(.+)", d)
    if m:
        return ("atom", m.group(1).strip("',"))
    return (d.split(":")[0].split("@")[0], None)


def _feat(kind, tok):
    """(kind,tok) -> sub-kind name"""
    if kind == "fvar":
        for keys, f in SYM_FEATURE:
            for k in keys:
                if tok.startswith(k):
                    return f
        if "." in tok:
            return "funcapp"
        if len(tok) == 1:
            return "freevar"
        return "freevar"
    if kind == "chr":
        return {"√": "sqrt", "∑": "bigop", "∏": "bigop", "π": "const_special",
                "|": "abs", "‖": "norm", "⌊": "floor", "⌈": "ceil", "∞": "inf",
                "α": "greek_type", "β": "greek_type", "θ": "greek_type",
                "λ": "lambda", "ℝ": "type_tok", "ℕ": "type_tok", "ℤ": "type_tok",
                "ℚ": "type_tok", "ℂ": "type_tok", "⁻": "superscript",
                ".": "dot_tok", "=": "reldup", ",": "comma", ";": "semi",
                "←": "arrow", "→": "arrow", "∀": "quant", "∃": "quant",
                "∧": "and", "∨": "or", "↔": "iff", "∈": "member", "∉": "member",
                "±": "pm"}.get(tok, "nonarith_token")
    if kind == "expnotnum":
        return "pow_sym"
    if kind == "expbig":
        return "exp_big"
    if kind in ("sumparse", "sumbounds", "sumcomma", "sumset", "sumtooloong",
                "sumbadrange", "summand_not_closed"):
        return "bigop"
    if kind == "fact_sym" or kind == "fact_nonint":
        return "fact"
    if kind == "absclose":
        return "abs"
    if kind == "rparen":
        return "paren"
    if kind in ("lex", "trailing"):
        return "lex"
    return "other"


def _first_ident(s):
    m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_']*)", s or "")
    return m.group(1) if m else ""


def _has_dot_capital(s):
    return bool(re.search(r"\.[A-Z][A-Za-z0-9_']*", s or ""))


def classify_group(ty):
    t = (ty or "").strip()
    rel, _i = S.top_rel(t)
    if rel:
        return "hyp_rel"
    if t[:1] in ("∀", "∃", "¬"):
        return "hyp_quant"
    if "∈" in t or "∉" in t:
        return "hyp_pred:member"
    if "[ZMOD" in t or t[:1] == "≡":
        return "hyp_pred:wmod"
    if _has_dot_capital(t):
        return "hyp_pred:dot"
    fi = _first_ident(t)
    if fi in PRED_HEADS and len(t) > len(fi):
        return "hyp_pred:head"
    if t in S.NUMERIC_TYPES:
        return "var_num"
    if "→" in t:
        parts = [p.strip() for p in re.split(r"→", t)]
        if all(p in S.NUMERIC_TYPES or p in ("Type*", "Type") for p in parts):
            return "var_nonnum:fn_num"
        return "var_nonnum:fn_other"
    if fi in SET_HEADS:
        return "var_nonnum:set"
    if fi in ("Matrix", "Vector", "Fin", "Module", "EuclideanSpace"):
        return "var_nonnum:matrix"
    return "var_nonnum:other"


def _depth0(s, chars):
    out, d = [], 0
    for i, ch in enumerate(s):
        if ch in "([{":
            d += 1
        elif ch in ")]}":
            d -= 1
        elif d == 0 and ch in chars:
            out.append((i, ch))
    return out


def concl_layer(concl):
    """v3: leading quantifier -> depth-0 connectives -> relation symbol;
    returns (label, rel, neg, nrel, nconn)"""
    ct = (concl or "").strip()
    neg = 0
    while ct.startswith("¬"):
        neg += 1
        ct = ct[1:].strip()
    for _ in range(8):
        if ct.startswith("(") and ct.endswith(")"):
            inner = ct[1:-1].strip()
            if (inner[:1] in ("∀", "∃") or _depth0(inner, RELS)
                    or _depth0(inner, CONNS)):
                ct = inner
                continue
        break
    if ct[:1] == "∀":
        return "C_forall", None, neg, 0, 0
    if ct[:1] == "∃":
        return "C_exists", None, neg, 0, 0
    conns = _depth0(ct, CONNS)
    rels = _depth0(ct, RELS)
    if conns:
        ch = conns[0][1]
        lab = {"∧": "C_and", "∨": "C_or", "↔": "C_iff", "→": "C_imp"}[ch]
        return lab + ("_post" if rels else "_only"), None, neg, len(rels), len(conns)
    if len(rels) >= 2:
        return "C_relchain:" + rels[0][1], rels[0][1], neg, len(rels), 0
    if len(rels) == 1:
        return "C_rel:" + rels[0][1], rels[0][1], neg, 1, 0
    if ct[:1] == "¬":
        return "C_not", None, neg, 0, 0
    return "C_other", None, neg, 0, 0


class P(object):
    __slots__ = ("name", "head", "concl", "split", "idx")

    def __init__(self, name, head, concl, split, idx):
        self.name, self.head, self.concl, self.split, self.idx = name, head, concl, split, idx


def analyse(rec, v6ops):
    gs = S.binder_groups(rec["binders"])
    labs, hyps = [], []
    for names, ty, br in gs:
        lab = classify_group(ty)
        labs.append(lab)
        if lab.startswith("hyp_"):
            hyps.append(lab)
    nhyp = collections.Counter(l.split(":")[0] for l in hyps)
    if nhyp.get("hyp_quant"):
        hlab = "H_quant"
    elif nhyp.get("hyp_pred"):
        hlab = "H_pred:" + collections.Counter(
            l for l in hyps if l.startswith("hyp_pred")).most_common(1)[0][0].split(":")[1]
    elif nhyp.get("hyp_rel"):
        hlab = "H_rel"
    else:
        hlab = "H_none"
    obad = [l for l in labs if l.startswith("var_nonnum")]
    osub = (collections.Counter(l.split(":")[1] for l in obad).most_common(1)[0][0]
            if obad else None)
    # Device-visible variable names (gen42 rule: no top-level relation and first char not ∀/∃/¬
    # => a variable).
    dev_vars, pred_names = set(), set()
    for names, ty, br in gs:
        rel, _i = S.top_rel(ty)
        if rel or ty.strip()[:1] in ("∀", "∃", "¬"):
            continue
        lab = classify_group(ty)
        if lab.startswith("hyp_pred"):
            pred_names |= set(names)
        elif lab.startswith("var"):
            dev_vars |= set(names)
    clab, crel, cneg, nrel, nconn = concl_layer(rec["concl"])
    row = {"idx": rec["idx"], "name": rec["name"], "sub": rec["subset"],
           "O": ("O_ok" if not obad else "O_nonnum:" + osub),
           "C": clab, "H": hlab, "nrel": nrel, "nconn": nconn,
           "n_grp": len(gs), "n_hyp": len(hyps), "n_var": len(dev_vars), "neg": cneg,
           "R": None, "st": None, "rs": None, "ops": None, "ops_v6": None,
           "pt_md5": None, "dev": None, "kind": None}
    if obad:
        row.update({"st": "state3", "rs": "O_nonnum:" + osub, "kind": osub})
        return row
    if nhyp.get("hyp_quant"):
        row.update({"st": "state3", "rs": "H_quant", "kind": "quant_hyp"})
        return row
    if not clab.startswith("C_rel:"):
        row.update({"st": "state3", "rs": clab, "kind": clab.split(":")[0]})
        return row
    prob = P(rec["name"], rec["head"], rec["concl"], rec["subset"], rec["idx"])
    d = S.parse_stmt(prob)
    if not d["ok"]:
        r = d["reason"] or "?"
        kind, tok = _norm_dev(r)
        base = r.split(":")[0]
        if base == "nonnumeric_type":
            pl = [l for l in labs if l.startswith("hyp_pred")]
            rs = ("H_pred:" + pl[0].split(":")[1]) if pl else "O_nonnum:other"
            row.update({"st": "state3", "rs": rs, "dev": r[:120], "kind": "nonnumeric"})
            return row
        if base in ("hyp_no_rel", "hyp_unparsed"):
            rs = "H_unparsed:" + _feat(kind, tok)
            row.update({"st": "state3", "rs": rs, "dev": r[:120], "kind": _feat(kind, tok)})
            return row
        if base == "concl_no_rel":
            row.update({"st": "state3", "rs": "C_other", "dev": r[:120], "kind": "no_rel"})
            return row
        f = _feat(kind, tok)
        if f == "freevar" and kind == "fvar":
            if tok in pred_names:
                f = "freevar_pred_binder"
            elif tok in dev_vars:
                f = "freevar_internal"
            else:
                f = "freevar_undefined"
        row.update({"st": "state3", "rs": "C_unparsed:" + f, "dev": r[:120], "kind": f})
        return row
    got, why = F.chain_of(d)
    if got is None:
        rr = F.route_reason(d)
        if str(rr) == "has_vars":
            rel = d["concl"]["rel"]
            side_closed = bool(S.closed(d["concl"]["lhs"]) or S.closed(d["concl"]["rhs"]))
            zero_side = False
            try:
                zero_side = (S.closed(d["concl"]["rhs"]) and
                             sp_nsimplify(d["concl"]["rhs"]) == 0)
            except Exception:                                           # noqa: BLE001
                pass
            if rel in ("≥", "≤", ">"):
                k = "ineq_nohyps" if not d["hyps"] else "ineq_withhyps"
            elif rel == "=":
                k = "eq_withhyps" if d["hyps"] else "eq_closed_edge"
            else:
                k = "other_rel"
            row.update({"st": "state2", "rs": "R_none:has_vars", "kind": k,
                        "zero_side": bool(zero_side), "side_closed": side_closed,
                        "dev": str(why)[:120]})
            return row
        row.update({"st": "state3", "rs": "R_none:" + str(rr)[:40],
                    "kind": "closed_false_or_x", "dev": str(why)[:120]})
        return row
    ops = got["chain_ops"]
    inv6 = all(o in v6ops for o in ops)
    pt = F.render_tactic(d, got["family"], got.get("route_payload"), ops)
    row.update({"st": "state1", "rs": "route_" + str(got.get("route") or got["family"]),
                "R": (got.get("route") or got["family"]), "ops": ",".join(ops),
                "ops_v6": bool(inv6), "fam": got["family"],
                "n_hyps": d.get("n_hyps", 0), "n_vars": d.get("n_vars", 0),
                "chain_len": len(ops), "kind": "chain",
                "pt_md5": hashlib.md5(pt.encode()).hexdigest()})
    row["_pt"] = pt
    return row


def sp_nsimplify(e):
    import sympy as sp
    return sp.nsimplify(e)


V6OPS = set()


def _worker(args):
    i, rec = args
    try:
        return analyse(rec, V6OPS)
    except Exception as e:                                              # noqa: BLE001
        return {"idx": rec.get("idx"), "name": rec.get("name"), "sub": rec.get("subset"),
                "O": None, "C": None, "H": None, "st": "error",
                "rs": "exc:" + type(e).__name__ + ":" + repr(e)[:50], "kind": "exc",
                "R": None, "ops": None, "ops_v6": None, "pt_md5": None}


def _init(v6path):
    global V6OPS
    d = json.load(open(v6path, encoding="utf-8"))
    V6OPS = set()
    for k in ("ops_ccp", "ops_bh", "ops_g24", "ops_g41", "ops_g32", "ops_gen32",
              "ops_g42", "op_index_map_ccp"):
        v = d.get(k)
        if isinstance(v, dict):
            V6OPS |= set(v.keys())
    im = d.get("init_map") or {}
    for k, v in im.items():
        if isinstance(v, dict) and v.get("name"):
            V6OPS.add(v["name"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(G44, "data", "LWB_ALL.jsonl"))
    ap.add_argument("--out", default=os.path.join(G44, "out", "GEN44_CENSUS_V3.json"))
    ap.add_argument("--rows", default=os.path.join(G44, "out", "GEN44_CENSUS_ROWS_V3.jsonl"))
    ap.add_argument("--struct", default=os.path.join(G44, "out", "GEN44_STRUCT_V3.jsonl"))
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--v6", default=V6_PATH)
    ap.add_argument("--v1", default=os.path.join(G44, "out", "GEN44_CENSUS.json"))
    a = ap.parse_args()
    t0 = time.time()
    recs = []
    with open(a.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
                if a.limit and len(recs) >= a.limit:
                    break
    print("census v3 over %d rows (jobs=%d)" % (len(recs), a.jobs), flush=True)
    import multiprocessing as mp
    rows = []
    with mp.Pool(a.jobs, initializer=_init, initargs=(a.v6,)) as pool:
        for k, r in enumerate(pool.imap(_worker, ((i, x) for i, x in enumerate(recs)),
                                        chunksize=64)):
            rows.append(r)
            if (k + 1) % 40000 == 0:
                print("  [%d/%d] %.1fs" % (k + 1, len(recs), time.time() - t0), flush=True)
    with open(a.rows, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "_pt"},
                               ensure_ascii=False) + "\n")
    structs = [r for r in rows if r["st"] == "state1"]
    with open(a.struct, "w", encoding="utf-8") as f:
        for r in structs:
            f.write(json.dumps({k: v for k, v in r.items() if k != "_pt"},
                               ensure_ascii=False) + "\n")

    def hist(key, sub=None):
        src = [r for r in rows if sub is None or r["st"] == sub]
        return dict(collections.Counter(str(r.get(key)) for r in src).most_common())

    by_sub = {}
    for s in ("lean_workbook", "lean_workbook_plus"):
        ss = [r for r in rows if r["sub"] == s]
        by_sub[s] = {"n": len(ss),
                     "state1": sum(1 for r in ss if r["st"] == "state1"),
                     "state2": sum(1 for r in ss if r["st"] == "state2"),
                     "state3": sum(1 for r in ss if r["st"] == "state3"),
                     "state1_rate": round(sum(1 for r in ss if r["st"] == "state1")
                                          / max(1, len(ss)), 5)}
    rng = {}
    for r in rows:
        c = r.get("rs") or "?"
        if r["st"] == "state2" and r.get("kind"):
            c = "R_none:has_vars/" + r["kind"]
        elif r["st"] == "state3" and r.get("kind") and not str(c).startswith("O_nonnum"):
            c = str(c).split(":")[0] + ":" + str(r["kind"])
        dd = rng.setdefault(c, {"n": 0, "min": None, "max": None})
        dd["n"] += 1
        i = r["idx"]
        dd["min"] = i if dd["min"] is None else min(dd["min"], i)
        dd["max"] = i if dd["max"] is None else max(dd["max"], i)
    v6bad = [r["name"] for r in structs if r.get("ops_v6") is False]
    ref = None
    if os.path.exists(a.v1):
        v1 = json.load(open(a.v1, encoding="utf-8"))
        ref = {"v1_n_struct": v1["n_struct"], "v3_n_struct": len(structs),
               "same_n_struct": v1["n_struct"] == len(structs),
               "v1_state_hist": v1["state_hist"], "v3_state_hist": hist("st")}
        ref["same_state_hist"] = (v1["state_hist"] == ref["v3_state_hist"])
    rep = {"schema": SCHEMA, "created": datetime.now(timezone.utc).astimezone()
           .isoformat(timespec="seconds"), "input": a.input, "n": len(rows),
           "state_hist": hist("st"), "by_subset": by_sub,
           "reject_hist": hist("rs"), "kind_hist_state2": hist("kind", "state2"),
           "kind_hist_state3": hist("kind", "state3"),
           "route_hist": dict(collections.Counter(r["R"] for r in rows if r["R"])
                              .most_common()),
           "O_hist": hist("O"), "C_hist": hist("C"), "H_hist": hist("H"),
           "chainlen_hist": dict(collections.Counter(
               r.get("chain_len") for r in rows if r.get("chain_len")).most_common()),
           "class_idx_range": rng, "v6_violations": len(v6bad),
           "gate_M6_pass": len(v6bad) == 0, "n_struct": len(structs),
           "v1_v3_reconcile": ref,
           "gate_M1": {"threshold": ">=500 or >=15%", "n_struct": len(structs),
                       "rate": round(len(structs) / max(1, len(rows)), 5),
                       "M1_pass": (len(structs) >= 500 or
                                   len(structs) / max(1, len(rows)) >= 0.15)},
           "wall_s": round(time.time() - t0, 1), "write_face": "scratch/ccp/gen44/**"}
    json.dump(rep, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: rep[k] for k in ("n", "state_hist", "by_subset", "n_struct",
                                          "gate_M1", "v6_violations",
                                          "v1_v3_reconcile", "wall_s")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
