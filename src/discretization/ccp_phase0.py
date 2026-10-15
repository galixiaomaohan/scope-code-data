# -*- coding: utf-8 -*-
# ccp_phase0.py -- CCP coverage-without-contamination program, Phase 0 analyzer (constitution change 35)
# Hard firewall: --design reads only the Valid split (256 problems; taxonomy design / tuning must
#               not read Test); --annotate does post-hoc annotation of Test 243 (Test reads are
#               allowed there; the output serves only the coverage mapping).
# Three-layer failure decomposition:
#   (1) translation layer (excluded symbols / outside the grammar / no relation symbol)
#   (2) vocabulary layer (inside the grammar but not substitution-solvable; needs operations
#       beyond the operator interface)
#   (3) navigation layer (the 19 in-domain problems: chain not formed / never judged).
# Reads probes/ (read-only import of lean_exam) and scratch/xcmp/runs/; writes only scratch/ccp/**.
# Single CPU process.
import argparse
import importlib.machinery
import importlib.util
import json
import os
import re
import sys
from collections import Counter, defaultdict

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
os.chdir(RES)
sys.path.insert(0, os.environ.get(
    "SCOPE_SUPEROPERATOR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "device")))


def _load(name, path):
    ld = importlib.machinery.SourceFileLoader(name, path)
    sp = importlib.util.spec_from_loader(name, ld)
    m = importlib.util.module_from_spec(sp)
    ld.exec_module(m)
    return m


XC = _load("xcmp_common_ccp", os.path.join(RES, "scratch/xcmp/xcmp_common.py"))
LE = XC.LE

# ---------------------------------------------------------------- classifier (designed and frozen on Valid)
DOMAIN_RULES = [
    ("numbertheory", [r"∣", r"dvd", r"Nat.Prime", r"prime", r"gcd", r"lcm", r"\bmod\b", r"≡ \[", r"Zsqrtd",
                      r"factorial", r"!\b", r"coprime", r"divisors", r"totient", r"order"]),
    ("combinatorics", [r"∑", r"Finset", r"choose", r"\bbinomial\b", r"card", r"cardinality", r"Pigeonhole",
                       r"injection", r"surjection", r"bipartite", r"choose\b", r"Set.count"]),
    ("analysis", [r"deriv", r"intervalIntegral", r"∫", r"limit", r"tendsto", r"Continuous", r"convex",
                  r"HasDeriv", r"series", r"exp\b", r"log\b", r"ln\b", r"sin\b", r"cos\b", r"tan\b", r"sqrt.*sqrt"]),
    ("geometry", [r"angle", r"triangle", r"circle", r"dist\b", r"IsConvex", r"perpendicular", r"parallel",
                  r"Euclidean", r"VAdd.*point", r"point", r"line.*intersect", r"chord", r"tangent"]),
    ("probability", [r"prob", r"expectation", r"bayes", r"distribution", r"random", r"coin", r"dice", r"die",
                     r"uniform"]),
    ("inequality", [r"≤|≥|<|>"]),  # fallback: the conclusion has an order relation and is not a pure equality
]
DOMAIN_ORDER = ["numbertheory", "combinatorics", "analysis", "geometry", "probability", "inequality", "algebra"]

TECH_RULES = [
    ("decision_arith", [r"norm_num", r"decide", r"omega", r"native_decide"]),   # placeholder: invisible at the statement level; proxied by structure
    ("substitution_compute", [r"= [0-9]", r"= \(", r"h[₀-₉]*\s*:.*="]),          # substitution-compute proxy
    ("divisibility_modular", [r"∣", r"\bmod\b", r"≡ \[", r"coprime", r"prime"]),
    ("summation_counting", [r"∑", r"Finset", r"choose", r"card", r"inclusion"]),
    ("bounding_squeeze", [r"≤.*≤", r"<.*<", r"max\b", r"min\b", r"abs", r"|"]),
    ("case_split", [r"∨", r"Or ", r"even.*odd|odd.*even", r"_cases"]),
    ("induction_recursion", [r"∀ n", r"∀ k", r"forall", r"recursive", r"fib", r"seq"]),
    ("trig_eval", [r"sin|cos|tan|cot|sec|csc"]),
]

STRUCT_RULES = [
    ("solve_value", [r"= [0-9]", r"= -[0-9]", r"answer\("]),
    ("prove_identity", [r"= \(", r"= [a-z]", r".* = .*[+*/^].*"]),
    ("prove_divisibility", [r"∣"]),
    ("prove_inequality", [r"≤|≥|<|>"]),
    ("prove_equality_bool", [r"= true", r"= false", r"Iff", r"↔"]),
    ("count_or_exist", [r"∃", r"Nonempty", r"card", r"≥ 1", r"> 0"]),
]


def _match(rules, text, concl_only):
    hay = concl_only if concl_only else text
    for name, pats in rules:
        for pat in pats:
            if re.search(pat, hay):
                return name
    return "other"


def classify(p):
    """Three-axis classification (designed and frozen on Valid; surface features only, no learning)."""
    text = (p.binders or "") + " " + p.concl
    concl = p.concl or ""
    src = "amc/aime" if re.match(r"(amc|aime|imo)", p.name) else ("mathd" if p.name.startswith("mathd") else "other")
    dom = _match(DOMAIN_RULES, text, concl)
    if dom == "inequality":  # the order-relation fallback fires only when no other domain matched
        pass
    elif dom == "other":
        dom = "algebra"
    tech = _match(TECH_RULES, text, False)
    struct = _match(STRUCT_RULES, text, concl)
    return {"domain": dom, "tech": tech, "struct": struct, "source": src}


def style_template(concl):
    """Style spectrum: canonicalized conclusion template (digits -> #, variables -> v) used for
    course style anchoring."""
    t = re.sub(r"\d+", "#", concl or "")
    t = re.sub(r"\b[a-zA-Z][a-zA-Z₀-₉'\d]*\b", "v", t)
    return re.sub(r"\s+", "", t)[:120]


def load_split(split):
    return [p for p in LE.read_minif2f(XC.MINIF2F) if p.split == split]


# ---------------------------------------------------------------- three-layer decomposition
def layer_decompose(names_test):
    """(1) translation layer: screen() why in the first three classes; (2) vocabulary layer:
    variables not eliminable (inside the grammar); (3) navigation layer: the 19 unsolved
    in-domain problems."""
    recs = {}
    for p in names_test:
        sc = LE.screen(p)
        why = sc.get("why") or []
        if sc["class"] != "REJECT":
            layer = "in_domain"
        elif any(("排除符号" in w or "文法外" in w or "无关系符" in w or "无结论" in w) for w in why):
            layer = "L1_translation"
        else:
            layer = "L2_vocabulary"
        recs[p.name] = {"class": sc["class"], "why": why, "layer": layer}
    # (3) navigation layer: in-domain problems joined with the v32 judging results
    # (solved in either state => solved; unsolved in both => L3)
    solved_any, nav_detail = set(), {}
    for tag in ("v32_off", "v32_on"):
        rp = os.path.join(RES, "scratch/xcmp/runs/%s_records.jsonl" % tag)
        if not os.path.exists(rp):
            continue
        solved_this = set()
        for line in open(rp, encoding="utf-8"):
            r = json.loads(line)
            if any((a or {}).get("certified") for a in r["attempts"].values()):
                solved_this.add(r["problem"])
        solved_any |= solved_this
        nav_detail[tag] = len(solved_this)
    for name, rec in recs.items():
        if rec["layer"] == "in_domain" and name not in solved_any:
            rec["layer"] = "L3_navigation"
        elif rec["layer"] == "in_domain":
            rec["layer"] = "solved"
    return recs, nav_detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", action="store_true", help="Valid-only taxonomy design stage")
    ap.add_argument("--annotate", action="store_true", help="post-hoc Test annotation + three-layer decomposition")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.design:
        valid = load_split("Valid")
        assert len(valid) == 256
        cnt = {"domain": Counter(), "tech": Counter(), "struct": Counter(),
               "dom_struct": Counter(), "source": Counter()}
        examples = defaultdict(list)
        styles = Counter()
        for p in valid:
            c = classify(p)
            for k in ("domain", "tech", "struct", "source"):
                cnt[k][c[k]] += 1
            cnt["dom_struct"][(c["domain"], c["struct"])] += 1
            if len(examples[(c["domain"], c["struct"])]) < 3:
                examples[(c["domain"], c["struct"])].append(p.name)
            styles[style_template(p.concl)] += 1
        rep = {
            "split": "Valid(256)-设计段-未读Test",
            "counts": {k: dict(v) for k, v in cnt.items() if k != "dom_struct"},
            "dom_struct": {"%s|%s" % k: v for k, v in cnt["dom_struct"].items()},
            "examples": {"%s|%s" % k: v for k, v in examples.items()},
            "style_templates_top30": styles.most_common(30),
            "style_templates_distinct": len(styles),
        }
        out = a.out or "scratch/ccp/TAXONOMY_VALID.json"
        json.dump(rep, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("[design] Valid domain distribution:", dict(cnt["domain"]))
        print("[design] structure distribution:", dict(cnt["struct"]))
        print("[design] style templates distinct=%d" % len(styles))
        print("[design] wrote %s" % out)

    if a.annotate:
        test = load_split("Test")
        assert len(test) == 243
        dec, nav_detail = layer_decompose(test)
        agg = Counter(r["layer"] for r in dec.values())
        dom_layer = Counter()
        dom_cls = Counter()
        for p in test:
            c = classify(p)
            dec[p.name]["cls"] = c
            dom_layer[(c["domain"], dec[p.name]["layer"])] += 1
            dom_cls[c["domain"]] += 1
        rep = {
            "split": "Test(243)-事后标注段",
            "nav_solved_per_state": nav_detail,
            "layer_counts": dict(agg),
            "domain_counts": dict(dom_cls),
            "domain_x_layer": {"%s|%s" % k: v for k, v in dom_layer.items()},
            "per_problem": dec,
        }
        out = a.out or "scratch/ccp/TEST_COVERAGE.json"
        json.dump(rep, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("[annotate] three-layer decomposition:", dict(agg))
        print("[annotate] domain distribution:", dict(dom_cls))
        print("[annotate] wrote %s" % out)


if __name__ == "__main__":
    main()
