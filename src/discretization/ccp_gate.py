# -*- coding: utf-8 -*-
# ccp_gate.py -- CCP contamination gate (R1-R5) + positive control + seed registry
# (advance team gen12, constitution change 58)
#
# Hard firewall: this module reads miniF2F Test/Valid **statement text for mechanical comparison
#   only** (hash / normalization / n-gram coverage) and emits **hit counts only**; no Test
#   statement fragment may enter logs / reports / generated artifacts.
#   (change 35 item 2: all training material must show zero hits against Test+Valid on both the
#   exact and near-duplicate routes; any hit rejects the whole batch.)
# Convention inherited from scratch/xcmp/xcmp_neardup_audit.py v2 (windows drawn from both
#   sources concl+head; bidirectional coverage; short-conclusion trap fix).
# Performance: n-gram inverted index + feature inverted index => exact scoring only against
#   reference problems sharing at least one n-gram / feature (complete and fast).
import collections
import hashlib
import json
import os
import re
import unicodedata

import ccp_lean as L

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = RES + "/scratch/ccp"
MINIF2F = os.environ.get("SCOPE_MINIF2F", os.path.join(RES, "lean", "miniF2F-main"))

_WS = re.compile(r"\s+")
_SYM_MAP = [("\u2264", "<="), ("\u2265", ">="), ("\u2260", "!="), ("\u2223", "|"),
            ("\u2225", "||"), ("\u00d7", "*"), ("\u2212", "-"), ("\u2013", "-"),
            ("\u2014", "-"), ("\u2018", "'"), ("\u2019", "'"), ("\u201c", '"'),
            ("\u201d", '"'), ("\u00b7", "*"), ("\u2044", "/")]
_NUMF = re.compile(r"(?<![0-9])(\d{3,})(?![0-9])")

REJECT = 0.50
REVIEW = 0.30
NG = 10


def norm(s):
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    for a, b in _SYM_MAP:
        s = s.replace(a, b)
    return _WS.sub("", s).lower()


def ngrams(s, n=NG):
    if len(s) < n:
        return set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def feats(s):
    return {int(x) for x in _NUMF.findall(s or "")}


def load_refs():
    """Reference corpus = miniF2F Test(243) + Valid(256); returns [(split, name, nhead, nconcl, nraw)].
    **Mechanical comparison only**; callers must not print its text."""
    import importlib.machinery
    import importlib.util
    ld = importlib.machinery.SourceFileLoader(
        "le_ccpg", os.path.join(RES, "probes", "lean_exam.py.txt"))
    sp = importlib.util.spec_from_loader("le_ccpg", ld)
    le = importlib.util.module_from_spec(sp)
    ld.exec_module(le)
    out = []
    for p in le.read_minif2f(MINIF2F):
        out.append((p.split, p.name, norm(p.head), norm(p.concl), norm(p.raw)))
    return out


class Gate(object):
    """R1 exact md5 / R2 normalized containment / R3 problem name / R4 n-gram bidirectional
    coverage / R5 feature-count fingerprint."""

    def __init__(self, refs):
        self.refs = refs
        self.md5map = {}
        self.entry = []                      # (split, name, nconcl, nhead, ft)
        for i, (split, name, head, concl, raw) in enumerate(refs):
            rec = (split, name, concl, head, feats(concl) | feats(head))
            self.entry.append(rec)
            for t in (concl, head, raw):
                if t:
                    self.md5map.setdefault(hashlib.md5(t.encode("utf-8")).hexdigest(), name)
        self.inv_ng = collections.defaultdict(set)
        self.inv_ft = collections.defaultdict(set)
        for i, (split, name, concl, head, ft) in enumerate(self.entry):
            for g in ngrams(concl) | ngrams(head):
                self.inv_ng[g].add(i)
            for f in ft:
                if f >= 100:
                    self.inv_ft[f].add(i)

    def _cand_refs(self, cands, cand_ft):
        s = set()
        for c in cands:
            for g in ngrams(c):
                v = self.inv_ng.get(g)
                if v:
                    s |= v
        for f in cand_ft:
            if f >= 100:
                v = self.inv_ft.get(f)
                if v:
                    s |= v
        return s

    def judge(self, text, head=None, concl=None):
        """Returns the verdict dict; counts and problem names only (names are metadata, not
        statement content)."""
        cands = [c for c in (norm(text), norm(concl) if concl else "",
                             norm(head) if head else "") if len(c) >= 12]
        if not cands:
            cands = [norm(text) or " "]
        hit = {"R1": [], "R2": [], "R3": []}
        r4 = []
        r4rev = []
        r5 = []
        cand_md5 = {hashlib.md5(c.encode("utf-8")).hexdigest() for c in cands}
        for h in cand_md5:
            if h in self.md5map:
                hit["R1"].append(self.md5map[h])
        cand_ft = set()
        for c in cands:
            cand_ft |= feats(c)
        # R4 coverage ratio is computed only on windows with len >= 24 (short-window ratios are
        # unstable: a 12-char window has only 3 10-grams, so 1 hit gives 0.333 -- same-origin
        # rule as R2's 24-char reject floor)
        long_c = [c for c in cands if len(c) >= 24] or ([max(cands, key=len)] if cands else [])
        idxs = self._cand_refs(cands, cand_ft)
        for i in idxs:
            split, name, rconcl, rhead, rft = self.entry[i]
            if re.search(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(name), text or ""):
                hit["R3"].append(name)
            done2 = False
            for c in cands:
                for r in (rconcl, rhead):
                    if not r:
                        continue
                    if len(c) >= 24 and (c in r or r in c):
                        hit["R2"].append(name)
                        done2 = True
                        break
                if done2:
                    break
            best = 0.0
            for c in long_c:
                cng = ngrams(c)
                if not cng:
                    continue
                for rng in (ngrams(rconcl), ngrams(rhead)):
                    if not rng:
                        continue
                    inter = len(cng & rng)
                    if inter:
                        best = max(best, inter / float(len(cng)), inter / float(len(rng)))
            if best >= REJECT:
                r4.append((name, round(best, 3)))
            elif best >= REVIEW:
                r4rev.append((name, round(best, 3)))
            big = {x for x in (cand_ft & rft) if x >= 100}
            if len(big) >= 3:
                r5.append(name)
        verdict = "REJECT" if (hit["R1"] or hit["R2"] or hit["R3"] or r4 or r5) else (
            "REVIEW" if r4rev else "PASS")
        return {"verdict": verdict,
                "counts": {"R1": len(hit["R1"]), "R2": len(hit["R2"]), "R3": len(hit["R3"]),
                           "R4": len(r4), "R5": len(r5), "R4rev": len(r4rev)},
                "n_review": len(r4rev),
                "max_R4": max([b for (_, b) in r4] + [b for (_, b) in r4rev] + [0.0])}

    def positive_control(self, idxs=(0, 100, 200)):
        """Positive control: reference text verbatim/perturbed as candidate must be caught by
        R1/R2/R4 (a failing control means the gate is invalid)."""
        out = {"n": 0, "caught": 0, "detail": []}
        for i in idxs:
            i = i % len(self.entry)
            split, name, concl, head, ft = self.entry[i]
            if not concl:
                continue
            for mut, tag in ((concl, "verbatim"),
                             ("  " + concl.replace("*", " * ") + "  ", "spacing")):
                r = self.judge(mut, head=head, concl=mut)
                out["n"] += 1
                if r["verdict"] == "REJECT":
                    out["caught"] += 1
                out["detail"].append({"i": i, "mut": tag, "verdict": r["verdict"],
                                      "max_R4": r["max_R4"], "counts": r["counts"]})
        return out


def seed_registry(cfg):
    """Generator seed and template-version registry (change 58 item 3, falsifiability: anyone
    can recompute that this course batch != Test)."""
    return {"schema": "ccp-seed-registry/1",
            "master_seed": cfg["master_seed"],
            "seed_formula": "md5('ccp|<master>|<cls>|<family>|<seq>')[:8] -> int (每题独立)",
            "families": cfg["families"],
            "row_mix": cfg["row_mix"],
            "risk_menus": cfg.get("risk_menus"),
            "recompute": cfg.get("recompute"),
            "template_files": cfg.get("template_files")}
