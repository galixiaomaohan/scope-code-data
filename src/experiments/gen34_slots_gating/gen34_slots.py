# -*- coding: utf-8 -*-
# gen34_slots.py -- integer slot annotation (V34_DESIGN §1): device re-render + bytewise
#                  assertion + token alignment
#
# Position: mark every **content integer** in proof_text as a slot carrying a device source
#   label (src).
#   Annotation method = **constructive re-render + bytewise assertion** (not a heuristic):
#   re-render the presented-step tactic text of each row from its own steps[].args through
#   the gen33 device evaluator (gen33_device.dev_args + gen32_ops.call) and assert
#   recon == gold; a passing assertion means every recorded device-value character span of
#   the re-render is that integer position's device source.
#
# Three conventions (V34_DESIGN §1.2):
#   naive        all [0-9]+ (including digits inside identifiers g0/h1/lean4) -- bitwise
#                comparable with gen33 §3b
#   content      [0-9]+ whose neighbouring characters are not letters/underscores -- the
#                main convention of this batch
#   attrib       content literals traceable to a device quantity -- the coverage numerator
#
# Read-only: gen32 corpus / gen33 sidecars and device files / gen13 vocab chain;
# write: scratch/ccp/gen34/** only.
import argparse
import collections
import json
import os
import re
import sys
import time
from fractions import Fraction

import gen34_common as C

G34, G33, G32, BH = C.G34, C.G33, C.G32, C.BH
OUT = os.path.join(G34, "out")
NUM = re.compile(r"[0-9]+")
ALNUM = re.compile(r"[A-Za-z0-9_]")

if BH not in sys.path:
    sys.path.insert(0, BH)
import bh_poly as BP                                                    # noqa: E402

DEV = None          # deferred import (needs sys.path to be ready)
OPS = None

# ---- slot classes (V34_DESIGN §1.3): 7 classes -> kind 0..6, n_slot_kinds = 8 ----
SLOT_KINDS = ("poly_coef", "poly_exp", "ell_coef", "ell_shift", "mult",
              "tail_const", "template")
NKIND = len(SLOT_KINDS)
KIND_ID = {k: i for i, k in enumerate(SLOT_KINDS)}


def kind_of_src(src):
    """src string -> kind id (V34_DESIGN §1.3 table)."""
    if src.startswith("tmpl:"):
        return KIND_ID["template"]
    if ".P@" in src:
        return KIND_ID["poly_coef"]
    if ".P^" in src:
        return KIND_ID["poly_exp"]
    if ".ell.c@" in src:
        return KIND_ID["ell_coef"]
    if ".ell.sh" in src:
        return KIND_ID["ell_shift"]
    if ".mult" in src:
        return KIND_ID["mult"]
    if ".rho" in src or ".sq" in src:
        return KIND_ID["tail_const"]
    return KIND_ID["template"]


# ---------------------------------------------------------------------------
# Offset-tracking buffer (for re-render: each text piece may carry a device source)
# ---------------------------------------------------------------------------

class Buf(object):
    def __init__(self):
        self.parts = []
        self.n = 0
        self.spans = []

    def add(self, t, src=None):
        if not t:
            return self
        a = self.n
        self.parts.append(t)
        self.n += len(t)
        if src is not None:
            self.spans.append({"cs": a, "ce": a + len(t), "src": src, "val": t})
        return self

    def text(self):
        return "".join(self.parts)


def mono_str(m):
    return ",".join(str(e) for e in m)


# ---------------------------------------------------------------------------
# 1) device polynomial re-render (convention = bh_poly.render_poly verbatim; add sources)
# ---------------------------------------------------------------------------

def render_poly_instr(p, vars_, sfx, tmpl0="tmpl:poly_zero"):
    """-> (text, spans). text must be bytewise equal to bh_poly.render_poly(p, vars_)."""
    items = sorted([(m, c) for m, c in p.items() if c != 0],
                   key=lambda kv: (-sum(kv[0]), kv[0]))
    b = Buf()
    if not items:
        b.add("0", tmpl0)
        return b.text(), b.spans
    for i, (m, c) in enumerate(items):
        c = Fraction(c)
        neg = c < 0
        a = abs(c)
        if i == 0:
            b.add("- " if neg else None, None if not neg else sfx(m, "sign"))
        else:
            b.add(" - " if neg else " + ", sfx(m, "sign"))
        mono = [(j, e) for j, e in enumerate(m) if e != 0]
        if not mono:
            b.add(str(a), sfx(m, "coef"))
            continue
        if a != 1:
            b.add(str(a), sfx(m, "coef"))
            b.add(" * ", None)
        pieces = []
        for j, e in mono:
            pieces.append((vars_[j], e))
        for t, (v, e) in enumerate(pieces):
            if t:
                b.add(" * ", None)
            b.add(v, None)
            if e != 1:
                b.add(" ^ ", None)
                b.add(str(e), sfx(m, "exp"))
    return b.text(), b.spans


def render_poly_sfx(k, op, nv=1):
    def sfx(m, what):
        if what == "coef":
            return "s%d.%s.P@%s" % (k, op, mono_str(m))
        if what == "exp":
            return "s%d.%s.P^%s" % (k, op, mono_str(m))
        return "s%d.%s.P@%s.sign" % (k, op, mono_str(m))
    return sfx


# ---------------------------------------------------------------------------
# 2) single-step tactic re-render (template = gen32_ops verbatim; add sources)
# ---------------------------------------------------------------------------

def ell_text_instr(ell, k, op):
    """Convention = gen32_ops.ell_text verbatim; adds sources (ell.c@<var> / ell.sh)."""
    b = Buf()
    parts = []
    for v, cf in ell["coefs"].items():
        cf = Fraction(cf)
        if cf == 0:
            continue
        parts.append((("+" if cf > 0 else "-"), v, cf))
    sh = Fraction(ell.get("shift", 0))
    if sh != 0 or not parts:
        parts.append((("-" if sh > 0 else "+"), None, sh))
    b.add("(", None)
    for i, (sg, v, cf) in enumerate(parts):
        if i == 0:
            if sg == "-":
                b.add("- ", None)
        else:
            b.add(" - " if sg == "-" else " + ", None)
        if v is None:
            b.add(str(int(abs(sh))), "s%d.%s.ell.sh" % (k, op))
        else:
            if abs(cf) != 1:
                b.add("%d" % int(abs(cf)), "s%d.%s.ell.c@%s" % (k, op, v))
                b.add(" * ", None)
            b.add(v, None)
    b.add(")", None)
    return b.text(), b.spans


def solve_block_instr(sym, closed_txt, closed_spans, h0, hstep, ref, intro, off):
    """Convention = gen32_ops.solve_block verbatim (closed text carries sources from the
    caller)."""
    b = Buf()
    if intro:
        b.add("intro n\n", None)
    head = "have h%s : \u2200 n, %s n = " % (sym, sym)
    b.parts.append(head)
    b.n += len(head)
    off = b.n
    b.parts.append(closed_txt)
    b.n += len(closed_txt)
    for s in closed_spans:
        b.spans.append({"cs": off + s["cs"], "ce": off + s["ce"], "src": s["src"],
                        "val": s["val"]})
    b.add(" := by\n", None)
    b.add("  intro n\n", None)
    b.add("  induction n with\n", None)
    b.add("  | zero => norm_num [%s]\n" % h0, None)
    b.add("  | succ n ih =>\n", None)
    b.add("    rw [%s, ih%s]\n" % (hstep, "" if not ref else ", %s n" % ref), None)
    b.add("    push_cast\n", None)
    b.add("    ring", None)
    return b.text(), b.spans


def step_instr(op, args, res, k, tmpl_src="tmpl"):
    """Re-render step k's tactic -> (text, spans). text must == res.tactic."""
    v = res.value or {}
    if op == "gen32_nest_solve":
        Pd = BP.load(v["P"])
        ct, cs = render_poly_instr(Pd, ["n"], render_poly_sfx(k, op))
        return solve_block_instr(args["sym"], ct, cs, args["h0"], args["hstep"],
                                 args.get("ref"), bool(args.get("intro", False)), 0)
    if op == "gen32_nest_close":
        b = Buf()
        b.add("rw [%s]" % args.get("close_lemma", "hclose"), None)
        if args.get("need_ring", True):
            b.add("\nring", None)
        return b.text(), b.spans
    if op in ("gen32_sos_block", "gen32_sos_cross"):
        mult = Fraction(args["mult"])
        et, es = ell_text_instr(args["ell"], k, op)
        b = Buf()
        b.add("have h%d : (" % int(args["hint_idx"]), None)
        b.add("0", tmpl_src + ":sos_pos_zero")
        b.add(":\u211a) \u2264 ", None)
        b.add("%d" % int(mult), "s%d.%s.mult" % (k, op))
        b.add(" * ", None)
        b.add(et, None)
        for s in es:
            b.spans.append({"cs": b.n - len(et) + s["cs"], "ce": b.n - len(et) + s["ce"],
                            "src": s["src"], "val": s["val"]})
        b.parts.append(" ^ ")
        b.n += 3
        b.add("2", tmpl_src + ":sq_exp")
        b.add(" := by positivity", None)
        return b.text(), b.spans
    if op in ("gen32_sos_tail_zero", "gen32_sos_tail_square", "gen32_sos_tail_other"):
        hs = list(args.get("hints") or [])
        b = Buf()
        if op == "gen32_sos_tail_zero":
            b.add("nlinarith only [%s]" % ", ".join(hs), None)
        elif op == "gen32_sos_tail_square":
            s = int(args["sq"])
            b.add("have hrho : (", None)
            b.add("0", tmpl_src + ":sos_pos_zero")
            b.add(":\u211a) \u2264 (", None)
            b.add("%d" % s, "s%d.%s.sq" % (k, op))
            b.add(" : \u211a) ^ ", None)
            b.add("2", tmpl_src + ":sq_exp")
            b.add(" := sq_nonneg (", None)
            b.add("%d" % s, "s%d.%s.sq" % (k, op))
            b.add(" : \u211a)\n", None)
            b.add("nlinarith only [%s, hrho]" % ", ".join(hs), None)
        else:
            b.add("have hrho : (", None)
            b.add("0", tmpl_src + ":sos_pos_zero")
            b.add(":\u211a) \u2264 (", None)
            b.add("%s" % args["rho"], "s%d.%s.rho" % (k, op))
            b.add(" : \u211a) := by norm_num\n", None)
            b.add("nlinarith only [%s, hrho]" % ", ".join(hs), None)
        return b.text(), b.spans
    raise KeyError("unregistered op re-render template: %s" % op)


#: presented steps (V34_DESIGN §1.1): allowlist of ops that enter proof_text
PRESENT_OPS = ("gen32_nest_solve", "gen32_nest_close",
               "gen32_sos_block", "gen32_sos_cross",
               "gen32_sos_tail_zero", "gen32_sos_tail_square", "gen32_sos_tail_other")

#: gen32_n1fix rewrite (source file `EXAM_DEEP_CANDIDATE.json` md5 780dbb5e = **pre-fix**
#:  text; `DEEP_CORPUS_V1.jsonl` = **post-fix** text). The device's live tactic is always the
#:  post-fix one => during annotation the pre-fix text gets the same-form normalization
#:  (replace rule = gen32_n1fix.py L30 verbatim), with the normalization hit count
#:  registered (never silent).
N1_OLD, N1_NEW = "nlinarith [", "nlinarith only ["


def apply_n1fix(text, spans, undo=False):
    """-> (new_text, new_spans, n_rep). Post-fix text has no "nlinarith [" substring =>
    the forward direction is naturally idempotent.

    undo=True: reverse direction (post-fix -> pre-fix). Reason (measured in this batch):
    `DEEP_CORPUS_V1.jsonl` holds **post-fix** text while `EXAM_DEEP_CANDIDATE.json`
    (md5 780dbb5e) holds **pre-fix** text => the exam side needs the reverse normalization
    to match the gold byte for byte.
    """
    old, new = (N1_NEW, N1_OLD) if undo else (N1_OLD, N1_NEW)
    idxs = [m.start() for m in re.finditer(re.escape(old), text)]
    if not idxs:
        return text, spans, 0
    ins = len(new) - len(old)
    cut = len("nlinarith ")          # common prefix of both sides
    step = len(old)                  # original characters skipped (includes "[", not cut!)

    def shift(x):
        return x + ins * sum(1 for p in idxs if p + step <= x)

    parts, prev = [], 0
    for p in idxs:
        parts.append(text[prev:p + cut])
        parts.append(new[cut:])
        prev = p + step
    parts.append(text[prev:])
    ns = [{"cs": shift(s["cs"]), "ce": shift(s["ce"]), "src": s["src"], "val": s["val"]}
          for s in spans]
    return "".join(parts), ns, len(idxs)


def row_proof_instr(row, gold=None, log_bad=None):
    """Whole-row device re-render -> (text, spans, info).

    info: {"recon_eq": bool, "n_step": n, "n_present": m, "dev_err": [...],
           "op_mismatch": [...], "n1_rev": 0|1}
    When gold is not None: if the forward normalization misses, the **reverse
    normalization** is tried automatically (exam artifacts carry pre-fix text; see
    apply_n1fix).
    """
    b = Buf()
    n_present = 0
    dev_err, mism = [], []
    for k in range(len(row["steps"])):
        s = row["steps"][k]
        op = s["op"]
        if op not in PRESENT_OPS:
            continue                        # device_only steps never enter proof_text
                                            # (§1.1); no live evaluation needed
        try:
            a = DEV.dev_args(row, k)
        except Exception as e:                                       # noqa: BLE001
            dev_err.append({"k": k, "op": op, "err": "DEV_ARGS:%r" % (e,)})
            continue
        r = OPS.call(op, a)
        if not getattr(r, "ok", False) or r.err:
            dev_err.append({"k": k, "op": op, "err": "DEV_ERR:%s" % (r.err or r.cert_detail)})
            continue
        try:
            t, sp = step_instr(op, a, r, k)
        except Exception as e:                                       # noqa: BLE001
            dev_err.append({"k": k, "op": op, "err": "INSTR:%r" % (e,)})
            continue
        if t != r.tactic:
            mism.append({"k": k, "op": op, "got": t[:120], "want": (r.tactic or "")[:120]})
        if n_present:
            b.add("\n", None)
            sp = [dict(x, cs=x["cs"] + b.n, ce=x["ce"] + b.n) for x in sp]
            for x in sp:
                b.spans.append(x)
            b.add(t, None)
        else:
            b.add(t, None)
            b.spans.extend(sp)
        n_present += 1
    txt, spans, n_fix = apply_n1fix(b.text(), b.spans)
    n1rev = 0
    if gold is not None and txt != gold:
        t2, s2, n2 = apply_n1fix(txt, spans, undo=True)
        if t2 == gold:
            txt, spans, n1rev = t2, s2, 1
    return txt, spans, {"n_step": len(row["steps"]), "n_present": n_present,
                        "n_n1fix": n_fix, "n1_rev": n1rev,
                        "dev_err": dev_err, "op_mismatch": mism}


# ---------------------------------------------------------------------------
# 3) literal conventions
# ---------------------------------------------------------------------------

def naive_lits(text):
    return [(m.start(), m.end()) for m in NUM.finditer(text)]


def content_lits(text):
    """[0-9]+ whose neighbouring characters are not letters/underscores (excludes digits
    inside identifiers such as g0/h1/lean4)."""
    out = []
    for m in NUM.finditer(text):
        a, bb = m.start(), m.end()
        if a > 0 and (text[a - 1].isalnum() or text[a - 1] == "_"):
            continue
        if bb < len(text) and (text[bb].isalnum() or text[bb] == "_"):
            continue
        out.append((a, bb))
    return out


def attrib_of(lits, spans):
    """literal -> device source (containment wins; none => None).
    Returns (list_of_src_or_None, n_cov)."""
    srcs, n = [], 0
    for (a, bb) in lits:
        hit = None
        for s in spans:
            if s["cs"] <= a and bb <= s["ce"]:
                hit = s["src"]
                break
        srcs.append(hit)
        if hit:
            n += 1
    return srcs, n


# ---------------------------------------------------------------------------
# 4) token alignment (target-string coordinates -> sidecar ids coordinates)
# ---------------------------------------------------------------------------

def tok_align(tok, target_text, lits, spans, ksrc):
    """-> (ids, offs, slots) with slots = [[tok_index_in_target, tokid, kind, src], ...]"""
    enc = tok(target_text, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(x) for x in enc["input_ids"]]
    offs = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    slots = []
    for j, (a, bb) in enumerate(offs):
        if bb <= a:
            continue
        for i, (la, lb) in enumerate(lits):
            if a < lb and bb > la:                      # token overlaps a content integer
                slots.append([j, ids[j], kind_of_src(ksrc[i]), ksrc[i]])
                break
    return ids, offs, slots


# ---------------------------------------------------------------------------
# 5) main flow
# ---------------------------------------------------------------------------

def init_dev():
    global DEV, OPS
    import gen33_device as _D
    import gen32_ops as _O
    DEV, OPS = _D, _O
    return DEV, OPS


def load_tok(vocab_path):
    import argparse as _ap
    import gen33_eval as E33
    a = _ap.Namespace(vocab=vocab_path, allow=os.path.join(G33, "cache", "ccp_allow_v4.bin"))
    ctx = E33._load_ctx_eval(a)
    return ctx["tok"], ctx


def dev_row_of(side_rec):
    """sidecar -> synthetic row for the live device evaluation (convention =
    gen33_eval._dev_row_of verbatim)."""
    ops = list(side_rec.get("ops") or [])
    sfa = list(side_rec.get("sfa") or [])
    steps = [{"op": op, "args": (dict(sfa[k]) if k < len(sfa) and sfa[k] else {})}
             for k, op in enumerate(ops)]
    meta = {}
    if side_rec.get("vars"):
        meta["vars"] = list(side_rec["vars"])
    if side_rec.get("dom_dump") is not None:
        meta["dom_dump"] = side_rec["dom_dump"]
    return {"name": side_rec.get("name"), "steps": steps, "meta": meta}


def annotate_rows(rows, gold_of, tok, tag, logp=None, limit=None):
    """-> (records, stats). rows = [(key, dev_row, side_rec)]"""
    recs = []
    st = {"n": 0, "n_recon_eq": 0, "n_recon_ne": 0, "n_recon_err": 0,
          "n_lit_naive": 0, "n_lit_content": 0, "n_lit_content_attrib": 0,
          "n_slots": 0, "n_tok_unaligned": 0,
          "kind_hist": collections.Counter(), "src_class_hist": collections.Counter(),
          "bad": [], "ne": [], "by_family": collections.defaultdict(
              lambda: {"n": 0, "content": 0, "attrib": 0, "naive": 0})}
    for (key, drow, srec) in (rows[:limit] if limit else rows):
        st["n"] += 1
        gold = gold_of(key, srec)
        txt, spans, info = row_proof_instr(drow, gold=gold)
        if info["dev_err"]:
            st["n_recon_err"] += 1
            if len(st["bad"]) < 6:
                st["bad"].append({"key": key, "err": info["dev_err"][:2]})
            continue
        recl = {"ri": srec.get("ri"), "name": srec.get("name"), "cls": srec.get("cls"),
                "seg": srec.get("seg"), "np": srec.get("np"), "n": srec.get("n"),
                "recon_eq": bool(gold is not None and txt == gold), "n1_rev": info["n1_rev"],
                "op_mismatch": info["op_mismatch"][:2], "n_present": info["n_present"]}
        if not recl["recon_eq"]:
            st["n_recon_ne"] += 1
            if len(st["ne"]) < 4:
                st["ne"].append({"key": key, "n_present": info["n_present"],
                                 "gold_len": None if gold is None else len(gold),
                                 "recon_len": len(txt),
                                 "gold_head": (gold or "")[:160], "recon_head": txt[:160],
                                 "op_mismatch": info["op_mismatch"][:1]})
            recs.append(recl)
            continue
        st["n_recon_eq"] += 1
        nl = naive_lits(gold)
        cl = content_lits(gold)
        csrc, ncov = attrib_of(cl, spans)
        recl.update({"n_lit_naive": len(nl), "n_lit_content": len(cl),
                     "n_lit_content_attrib": ncov,
                     "unattrib": [gold[a:bb] for (a, bb), s in zip(cl, csrc) if not s][:8]})
        st["n_lit_naive"] += len(nl)
        st["n_lit_content"] += len(cl)
        st["n_lit_content_attrib"] += ncov
        fam = (srec.get("family") or "?")
        d = st["by_family"][fam]
        d["n"] += 1
        d["naive"] += len(nl)
        d["content"] += len(cl)
        d["attrib"] += ncov
        for s in csrc:
            st["src_class_hist"][("none" if not s else kind_of_src(s))] += 1
        # ---- token alignment ----
        ids = list(srec.get("ids") or [])
        np_ = int(srec.get("np") or 0)
        target = gold                     # sidecar target = "```lean4\n" + proof + "\n```"
        pre = "```lean4\n"
        tgt_full = pre + target + "\n```"
        tids, offs, slots = tok_align(tok, tgt_full, cl, spans, csrc)
        n_unal = 0
        # The exam sidecar carries **no gold target** (has_target=False, np==n) => it can
        # and should not be compared; only the "annotation-time" alignment is done (slot
        # indices start at np, sharing the generation-time coordinate origin).
        if np_ + len(tids) <= len(ids):
            want = ids[np_:np_ + len(tids)]
            if want != tids:
                n_unal = -1
                if len(st["ne"]) < 4:
                    st["ne"].append({"key": key, "err": "TOK_MISMATCH",
                                     "want_head": want[:12], "got_head": tids[:12]})
        elif not bool(srec.get("has_target", True)):
            n_unal = 0                        # no gold target: skip comparison (no_target)
        else:
            n_unal = -1
            if len(st["ne"]) < 4:
                st["ne"].append({"key": key, "err": "TOK_SHORT",
                                 "len_ids": len(ids), "np": np_, "n_tgt": len(tids)})
        st["n_tok_unaligned"] += (1 if n_unal else 0)
        for s in slots:
            st["kind_hist"][SLOT_KINDS[s[2]]] += 1
            s[0] += np_                    # target-relative index -> absolute sidecar ids index
        recl.update({"slots": slots, "n_slots": len(slots),
                     "tok_unaligned": bool(n_unal)})
        st["n_slots"] += len(slots)
        recs.append(recl)
    rep = {k: (dict(v) if isinstance(v, collections.Counter) else v)
           for k, v in st.items() if k != "by_family"}
    rep["by_family"] = {k: dict(v, cov=round(v["attrib"] / max(1, v["content"]), 6))
                        for k, v in sorted(st["by_family"].items())}
    rep["coverage"] = round(st["n_lit_content_attrib"] / max(1, st["n_lit_content"]), 6)
    rep["recon_rate"] = round(st["n_recon_eq"] / max(1, st["n"]), 6)
    return recs, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default=G33 + "/cache/CCP_VOCAB_V4.json")
    ap.add_argument("--corpus", default=C.DEEP_CORPUS)
    ap.add_argument("--render-side", default=G33 + "/out/DEEP_RENDER_SIDECAR.jsonl")
    ap.add_argument("--exam-side", default=G33 + "/out/DEEP_EXAM_SIDECAR.jsonl")
    ap.add_argument("--exam-cand", default=C.EXAM_CAND)
    ap.add_argument("--out-corpus", default=OUT + "/DEEP_SLOT_SIDECAR.jsonl")
    ap.add_argument("--out-exam", default=OUT + "/DEEP_EXAM_SLOT_SIDECAR.jsonl")
    ap.add_argument("--out-report", default=OUT + "/GEN34_SLOTS.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log", default=G34 + "/logs/gen34.log")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    t0 = time.time()
    C.log("== gen34 integer slot annotation (device re-render + bytewise assertion) ==", a.log)
    led = C.parent_ledger()
    C.log("parent ledger: %s" % json.dumps({k: v["md5"] for k, v in led.items()},
                                           ensure_ascii=False), a.log)
    assert led["drct_common"]["md5"] == C.JUDGE_MD5_EXPECT
    init_dev()
    tok, ctx = load_tok(a.vocab)
    C.log("tokenizer loaded: %s vocab_size=%s fast=%s"
          % (type(tok).__name__, getattr(tok, "vocab_size", "?"),
             tok.is_fast if hasattr(tok, "is_fast") else "?"), a.log)

    rows = C.jlload(a.corpus)
    C.log("corpus %s n=%d md5=%s" % (a.corpus, len(rows), C.md5f(a.corpus)), a.log)
    side = {int(r["ri"]): r for r in C.jlload(a.render_side)}
    assert len(side) == len(rows), (len(side), len(rows))
    gold_of = {r["name"]: r["proof_text"] for r in rows}
    # **Pairing discipline (lesson measured at launch)**: the row order of
    #   `DEEP_RENDER_SIDECAR` / `deep_all.bin` is `ordered = train + held` (gen33_build),
    #   **not** the corpus order => pairing must be by name, and the slot sidecar must be
    #   keyed by the sidecar's `ri` (= cache row index = the slot key used during training).
    #   (The first version paired by enumerate index: text annotation was correct but every
    #   token index was wrong: n_tok_unaligned 2399/2399)
    by_name = {r["name"]: r for r in side.values()}
    assert len(by_name) == len(side), "sidecar names are not unique"
    items = []
    for i, r in enumerate(rows):
        s = by_name.get(r["name"])
        if s is None:
            raise SystemExit("sidecar row missing: %s" % r["name"])
        assert s["cls"] == r["cls"] and s["chain_len"] == r["chain_len"], r["name"]
        items.append((r["name"], r, s))
    C.log("pairing: %d corpus rows <-> %d sidecar rows (by name; ri range %d..%d)"
          % (len(items), len(side), min(side), max(side)), a.log)

    assert tok("x", return_offsets_mapping=True) is not None
    recs, rep = annotate_rows(items, lambda k, s: gold_of.get(k), tok, "corpus",
                              logp=a.log, limit=a.limit or None)
    recs.sort(key=lambda r: int(r["ri"]))
    exc = {r["name"]: r["proof_text"]
           for r in json.load(open(a.exam_cand, encoding="utf-8"))["items"]}
    exside = {int(r["ri"]): r for r in C.jlload(a.exam_side)}
    exrows = [(r["name"], dev_row_of(r), r) for r in
              [exside[i] for i in sorted(exside)] if r["seg"] == "exam_deep"]
    erecs, erep = annotate_rows(exrows, lambda k, s: exc.get(k), tok, "exam", logp=a.log)

    outc = {"schema": "gen34-slot-sidecar/1", "tag": "corpus",
            "n_slot_kinds": NKIND, "slot_kinds": list(SLOT_KINDS),
            "note": ("slots = [absolute sidecar_ids index, token id, kind, src]; "
                     "src = s{k}.{op}.{field} device source label (V34_DESIGN §1.3)")}
    mdc = C.jlsave([dict(outc, **r) for r in recs], a.out_corpus)
    mde = C.jlsave([dict(outc, tag="exam", **r) for r in erecs], a.out_exam)
    report = {"schema": "gen34-slots/1", "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "law": {"用户令": "你想到的解决方法是什么？派出智能体去做",
                      "预注册": "scratch/ccp/gen34/V34_DESIGN.md §1"},
              "parent_ledger": {k: v["md5"] for k, v in led.items()},
              "tokenizer": {"cls": type(tok).__name__,
                            "vocab_size": getattr(tok, "vocab_size", None)},
              "corpus": rep, "exam": erep,
              "artifacts": {"slot_sidecar": {"path": a.out_corpus, "md5": mdc,
                                             "n": len(recs)},
                            "exam_slot_sidecar": {"path": a.out_exam, "md5": mde,
                                                  "n": len(erecs)}},
              "gate": {"coverage_target": 0.95, "coverage_stopline": 0.80,
                       "coverage_corpus": rep["coverage"], "coverage_exam": erep["coverage"],
                       "pass": bool(rep["coverage"] >= 0.95 and erep["coverage"] >= 0.95),
                       "stopline_hit": bool(rep["coverage"] < 0.80 or erep["coverage"] < 0.80),
                       "tok_aligned_corpus": bool(rep["n_tok_unaligned"] == 0),
                       "tok_aligned_exam": bool(erep["n_tok_unaligned"] == 0),
                       "recon_rate_corpus": rep["recon_rate"],
                       "recon_rate_exam": erep["recon_rate"]},
              "wall_s": round(time.time() - t0, 2)}
    report["md5"] = C.save_json(a.out_report, report)
    C.log("SLOTS corpus: recon %d/%d (%.4f) content=%d attrib=%d cov=%.6f naive=%d"
          % (rep["n_recon_eq"], rep["n"], rep["recon_rate"], rep["n_lit_content"],
             rep["n_lit_content_attrib"], rep["coverage"], rep["n_lit_naive"]), a.log)
    C.log("SLOTS exam:   recon %d/%d (%.4f) content=%d attrib=%d cov=%.6f naive=%d"
          % (erep["n_recon_eq"], erep["n"], erep["recon_rate"], erep["n_lit_content"],
             erep["n_lit_content_attrib"], erep["coverage"], erep["n_lit_naive"]), a.log)
    C.log("SLOTS kind_hist=%s" % json.dumps(rep["kind_hist"], ensure_ascii=False), a.log)
    C.log("SLOTS src_class_hist=%s" % json.dumps(
        {SLOT_KINDS[k] if isinstance(k, int) else k: v
         for k, v in rep["src_class_hist"].items()}, ensure_ascii=False), a.log)
    C.heartbeat("slots", "cov=%.4f" % rep["coverage"])
    print("GEN34_SLOTS_RC=0 -> %s" % a.out_report)
    return report


if __name__ == "__main__":
    main()
