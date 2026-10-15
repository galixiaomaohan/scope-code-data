# -*- coding: utf-8 -*-
# gen36_forkpatch.py -- node [2b] fork bench (V36_DESIGN §3 D0): generate the three gen36 artifacts from their gen35 parents.
#
# Discipline: parents are **untouched** (fork-copy regime); this file writes only under scratch/ccp/gen36/**.
#   Generates: gen36_hybrid.py (trainer; +--w-tail terminal-step weighting) . gen36_eval.py (exam section; +--state-shift)
#          gen36_armrun.py (judging/compiler payload port; LEAN work dir retargeted to gen36/lean)
# Every replacement asserts its hit count (never silent); artifact md5 + diff list go to out/GEN36_FORK_LEDGER.json.
import argparse
import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True

RES = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
CCP = RES + "/scratch/ccp"
G36 = CCP + "/gen36"
G35 = CCP + "/gen35"
OUT = G36 + "/out"

HDR_HYBRID = """# ---- gen36 fork registry (terminal-step routing push . change 98 continued) ----
# This file is a **minimal-surface fork copy** of gen35_hybrid.py (md5 %(pmd5)s); parent untouched (fork-copy regime).
# Differences in this batch (V36_DESIGN §3; the only training-semantics difference is [2]):
#   [1] Paths: add G36 (write scope); default OUT/artifacts point to gen36; gen35 kept read-only (ledger)
#   [2] **Terminal-step decision-position chain CE weighting** `--w-tail W` (labels in {tail_zero, tail_square, tail_other});
#       with W == 1.0, loss_f **falls back to the parent verbatim** (CCPModel.loss_f, structurally equivalent)
#   [3] Registered: curve/summary add w_tail and cache_md5 (readings carry their provenance)
# Redlines same as parent: write scope is scratch/ccp/gen36/** only; GPU0 user redline / GPU1 reserved zone strictly forbidden; drct_common frozen read-only.
# ---- gen35 fork registry (pure planning + mechanical rendering . change 98) ----
"""

HDR_EVAL = """# ---- gen36 fork registry (terminal-step routing push, exam section) ----
# This file is a **minimal-surface fork copy** of gen35_eval.py (md5 %(pmd5)s); parent untouched.
# Differences (V36_DESIGN §3 D3): [1] default paths OUT/LOGS/ckpt -> gen36;
#   [2] `--state-shift 1` = payload res **shift convention** (res(k) := device-recomputed R_in(k+1), terminal frame kept),
#       bitwise the same convention as the G-SW arm training cache (gen36/cache/deep_all_a2.bin); default 0 = parent convention.
#   [3] judging/compile go through gen36_armrun (Lean work dir gen36/lean).
# ---- gen35 fork registry ----
"""

HDR_ARMRUN = """# ---- gen36 fork registry (judging/compiler payload port) ----
# This file is a **minimal-surface fork copy** of gen35_armrun.py (md5 %(pmd5)s); parent untouched.
# Differences: [1] import gen36_common; [2] LEAN work dir -> gen36/lean (write-scope discipline);
#   the compiler still imports gen35_compile (frozen read-only).
# ---- gen35 fork registry ----
"""

HDR_JUDGE = """# ---- gen36 fork registry (literal/reverse dual-convention judging) ----
# This file is a **minimal-surface fork copy** of gen35_judge.py (md5 %(pmd5)s); parent untouched; judging logic unchanged.
# Differences: only [1] import gen36_common (write scope -> gen36/**).
# ---- gen35 fork registry ----
"""

MD5_HELPER = '''
def _md5f36(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def tail_op_idx36(vocab_path):
    """Terminal-step operator label indices (read from vocabulary ops_gen32; the three tail operators)."""
    v = json.load(open(vocab_path, encoding="utf-8"))
    return sorted(int(i) for n, i in (v.get("ops_gen32") or {}).items() if "tail" in n)
'''

LOSSF = '''
    # ---- gen36 [2]: terminal-step decision-position chain CE weighting (V36_DESIGN §3 D1) ----
    def loss_f(self, step_logits, zb, tgt, bc, lam_bind=1.0):
        """Formula-for-formula the same as gen13 CCPModel.loss_f; the only increment = CE at decision positions
        whose label is a **terminal-step operator**, multiplied by w_tail.

        w_tail == 1.0 => **falls back to the parent verbatim** (structurally equivalent; no new tensors/params => no new module).
        """
        w = float(getattr(self, "w_tail", 1.0) or 1.0)
        if w == 1.0:
            return CCPModel.loss_f(self, step_logits, zb, tgt, bc, lam_bind=lam_bind)
        lab = torch.tensor(bc.labels, dtype=torch.long, device=step_logits.device)
        per = F.cross_entropy(step_logits.reshape(-1, self.HEAD_C), lab.reshape(-1),
                              ignore_index=-100, reduction="none")
        pu = lab.reshape(-1)
        wt = torch.ones_like(per)
        for _ix in TAIL_IDX:
            wt = torch.where(pu == _ix, torch.full_like(wt, w), wt)
        act = (pu != -100)
        ls = (per * wt)[act].sum() / wt[act].sum().clamp_min(1.0)
        t = tgt.clone()
        n_unk = int((t == HC.UNK).sum())
        t[t == HC.UNK] = -100
        nbf = int((t != -100).sum())
        lb = F.cross_entropy(zb.reshape(-1, self.NVALS_C), t.reshape(-1),
                             ignore_index=-100, reduction="sum") / max(nbf, 1)
        return ls + (lam_bind / HC.NSLOT) * lb, ls, lb, n_unk, nbf
'''


def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", type=int, default=1)
    ap.add_argument("--ledger", default=OUT + "/GEN36_FORK_LEDGER.json")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rep = {"schema": "gen36-fork-ledger/1", "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "law": {"pre": "scratch/ccp/gen36/V36_DESIGN.md §3"}, "items": {}}

    # =============== gen36_hybrid ===============
    src = os.path.join(G35, "gen35_hybrid.py")
    txt = open(src, encoding="utf-8", newline="").read()
    n_crlf = txt.count("\r\n")
    txt = txt.replace("\r\n", "\n")
    pmd5 = hashlib.md5(open(src, "rb").read()).hexdigest()
    reps = [
        ("header", 1, "", HDR_HYBRID % {"pmd5": pmd5}, "prepend"),
        ("paths", 1,
         'G35 = os.path.join(CCP, "gen35")            # gen35 fork write scope\n'
         'OUT = os.path.join(G35, "out")\n'
         'GEN35_PURE_PLAN = 1\n'
         'GEN35_FORK_OF = "gen34_hybrid.py"\n'
         'GEN35_PARENT_MD5 = "094515124766d60499a81b7db2388ed4"',
         'G36 = os.path.join(CCP, "gen36")            # gen36 fork write scope\n'
         'G35 = os.path.join(CCP, "gen35")            # read-only previous generation (ledger)\n'
         'OUT = os.path.join(G36, "out")\n'
         'GEN36_TAIL_W = 1\n'
         'GEN36_FORK_OF = "gen35_hybrid.py"\n'
         'GEN36_PARENT_MD5 = "%s"\n'
         'TAIL_IDX = []                               # gen36 [2]: terminal-step operator label indices' % pmd5),
        ("syspath", 1,
         'G26, G30, G33, G34, G35,\n'
         '           os.environ.get("SCOPE_SUPEROPERATOR", "../device")):',
         'G26, G30, G33, G34, G35, G36,\n'
         '           os.environ.get("SCOPE_SUPEROPERATOR", "../device")):'),
        ("helpers", 1,
         'import numpy as np                                                          # noqa: E402',
         MD5_HELPER.strip("\n") + '\n\nimport numpy as np'
         '                                                          # noqa: E402'),
        ("side_dflt", 1,
         'ap.add_argument("--side", default=os.path.join(OUT, "DEEP_RENDER_SIDECAR.jsonl"))',
         'ap.add_argument("--side", default=os.path.join(G33, "out",\n'
         '                                              "DEEP_RENDER_SIDECAR.jsonl"))'),
        ("wt_arg", 1,
         'ap.add_argument("--lam-txt", type=float, default=1.0)',
         'ap.add_argument("--lam-txt", type=float, default=1.0)\n'
         '    ap.add_argument("--w-tail", type=float, default=1.0,\n'
         '                    help="gen36 [2]: terminal-step decision-position chain CE weight (1.0 = parent fallback, verbatim)")'),
        ("wt_init", 1,
         '        self.lam_txt = 1.0\n',
         '        self.lam_txt = 1.0\n'
         '        self.w_tail = 1.0        # gen36 [2]: terminal-step decision-position CE weight (1.0 = parent verbatim)\n'),
        ("wt_lossf", 1,
         '    # ---- render-segment payload ----\n',
         LOSSF + '\n    # ---- render-segment payload ----\n'),
        ("wt_wire", 1,
         'model.lam_bind, model.lam_txt = a.lam_bind, a.lam_txt\n',
         'model.lam_bind, model.lam_txt = a.lam_bind, a.lam_txt\n'
         '    model.w_tail = float(a.w_tail)                     # gen36 [2]\n'
         '    TAIL_IDX[:] = tail_op_idx36(a.vocab)\n'
         '    log("gen36 terminal-step weighting: w_tail=%g TAIL_IDX=%s (1.0 falls back to the parent bitwise)"\n'
         '        % (a.w_tail, TAIL_IDX), a.log)\n'),
        ("wt_obj", 1,
         '"lam_bind": a.lam_bind, "lam_txt": a.lam_txt, "NSLOT": NSLOT,',
         '"lam_bind": a.lam_bind, "lam_txt": a.lam_txt, "NSLOT": NSLOT,\n'
         '                        "w_tail": a.w_tail, "cache": a.cache,\n'
         '                        "cache_md5": _md5f36(a.cache),'),
    ]
    out_txt, items = apply_reps(txt, reps)
    dst = os.path.join(G36, "gen36_hybrid.py")
    rep["items"]["gen36_hybrid.py"] = {"src": src, "src_md5": pmd5, "n_crlf_parent": n_crlf,
                                      "dst": dst, "patches": items}
    if a.apply:
        open(dst, "w", encoding="utf-8", newline="\n").write(out_txt)
        rep["items"]["gen36_hybrid.py"]["dst_md5"] = md5f(dst)
        rep["items"]["gen36_hybrid.py"]["bytes"] = os.path.getsize(dst)
    rep["items"]["gen36_hybrid.py"]["leftover_G35_refs"] = out_txt.count("G35")

    # =============== gen36_eval ===============
    src = os.path.join(G35, "gen35_eval.py")
    txt = open(src, encoding="utf-8", newline="").read()
    n_crlf = txt.count("\r\n")
    txt = txt.replace("\r\n", "\n")
    pmd5 = hashlib.md5(open(src, "rb").read()).hexdigest()
    reps = [
        ("header", 1, "", HDR_EVAL % {"pmd5": pmd5}, "prepend"),
        ("common", 1, "import gen35_common as C", "import gen36_common as C"),
        ("paths", 1, 'OUT = C.G35 + "/out"\nLOGS = C.G35 + "/logs"',
         'OUT = C.G36 + "/out"\nLOGS = C.G36 + "/logs"'),
        ("armrun", 1, 'import gen35_armrun as AR', 'import gen36_armrun as AR'),
        ("ckpt", 1,
         'ap.add_argument("--ckpt", default=C.G35 + "/ckpt/deep_plan_formal_6000.pt")',
         'ap.add_argument("--ckpt", default=C.G36 + "/ckpt/deep_plan_wt5_6000.pt")'),
        ("shift_arg", 1, 'ap.add_argument("--dev", default="cuda:2")',
         'ap.add_argument("--state-shift", type=int, default=0,\n'
         '                    help="gen36 [D3]: 1 = payload res shift convention (same convention as the G-SW training cache)")\n'
         '    ap.add_argument("--dev", default="cuda:2")'),
        ("shift_fn", 1,
         'def build_args(a):',
         'def _payload_of_shift(side_rec):\n'
         '    """gen36 [D3] payload convention = the gen36/cache/deep_all_a2 training-cache convention (single state source gen36_state).\n\n'
         '    Shift only for the cascade family (bitwise identical to cache scope=cascade; other families use the parent convention):\n'
         '      frame k (k <= n-2) carries res := evals(R_out(k)) = post-step device residual;\n'
         '      terminal frame (k = n-1) keeps the parent value (dev_resid(row, n-1); only read by the END decision).\n'
         '    """\n'
         '    base = E34._payload_of(side_rec)\n'
         '    if (side_rec.get("family") or "") != "gen32_ineq_cascade":\n'
         '        return base\n'
         '    import gen36_state as S36\n'
         '    sfa = list(side_rec.get("sfa") or [])\n'
         '    res = list(base["res"])\n'
         '    errs = list(base.get("dev_errs") or [])\n'
         '    n = len(sfa)\n'
         '    for k in range(n):\n'
         '        if k >= n - 1:\n'
         '            continue\n'
         '        try:\n'
         '            r8, _ro, _nv = S36.post_action_res(sfa[k])\n'
         '            res[k] = r8\n'
         '        except Exception as e:                                    # noqa: BLE001\n'
         '            errs.append({"k": k, "err": "SHIFT:" + repr(e)})\n'
         '    return {"res": res, "sfa": sfa, "dev_errs": errs,\n'
         '            "res_src": "device_live(post_action/cascade)"}\n\n\n'
         'def build_args(a):'),
        ("shift_wire", 1,
         '    na = build_args(a)\n    E34._set_vocab(a.vocab)',
         '    na = build_args(a)\n    E34._set_vocab(a.vocab)\n'
         '    if int(getattr(a, "state_shift", 0)):      # gen36 [D3] payload shift convention\n'
         '        E34._payload_of = _payload_of_shift\n'
         '        C.log("gen36 payload shift convention: res(k) := dev_resid(k+1), terminal frame kept", logp)'),
    ]
    out_txt, items = apply_reps(txt, reps)
    dst = os.path.join(G36, "gen36_eval.py")
    rep["items"]["gen36_eval.py"] = {"src": src, "src_md5": pmd5, "n_crlf_parent": n_crlf,
                                     "dst": dst, "patches": items}
    if a.apply:
        open(dst, "w", encoding="utf-8", newline="\n").write(out_txt)
        rep["items"]["gen36_eval.py"]["dst_md5"] = md5f(dst)
        rep["items"]["gen36_eval.py"]["bytes"] = os.path.getsize(dst)

    # =============== gen36_armrun ===============
    src = os.path.join(G35, "gen35_armrun.py")
    txt = open(src, encoding="utf-8", newline="").read()
    n_crlf = txt.count("\r\n")
    txt = txt.replace("\r\n", "\n")
    pmd5 = hashlib.md5(open(src, "rb").read()).hexdigest()
    reps = [
        ("header", 1, "", HDR_ARMRUN % {"pmd5": pmd5}, "prepend"),
        ("common", 1, "import gen35_common as C", "import gen36_common as C"),
        ("lean", 1, 'LEAN = C.G35 + "/lean"', 'LEAN = C.G36 + "/lean"'),
    ]
    out_txt, items = apply_reps(txt, reps)
    dst = os.path.join(G36, "gen36_armrun.py")
    rep["items"]["gen36_armrun.py"] = {"src": src, "src_md5": pmd5, "n_crlf_parent": n_crlf,
                                       "dst": dst, "patches": items}
    if a.apply:
        open(dst, "w", encoding="utf-8", newline="\n").write(out_txt)
        rep["items"]["gen36_armrun.py"]["dst_md5"] = md5f(dst)
        rep["items"]["gen36_armrun.py"]["bytes"] = os.path.getsize(dst)

    # =============== gen36_judge ===============
    src = os.path.join(G35, "gen35_judge.py")
    txt = open(src, encoding="utf-8", newline="").read()
    n_crlf = txt.count("\r\n")
    txt = txt.replace("\r\n", "\n")
    pmd5 = hashlib.md5(open(src, "rb").read()).hexdigest()
    reps = [
        ("header", 1, "", HDR_JUDGE % {"pmd5": pmd5}, "prepend"),
        ("common", 1, "import gen35_common as C", "import gen36_common as C"),
        ("out_dir", 1, 'OUT = C.G35 + "/out"', 'OUT = C.G36 + "/out"'),
        ("lean_dir", 1, 'LEAN = C.G35 + "/lean"', 'LEAN = C.G36 + "/lean"'),
        ("log", 1, 'default=C.G35 + "/logs/gen35_judge.log"',
         'default=C.G36 + "/logs/gen36_judge.log"'),
    ]
    out_txt, items = apply_reps(txt, reps)
    dst = os.path.join(G36, "gen36_judge.py")
    rep["items"]["gen36_judge.py"] = {"src": src, "src_md5": pmd5, "n_crlf_parent": n_crlf,
                                      "dst": dst, "patches": items}
    if a.apply:
        open(dst, "w", encoding="utf-8", newline="\n").write(out_txt)
        rep["items"]["gen36_judge.py"]["dst_md5"] = md5f(dst)
        rep["items"]["gen36_judge.py"]["bytes"] = os.path.getsize(dst)

    rep["n_patches_total"] = sum(len(v["patches"]) for v in rep["items"].values())
    rep["all_hits_ok"] = all(p["hits"] == p["expect"]
                             for v in rep["items"].values() for p in v["patches"])
    if a.apply:
        with open(a.ledger, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)
        rep["ledger_md5"] = md5f(a.ledger)
    print(json.dumps(rep, ensure_ascii=False, indent=1)[:4000])


def apply_reps(txt, reps):
    items = []
    for name, expect, old, new, *mode in reps:
        if mode and mode[0] == "prepend":
            txt = new + txt
            items.append({"name": name, "hits": 1, "expect": 1, "mode": "prepend"})
            continue
        n = txt.count(old)
        items.append({"name": name, "hits": n, "expect": expect,
                      "old_head": old.split("\n")[0][:70]})
        if n != expect:
            raise SystemExit("补丁命中数不符: %s hits=%d expect=%d" % (name, n, expect))
        txt = txt.replace(old, new)
    return txt, items


if __name__ == "__main__":
    main()
