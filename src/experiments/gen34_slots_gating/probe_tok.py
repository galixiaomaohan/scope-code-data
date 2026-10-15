# -*- coding: utf-8 -*-
# probe_tok.py -- diagnostic: token-by-token comparison of slot-sidecar token indices
#                against the gold sidecar ids
import json
import os
import sys

R = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
sys.path.insert(0, R + "/scratch/ccp/gen34")
import gen34_slots as S                                                   # noqa: E402

G = R + "/scratch/ccp/"
BT = chr(96) * 3
tok, ctx = S.load_tok(G + "gen33/cache/CCP_VOCAB_V4.json")
rows = S.C.jlload(G + "gen32/DEEP_CORPUS_V1.jsonl")
side = [json.loads(l) for l in open(G + "gen33/out/DEEP_RENDER_SIDECAR.jsonl",
                                    encoding="utf-8")]
for ii in (0, 1, 800, 1600):
    r, s0 = rows[ii], side[ii]
    gold = r["proof_text"]
    full = BT + "lean4\n" + gold + "\n" + BT
    e = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    tids = [int(x) for x in e["input_ids"]]
    ids, np_ = s0["ids"], s0["np"]
    print("== idx %d %s cls=%s | len(tids)=%d len(ids)-np=%d"
          % (ii, r["name"], r["cls"], len(tids), len(ids) - np_))
    w = ids[np_:np_ + len(tids)]
    k = 0
    while k < min(len(w), len(tids)) and w[k] == tids[k]:
        k += 1
    print("   first mismatch at %d / %d" % (k, len(tids)))
    print("   want", w[max(0, k - 3):k + 4], " got", tids[max(0, k - 3):k + 4])
    print("   dec want:", repr(tok.decode(w[max(0, k - 4):k + 4])))
    print("   dec got :", repr(tok.decode(tids[max(0, k - 4):k + 4])))
    print("   gold tail:", repr(gold[-60:]))
