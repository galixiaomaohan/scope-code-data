# -*- coding: utf-8 -*-
import os
import sys

R = os.environ.get("SCOPE_RESEARCH_ROOT", ".")
sys.path.insert(0, R + "/scratch/ccp/gen33")
sys.path.insert(0, R + "/scratch/ccp/gen32")
sys.path.insert(0, R + "/scratch/ccp/bh")
sys.path.insert(0, os.environ.get("SCOPE_SUPEROPERATOR",
                                  os.path.join(R, os.pardir, "superoperator")))
import g33_common as C
import gen33_device as DEV
import gen32_ops as OPS

rows = C.jlload(C.DEEP_CORPUS)
print('rows', len(rows))
n_eq = n_ne = n_err = 0
bad = []
for r in rows:
    tactics = []
    for k in range(len(r['steps'])):
        res = OPS.call(r['steps'][k]['op'], DEV.dev_args(r, k))
        tactics.append(res.tactic)
    joined = chr(10).join([l for l in tactics if l.strip()])
    if joined == r['proof_text']:
        n_eq += 1
    else:
        n_ne += 1
        if len(bad) < 3:
            bad.append({'name': r['name'], 'cls': r['cls'],
                        'gold': r['proof_text'][:300], 'recon': joined[:300]})
print('EXACT_EQ', n_eq, '/', len(rows))
for b in bad:
    print('---', b['name'], b['cls'])
    print('GOLD:', repr(b['gold']))
    print('RECO:', repr(b['recon']))
