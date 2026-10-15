# -*- coding: utf-8 -*-
"""V32 codebook instance -- does **not modify** the frozen codebooks of svokernel / svokernel_v31.

Why a third codebook (exactly what V31 did to the legacy one):
  - the v31 codebook has exactly 155 rows (154 operators + 1 END sentinel), with no free row to
    append to;
  - the harder reason: the v31 codebook anchors the v31 model-side 155-way routing head; any
    append would move that anchor and the row numbers of the trained head. V32 therefore uses a
    **separate codebook instance**, replicating the v31 rows row by row (row numbers and indices
    unchanged).

V32 codebook = the 155 v31 rows (replicated row by row, same order) + new band 9950 (6 skeleton
meta-operators, rows 9951..9956):
  - index space: 0..153 = v31 operator rows, **154 = END slot (as in v31)**, 155..160 = skeleton
    operators;
  - therefore **the op index/label of every v31 row is identical in the v32 space** -- this is the
    structural premise of "legacy does not regress" (the trained v31 155-row step head can be moved
    into the v32 161-row head row by row);
  - the embedding function latent_vec(row) is reused bit for bit (deterministic derivation from the
    same GLOBAL_SEED), so the 64-dim embeddings of the new rows are exactly what they would be if
    those rows existed in the old codebook -- V32 did not invent a second vector film.
"""
from __future__ import annotations

import os
import sys

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from svokernel.codebook import Handle, latent_vec                                  # noqa: E402
from svokernel.errors import Err, SvoError                                         # noqa: E402

from svokernel_v31.v31codebook import (                                            # noqa: E402
    V31_BANDS, V31_END_MARKER, V31_END_MARKER_ROW, V31Codebook,
    V31_NEW_BANDS, V31_CODEBOOK,
)

__all__ = [
    "V32_BANDS", "V32_NEW_BANDS", "V32Codebook", "V32_CODEBOOK", "v32_band_of",
    "V32_SKEL_BAND", "V32_END_MARKER", "V32_END_MARKER_ROW", "V32_HEAD_DIM_FROZEN",
    "v32_action_ops", "v31_prefix_rows", "v32_op_index_map", "V32_SCHEMA",
]

V32_SCHEMA = "svo-32-skel/1"

#: New V32 band (read-only copy; never written back to the v31/v31codebook band tables).
V32_NEW_BANDS = {
    # WARNING: the band base must be **greater than the END sentinel row 9999**: the index space
    # is numbered by **ascending codebook row**, and END must always be index 154 (the v31
    # convention). If the skeleton band sat before 9999, skeleton operators would be squeezed in
    # front of END and the END index would drift. This is not an aesthetic issue: the index anchors
    # the model-side routing head, and a drift means the structural premise of "legacy does not
    # regress" is gone.
    10000: "skel_proof",         # proof-skeleton meta-operators (the new layer of this direction, rows 10001..10006)
}

#: Full V32 band table = v31 band table (replicated) + new band.
V32_BANDS = dict(V31_BANDS)
V32_BANDS.update(V32_NEW_BANDS)

V32_SKEL_BAND = 10000
V32_END_MARKER = V31_END_MARKER
V32_END_MARKER_ROW = V31_END_MARKER_ROW

#: Frozen head dimension (6 skeleton operators; op indices 155..160, END still 154).
V32_HEAD_DIM_FROZEN = 161
N_SKEL_OPS = 6


class V32Codebook(V31Codebook):
    """Separate codebook instance: its band table carries the new V32 band; append-only /
    rows-are-immutable semantics match the parent class.

    Note: allocate must be **rewritten in full** -- the parent implementation hard-codes the V31
    band table (`band not in V31_BANDS`), so calling super() directly would judge 9950 an
    "unknown band".
    """

    def __init__(self, seed=None) -> None:
        super().__init__()
        self._next = {base: base for base in V32_BANDS}

    def allocate(self, name: str, band: int, sub=None) -> Handle:
        if name in self._by_name:
            return self._by_name[name]
        if band not in V32_BANDS:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="v32codebook",
                           detail="未知段基址 %r (V32 段表: %s)" % (band, sorted(V32_BANDS)))
        base = band
        row = self._next[base] if sub is None else band + sub
        if row in self._by_row:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="v32codebook",
                           detail="行 %d 已被 %s 占用 (append-only)" % (row, self._by_row[row].name))
        if not (band <= row < band + 100):
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="v32codebook",
                           detail="行 %d 越出段 %d" % (row, band))
        h = Handle(row, name)
        self._by_name[name] = h
        self._by_row[row] = h
        self._row_band[row] = band
        self._next[band] = max(self._next[band], row + 1)
        return h

    def band_of(self, row: int) -> str:
        base = self._row_band.get(row)
        if base is None:
            return ""
        return V32_BANDS.get(base, "unknown")


#: The single V32 process-level codebook.
V32_CODEBOOK = V32Codebook()


def v32_band_of(row: int) -> str:
    b = V32_CODEBOOK.band_of(row)
    if b:
        return b
    if row in V31_NEW_BANDS:
        return V31_NEW_BANDS[row]
    for base in sorted(V32_BANDS):
        if base <= row < base + 100:
            return V32_BANDS[base]
    return "unknown"


def v31_prefix_rows() -> list:
    """v31 codebook row order (155 rows: 154 operators + END sentinel) -- the prefix of the v32
    index space."""
    return [n for n, _ in sorted(V31_CODEBOOK.rows().items(), key=lambda kv: kv[1])]


def v32_action_ops() -> list:
    """Action-space operator names (ordered by row, **excluding** the END sentinel)."""
    return [n for n, _ in sorted(V32_CODEBOOK.rows().items(), key=lambda kv: kv[1])
            if n != V32_END_MARKER]


def v32_op_index_map() -> dict:
    """Symbol -> action index = **the ordinal in codebook row order (END sentinel included)**.

    The v31 prefix is unchanged bit for bit: 0..153 operators, **154 = END**; skeleton operators
    are 155..160. (Note: enumerate over v32_action_ops() must **not** be used -- it excludes END and
    would mislabel the skeleton operators as 154+.)
    """
    rows = sorted(V32_CODEBOOK.rows().items(), key=lambda kv: kv[1])
    return {n: i for i, (n, _) in enumerate(rows)}
