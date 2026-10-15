# -*- coding: utf-8 -*-
"""V31 codebook instance (carrier of the new head dimension) -- does **not modify** the
frozen svokernel codebook.

Why a separate instance is mandatory (investigation result, see STATUS.md stage 0):
  - the legacy CODEBOOK has exactly 118 rows (checksum bd233ec24e84deec), with no free row
    to append to;
  - legacy codebook.BANDS declares only 9000..9800; band base 9900 is **not registered** --
    allocate(..., band=9900) on the legacy codebook raises "unknown band base";
  - the harder reason: the legacy codebook anchors the V30 model-side 119-way routing head
    (probes/sla_data.py.txt `END = len(OPS)`); any append would move that anchor and the
    historical numbers. V31 therefore uses a **separate codebook instance**.

V31 codebook = the 118 legacy rows (replicated row by row so row numbers survive
re-registration) + new bands + 1 END sentinel row:
  - the embedding function latent_vec(row) is reused bit for bit (deterministic derivation
    from the same GLOBAL_SEED), so the 64-dim embeddings of the new V31 rows are exactly
    what they would be if those rows existed in the old codebook -- no home-made vector film;
  - new bands:
      9900 nt_comp_L0      layer 0 contest number theory (10 operators, rows 9901..9910)
      9910 nt_struct_L1    layer 1 structural number theory (8 verified, rows 9911..9918)
      9920 nt_research_L2  layer 2 research grade (3 verified, rows 9921..9923)
      9930 goal_L3         layer 3 invocation mechanism (4, rows 9931..9934)
      9940 v31_stub        explicit stub band (8 stubs, rows 9941..9948)
      9990 v31_meta        V31 meta/sentinel band (1 sentinel: v31_end_marker, row 9999)

Row convention (same as legacy): row = band + sub, sub >= 1; no row sits on a band base.
The END sentinel v31_end_marker is a **row**, not an operator: it occupies the slot that
`END = len(OPS)` denotes in the V30 measurement convention, so the V31 head dimension
equals the codebook row count (see V31_HEAD_DIM).
"""
from __future__ import annotations

import os
import sys

# --- import bootstrap: let this package import sibling svokernel from any cwd
# --- (zero-modification reference) ---
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from svokernel.codebook import (  # noqa: E402
    BANDS as LEGACY_BANDS,
    Codebook,
    GLOBAL_SEED,
    EMB_DIM,
    RESID_DIM,
    Handle,
    band_of as legacy_band_of,
    latent_vec,
)
from svokernel.errors import Err, SvoError  # noqa: E402

__all__ = [
    "V31_BANDS", "V31_NEW_BANDS", "V31Codebook", "V31_CODEBOOK", "v31_band_of",
    "V31_END_MARKER", "V31_END_MARKER_ROW", "V31_INFRA_BAND", "V31_STUB_BAND",
    "EMB_DIM", "RESID_DIM", "GLOBAL_SEED", "latent_vec", "Handle",
]

#: New V31 band bases -> purpose. **Read-only**: never written back to svokernel.codebook.BANDS.
V31_NEW_BANDS = {
    9900: "nt_comp_L0",
    9910: "nt_struct_L1",
    9920: "nt_research_L2",
    9930: "goal_L3",
    9940: "v31_stub",
    9990: "v31_meta",
}

#: Full V31 band table = legacy bands (replicated, names untouched) + new bands.
#: This is a **copy**, not the legacy dict itself.
V31_BANDS = dict(LEGACY_BANDS)
V31_BANDS.update(V31_NEW_BANDS)

V31_INFRA_BAND = 9000            # same band as legacy infra (legacy occupies only 9001; 9002+ free)
V31_STUB_BAND = 9940             # explicit isolated stub band (stubs must not mix into implementation bands)

#: END sentinel: occupies one codebook row, corresponds to no operator (the V31 counterpart
#: of the V30 convention `END = len(OPS)`).
V31_END_MARKER = "v31_end_marker"
V31_END_MARKER_ROW = 9999


def v31_band_of(row: int) -> str:
    """V31 row -> band name. The instance registry (authoritative) is queried first, then
    legacy rows fall back to band-window scanning."""
    b = V31_CODEBOOK.band_of(row)
    if b:
        return b
    if row in V31_NEW_BANDS:
        return V31_NEW_BANDS[row]
    if row in LEGACY_BANDS:
        return LEGACY_BANDS[row]
    for base in sorted(V31_BANDS):
        if base <= row < base + 100:
            return V31_BANDS[base]
    return "unknown"


class V31Codebook(Codebook):
    """Separate codebook instance: its band table carries the new V31 bands and records
    row->band authoritatively at allocation time (no window ambiguity).

    append-only / rows-are-immutable semantics match the parent class (every parent check
    is kept). The CODEBOOK singleton is **never touched**.
    """

    def __init__(self, seed: int = GLOBAL_SEED) -> None:
        super().__init__(seed)
        # parent __init__ built _next from the legacy BANDS; swap in the V31 band table (a copy)
        self._next = {base: base for base in V31_BANDS}
        self._row_band = {}

    def allocate(self, name: str, band: int, sub=None) -> Handle:
        if name in self._by_name:
            return self._by_name[name]
        if band not in V31_BANDS:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="v31codebook",
                           detail="未知段基址 %r (V31 段表: %s)" % (band, sorted(V31_BANDS)))
        row = self._next[band] if sub is None else band + sub
        if row in self._by_row:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="v31codebook",
                           detail="行 %d 已被 %s 占用 (append-only)" % (row, self._by_row[row].name))
        if not (band <= row < band + 100):
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="v31codebook",
                           detail="行 %d 越出段 %d" % (row, band))
        h = Handle(row, name)
        self._by_name[name] = h
        self._by_row[row] = h
        self._row_band[row] = band
        self._next[band] = max(self._next[band], row + 1)
        return h

    # -- queries ------------------------------------------------------------
    def band_of(self, row: int) -> str:
        """Authoritative band name recorded at allocation; empty string when unregistered
        (the caller decides the fallback)."""
        base = self._row_band.get(row)
        if base is None:
            return ""
        return V31_BANDS.get(base, "unknown")

    def row_bands(self) -> dict:
        return dict(self._row_band)

    def table(self) -> list:
        return [(h.row, h.name, self.band_of(h.row)) for h in sorted(self._by_row.values(), key=lambda x: x.row)]


#: The single V31 process-level codebook (written at operator registration).
#: Independent of the legacy CODEBOOK.
V31_CODEBOOK = V31Codebook()


def v31_head_dim() -> int:
    """V31 model-side routing head dimension = operator row count + 1 (END slot).

    Rationale: in the V30 convention `OPS = codebook rows; END = len(OPS)` => head = len(OPS)+1.
    V31 makes the END slot a real codebook row (v31_end_marker), hence head == codebook rows.
    Computed from the codebook here; the frozen constant is asserted in the ops_infra census
    self-check.
    """
    n_rows = V31_CODEBOOK.size()
    has_end = V31_CODEBOOK.has(V31_END_MARKER)
    return n_rows if has_end else n_rows + 1


def v31_action_ops() -> list:
    """Action-space operator names (ordered by row, **excluding** the END sentinel)."""
    return [n for n, _ in sorted(V31_CODEBOOK.rows().items(), key=lambda kv: kv[1])
            if n != V31_END_MARKER]
