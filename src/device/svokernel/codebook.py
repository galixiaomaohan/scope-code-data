"""Vector side of the codebook and the vector-membrane protocol (VMP) (§4 + §12).

Three things:
1. **Handle allocation**: every operator/theorem/tactic occupies one frozen codebook row (handle, int);
2. **Handle → vector**: the row number maps to a 64-dim frozen embedding through a
   **deterministic** pseudorandom map — a lossless symbol-vector bijection (re-indexing, not
   compression). The same handle yields bit-identical vectors in any process and on any
   platform: the seed derives from sha512 over (global seed, row number), independent of
   PYTHONHASHSEED.
3. **decode**: kernel result → (latent-state vector, verification residual vector). The residual
   vector is **not normalized** (normalization would destroy magnitude information, and both
   the AIC score and the acceptance criterion consume magnitude).

Honesty boundary (§8 hard constraint b): for exact objects the latent vector carries a
**reference**, not the value itself — float64 has a ~1e-16 relative-precision wall, while exact
integer/symbolic objects live inside the kernel. `Handle` is the typed expression of that fact.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from .errors import Err, SvoError

EMB_DIM = 64
"""Latent-state embedding dimension (§12: each operator/theorem/tactic takes one 64-dim frozen embedding row)."""

RESID_DIM = 8
"""Verification-residual vector dimension (aligned with the v26-side P-RO readout interface)."""

GLOBAL_SEED = 20260920
"""Codebook freeze seed; changing it equals re-issuing the codebook, so it requires a version change."""

#: Codebook bands (frozen): band base → purpose. Row = band base + in-band ordinal.
BANDS: dict[int, str] = {
    9000: "infra",           # skeleton/reference implementation/tooling base
    9100: "math_basic",      # §8.1 basic operators
    9200: "math_theorem",    # §8.2 theorem registry (34 entries)
    9300: "math_strategy",   # §8.3 strategy operators (8)
    9400: "code",            # §9 code domain (incl. heavy cores)
    9500: "logic",           # §10 logic domain
    9600: "agent",           # §11 agent domain
    9700: "se",              # §9.5 software-engineering operators
    9800: "core_cert",       # §8.5/§13 certified numerics + certificate system
}


def band_of(handle: int) -> str:
    """Reverse lookup of the band from a row number (used for audit and catalog printing)."""
    for base in sorted(BANDS):
        if base <= handle < base + 100:
            return BANDS[base]
    return "unknown"


@dataclass(frozen=True)
class Handle:
    """Lossless symbolic reference in latent state (handle ≠ value)."""

    row: int
    name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.row, int) or isinstance(self.row, bool):
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="codebook", detail=f"行号须为 int, 得 {self.row!r}")

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"#{self.row}{'(' + self.name + ')' if self.name else ''}"


def latent_vec(row: int, dim: int = EMB_DIM) -> tuple[float, ...]:
    """Row number → deterministic 64-dim frozen embedding.

    Derivation: block_i = sha512(f"{GLOBAL_SEED}:{row}:{i}") → 8 float64 in [-1,1).
    8 blocks = 64 dims. Determinism comes from sha512, independent of the interpreter hash seed.
    """
    if dim % 8 != 0:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx="codebook.latent_vec", detail=f"dim 须为 8 的倍数, 得 {dim}")
    out: list[float] = []
    for i in range(dim // 8):
        h = hashlib.sha512(f"{GLOBAL_SEED}:{row}:{i}".encode("utf-8")).digest()
        for j in range(8):
            u = int.from_bytes(h[8 * j : 8 * j + 8], "big") / 2.0**64  # in [0,1)
            out.append(round(2.0 * u - 1.0, 17))
    return tuple(out)


def cos(a: Iterable[float], b: Iterable[float]) -> float:
    aa, bb = tuple(a), tuple(b)
    if len(aa) != len(bb):
        raise SvoError(Err.SCHEMA_MISMATCH, ctx="codebook.cos", detail=f"维度不等 {len(aa)} vs {len(bb)}")
    na = sum(x * x for x in aa) ** 0.5
    nb = sum(x * x for x in bb) ** 0.5
    if na == 0.0 or nb == 0.0:
        raise SvoError(Err.DEG_UNSUPPORTED, ctx="codebook.cos", detail="零向量无方向")
    return sum(x * y for x, y in zip(aa, bb)) / (na * nb)


def resid_vec(resids: Iterable[float], dim: int = RESID_DIM) -> tuple[float, ...]:
    """Verification residuals → fixed-length vector (zero-padded; not normalized, magnitude preserved)."""
    vals = [float(r) for r in resids]
    if len(vals) > dim:
        # When too long: keep the worst value + truncate the tail. The worst value is acceptance
        # information that must not be dropped.
        worst = max(abs(v) for v in vals)
        vals = vals[: dim - 1] + [worst]
    return tuple(vals) + (0.0,) * (dim - len(vals))


def resid_summary(rv: Iterable[float]) -> float:
    """Scalar summary of the residual vector = max|·| (acceptance criterion; same source as the vector side)."""
    vals = [abs(float(x)) for x in rv]
    return max(vals) if vals else 0.0


class Codebook:
    """Frozen codebook: name ↔ handle ↔ 64-dim embedding.

    Names are unique; rows are immutable; append-only. `checksum()` is a registrable fingerprint.
    """

    def __init__(self, seed: int = GLOBAL_SEED) -> None:
        self.seed = seed
        self._by_name: dict[str, Handle] = {}
        self._by_row: dict[int, Handle] = {}
        self._next: dict[int, int] = {base: base for base in BANDS}

    # -- allocation ---------------------------------------------------------
    def allocate(self, name: str, band: int, sub: int | None = None) -> Handle:
        if name in self._by_name:
            return self._by_name[name]
        if band not in BANDS:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="codebook", detail=f"未知段基址 {band}")
        row = self._next[band] if sub is None else band + sub
        if row in self._by_row:
            raise SvoError(
                Err.SCHEMA_MISMATCH,
                ctx="codebook",
                detail=f"行 {row} 已被 {self._by_row[row].name} 占用 (append-only, 禁改行)",
            )
        if not (band <= row < band + 100):
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="codebook", detail=f"行 {row} 越出段 {band}")
        h = Handle(row, name)
        self._by_name[name] = h
        self._by_row[row] = h
        self._next[band] = max(self._next[band], row + 1)
        return h

    # -- lookup -------------------------------------------------------------
    def row_of(self, name: str) -> int:
        h = self._by_name.get(name)
        if h is None:
            raise SvoError(Err.SCHEMA_MISMATCH, ctx="codebook", detail=f"未登记算子/定理: {name!r}")
        return h.row

    def name_of(self, row: int) -> str:
        h = self._by_row.get(row)
        return h.name if h else ""

    def has(self, name: str) -> bool:
        return name in self._by_name

    def rows(self) -> dict[str, int]:
        return {n: h.row for n, h in sorted(self._by_name.items(), key=lambda kv: kv[1].row)}

    def size(self) -> int:
        return len(self._by_name)

    def latent(self, name_or_row: "str | int") -> tuple[float, ...]:
        row = name_or_row if isinstance(name_or_row, int) else self.row_of(name_or_row)
        return latent_vec(row)

    def checksum(self) -> str:
        """Codebook fingerprint: ordered sha256 over the full name:row table."""
        blob = "|".join(f"{n}:{h.row}" for n, h in sorted(self._by_name.items(), key=lambda kv: kv[1].row))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def table(self) -> list[tuple[int, str, str]]:
        return [(h.row, h.name, band_of(h.row)) for h in sorted(self._by_row.values(), key=lambda x: x.row)]


#: Process-wide unique codebook (written when operators register).
CODEBOOK = Codebook()
