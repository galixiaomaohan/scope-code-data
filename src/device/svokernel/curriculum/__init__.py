"""Curriculum generators — §8.4 / §9.7 / §10.7.

Every generator must:
  1. **Be deterministic**: same (n, seed, kwargs) → bit-identical exam set (no unseeded randomness);
  2. **Carry known answers**: each problem ships its answer plus a machine-verifiable acceptance
     equation (V-pair) so P-TEACH can act as teacher;
  3. **Argue identifiability explicitly**: inverse problems must supply an argument string for
     "why the unique solution can be identified" (§17.3); problems that cannot be argued must not
     enter the set;
  4. **Measure machine-verifiability for real**: `verify_all()` actually runs all problems and
     returns pass/fail counts; claiming 100% verifiable without running is forbidden.

Unified interface:
    generate(n: int, seed: int, **kw) -> list[dict]     # each item carries a 'verify' field description
    verify_all(items, **kw) -> dict                     # {"total","passed","failed","failures"}
    manifest(items) -> dict                             # summary (counts/template distribution/seed/fingerprint)
"""
from __future__ import annotations

import hashlib
import importlib
import json

MANIFEST_GENERATORS: tuple[tuple[str, str], ...] = (
    ("gen_L1", "§8.4 数学出题器 (正向/逆向/混合/模板实例化)"),
    ("gen_SE", "§9.7 代码出题器 (注入缺陷/重构等价/规格实现)"),
    ("gen_L3", "§10.7 逻辑出题器 (随机 k-SAT/Hilbert 证明/论证图修补)"),
)

LOADED: list[str] = []
MISSING: list[str] = []


def load_all(strict: bool = False) -> tuple[list[str], list[str]]:
    global LOADED, MISSING
    LOADED, MISSING = [], []
    for mod, _d in MANIFEST_GENERATORS:
        try:
            importlib.import_module(f".{mod}", __name__)
            LOADED.append(mod)
        except ModuleNotFoundError as e:
            if e.name and e.name.split(".")[-1] != mod:
                raise
            MISSING.append(mod)
            if strict:
                raise
    return LOADED, MISSING


def fingerprint(items: list[dict]) -> str:
    """Exam-set fingerprint: first 16 hex digits of sha256 over the structured dump (for reproduction checks)."""
    blob = json.dumps(items, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def describe() -> str:
    lines = [f"generators loaded {len(LOADED)}/{len(MANIFEST_GENERATORS)}:"]
    for mod, desc in MANIFEST_GENERATORS:
        lines.append(f"  [{'OK ' if mod in LOADED else 'MISS'}] {mod:8s} {desc}")
    return "\n".join(lines)


load_all()
