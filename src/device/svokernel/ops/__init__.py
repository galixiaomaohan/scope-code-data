"""Operator-library loading manifest (import registers).

MANIFEST is an **explicit allowlist**: modules not listed are never loaded; modules that are
listed but missing are recorded in MISSING and printed prominently in the selfcheck report
(with strict=True the run fails and exits).

Rationale (audit-friendly): the loaded set must be enumerable and verifiable; "load whatever
is in the directory" is forbidden — otherwise anyone could change the system's capability
surface just by dropping a file into the directory.
"""
from __future__ import annotations

import importlib

MANIFEST: tuple[tuple[str, str], ...] = (
    ("p0_reference", "P0 骨架: §14 参考实现 + 码本自证件"),
    ("math_basic", "P1 §8.1 数学基础算子"),
    ("math_theorems", "P1 §8.2 定理注册表 (34 条)"),
    ("math_strategy", "P1 §8.3 策略算子 (8 个)"),
    ("code_core", "P2 §9.1 沙箱 / AST 码本 / 属性测试 / V 对"),
    ("logic_ops", "P3 §10.1–10.5 逻辑算子 (归结/合一/核/论证图/战术/反例)"),
    ("logic_cert", "P3 §10.2 证书链 (DRAT / SMT proof log / Lean 适配)"),
    ("agent_ops", "P4 §11 Agent 工具与调用循环"),
    ("code_analysis", "P5 §9.2 静态分析算子"),
    ("code_repair", "P5 §9.3 纠正算子与三重门"),
    ("code_se", "P5 §9.4–9.5 契约式生成与软件工程算子"),
    ("heavy_cores", "P5 §9.6 重载核 (有界模型检查 / 模型检查 / 符号执行)"),
    ("cert_math", "P6 §8.5 区间算术与认证数值方法"),
)

LOADED: list[str] = []
MISSING: list[str] = []


def load_all(strict: bool = False) -> tuple[list[str], list[str]]:
    """Load every available module in MANIFEST; return (loaded, missing)."""
    global LOADED, MISSING
    LOADED, MISSING = [], []
    for mod, _desc in MANIFEST:
        try:
            importlib.import_module(f".{mod}", __name__)
            LOADED.append(mod)
        except ModuleNotFoundError as e:
            # Swallow only "the module file itself is absent"; an import failure inside the module must propagate
            if e.name and e.name.split(".")[-1] != mod:
                raise
            MISSING.append(mod)
            if strict:
                raise
    return LOADED, MISSING


def describe() -> str:
    lines = [f"loaded {len(LOADED)}/{len(MANIFEST)} operator modules:"]
    for mod, desc in MANIFEST:
        mark = "OK " if mod in LOADED else "MISS"
        lines.append(f"  [{mark}] {mod:16s} {desc}")
    return "\n".join(lines)


load_all()
