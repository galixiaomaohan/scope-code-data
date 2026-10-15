"""Typed IO schema (§7.1: explicit schemas, bare strings rejected).

This module is a **controlled subset** of JSON-Schema, keeping only constructs that this
package can decide on its own and whose decision does not rely on float. Design constraints:

1. a top-level schema must not be {"kind": "any"} — §7.1 explicitly rejects untyped interfaces;
2. bool is not int (Python's `True == 1` trap is plugged explicitly);
3. exact types ("int"/"Fraction") are separate from "float" — exact arithmetic avoids the float
   channel (§17.5), so operators that need exactness declare Fraction in their schema and the
   validator rejects float;
4. list/tuple length may declare an exact value or a range, so the §17.2 underdetermination
   check can stop bad calls at the interface layer.

schema grammar (dict):
  {"kind": "int",   "min":0, "max":10}                  # bool rejected
  {"kind": "Fraction", "min":"-1/2"}                    # Fraction|int, exact
  {"kind": "float", "min":-1e9, "max":1e9, "finite":True}
  {"kind": "bool"}
  {"kind": "str",   "enum":["a","b"]}
  {"kind": "handle"}                                    # codebook row number int
  {"kind": "list",  "items": <schema>, "len":8, "minlen":1, "maxlen":16}
  {"kind": "tuple", "items":[<schema>,...]}              # fixed length, heterogeneous
  {"kind": "dict",  "fields":{...}, "required":[...], "extra":False}
  {"kind": "null"}
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any

from .errors import Err, SvoError

_KINDS = frozenset(
    {"int", "Fraction", "float", "bool", "str", "handle", "list", "tuple", "dict", "null"}
)


def _fail(msg: str) -> bool:
    return False


def check(schema: dict, value: Any, path: str = "$") -> tuple[bool, str]:
    """Return (ok, message). Pure decision, no side effects."""
    if not isinstance(schema, dict) or "kind" not in schema:
        return False, f"{path}: 非法 schema (缺 kind): {schema!r}"
    kind = schema["kind"]
    if kind not in _KINDS:
        return False, f"{path}: 未知 kind {kind!r}"

    if kind == "null":
        return (True, "") if value is None else (False, f"{path}: 期望 null, 得 {type(value).__name__}")

    if kind == "bool":
        if isinstance(value, bool):
            return True, ""
        return False, f"{path}: 期望 bool, 得 {type(value).__name__}"

    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            return False, f"{path}: 期望 int(非 bool), 得 {type(value).__name__}={value!r}"
        return _num_range(schema, value, path)

    if kind == "handle":
        if isinstance(value, bool) or not isinstance(value, int):
            return False, f"{path}: 期望 handle(int), 得 {type(value).__name__}"
        return _num_range(schema, value, path)

    if kind == "Fraction":
        # exact channel: accepts int / Fraction / canonical rational literal ("-3/2"); rejects float and bool
        if isinstance(value, bool):
            return False, f"{path}: bool 不是有理数"
        if isinstance(value, Fraction):
            pass
        elif isinstance(value, int):
            value = Fraction(value)
        elif isinstance(value, str):
            from .exact import is_exact_literal

            if not is_exact_literal(value):
                return False, f"{path}: 非规范有理字面量 {value!r} (只允许 -?\\d+(/\\d+)?; 小数写法属 float 通道)"
        else:
            return False, f"{path}: 期望 Fraction|int|有理字面量, 得 {type(value).__name__}={value!r}"
        return _num_range(schema, value, path)

    if kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False, f"{path}: 期望 float, 得 {type(value).__name__}={value!r}"
        f = float(value)
        if schema.get("finite", True):
            if f != f or f in (float("inf"), float("-inf")):
                return False, f"{path}: 期望有限浮点, 得 {f!r}"
        return _num_range(schema, f, path)

    if kind == "str":
        if not isinstance(value, str):
            return False, f"{path}: 期望 str, 得 {type(value).__name__}"
        if "enum" in schema and value not in schema["enum"]:
            return False, f"{path}: {value!r} 不在 enum {schema['enum']!r}"
        return True, ""

    if kind == "list":
        if not isinstance(value, list):
            return False, f"{path}: 期望 list, 得 {type(value).__name__}"
        if "len" in schema and len(value) != schema["len"]:
            return False, f"{path}: 长度 {len(value)} != {schema['len']}"
        if "minlen" in schema and len(value) < schema["minlen"]:
            return False, f"{path}: 长度 {len(value)} < minlen {schema['minlen']}"
        if "maxlen" in schema and len(value) > schema["maxlen"]:
            return False, f"{path}: 长度 {len(value)} > maxlen {schema['maxlen']}"
        it = schema.get("items")
        if it is not None:
            for i, v in enumerate(value):
                ok, m = check(it, v, f"{path}[{i}]")
                if not ok:
                    return False, m
        return True, ""

    if kind == "tuple":
        items = schema.get("items", [])
        if not isinstance(value, (list, tuple)):
            return False, f"{path}: 期望 tuple, 得 {type(value).__name__}"
        if len(value) != len(items):
            return False, f"{path}: 元数 {len(value)} != {len(items)}"
        for i, (s, v) in enumerate(zip(items, value)):
            ok, m = check(s, v, f"{path}[{i}]")
            if not ok:
                return False, m
        return True, ""

    if kind == "dict":
        if not isinstance(value, dict):
            return False, f"{path}: 期望 dict, 得 {type(value).__name__}"
        fields = schema.get("fields", {})
        for req in schema.get("required", []):
            if req not in value:
                return False, f"{path}: 缺必填字段 {req!r}"
        extra = schema.get("extra", False)
        for k, v in value.items():
            if k in fields:
                ok, m = check(fields[k], v, f"{path}.{k}")
                if not ok:
                    return False, m
            elif not extra:
                return False, f"{path}: 未知字段 {k!r} (extra=False)"
        return True, ""

    return _fail(f"{path}: 未覆盖 kind {kind!r}")


def _num_range(schema: dict, value: Any, path: str) -> tuple[bool, str]:
    lo, hi = schema.get("min"), schema.get("max")
    if lo is not None and value < lo:
        return False, f"{path}: {value} < min {lo}"
    if hi is not None and value > hi:
        return False, f"{path}: {value} > max {hi}"
    return True, ""


def validate(schema: dict, value: Any, ctx: str, path: str = "$") -> None:
    """Raise SCHEMA_MISMATCH (carrying the ctx handle) when the value does not match the schema."""
    ok, msg = check(schema, value, path)
    if not ok:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail=msg)


def check_declaration(schema: dict, ctx: str) -> None:
    """Validate the schema declaration itself at operator registration (§7.1: reject bare strings / any top level)."""
    if not isinstance(schema, dict) or "kind" not in schema:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail=f"schema 缺 kind: {schema!r}")
    if schema["kind"] not in _KINDS:
        raise SvoError(Err.SCHEMA_MISMATCH, ctx=ctx, detail=f"未登记 kind: {schema['kind']!r}")


def schema_str(schema: dict) -> str:
    """Human-readable one-line form, used by the five-tuple table."""
    if not isinstance(schema, dict):
        return repr(schema)
    k = schema.get("kind")
    if k == "list":
        core = f"[{schema_str(schema.get('items', {'kind': 'any'}))}]"
        if "len" in schema:
            core += f"^{schema['len']}"
        elif "minlen" in schema or "maxlen" in schema:
            core += f"^{{{schema.get('minlen', 0)},{schema.get('maxlen', 'n')}}}"
        return core
    if k == "tuple":
        return "(" + ",".join(schema_str(s) for s in schema.get("items", [])) + ")"
    if k == "dict":
        fs = schema.get("fields", {})
        req = set(schema.get("required", []))
        return "{" + ", ".join(
            f"{n}{'' if n in req else '?'}: {schema_str(s)}" for n, s in fs.items()
        ) + "}"
    if k == "str" and "enum" in schema:
        return "str∈" + repr(schema["enum"])
    if k in ("int", "float", "Fraction"):
        lo, hi = schema.get("min"), schema.get("max")
        suffix = ""
        if lo is not None or hi is not None:
            suffix = f"[{lo if lo is not None else '-inf'},{hi if hi is not None else 'inf'}]"
        return f"{k}{suffix}"
    return str(k)
