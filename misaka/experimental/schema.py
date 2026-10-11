"""The runtime validation pi's experimental processes do with ``typebox``'s ``Check``.

MISAKA fork: a schema is the JSON Schema TypeBox would build, written as a dict (``strictObject`` is TypeBox's
``StrictObject`` helper, an object without additional properties), and ``Check`` validates with ``jsonschema``.
"""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

_validators: dict[int, tuple[dict[str, Any], Draft202012Validator]] = {}


def Check(schema: dict[str, Any], value: Any) -> bool:
    """Whether ``value`` conforms to ``schema``. A schema is compiled once, by identity."""
    cached = _validators.get(id(schema))
    if cached is None or cached[0] is not schema:
        cached = (schema, Draft202012Validator(schema))
        _validators[id(schema)] = cached
    return cached[1].is_valid(value)


def strictObject(properties: dict[str, Any], optional: tuple[str, ...] = ()) -> dict[str, Any]:
    """An object schema that requires every property except those in ``optional`` and allows no others."""
    return {
        "type": "object",
        "properties": properties,
        "required": [name for name in properties if name not in optional],
        "additionalProperties": False,
    }


def looseObject(properties: dict[str, Any], optional: tuple[str, ...] = ()) -> dict[str, Any]:
    """An object schema that requires every property except those in ``optional`` and allows others."""
    return {
        "type": "object",
        "properties": properties,
        "required": [name for name in properties if name not in optional],
    }


def literal(value: str | bool) -> dict[str, Any]:
    return {"const": value}


def string(minLength: int | None = None, pattern: str | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string"}
    if minLength is not None:
        schema["minLength"] = minLength
    if pattern is not None:
        schema["pattern"] = pattern
    return schema


def integer(minimum: int | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "integer"}
    if minimum is not None:
        schema["minimum"] = minimum
    return schema


NUMBER: dict[str, Any] = {"type": "number"}
BOOLEAN: dict[str, Any] = {"type": "boolean"}
UNKNOWN: dict[str, Any] = {}


def array(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}


def union(*schemas: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": list(schemas)}
