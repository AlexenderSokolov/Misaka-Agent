"""Translated from pi's ``packages/ai/src/utils/typebox-helpers.ts`` (v1.1.0)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict

from misaka.ai.utils._typebox import TSchema, Type


class _StringEnumOptions(TypedDict, total=False):
    description: str
    default: str


def StringEnum(values: Sequence[str], options: _StringEnumOptions | None = None) -> TSchema:
    """Creates a string enum schema compatible with Google's API and other providers
    that don't support anyOf/const patterns.

    Example::

        OperationSchema = StringEnum(["add", "subtract", "multiply", "divide"], {
            "description": "The operation to perform",
        })
    """
    schema: dict[str, Any] = {"type": "string", "enum": list(values)}
    if options is not None and options.get("description"):
        schema["description"] = options["description"]
    if options is not None and options.get("default"):
        schema["default"] = options["default"]
    return Type.Unsafe(schema)


__all__ = ["StringEnum"]
