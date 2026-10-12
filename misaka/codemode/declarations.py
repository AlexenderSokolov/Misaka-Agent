"""Translated from pi's ``packages/codemode/src/declarations.ts``."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypedDict, TypeGuard
from urllib.parse import unquote

from misaka.utils.js import UNDEFINED, jsLength, jsonStringify, jsTrim, nullishCoalesce
from misaka.utils.values import read_field

from .identifier import toCodemodeIdentifier
from .types import CodemodeJsonSchema

# `\Z`, not `$`: Python's `$` also matches before a final newline.
IDENTIFIER = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*\Z")
INDENT = "  "
# Largest rendered input type, in characters, before it becomes `unknown`.
DEFAULT_INPUT_SCHEMA_MAX_CHARS = 16_000
# Local `$ref` expansions per rendered schema, so shared definitions cannot blow up the output.
MAX_REF_EXPANSIONS = 32

# TypeScript types for MCP results, from the MCP `CallToolResult` schema, so `CallToolResult<T>`
# declarations can refer to them.
MCP_TYPESCRIPT_PREAMBLE = """type Role = "user" | "assistant";
type MetaObject = Record<string, unknown>;
type Annotations = {
  audience?: Role[];
  priority?: number;
  lastModified?: string;
};
type Icon = {
  src: string;
  mimeType?: string;
  sizes?: string[];
  theme?: "light" | "dark";
};
type TextResourceContents = {
  uri: string;
  mimeType?: string;
  _meta?: MetaObject;
  text: string;
};
type BlobResourceContents = {
  uri: string;
  mimeType?: string;
  _meta?: MetaObject;
  blob: string;
};
type TextContent = {
  type: "text";
  text: string;
  annotations?: Annotations;
  _meta?: MetaObject;
};
type ImageContent = {
  type: "image";
  data: string;
  mimeType: string;
  annotations?: Annotations;
  _meta?: MetaObject;
};
type AudioContent = {
  type: "audio";
  data: string;
  mimeType: string;
  annotations?: Annotations;
  _meta?: MetaObject;
};
type ResourceLink = {
  icons?: Icon[];
  name: string;
  title?: string;
  uri: string;
  description?: string;
  mimeType?: string;
  annotations?: Annotations;
  size?: number;
  _meta?: MetaObject;
  type: "resource_link";
};
type EmbeddedResource = {
  type: "resource";
  resource: TextResourceContents | BlobResourceContents;
  annotations?: Annotations;
  _meta?: MetaObject;
};
type ContentBlock =
  | TextContent
  | ImageContent
  | AudioContent
  | ResourceLink
  | EmbeddedResource;
type CallToolResult<TStructured = { [key: string]: unknown }> = {
  _meta?: MetaObject;
  content: ContentBlock[];
  isError?: boolean;
  structuredContent?: TStructured;
  [key: string]: unknown;
};"""


class RenderDeclarationsOptions(TypedDict, total=False):
    # `CodemodeTool`s, or anything carrying their fields.
    tools: Sequence[Any]
    globals: Sequence[Any]


def renderDeclarations(options: RenderDeclarationsOptions) -> str:
    """Render TypeScript declarations for the script-visible API. Tools become members of
    ``declare const tools``, globals become ``declare function`` statements, and ``ns.member`` globals
    members of ``declare const ns``. Descriptions become doc comments; schemas become types."""
    sections: list[str] = []
    tools = options.get("tools") or []
    if len(tools) > 0:
        members = [
            f"{docComment(read_field(tool, 'description'), INDENT)}{INDENT}{renderToolSignature(tool)}" for tool in tools
        ]
        sections.append("declare const tools: {\n" + "\n".join(members) + "\n};")
    namespaces: dict[str, list[str]] = {}
    for global_ in options.get("globals") or []:
        name: str = read_field(global_, "name")
        dot = name.find(".")
        if dot == -1:
            sections.append(renderGlobal(f"declare function {name}", global_, ""))
            continue
        namespace = name[:dot]
        members = namespaces.setdefault(namespace, [])
        members.append(renderGlobal(name[dot + 1 :], global_, INDENT))
    for namespace, members in namespaces.items():
        sections.append(f"declare const {namespace}: {{\n" + "\n".join(members) + "\n};")
    return "\n\n".join(sections)


def renderToolSignature(tool: Any, options: Mapping[str, int] | None = None) -> str:
    """One tool as a member of the ``tools`` object: ``name(args: T): Promise<R>;`` with the name as
    the identifier scripts use. Input types longer than ``inputMaxChars`` render as ``unknown``. Tools
    whose output schema is an MCP ``CallToolResult`` render as ``Promise<CallToolResult<T>>``, which
    needs :data:`MCP_TYPESCRIPT_PREAMBLE`."""
    inputSchema = read_field(tool, "inputSchema")
    input = (
        "unknown"
        if inputSchema is None
        else schemaToType(
            inputSchema,
            {"maxChars": nullishCoalesce((options or {}).get("inputMaxChars"), DEFAULT_INPUT_SCHEMA_MAX_CHARS)},
        )
    )
    output = renderToolOutputType(read_field(tool, "outputSchema"))
    return f"{toCodemodeIdentifier(read_field(tool, 'name'))}(args: {input}): Promise<{output}>;"


def renderToolSample(tool: Any, options: Mapping[str, int] | None = None) -> str:
    """A tool's sample: the description followed by the tool's declaration. Used for tool listings
    and ``ALL_TOOLS`` entries."""
    declaration = f"declare const tools: {{ {renderToolSignature(tool, options)} }};"
    description = read_field(tool, "description")
    return f"{jsTrim(description) if description is not None else ''}\n\ncodemode tool declaration:\n```ts\n{declaration}\n```"


def mcpStructuredContentSchema(schema: CodemodeJsonSchema | None) -> CodemodeJsonSchema | None:
    """The ``structuredContent`` schema of an MCP ``CallToolResult`` output schema (detected by a
    ``content`` array of objects, boolean ``isError``, and object ``_meta``), ``True`` when it declares
    none, or ``None`` when the schema is not a ``CallToolResult``."""
    if not isObject(schema) or not isObject(schema.get("properties")):
        return None
    properties = schema["properties"]
    content, isError, _meta = properties.get("content"), properties.get("isError"), properties.get("_meta")
    structuredContent = properties.get("structuredContent", UNDEFINED)
    if (
        not isObject(content)
        or content.get("type") != "array"
        or not isObject(content.get("items"))
        or content["items"].get("type") != "object"
    ):
        return None
    if not isObject(isError) or isError.get("type") != "boolean" or not isObject(_meta) or _meta.get("type") != "object":
        return None
    return structuredContent if isObject(structuredContent) or isinstance(structuredContent, bool) else True


def renderToolOutputType(schema: CodemodeJsonSchema | None) -> str:
    """The type a tool call resolves to: ``CallToolResult<T>`` for MCP output schemas (needs
    :data:`MCP_TYPESCRIPT_PREAMBLE`), the schema's type otherwise, and ``unknown`` without a schema."""
    structured = mcpStructuredContentSchema(schema)
    if structured is not None:
        type = schemaToType(structured)
        return "CallToolResult" if type == "unknown" else f"CallToolResult<{type}>"
    return "unknown" if schema is None else schemaToType(schema)


def renderGlobal(head: str, global_: Any, indent: str) -> str:
    signature = read_field(global_, "signature")
    description = read_field(global_, "description")
    if signature is not None:
        return f"{docComment(description, indent)}{indent}{head}{signature};"
    inputSchema, outputSchema = read_field(global_, "inputSchema"), read_field(global_, "outputSchema")
    input = "unknown" if inputSchema is None else schemaToType(inputSchema)
    output = "unknown" if outputSchema is None else schemaToType(outputSchema)
    return f"{docComment(description, indent)}{indent}{head}(args: {input}): Promise<{output}>;"


def docComment(description: str | None, indent: str) -> str:
    text = jsTrim(description) if description is not None else ""
    if not text:
        return ""
    lines = re.split(r"\r?\n", text.replace("*/", "*\\/"))
    if len(lines) == 1:
        return f"{indent}/** {lines[0]} */\n"
    body = "\n".join(f"{indent} *{f' {line}' if line else ''}" for line in lines)
    return f"{indent}/**\n{body}\n{indent} */\n"


def propertyKey(name: str) -> str:
    return name if IDENTIFIER.match(name) else jsonStringify(name)


def isObject(value: Any) -> TypeGuard[dict[str, Any]]:
    return isinstance(value, dict)


def union(types: Sequence[str]) -> str:
    unique = list(dict.fromkeys(types))
    if "unknown" in unique:
        return "unknown"
    return "never" if len(unique) == 0 else " | ".join(unique)


def schemaToType(schema: CodemodeJsonSchema, options: Mapping[str, int] | None = None) -> str:
    """Convert a JSON Schema to a TypeScript type expression: objects on one line
    (``{ a: string; b?: number; }``) with properties sorted by name, or one property per line with
    ``//`` comments when a property has a description; ``Array<T>`` for arrays. Local references
    (``#/$defs/...``, ``#/definitions/...``) resolve against ``schema``; recursive and remote
    references render as ``unknown``. A result longer than ``maxChars`` renders as ``unknown``."""
    type = toType(schema, SchemaContext(root=schema))
    maxChars = (options or {}).get("maxChars")
    return "unknown" if maxChars is not None and jsLength(type) > maxChars else type


@dataclass(slots=True)
class SchemaContext:
    root: CodemodeJsonSchema
    # References being expanded on the current path, to stop at recursive types.
    resolving: set[str] = field(default_factory=set)
    expansions: int = 0


_MALFORMED_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")


def _decodeURIComponent(text: str) -> str:
    """``decodeURIComponent``: a malformed escape or invalid UTF-8 raises, where ``unquote`` leaves the
    escape as it is or substitutes U+FFFD."""
    if _MALFORMED_ESCAPE.search(text):
        raise ValueError("URI malformed")
    try:
        return unquote(text, errors="strict")
    except UnicodeDecodeError:
        raise ValueError("URI malformed") from None


def resolveRef(ref: str, root: CodemodeJsonSchema) -> CodemodeJsonSchema | None:
    if ref != "#" and not ref.startswith("#/"):
        return None
    current: Any = root
    for segment in filter(None, ref[2:].split("/")):
        key = _decodeURIComponent(segment).replace("~1", "/").replace("~0", "~")
        if not isObject(current) or key not in current:
            return None
        current = current[key]
    return current if isinstance(current, bool) or isObject(current) else None


def toType(schema: Any, context: SchemaContext) -> str:
    if schema is True:
        return "unknown"
    if schema is False:
        return "never"
    if not isObject(schema):
        return "unknown"
    if isinstance(schema.get("$ref"), str):
        ref = schema["$ref"]
        if ref in context.resolving or context.expansions >= MAX_REF_EXPANSIONS:
            return "unknown"
        target = resolveRef(ref, context.root)
        if target is None:
            return "unknown"
        context.expansions += 1
        context.resolving.add(ref)
        try:
            return toType(target, context)
        finally:
            context.resolving.discard(ref)

    if "const" in schema:
        return _stringifyOrUnknown(schema["const"])
    if isinstance(schema.get("enum"), list):
        return union([_stringifyOrUnknown(value) for value in schema["enum"]])

    anyOf, oneOf = schema.get("anyOf"), schema.get("oneOf")
    variants = anyOf if isinstance(anyOf, list) else oneOf if isinstance(oneOf, list) else None
    if variants is not None:
        return union([toType(variant, context) for variant in variants])
    if isinstance(schema.get("allOf"), list):
        parts = [part for part in (toType(part, context) for part in schema["allOf"]) if part != "unknown"]
        return "unknown" if len(parts) == 0 else " & ".join(f"({part})" if " | " in part else part for part in parts)

    type = schema.get("type", UNDEFINED)
    if isinstance(type, list):
        return union([toType({**schema, "type": entry}, context) for entry in type])
    match type:
        case "string":
            return "string"
        case "number" | "integer":
            return "number"
        case "boolean":
            return "boolean"
        case "null":
            return "null"
        case "array":
            return arrayType(schema, context)
        case "object":
            return objectType(schema, context)
        case _ if type is UNDEFINED:
            if "properties" in schema or "additionalProperties" in schema or "required" in schema:
                return objectType(schema, context)
            if "items" in schema or "prefixItems" in schema:
                return arrayType(schema, context)
            return "unknown"
        case _:
            return "unknown"


def _stringifyOrUnknown(value: Any) -> str:
    """``JSON.stringify(value) ?? "unknown"``: undefined and functions have no JSON; a value that cannot be
    serialized (circular) throws."""
    if value is UNDEFINED or callable(value):
        return "unknown"
    return jsonStringify(value)


def arrayType(schema: dict[str, Any], context: SchemaContext) -> str:
    items = schema.get("items", UNDEFINED)
    if items is not UNDEFINED and not isinstance(items, list):
        return f"Array<{toType(items, context)}>"
    prefixItems = schema.get("prefixItems")
    tuple_ = prefixItems if isinstance(prefixItems, list) else items if isinstance(items, list) else []
    if len(tuple_) > 0:
        return "[" + ", ".join(toType(item, context) for item in tuple_) + "]"
    return "unknown[]"


def descriptionOf(property: Any) -> str:
    return jsTrim(property["description"]) if isObject(property) and isinstance(property.get("description"), str) else ""


def _utf16Order(name: str) -> bytes:
    """``Array.prototype.sort()`` compares UTF-16 code units."""
    return name.encode("utf-16-be", "surrogatepass")


def objectType(schema: dict[str, Any], context: SchemaContext) -> str:
    properties: dict[str, Any] = schema["properties"] if isObject(schema.get("properties")) else {}
    required_ = schema.get("required")
    required = {entry for entry in required_ if isinstance(entry, str)} if isinstance(required_, list) else set()
    names = sorted(properties, key=_utf16Order)
    members = [f"{propertyKey(name)}{'' if name in required else '?'}: {toType(properties[name], context)};" for name in names]
    additional = schema.get("additionalProperties", UNDEFINED)
    if additional is not UNDEFINED and additional is not False:
        type = "unknown" if additional is True else toType(additional, context)
        members.append(f"[key: string]: {type};")
    elif additional is UNDEFINED and len(names) == 0:
        members.append("[key: string]: unknown;")
    if len(members) == 0:
        return "{}"
    if not any(descriptionOf(properties[name]) for name in names):
        return "{ " + " ".join(members) + " }"

    lines = ["{"]
    for index, name in enumerate(names):
        for line in re.split(r"\r?\n", descriptionOf(properties[name])):
            if jsTrim(line):
                lines.append(f"{INDENT}// {jsTrim(line)}")
        lines.append(f"{INDENT}{members[index].replace(chr(10), chr(10) + INDENT)}")
    for member in members[len(names) :]:
        lines.append(f"{INDENT}{member}")
    lines.append("}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_INPUT_SCHEMA_MAX_CHARS",
    "MCP_TYPESCRIPT_PREAMBLE",
    "RenderDeclarationsOptions",
    "mcpStructuredContentSchema",
    "renderDeclarations",
    "renderToolOutputType",
    "renderToolSample",
    "renderToolSignature",
    "schemaToType",
    "toCodemodeIdentifier",
]
