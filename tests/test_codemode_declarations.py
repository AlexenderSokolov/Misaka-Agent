"""pi codemode's ``declarations.test.ts``."""

from __future__ import annotations

from misaka.codemode import (
    mcpStructuredContentSchema,
    renderDeclarations,
    renderToolSample,
    renderToolSignature,
    schemaToType,
)


def execute(_args, _context):
    return None


def mcpResultSchema(structuredContent=None):
    return {
        "type": "object",
        "properties": {
            "content": {"type": "array", "items": {"type": "object"}},
            **({} if structuredContent is None else {"structuredContent": structuredContent}),
            "isError": {"type": "boolean"},
            "_meta": {"type": "object"},
        },
        "required": ["content"],
    }


# -- schemaToType ------------------------------------------------------------------------------


def test_renders_primitives_literals_and_unions():
    assert schemaToType({"type": "string"}) == "string"
    assert schemaToType({"type": "integer"}) == "number"
    assert schemaToType({"type": ["string", "null"]}) == "string | null"
    assert schemaToType({"const": "a"}) == '"a"'
    assert schemaToType({"enum": ["a", 1, None]}) == '"a" | 1 | null'
    assert schemaToType({"anyOf": [{"type": "string"}, {"type": "number"}]}) == "string | number"
    assert schemaToType({"anyOf": [{"type": "string"}, {}]}) == "unknown"
    assert schemaToType({"allOf": [{"anyOf": [{"type": "string"}, {"type": "number"}]}, {"const": 1}]}) == "(string | number) & 1"
    assert schemaToType({"$ref": "#/defs/x"}) == "unknown"
    assert schemaToType(True) == "unknown"
    assert schemaToType(False) == "never"


def test_renders_objects_on_one_line_with_sorted_properties():
    assert schemaToType({
        "type": "object",
        "properties": {"city": {"type": "string"}, "max-lines": {"type": "number"}},
        "required": ["city"],
        "additionalProperties": False,
    }) == '{ city: string; "max-lines"?: number; }'
    assert schemaToType({"type": "object", "additionalProperties": {"type": "number"}}) == "{ [key: string]: number; }"
    assert schemaToType({"type": "object"}) == "{ [key: string]: unknown; }"
    assert schemaToType({"type": "object", "properties": {}, "additionalProperties": False}) == "{}"


def test_puts_property_descriptions_on_comment_lines():
    assert schemaToType({
        "type": "object",
        "properties": {
            "weather": {
                "type": "array",
                "description": "look up weather for a given list of locations",
                "items": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
            },
        },
        "required": ["weather"],
    }) == "{\n  // look up weather for a given list of locations\n  weather: Array<{ location: string; }>;\n}"
    assert schemaToType({
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "description": "Outer",
                "properties": {"inner": {"type": "string", "description": "Inner"}},
            },
        },
    }) == "{\n  // Outer\n  outer?: {\n    // Inner\n    inner?: string;\n  };\n}"


def test_resolves_local_references_and_stops_at_recursive_ones():
    schema = {
        "type": "object",
        "properties": {
            "item": {"$ref": "#/$defs/Item"},
            "legacy": {"$ref": "#/definitions/Legacy"},
            "remote": {"$ref": "https://example.com/schema.json"},
        },
        "required": ["item"],
        "$defs": {
            "Item": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "parent": {"$ref": "#/$defs/Item"}},
                "required": ["id"],
            },
        },
        "definitions": {"Legacy": {"enum": ["a", "b"]}},
    }
    assert schemaToType(schema) == '{ item: { id: string; parent?: unknown; }; legacy?: "a" | "b"; remote?: unknown; }'


def test_renders_arrays_and_tuples():
    assert schemaToType({"type": "array", "items": {"type": "string"}}) == "Array<string>"
    assert schemaToType({"type": "array", "prefixItems": [{"type": "string"}, {"type": "number"}]}) == "[string, number]"
    assert schemaToType({"type": "array"}) == "unknown[]"


def test_renders_types_over_the_budget_as_unknown():
    properties = {f"field{i}": {"type": "string"} for i in range(50)}
    schema = {"type": "object", "properties": properties}
    assert schemaToType(schema, {"maxChars": 100}) == "unknown"
    assert "field49?: string;" in schemaToType(schema)


# -- tool declarations -------------------------------------------------------------------------


def test_renders_signatures_with_normalized_identifiers():
    assert renderToolSignature({
        "name": "hidden-dynamic-tool",
        "inputSchema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        "outputSchema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
    }) == "hidden_dynamic_tool(args: { city: string; }): Promise<{ ok: boolean; }>;"
    assert renderToolSignature({"name": "free"}) == "free(args: unknown): Promise<unknown>;"


def test_renders_mcp_call_tool_result_output_schemas_as_call_tool_result_t():
    inputSchema = {"type": "object", "properties": {}, "additionalProperties": False}
    assert renderToolSignature({
        "name": "mcp__sample__search",
        "inputSchema": inputSchema,
        "outputSchema": mcpResultSchema({
            "type": "object",
            "properties": {"results": {"type": "array", "items": {"$ref": "#/definitions/Result~1item~0v1"}}},
            "required": ["results"],
            "additionalProperties": False,
            "definitions": {
                "Result/item~v1": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "score": {"type": "number"}},
                    "required": ["id", "score"],
                    "additionalProperties": False,
                },
            },
        }),
    }) == "mcp__sample__search(args: {}): Promise<CallToolResult<{ results: Array<{ id: string; score: number; }>; }>>;"
    assert renderToolSignature({"name": "plain", "inputSchema": inputSchema, "outputSchema": mcpResultSchema()}) == (
        "plain(args: {}): Promise<CallToolResult>;"
    )
    assert mcpStructuredContentSchema({"type": "object", "properties": {"content": {"type": "array"}}}) is None


def test_renders_the_per_tool_sample():
    assert renderToolSample({"name": "foo", "description": "bar", "inputSchema": {"type": "string"}}) == (
        "bar\n\ncodemode tool declaration:\n```ts\ndeclare const tools: { foo(args: string): Promise<unknown>; };\n```"
    )


# -- renderDeclarations ------------------------------------------------------------------------


def test_renders_tools_and_globals():
    text = renderDeclarations({
        "tools": [
            {
                "name": "read",
                "description": "Read a file.\nSecond line.",
                "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                "outputSchema": {"type": "string"},
                "execute": execute,
            },
            {"name": "remote-api", "execute": execute},
        ],
        "globals": [{"name": "attach", "description": "Attach it.", "inputSchema": {"type": "string"}, "execute": execute}],
    })
    assert text == "declare const tools: {\n  /**\n   * Read a file.\n   * Second line.\n   */\n  read(args: { path: string; }): Promise<string>;\n  remote_api(args: unknown): Promise<unknown>;\n};\n\n/** Attach it. */\ndeclare function attach(args: string): Promise<unknown>;"


def test_renders_namespaced_globals_and_explicit_signatures():
    text = renderDeclarations({
        "globals": [
            {"name": "models.list", "description": "List models.", "signature": "(type: string): Promise<string[]>", "execute": execute},
            {"name": "models.get", "inputSchema": {"type": "string"}, "execute": execute},
            {"name": "plain", "signature": "(): void", "execute": execute},
        ],
    })
    assert text == "declare function plain(): void;\n\ndeclare const models: {\n  /** List models. */\n  list(type: string): Promise<string[]>;\n  get(args: string): Promise<unknown>;\n};"


def test_escapes_comment_terminators_in_descriptions():
    text = renderDeclarations({"tools": [{"name": "x", "description": "a */ b", "execute": execute}]})
    assert "/** a *\\/ b */" in text
