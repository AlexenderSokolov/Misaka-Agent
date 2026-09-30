# ruff: noqa: UP040 - TypeAlias spelling kept in step with misaka/ai/types.py.

"""Shared Google Generative AI and Vertex request helpers."""

from __future__ import annotations

import base64
import copy
import re
from collections.abc import Mapping
from typing import Any, Literal, TypeAlias

from misaka.ai.models import clamp_thinking_level
from misaka.ai.providers.constrained_sampling import (
    get_json_schema_tool_parameters,
    resolve_json_schema_strict_sampling,
)
from misaka.ai.providers.transform_messages import transform_messages
from misaka.ai.types import ImageContent, Model, StopReason, Tool, TranscriptContext
from misaka.ai.utils.provider_retry import retry_provider_request
from misaka.ai.utils.sanitize_unicode import sanitize_surrogates
from misaka.ai.utils.transcript import (
    collapse_system_messages,
    without_initial_system_message,
)

GoogleThinkingLevel: TypeAlias = Literal["THINKING_LEVEL_UNSPECIFIED", "MINIMAL", "LOW", "MEDIUM", "HIGH"]

_BASE64_SIGNATURE_PATTERN = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")
_JSON_SCHEMA_META_DECLARATIONS = frozenset(
    {
        "$schema",
        "$id",
        "$anchor",
        "$dynamicAnchor",
        "$vocabulary",
        "$comment",
        "$defs",
        "definitions",
    }
)


def is_thinking_part(part: dict[str, Any] | Any) -> bool:
    if isinstance(part, dict):
        return part.get("thought") is True
    return getattr(part, "thought", None) is True


def coerce_thought_signature(value: object) -> str | None:
    """Coerce a thought_signature to str.

    google-genai declares `Part.thought_signature` as `bytes` while ToolCall declares str,
    so what comes off an SDK part needs converting; a value that is already a str is kept
    as-is. Google's field is TYPE_BYTES, i.e. base64 on the wire, so base64 both round-trips
    losslessly and is the shape `_is_valid_thought_signature` demands before a signature is
    echoed back.
    """
    if isinstance(value, (bytes, bytearray)):
        return base64.b64encode(bytes(value)).decode("ascii")
    return value if isinstance(value, str) else None


def retain_thought_signature(existing: str | None, incoming: object) -> str | None:
    coerced = coerce_thought_signature(incoming)
    if coerced:
        return coerced
    return existing


def _is_valid_thought_signature(signature: str | None) -> bool:
    if not signature:
        return False
    if len(signature) % 4 != 0:
        return False
    return _BASE64_SIGNATURE_PATTERN.fullmatch(signature) is not None


def _resolve_thought_signature(is_same_provider_and_model: bool, signature: str | None) -> str | None:
    return signature if is_same_provider_and_model and _is_valid_thought_signature(signature) else None


ResolvedGoogleThinkingLevel: TypeAlias = Literal["minimal", "low", "medium", "high"]


def resolve_google_thinking_level(model: Model, level: str) -> ResolvedGoogleThinkingLevel:
    """Resolve a supported pi level or model-specific Google mapping to a standard Google level."""
    mapped = model.thinkingLevelMap.get(level) if model.thinkingLevelMap else None
    resolved = mapped.lower() if isinstance(mapped, str) else level
    if resolved in {"minimal", "low", "medium", "high"}:
        return resolved  # type: ignore[return-value]
    raise ValueError(
        f"Unsupported Google thinking level mapping for {model.provider}/{model.id}: {level} -> {mapped}"
    )


def uses_google_thinking_level(model: Model) -> bool:
    """Whether this model uses Gemini's discrete `thinkingLevel` control instead of
    the token-based `thinkingBudget` control. Supported levels come from the
    model's `thinkingLevelMap`; this only selects the Google wire format."""
    model_id = model.id.lower()
    return (
        # Match Gemini 3 Pro/Flash IDs with or without a minor version, such as
        # gemini-3-flash-preview, gemini-3.1-pro-preview, and gemini-3.8-flash.
        re.search(r"gemini-3(?:\.\d+)?-(?:pro|flash)", model_id) is not None
        or model_id == "gemini-flash-latest"
        or model_id == "gemini-flash-lite-latest"
        # Match both hosted Gemma 4 naming forms: gemma-4-* and gemma4-*.
        or re.search(r"gemma-?4", model_id) is not None
    )


def to_google_thinking_level(level: ResolvedGoogleThinkingLevel) -> GoogleThinkingLevel:
    if level == "minimal":
        return "MINIMAL"
    if level == "low":
        return "LOW"
    if level == "medium":
        return "MEDIUM"
    return "HIGH"


def to_google_sdk_thinking_level(level: GoogleThinkingLevel) -> Any:
    """The SDK's own enum member for a wire level, when the SDK is installed.

    pi maps through `@google/genai`'s `ThinkingLevel`; the Python SDK's `types.ThinkingLevel`
    is a `str` enum with the same members, and its config validation accepts the plain string
    too, so the string stands in when the SDK is absent."""
    try:
        from google.genai import types as genai_types
    except ImportError:  # optional extra: misaka[google]
        return level
    return getattr(genai_types.ThinkingLevel, level, level)


def get_disabled_google_thinking_config(model: Model) -> dict[str, Any]:
    if not uses_google_thinking_level(model):
        return {"thinkingBudget": 0}
    fallback = clamp_thinking_level(model, "off")
    if fallback == "off":
        return {"thinkingBudget": 0}
    resolved_level = resolve_google_thinking_level(model, fallback)
    api_level = to_google_thinking_level(resolved_level)
    return {"thinkingLevel": to_google_sdk_thinking_level(api_level)}


def requires_tool_call_id(model_id: str) -> bool:
    major = _get_gemini_major_version(model_id)
    return model_id.startswith(("claude-", "gpt-oss-")) or (major is not None and major >= 3)


def _get_gemini_major_version(model_id: str) -> int | None:
    match = re.match(r"^gemini(?:-live)?-(\d+)", model_id.lower())
    if not match:
        return None
    return int(match.group(1))


def _supports_multimodal_function_response(model_id: str) -> bool:
    major = _get_gemini_major_version(model_id)
    if major is not None:
        return major >= 3
    return True


def convert_messages(model: Model, context: TranscriptContext) -> list[dict[str, Any]]:
    # Gemini has no mid-conversation system messages; the leading prompt is sent as systemInstruction.
    conversation = without_initial_system_message(collapse_system_messages(context).messages)
    contents: list[dict[str, Any]] = []

    def normalize_tool_call_id(tool_call_id: str, _target_model: Model, _source: Any) -> str:
        if not requires_tool_call_id(model.id):
            return tool_call_id
        return re.sub(r"[^a-zA-Z0-9_-]", "_", tool_call_id)[:64]

    transformed_messages = transform_messages(conversation, model, normalize_tool_call_id)
    for message in transformed_messages:
        if message.role == "user":
            if isinstance(message.content, str):
                contents.append({"role": "user", "parts": [{"text": sanitize_surrogates(message.content)}]})
            else:
                parts: list[dict[str, Any]] = []
                for item in message.content:
                    if item.type == "text":
                        parts.append({"text": sanitize_surrogates(item.text)})
                    else:
                        parts.append({"inlineData": {"mimeType": item.mimeType, "data": item.data}})
                if parts:
                    contents.append({"role": "user", "parts": parts})
            continue

        if message.role == "assistant":
            parts: list[dict[str, Any]] = []
            # Check if message is from same provider and model - only then keep thinking blocks
            is_same_provider_and_model = message.provider == model.provider and message.model == model.id

            for block in message.content:
                if block.type == "text":
                    thought_signature = _resolve_thought_signature(is_same_provider_and_model, block.textSignature)
                    # Skip empty text blocks — unless they carry a thought signature. Gemini can attach
                    # the signature to a part whose visible text is empty and requires it echoed back;
                    # dropping it breaks the reasoning chain and the model intermittently ends mid-task
                    # turns with a thought-only STOP (empty completion, no tool call).
                    if (not block.text or not block.text.strip()) and not thought_signature:
                        continue
                    part: dict[str, Any] = {"text": sanitize_surrogates(block.text)}
                    if thought_signature:
                        part["thoughtSignature"] = thought_signature
                    parts.append(part)
                    continue

                if block.type == "thinking":
                    # Only keep as thinking block if same provider AND same model
                    # Otherwise convert to plain text (no tags to avoid model mimicking them)
                    if is_same_provider_and_model:
                        thought_signature = _resolve_thought_signature(
                            is_same_provider_and_model, block.thinkingSignature
                        )
                        # Same rule as text blocks: an empty thinking block is dropped only when it
                        # carries no signature (mirrors the anthropic converter's handling).
                        if (not block.thinking or not block.thinking.strip()) and not thought_signature:
                            continue
                        part = {"thought": True, "text": sanitize_surrogates(block.thinking)}
                        if thought_signature:
                            part["thoughtSignature"] = thought_signature
                        parts.append(part)
                    else:
                        # Cross-provider/model: the signature is unusable, empty blocks stay dropped.
                        if not block.thinking or not block.thinking.strip():
                            continue
                        parts.append({"text": sanitize_surrogates(block.thinking)})
                    continue

                if block.type == "toolCall":
                    thought_signature = _resolve_thought_signature(is_same_provider_and_model, block.thoughtSignature)
                    function_call: dict[str, Any] = {"name": block.name, "args": block.arguments or {}}
                    if requires_tool_call_id(model.id):
                        function_call["id"] = block.id
                    part = {"functionCall": function_call}
                    if thought_signature:
                        part["thoughtSignature"] = thought_signature
                    parts.append(part)

            if parts:
                contents.append({"role": "model", "parts": parts})
            continue

        if message.role == "toolResult":
            text_result = "\n".join(block.text for block in message.content if block.type == "text")
            image_content = (
                [block for block in message.content if isinstance(block, ImageContent) or block.type == "image"]
                if "image" in model.input
                else []
            )

            has_text = len(text_result) > 0
            has_images = len(image_content) > 0
            supports_multimodal = _supports_multimodal_function_response(model.id)
            response_value = sanitize_surrogates(text_result) if has_text else "(see attached image)" if has_images else ""
            image_parts = [
                {"inlineData": {"mimeType": image_block.mimeType, "data": image_block.data}}
                for image_block in image_content
            ]

            function_response: dict[str, Any] = {
                "name": message.toolName,
                "response": {"error": response_value} if message.isError else {"output": response_value},
            }
            if has_images and supports_multimodal:
                function_response["parts"] = image_parts
            if requires_tool_call_id(model.id):
                function_response["id"] = message.toolCallId

            function_response_part = {"functionResponse": function_response}

            last_content = contents[-1] if contents else None
            if (
                isinstance(last_content, dict)
                and last_content.get("role") == "user"
                and any(isinstance(part, dict) and part.get("functionResponse") for part in last_content.get("parts", []))
            ):
                last_content["parts"].append(function_response_part)
            else:
                contents.append({"role": "user", "parts": [function_response_part]})

            if has_images and not supports_multimodal:
                contents.append({"role": "user", "parts": [{"text": "Tool result image:"}, *image_parts]})

    return contents


def sanitize_for_openapi(schema: Any) -> Any:
    if not isinstance(schema, dict):
        if isinstance(schema, list):
            return schema
        return schema

    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _JSON_SCHEMA_META_DECLARATIONS:
            continue
        result[key] = sanitize_for_openapi(value)
    return result


def convert_tools(
    tools: list[Tool],
    use_parameters: bool = False,
    supports_strict_mode: bool = True,
) -> list[dict[str, Any]] | None:
    if not tools:
        return None

    declarations: list[dict[str, Any]] = []
    for tool in tools:
        strict = resolve_json_schema_strict_sampling(tool, supports_strict_mode)
        parameters = get_json_schema_tool_parameters(tool, strict)
        declaration: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
        }
        if use_parameters:
            declaration["parameters"] = sanitize_for_openapi(copy.deepcopy(parameters))
        else:
            declaration["parametersJsonSchema"] = copy.deepcopy(parameters)
        declarations.append(declaration)

    return [{"functionDeclarations": declarations}]


def map_tool_choice(choice: str) -> str:
    if choice == "none":
        return "NONE"
    if choice == "any":
        return "ANY"
    return "AUTO"


def _option(options: Any, name: str) -> Any:
    if options is None:
        return None
    if isinstance(options, Mapping):
        return options.get(name)
    return getattr(options, name, None)


def supports_google_strict_tool_sampling(model_id: str) -> bool:
    """Gemini 3 and later accept a constrained tool schema; earlier versions reject it."""
    major = _get_gemini_major_version(model_id)
    return major is not None and major >= 3


def resolve_google_function_calling_mode(
    tools: list[Tool], tool_choice: str | None, supports_strict_mode: bool
) -> str | None:
    """Which function-calling mode this request asks Google for.

    ``VALIDATED`` is Google's side of strict sampling: sending the constrained schema is
    only half of it, the mode has to ask for validation too. An explicit ``none``/``any``
    from the caller wins over that, as upstream's does.
    """
    use_strict_mode = any(
        resolve_json_schema_strict_sampling(tool, supports_strict_mode) is True for tool in tools
    )
    if tool_choice in ("none", "any"):
        return map_tool_choice(tool_choice)
    if use_strict_mode:
        return "VALIDATED"
    return map_tool_choice(tool_choice) if tool_choice else None


async def retry_google_request(request, options: Any = None):
    """Run a Google request under the shared retry policy.

    Upstream wraps this same call in ``retryProviderRequest`` and normalizes the error
    first: the Google SDK's ``ApiError`` carries ``status`` but no ``headers``, and pi's
    gate requires literally both property names. This port's gate asks
    ``provider_error_status`` instead -- because the Python SDKs raise ``status_code`` and
    keep headers on ``response``, so a name check was never going to work here -- and that
    already recognises the Google shape. The normalization has nothing left to do, so it
    is not carried over; the retry is the part that was missing.
    """
    return await retry_provider_request(
        request,
        max_retries=_option(options, "maxRetries") or 0,
        max_retry_delay_ms=_option(options, "maxRetryDelayMs"),
        signal=_option(options, "signal"),
    )


async def open_content_stream(client: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    """``(the SDK's stream, its first chunk or None)`` with the request sent. pi's JS SDK sends it
    when ``generateContentStream`` is called; the Python SDK's call returns a generator that sends
    it on the first read, so a refusal surfaced inside the read loop -- after "start", and past
    ``retry_google_request``, which never retried it (0.18.9 sweep)."""
    google_stream = await client.aio.models.generate_content_stream(**params)
    try:
        first = await google_stream.__anext__()
    except StopAsyncIteration:
        first = None
    return google_stream, first


async def chunks_from(first: Any, google_stream: Any):
    """The stream ``open_content_stream`` opened, its first chunk put back in front."""
    if first is not None:
        yield first
    async for chunk in google_stream:
        yield chunk


def map_stop_reason(reason: Any) -> StopReason:
    name = getattr(reason, "name", None)
    if isinstance(name, str):
        normalized = name
    elif isinstance(reason, str):
        normalized = reason.split(".")[-1]
    else:
        normalized = str(reason)

    if normalized == "STOP":
        return "stop"
    if normalized == "MAX_TOKENS":
        return "length"
    if normalized in {
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "SAFETY",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
        "IMAGE_OTHER",
        "RECITATION",
        "FINISH_REASON_UNSPECIFIED",
        "OTHER",
        "LANGUAGE",
        "MALFORMED_FUNCTION_CALL",
        "UNEXPECTED_TOOL_CALL",
        "TOO_MANY_TOOL_CALLS",
        "NO_IMAGE",
    }:
        return "error"

    raise RuntimeError(f"Unhandled stop reason: {normalized}")


__all__ = [
    "GoogleThinkingLevel",
    ]
