"""OpenAI Codex Responses provider adapter (pi ``api/openai-codex-responses.ts``)."""

from __future__ import annotations

import asyncio
import base64
import importlib
import json
import math
import re
import time
from collections.abc import AsyncIterable, AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field, replace
from email.utils import parsedate_to_datetime
from typing import Any, Literal, TypedDict
from urllib.parse import urlparse

import httpx

from misaka.agent.harness.session.uuid import uuidv7
from misaka.ai.models import clamp_thinking_level
from misaka.ai.providers._common import _create_abort_wait_task, _option
from misaka.ai.providers.constrained_sampling import (
    create_grammar_tool_input_properties,
)
from misaka.ai.providers.openai_prompt_cache import clamp_openai_prompt_cache_key
from misaka.ai.providers.openai_responses_shared import (
    convert_responses_messages,
    convert_responses_tools,
    process_responses_stream,
)
from misaka.ai.providers.simple_options import build_base_options
from misaka.ai.session_resources import register_session_resource_cleanup
from misaka.ai.types import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    Model,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
    TranscriptContext,
    Usage,
)
from misaka.ai.utils.diagnostics import (
    append_assistant_message_diagnostic,
    create_assistant_message_diagnostic,
    format_thrown_value,
)
from misaka.ai.utils.error_body import format_provider_error, normalize_provider_error
from misaka.ai.utils.event_stream import AssistantMessageEventStream, spawn_stream_task
from misaka.ai.utils.headers import headers_to_record
from misaka.ai.utils.text import get_system_message_text
from misaka.ai.utils.transcript import (
    get_declared_tools,
    get_initial_system_message,
    normalize_context,
    resolve_transcript,
    resolve_transcript_tools,
)
from misaka.ai.utils.user_agent import get_misaka_user_agent
from misaka.core.provider_attribution import CLIENT_NAME
from misaka.utils.values import maybe_await, read_field, signal_aborted

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api"
JWT_CLAIM_PATH = "https://api.openai.com/auth"
DEFAULT_MAX_RETRIES = 0
BASE_DELAY_MS = 1000
DEFAULT_MAX_RETRY_DELAY_MS = 60_000
DEFAULT_WEBSOCKET_CONNECT_TIMEOUT_MS = 15_000
# The Codex backend accepts zstd-compressed request bodies on the SSE responses
# endpoint (the same endpoint the official Codex client compresses against).
REQUEST_COMPRESSION_ZSTD_LEVEL = 3
CODEX_TOOL_CALL_PROVIDERS = {"openai", "openai-codex", "opencode"}
WEBSOCKET_MESSAGE_TOO_BIG_CLOSE_CODE = 1009
WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE = "websocket_connection_limit_reached"
PREVIOUS_RESPONSE_NOT_FOUND_CODE = "previous_response_not_found"

CODEX_RESPONSE_STATUSES = {"completed", "incomplete", "failed", "cancelled", "queued", "in_progress"}

# ============================================================================
# Types
# ============================================================================


class OpenAICodexResponsesOptions(TypedDict, total=False):
    apiKey: str
    headers: dict[str, str | None]
    env: dict[str, str]
    signal: Any
    sessionId: str
    cacheRetention: str
    transport: str
    onPayload: Any
    onResponse: Any
    onProviderStreamEvent: Any
    timeoutMs: int
    websocketConnectTimeoutMs: int
    maxRetries: int
    maxRetryDelayMs: int
    reasoningEffort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
    reasoningSummary: Literal["auto", "concise", "detailed", "off", "on"] | None
    serviceTier: str
    textVerbosity: Literal["low", "medium", "high"]
    toolChoice: Literal["auto", "none", "required"]


def assert_successful_output(output: AssistantMessage) -> None:
    if output.stopReason == "pending":
        raise RuntimeError("Codex stream ended without a stop reason")
    if output.stopReason in {"error", "aborted"}:
        raise RuntimeError(output.errorMessage or "An unknown error occurred")


# ============================================================================
# Retry Helpers
# ============================================================================

_TERMINAL_RATE_LIMIT_PATTERN = re.compile(
    r"GoUsageLimitError|FreeUsageLimitError|Monthly usage limit reached|available balance|insufficient_quota"
    r"|out of budget|quota exceeded|billing",
    re.IGNORECASE,
)
_RETRYABLE_ERROR_PATTERN = re.compile(
    r"rate.?limit|overloaded|service.?unavailable|upstream.?connect|connection.?refused", re.IGNORECASE
)


def is_terminal_rate_limit_error(error_text: str) -> bool:
    return bool(_TERMINAL_RATE_LIMIT_PATTERN.search(error_text))


def is_retryable_error(status: int, error_text: str) -> bool:
    if status == 429 and is_terminal_rate_limit_error(error_text):
        return False
    if status in {429, 500, 502, 503, 504}:
        return True
    return bool(_RETRYABLE_ERROR_PATTERN.search(error_text))


def get_retry_after_delay_ms(headers: Mapping[str, str]) -> float | None:
    retry_after_ms = headers.get("retry-after-ms")
    if retry_after_ms is not None:
        try:
            millis = float(retry_after_ms)
        except ValueError:
            millis = math.nan
        if math.isfinite(millis):
            return max(0, millis)

    retry_after = headers.get("retry-after")
    if not retry_after:
        return None

    try:
        seconds = float(retry_after)
    except ValueError:
        seconds = math.nan
    if math.isfinite(seconds):
        return max(0, seconds * 1000)

    try:
        date = parsedate_to_datetime(retry_after).timestamp()
    except (TypeError, ValueError, OverflowError):
        return None
    return max(0, date * 1000 - time.time() * 1000)


class RetryDelayExceededError(RuntimeError):
    pass


def validate_retry_delay_ms(delay_ms: float, options: Any = None) -> float:
    max_retry_delay_ms = _option(options, "maxRetryDelayMs", DEFAULT_MAX_RETRY_DELAY_MS)
    if max_retry_delay_ms > 0 and delay_ms > max_retry_delay_ms:
        raise RetryDelayExceededError(
            f"Server requested {math.ceil(delay_ms / 1000)}s retry delay (max: {math.ceil(max_retry_delay_ms / 1000)}s)"
        )
    return delay_ms


async def sleep(ms: float, signal: Any = None) -> None:
    await _await_with_abort(asyncio.sleep(ms / 1000), signal)


def normalize_timeout_ms(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise RuntimeError(f"Invalid timeoutMs: {value}")
    return math.floor(value)


# ============================================================================
# Request Compression
# ============================================================================


def compress_request_body_zstd(body_json: str) -> bytes | None:
    """The zstd-compressed body bytes, or ``None`` when compression is unavailable.

    MISAKA fork: pi reaches zstd through ``node:zlib``. Python's zlib has no zstd; the codec
    is ``compression.zstd`` from 3.14 and a third-party package before that. Callers fall
    back to sending the uncompressed JSON when this returns ``None``, as upstream does.
    """
    try:
        from compression import zstd  # type: ignore[import-not-found]

        return zstd.compress(body_json.encode("utf-8"), level=REQUEST_COMPRESSION_ZSTD_LEVEL)
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 - compression is best effort; the plain body still works
        return None
    try:
        import zstandard  # type: ignore[import-not-found]

        return zstandard.ZstdCompressor(level=REQUEST_COMPRESSION_ZSTD_LEVEL).compress(body_json.encode("utf-8"))
    except Exception:  # noqa: BLE001 - as above, and ImportError when the package is absent
        return None


# ============================================================================
# Main Stream Function
# ============================================================================


def stream_openai_codex_responses(
    model: Model,
    context: TranscriptContext,
    options: StreamOptions | dict[str, Any] | None = None,
) -> AssistantMessageEventStream:
    stream = AssistantMessageEventStream()
    normalized_context = resolve_transcript(context, read_field(model.compat, "supportsMidConvoSystemMessages"))

    async def run() -> None:
        output = AssistantMessage(
            content=[],
            api="openai-codex-responses",
            provider=model.provider,
            model=model.id,
            usage={
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0,
                "totalTokens": 0,
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
            },
            stopReason="pending",
            timestamp=time.time_ns() // 1_000_000,
        )

        response: httpx.Response | None = None
        try:
            api_key = _option(options, "apiKey")
            if not api_key:
                raise RuntimeError(f"No API key for provider: {model.provider}")

            account_id = extract_account_id(api_key)
            grammar_tool_input_properties = create_grammar_tool_input_properties(
                get_declared_tools(normalized_context.messages),
                read_field(model.compat, "supportsOpenAIGrammarTools", False),
            )
            cache_session_id = None if _option(options, "cacheRetention") == "none" else _option(options, "sessionId")
            codex_session_id = clamp_openai_prompt_cache_key(cache_session_id)
            body = build_request_body(
                model, normalized_context, options, codex_session_id, grammar_tool_input_properties
            )
            on_payload = _option(options, "onPayload")
            if callable(on_payload):
                next_body = await maybe_await(on_payload(body, model))
                if next_body is not None:
                    body = next_body
            websocket_request_id = codex_session_id or uuidv7()
            sse_headers = build_sse_headers(model.headers, _option(options, "headers"), account_id, api_key, codex_session_id)
            websocket_headers = build_websocket_headers(
                model.headers,
                _option(options, "headers"),
                account_id,
                api_key,
                websocket_request_id,
            )
            body_json = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
            http_timeout_ms = normalize_timeout_ms(_option(options, "timeoutMs"))
            websocket_connect_timeout_ms = normalize_timeout_ms(_option(options, "websocketConnectTimeoutMs"))
            transport = _option(options, "transport") or "auto"
            start_emitted = False
            websocket_disabled_for_session = transport != "sse" and is_websocket_sse_fallback_active(cache_session_id)
            if websocket_disabled_for_session:
                record_websocket_sse_fallback(cache_session_id)

            if transport != "sse" and not websocket_disabled_for_session:
                websocket_started = False
                retried_websocket_connection_limit = False
                retried_missing_websocket_continuation = False
                while True:
                    websocket_started = False

                    def on_start() -> None:
                        nonlocal websocket_started, start_emitted
                        websocket_started = True
                        if not start_emitted:
                            start_emitted = True
                            stream.push(StartEvent(partial=output))

                    try:
                        await process_websocket_stream(
                            resolve_codex_websocket_url(model.baseUrl),
                            body,
                            websocket_headers,
                            output,
                            stream,
                            model,
                            on_start,
                            http_timeout_ms,
                            websocket_connect_timeout_ms,
                            cache_session_id,
                            account_id,
                            grammar_tool_input_properties,
                            options,
                        )

                        if signal_aborted(_option(options, "signal")):
                            raise RuntimeError("Request was aborted")
                        assert_successful_output(output)
                        stream.push(DoneEvent(reason=output.stopReason, message=output))
                        return
                    except Exception as error:
                        aborted = signal_aborted(_option(options, "signal"))
                        connection_limit_before_start = (
                            not websocket_started and is_websocket_connection_limit_reached_error(error)
                        )
                        previous_response_not_found = is_previous_response_not_found_error(error)
                        if not aborted and previous_response_not_found and not retried_missing_websocket_continuation:
                            retried_missing_websocket_continuation = True
                            continue
                        if not aborted and connection_limit_before_start and not retried_websocket_connection_limit:
                            retried_websocket_connection_limit = True
                            continue
                        if aborted or (is_codex_non_transport_error(error) and not connection_limit_before_start):
                            raise
                        append_assistant_message_diagnostic(
                            output,
                            create_assistant_message_diagnostic(
                                "provider_transport_failure",
                                error,
                                {
                                    "configuredTransport": transport,
                                    **({} if websocket_started else {"fallbackTransport": "sse"}),
                                    "eventsEmitted": websocket_started,
                                    "phase": "after_message_stream_start"
                                    if websocket_started
                                    else "before_message_stream_start",
                                    "requestBytes": len(body_json.encode("utf-8")),
                                },
                            ),
                        )
                        record_websocket_failure(cache_session_id, error)
                        if websocket_started:
                            raise
                        record_websocket_sse_fallback(cache_session_id)
                        break

            # Compress the request body once for the SSE path. The Codex backend
            # decodes Content-Encoding: zstd; the WebSocket transport above sends the
            # uncompressed JSON frame, matching the official Codex client.
            compressed_body = compress_request_body_zstd(body_json)
            if compressed_body is not None:
                sse_headers["content-encoding"] = "zstd"
            sse_body: bytes = compressed_body if compressed_body is not None else body_json.encode("utf-8")

            # Fetch with retry logic for rate limits and transient errors
            last_error: BaseException | None = None
            max_retries = _option(options, "maxRetries", DEFAULT_MAX_RETRIES)
            signal = _option(options, "signal")

            # MISAKA fork: pi's `fetch` has no read timeout; the header wait below is its only
            # bound. httpx needs its read leg switched off to behave the same.
            async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(None)) as client:
                attempt = 0
                while attempt <= max_retries:
                    if signal_aborted(signal):
                        raise RuntimeError("Request was aborted")

                    try:
                        request = client.build_request("POST", resolve_codex_url(model.baseUrl), headers=sse_headers,
                                                       content=sse_body)
                        send = client.send(request, stream=True)
                        try:
                            response = await _await_with_abort(
                                asyncio.wait_for(send, http_timeout_ms / 1000)
                                if http_timeout_ms is not None and http_timeout_ms > 0
                                else send,
                                signal,
                            )
                        except TimeoutError as error:
                            raise RuntimeError(
                                f"Codex SSE response headers timed out after {http_timeout_ms}ms"
                            ) from error
                        on_response = _option(options, "onResponse")
                        if callable(on_response):
                            await maybe_await(
                                on_response(
                                    {"status": response.status_code, "headers": headers_to_record(response.headers)},
                                    model,
                                )
                            )

                        if response.is_success:
                            break

                        error_text = (await response.aread()).decode("utf-8", errors="replace")
                        await response.aclose()
                        if attempt < max_retries and is_retryable_error(response.status_code, error_text):
                            retry_after_delay_ms = get_retry_after_delay_ms(response.headers)
                            delay_ms = (
                                BASE_DELAY_MS * 2**attempt
                                if retry_after_delay_ms is None
                                else validate_retry_delay_ms(retry_after_delay_ms, options)
                            )

                            await sleep(delay_ms, signal)
                            attempt += 1
                            continue

                        # Parse error for friendly message on final attempt or non-retryable error
                        info = parse_error_response(error_text, response.status_code, response.reason_phrase)
                        raise RuntimeError(info.get("friendlyMessage") or info["message"])
                    except Exception as error:
                        if str(error) == "Request was aborted" or isinstance(error, asyncio.CancelledError):
                            raise RuntimeError("Request was aborted") from error
                        last_error = error
                        # Network errors are retryable
                        if (
                            attempt < max_retries
                            and not isinstance(last_error, RetryDelayExceededError)
                            and "usage limit" not in str(last_error)
                        ):
                            delay_ms = BASE_DELAY_MS * 2**attempt
                            await sleep(delay_ms, signal)
                            attempt += 1
                            continue
                        raise

                if response is None or not response.is_success:
                    raise last_error or RuntimeError("Failed after retries")

                if not start_emitted:
                    start_emitted = True
                    stream.push(StartEvent(partial=output))
                await process_stream(response, output, stream, model, grammar_tool_input_properties, options)

                if signal_aborted(signal):
                    raise RuntimeError("Request was aborted")

                assert_successful_output(output)
                stream.push(DoneEvent(reason=output.stopReason, message=output))
        except Exception as error:  # noqa: BLE001 - every failure becomes an error event on the stream
            output.stopReason = "aborted" if signal_aborted(_option(options, "signal")) else "error"
            output.errorMessage = format_provider_error(normalize_provider_error(error))
            stream.push(ErrorEvent(reason=output.stopReason, error=output), cause=error)
        finally:
            if response is not None:
                await response.aclose()
            stream.end()

    spawn_stream_task(run(), stream=stream)
    return stream


def stream_simple_openai_codex_responses(
    model: Model,
    context: TranscriptContext,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    api_key = _option(options, "apiKey")
    if not api_key:
        raise RuntimeError(f"No API key for provider: {model.provider}")

    base = {
        **build_base_options(model, context, options, api_key).model_dump(),
        "toolChoice": _option(options, "toolChoice"),
    }
    clamped_reasoning = clamp_thinking_level(model, options.reasoning) if options and options.reasoning else None
    reasoning_effort = None if clamped_reasoning == "off" else clamped_reasoning

    return stream_openai_codex_responses(model, context, {**base, "reasoningEffort": reasoning_effort})


# ============================================================================
# Request Building
# ============================================================================


def build_request_body(
    model: Model,
    context: TranscriptContext,
    options: Any,
    cache_session_id: str | None,
    grammar_tool_input_properties: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    supports_strict_mode = read_field(model.compat, "supportsStrictMode", True)
    supports_openai_grammar_tools = read_field(model.compat, "supportsOpenAIGrammarTools", False)
    supports_additional_tools = read_field(model.compat, "supportsAdditionalTools", False)
    supports_tool_search = read_field(model.compat, "supportsToolSearch", False)
    if grammar_tool_input_properties is None:
        grammar_tool_input_properties = create_grammar_tool_input_properties(
            get_declared_tools(context.messages), supports_openai_grammar_tools
        )
    transcript_tools = resolve_transcript_tools(context.messages, supports_additional_tools or supports_tool_search)
    messages = convert_responses_messages(
        model,
        context,
        CODEX_TOOL_CALL_PROVIDERS,
        {
            "includeSystemPrompt": False,
            "grammarToolInputProperties": grammar_tool_input_properties,
            "supportsMidConvoSystemMessages": read_field(model.compat, "supportsMidConvoSystemMessages", False),
            "supportsAdditionalTools": supports_additional_tools,
            "supportsToolSearch": supports_tool_search,
            "toolOptions": {
                "strict": None,
                "supportsStrictMode": supports_strict_mode,
                "supportsOpenAIGrammarTools": supports_openai_grammar_tools,
            },
        },
    )

    initial_system_message = get_initial_system_message(context.messages)
    instructions = get_system_message_text(initial_system_message) if initial_system_message else ""
    body: dict[str, Any] = {
        "model": model.id,
        "store": False,
        "stream": True,
        "instructions": instructions or "You are a helpful assistant.",
        "input": messages,
        "text": {"verbosity": _option(options, "textVerbosity") or "low"},
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": cache_session_id,
        "tool_choice": _option(options, "toolChoice", "auto"),
        "parallel_tool_calls": True,
    }

    if _option(options, "temperature") is not None:
        body["temperature"] = _option(options, "temperature")

    if _option(options, "serviceTier") is not None:
        body["service_tier"] = _option(options, "serviceTier")

    if len(transcript_tools.requestTools) > 0:
        body["tools"] = convert_responses_tools(
            transcript_tools.requestTools,
            {
                "strict": None,
                "supportsStrictMode": supports_strict_mode,
                "supportsOpenAIGrammarTools": supports_openai_grammar_tools,
            },
        )

    reasoning_effort = _option(options, "reasoningEffort")
    thinking_level_map = model.thinkingLevelMap or {}
    if reasoning_effort is not None:
        if reasoning_effort == "none":
            effort = thinking_level_map.get("off", "none")
        else:
            mapped = thinking_level_map.get(reasoning_effort)
            effort = reasoning_effort if mapped is None else mapped
        if effort is not None:
            summary = _option(options, "reasoningSummary")
            body["reasoning"] = {"effort": effort, "summary": "auto" if summary is None else summary}
    elif model.reasoning and (model.thinkingLevelMap is None or thinking_level_map.get("off", ...) is not None):
        off_value = thinking_level_map.get("off")
        body["reasoning"] = {"effort": "none" if off_value is None else off_value}

    # MISAKA fork: pi's `undefined` fields are left out of the JSON; a Python `None` would be
    # sent as `null`.
    return {key: value for key, value in body.items() if value is not None}


def get_service_tier_cost_multiplier(model: Model, service_tier: str | None) -> float:
    if service_tier == "flex":
        return 0.5
    if service_tier == "priority":
        return 2.5 if model.id == "gpt-5.5" else 2
    return 1


def apply_service_tier_pricing(usage: Usage, service_tier: str | None, model: Model) -> None:
    multiplier = get_service_tier_cost_multiplier(model, service_tier)
    if multiplier == 1:
        return

    usage.cost.input *= multiplier
    usage.cost.output *= multiplier
    usage.cost.cacheRead *= multiplier
    usage.cost.cacheWrite *= multiplier
    usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cacheRead + usage.cost.cacheWrite


def resolve_codex_service_tier(response_service_tier: str | None, request_service_tier: str | None) -> str | None:
    if response_service_tier == "default" and request_service_tier in {"flex", "priority"}:
        return request_service_tier
    return request_service_tier if response_service_tier is None else response_service_tier


def resolve_codex_url(base_url: str | None = None) -> str:
    raw = base_url if base_url and base_url.strip() else DEFAULT_CODEX_BASE_URL
    normalized = raw.rstrip("/")
    if normalized.endswith("/codex/responses"):
        return normalized
    if normalized.endswith("/codex"):
        return f"{normalized}/responses"
    return f"{normalized}/codex/responses"


def resolve_codex_websocket_url(base_url: str | None = None) -> str:
    url = urlparse(resolve_codex_url(base_url))
    scheme = "wss" if url.scheme == "https" else "ws" if url.scheme == "http" else url.scheme
    return url._replace(scheme=scheme).geturl()


# ============================================================================
# Response Processing
# ============================================================================


async def process_stream(
    response: httpx.Response,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    model: Model,
    grammar_tool_input_properties: Mapping[str, str],
    options: Any = None,
) -> None:
    await process_responses_stream(
        map_codex_events(
            parse_sse(response, _option(options, "signal")), output, model, _option(options, "onProviderStreamEvent")
        ),
        output,
        stream,
        model,
        {
            "serviceTier": _option(options, "serviceTier"),
            "grammarToolInputProperties": grammar_tool_input_properties,
            "resolveServiceTier": resolve_codex_service_tier,
            "applyServiceTierPricing": lambda usage, service_tier: apply_service_tier_pricing(usage, service_tier, model),
        },
    )


class CodexApiError(RuntimeError):
    def __init__(self, message: str, *, code: str | None = None, payload: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.payload = payload


class CodexProtocolError(RuntimeError):
    def __init__(self, message: str, *, payload: Any = None) -> None:
        super().__init__(message)
        self.payload = payload


class ProviderStreamEventCallbackError(RuntimeError):
    def __init__(self, cause: BaseException) -> None:
        super().__init__(format_thrown_value(cause))
        self.__cause__ = cause


def is_codex_non_transport_error(error: Any) -> bool:
    return isinstance(error, (CodexApiError, CodexProtocolError, ProviderStreamEventCallbackError))


def is_websocket_connection_limit_reached_error(error: Any) -> bool:
    return isinstance(error, CodexApiError) and error.code == WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE


def is_previous_response_not_found_error(error: Any) -> bool:
    return isinstance(error, CodexApiError) and error.code == PREVIOUS_RESPONSE_NOT_FOUND_CODE


def extract_codex_event_error(event: Mapping[str, Any]) -> dict[str, str | None]:
    nested = event.get("error") if isinstance(event.get("error"), Mapping) else None
    code = event.get("code")
    message = event.get("message")
    return {
        "code": code if isinstance(code, str) else nested.get("code") if nested and isinstance(nested.get("code"), str) else None,
        "message": message
        if isinstance(message, str)
        else nested.get("message")
        if nested and isinstance(nested.get("message"), str)
        else None,
    }


async def map_codex_events(
    events: AsyncIterable[dict[str, Any]],
    output: AssistantMessage,
    model: Model,
    on_provider_stream_event: Any = None,
) -> AsyncIterator[dict[str, Any]]:
    async for event in events:
        try:
            if callable(on_provider_stream_event):
                await maybe_await(on_provider_stream_event(event, model))
        except Exception as error:
            # Keep callback failures out of Codex's WebSocket retry and SSE fallback path.
            raise ProviderStreamEventCallbackError(error) from error
        event_type = event.get("type") if isinstance(event.get("type"), str) else None
        if not event_type:
            continue

        if event_type == "error":
            error = extract_codex_event_error(event)
            raise CodexApiError(
                f"Codex error: {error['message'] or error['code'] or json.dumps(event)}",
                code=error["code"],
                payload=event,
            )

        if event_type == "response.failed":
            response = event.get("response") if isinstance(event.get("response"), Mapping) else {}
            response_error = response.get("error") if isinstance(response.get("error"), Mapping) else {}
            raise CodexApiError(
                response_error.get("message") or "Codex response failed",
                code=response_error.get("code"),
                payload=event,
            )

        if event_type in {"response.done", "response.completed", "response.incomplete"}:
            response = event.get("response")
            if isinstance(response, Mapping) and isinstance(response.get("end_turn"), bool):
                output.endTurn = response["end_turn"]
            normalized_response = (
                {**response, "status": normalize_codex_status(response.get("status"))}
                if isinstance(response, Mapping)
                else response
            )
            yield {**event, "type": "response.completed", "response": normalized_response}
            return

        yield event


def normalize_codex_status(status: Any) -> str | None:
    if not isinstance(status, str):
        return None
    return status if status in CODEX_RESPONSE_STATUSES else None


# ============================================================================
# SSE Parsing
# ============================================================================


async def parse_sse(response: httpx.Response, signal: Any = None) -> AsyncIterator[dict[str, Any]]:
    buffer = ""
    iterator = response.aiter_text().__aiter__()
    while True:
        if signal_aborted(signal):
            raise RuntimeError("Request was aborted")
        try:
            chunk = await _await_with_abort(iterator.__anext__(), signal, on_abort=lambda: response.aclose())
            done = False
        except StopAsyncIteration:
            chunk = ""
            done = True
        if signal_aborted(signal):
            raise RuntimeError("Request was aborted")
        buffer += chunk
        # Treat EOF as terminating the residual SSE frame.
        if done and buffer.strip():
            buffer += "\n\n"

        index = buffer.find("\n\n")
        while index != -1:
            frame = buffer[:index]
            buffer = buffer[index + 2 :]

            data_lines = [line[5:].strip() for line in frame.split("\n") if line.startswith("data:")]
            if len(data_lines) > 0:
                data = "\n".join(data_lines).strip()
                if data and data != "[DONE]":
                    try:
                        yield json.loads(data)
                    except ValueError as cause:
                        raise CodexProtocolError(
                            f"Invalid Codex SSE JSON: {format_thrown_value(cause)}", payload=data
                        ) from cause
            index = buffer.find("\n\n")

        if done:
            break


# ============================================================================
# WebSocket Parsing
# ============================================================================

OPENAI_BETA_RESPONSES_WEBSOCKETS = "responses_websockets=2026-02-06"
SESSION_WEBSOCKET_CACHE_TTL_MS = 5 * 60 * 1000
SESSION_WEBSOCKET_MAX_AGE_MS = 55 * 60 * 1000


@dataclass(slots=True)
class CachedWebSocketContinuationState:
    lastRequestBody: dict[str, Any]
    lastResponseId: str
    lastResponseItems: list[dict[str, Any]]


@dataclass(slots=True)
class CachedWebSocketConnection:
    socket: Any
    busy: bool
    createdAt: float
    idleTimer: asyncio.TimerHandle | None = None
    continuation: CachedWebSocketContinuationState | None = None


@dataclass(slots=True)
class OpenAICodexWebSocketDebugStats:
    requests: int = 0
    connectionsCreated: int = 0
    connectionsReused: int = 0
    cachedContextRequests: int = 0
    storeTrueRequests: int = 0
    fullContextRequests: int = 0
    deltaRequests: int = 0
    lastInputItems: int = 0
    lastDeltaInputItems: int | None = None
    lastPreviousResponseId: str | None = None
    websocketFailures: int = 0
    sseFallbacks: int = 0
    websocketFallbackActive: bool | None = None
    lastWebSocketError: str | None = None


websocket_session_cache: dict[str, dict[str, CachedWebSocketConnection]] = {}
websocket_debug_stats: dict[str, OpenAICodexWebSocketDebugStats] = {}
websocket_sse_fallback_sessions: set[str] = set()


def get_or_create_websocket_debug_stats(session_id: str) -> OpenAICodexWebSocketDebugStats:
    stats = websocket_debug_stats.get(session_id)
    if stats is None:
        stats = OpenAICodexWebSocketDebugStats()
        websocket_debug_stats[session_id] = stats
    return stats


def get_openai_codex_websocket_debug_stats(session_id: str) -> OpenAICodexWebSocketDebugStats | None:
    stats = websocket_debug_stats.get(session_id)
    return replace(stats) if stats is not None else None


def reset_openai_codex_websocket_debug_stats(session_id: str | None = None) -> None:
    if session_id:
        websocket_debug_stats.pop(session_id, None)
        websocket_sse_fallback_sessions.discard(session_id)
        return
    websocket_debug_stats.clear()
    websocket_sse_fallback_sessions.clear()


def close_openai_codex_websocket_sessions(session_id: str | None = None) -> None:
    def close_entry(entry: CachedWebSocketConnection) -> None:
        if entry.idleTimer is not None:
            entry.idleTimer.cancel()
        close_websocket_silently(entry.socket, 1000, "debug_close")

    if session_id:
        for entry in list((websocket_session_cache.get(session_id) or {}).values()):
            close_entry(entry)
        websocket_session_cache.pop(session_id, None)
        return
    for account_entries in websocket_session_cache.values():
        for entry in account_entries.values():
            close_entry(entry)
    websocket_session_cache.clear()


register_session_resource_cleanup(close_openai_codex_websocket_sessions)


def is_websocket_sse_fallback_active(session_id: str | None) -> bool:
    return session_id in websocket_sse_fallback_sessions if session_id else False


def record_websocket_sse_fallback(session_id: str | None) -> None:
    if not session_id:
        return
    stats = get_or_create_websocket_debug_stats(session_id)
    stats.sseFallbacks += 1
    stats.websocketFallbackActive = is_websocket_sse_fallback_active(session_id)


def record_websocket_failure(session_id: str | None, error: Any) -> None:
    if not session_id:
        return
    websocket_sse_fallback_sessions.add(session_id)

    stats = get_or_create_websocket_debug_stats(session_id)
    stats.websocketFailures += 1
    stats.lastWebSocketError = format_thrown_value(error)
    stats.websocketFallbackActive = True


_cached_websocket_connector: Callable[..., Any] | None = None


async def get_websocket_connector() -> Callable[..., Any] | None:
    """MISAKA fork: pi picks a `WebSocket` constructor (with a proxy shim under Bun); Python
    has none built in, so the ``websockets`` package's client stands in, when installed."""
    global _cached_websocket_connector
    if _cached_websocket_connector is not None:
        return _cached_websocket_connector
    try:
        module = importlib.import_module("websockets.asyncio.client")
    except ImportError:
        return None
    connector = getattr(module, "connect", None)
    if connector is None:
        return None
    _cached_websocket_connector = connector
    return connector


class WebSocketCloseError(RuntimeError):
    def __init__(
        self, message: str, *, code: int | None = None, reason: str | None = None, was_clean: bool | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.reason = reason
        self.wasClean = was_clean


def is_websocket_reusable(socket: Any) -> bool:
    state = getattr(socket, "state", None)
    state_name = getattr(state, "name", None)
    if isinstance(state_name, str):
        return state_name == "OPEN"
    # If the state is unavailable, assume the runtime keeps it open/reusable.
    return True


def is_websocket_session_expired(entry: CachedWebSocketConnection) -> bool:
    return time.monotonic() * 1000 - entry.createdAt >= SESSION_WEBSOCKET_MAX_AGE_MS


def close_websocket_silently(socket: Any, code: int = 1000, reason: str = "done") -> None:
    try:
        result = socket.close(code=code, reason=reason)
    except Exception:  # noqa: BLE001 - closing a dead websocket must not raise
        return
    if not hasattr(result, "__await__"):
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        try:
            asyncio.run(result)
        except Exception:  # noqa: BLE001 - as above
            return
    else:
        loop.create_task(result)


def schedule_session_websocket_expiry(session_id: str, account_id: str, entry: CachedWebSocketConnection) -> None:
    if entry.idleTimer is not None:
        entry.idleTimer.cancel()

    def expire() -> None:
        if entry.busy:
            return
        close_websocket_silently(entry.socket, 1000, "idle_timeout")
        account_entries = websocket_session_cache.get(session_id)
        if account_entries is not None and account_entries.get(account_id) is entry:
            del account_entries[account_id]
        if account_entries is not None and len(account_entries) == 0:
            websocket_session_cache.pop(session_id, None)

    entry.idleTimer = asyncio.get_running_loop().call_later(SESSION_WEBSOCKET_CACHE_TTL_MS / 1000, expire)


async def connect_websocket(
    url: str,
    headers: Mapping[str, str],
    signal: Any = None,
    connect_timeout_ms: int | None = DEFAULT_WEBSOCKET_CONNECT_TIMEOUT_MS,
) -> Any:
    connector = await get_websocket_connector()
    if connector is None:
        raise RuntimeError("WebSocket transport is not available in this runtime")
    if signal_aborted(signal):
        raise RuntimeError("Request was aborted")

    # pi deletes `OpenAI-Beta` from a record whose keys `Headers` has already lower-cased,
    # so the websocket beta header goes out; the same headers are sent here.
    connect_timeout_ms = DEFAULT_WEBSOCKET_CONNECT_TIMEOUT_MS if connect_timeout_ms is None else connect_timeout_ms
    try:
        return await _await_with_abort(
            maybe_await(
                connector(
                    url,
                    additional_headers=dict(headers),
                    max_size=None,
                    open_timeout=connect_timeout_ms / 1000 if connect_timeout_ms > 0 else None,
                )
            ),
            signal,
        )
    except TimeoutError as error:
        raise RuntimeError(f"WebSocket connect timeout after {connect_timeout_ms}ms") from error


@dataclass(slots=True)
class AcquiredWebSocket:
    socket: Any
    entry: CachedWebSocketConnection | None
    reused: bool
    release: Callable[..., None] = field(repr=False)


async def acquire_websocket(
    url: str,
    headers: Mapping[str, str],
    session_id: str | None,
    account_id: str,
    signal: Any = None,
    connect_timeout_ms: int | None = None,
) -> AcquiredWebSocket:
    if not session_id:
        socket = await connect_websocket(url, headers, signal, connect_timeout_ms)
        return AcquiredWebSocket(
            socket=socket, entry=None, reused=False, release=lambda keep=False: close_websocket_silently(socket)
        )

    account_entries = websocket_session_cache.get(session_id)
    cached = account_entries.get(account_id) if account_entries is not None else None
    if cached is not None:
        if cached.idleTimer is not None:
            cached.idleTimer.cancel()
            cached.idleTimer = None
        if not cached.busy and is_websocket_session_expired(cached):
            close_websocket_silently(cached.socket, 1000, "connection_age_limit")
            account_entries.pop(account_id, None)
            if len(account_entries) == 0:
                websocket_session_cache.pop(session_id, None)
        elif not cached.busy and is_websocket_reusable(cached.socket):
            cached.busy = True

            def release_cached(keep: bool = False) -> None:
                if not keep or not is_websocket_reusable(cached.socket):
                    close_websocket_silently(cached.socket)
                    current_entries = websocket_session_cache.get(session_id)
                    if current_entries is not None and current_entries.get(account_id) is cached:
                        del current_entries[account_id]
                    if current_entries is not None and len(current_entries) == 0:
                        websocket_session_cache.pop(session_id, None)
                    return
                cached.busy = False
                schedule_session_websocket_expiry(session_id, account_id, cached)

            return AcquiredWebSocket(socket=cached.socket, entry=cached, reused=True, release=release_cached)
        if cached.busy:
            socket = await connect_websocket(url, headers, signal, connect_timeout_ms)
            return AcquiredWebSocket(
                socket=socket, entry=None, reused=False, release=lambda keep=False: close_websocket_silently(socket)
            )
        if not is_websocket_reusable(cached.socket):
            close_websocket_silently(cached.socket)
            account_entries.pop(account_id, None)
            if len(account_entries) == 0:
                websocket_session_cache.pop(session_id, None)

    socket = await connect_websocket(url, headers, signal, connect_timeout_ms)
    entry = CachedWebSocketConnection(socket=socket, busy=True, createdAt=time.monotonic() * 1000)
    websocket_session_cache.setdefault(session_id, {})[account_id] = entry

    def release_entry(keep: bool = False) -> None:
        if not keep or not is_websocket_reusable(entry.socket):
            close_websocket_silently(entry.socket)
            if entry.idleTimer is not None:
                entry.idleTimer.cancel()
            current_entries = websocket_session_cache.get(session_id)
            if current_entries is not None and current_entries.get(account_id) is entry:
                del current_entries[account_id]
            if current_entries is not None and len(current_entries) == 0:
                websocket_session_cache.pop(session_id, None)
            return
        entry.busy = False
        schedule_session_websocket_expiry(session_id, account_id, entry)

    return AcquiredWebSocket(socket=socket, entry=entry, reused=False, release=release_entry)


def extract_websocket_error(error: Any) -> BaseException:
    if isinstance(error, BaseException) and str(error):
        return error
    return RuntimeError("WebSocket error")


def extract_websocket_close_error(error: Any) -> BaseException:
    code = getattr(error, "code", None)
    reason = getattr(error, "reason", None)
    rcvd = getattr(error, "rcvd", None)
    if code is None and rcvd is not None:
        code = getattr(rcvd, "code", None)
        reason = getattr(rcvd, "reason", None)
    code_text = f" {code}" if isinstance(code, int) else ""
    reason_text = f" {reason}" if isinstance(reason, str) and reason else ""
    if not reason_text and code == WEBSOCKET_MESSAGE_TOO_BIG_CLOSE_CODE:
        reason_text = " message too big"
    return WebSocketCloseError(
        f"WebSocket closed{code_text}{reason_text}".strip(),
        code=code if isinstance(code, int) else None,
        reason=reason if isinstance(reason, str) and reason else None,
    )


def decode_websocket_data(data: Any) -> str | None:
    if isinstance(data, str):
        return data
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data).decode("utf-8", errors="replace")
    return None


async def parse_websocket(
    socket: Any, signal: Any = None, idle_timeout_ms: int | None = None
) -> AsyncIterator[dict[str, Any]]:
    saw_completion = False
    while True:
        if signal_aborted(signal):
            raise RuntimeError("Request was aborted")
        try:
            receive = maybe_await(socket.recv())
            raw_message = await _await_with_abort(
                asyncio.wait_for(receive, idle_timeout_ms / 1000)
                if idle_timeout_ms is not None and idle_timeout_ms > 0
                else receive,
                signal,
            )
        except TimeoutError as error:
            close_websocket_silently(socket, 1000, "idle_timeout")
            raise RuntimeError(f"WebSocket idle timeout after {idle_timeout_ms}ms") from error
        except RuntimeError as error:
            if str(error) == "Request was aborted":
                raise
            raise extract_websocket_error(error) from error
        except Exception as error:
            if saw_completion:
                return
            if error.__class__.__name__.startswith("ConnectionClosed"):
                raise extract_websocket_close_error(error) from error
            raise extract_websocket_error(error) from error

        text = decode_websocket_data(raw_message)
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except ValueError as cause:
            raise CodexProtocolError(
                f"Invalid Codex WebSocket JSON: {format_thrown_value(cause)}", payload=text
            ) from cause
        event_type = parsed.get("type") if isinstance(parsed, dict) and isinstance(parsed.get("type"), str) else ""
        if event_type in {"response.completed", "response.done", "response.incomplete"}:
            saw_completion = True
        yield parsed
        if saw_completion:
            return


def request_body_without_input(body: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in body.items() if key not in {"input", "previous_response_id"}}


def response_inputs_equal(a: Any, b: Any) -> bool:
    return json.dumps(a or []) == json.dumps(b or [])


def request_bodies_match_except_input(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return json.dumps(request_body_without_input(a)) == json.dumps(request_body_without_input(b))


def get_cached_websocket_input_delta(
    body: Mapping[str, Any], continuation: CachedWebSocketContinuationState
) -> list[Any] | None:
    if not request_bodies_match_except_input(body, continuation.lastRequestBody):
        return None

    current_input = list(body.get("input") or [])
    baseline = [*(continuation.lastRequestBody.get("input") or []), *continuation.lastResponseItems]
    if len(current_input) < len(baseline):
        return None

    prefix = current_input[: len(baseline)]
    if not response_inputs_equal(prefix, baseline):
        return None

    return current_input[len(baseline) :]


def build_cached_websocket_request_body(entry: CachedWebSocketConnection, body: dict[str, Any]) -> dict[str, Any]:
    continuation = entry.continuation
    if continuation is None:
        return body

    delta = get_cached_websocket_input_delta(body, continuation)
    if not delta or not continuation.lastResponseId:
        entry.continuation = None
        return body

    return {**body, "previous_response_id": continuation.lastResponseId, "input": delta}


async def start_websocket_output_on_first_event(
    events: AsyncIterable[dict[str, Any]], on_start: Callable[[], None]
) -> AsyncIterator[dict[str, Any]]:
    started = False
    async for event in events:
        if not started:
            started = True
            on_start()
        yield event


async def process_websocket_stream(
    url: str,
    body: dict[str, Any],
    headers: Mapping[str, str],
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    model: Model,
    on_start: Callable[[], None],
    idle_timeout_ms: int | None,
    websocket_connect_timeout_ms: int | None,
    cache_session_id: str | None,
    account_id: str,
    grammar_tool_input_properties: Mapping[str, str],
    options: Any = None,
) -> None:
    acquired = await acquire_websocket(
        url,
        headers,
        cache_session_id,
        account_id,
        _option(options, "signal"),
        websocket_connect_timeout_ms,
    )
    keep_connection = True
    use_cached_context = _option(options, "transport") in {"websocket-cached", "auto"}
    # ChatGPT Codex Responses rejects `store: true` ("Store must be set to false").
    # WebSocket continuation still works via connection-scoped previous_response_id state.
    full_body = body
    request_body = (
        build_cached_websocket_request_body(acquired.entry, full_body)
        if use_cached_context and acquired.entry is not None
        else full_body
    )
    stats = get_or_create_websocket_debug_stats(cache_session_id) if cache_session_id else None
    if stats is not None:
        stats.requests += 1
        if acquired.reused:
            stats.connectionsReused += 1
        else:
            stats.connectionsCreated += 1
        if use_cached_context:
            stats.cachedContextRequests += 1
        if request_body.get("store") is True:
            stats.storeTrueRequests += 1
        stats.lastInputItems = len(request_body.get("input") or [])
        if request_body.get("previous_response_id"):
            stats.deltaRequests += 1
            stats.lastDeltaInputItems = len(request_body.get("input") or [])
            stats.lastPreviousResponseId = request_body["previous_response_id"]
        else:
            stats.fullContextRequests += 1
            stats.lastDeltaInputItems = None
            stats.lastPreviousResponseId = None
    try:
        await maybe_await(acquired.socket.send(json.dumps({"type": "response.create", **request_body})))
        await process_responses_stream(
            start_websocket_output_on_first_event(
                map_codex_events(
                    parse_websocket(acquired.socket, _option(options, "signal"), idle_timeout_ms),
                    output,
                    model,
                    _option(options, "onProviderStreamEvent"),
                ),
                on_start,
            ),
            output,
            stream,
            model,
            {
                "serviceTier": _option(options, "serviceTier"),
                "grammarToolInputProperties": grammar_tool_input_properties,
                "resolveServiceTier": resolve_codex_service_tier,
                "applyServiceTierPricing": lambda usage, service_tier: apply_service_tier_pricing(
                    usage, service_tier, model
                ),
            },
        )
        if signal_aborted(_option(options, "signal")):
            keep_connection = False
        elif use_cached_context and acquired.entry is not None and output.responseId:
            response_items = [
                item
                for item in convert_responses_messages(
                    model,
                    normalize_context(Context(messages=[output])),
                    CODEX_TOOL_CALL_PROVIDERS,
                    {"includeSystemPrompt": False, "grammarToolInputProperties": grammar_tool_input_properties},
                )
                if item.get("type") not in {"function_call_output", "custom_tool_call_output"}
            ]
            acquired.entry.continuation = CachedWebSocketContinuationState(
                lastRequestBody=full_body,
                lastResponseId=output.responseId,
                lastResponseItems=response_items,
            )
    except BaseException:
        if acquired.entry is not None:
            acquired.entry.continuation = None
        keep_connection = False
        raise
    finally:
        acquired.release(keep=keep_connection)


# ============================================================================
# Error Handling
# ============================================================================


def parse_error_response(raw: str, status: int, status_text: str | None = None) -> dict[str, str]:
    message = raw or status_text or "Request failed"
    friendly_message: str | None = None

    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = None
    error = parsed.get("error") if isinstance(parsed, Mapping) else None
    if isinstance(error, Mapping):
        code = str(error.get("code") or error.get("type") or "")
        if re.search(r"usage_limit_reached|usage_not_included|rate_limit_exceeded", code, re.IGNORECASE) or status == 429:
            plan_type = error.get("plan_type")
            plan = f" ({plan_type.lower()} plan)" if isinstance(plan_type, str) and plan_type else ""
            resets_at = error.get("resets_at")
            mins = (
                max(0, round((resets_at * 1000 - time.time() * 1000) / 60000))
                if isinstance(resets_at, (int, float)) and resets_at
                else None
            )
            when = f" Try again in ~{mins} min." if mins is not None else ""
            friendly_message = f"You have hit your ChatGPT usage limit{plan}.{when}".strip()
        message = error.get("message") or friendly_message or message

    result = {"message": message}
    if friendly_message is not None:
        result["friendlyMessage"] = friendly_message
    return result


# ============================================================================
# Auth & Headers
# ============================================================================


def extract_account_id(token: str) -> str:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid token")
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
        account_id = payload.get(JWT_CLAIM_PATH, {}).get("chatgpt_account_id")
        if not account_id:
            raise ValueError("No account ID in token")
        return account_id
    except Exception as error:
        raise RuntimeError("Failed to extract accountId from token") from error


def _set_header(headers: dict[str, str], name: str, value: str) -> None:
    for existing in [key for key in headers if key.lower() == name.lower()]:
        del headers[existing]
    headers[name] = value


def _delete_header(headers: dict[str, str], name: str) -> None:
    for existing in [key for key in headers if key.lower() == name.lower()]:
        del headers[existing]


def build_base_codex_headers(
    init_headers: Mapping[str, str] | None,
    additional_headers: Mapping[str, str | None] | None,
    account_id: str,
    token: str,
) -> dict[str, str]:
    """A case-insensitive header set, as pi's `Headers` is."""
    headers: dict[str, str] = {}
    for key, value in (init_headers or {}).items():
        _set_header(headers, key, value)
    for key, value in (additional_headers or {}).items():
        if value is None:
            _delete_header(headers, key)
        else:
            _set_header(headers, key, value)
    _set_header(headers, "Authorization", f"Bearer {token}")
    _set_header(headers, "chatgpt-account-id", account_id)
    # MISAKA fork: pi sends its own name; sending another project's would credit this
    # traffic to it, which is what CLIENT_NAME exists to prevent.
    _set_header(headers, "originator", CLIENT_NAME)
    _set_header(headers, "User-Agent", get_misaka_user_agent())
    return headers


def build_sse_headers(
    init_headers: Mapping[str, str] | None,
    additional_headers: Mapping[str, str | None] | None,
    account_id: str,
    token: str,
    session_id: str | None = None,
) -> dict[str, str]:
    headers = build_base_codex_headers(init_headers, additional_headers, account_id, token)
    _set_header(headers, "OpenAI-Beta", "responses=experimental")
    _set_header(headers, "accept", "text/event-stream")
    _set_header(headers, "content-type", "application/json")

    if session_id:
        _set_header(headers, "session-id", session_id)
        _set_header(headers, "x-client-request-id", session_id)

    return headers


def build_websocket_headers(
    init_headers: Mapping[str, str] | None,
    additional_headers: Mapping[str, str | None] | None,
    account_id: str,
    token: str,
    request_id: str,
) -> dict[str, str]:
    headers = build_base_codex_headers(init_headers, additional_headers, account_id, token)
    _delete_header(headers, "accept")
    _delete_header(headers, "content-type")
    _delete_header(headers, "OpenAI-Beta")
    _set_header(headers, "OpenAI-Beta", OPENAI_BETA_RESPONSES_WEBSOCKETS)
    _set_header(headers, "x-client-request-id", request_id)
    _set_header(headers, "session-id", request_id)
    return headers


# --- MISAKA: asyncio plumbing ------------------------------------------------------------


async def _await_with_abort(awaitable: Any, signal: Any, *, on_abort: Any = None) -> Any:
    if signal_aborted(signal):
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        if on_abort is not None:
            await maybe_await(on_abort())
        raise RuntimeError("Request was aborted")

    task = asyncio.ensure_future(awaitable)
    abort_task = _create_abort_wait_task(signal)
    try:
        if abort_task is not None:
            done, _ = await asyncio.wait({task, abort_task}, return_when=asyncio.FIRST_COMPLETED)
            if abort_task in done and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                if on_abort is not None:
                    await maybe_await(on_abort())
                raise RuntimeError("Request was aborted")
        return await task
    finally:
        if abort_task is not None:
            abort_task.cancel()
            await asyncio.gather(abort_task, return_exceptions=True)


streamOpenAICodexResponses = stream_openai_codex_responses
streamSimpleOpenAICodexResponses = stream_simple_openai_codex_responses
applyServiceTierPricing = apply_service_tier_pricing

__all__ = [
    "DEFAULT_CODEX_BASE_URL",
    "JWT_CLAIM_PATH",
    "OpenAICodexResponsesOptions",
    "applyServiceTierPricing",
    "close_openai_codex_websocket_sessions",
    "get_openai_codex_websocket_debug_stats",
    "reset_openai_codex_websocket_debug_stats",
    "streamOpenAICodexResponses",
    "streamSimpleOpenAICodexResponses",
]
