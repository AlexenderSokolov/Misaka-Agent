"""Public coding-agent SDK entrypoints."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

from misaka.agent.agent import Agent
from misaka.agent.stream_fn import setDefaultStreamFn
from misaka.agent.types import AgentMessage, ThinkingLevel
from misaka.ai.compat import streamSimple
from misaka.ai.models import clampThinkingLevel
from misaka.ai.types import (
    Model,
    TextContent,
    validate_message,
)
from misaka.config import get_agent_dir
from misaka.core.agent_session import AgentSession
from misaka.core.auth_guidance import formatNoModelsAvailableMessage
from misaka.core.cache_warmer import CacheWarmer, CacheWarmRequest
from misaka.core.defaults import DEFAULT_THINKING_LEVEL
from misaka.core.extensions import (
    ExtensionAPI,
    ExtensionCommandContext,
    ExtensionContext,
    ExtensionFactory,
    LoadExtensionsResult,
    SessionStartEvent,
    SlashCommandInfo,
    SlashCommandSource,
    ToolDefinition,
)
from misaka.core.http_dispatcher import applyHttpProxySettings
from misaka.core.mcp_servers import createToolNameMatcher
from misaka.core.messages import convertToLlm
from misaka.core.model_resolver import findInitialModel
from misaka.core.model_runtime import ModelRuntime
from misaka.core.prompt_templates import PromptTemplate
from misaka.core.provider_attribution import mergeProviderAttributionHeaders
from misaka.core.resource_loader import DefaultResourceLoader, ResourceLoader
from misaka.core.session_manager import SessionManager, getDefaultSessionDir
from misaka.core.settings_manager import (
    DEFAULT_TOOL_NAMES,
    SettingsManager,
    applyToolModifiers,
    getToolListError,
    isToolModifier,
)
from misaka.core.timings import time
from misaka.core.tools import (
    Tool,
    createBashTool,
    createCodingTools,
    createEditTool,
    createFindTool,
    createGrepTool,
    createLsTool,
    createPowerShellTool,
    createReadOnlyTools,
    createReadTool,
    createWriteTool,
    withFileMutationQueue,
)
from misaka.core.virtual_models import getBranchSelection
from misaka.utils.paths import resolvePath
from misaka.utils.values import read_field

# Preserve the pre-0.81 fallback for extensions that construct Agent instances
# or invoke low-level agent loops without supplying streamFn. Agent core remains
# provider-agnostic and does not import pi-ai/compat itself.
setDefaultStreamFn(streamSimple)

if TYPE_CHECKING:
    from misaka.core.agent_session_runtime import (
        AgentSessionRuntime,
        AgentSessionRuntimeDiagnostic,
        AgentSessionServices,
        CreateAgentSessionFromServicesOptions,
        CreateAgentSessionRuntimeFactory,
        CreateAgentSessionRuntimeResult,
        CreateAgentSessionServicesOptions,
        SessionImportFileNotFoundError,
        createAgentSessionFromServices,
        createAgentSessionRuntime,
        createAgentSessionServices,
    )


_AGENT_SESSION_RUNTIME_EXPORTS = {
    "AgentSessionRuntime",
    "AgentSessionRuntimeDiagnostic",
    "AgentSessionServices",
    "CreateAgentSessionFromServicesOptions",
    "CreateAgentSessionRuntimeFactory",
    "CreateAgentSessionRuntimeResult",
    "CreateAgentSessionServicesOptions",
    "SessionImportFileNotFoundError",
    "createAgentSessionFromServices",
    "createAgentSessionRuntime",
    "createAgentSessionServices",
}


class ScopedModel(TypedDict):
    model: Model[Any]
    thinkingLevel: NotRequired[ThinkingLevel]


class CreateAgentSessionOptions(TypedDict, total=False):
    cwd: str
    agentDir: str
    modelRuntime: ModelRuntime
    model: Model[Any]
    thinkingLevel: ThinkingLevel
    scopedModels: list[ScopedModel]
    noTools: Literal["all", "builtin"]
    # Optional allowlist of tool names or patterns, where `*` matches any characters.
    #
    # When omitted, the resolved `defaultTools` setting gives the initial selection when configured.
    # Otherwise the default built-in tools are enabled. Extension/custom tools remain enabled unless
    # `noTools` changes that default. When provided, only matching tools are enabled. MCP tools stay
    # registered for codemode and tool search unless an entry starts with `mcp__`; then only matching
    # MCP tools are kept. An empty list, like `noTools: "all"`, disables MCP tools too.
    #
    # A list of only `+name` and `-name` entries is not an allowlist: it adds tools to or removes them
    # from the default selection, like the `defaultTools` setting. These entries take exact names.
    # Mixing them with plain entries throws.
    tools: list[str]
    # Optional denylist of tool names or patterns to disable. Applies after `tools` when both are
    # provided, MCP tools included.
    excludeTools: list[str]
    customTools: list[ToolDefinition[Any, Any]]
    parts: list[Any]
    resourceLoader: ResourceLoader
    sessionManager: SessionManager
    settingsManager: SettingsManager
    modelProfile: str
    sessionStartEvent: SessionStartEvent


class _CreateAgentSessionResultRequired(TypedDict):
    session: AgentSession
    extensionsResult: LoadExtensionsResult


class CreateAgentSessionResult(_CreateAgentSessionResultRequired, total=False):
    modelFallbackMessage: str


def getDefaultAgentDir() -> str:
    return get_agent_dir()


async def createAgentSession(options: CreateAgentSessionOptions | None = None) -> CreateAgentSessionResult:
    resolved_options = dict(options or {})
    explicit_session_manager = resolved_options.get("sessionManager")
    cwd_option = resolved_options.get("cwd", None)
    cwd = resolvePath(cwd_option if cwd_option is not None else (
        explicit_session_manager.getCwd() if explicit_session_manager else os.getcwd()
    ))
    agent_dir = (
        resolvePath(resolved_options["agentDir"])
        if resolved_options.get("agentDir")
        else getDefaultAgentDir()
    )
    resource_loader = resolved_options.get("resourceLoader")

    # The home lays credentials out under credentials/; a custom agent directory keeps the same
    # shape (config.home.path: a directory laying out what it keeps the way the home does).
    from misaka.config import home

    custom_dir = resolved_options.get("agentDir")
    auth_path = str(home.path("auth", agent_dir)) if custom_dir else None
    models_path = str(home.path("models", agent_dir if custom_dir else None))
    settings_manager = resolved_options.get("settingsManager") or SettingsManager.create(cwd, agent_dir)
    applyHttpProxySettings(settings_manager.getGlobalSettings().get("httpProxy"))
    model_runtime = resolved_options.get("modelRuntime") or await ModelRuntime.create(
        authPath=auth_path, modelsPath=models_path
    )
    if resolved_options.get("modelProfile"):
        settings_manager.bindModelProfile(resolved_options["modelProfile"], model_runtime)
    # The engine home only decides the session store when the caller named one; otherwise
    # sessions go to the product tree (config.sessions), where /resume and the panel look.
    session_manager = explicit_session_manager or SessionManager.create(
        cwd, getDefaultSessionDir(cwd, resolved_options.get("agentDir") and agent_dir))

    if resource_loader is None:
        resource_loader = DefaultResourceLoader(
            {"cwd": cwd, "agentDir": agent_dir, "settingsManager": settings_manager}
        )
        await resource_loader.reload()
        time("resourceLoader.reload")

    existing_session = session_manager.buildSessionContext()
    has_existing_session = len(existing_session.messages) > 0
    has_thinking_entry = any(entry.get("type") == "thinking_level_change" for entry in session_manager.getBranch())

    model = resolved_options.get("model")
    model_fallback_message: str | None = None

    # Assistant messages name the physical model that answered, so a virtual selection is only in
    # model_change entries.
    session_model = getBranchSelection(session_manager.getBranch(), model_runtime.getModel)

    if model is None and has_existing_session and session_model:
        restored_model = model_runtime.getModel(session_model["provider"], session_model["modelId"])
        if restored_model is not None and model_runtime.hasConfiguredAuth(restored_model.provider):
            model = restored_model
        if model is None:
            model_fallback_message = f'Could not restore model {session_model["provider"]}/{session_model["modelId"]}'

    if model is None and getattr(settings_manager, "getModelProfile", lambda: None)():
        from misaka.config import profiles

        pin = profiles.pinned_model(settings_manager.getModelProfile(), strict=True)
        if pin and pin != "inherit":
            provider, model_id = settings_manager.getDefaultModelPair()
            model = model_runtime.getModel(provider, model_id)
            if model is None:
                raise RuntimeError(f"Role default model is not available: {provider}/{model_id}")
            if not model_runtime.hasConfiguredAuth(model.provider):
                model_fallback_message = (
                    f"No configured authentication for role default {provider}/{model_id}. "
                    "Use /login or misaka setup; the selected provider has not been changed."
                )

    if model is None:
        default_provider, default_model = settings_manager.getDefaultModelPair()
        result = await findInitialModel(
            {
                "scopedModels": [],
                "isContinuing": has_existing_session,
                "defaultProvider": default_provider,
                "defaultModelId": default_model,
                "defaultThinkingLevel": settings_manager.getDefaultThinkingLevel(),
                "modelThinkingLevels": settings_manager.getAllModelThinkingLevels(),
                "modelRuntime": model_runtime,
            }
        )
        model = result.model
        if model is None:
            model_fallback_message = formatNoModelsAvailableMessage()
        elif model_fallback_message:
            model_fallback_message += f". Using {model.provider}/{model.id}"

    thinking_level = resolved_options.get("thinkingLevel")
    if thinking_level is None and has_existing_session:
        thinking_level = (
            existing_session.thinkingLevel
            if has_thinking_entry
            else settings_manager.getDefaultThinkingLevel() or DEFAULT_THINKING_LEVEL
        )
    if thinking_level is None:
        # MISAKA fork: the agent's own default (her role's defaultThinkingLevel) comes before the
        # shared per-model table.
        thinking_level = settings_manager.getRoleThinkingLevel()
    if thinking_level is None and model is not None:
        thinking_level = settings_manager.getModelThinkingLevel(model.provider, model.id)
    if thinking_level is None:
        thinking_level = settings_manager.getDefaultThinkingLevel() or DEFAULT_THINKING_LEVEL
    thinking_level = "off" if model is None else clampThinkingLevel(model, thinking_level)

    # defaultTools selects initial built-ins; it does not populate the registry allowlist (pi 541045ae0).
    tools_option = resolved_options.get("tools")
    tool_list_error = getToolListError(tools_option) if tools_option else None
    if tool_list_error:
        raise ValueError(f"Invalid tools option: {tool_list_error}")
    configured_default_tools = settings_manager.getDefaultTools()
    default_tool_names = (
        []
        if resolved_options.get("noTools")
        else list(DEFAULT_TOOL_NAMES if configured_default_tools is None else configured_default_tools)
    )
    # A `tools` list of only `+name`/`-name` entries changes the default selection instead of
    # replacing it, like the `defaultTools` setting.
    tool_modifiers = list(tools_option) if tools_option and any(map(isToolModifier, tools_option)) else None
    selected_tool_names = (
        applyToolModifiers(default_tool_names, tool_modifiers)
        if tool_modifiers
        else (list(tools_option) if tools_option is not None else None)
    )
    if tool_modifiers:
        allowed_tool_names = selected_tool_names if resolved_options.get("noTools") == "all" else None
    else:
        allowed_tool_names = (
            list(tools_option)
            if tools_option is not None
            else ([] if resolved_options.get("noTools") == "all" else None)
        )
    excluded_tool_names = (
        list(resolved_options["excludeTools"]) if resolved_options.get("excludeTools") is not None else None
    )
    is_excluded_tool = createToolNameMatcher(excluded_tool_names) if excluded_tool_names is not None else None
    initial_active_tool_names = [
        name
        for name in (selected_tool_names if selected_tool_names is not None else default_tool_names)
        if not (is_excluded_tool is not None and is_excluded_tool(name))
    ]

    extension_runner_ref: dict[str, Any] = {}

    def convert_to_llm_with_block_images(messages: list[AgentMessage]) -> list[Any]:
        converted = convertToLlm(messages)
        if not settings_manager.getBlockImages():
            return converted

        filtered_messages: list[Any] = []
        for message in converted:
            role = read_field(message, "role")
            content = read_field(message, "content")
            if role not in {"user", "toolResult"} or not isinstance(content, list):
                filtered_messages.append(message)
                continue
            if not any(_content_type(item) == "image" for item in content):
                filtered_messages.append(message)
                continue

            def is_disabled_text(item: Any) -> bool:
                return _content_type(item) == "text" and _content_text(item) == "Image reading is disabled."

            mapped = [
                TextContent(text="Image reading is disabled.") if _content_type(item) == "image" else item
                for item in content
            ]
            # Dedupe consecutive "Image reading is disabled." texts
            filtered_content = [
                item
                for index, item in enumerate(mapped)
                if not (is_disabled_text(item) and index > 0 and is_disabled_text(mapped[index - 1]))
            ]

            payload = _message_dump(message)
            payload["content"] = [_message_dump(item) for item in filtered_content]
            filtered_messages.append(validate_message(payload))

        return filtered_messages

    def build_request_options(request_model: Model[Any], options: Any = None) -> dict[str, Any]:
        resolved = _to_dict(options)
        provider_retry_settings = settings_manager.getProviderRetrySettings()
        http_idle_timeout_ms = settings_manager.getHttpIdleTimeoutMs()
        effective_timeout_ms = 2_147_483_647 if http_idle_timeout_ms == 0 else http_idle_timeout_ms
        header_runner = extension_runner_ref.get("current")

        async def transform_headers(request_headers: dict[str, Any]) -> dict[str, Any]:
            headers = mergeProviderAttributionHeaders(
                request_model,
                settings_manager,
                resolved.get("sessionId"),
                request_headers,
            )
            if header_runner is not None and header_runner.hasHandlers("before_provider_headers"):
                return await header_runner.emitBeforeProviderHeaders(dict(headers or {}))
            return headers or {}

        def first_defined(*values: Any) -> Any:
            return next((value for value in values if value is not None), None)

        return {
            **resolved,
            "timeoutMs": first_defined(
                resolved.get("timeoutMs"), provider_retry_settings.get("timeoutMs"), effective_timeout_ms
            ),
            "websocketConnectTimeoutMs": first_defined(
                resolved.get("websocketConnectTimeoutMs"), settings_manager.getWebSocketConnectTimeoutMs()
            ),
            "maxRetries": first_defined(resolved.get("maxRetries"), provider_retry_settings.get("maxRetries")),
            "maxRetryDelayMs": first_defined(
                resolved.get("maxRetryDelayMs"), provider_retry_settings.get("maxRetryDelayMs")
            ),
            "transformHeaders": transform_headers,
        }

    async def stream_fn(model_value: Model[Any], context: Any, stream_options: Any = None) -> Any:
        request_options = build_request_options(model_value, stream_options)
        # Compaction and summaries use their own routing ids; only session requests
        # replace the cache entry, so warming restarts from them. Keep warming while
        # the current transcript still extends the request's prefix. Agent state may
        # shallow-copy the messages array or refresh the model object without changing
        # the provider request, so top-level object identity is not a valid cache key.
        if request_options.get("sessionId") == session_manager.getSessionId():
            cache_warmer.start(
                CacheWarmRequest(model=model_value, context=context, options=request_options),
                cache_context_is_current(model_value),
            )
        return model_runtime.streamSimple(model_value, context, request_options)

    def cache_context_is_current(request_model: Model[Any]) -> Callable[[], bool]:
        messages = agent.state.messages

        def is_current() -> bool:
            current_model = agent.state.model
            current_messages = agent.state.messages
            return (
                current_model is not None
                and current_model.provider == request_model.provider
                and current_model.id == request_model.id
                and len(messages) <= len(current_messages)
                and all(message is current_messages[index] for index, message in enumerate(messages))
            )

        return is_current

    async def decide_cache_warming(event: dict[str, Any]) -> Any:
        runner = extension_runner_ref.get("current")
        if runner is None:
            return event["action"]
        return await runner.emitCacheWarmingDecision(event)

    cache_warmer = CacheWarmer(
        model_runtime, session_manager, settings_manager.getCacheWarmingMode, decide_cache_warming
    )

    async def on_payload(payload: dict[str, Any], _model: Model[Any]) -> Any:
        runner = extension_runner_ref.get("current")
        if runner is None or not runner.hasHandlers("before_provider_request"):
            return payload
        return await runner.emitBeforeProviderRequest(payload)

    async def on_response(response: Any, _model: Model[Any]) -> None:
        runner = extension_runner_ref.get("current")
        if runner is None or not runner.hasHandlers("after_provider_response"):
            return
        await runner.emit(
            {
                "type": "after_provider_response",
                "status": read_field(response, "status"),
                "headers": read_field(response, "headers"),
            }
        )

    async def on_provider_stream_event(data: Any, model: Model[Any]) -> None:
        runner = extension_runner_ref.get("current")
        if runner is None or not runner.hasHandlers("provider_stream_event"):
            return
        await runner.emit(
            {
                "data": data,
                "type": "provider_stream_event",
                "provider": model.provider,
                "api": model.api,
                "model": model.id,
            }
        )

    async def transform_context(messages: list[AgentMessage], _signal: Any | None = None) -> list[AgentMessage]:
        session = extension_runner_ref.get("session")
        if session is not None:
            messages = await session.prepareContextMessages(messages, _signal)
            # MISAKA fork: transient parts/extensions see the engine's durable view.
            messages = await session.moments.context(messages)
        runner = extension_runner_ref.get("current")
        if runner is None:
            return messages
        return await runner.emitContext(messages)

    initial_model = model or _unknown_model()
    agent = Agent(
        {
            "initialState": {
                "systemPrompt": "",
                "model": initial_model,
                "thinkingLevel": thinking_level,
                "tools": [],
                "messages": existing_session.messages,
            },
            "convertToLlm": convert_to_llm_with_block_images,
            "streamFn": stream_fn,
            "onPayload": on_payload,
            "onResponse": on_response,
            "onProviderStreamEvent": on_provider_stream_event,
            "sessionId": session_manager.getSessionId(),
            "transformContext": transform_context,
            "steeringMode": settings_manager.getSteeringMode(),
            "followUpMode": settings_manager.getFollowUpMode(),
            "transport": settings_manager.getTransport(),
            "thinkingBudgets": settings_manager.getThinkingBudgets(),
            "maxRetryDelayMs": settings_manager.getProviderRetrySettings().get("maxRetryDelayMs"),
        }
    )
    if model is None:
        agent.state.model = None

    # Restore missing settings metadata for older sessions.
    if has_existing_session:
        if not has_thinking_entry:
            session_manager.appendThinkingLevelChange(thinking_level)
    else:
        if model is not None:
            session_manager.appendModelChange(model.provider, model.id)
        session_manager.appendThinkingLevelChange(thinking_level)

    from misaka.core.session_catalog import CatalogPart
    parts = list(resolved_options.get("parts") or [])
    if not any(isinstance(part, CatalogPart) for part in parts):
        parts.insert(0, CatalogPart())  # raw SDK/helper sessions have no product assembly
    session = AgentSession(
        {
            "agent": agent,
            "sessionManager": session_manager,
            "settingsManager": settings_manager,
            "cwd": cwd,
            "scopedModels": resolved_options.get("scopedModels") or [],
            "resourceLoader": resource_loader,
            "customTools": resolved_options.get("customTools") or [],
            "parts": parts,
            "modelRuntime": model_runtime,
            "cacheWarmer": cache_warmer,
            "initialActiveToolNames": initial_active_tool_names,
            "usesDefaultTools": (resolved_options.get("tools") is None or tool_modifiers is not None)
            and not resolved_options.get("noTools"),
            "defaultToolModifiers": tool_modifiers,
            "allowedToolNames": allowed_tool_names,
            "excludedToolNames": excluded_tool_names,
            "extensionRunnerRef": extension_runner_ref,
            "sessionStartEvent": resolved_options.get("sessionStartEvent"),
        }
    )
    extensions_result = resource_loader.getExtensions()

    return {
        "session": session,
        "extensionsResult": extensions_result,
        "modelFallbackMessage": model_fallback_message,
    }


def _to_dict(value: Any) -> dict[str, Any]:
    """Convert an options-like value to a plain dict.

    Handles dicts, Pydantic models, SimpleNamespace subclasses (e.g.
    ``StreamOptionsNamespace``), and other objects with ``__dict__``.
    Falls back to an empty dict for ``None``.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    # SimpleNamespace and its subclasses (e.g. StreamOptionsNamespace) store
    # their attributes in __dict__ but are not iterable, so dict() on them
    # raises TypeError.  We read __dict__ directly instead.
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return dict(value)


def _unknown_model() -> Model[Any]:
    return Model(
        id="unknown",
        name="unknown",
        api="unknown",
        provider="unknown",
        baseUrl="",
        reasoning=False,
        input=[],
        cost={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        contextWindow=0,
        maxTokens=0,
    )


def _message_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: _message_dump(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_message_dump(item) for item in value]
    return value


def _content_type(block: Any) -> str | None:
    value = read_field(block, "type")
    return value if isinstance(value, str) else None


def _content_text(block: Any) -> str | None:
    value = read_field(block, "text")
    return value if isinstance(value, str) else None




def __getattr__(name: str) -> Any:
    if name in _AGENT_SESSION_RUNTIME_EXPORTS:
        from misaka.core import agent_session_runtime as runtime_module

        return getattr(runtime_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AgentSessionRuntime",
    "AgentSessionRuntimeDiagnostic",
    "AgentSessionServices",
    "CreateAgentSessionFromServicesOptions",
    "CreateAgentSessionOptions",
    "CreateAgentSessionResult",
    "CreateAgentSessionRuntimeFactory",
    "CreateAgentSessionRuntimeResult",
    "CreateAgentSessionServicesOptions",
    "ExtensionAPI",
    "ExtensionCommandContext",
    "ExtensionContext",
    "ExtensionFactory",
    "PromptTemplate",
    "SessionImportFileNotFoundError",
    "SlashCommandInfo",
    "SlashCommandSource",
    "Tool",
    "ToolDefinition",
    "createAgentSessionFromServices",
    "createAgentSessionRuntime",
    "createAgentSessionServices",
    "createBashTool",
    "createCodingTools",
    "createEditTool",
    "createFindTool",
    "createGrepTool",
    "createLsTool",
    "createPowerShellTool",
    "createReadOnlyTools",
    "createReadTool",
    "createWriteTool",
    "withFileMutationQueue",
]
