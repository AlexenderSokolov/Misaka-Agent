"""Resource loading for prompts, skills, context files, and extensions.

Translated from pi's ``coding-agent/src/core/resource-loader.ts``.

MISAKA fork: skills belong to MISAKA's own layer system (``core/skills``), so this loader only collects the skill paths
extensions contribute (no ``updateSkillsFromPaths``/``mapSkillPath``); and ``HOST_PROVIDED_EXTENSION_PACKAGES`` with
``collectExtensionPackageWarnings`` are not ported, see ``mergeExtensionWarnings``.
"""

from __future__ import annotations

import dataclasses
import os
import stat as stat_module
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypedDict

from misaka.core.diagnostics import ResourceCollision, ResourceDiagnostic
from misaka.core.event_bus import createEventBus
from misaka.core.extensions.loader import (
    clearExtensionCache,
    createExtensionRuntime,
    loadExtensionFromFactory,
    loadExtensionsCached,
)
from misaka.core.extensions.types import (
    Extension,
    ExtensionRuntime,
    InlineExtension,
    LoadExtensionsResult,
)
from misaka.core.footer_data_provider import findGitPaths
from misaka.core.package_manager import (
    DefaultPackageManager,
    PathMetadata,
    ResolvedResource,
)
from misaka.core.prompt_templates import PromptTemplate, loadPromptTemplates
from misaka.core.settings_manager import SettingsManager
from misaka.core.source_info import (
    BUILTIN_PATH_PREFIX,
    SourceInfo,
    createSourceInfo,
    getSyntheticPathSource,
    isSyntheticPath,
)
from misaka.core.timings import resetTimings
from misaka.ui.tui.interactive.theme.theme import Theme, loadThemeFromPath
from misaka.ui.tui.terminal_image import (
    TerminalColorMode,
    detectCapabilities,
    getTerminalColorMode,
)
from misaka.utils.node_fs import readdirEntries
from misaka.utils.paths import canonicalizePath, isLocalPath, resolvePath


class ResourcePathEntry(TypedDict):
    path: str
    metadata: PathMetadata


class ResourceExtensionPaths(TypedDict, total=False):
    promptPaths: list[ResourcePathEntry]
    themePaths: list[ResourcePathEntry]
    skillPaths: list[ResourcePathEntry]


class ResourceLoaderReloadOptions(TypedDict, total=False):
    resolveProjectTrust: Callable[[dict[str, LoadExtensionsResult]], Awaitable[bool]]


class _DefaultResourceLoaderOptionsRequired(TypedDict):
    cwd: str
    agentDir: str


class DefaultResourceLoaderOptions(_DefaultResourceLoaderOptionsRequired, total=False):
    settingsManager: SettingsManager
    eventBus: Any
    additionalExtensionPaths: list[str]
    additionalPromptTemplatePaths: list[str]
    additionalThemePaths: list[str]
    extensionFactories: list[InlineExtension]
    noExtensions: bool
    # Built-in extensions not to load, by name (such as `mcp`), even when settings or `-e` enable them.
    disabledBuiltinExtensions: list[str]
    noPromptTemplates: bool
    noThemes: bool
    noContextFiles: bool
    systemPrompt: str
    appendSystemPrompt: list[str]
    extensionsOverride: Any
    promptsOverride: Any
    themesOverride: Any
    agentsFilesOverride: Any
    systemPromptOverride: Any
    appendSystemPromptOverride: Any


class PromptsResult(TypedDict):
    prompts: list[PromptTemplate]
    diagnostics: list[ResourceDiagnostic]


class ThemesResult(TypedDict):
    themes: list[Theme]
    diagnostics: list[ResourceDiagnostic]


class AgentsFilesResult(TypedDict):
    agentsFiles: list[dict[str, str]]


class ResourceLoader(Protocol):
    def getExtensions(self) -> LoadExtensionsResult: ...

    def getExtensionSkillPaths(self) -> list[str]: ...

    def getExtensionSkillResources(self) -> list[ResourcePathEntry]: ...

    def getPrompts(self) -> PromptsResult: ...

    def getThemes(self) -> ThemesResult: ...

    def getAgentsFiles(self) -> AgentsFilesResult: ...

    def getSystemPrompt(self) -> str | None: ...

    def getSystemPromptSource(self) -> dict[str, str] | None: ...

    def getAppendSystemPrompt(self) -> list[str]: ...

    def getAppendSystemPromptSources(self) -> list[dict[str, str]]: ...

    def extendResources(self, paths: ResourceExtensionPaths) -> None: ...

    async def reload(self, options: ResourceLoaderReloadOptions | None = None) -> None: ...


def _warn(message: str) -> None:
    print(message, file=sys.stderr)


def _error_message(error: Exception, fallback: str) -> str:
    message = str(error)
    return message or fallback


def resolvePromptInput(input_value: str | None, description: str) -> str | None:
    if not input_value:
        return None
    if os.path.exists(input_value):
        try:
            return Path(input_value).read_text(encoding="utf-8-sig")
        except OSError as error:
            _warn(f"Warning: Could not read {description} file {input_value}: {error}")
            return input_value
    return input_value


def findShadowedContextFile(cwd: str) -> str | None:
    git_paths = findGitPaths(cwd)
    if git_paths is None:
        return None

    common_git_dir = canonicalizePath(git_paths.commonGitDir)
    worktree_root = canonicalizePath(git_paths.repoDir)
    main_repo_root = os.path.dirname(common_git_dir)
    if not worktree_root.startswith(f"{main_repo_root}{os.sep}"):
        return None
    if canonicalizePath(os.path.join(main_repo_root, ".git")) != common_git_dir:
        return None

    worktree_context_file = loadContextFileFromDir(worktree_root)
    if worktree_context_file is None:
        return None
    return os.path.join(main_repo_root, os.path.basename(worktree_context_file["path"]))


def loadProjectContextFiles(options: dict[str, str]) -> list[dict[str, str]]:
    resolved_cwd = resolvePath(options["cwd"])
    resolved_agent_dir = resolvePath(options["agentDir"])
    context_files: list[dict[str, str]] = []
    seen_paths: set[str] = set()

    global_context = loadContextFileFromDir(resolved_agent_dir)
    if global_context is not None:
        context_files.append(global_context)
        seen_paths.add(global_context["path"])

    ancestor_context_files: list[dict[str, str]] = []
    shadowed_context_file = findShadowedContextFile(resolved_cwd)
    current_dir = resolved_cwd
    root = os.path.abspath(os.sep)
    while True:
        context_file = loadContextFileFromDir(current_dir)
        is_shadowed = (
            shadowed_context_file is not None
            and canonicalizePath(context_file["path"] if context_file is not None else "") == shadowed_context_file
        )
        if context_file is not None and not is_shadowed and context_file["path"] not in seen_paths:
            ancestor_context_files.insert(0, context_file)
            seen_paths.add(context_file["path"])
        if current_dir == root:
            break
        parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
        if parent_dir == current_dir:
            break
        current_dir = parent_dir

    context_files.extend(ancestor_context_files)
    return context_files


# MISAKA fork: upstream's `collectExtensionPackageWarnings` flags npm extension packages that bundle a host package
# as a dependency instead of a peer dependency, which loads a second copy of it; a Python package cannot carry its own
# copy of an imported package, so there is nothing to warn about and only extension warnings are merged here.
def mergeExtensionWarnings(result: LoadExtensionsResult, warnings: list[dict[str, str]]) -> None:
    # Deduplicated by path: the first position of a path, its last warning (a JS Map).
    result.warnings = list({warning["path"]: warning for warning in [*(result.warnings or []), *warnings]}.values())


def isBuiltinExtension(input_: InlineExtension) -> bool:
    """Built-in extensions supply the code of ``builtin:<name>`` extension paths (see
    ``InlineExtension``)."""
    return not callable(input_) and input_.get("builtin") is True


def omitReplacedExtensions(
    extensions: list[Extension], warnings: list[dict[str, str]] | None = None
) -> list[Extension]:
    """Leave out replaceable extensions (see ``InlineExtension``) that share a tool, command, or
    flag name with another extension. For example, a third-party MCP extension that registers
    ``/mcp`` replaces the built-in MCP extension instead of both connecting the same servers."""

    def names(extension: Extension) -> list[str]:
        return [
            *(f"tool:{name}" for name in extension.tools),
            *(f"command:{name}" for name in extension.commands),
            *(f"flag:{name}" for name in extension.flags),
        ]

    taken = {name: extension for extension in extensions if not extension.replaceable for name in names(extension)}
    kept: list[Extension] = []
    for extension in extensions:
        if not extension.replaceable:
            kept.append(extension)
            continue
        replacement = next(((name, taken[name]) for name in names(extension) if name in taken), None)
        if replacement is None:
            kept.append(extension)
            continue
        if extension.path.startswith(BUILTIN_PATH_PREFIX) and warnings is not None:
            builtin_name = extension.path[len(BUILTIN_PATH_PREFIX):]
            kind, raw_name = replacement[0].split(":")[:2]
            registered_name = f"/{raw_name}" if kind == "command" else f"--{raw_name}" if kind == "flag" else raw_name
            # MISAKA fork: misaka has no `pi config` screen; built-in extensions are enabled in the
            # `extensions` setting.
            warnings.append({
                "path": extension.path,
                "warning": (
                    f"Extension {replacement[1].path} registers {kind} `{registered_name}`, so built-in extension "
                    f"`{builtin_name}` was not loaded. To use `{builtin_name}`, make sure the `extensions` setting "
                    f"does not disable it (`-{BUILTIN_PATH_PREFIX}{builtin_name}`), then disable or remove the "
                    "existing extension. We recommend only having one or the other loaded at a time."
                ),
            })
    return kept


@dataclass(slots=True)
class DefaultResourceLoader:
    cwd: str
    agentDir: str
    settingsManager: SettingsManager
    packageManager: DefaultPackageManager
    eventBus: Any | None = None
    additionalExtensionPaths: list[str] = field(default_factory=list)
    additionalPromptTemplatePaths: list[str] = field(default_factory=list)
    additionalThemePaths: list[str] = field(default_factory=list)
    extensionFactories: list[InlineExtension] = field(default_factory=list)
    builtinExtensions: dict[str, Any] = field(default_factory=dict)
    noExtensions: bool = False
    disabledBuiltinExtensions: set[str] = field(default_factory=set)
    noPromptTemplates: bool = False
    noThemes: bool = False
    noContextFiles: bool = False
    systemPromptSource: str | None = None
    appendSystemPromptSource: list[str] | None = None
    extensionsOverride: Any = None
    promptsOverride: Any = None
    themesOverride: Any = None
    agentsFilesOverride: Any = None
    systemPromptOverride: Any = None
    appendSystemPromptOverride: Any = None
    extensionsResult: LoadExtensionsResult = field(
        default_factory=lambda: LoadExtensionsResult(extensions=[], errors=[], runtime=createExtensionRuntime())
    )
    prompts: list[PromptTemplate] = field(default_factory=list)
    promptDiagnostics: list[ResourceDiagnostic] = field(default_factory=list)
    themes: list[Theme] = field(default_factory=list)
    themeDiagnostics: list[ResourceDiagnostic] = field(default_factory=list)
    agentsFiles: list[dict[str, str]] = field(default_factory=list)
    systemPrompt: str | None = None
    systemPromptSourcePath: str | None = None
    appendSystemPrompt: list[str] = field(default_factory=list)
    appendSystemPromptSourcePaths: list[str] = field(default_factory=list)
    lastPromptPaths: list[str] = field(default_factory=list)
    lastThemePaths: list[str] = field(default_factory=list)
    extensionSkillPaths: list[str] = field(default_factory=list)
    extensionSkillResources: list[ResourcePathEntry] = field(default_factory=list)
    extensionPromptSourceInfos: dict[str, SourceInfo] = field(default_factory=dict)
    extensionThemeSourceInfos: dict[str, SourceInfo] = field(default_factory=dict)
    resourceMetadataByPath: dict[str, PathMetadata] = field(default_factory=dict)
    loaded: bool = False

    def __init__(self, options: DefaultResourceLoaderOptions) -> None:
        self.cwd = resolvePath(options["cwd"])
        self.agentDir = resolvePath(options["agentDir"])
        self.settingsManager = options.get("settingsManager") or SettingsManager.create(self.cwd, self.agentDir)
        factories = list(options.get("extensionFactories", []))
        self.extensionFactories = [input_ for input_ in factories if not isBuiltinExtension(input_)]
        self.builtinExtensions = {input_["name"]: input_ for input_ in factories if isBuiltinExtension(input_)}
        self.packageManager = DefaultPackageManager(
            {
                "cwd": self.cwd,
                "agentDir": self.agentDir,
                "settingsManager": self.settingsManager,
                "builtinExtensions": list(self.builtinExtensions),
            }
        )
        self.eventBus = options.get("eventBus") or createEventBus()
        self.additionalExtensionPaths = list(options.get("additionalExtensionPaths", []))
        self.additionalPromptTemplatePaths = list(options.get("additionalPromptTemplatePaths", []))
        self.additionalThemePaths = list(options.get("additionalThemePaths", []))
        self.noExtensions = options.get("noExtensions", False)
        self.disabledBuiltinExtensions = set(options.get("disabledBuiltinExtensions") or [])
        self.noPromptTemplates = options.get("noPromptTemplates", False)
        self.noThemes = options.get("noThemes", False)
        self.noContextFiles = options.get("noContextFiles", False)
        self.systemPromptSource = options.get("systemPrompt")
        self.appendSystemPromptSource = options.get("appendSystemPrompt")
        self.extensionsOverride = options.get("extensionsOverride")
        self.promptsOverride = options.get("promptsOverride")
        self.themesOverride = options.get("themesOverride")
        self.agentsFilesOverride = options.get("agentsFilesOverride")
        self.systemPromptOverride = options.get("systemPromptOverride")
        self.appendSystemPromptOverride = options.get("appendSystemPromptOverride")
        self.extensionsResult = LoadExtensionsResult(extensions=[], errors=[], runtime=createExtensionRuntime())
        self.prompts = []
        self.promptDiagnostics = []
        self.themes = []
        self.themeDiagnostics = []
        self.agentsFiles = []
        self.systemPrompt = None
        self.systemPromptSourcePath = None
        self.appendSystemPrompt = []
        self.appendSystemPromptSourcePaths = []
        self.lastPromptPaths = []
        self.lastThemePaths = []
        self.extensionSkillPaths = []
        self.extensionSkillResources = []
        self.extensionPromptSourceInfos = {}
        self.extensionThemeSourceInfos = {}
        self.resourceMetadataByPath = {}
        self.loaded = False

    def getExtensions(self) -> LoadExtensionsResult:
        return self.extensionsResult

    def getPrompts(self) -> PromptsResult:
        return {"prompts": self.prompts, "diagnostics": self.promptDiagnostics}

    def getThemes(self) -> ThemesResult:
        return {"themes": self.themes, "diagnostics": self.themeDiagnostics}

    def getAgentsFiles(self) -> AgentsFilesResult:
        return {"agentsFiles": self.agentsFiles}

    def getSystemPrompt(self) -> str | None:
        return self.systemPrompt

    def getSystemPromptSource(self) -> dict[str, str] | None:
        return {"path": self.systemPromptSourcePath} if self.systemPromptSourcePath is not None else None

    def getAppendSystemPrompt(self) -> list[str]:
        return self.appendSystemPrompt

    def getAppendSystemPromptSources(self) -> list[dict[str, str]]:
        return [{"path": path} for path in self.appendSystemPromptSourcePaths]

    def extendResources(self, paths: ResourceExtensionPaths) -> None:
        prompt_paths = self.normalizeExtensionPaths(paths.get("promptPaths", []))
        theme_paths = self.normalizeExtensionPaths(paths.get("themePaths", []))

        for entry in prompt_paths:
            self.extensionPromptSourceInfos[entry["path"]] = createSourceInfo(entry["path"], entry["metadata"])
        for entry in theme_paths:
            self.extensionThemeSourceInfos[entry["path"]] = createSourceInfo(entry["path"], entry["metadata"])

        if prompt_paths:
            self.lastPromptPaths = self.mergePaths(self.lastPromptPaths, [entry["path"] for entry in prompt_paths])
            self.updatePromptsFromPaths(self.lastPromptPaths, self.resourceMetadataByPath)

        if theme_paths:
            self.lastThemePaths = self.mergePaths(self.lastThemePaths, [entry["path"] for entry in theme_paths])
            self.updateThemesFromPaths(self.lastThemePaths, self.resourceMetadataByPath)

        # MISAKA fork: skills are misaka's own layer system (core/skills), not this loader's;
        # the roots an extension contributes are kept here for that system to read.
        skill_paths = self.normalizeExtensionPaths(paths.get("skillPaths", []))
        if skill_paths:
            for entry in skill_paths:
                if entry not in self.extensionSkillResources:
                    self.extensionSkillResources.append(entry)
            self.extensionSkillPaths = self.mergePaths(
                self.getExtensionSkillPaths(), [entry["path"] for entry in skill_paths]
            )

    def getExtensionSkillPaths(self) -> list[str]:
        return list(self.extensionSkillPaths)

    def getExtensionSkillResources(self) -> list[ResourcePathEntry]:
        return [{"path": e["path"], "metadata": dict(e["metadata"])} for e in self.extensionSkillResources]

    async def loadProjectTrustExtensions(self) -> LoadExtensionsResult:
        """Bootstrap only user/CLI/inline extensions while project settings are gated."""
        self.settingsManager.setProjectTrusted(False)
        await self.settingsManager.reload()
        return await self.loadCurrentExtensionSet(include_inline_factories=True)

    async def reload(self, options: ResourceLoaderReloadOptions | None = None) -> None:
        resetTimings("extensions")

        if self.loaded:
            clearExtensionCache()

        pre_trust_extensions: LoadExtensionsResult | None = None
        resolve_project_trust = (options or {}).get("resolveProjectTrust")
        if resolve_project_trust is not None:
            pre_trust_extensions = await self.loadProjectTrustExtensions()
            project_trusted = await resolve_project_trust(
                {"extensionsResult": pre_trust_extensions}
            )
            self.settingsManager.setProjectTrusted(project_trusted)

        await self.settingsManager.reload()
        resolved_paths = await self.packageManager.resolve()
        cli_extension_paths = await self.packageManager.resolveExtensionSources(
            self.additionalExtensionPaths,
            {"temporary": True},
        )

        self.resourceMetadataByPath = {}
        metadata_by_path = self.resourceMetadataByPath
        self.extensionSkillPaths = []
        self.extensionSkillResources = []
        self.extensionPromptSourceInfos = {}
        self.extensionThemeSourceInfos = {}

        def getEnabledResources(resources: list[ResolvedResource]) -> list[ResolvedResource]:
            for resource in resources:
                if resource.path not in metadata_by_path:
                    metadata_by_path[resource.path] = resource.metadata
            return [resource for resource in resources if resource.enabled]

        def enabled_paths(resources: list[ResolvedResource]) -> list[str]:
            return [resource.path for resource in getEnabledResources(resources)]

        enabled_extensions = enabled_paths(resolved_paths.extensions)
        enabled_prompts = enabled_paths(resolved_paths.prompts)
        enabled_themes = enabled_paths(resolved_paths.themes)

        for resource in cli_extension_paths.extensions:
            if resource.path not in metadata_by_path:
                metadata_by_path[resource.path] = {"source": "cli", "scope": "temporary", "origin": "top-level"}

        cli_enabled_extensions = enabled_paths(cli_extension_paths.extensions)
        cli_enabled_prompts = enabled_paths(cli_extension_paths.prompts)
        cli_enabled_themes = enabled_paths(cli_extension_paths.themes)

        extension_paths = [
            path
            for path in (
                cli_enabled_extensions
                if self.noExtensions
                else self.mergePaths(cli_enabled_extensions, enabled_extensions)
            )
            if not path.startswith(BUILTIN_PATH_PREFIX)
            or path[len(BUILTIN_PATH_PREFIX) :] not in self.disabledBuiltinExtensions
        ]

        extensions_result = await self.loadFinalExtensionSet(
            extension_paths,
            pre_trust_extensions,
        )

        for raw_path in self.additionalExtensionPaths:
            if isLocalPath(raw_path):
                resolved = self.resolveResourcePath(raw_path)
                if not os.path.exists(resolved):
                    extensions_result.errors.append(
                        {"path": resolved, "error": f"Extension path does not exist: {resolved}"}
                    )

        self.extensionsResult = (
            self.extensionsOverride(extensions_result)
            if callable(self.extensionsOverride)
            else extensions_result
        )
        self.applyExtensionSourceInfo(self.extensionsResult.extensions, metadata_by_path)

        prompt_paths = (
            self.mergePaths(cli_enabled_prompts, self.additionalPromptTemplatePaths)
            if self.noPromptTemplates
            else self.mergePaths([*cli_enabled_prompts, *enabled_prompts], self.additionalPromptTemplatePaths)
        )
        self.lastPromptPaths = prompt_paths
        self.updatePromptsFromPaths(prompt_paths, metadata_by_path)
        for raw_path in self.additionalPromptTemplatePaths:
            if isLocalPath(raw_path):
                resolved = self.resolveResourcePath(raw_path)
                if not os.path.exists(resolved) and not any(
                    diagnostic.path == resolved for diagnostic in self.promptDiagnostics
                ):
                    self.promptDiagnostics.append(
                        ResourceDiagnostic(
                            type="error",
                            message="Prompt template path does not exist",
                            path=resolved,
                        )
                    )

        theme_paths = (
            self.mergePaths(cli_enabled_themes, self.additionalThemePaths)
            if self.noThemes
            else self.mergePaths([*cli_enabled_themes, *enabled_themes], self.additionalThemePaths)
        )
        self.lastThemePaths = theme_paths
        self.updateThemesFromPaths(theme_paths, metadata_by_path)
        for raw_path in self.additionalThemePaths:
            resolved = self.resolveResourcePath(raw_path)
            if not os.path.exists(resolved) and not any(
                diagnostic.path == resolved for diagnostic in self.themeDiagnostics
            ):
                self.themeDiagnostics.append(
                    ResourceDiagnostic(type="error", message="Theme path does not exist", path=resolved)
                )

        agents_files = {
            "agentsFiles": (
                [] if self.noContextFiles else loadProjectContextFiles({"cwd": self.cwd, "agentDir": self.agentDir})
            )
        }
        resolved_agents_files = (
            self.agentsFilesOverride(agents_files) if callable(self.agentsFilesOverride) else agents_files
        )
        self.agentsFiles = resolved_agents_files["agentsFiles"]

        discovered_system_prompt = self.discoverSystemPromptFile()
        system_prompt_source = self.systemPromptSource if self.systemPromptSource is not None else discovered_system_prompt
        base_system_prompt = resolvePromptInput(system_prompt_source, "system prompt")
        self.systemPrompt = (
            self.systemPromptOverride(base_system_prompt)
            if callable(self.systemPromptOverride)
            else base_system_prompt
        )
        self.systemPromptSourcePath = (
            resolvePath(system_prompt_source)
            if system_prompt_source is not None and os.path.exists(system_prompt_source)
            else None
        )

        discovered_append_prompt = self.discoverAppendSystemPromptFile()
        append_sources = self.appendSystemPromptSource if self.appendSystemPromptSource is not None else (
            [discovered_append_prompt] if discovered_append_prompt else []
        )
        base_append = [
            content
            for source in append_sources
            if (content := resolvePromptInput(source, "append system prompt")) is not None
        ]
        self.appendSystemPrompt = (
            self.appendSystemPromptOverride(base_append)
            if callable(self.appendSystemPromptOverride)
            else base_append
        )
        self.appendSystemPromptSourcePaths = [
            resolvePath(source) for source in append_sources if os.path.exists(source)
        ]
        self.loaded = True

    async def loadCurrentExtensionSet(
        self,
        *,
        include_inline_factories: bool,
    ) -> LoadExtensionsResult:
        resolved_paths = await self.packageManager.resolve()
        cli_paths = await self.packageManager.resolveExtensionSources(
            self.additionalExtensionPaths,
            {"temporary": True},
        )
        enabled_extensions = [item.path for item in resolved_paths.extensions if item.enabled]
        cli_enabled_extensions = [item.path for item in cli_paths.extensions if item.enabled]
        # Built-in extensions wait for the final pass: project settings can disable them, and a
        # loaded extension cannot be unloaded.
        extension_paths = [
            path
            for path in (
                cli_enabled_extensions
                if self.noExtensions
                else self.mergePaths(cli_enabled_extensions, enabled_extensions)
            )
            if not path.startswith(BUILTIN_PATH_PREFIX)
        ]
        result = await loadExtensionsCached(extension_paths, self.cwd, self.eventBus)
        if not include_inline_factories:
            return result

        inline = await self.loadExtensionFactories(result.runtime)
        result.extensions.extend(inline["extensions"])
        result.errors.extend(inline["errors"])
        replacement_warnings: list[dict[str, str]] = []
        result.extensions = omitReplacedExtensions(result.extensions, replacement_warnings)
        mergeExtensionWarnings(result, replacement_warnings)
        return result

    async def loadExtensionPaths(
        self, paths: list[str], runtime: ExtensionRuntime | None = None
    ) -> LoadExtensionsResult:
        """Load extension paths: files from disk and ``builtin:<name>`` paths from the built-in
        extensions."""

        def isBuiltinPath(path: str) -> bool:
            return path.startswith(BUILTIN_PATH_PREFIX)

        result = await loadExtensionsCached(
            [path for path in paths if not isBuiltinPath(path)], self.cwd, self.eventBus, runtime
        )
        for path in [path for path in paths if isBuiltinPath(path)]:
            builtin = self.builtinExtensions.get(path[len(BUILTIN_PATH_PREFIX):])
            if builtin is None:
                result.errors.append({"path": path, "error": f"Unknown built-in extension: {path}"})
                continue
            try:
                extension = await loadExtensionFromFactory(
                    builtin["factory"], self.cwd, self.eventBus, result.runtime, path
                )
                extension.hidden = True
                extension.replaceable = builtin.get("replaceable") is True
                result.extensions.append(extension)
            except Exception as error:  # noqa: BLE001 - extension code: a failing load is reported as an error entry
                result.errors.append({"path": path, "error": _error_message(error, "failed to load extension")})
        return result

    async def loadFinalExtensionSet(
        self,
        extension_paths: list[str],
        pre_trust_extensions: LoadExtensionsResult | None,
    ) -> LoadExtensionsResult:
        # Without a pre-trust pass nothing is preloaded, and inline extensions load here.
        preloaded = pre_trust_extensions.extensions if pre_trust_extensions is not None else []
        preloaded_by_path = {
            extension.resolvedPath: extension for extension in preloaded if not extension.path.startswith("<inline:")
        }
        failed_preload_paths = {
            self.resolveExtensionLoadPath(error["path"])
            for error in (pre_trust_extensions.errors if pre_trust_extensions is not None else [])
        }
        remaining_paths = [
            path
            for path in extension_paths
            if self.resolveExtensionLoadPath(path) not in preloaded_by_path
            and self.resolveExtensionLoadPath(path) not in failed_preload_paths
        ]
        remaining = await self.loadExtensionPaths(
            remaining_paths, pre_trust_extensions.runtime if pre_trust_extensions is not None else None
        )
        loaded_by_path = dict(preloaded_by_path)
        loaded_by_path.update({extension.resolvedPath: extension for extension in remaining.extensions})

        inline = (
            {"extensions": [extension for extension in preloaded if extension.path.startswith("<inline:")], "errors": []}
            if pre_trust_extensions is not None
            else await self.loadExtensionFactories(remaining.runtime)
        )
        ordered = [
            loaded_by_path[resolved]
            for path in extension_paths
            if (resolved := self.resolveExtensionLoadPath(path)) in loaded_by_path
        ]
        ordered.extend(inline["extensions"])

        replacement_warnings: list[dict[str, str]] = []
        result = LoadExtensionsResult(
            extensions=omitReplacedExtensions(ordered, replacement_warnings),
            errors=[
                *(pre_trust_extensions.errors if pre_trust_extensions is not None else []),
                *remaining.errors,
                *inline["errors"],
            ],
            warnings=[
                *(pre_trust_extensions.warnings if pre_trust_extensions is not None else []),
                *(remaining.warnings or []),
            ],
            runtime=remaining.runtime,
        )
        mergeExtensionWarnings(result, replacement_warnings)
        self.addExtensionConflictDiagnostics(result)
        return result

    def resolveExtensionLoadPath(self, path: str) -> str:
        # Keep the pre-trust/final-pass identity key byte-for-byte aligned with
        # extensions.loader._load_extension(), otherwise Unicode-space paths can
        # activate the same factory twice.
        return path if isSyntheticPath(path) else resolvePath(path, self.cwd, normalize_unicode_spaces=True)

    def addExtensionConflictDiagnostics(self, result: LoadExtensionsResult) -> None:
        for conflict in self.detectExtensionConflicts(result.extensions):
            result.errors.append({"path": conflict["path"], "error": conflict["message"]})

    def normalizeExtensionPaths(self, entries: list[ResourcePathEntry]) -> list[ResourcePathEntry]:
        normalized: list[ResourcePathEntry] = []
        for entry in entries:
            metadata = dict(entry.get("metadata", {}))
            if metadata.get("baseDir") is not None:
                metadata["baseDir"] = self.resolveResourcePath(str(metadata["baseDir"]))
            normalized.append(
                {
                    "path": self.resolveResourcePath(entry["path"]),
                    "metadata": metadata,  # type: ignore[typeddict-item]
                }
            )
        return normalized

    def updatePromptsFromPaths(
        self,
        prompt_paths: list[str],
        metadata_by_path: dict[str, PathMetadata] | None = None,
    ) -> None:
        if self.noPromptTemplates and not prompt_paths:
            prompts_result = {"prompts": [], "diagnostics": []}
        else:
            loaded = loadPromptTemplates(
                {
                    "cwd": self.cwd,
                    "agentDir": self.agentDir,
                    "promptPaths": prompt_paths,
                }
            )
            deduped = self.dedupePrompts(loaded.templates)
            prompts_result = {
                "prompts": deduped["prompts"],
                "diagnostics": [*loaded.diagnostics, *deduped["diagnostics"]],
            }

        resolved = self.promptsOverride(prompts_result) if callable(self.promptsOverride) else prompts_result
        self.prompts = [
            PromptTemplate(
                name=prompt.name,
                description=prompt.description,
                content=prompt.content,
                sourceInfo=self.findSourceInfoForPath(
                    prompt.filePath,
                    self.extensionPromptSourceInfos,
                    metadata_by_path,
                )
                or prompt.sourceInfo
                or self.getDefaultSourceInfoForPath(prompt.filePath),
                filePath=prompt.filePath,
                argumentHint=prompt.argumentHint,
            )
            for prompt in resolved["prompts"]
        ]
        self.promptDiagnostics = list(resolved["diagnostics"])

    def updateThemesFromPaths(
        self,
        theme_paths: list[str],
        metadata_by_path: dict[str, PathMetadata] | None = None,
    ) -> None:
        if self.noThemes and not theme_paths:
            themes_result = {"themes": [], "diagnostics": []}
        else:
            # Theme construction only needs trueColor, so skip the unrelated tmux hyperlink probe.
            colorMode = getTerminalColorMode(
                dataclasses.replace(
                    detectCapabilities(lambda: False),
                    **self.settingsManager.getTerminalCapabilityOverrides(),
                )
            )
            loaded = self.loadThemes(theme_paths, colorMode)
            deduped = self.dedupeThemes(loaded["themes"])
            themes_result = {
                "themes": deduped["themes"],
                "diagnostics": [*loaded["diagnostics"], *deduped["diagnostics"]],
            }

        resolved = self.themesOverride(themes_result) if callable(self.themesOverride) else themes_result
        self.themes = []
        for theme in resolved["themes"]:
            source_path = theme.sourcePath
            theme.sourceInfo = (
                self.findSourceInfoForPath(
                    source_path,
                    self.extensionThemeSourceInfos,
                    metadata_by_path,
                )
                if source_path
                else None
            ) or theme.sourceInfo
            if source_path and theme.sourceInfo is None:
                theme.sourceInfo = self.getDefaultSourceInfoForPath(source_path)
            self.themes.append(theme)
        self.themeDiagnostics = list(resolved["diagnostics"])

    def applyExtensionSourceInfo(
        self,
        extensions: list[Extension],
        metadata_by_path: dict[str, PathMetadata] | None = None,
    ) -> None:
        for extension in extensions:
            extension.sourceInfo = (
                self.findSourceInfoForPath(extension.path, None, metadata_by_path)
                or self.getDefaultSourceInfoForPath(extension.path)
            )
            for command in extension.commands.values():
                command.sourceInfo = extension.sourceInfo
            for tool in extension.tools.values():
                tool.sourceInfo = extension.sourceInfo

    def findSourceInfoForPath(
        self,
        resource_path: str,
        extra_source_infos: dict[str, SourceInfo] | None = None,
        metadata_by_path: dict[str, PathMetadata] | None = None,
    ) -> SourceInfo | None:
        if not resource_path:
            return None
        if isSyntheticPath(resource_path):
            return self.getDefaultSourceInfoForPath(resource_path)

        normalized_resource_path = os.path.abspath(resource_path)
        for source_path, source_info in (extra_source_infos or {}).items():
            normalized_source_path = os.path.abspath(source_path)
            if normalized_resource_path == normalized_source_path or normalized_resource_path.startswith(
                f"{normalized_source_path}{os.sep}"
            ):
                return SourceInfo(
                    path=resource_path,
                    source=source_info.source,
                    scope=source_info.scope,
                    origin=source_info.origin,
                    baseDir=source_info.baseDir,
                )

        if metadata_by_path is not None:
            exact = metadata_by_path.get(normalized_resource_path) or metadata_by_path.get(resource_path)
            if exact is not None:
                return createSourceInfo(resource_path, exact)

            for source_path, metadata in metadata_by_path.items():
                normalized_source_path = os.path.abspath(source_path)
                if normalized_resource_path == normalized_source_path or normalized_resource_path.startswith(
                    f"{normalized_source_path}{os.sep}"
                ):
                    return createSourceInfo(resource_path, metadata)
        return None

    def getDefaultSourceInfoForPath(self, file_path: str) -> SourceInfo:
        synthetic_source = getSyntheticPathSource(file_path)
        if synthetic_source:
            return SourceInfo(path=file_path, source=synthetic_source, scope="temporary", origin="top-level", baseDir=None)

        normalized_path = os.path.abspath(file_path)
        agent_roots = [
            os.path.join(self.agentDir, "skills"),
            os.path.join(self.agentDir, "prompts"),
            os.path.join(self.agentDir, "themes"),
            os.path.join(self.agentDir, "extensions"),
        ]
        # Trusted project prompts/themes are classified from package metadata above.
        # Only user roots need this fallback; project extensions/system prompts stay disabled,
        # while skills remain owned by misaka.core.skills.layers.
        for root in agent_roots:
            if self.isUnderPath(normalized_path, root):
                return SourceInfo(path=file_path, source="local", scope="user", origin="top-level", baseDir=root)

        stats = os.stat(normalized_path)
        base_dir = (
            normalized_path
            if stat_module.S_ISDIR(stats.st_mode)
            else os.path.abspath(os.path.join(normalized_path, ".."))
        )
        return SourceInfo(
            path=file_path,
            source="local",
            scope="temporary",
            origin="top-level",
            baseDir=base_dir,
        )

    def mergePaths(self, primary: list[str], additional: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for value in [*primary, *additional]:
            resolved = self.resolveResourcePath(value)
            canonical = canonicalizePath(resolved)
            if canonical in seen:
                continue
            seen.add(canonical)
            merged.append(resolved)
        return merged

    def resolveResourcePath(self, path: str) -> str:
        return path if isSyntheticPath(path) else resolvePath(path, self.cwd, trim=True)

    def loadThemes(self, paths: list[str], colorMode: TerminalColorMode) -> dict[str, list[Any]]:
        # MISAKA fork: no `includeDefaults` parameter (only ever false); it used to auto-load
        # `<agentDir>/themes` here and was removed with its dead branch.
        themes: list[Theme] = []
        diagnostics: list[ResourceDiagnostic] = []

        for path in paths:
            resolved = self.resolveResourcePath(path)
            if not os.path.exists(resolved):
                diagnostics.append(
                    ResourceDiagnostic(type="warning", message="theme path does not exist", path=resolved)
                )
                continue
            try:
                if os.path.isdir(resolved):
                    self.loadThemesFromDir(resolved, themes, diagnostics, colorMode)
                elif os.path.isfile(resolved) and resolved.endswith(".json"):
                    self.loadThemeFromFile(resolved, themes, diagnostics, colorMode)
                else:
                    diagnostics.append(
                        ResourceDiagnostic(type="warning", message="theme path is not a json file", path=resolved)
                    )
            except Exception as error:  # noqa: BLE001
                diagnostics.append(
                    ResourceDiagnostic(
                        type="warning",
                        message=_error_message(error, "failed to read theme path"),
                        path=resolved,
                    )
                )
        return {"themes": themes, "diagnostics": diagnostics}

    def loadThemesFromDir(
        self,
        dir_path: str,
        themes: list[Theme],
        diagnostics: list[ResourceDiagnostic],
        colorMode: TerminalColorMode,
    ) -> None:
        if not os.path.exists(dir_path):
            return
        try:
            for entry in readdirEntries(dir_path):
                is_file = entry.is_file(follow_symlinks=False)
                if entry.is_symlink():
                    try:
                        is_file = os.path.isfile(entry.path)
                    except Exception:  # noqa: BLE001, S112 - an unreadable entry is skipped
                        continue
                if not is_file or not entry.name.endswith(".json"):
                    continue
                self.loadThemeFromFile(entry.path, themes, diagnostics, colorMode)
        except Exception as error:  # noqa: BLE001
            diagnostics.append(
                ResourceDiagnostic(
                    type="warning",
                    message=_error_message(error, "failed to read theme directory"),
                    path=dir_path,
                )
            )

    def loadThemeFromFile(
        self,
        file_path: str,
        themes: list[Theme],
        diagnostics: list[ResourceDiagnostic],
        colorMode: TerminalColorMode,
    ) -> None:
        try:
            themes.append(loadThemeFromPath(file_path, colorMode))
        except Exception as error:  # noqa: BLE001 - a broken theme file is a diagnostic, not a crash
            diagnostics.append(
                ResourceDiagnostic(
                    type="warning",
                    message=_error_message(error, "failed to load theme"),
                    path=file_path,
                )
            )

    async def loadExtensionFactories(self, runtime: ExtensionRuntime) -> dict[str, list[Any]]:
        extensions: list[Extension] = []
        errors: list[dict[str, str]] = []
        for index, input_ in enumerate(self.extensionFactories, start=1):
            # Port of pi resource-loader.ts loadExtensionFactories: an InlineExtension is a bare
            # factory or {name, factory, hidden}; named ones show as <inline:name>, hidden ones
            # stay off the startup screen.
            is_named = not callable(input_)
            factory = input_.get("factory") if is_named else input_
            extension_path = f"<inline:{input_.get('name') if is_named else index}>"
            try:
                extension = await loadExtensionFromFactory(
                    factory,
                    self.cwd,
                    self.eventBus,
                    runtime,
                    extension_path,
                )
                extension.hidden = bool(is_named and input_.get("hidden"))
                extension.replaceable = bool(is_named and input_.get("replaceable") is True)
                extensions.append(extension)
            except Exception as error:  # noqa: BLE001 - extension code: a failing load is reported as an error entry
                errors.append({"path": extension_path, "error": _error_message(error, "failed to load extension")})
        return {"extensions": extensions, "errors": errors}

    def dedupePrompts(self, prompts: list[PromptTemplate]) -> dict[str, list[Any]]:
        seen: dict[str, PromptTemplate] = {}
        diagnostics: list[ResourceDiagnostic] = []
        for prompt in prompts:
            existing = seen.get(prompt.name)
            if existing is None:
                seen[prompt.name] = prompt
                continue
            diagnostics.append(
                ResourceDiagnostic(
                    type="collision",
                    message=f'name "/{prompt.name}" collision',
                    path=prompt.filePath,
                    collision=ResourceCollision(
                        resourceType="prompt",
                        name=prompt.name,
                        winnerPath=existing.filePath,
                        loserPath=prompt.filePath,
                    ),
                )
            )
        return {"prompts": list(seen.values()), "diagnostics": diagnostics}

    def dedupeThemes(self, themes: list[Theme]) -> dict[str, list[Any]]:
        seen: dict[str, Theme] = {}
        diagnostics: list[ResourceDiagnostic] = []
        for theme in themes:
            name = theme.name or "unnamed"
            existing = seen.get(name)
            if existing is None:
                seen[name] = theme
                continue
            diagnostics.append(
                ResourceDiagnostic(
                    type="collision",
                    message=f'name "{name}" collision',
                    path=theme.sourcePath,
                    collision=ResourceCollision(
                        resourceType="theme",
                        name=name,
                        winnerPath=existing.sourcePath or "<builtin>",
                        loserPath=theme.sourcePath or "<builtin>",
                    ),
                )
            )
        return {"themes": list(seen.values()), "diagnostics": diagnostics}

    def discoverSystemPromptFile(self) -> str | None:
        global_path = os.path.join(self.agentDir, "SYSTEM.md")
        if os.path.exists(global_path):
            return global_path
        return None

    def discoverAppendSystemPromptFile(self) -> str | None:
        global_path = os.path.join(self.agentDir, "APPEND_SYSTEM.md")
        if os.path.exists(global_path):
            return global_path
        return None

    def isUnderPath(self, target: str, root: str) -> bool:
        normalized_root = os.path.abspath(root)
        normalized_target = os.path.abspath(target)
        if normalized_target == normalized_root:
            return True
        prefix = normalized_root if normalized_root.endswith(os.sep) else f"{normalized_root}{os.sep}"
        return normalized_target.startswith(prefix)

    def detectExtensionConflicts(self, extensions: list[Extension]) -> list[dict[str, str]]:
        conflicts: list[dict[str, str]] = []
        tool_owners: dict[str, str] = {}
        flag_owners: dict[str, str] = {}
        for extension in extensions:
            for tool_name in extension.tools:
                existing_owner = tool_owners.get(tool_name)
                if existing_owner is not None and existing_owner != extension.path:
                    conflicts.append(
                        {"path": extension.path, "message": f'Tool "{tool_name}" conflicts with {existing_owner}'}
                    )
                else:
                    tool_owners[tool_name] = extension.path
            for flag_name in extension.flags:
                existing_owner = flag_owners.get(flag_name)
                if existing_owner is not None and existing_owner != extension.path:
                    conflicts.append(
                        {"path": extension.path, "message": f'Flag "--{flag_name}" conflicts with {existing_owner}'}
                    )
                else:
                    flag_owners[flag_name] = extension.path
        return conflicts


# MISAKA fork: named, because the skills guard refuses writes to them (issue #10 audit H4).
# The files a session in a folder loads as instructions, from that folder and every one above it.
CONTEXT_FILE_NAMES = ("PROJECT.md", "AGENTS.override.md", "AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD")  # pi 8ecf8a9 + the project brief


def loadContextFileFromDir(dir_path: str) -> dict[str, str] | None:
    for filename in CONTEXT_FILE_NAMES:
        file_path = os.path.join(dir_path, filename)
        # `isfile` rather than `exists`: pi's resource-loader.ts:71-90 stats and skips
        # anything that is not a regular file before reading, and the port dropped that
        # line. A directory only costs an IsADirectoryError, but a FIFO named AGENTS.md
        # makes `read_text` block forever -- and `loadProjectContextFiles` walks every
        # ancestor up to the filesystem root, so any one of them can hang startup with no
        # timeout and no diagnostic. A symlink to /dev/zero reads until OOM.
        if not os.path.isfile(file_path):
            continue
        try:
            return {"path": file_path, "content": Path(file_path).read_text(encoding="utf-8-sig")}
        except OSError as error:
            _warn(f"Warning: Could not read {file_path}: {error}")
            continue
    return None


ResourceLoaderLike = ResourceLoader


__all__ = [
    "DefaultResourceLoader",
    "DefaultResourceLoaderOptions",
    "ResourceCollision",
    "ResourceDiagnostic",
    "ResourceExtensionPaths",
    "ResourceLoader",
    "ResourceLoaderLike",
    "ResourceLoaderReloadOptions",
    "isBuiltinExtension",
]

# Kept for misaka/core/subagent, which imports these names.
resolve_prompt_input = resolvePromptInput
