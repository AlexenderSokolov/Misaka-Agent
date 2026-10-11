"""pi coding-agent's ``test/experimental-presentation-facets.test.ts``.

MISAKA fork: a plugin package is a directory with a ``pyproject.toml`` and Python entries (``src/session.py``, ``src/tui.py``);
the example plugin is ``examples/plugins/pi-example-plugin``. pi's test of the server-built plugin cache is one test; the
part that starts a server needs Unix sockets, so it is a second test here and the build part runs everywhere.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import uuid
from pathlib import Path

import pytest

from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.chord.node import readFacetBundleManifest
from misaka.experimental.client_runtime import (
    activateBuiltinClientServices,
    openClientRuntime,
)
from misaka.experimental.plugins.bundled import (
    createPresentationFacetData,
    createPresentationFacetLoaders,
)
from misaka.experimental.plugins.package import (
    createServerPluginPackage,
    restoreServerPluginPackageProfile,
)
from misaka.experimental.server import startServer
from misaka.experimental.services.plugins import PresentationPlugins
from tests.experimental_session_support import makeShortDirectory

EXAMPLE_PLUGIN = str(Path(__file__).resolve().parents[1] / "examples" / "plugins" / "pi-example-plugin")
posixOnly = pytest.mark.skipif(sys.platform == "win32", reason="the experimental server listens on a Unix socket")


@pytest.fixture
def directory():
    path = makeShortDirectory("pi-presentation-")
    yield path
    shutil.rmtree(path, ignore_errors=True)


def facetSource(facetId: str, uses: bool = True) -> str:
    return (
        "from types import SimpleNamespace\n"
        "\n"
        "from misaka.chord import defineFacet\n"
        "from misaka.experimental.plugin import SlashCommands\n"
        "\n"
        "\n"
        "def setup(env):\n"
        f"    {'env.use(SlashCommands)' if uses else 'pass'}\n"
        "\n"
        "\n"
        f'default = defineFacet(SimpleNamespace(id="{facetId}", setup=setup))\n'
    )


def writePackage(path: str, name: str, source: str) -> str:
    os.makedirs(os.path.join(path, "src"), exist_ok=True)
    with open(os.path.join(path, "pyproject.toml"), "w", encoding="utf-8", newline="\n") as file:
        file.write(f'[project]\nname = "{name}"\nversion = "1.0.0"\n')
    sourcePath = os.path.join(path, "src", "tui.py")
    with open(sourcePath, "w", encoding="utf-8", newline="\n") as file:
        file.write(source)
    return sourcePath


async def loadedIds(loaders):
    loaded = [await loader.load() for loader in loaders]
    ids = [facet.id for entry in loaded for facet in entry.facets]
    for entry in loaded:
        await entry.dispose()
    return ids


async def test_rejects_local_plugin_paths_for_radius_servers():
    with pytest.raises(RuntimeError, match="only be configured on a local Unix server"):
        await openClientRuntime(
            {
                "command": "client",
                "connect": {"transport": "radius", "serverId": str(uuid.uuid4())},
                "pluginPackages": ["./local-plugin"],
            }
        )


async def test_restores_plugin_package_selections_for_later_server_generations(directory):
    serverId = str(uuid.uuid4())
    packagePaths = [os.path.join(directory, "first-plugin"), os.path.join(directory, "second-plugin")]
    assert list(await restoreServerPluginPackageProfile(directory, serverId, packagePaths)) == packagePaths
    assert list(await restoreServerPluginPackageProfile(directory, serverId)) == packagePaths
    assert list(await restoreServerPluginPackageProfile(directory, serverId, [])) == []
    assert list(await restoreServerPluginPackageProfile(directory, serverId)) == []


async def test_builds_conventional_plugin_entries_into_the_server_owned_plugin_cache(directory):
    serverId = str(uuid.uuid4())
    packagePath = os.path.join(directory, "pi-example-plugin")
    sourcePath = writePackage(packagePath, "test-plugin", facetSource("built-a"))
    plugin = createServerPluginPackage(directory, serverId, packagePath)

    first = await plugin.build()
    assert len(first) == 1
    assert re.search(
        rf"[\\/]plugin-builds[\\/]{serverId}[\\/]pi-example-plugin-[a-f0-9]{{12}}[\\/]chord-facets\.json$", plugin.manifestPath
    )
    assert first[0]["plugin"] == {"id": "test-plugin", "version": "1.0.0"}
    assert await loadedIds(createPresentationFacetLoaders(createPresentationFacetData(first))) == ["built-a"]

    Path(sourcePath).write_bytes(facetSource("built-b").encode("utf-8"))
    second = await plugin.build()
    assert second[0]["source"] != first[0]["source"]
    assert await loadedIds(createPresentationFacetLoaders(createPresentationFacetData(second))) == ["built-b"]


@posixOnly
async def test_serves_server_selected_presentation_facets_and_reloads_them(directory):
    packagePath = os.path.join(directory, "pi-example-plugin")
    sourcePath = writePackage(packagePath, "test-plugin", facetSource("built-b"))
    secondPackagePath = os.path.join(directory, "second-plugin")
    writePackage(secondPackagePath, "second-test-plugin", facetSource("second-built", uses=False))
    running = await startServer(
        {"directory": os.path.join(directory, "server"), "sessionDir": os.path.join(directory, "sessions")}
    )
    runtime = None
    try:
        runtime = await openClientRuntime(
            {
                "command": "client",
                "connect": {"transport": "unix", "path": running.socketPath},
                "pluginPackages": [packagePath, secondPackagePath],
            }
        )
        activated = await activateBuiltinClientServices(runtime.servers[0])
        sessionId = str(uuid.uuid4())
        await activated.management.create({"id": sessionId}, BACKGROUND_CONTEXT)
        presentationPlugins = await activated.plugins.prepareSession(
            {"sessionId": sessionId, "packagePaths": [packagePath, secondPackagePath]}, BACKGROUND_CONTEXT
        )
        assert list(await restoreServerPluginPackageProfile(os.path.join(directory, "server"), running.serverId)) == []
        assert await loadedIds(createPresentationFacetLoaders(presentationPlugins)) == ["built-b", "second-built"]

        Path(sourcePath).write_bytes(facetSource("built-c").encode("utf-8"))
        services = runtime.servers[0].server.open(
            {"services": [PresentationPlugins], "assertAccess": lambda: None, "onError": lambda error: None}
        )
        try:
            await services.ready(BACKGROUND_CONTEXT)
            data = await services.use(PresentationPlugins).reload(BACKGROUND_CONTEXT)
            assert await loadedIds(createPresentationFacetLoaders(data)) == ["built-c", "second-built"]
        finally:
            await services.dispose(BACKGROUND_CONTEXT)
    finally:
        if runtime is not None:
            await runtime.dispose()
        await running.close()


async def test_builds_the_example_plugin_package_without_a_package_owned_build_script(directory):
    plugin = createServerPluginPackage(directory, str(uuid.uuid4()), EXAMPLE_PLUGIN)

    artifacts = await plugin.build()
    manifest = await readFacetBundleManifest(plugin.manifestPath)
    assert manifest["plugin"] == {"id": "pi-example-plugin", "version": "1.0.0"}
    assert list(manifest["entries"]) == ["session", "tui"]
    assert await loadedIds(createPresentationFacetLoaders(createPresentationFacetData(artifacts))) == [
        "pi-example-plugin/tui"
    ]
