"""The Python facet bundles of ``misaka.chord.node``, adapted from pi chord's ``test/bundle.test.ts``.

pi bundles TypeScript facet entries with esbuild into CommonJS files; here an entry is a Python module and a bundle is a zip
archive of the module and the files it imports relatively (see ``misaka/chord/node/manifest.py``). The cases are the
upstream ones that have a counterpart: content-addressed entries, fresh generations that reload without disturbing
consumers, packages with conventional and configured entries, configuration that escapes the package, corrupt entries and
invalid exports.
"""

from __future__ import annotations

import copy
import os
import re
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from misaka.chord import createFacetHost, defineFacet, defineService
from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.chord.node import (
    FACET_BUNDLE_MANIFEST_FILE,
    bundleFacetPackage,
    bundleFacets,
    createFacetBundleArtifactLoader,
    createFacetBundleLoader,
    readFacetBundleArtifact,
    readFacetBundleManifest,
)

GenerationValue = defineService("test.bundle.generation", local=True)


@pytest.fixture
def directory() -> Iterator[Path]:
    path = Path(tempfile.mkdtemp(prefix="chord-node-test-"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def writeGeneration(path: Path, generation: str) -> None:
    write(
        path,
        "from types import SimpleNamespace\n"
        "from misaka.chord import defineService\n"
        "from .helper import decorate\n"
        'Value = defineService("test.bundle.generation", local=True)\n'
        "def setup(env):\n"
        f"    env.provide(Value, {{'read': lambda *_: decorate({generation!r})}})\n"
        'default = SimpleNamespace(id="bundle-provider", setup=setup)\n',
    )


def archiveNames(outputDirectory: Path, file: str) -> list[str]:
    with zipfile.ZipFile(outputDirectory / file) as archive:
        return sorted(archive.namelist())


async def test_builds_independent_content_addressed_entries_and_loads_fresh_reloadable_generations(directory: Path):
    sourceDirectory = directory / "src"
    outputDirectory = directory / "bundle"
    write(sourceDirectory / "helper.py", 'def decorate(value):\n    return "generation:" + value\n')
    entryPath = sourceDirectory / "entry.py"
    presentationPath = sourceDirectory / "presentation.py"
    writeGeneration(entryPath, "A")
    write(
        presentationPath,
        'from types import SimpleNamespace\ndefault = SimpleNamespace(id="bundle-presentation", setup=lambda env: None)\n',
    )
    facetEntries = {"presentation": str(presentationPath), "worker": str(entryPath)}

    firstBuild = await bundleFacets(
        {
            "plugin": {"id": "test-bundle", "version": "1"},
            "entries": facetEntries,
            "outdir": str(outputDirectory),
            "workingDirectory": str(directory),
        }
    )
    firstEntry = firstBuild["manifest"]["entries"]["worker"]
    assert list(firstBuild["manifest"]["entries"]) == ["presentation", "worker"]
    assert re.fullmatch(r"facet-[a-f0-9]{12}-[0-9A-F]{16}\.zip", firstEntry["file"])
    assert firstEntry["module"] == "src/entry.py"
    assert firstEntry["integrity"].startswith("sha256-")
    assert firstBuild["manifest"]["entries"]["presentation"]["file"] != firstEntry["file"]
    assert len([name for name in os.listdir(outputDirectory) if name.endswith(".zip")]) == 2
    # The worker's bundle holds the entry and what it imports, not the other entry.
    assert archiveNames(outputDirectory, firstEntry["file"]) == ["src/entry.py", "src/helper.py"]
    assert archiveNames(outputDirectory, firstBuild["manifest"]["entries"]["presentation"]["file"]) == [
        "src/presentation.py"
    ]
    assert Path(firstBuild["manifestPath"]).read_bytes().endswith(b"\n")
    assert Path(firstBuild["manifestPath"]).name == FACET_BUNDLE_MANIFEST_FILE

    presentation = await createFacetBundleLoader(
        {"manifestPath": firstBuild["manifestPath"], "entry": "presentation"}
    ).load()
    assert [facet.id for facet in presentation.facets] == ["bundle-presentation"]
    await presentation.dispose()

    artifact = await readFacetBundleArtifact({"manifestPath": firstBuild["manifestPath"], "entry": "worker"})
    materializedDirectory = directory / "materialized"
    transported = await createFacetBundleArtifactLoader(
        {"artifact": copy.deepcopy(artifact), "temporaryDirectory": str(materializedDirectory)}
    ).load()
    assert [facet.id for facet in transported.facets] == ["bundle-provider"]
    assert len(os.listdir(materializedDirectory)) == 1
    await transported.dispose()
    assert os.listdir(materializedDirectory) == []

    secondBuild = await bundleFacets(
        {
            "plugin": {"id": "test-bundle", "version": "1"},
            "entries": facetEntries,
            "outdir": str(outputDirectory),
            "workingDirectory": str(directory),
        }
    )
    assert secondBuild["manifest"]["entries"]["worker"] == firstEntry

    loader = createFacetBundleLoader({"manifestPath": secondBuild["manifestPath"], "entry": "worker"})
    loadedA = await loader.load()
    loadedACopy = await loader.load()
    assert loadedACopy.facets[0] is not loadedA.facets[0]
    await loadedACopy.dispose()

    retained: list[Any] = []
    consumer = defineFacet(SimpleNamespace(id="bundle-consumer", setup=lambda env: retained.append(env.use(GenerationValue))))
    host = await createFacetHost({"facets": [consumer, *loadedA.facets]})
    assert retained[0].read(BACKGROUND_CONTEXT) == "generation:A"

    writeGeneration(entryPath, "B")
    thirdBuild = await bundleFacets(
        {
            "plugin": {"id": "test-bundle", "version": "2"},
            "entries": facetEntries,
            "outdir": str(outputDirectory),
            "workingDirectory": str(directory),
        }
    )
    assert thirdBuild["manifest"]["entries"]["worker"]["file"] != firstEntry["file"]
    loadedB = await loader.load()
    await host.reload(loadedB.facets)
    await loadedA.dispose()
    assert retained[0].read(BACKGROUND_CONTEXT) == "generation:B"

    await host.dispose()
    await loadedB.dispose()


async def test_a_generation_does_not_leave_its_modules_or_files_behind(directory: Path):
    write(directory / "src" / "helper.py", "def decorate(value):\n    return value\n")
    entryPath = directory / "src" / "entry.py"
    writeGeneration(entryPath, "A")
    result = await bundleFacets(
        {"plugin": {"id": "p"}, "entries": {"worker": str(entryPath)}, "outdir": str(directory / "out"), "workingDirectory": str(directory)}
    )
    before = {name for name in sys.modules if name.startswith("chord_facet_")}
    loaded = await createFacetBundleLoader({"manifestPath": result["manifestPath"], "entry": "worker"}).load()
    during = {name for name in sys.modules if name.startswith("chord_facet_")} - before
    assert any(name.endswith(".src.entry") for name in during)
    assert any(name.endswith(".src.helper") for name in during)
    await loaded.dispose()
    assert {name for name in sys.modules if name.startswith("chord_facet_")} == before
    assert loaded.facets == ()


async def test_leaves_imports_of_other_modules_to_the_loading_application(directory: Path):
    write(directory / "sibling.py", "VALUE = 1\n")
    write(
        directory / "entry.py",
        "import json\n"
        "from types import SimpleNamespace\n"
        "import sibling\n"
        'default = SimpleNamespace(id="external-facet", setup=lambda env: None)\n',
    )
    result = await bundleFacets(
        {"plugin": {"id": "p"}, "entries": {"worker": str(directory / "entry.py")}, "outdir": str(directory / "out"), "workingDirectory": str(directory)}
    )
    # The absolute import of a sibling is not followed, so nothing bundles it.
    assert archiveNames(directory / "out", result["manifest"]["entries"]["worker"]["file"]) == ["entry.py"]
    with pytest.raises(Exception, match="Could not load facet bundle entry p/worker"):
        await createFacetBundleLoader({"manifestPath": result["manifestPath"], "entry": "worker"}).load()
    sys.path.insert(0, str(directory))
    try:
        loaded = await createFacetBundleLoader({"manifestPath": result["manifestPath"], "entry": "worker"}).load()
        assert [facet.id for facet in loaded.facets] == ["external-facet"]
        await loaded.dispose()
    finally:
        sys.path.remove(str(directory))
        sys.modules.pop("sibling", None)


async def test_a_list_of_facets_is_exported_as_default(directory: Path):
    write(
        directory / "entry.py",
        "from types import SimpleNamespace\n"
        "default = [SimpleNamespace(id='one', setup=lambda env: None), SimpleNamespace(id='two', setup=lambda env: None)]\n",
    )
    result = await bundleFacets(
        {"plugin": {"id": "p"}, "entries": {"worker": str(directory / "entry.py")}, "outdir": str(directory / "out"), "workingDirectory": str(directory)}
    )
    loaded = await createFacetBundleLoader({"manifestPath": result["manifestPath"], "entry": "worker"}).load()
    assert [facet.id for facet in loaded.facets] == ["one", "two"]
    await loaded.dispose()


async def test_builds_plugin_packages_from_conventional_and_configured_facet_entries(directory: Path):
    sourceDirectory = directory / "src"
    write(directory / "pyproject.toml", '[project]\nname = "example-conventional-plugin"\nversion = "1.2.3"\n')
    for name in ("session", "tui", "presentation"):
        write(
            sourceDirectory / f"{name}.py",
            "from types import SimpleNamespace\n"
            + (
                'default = SimpleNamespace(id="configured-tui", setup=lambda env: None)\n'
                if name == "presentation"
                else f'default = SimpleNamespace(id="package-{name}", setup=lambda env: None)\n'
            ),
        )
    write(sourceDirectory / "contract.py", "ignored = True\n")

    conventional = await bundleFacetPackage(
        {
            "packagePath": str(directory),
            "outdir": str(directory / "build"),
            "defaultFacets": {"session": "src/session.py", "tui": "src/tui.py", "browser": "src/browser.py"},
        }
    )
    assert conventional["packageDirectory"] == os.path.realpath(directory)
    assert conventional["manifest"]["plugin"] == {"id": "example-conventional-plugin", "version": "1.2.3"}
    assert list(conventional["manifest"]["entries"]) == ["session", "tui"]
    # What no entry imports is not bundled.
    assert archiveNames(directory / "build", conventional["manifest"]["entries"]["tui"]["file"]) == ["src/tui.py"]

    write(
        directory / "pyproject.toml",
        '[project]\nname = "example-conventional-plugin"\nversion = "2.0.0"\n\n'
        '[tool.chord.facets]\nsession = false\ntui = "src/presentation.py"\n',
    )
    configured = await bundleFacetPackage(
        {
            "packagePath": str(directory / "pyproject.toml"),
            "outdir": str(directory / "build"),
            "defaultFacets": {"session": "src/session.py", "tui": "src/tui.py"},
        }
    )
    assert list(configured["manifest"]["entries"]) == ["tui"]
    loaded = await createFacetBundleLoader({"manifestPath": configured["manifestPath"], "entry": "tui"}).load()
    assert [facet.id for facet in loaded.facets] == ["configured-tui"]
    await loaded.dispose()


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ('[project]\nname = "invalid"\nversion = "1.0.0"\n[tool.chord.facets]\ntui = "../outside.py"\n', "escapes the package directory"),
        ('[project]\nname = "invalid"\nversion = "1.0.0"\n[tool.chord.facets]\ntui = "/abs/outside.py"\n', "must be relative to the package directory|escapes the package directory"),
        ('[project]\nname = "invalid"\nversion = "1.0.0"\n[tool.chord.facets]\ntui = "src/missing.py"\n', "Could not access configured facet entry tui"),
        ('[project]\nname = "invalid"\nversion = "1.0.0"\n[tool.chord]\nexternal = []\n', "has an unknown field"),
        ('[project]\nname = "invalid"\nversion = "1.0.0"\n', "has no configured or conventional facet entries"),
        ('[project]\nname = ""\nversion = "1.0.0"\n', "non-empty name"),
        ('[project]\nname = "x"\n', "non-empty version"),
        ("not toml [", "Could not read facet package metadata"),
    ],
)
async def test_rejects_invalid_plugin_package_configuration(directory: Path, metadata: str, message: str):
    write(directory / "pyproject.toml", metadata)
    with pytest.raises(Exception, match=message):
        await bundleFacetPackage({"packagePath": str(directory), "outdir": str(directory / "build")})


async def test_rejects_corrupt_entries_and_invalid_module_exports(directory: Path):
    entryPath = directory / "entry.py"
    outputDirectory = directory / "bundle"
    write(entryPath, "from types import SimpleNamespace\ndefault = SimpleNamespace(id='missing-setup')\n")
    result = await bundleFacets(
        {"plugin": {"id": "invalid-bundle"}, "entries": {"invalid": str(entryPath)}, "outdir": str(outputDirectory), "workingDirectory": str(directory)}
    )
    loader = createFacetBundleLoader({"manifestPath": result["manifestPath"], "entry": "invalid"})
    with pytest.raises(Exception, match="has no setup function"):
        await loader.load()

    manifest = await readFacetBundleManifest(result["manifestPath"])
    (outputDirectory / manifest["entries"]["invalid"]["file"]).write_bytes(b"PK corrupted")
    with pytest.raises(Exception, match="integrity check failed"):
        await loader.load()
    with pytest.raises(Exception, match="integrity check failed"):
        await readFacetBundleArtifact({"manifestPath": result["manifestPath"], "entry": "invalid"})


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("default = None\n", "has a facet with an invalid ID"),
        ("default = []\n", "exported no facets"),
        ("x = 1\n", "has a facet with an invalid ID"),
        (
            (
                "from types import SimpleNamespace\n"
                "default = [SimpleNamespace(id='same', setup=lambda e: None), SimpleNamespace(id='same', setup=lambda e: None)]\n"
            ),
            "exports duplicate facet IDs",
        ),
        ("raise RuntimeError('boom')\n", "boom"),
    ],
)
async def test_rejects_entries_that_do_not_export_facets(directory: Path, source: str, message: str):
    write(directory / "entry.py", source)
    result = await bundleFacets(
        {"plugin": {"id": "p"}, "entries": {"worker": str(directory / "entry.py")}, "outdir": str(directory / "out"), "workingDirectory": str(directory)}
    )
    with pytest.raises(Exception, match=message):
        await createFacetBundleLoader({"manifestPath": result["manifestPath"], "entry": "worker"}).load()


async def test_rejects_invalid_bundle_options_and_unreadable_entries(directory: Path):
    with pytest.raises(TypeError, match="plugin ID must not be empty"):
        await bundleFacets({"plugin": {"id": ""}, "entries": {"a": "x.py"}, "outdir": "out"})
    with pytest.raises(TypeError, match="at least one entry"):
        await bundleFacets({"plugin": {"id": "p"}, "entries": {}, "outdir": "out"})
    with pytest.raises(TypeError, match="entry name must not be empty"):
        await bundleFacets({"plugin": {"id": "p"}, "entries": {"": "x.py"}, "outdir": "out"})
    write(directory / "broken.py", "def (:\n")
    with pytest.raises(Exception, match="Could not bundle facet entry broken"):
        await bundleFacets(
            {"plugin": {"id": "p"}, "entries": {"broken": str(directory / "broken.py")}, "outdir": str(directory / "out"), "workingDirectory": str(directory)}
        )
    with pytest.raises(Exception, match="outside the working directory"):
        await bundleFacets(
            {"plugin": {"id": "p"}, "entries": {"outside": str(directory / "broken.py")}, "outdir": str(directory / "out"), "workingDirectory": str(directory / "src")}
        )


async def test_rejects_invalid_manifests_and_artifacts(directory: Path):
    write(directory / "entry.py", "from types import SimpleNamespace\ndefault = SimpleNamespace(id='a', setup=lambda e: None)\n")
    result = await bundleFacets(
        {"plugin": {"id": "p", "version": "1"}, "entries": {"worker": str(directory / "entry.py")}, "outdir": str(directory / "out"), "workingDirectory": str(directory)}
    )
    manifest = await readFacetBundleManifest(result["manifestPath"])
    manifestFile = Path(result["manifestPath"])
    original = manifestFile.read_text(encoding="utf-8")

    for find, replace, message in [
        ('"chord.facet-bundle"', '"other"', "Invalid facet bundle manifest format"),
        ('"formatVersion": 2', '"formatVersion": 3', "Unsupported facet bundle manifest version"),
        ('"module": "entry.py"', '"module": "../entry.py"', "invalid module path"),
        ('"module": "entry.py"', '"module": "/entry.py"', "invalid module path"),
        ('"module": "entry.py"', '"module": "entry.txt"', "invalid module path"),
    ]:
        assert find in original
        manifestFile.write_text(original.replace(find, replace, 1), encoding="utf-8", newline="")
        with pytest.raises(Exception, match=message):
            await readFacetBundleManifest(manifestFile)
    manifestFile.write_text("{}", encoding="utf-8")
    with pytest.raises(Exception, match="Invalid facet bundle manifest format"):
        await readFacetBundleManifest(manifestFile)
    manifestFile.write_text("not json", encoding="utf-8")
    with pytest.raises(Exception, match="Could not read facet bundle manifest"):
        await readFacetBundleManifest(manifestFile)
    manifestFile.write_text(original, encoding="utf-8", newline="")

    artifact = await readFacetBundleArtifact({"manifestPath": result["manifestPath"], "entry": "worker"})
    assert artifact["plugin"] == {"id": "p", "version": "1"}
    assert artifact["entryName"] == "worker"
    assert artifact["entry"] == manifest["entries"]["worker"]
    with pytest.raises(Exception, match="has no entry named missing"):
        await readFacetBundleArtifact({"manifestPath": result["manifestPath"], "entry": "missing"})
    with pytest.raises(Exception, match="Invalid facet bundle artifact"):
        createFacetBundleArtifactLoader({"artifact": {**artifact, "format": "other"}})
    with pytest.raises(Exception, match="integrity check failed"):
        createFacetBundleArtifactLoader({"artifact": {**artifact, "source": "AAAA"}})
    with pytest.raises(Exception, match="is not base64"):
        createFacetBundleArtifactLoader({"artifact": {**artifact, "source": "!!!"}})
    with pytest.raises(TypeError, match="entry name must not be empty"):
        createFacetBundleLoader({"manifestPath": result["manifestPath"], "entry": ""})


async def bundleOne(directory: Path, source: str, init: str | None = None, name: str = "entry") -> dict[str, Any]:
    write(directory / "src" / f"{name}.py", source)
    if init is not None:
        write(directory / "src" / "__init__.py", init)
    return await bundleFacets(
        {
            "plugin": {"id": "p"},
            "entries": {"worker": str(directory / "src" / f"{name}.py")},
            "outdir": str(directory / "out"),
            "workingDirectory": str(directory),
        }
    )


async def test_an_entry_that_a_package_initializer_imports_runs_once(directory: Path):
    result = await bundleOne(
        directory,
        "import builtins\n"
        "from types import SimpleNamespace\n"
        "builtins.CHORD_ENTRY_RUNS = getattr(builtins, 'CHORD_ENTRY_RUNS', 0) + 1\n"
        'default = SimpleNamespace(id="once", setup=lambda env: None)\n',
        init="from .entry import default\n",
    )
    import builtins

    builtins.CHORD_ENTRY_RUNS = 0  # type: ignore[attr-defined]
    loaded = await createFacetBundleLoader({"manifestPath": result["manifestPath"], "entry": "worker"}).load()
    try:
        assert builtins.CHORD_ENTRY_RUNS == 1  # type: ignore[attr-defined]
        package = next(module for name, module in sys.modules.items() if name.startswith("chord_facet_") and name.endswith(".src"))
        assert loaded.facets[0] is package.default
    finally:
        await loaded.dispose()
        del builtins.CHORD_ENTRY_RUNS  # type: ignore[attr-defined]


async def test_loading_leaves_no_importer_cache_entry_for_the_generation(directory: Path):
    write(directory / "src" / "helper.py", "def decorate(value):\n    return value\n")
    entryPath = directory / "src" / "entry.py"
    writeGeneration(entryPath, "A")
    result = await bundleFacets(
        {"plugin": {"id": "p"}, "entries": {"worker": str(entryPath)}, "outdir": str(directory / "out"), "workingDirectory": str(directory)}
    )
    before = set(sys.path_importer_cache)
    for _ in range(3):
        loaded = await createFacetBundleLoader({"manifestPath": result["manifestPath"], "entry": "worker"}).load()
        await loaded.dispose()
    assert set(sys.path_importer_cache) == before


async def test_an_artifact_is_materialized_under_the_temporary_directory_only(directory: Path, monkeypatch: pytest.MonkeyPatch):
    result = await bundleOne(
        directory,
        'from types import SimpleNamespace\ndefault = SimpleNamespace(id="a", setup=lambda env: None)\n',
    )
    artifact = await readFacetBundleArtifact({"manifestPath": result["manifestPath"], "entry": "worker"})
    parent = directory / "parent"
    # The OS temporary directory is not used at all.
    monkeypatch.setattr(tempfile, "tempdir", str(directory / "does-not-exist"))
    loaded = await createFacetBundleArtifactLoader({"artifact": artifact, "temporaryDirectory": str(parent)}).load()
    try:
        [name] = os.listdir(parent)
        assert name.startswith("chord-facet-")
    finally:
        await loaded.dispose()
    assert os.listdir(parent) == []


async def test_a_failed_artifact_cleanup_is_reported(directory: Path, monkeypatch: pytest.MonkeyPatch):
    result = await bundleOne(
        directory,
        'from types import SimpleNamespace\ndefault = SimpleNamespace(id="a", setup=lambda env: None)\n',
    )
    artifact = await readFacetBundleArtifact({"manifestPath": result["manifestPath"], "entry": "worker"})
    loaded = await createFacetBundleArtifactLoader({"artifact": artifact, "temporaryDirectory": str(directory / "parent")}).load()

    def refuse(path: str, *args: Any, **kwargs: Any) -> None:
        raise PermissionError("in use")

    monkeypatch.setattr(shutil, "rmtree", refuse)
    with pytest.raises(Exception, match="Failed to dispose facet bundle artifact|in use"):
        await loaded.dispose()
    monkeypatch.undo()


async def test_a_module_path_cannot_name_another_drive(directory: Path):
    result = await bundleOne(
        directory,
        'from types import SimpleNamespace\ndefault = SimpleNamespace(id="a", setup=lambda env: None)\n',
    )
    manifestFile = Path(result["manifestPath"])
    original = manifestFile.read_text(encoding="utf-8")
    for module in ("D:/evil/entry.py", "D:evil.py", "src/a:b.py"):
        manifestFile.write_text(original.replace('"src/entry.py"', f'"{module}"', 1), encoding="utf-8", newline="")
        with pytest.raises(Exception, match="invalid module path"):
            await readFacetBundleManifest(manifestFile)


async def test_a_manifest_follows_json_parse_and_the_version_rules(directory: Path):
    result = await bundleOne(
        directory,
        'from types import SimpleNamespace\ndefault = SimpleNamespace(id="a", setup=lambda env: None)\n',
    )
    manifestFile = Path(result["manifestPath"])
    original = manifestFile.read_text(encoding="utf-8")
    manifestFile.write_text(original.replace('"plugin": {', '"note": NaN, "plugin": {', 1), encoding="utf-8", newline="")
    with pytest.raises(Exception, match="Could not read facet bundle manifest"):
        await readFacetBundleManifest(manifestFile)
    manifestFile.write_text(original.replace('"plugin": {', '"plugin": {"version": null, ', 1), encoding="utf-8", newline="")
    with pytest.raises(Exception, match="invalid plugin version"):
        await readFacetBundleManifest(manifestFile)
    manifestFile.write_text(original.replace('"formatVersion": 2,', "", 1), encoding="utf-8", newline="")
    with pytest.raises(Exception, match="Unsupported facet bundle manifest version in .*: undefined"):
        await readFacetBundleManifest(manifestFile)


async def test_an_entry_failure_without_a_message_keeps_the_colon(directory: Path):
    result = await bundleOne(directory, "raise RuntimeError()\n")
    with pytest.raises(Exception) as caught:
        await createFacetBundleLoader({"manifestPath": result["manifestPath"], "entry": "worker"}).load()
    assert str(caught.value) == "Could not load facet bundle entry p/worker: "


async def test_the_package_walk_up_stops_at_the_working_directory_whatever_its_case(directory: Path):
    write(directory / "__init__.py", "from . import plugin\n")
    working = directory / "plugin"
    write(working / "src" / "entry.py", 'from types import SimpleNamespace\ndefault = SimpleNamespace(id="a", setup=lambda env: None)\n')
    result = await bundleFacets(
        {
            "plugin": {"id": "p"},
            "entries": {"worker": str(working / "src" / "entry.py")},
            "outdir": str(directory / "out"),
            # On Windows a different case names the same directory; elsewhere the paths are the same.
            "workingDirectory": str(working).swapcase() if os.name == "nt" else str(working),
        }
    )
    assert archiveNames(directory / "out", result["manifest"]["entries"]["worker"]["file"]) == ["src/entry.py"]


async def test_a_package_with_unreadable_or_malformed_metadata_gets_the_documented_errors(directory: Path):
    (directory / "pyproject.toml").write_bytes(b'[project]\nname = "caf\xe9"\nversion = "1"\n[tool.chord.facets]\nsession = "x.py"\n')
    with pytest.raises(Exception, match="Could not access configured facet entry session"):
        await bundleFacetPackage({"packagePath": str(directory), "outdir": str(directory / "out")})
    write(directory / "pyproject.toml", 'tool = "nope"\n[project]\nname = "p"\nversion = "1"\n')
    with pytest.raises(Exception, match="chord configuration must be a table"):
        await bundleFacetPackage({"packagePath": str(directory), "outdir": str(directory / "out")})
    write(directory / "pyproject.toml", '[project]\nname = "p"\nversion = "1"\n[tool.chord.facets]\nsession = "D:x.py"\n')
    with pytest.raises(Exception, match="(escapes the package directory|Could not access configured facet entry)"):
        await bundleFacetPackage({"packagePath": str(directory), "outdir": str(directory / "out")})
