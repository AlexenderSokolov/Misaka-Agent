"""pi chord's ``test/facet-loader.test.ts``."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from misaka.chord import (
    combineFacetLoaders,
    createFacetHost,
    createRemoteServiceBinding,
    createStaticFacetLoader,
    defineFacet,
    defineService,
)
from misaka.chord.context import BACKGROUND_CONTEXT
from tests.chord_helpers import createLoopbackServiceTransport

LocalGenerationValue = defineService("test.experimental.local-generation-value", {"local": True})
RemoteGenerationValue = defineService("test.experimental.remote-generation-value")


def facet(id_: str, setup: Any = None) -> Any:
    return defineFacet(SimpleNamespace(id=id_, setup=setup or (lambda env: None)))


firstFacet = facet("first")
secondFacet = facet("second")


def reader(value: str) -> Any:
    async def read(*_: Any) -> str:
        return value

    return read


async def test_combines_loaded_facets_in_loader_order_and_disposes_generations_in_reverse():
    trace: list[str] = []

    def loader(name: str, loadedFacet: Any) -> Any:
        class Loaded:
            facets = (loadedFacet,)

            async def dispose(self) -> None:
                trace.append(f"dispose {name}")

        class Loader:
            async def load(self) -> Any:
                trace.append(f"load {name}")
                return Loaded()

        return Loader()

    loaded = await combineFacetLoaders([loader("first", firstFacet), loader("second", secondFacet)]).load()
    assert list(loaded.facets) == [firstFacet, secondFacet]
    await loaded.dispose()
    await loaded.dispose()
    assert trace == ["load first", "load second", "dispose second", "dispose first"]


async def test_cleans_up_loaded_facets_when_a_later_loader_fails():
    disposed: list[int] = []
    failure = RuntimeError("load failed")

    class FirstLoaded:
        facets = (firstFacet,)

        async def dispose(self) -> None:
            disposed.append(1)

    class First:
        async def load(self) -> Any:
            return FirstLoaded()

    class Second:
        async def load(self) -> Any:
            raise failure

    with pytest.raises(RuntimeError) as raised:
        await combineFacetLoaders([First(), Second()]).load()
    assert raised.value is failure
    assert len(disposed) == 1


async def test_keeps_local_and_rpc_service_handles_stable_when_their_provider_facet_reloads():
    trace: list[str] = []
    handles: dict[str, Any] = {}
    state = {"generation": 0}
    replacementStarted: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    replacementCanContinue: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    def setupConsumer(env: Any) -> None:
        trace.append("setup consumer")
        handles["local"] = env.use(LocalGenerationValue)

        async def activate() -> None:
            trace.append(f"activate consumer:{await handles['local'].read(BACKGROUND_CONTEXT)}")

        env.onActivate(activate)
        env.onDeactivate(lambda: trace.append("deactivate consumer"))

    consumer = facet("consumer", setupConsumer)

    class Loader:
        async def load(self) -> Any:
            name = "A" if state["generation"] == 0 else "B"
            state["generation"] += 1
            trace.append(f"load {name}")

            def setupProvider(env: Any) -> None:
                trace.append(f"setup provider {name}")
                implementation = {"read": reader(name)}
                env.provide(LocalGenerationValue, implementation)
                env.provide(RemoteGenerationValue, implementation)

                async def activate() -> None:
                    trace.append(f"activate provider {name}")
                    if name == "B":
                        replacementStarted.set_result(None)
                        await replacementCanContinue

                env.onActivate(activate)
                env.onDeactivate(lambda: trace.append(f"deactivate provider {name}"))

            class Loaded:
                facets = (facet("provider", setupProvider),)

                async def dispose(self) -> None:
                    trace.append(f"unload {name}")

            return Loaded()

    loader = Loader()
    loadedA = await loader.load()
    host = await createFacetHost({"facets": [consumer, *loadedA.facets]})
    originalLocalHandle = handles["local"]
    localRead = originalLocalHandle.read
    remoteServices = createRemoteServiceBinding(
        {"services": [RemoteGenerationValue], "transport": createLoopbackServiceTransport(host.services)}
    )
    originalRemoteHandle = remoteServices.use(RemoteGenerationValue)
    remoteRead = originalRemoteHandle.read
    await remoteServices.ready(BACKGROUND_CONTEXT)
    assert await localRead(BACKGROUND_CONTEXT) == "A"
    assert await remoteRead(BACKGROUND_CONTEXT) == "A"

    with pytest.raises(Exception, match="Reloaded facet provider must preserve its service requirements and provisions"):
        await host.reload(
            [
                facet(
                    "provider",
                    lambda env: env.provide(LocalGenerationValue, {"read": reader("invalid")}),
                )
            ]
        )
    assert await localRead(BACKGROUND_CONTEXT) == "A"
    assert await remoteRead(BACKGROUND_CONTEXT) == "A"

    loadedB = await loader.load()
    reload = asyncio.ensure_future(host.reload(loadedB.facets))
    await replacementStarted
    assert await localRead(BACKGROUND_CONTEXT) == "A"
    assert await remoteRead(BACKGROUND_CONTEXT) == "A"
    replacementCanContinue.set_result(None)
    await reload
    await loadedA.dispose()

    assert handles["local"] is originalLocalHandle
    assert remoteServices.use(RemoteGenerationValue) is originalRemoteHandle
    assert await localRead(BACKGROUND_CONTEXT) == "B"
    assert await remoteRead(BACKGROUND_CONTEXT) == "B"

    await remoteServices.dispose(BACKGROUND_CONTEXT)
    await host.dispose()
    await loadedB.dispose()
    assert trace == [
        "load A",
        "setup consumer",
        "setup provider A",
        "activate provider A",
        "activate consumer:A",
        "load B",
        "setup provider B",
        "activate provider B",
        "deactivate provider A",
        "unload A",
        "deactivate consumer",
        "deactivate provider B",
        "unload B",
    ]


async def test_rejects_remote_singleton_member_shape_changes_before_reload_cutover():
    state: dict[str, Any] = {"retained": None, "providerDisposed": False}

    consumer = facet("shape-consumer", lambda env: state.__setitem__("retained", env.use(RemoteGenerationValue)))

    def setupProvider(env: Any) -> None:
        env.provide(RemoteGenerationValue, {"read": reader("A")})
        env.onDeactivate(lambda: state.__setitem__("providerDisposed", True))

    host = await createFacetHost({"facets": [consumer, facet("shape-provider", setupProvider)]})

    with pytest.raises(Exception, match="replacement must preserve its member shape"):
        await host.reload(
            [facet("shape-provider", lambda env: env.provide(RemoteGenerationValue, {"renamed": reader("B")}))]
        )
    assert state["providerDisposed"] is False
    assert await state["retained"].read(BACKGROUND_CONTEXT) == "A"

    await host.dispose()
    assert state["providerDisposed"] is True


async def test_terminates_the_host_when_old_cleanup_fails_after_cutover():
    cleanupFailure = RuntimeError("cleanup failed")

    def provider(name: str, failCleanup: bool) -> Any:
        def setup(env: Any) -> None:
            env.provide(RemoteGenerationValue, {"read": reader(name)})

            def deactivate() -> None:
                if failCleanup:
                    raise cleanupFailure

            env.onDeactivate(deactivate)

        return facet("cleanup-provider", setup)

    host = await createFacetHost({"facets": [provider("A", True)]})

    with pytest.raises(Exception, match="Facet reload failed after cutover"):
        await host.reload([provider("B", False)])
    with pytest.raises(Exception, match="Facet host cannot reload while dead"):
        await host.reload([])
    await host.dispose()


async def test_cleans_failed_candidate_activation_in_reverse_dependency_order():
    failure = RuntimeError("consumer activation failed")
    trace: list[str] = []

    def provider(name: str) -> Any:
        def setup(env: Any) -> None:
            env.provide(RemoteGenerationValue, {"read": reader(name)})
            env.onActivate(lambda: trace.append(f"activate provider {name}"))
            env.onDeactivate(lambda: trace.append(f"deactivate provider {name}"))

        return facet("ordered-provider", setup)

    def consumer(name: str, fail: bool) -> Any:
        def setup(env: Any) -> None:
            env.use(RemoteGenerationValue)

            def activate() -> None:
                trace.append(f"activate consumer {name}")
                if fail:
                    raise failure

            env.onActivate(activate)
            env.onDeactivate(lambda: trace.append(f"deactivate consumer {name}"))

        return facet("ordered-consumer", setup)

    host = await createFacetHost({"facets": [consumer("A", False), provider("A")]})
    trace.clear()

    with pytest.raises(RuntimeError) as raised:
        await host.reload([consumer("B", True), provider("B")])
    assert raised.value is failure
    assert trace == [
        "activate provider B",
        "activate consumer B",
        "deactivate consumer B",
        "deactivate provider B",
    ]
    await host.dispose()


async def test_terminates_the_host_when_replacement_publication_fails_after_cutover():
    publicationFailure = RuntimeError("publication failed")

    def provider(name: str) -> Any:
        return facet("publication-provider", lambda env: env.provide(RemoteGenerationValue, {"read": reader(name)}))

    host = await createFacetHost({"facets": [provider("A")]})

    def fail(*_: Any) -> None:
        raise publicationFailure

    subscription = host.services.subscribe(RemoteGenerationValue.id, "singleton", fail)
    subscription.activate()

    with pytest.raises(Exception, match="Facet reload failed after cutover"):
        await host.reload([provider("B")])
    with pytest.raises(Exception, match="Facet host cannot reload while dead"):
        await host.reload([])
    await host.dispose()


async def test_keeps_the_old_generation_active_when_replacement_activation_fails_before_cutover():
    failure = RuntimeError("replacement activation failed")
    trace: list[str] = []
    state: dict[str, Any] = {}

    def setupConsumer(env: Any) -> None:
        state["retained"] = env.use(RemoteGenerationValue)
        env.onDeactivate(lambda: trace.append("deactivate consumer"))

    consumer = facet("terminal-consumer", setupConsumer)

    def provider(name: str, fail: bool) -> Any:
        def setup(env: Any) -> None:
            env.provide(RemoteGenerationValue, {"read": reader(name)})

            def activate() -> None:
                trace.append(f"activate {name}")
                if fail:
                    raise failure

            env.onActivate(activate)
            env.onDeactivate(lambda: trace.append(f"deactivate {name}"))

        return facet("terminal-provider", setup)

    host = await createFacetHost({"facets": [consumer, provider("A", False)]})
    assert await state["retained"].read(BACKGROUND_CONTEXT) == "A"

    with pytest.raises(RuntimeError) as raised:
        await host.reload([provider("B", True)])
    assert raised.value is failure
    assert trace == ["activate A", "activate B", "deactivate B"]
    assert await state["retained"].read(BACKGROUND_CONTEXT) == "A"
    await host.reload([])
    await host.dispose()
    assert trace == ["activate A", "activate B", "deactivate B", "deactivate consumer", "deactivate A"]


async def test_creates_a_reusable_static_loader():
    loader = createStaticFacetLoader([firstFacet])
    first = await loader.load()
    second = await loader.load()
    assert list(first.facets) == [firstFacet]
    assert list(second.facets) == [firstFacet]
    await asyncio.gather(first.dispose(), second.dispose())
