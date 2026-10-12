"""pi chord's ``test/facets.test.ts``.

A facet is any object with an ``id`` and a ``setup(env)``; ``facet(id, setup)`` below builds one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from misaka.chord import (
    RemoteServiceProvider,
    createFacetHost,
    createRemoteServiceBinding,
    defineFacet,
    defineService,
)
from misaka.chord.context import BACKGROUND_CONTEXT
from tests.chord_helpers import createLoopbackServiceTransport, waitFor

Source = defineService("test.experimental.source")
Projection = defineService("test.experimental.projection")
KeyedValue = defineService("test.experimental.keyed-value")
Watched = defineService("test.experimental.watched")
HostValues = defineService("test.experimental.host-values", {"local": True})
LocalKeyedValue = defineService("test.experimental.local-keyed-value", {"local": True})
LeftValue = defineService("test.experimental.left-value")
RightValue = defineService("test.experimental.right-value")
CombinedValue = defineService("test.experimental.combined-value")


def facet(id_, setup):
    return defineFacet(SimpleNamespace(id=id_, setup=setup))


def constant(value):
    async def read(*_args):
        return value

    return read


def setValue(state, value):
    state.change(BACKGROUND_CONTEXT, lambda draft: draft.__setitem__("value", value))


async def test_discovers_setup_dependencies_before_connecting_stable_service_handles():
    trace = []
    handles = []

    def setupProjection(env):
        trace.append("setup projection")
        sourceHandle = env.use(Source)
        handles.append(sourceHandle)
        with pytest.raises(Exception, match="Facet projection service handles cannot be used while setting_up"):
            sourceHandle.read(BACKGROUND_CONTEXT)
        env.provide(Projection, {"read": lambda context: sourceHandle.read(context)})
        env.onActivate(lambda: trace.append("activate projection"))
        env.onDeactivate(lambda: trace.append("dispose projection"))

    def setupSource(env):
        trace.append("setup source")
        env.provide(Source, {"read": constant("value")})
        env.onActivate(lambda: trace.append("activate source"))
        env.onDeactivate(lambda: trace.append("dispose source"))

    host = await createFacetHost({"facets": [facet("projection", setupProjection), facet("source", setupSource)]})

    assert trace == ["setup projection", "setup source", "activate source", "activate projection"]
    assert await handles[0].read(BACKGROUND_CONTEXT) == "value"
    assert await host.services.use(Projection).read(BACKGROUND_CONTEXT) == "value"

    await host.dispose()
    assert trace[-2:] == ["dispose projection", "dispose source"]


async def test_connects_keyed_observations_only_when_the_observing_facet_activates():
    trace = []

    def setupObserver(env):
        async def observe(service, context):
            trace.append(f"observe {await service.read(context)}")

        env.observe(KeyedValue, observe)
        env.onActivate(lambda: trace.append("activate observer"))

    def setupProvider(env):
        values = env.provideMany(KeyedValue)

        def activate():
            trace.append("activate provider")
            values.spawn("one", {"read": constant("one")})

        env.onActivate(activate)

    host = await createFacetHost({"facets": [facet("observer", setupObserver), facet("provider", setupProvider)]})
    await waitFor(lambda: "observe one" in trace)

    assert trace == ["activate provider", "activate observer", "observe one"]
    remoteServices = createRemoteServiceBinding({
        "services": [KeyedValue],
        "transport": createLoopbackServiceTransport(host.services),
    })
    remoteValues = []

    async def observeRemote(service, context):
        remoteValues.append(await service.read(context))

    remoteServices.observe(KeyedValue, observeRemote)
    await remoteServices.ready(BACKGROUND_CONTEXT)
    await waitFor(lambda: remoteValues == ["one"])

    await remoteServices.dispose(BACKGROUND_CONTEXT)
    await host.dispose()


def keyedProvider(id_, value, mode="remote"):
    service = KeyedValue if mode == "remote" else LocalKeyedValue

    def setup(env):
        values = env.provideMany(service)
        if mode == "remote":
            env.onActivate(lambda: values.spawn("current", {"read": constant(value)}))
        else:
            env.onActivate(lambda: values.spawn("current", {"metadata": {"value": value}, "read": lambda: value}))

    return facet(id_, setup)


async def test_routes_remotely_exposable_keyed_services_through_the_host_provider():
    observed = []

    def setupConsumer(env):
        env.observe(KeyedValue, lambda service, context: observed.append({"service": service, "context": context}))

    host = await createFacetHost({
        "facets": [facet("remote-keyed-consumer", setupConsumer), keyedProvider("remote-keyed-provider", "A")]
    })
    await waitFor(lambda: len(observed) == 1)
    first = observed[0]
    assert await first["service"].read(first["context"]) == "A"

    await host.reload([keyedProvider("remote-keyed-provider", "B")])
    await waitFor(lambda: len(observed) == 2)
    assert first["context"].abortSignal.aborted is True
    with pytest.raises(Exception, match="observation is closed"):
        first["service"].read(first["context"])
    assert await observed[1]["service"].read(observed[1]["context"]) == "B"

    await host.dispose()
    assert observed[1]["context"].abortSignal.aborted is True


async def test_terminates_the_host_when_keyed_replacement_publication_fails():
    failure = RuntimeError("spawn publication failed")
    host = await createFacetHost({"facets": [keyedProvider("failing-keyed-provider", "A")]})

    def listener(update, _context):
        if update["type"] == "spawned":
            raise failure

    subscription = host.services.subscribe(KeyedValue.id, "keyed", listener)
    subscription.activate()

    with pytest.raises(ExceptionGroup, match="Facet reload failed after cutover"):
        await host.reload([keyedProvider("failing-keyed-provider", "B")])
    with pytest.raises(Exception, match="Facet host cannot reload while dead"):
        await host.reload([])
    with pytest.raises(Exception, match="Remote service provider is disposed"):
        host.services.use(KeyedValue)
    await host.dispose()


async def test_terminates_the_host_when_keyed_retirement_publication_fails():
    failure = RuntimeError("close publication failed")
    host = await createFacetHost({"facets": [keyedProvider("failing-keyed-retirement-provider", "A")]})

    def listener(update, _context):
        if update["type"] == "closed":
            raise failure

    subscription = host.services.subscribe(KeyedValue.id, "keyed", listener)
    subscription.activate()

    with pytest.raises(ExceptionGroup, match="Facet reload failed after cutover"):
        await host.reload([keyedProvider("failing-keyed-retirement-provider", "B")])
    with pytest.raises(Exception, match="Facet host cannot reload while dead"):
        await host.reload([])
    await host.dispose()


async def test_keeps_unrestricted_local_keyed_services_process_local_across_provider_reloads():
    observed = []

    def setupConsumer(env):
        env.observe(LocalKeyedValue, lambda service, context: observed.append({"service": service, "context": context}))

    host = await createFacetHost({
        "facets": [facet("local-keyed-consumer", setupConsumer), keyedProvider("local-keyed-provider", "A", "local")]
    })
    await waitFor(lambda: len(observed) == 1)
    first = observed[0]
    assert first["service"].read() == "A"
    assert first["service"].metadata.get("value") == "A"
    assert {"serviceId": LocalKeyedValue.id, "mode": "keyed"} not in host.services.catalogue
    with pytest.raises(Exception, match="process-local"):
        host.services.use(LocalKeyedValue)

    await host.reload([keyedProvider("local-keyed-provider", "B", "local")])
    await waitFor(lambda: len(observed) == 2)
    assert first["context"].abortSignal.aborted is True
    with pytest.raises(Exception, match=f"Keyed service {LocalKeyedValue.id} observation is closed"):
        first["service"].read()
    assert observed[1]["service"].read() == "B"
    assert observed[1]["service"].metadata.get("value") == "B"

    await host.dispose()
    assert observed[1]["context"].abortSignal.aborted is True


async def test_keeps_remotely_exposable_local_state_replicas_stable_across_provider_reloads():
    sources = []
    revisions = []
    watched = []

    def setupConsumer(env):
        watched.append(env.use(Watched))

        def activate():
            env.own(watched[0].state.subscribe(lambda value, _context, _delivery: revisions.append(value["value"])))

        env.onActivate(activate)

    def provider(value):
        def setup(env):
            state = env.replicatedState({"value": value})
            sources.append(state)
            env.provide(Watched, {"state": state})

        return facet("state-provider", setup)

    host = await createFacetHost({"facets": [facet("state-consumer", setupConsumer), provider(1)]})
    retainedState = watched[0].state
    assert retainedState.value == {"value": 1}
    assert revisions == [1]

    await host.reload([provider(2)])
    assert watched[0].state is retainedState
    assert retainedState.value == {"value": 2}
    assert revisions == [1, 2]
    setValue(sources[0], 3)
    assert retainedState.value == {"value": 2}
    assert revisions == [1, 2]

    await host.dispose()
    with pytest.raises(Exception, match="Facet state-consumer service handles cannot be used while dead"):
        retainedState.value  # noqa: B018


async def test_scopes_singleton_service_views_to_each_facet_lifecycle():
    consumerHandles = []
    cleanupValues = []
    peerHandle = []

    def consumer(generation):
        def setup(env):
            source = env.use(Source)
            assert env.use(Source) is source
            consumerHandles.append(source)
            with pytest.raises(Exception, match="Facet scoped-consumer service handles cannot be used while setting_up"):
                source.read(BACKGROUND_CONTEXT)

            async def cleanup():
                cleanupValues.append(f"{generation}:{await source.read(BACKGROUND_CONTEXT)}")

            env.onDeactivate(cleanup)

        return facet("scoped-consumer", setup)

    peer = facet("peer-consumer", lambda env: peerHandle.append(env.use(Source)))
    provider = facet("scoped-provider", lambda env: env.provide(Source, {"read": constant("value")}))
    host = await createFacetHost({"facets": [consumer("A"), peer, provider]})
    assert consumerHandles[0] is not peerHandle[0]
    retainedOldRead = consumerHandles[0].read

    await host.reload([consumer("B")])
    assert cleanupValues == ["A:value"]
    assert consumerHandles[1] is not consumerHandles[0]
    with pytest.raises(Exception, match="Facet scoped-consumer service handles cannot be used while dead"):
        consumerHandles[0].read(BACKGROUND_CONTEXT)
    with pytest.raises(Exception, match="Facet scoped-consumer service handles cannot be used while dead"):
        retainedOldRead(BACKGROUND_CONTEXT)
    assert await consumerHandles[1].read(BACKGROUND_CONTEXT) == "value"
    assert await peerHandle[0].read(BACKGROUND_CONTEXT) == "value"

    await host.dispose()
    assert cleanupValues == ["A:value", "B:value"]
    with pytest.raises(Exception, match="Facet scoped-consumer service handles cannot be used while dead"):
        consumerHandles[1].read(BACKGROUND_CONTEXT)
    with pytest.raises(Exception, match="Facet peer-consumer service handles cannot be used while dead"):
        peerHandle[0].read(BACKGROUND_CONTEXT)


async def test_owns_resources_registered_during_activation():
    state = []
    deliveries = [0]

    def setupConsumer(env):
        watched = env.use(Watched)

        def activate():
            env.own(watched.state.subscribe(lambda *_args: deliveries.__setitem__(0, deliveries[0] + 1)))

        env.onActivate(activate)

    def setupProvider(env):
        state.append(env.replicatedState({"value": 0}))
        env.provide(Watched, {"state": state[0]})

    host = await createFacetHost({"facets": [facet("consumer", setupConsumer), facet("provider", setupProvider)]})
    assert deliveries[0] == 1
    setValue(state[0], 1)
    assert deliveries[0] == 2

    await host.dispose()
    setValue(state[0], 2)
    assert deliveries[0] == 2


async def test_provides_arbitrary_host_services_through_the_facet_graph():
    values = {"name": "session", "use": "host value"}

    def setupConsumer(env):
        hostValues = env.use(HostValues)
        with pytest.raises(Exception, match="Facet host-service-consumer service handles cannot be used while setting_up"):
            hostValues.use  # noqa: B018

        def activate():
            assert hostValues is not values
            assert hostValues.name == "session"
            assert hostValues.use == "host value"

        env.onActivate(activate)

    host = await createFacetHost({
        "facets": [
            facet("host-service-consumer", setupConsumer),
            facet("host-service-provider", lambda env: env.provide(HostValues, values)),
        ]
    })
    with pytest.raises(Exception, match="Service test.experimental.host-values is process-local"):
        host.services.use(HostValues)
    await host.dispose()


class ConnectedSource:
    """A remote service source over one provider, opened bound only at host activation."""

    acceptsUnavailableServices = False

    def __init__(self, namespace, provider):
        self.namespace = namespace
        self.provider = provider

    async def catalogue(self, context):
        return self.provider.catalogue

    def open(self, options):
        return self

    def use(self, service):
        return self.namespace.use(service)

    def observe(self, service, handler):
        return self.namespace.observe(service, handler)

    async def ready(self, context):
        await self.namespace.rebind(True, context)
        await self.namespace.ready(context)

    async def dispose(self, context):
        await self.namespace.dispose(context)


async def test_combines_connected_services_and_facet_provided_services_in_one_host():
    leftProvider = RemoteServiceProvider([LeftValue])
    leftProvider.provide(LeftValue, {"read": constant("left")})
    rightProvider = RemoteServiceProvider([RightValue])
    rightProvider.provide(RightValue, {"read": constant("right")})
    leftNamespace = createRemoteServiceBinding({
        "services": [LeftValue],
        "transport": createLoopbackServiceTransport(leftProvider),
        "bound": False,
    })
    rightNamespace = createRemoteServiceBinding({
        "services": [RightValue],
        "transport": createLoopbackServiceTransport(rightProvider),
        "bound": False,
    })
    serviceSources = [ConnectedSource(leftNamespace, leftProvider), ConnectedSource(rightNamespace, rightProvider)]

    def setup(env):
        left = env.use(LeftValue)
        right = env.use(RightValue)

        async def read(context):
            return f"{await left.read(context)} {await right.read(context)}"

        env.provide(CombinedValue, {"read": read})

    host = await createFacetHost({"facets": [facet("combined", setup)], "serviceSources": serviceSources})
    assert await host.services.use(CombinedValue).read(BACKGROUND_CONTEXT) == "left right"

    await host.dispose()
    await leftNamespace.dispose(BACKGROUND_CONTEXT)
    await rightNamespace.dispose(BACKGROUND_CONTEXT)
    leftProvider.dispose()
    rightProvider.dispose()


async def test_rejects_a_service_offered_by_multiple_sources():
    class Duplicate:
        acceptsUnavailableServices = False

        async def catalogue(self, context):
            return [{"serviceId": LeftValue.id, "mode": "singleton"}]

        def open(self, options):
            raise RuntimeError("Ambiguous sources must not open")

    duplicate = Duplicate()
    consumer = facet("duplicate-consumer", lambda env: env.use(LeftValue))
    with pytest.raises(Exception, match=f"Facet host service {LeftValue.id} is offered by more than one source"):
        await createFacetHost({"facets": [consumer], "serviceSources": [duplicate, duplicate]})


async def test_reopens_source_bindings_from_changed_catalogues_for_a_replacement_generation():
    leftProvider = RemoteServiceProvider([LeftValue])
    leftProvider.provide(LeftValue, {"read": constant("left")})
    rightProvider = RemoteServiceProvider([RightValue])
    rightProvider.provide(RightValue, {"read": constant("right")})
    state = {"currentProvider": leftProvider, "opened": 0, "disposed": 0}

    class Opened:
        def __init__(self, namespace):
            self.namespace = namespace

        def use(self, service):
            return self.namespace.use(service)

        def observe(self, service, handler):
            return self.namespace.observe(service, handler)

        async def ready(self, context):
            await self.namespace.rebind(True, context)
            await self.namespace.ready(context)

        async def dispose(self, context):
            state["disposed"] += 1
            await self.namespace.dispose(context)

    class ChangingSource:
        acceptsUnavailableServices = False

        async def catalogue(self, context):
            return state["currentProvider"].catalogue

        def open(self, options):
            state["opened"] += 1
            return Opened(
                createRemoteServiceBinding({
                    "services": options["services"],
                    "transport": createLoopbackServiceTransport(state["currentProvider"]),
                    "bound": False,
                    "assertAccess": options["assertAccess"],
                    "onError": options["onError"],
                })
            )

    source = ChangingSource()
    values = []

    def setupLeft(env):
        left = env.use(LeftValue)

        async def activate():
            values.append(await left.read(BACKGROUND_CONTEXT))

        env.onActivate(activate)

    first = await createFacetHost({"facets": [facet("left-consumer", setupLeft)], "serviceSources": [source]})
    await first.dispose()

    state["currentProvider"] = rightProvider

    def setupRight(env):
        right = env.use(RightValue)

        async def activate():
            values.append(await right.read(BACKGROUND_CONTEXT))

        env.onActivate(activate)

    second = await createFacetHost({"facets": [facet("right-consumer", setupRight)], "serviceSources": [source]})
    await second.dispose()

    assert values == ["left", "right"]
    assert state["opened"] == 2
    assert state["disposed"] == 2
    leftProvider.dispose()
    rightProvider.dispose()


async def test_rejects_missing_dependencies_cycles_and_asynchronous_setup():
    activated = [False]

    def setupMissing(env):
        env.use(Source)
        env.onActivate(lambda: activated.__setitem__(0, True))

    with pytest.raises(
        Exception, match="Facet missing requires local/test.experimental.source/singleton, but no facet provides it"
    ):
        await createFacetHost({"facets": [facet("missing", setupMissing)]})
    assert activated[0] is False

    def setupFirst(env):
        env.use(Projection)
        env.provide(Source, {"read": constant("first")})

    def setupSecond(env):
        env.use(Source)
        env.provide(Projection, {"read": constant("second")})

    with pytest.raises(Exception, match="Facet dependency cycle: first, second"):
        await createFacetHost({"facets": [facet("first", setupFirst), facet("second", setupSecond)]})

    async def setupAsynchronous(env):
        return None

    with pytest.raises(Exception, match="Facet asynchronous setup must be synchronous"):
        await createFacetHost({"facets": [facet("asynchronous", setupAsynchronous)]})
