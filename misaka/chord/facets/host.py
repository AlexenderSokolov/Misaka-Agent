"""The facet host kernel: facet lifecycles, service wiring, and atomic reload.

Translated from pi's ``packages/chord/src/facets/host.ts``.

MISAKA forks, forced by the language: a callback or disposal may be a plain function or a coroutine function
(an awaitable result is awaited); ``AggregateError`` is an ``ExceptionGroup``; a service is a ``Service`` (or
any object with ``id`` and ``local``), an implementation a mapping or an object (see ``services/provider.py``);
a facet's ``setup`` is a plain function and answering an awaitable is an error, as answering a ``PromiseLike``
is upstream.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.chord.services.consumer import RemoteServiceBindingImpl
from misaka.chord.services.handle import ServiceSlot
from misaka.chord.services.instances import InstanceDirectory
from misaka.chord.services.loopback import createLoopbackServiceTransport
from misaka.chord.services.provider import (
    RemoteServiceProvider,
    validateRemoteServiceImplementation,
)
from misaka.chord.services.state import MutableReplicatedStateImpl
from misaka.chord.types import (
    Context,
    Facet,
    FacetOptions,
    RemoteServices,
    RemoteServiceSource,
)
from misaka.utils.js import awaitValue, eager

type Disposal = Callable[[], Any]


def _serviceId(service: Any) -> str:
    return service["id"] if isinstance(service, Mapping) else service.id


def _isLocal(service: Any) -> bool:
    local = service.get("local") if isinstance(service, Mapping) else getattr(service, "local", None)
    return local is True


def _isPlainObject(value: Any) -> bool:
    # ``typeof`` calls a function (and a class) a "function", not an "object".
    if inspect.isroutine(value) or inspect.isclass(value) or isinstance(value, functools.partial):
        return False
    return not (value is None or isinstance(value, str | bytes | int | float | bool | list | tuple | set))


class FacetServiceReference:
    __slots__ = ("mode", "service", "serviceId")

    def __init__(self, serviceId: str, service: Any, mode: str) -> None:
        self.serviceId = serviceId
        self.service = service
        self.mode = mode


class ExternalService:
    __slots__ = ("mode", "service", "source")

    def __init__(self, service: Any, mode: str, source: RemoteServiceSource) -> None:
        self.service = service
        self.mode = mode
        self.source = source


class FacetLifecycle:
    def __init__(self, id: str) -> None:
        self.id = id
        self._effects: list[Disposal] = []
        self._observations: list[Callable[[], Disposal]] = []
        self._activate: list[Callable[[], Any]] = []
        self._state = "setting_up"
        self._serviceAccess = False

    def assertSettingUp(self, operation: str) -> None:
        if self._state != "setting_up":
            raise RuntimeError(f"Facet {self.id} can {operation} only during setup")

    def assertRunning(self, operation: str) -> None:
        if self._state != "setting_up" and self._state != "active":
            raise RuntimeError(f"Facet {self.id} cannot {operation} while {self._state}")

    def assertActive(self, operation: str) -> None:
        if self._state != "active":
            raise RuntimeError(f"Facet {self.id} can {operation} only while active")

    def assertServiceAccess(self) -> None:
        if not self._serviceAccess:
            raise RuntimeError(f"Facet {self.id} service handles cannot be used while {self._state}")

    def revoke(self) -> None:
        self._serviceAccess = False

    def own(self, disposal: Disposal) -> None:
        self.assertRunning("own resources")
        self._effects.append(disposal)

    def observe(self, start: Callable[[], Disposal]) -> None:
        self.assertSettingUp("observe services")
        self._observations.append(start)

    def onActivate(self, callback: Callable[[], Any]) -> None:
        self.assertSettingUp("register activation callbacks")
        self._activate.append(callback)

    def prepared(self) -> None:
        self.assertSettingUp("finish setup")
        self._state = "prepared"

    async def activate(self) -> None:
        if self._state != "prepared":
            raise RuntimeError(f"Facet {self.id} is not prepared")
        self._state = "active"
        self._serviceAccess = True
        for start in self._observations:
            self._effects.append(start())
        for callback in self._activate:
            await awaitValue(callback())

    async def dispose(self) -> None:
        if self._state == "dead":
            return
        self._state = "disposing"
        errors: list[Exception] = []
        effects, self._effects = self._effects, []
        for effect in reversed(effects):
            try:
                await awaitValue(effect())
            except Exception as error:  # noqa: BLE001 - collected
                errors.append(error)
        self._observations.clear()
        self._activate.clear()
        self._serviceAccess = False
        self._state = "dead"
        if len(errors) == 1:
            raise errors[0]
        if len(errors) > 1:
            raise ExceptionGroup(f"Failed to dispose facet {self.id}", errors)


class FacetProvision:
    """One thing a facet provides: a singleton implementation or the spawning of keyed instances."""

    __slots__ = ("connectLocal", "connectRemote", "implementation", "install", "kind", "replace", "service", "validateReplacement")

    def __init__(self, kind: str, service: Any) -> None:
        self.kind = kind
        self.service = service
        self.implementation: Any = None
        self.install: Callable[[RemoteServiceProvider], None] = _unsupported
        self.validateReplacement: Callable[[RemoteServiceProvider], None] = _unsupported
        self.replace: Callable[[RemoteServiceProvider], None] = _unsupported
        self.connectLocal: Callable[[LocalKeyedServiceRegistry], None] = _unsupported
        self.connectRemote: Callable[[RemoteServiceProvider], None] = _unsupported


def _unsupported(_target: Any) -> None:
    raise RuntimeError("Unsupported facet provision operation")


class _LocalInstance:
    __slots__ = ("generation", "key", "service")

    def __init__(self, key: str, generation: int, service: Any) -> None:
        self.key = key
        self.generation = generation
        self.service = service

    def deactivate(self) -> None:
        pass


class LocalKeyedServiceRegistry:
    def __init__(self, services: Sequence[Any], onError: Callable[[Exception], None]) -> None:
        ids = [_serviceId(service) for service in services]
        if len(set(ids)) != len(ids):
            raise TypeError("Local keyed service registry has duplicate IDs")
        self._registrations: dict[str, tuple[dict[str, int], InstanceDirectory[_LocalInstance]]] = {}
        self._disposed = False
        for serviceId in ids:
            self._registrations[serviceId] = ({}, InstanceDirectory(ready=True, onError=onError))

    def spawn(self, service: Any, key: str, implementation: Any) -> Callable[[], None]:
        self._assertActive()
        if len(key) == 0:
            raise TypeError("Local service instance key must not be empty")
        if not _isPlainObject(implementation):
            raise TypeError(f"Local service {_serviceId(service)} implementation must be an object")
        generations, directory = self._registration(_serviceId(service))
        if directory.get(key) is not None:
            raise RuntimeError(f"Local service {_serviceId(service)} already has a live instance with key {key}")
        generation = generations.get(key, 0) + 1
        generations[key] = generation
        instance = _LocalInstance(key, generation, implementation)
        directory.insert(instance)
        closed = False

        def close() -> None:
            nonlocal closed
            if closed:
                return
            closed = True
            directory.remove(instance)

        return close

    def observe(self, service: Any, handler: Callable[[Any, Context], Any]) -> Callable[[], None]:
        self._assertActive()
        return self._registration(_serviceId(service))[1].observe(handler)

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        for _generations, directory in self._registrations.values():
            directory.dispose()
        self._registrations.clear()

    def _registration(self, serviceId: str) -> tuple[dict[str, int], InstanceDirectory[_LocalInstance]]:
        registration = self._registrations.get(serviceId)
        if registration is None:
            raise RuntimeError(f"Local keyed service {serviceId} is not registered")
        return registration

    def _assertActive(self) -> None:
        if self._disposed:
            raise RuntimeError("Local keyed service registry is disposed")


class HostServiceSlots:
    def __init__(self) -> None:
        self._singletons: dict[str, ServiceSlot] = {}
        self._keyedSources: dict[str, Any] = {}

    def getSingleton(self, service: Any, assertAccess: Callable[[], None]) -> Any:
        slot = self._singletons.get(_serviceId(service))
        if slot is None:
            slot = ServiceSlot(_serviceId(service), not _isLocal(service))
            self._singletons[_serviceId(service)] = slot
        return slot.view(assertAccess)

    def hasSingleton(self, serviceId: str) -> bool:
        return serviceId in self._singletons

    def observe(
        self, service: Any, assertAccess: Callable[[], None], handler: Callable[[Any, Context], Any]
    ) -> Disposal:
        source = self._keyedSources.get(_serviceId(service))
        if source is None:
            raise RuntimeError(f"Service {_serviceId(service)} is disconnected")
        stopped = False

        def observeTarget(target: Any, context: Context) -> Any:
            slot = ServiceSlot(_serviceId(service), not _isLocal(service))
            slot.bind(target)

            def guard() -> None:
                assertAccess()
                signal = context.abortSignal
                if stopped or (signal is not None and signal.aborted):
                    raise RuntimeError(f"Keyed service {_serviceId(service)} observation is closed")

            return handler(slot.view(guard), context)

        stop = source.observe(service, observeTarget)

        def stopObserving() -> None:
            nonlocal stopped
            if stopped:
                return
            stopped = True
            stop()

        return stopObserving

    def bindSingleton(self, serviceId: str, target: Any) -> None:
        slot = self._singletons.get(serviceId)
        if slot is not None:
            slot.bind(target)

    def bindKeyed(self, serviceId: str, services: Any) -> None:
        self._keyedSources[serviceId] = services

    def dispose(self) -> None:
        for slot in self._singletons.values():
            slot.unbind()
        self._singletons.clear()
        self._keyedSources.clear()


class _StagedServiceInstance:
    __slots__ = ("implementation", "key", "release")

    def __init__(self, key: str, implementation: Any) -> None:
        self.key = key
        self.implementation = implementation
        self.release: Callable[[], None] | None = None


class StagedServiceSpawner:
    def __init__(self, lifecycle: FacetLifecycle, validate: Callable[[str, Any], None]) -> None:
        self._lifecycle = lifecycle
        self._validate = validate
        self._instances: dict[str, _StagedServiceInstance] = {}
        self._installer: Callable[[str, Any], Callable[[], None]] | None = None

    def connect(self, installer: Callable[[str, Any], Callable[[], None]]) -> None:
        if self._installer is not None:
            raise RuntimeError("Facet service provider is already connected")
        self._installer = installer
        for instance in list(self._instances.values()):
            instance.release = installer(instance.key, instance.implementation)

    def spawn(self, key: str, implementation: Any) -> Callable[[], None]:
        self._lifecycle.assertActive("spawn service instances")
        self._validate(key, implementation)
        if key in self._instances:
            raise RuntimeError(f"Facet service already has a live instance with key {key}")
        instance = _StagedServiceInstance(key, implementation)
        self._instances[key] = instance
        if self._installer is not None:
            instance.release = self._installer(key, implementation)

        def close() -> None:
            if self._instances.get(key) is not instance:
                return
            del self._instances[key]
            if instance.release is not None:
                instance.release()

        self._lifecycle.own(close)
        return close


class FacetRuntime:
    """The record of one facet generation: what it requires and provides, and its lifecycle.

    MISAKA fork: upstream's `interface FacetRuntime`, built by `FacetKernel.#createFacetRuntime`; here the
    constructor builds it."""

    def __init__(self, facetId: str) -> None:
        self.facetId = facetId
        self.requires: list[FacetServiceReference] = []
        self.provides: list[FacetServiceReference] = []
        self.lifecycle = FacetLifecycle(facetId)
        self.provisions: list[FacetProvision] = []
        self.singletonViews: dict[str, Any] = {}


class _Environment:
    """The ``FacetEnvironment`` one facet's ``setup`` receives.

    MISAKA fork: upstream's `FacetKernel.#environment(runtime)` returns an object literal closing over the kernel;
    this class holds the kernel and the runtime instead."""

    def __init__(self, kernel: FacetKernel, runtime: FacetRuntime) -> None:
        self._kernel = kernel
        self._runtime = runtime

    def provide(self, service: Any, implementation: Any) -> None:
        runtime = self._runtime
        runtime.lifecycle.assertSettingUp("provide services")
        if not _isPlainObject(implementation):
            raise TypeError(f"Service {_serviceId(service)} implementation must be an object")
        recordServiceReference(runtime.provides, service, "singleton")
        provision = FacetProvision("singleton", service)
        provision.implementation = implementation
        provision.install = lambda provider: provider.provide(service, implementation)
        provision.validateReplacement = lambda provider: provider.validateReplacement(service, implementation)
        provision.replace = lambda provider: provider.replace(service, implementation)
        runtime.provisions.append(provision)

    def provideMany(self, service: Any) -> StagedServiceSpawner:
        runtime = self._runtime
        runtime.lifecycle.assertSettingUp("provide service instances")
        recordServiceReference(runtime.provides, service, "keyed")

        def validate(key: str, implementation: Any) -> None:
            if len(key) == 0:
                raise TypeError("Facet service instance key must not be empty")
            if not _isPlainObject(implementation):
                raise TypeError(f"Facet service {_serviceId(service)} implementation must be an object")
            if not _isLocal(service):
                validateRemoteServiceImplementation(_serviceId(service), implementation)

        instances = StagedServiceSpawner(runtime.lifecycle, validate)
        provision = FacetProvision("keyed", service)
        provision.connectLocal = lambda registry: instances.connect(
            lambda key, implementation: registry.spawn(service, key, implementation)
        )
        provision.connectRemote = lambda provider: instances.connect(
            lambda key, implementation: provider.spawn(service, key, implementation)
        )
        runtime.provisions.append(provision)
        return instances

    def use(self, service: Any) -> Any:
        runtime = self._runtime
        runtime.lifecycle.assertSettingUp("acquire services")
        recordServiceReference(runtime.requires, service, "singleton")
        view = runtime.singletonViews.get(_serviceId(service))
        if view is None:
            view = self._kernel.serviceSlots.getSingleton(service, runtime.lifecycle.assertServiceAccess)
            runtime.singletonViews[_serviceId(service)] = view
        return view

    def observe(self, service: Any, handler: Callable[[Any, Context], Any]) -> None:
        runtime = self._runtime
        runtime.lifecycle.assertSettingUp("observe services")
        recordServiceReference(runtime.requires, service, "keyed")
        runtime.lifecycle.observe(
            lambda: self._kernel.serviceSlots.observe(service, runtime.lifecycle.assertServiceAccess, handler)
        )

    def replicatedState(self, initial: Any) -> MutableReplicatedStateImpl[Any]:
        self._runtime.lifecycle.assertRunning("create replicated state")
        return MutableReplicatedStateImpl(initial)

    def own(self, disposal: Disposal) -> None:
        self._runtime.lifecycle.own(disposal)

    def onActivate(self, callback: Callable[[], Any]) -> None:
        self._runtime.lifecycle.onActivate(callback)

    def onDeactivate(self, callback: Callable[[], Any]) -> None:
        self._runtime.lifecycle.own(callback)


class FacetKernel:
    """Private lifecycle and dependency kernel behind the atomic host entry point."""

    def __init__(self, options: FacetOptions) -> None:
        facets = options["facets"]
        ids = [facet.id for facet in facets]
        if any(len(id_) == 0 for id_ in ids):
            raise RuntimeError("Facet ID must not be empty")
        if len(set(ids)) != len(ids):
            raise RuntimeError("Facet IDs must be unique within a generation")
        self._initialFacets = facets
        self._serviceSources: Sequence[RemoteServiceSource] = options.get("serviceSources") or []
        self._onError: Callable[[Exception], None] = options.get("onError") or (lambda error: None)
        self._facets: dict[str, FacetRuntime] = {}
        # MISAKA fork: upstream's `#serviceSlots`; public so `_Environment` (upstream's `#environment` closure) can
        # reach it.
        self.serviceSlots = HostServiceSlots()
        # `Map<RemoteServiceSource, RemoteServices>`, keyed by identity: a source need not be hashable.
        self._sourceBindings: dict[int, tuple[RemoteServiceSource, RemoteServices]] = {}
        self._activationOrder: list[str] = []
        self._provider: RemoteServiceProvider | None = None
        self._internalServices: RemoteServices | None = None
        self._localKeyedServices: LocalKeyedServiceRegistry | None = None
        self._phase = "setup"

    @property
    def provider(self) -> RemoteServiceProvider:
        if self._provider is None:
            raise RuntimeError("Facet service provider is not assembled")
        return self._provider

    def _setupFacet(self, facet: Facet, record: FacetRuntime) -> None:
        result: Any = facet.setup(_Environment(self, record))  # type: ignore[arg-type, func-returns-value]
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            raise RuntimeError(f"Facet {facet.id} setup must be synchronous")
        record.lifecycle.prepared()

    async def activate(self) -> None:
        records: list[FacetRuntime] = []
        try:
            for facet in self._initialFacets:
                record = FacetRuntime(facet.id)
                self._facets[facet.id] = record
                self._setupFacet(facet, record)
                records.append(record)

            self._phase = "assembling"
            externalServices = await self._resolveExternalServices(records)
            self._activationOrder = validateFacets(records, externalServices)
            self._assembleProviders()
            self._bindServices(externalServices)

            self._phase = "connecting"
            await asyncio.gather(
                *(services.ready(BACKGROUND_CONTEXT) for _source, services in list(self._sourceBindings.values())),
                self._internalServiceBinding().ready(BACKGROUND_CONTEXT),
            )

            self._phase = "activating"
            for id_ in self._activationOrder:
                await awaitValue(self._facets[id_].lifecycle.activate())
            self._phase = "active"
        except Exception as error:
            cleanupErrors = await self._terminate()
            if cleanupErrors:
                raise ExceptionGroup("Facet generation startup and cleanup failed", [error, *cleanupErrors]) from None
            raise

    @eager
    async def reload(self, facets: Sequence[Facet]) -> None:
        if self._phase != "active":
            raise RuntimeError(f"Facet host cannot reload while {self._phase}")
        ids = [facet.id for facet in facets]
        if any(len(id_) == 0 for id_ in ids):
            raise RuntimeError("Facet ID must not be empty")
        if len(set(ids)) != len(ids):
            raise RuntimeError("Reloaded facet IDs must be unique")
        for id_ in ids:
            if id_ not in self._facets:
                raise RuntimeError(f"Facet {id_} is not active")
        self._phase = "reloading"

        staged: list[FacetRuntime] = []
        candidates: list[FacetRuntime] = []
        try:
            for facet in facets:
                record = FacetRuntime(facet.id)
                staged.append(record)
                self._setupFacet(facet, record)
                previous = self._facets[facet.id]
                if not sameFacetShape(previous, record):
                    raise RuntimeError(
                        f"Reloaded facet {facet.id} must preserve its service requirements and provisions"
                    )
                self._validateReplacementProvisions(record.provisions)
                candidates.append(record)
        except Exception as error:
            cleanupErrors = await disposeFacetRecords(list(reversed(staged)))
            if cleanupErrors:
                abortErrors = await self._abort()
                raise ExceptionGroup(
                    "Facet reload setup and cleanup failed", [error, *cleanupErrors, *abortErrors]
                ) from None
            self._phase = "active"
            raise

        replacements = {record.facetId: record for record in candidates}
        candidateOrder = [replacements[id_] for id_ in self._activationOrder if id_ in replacements]
        try:
            for candidate in candidateOrder:
                await awaitValue(candidate.lifecycle.activate())
            for candidate in candidateOrder:
                self._validateReplacementProvisions(candidate.provisions)
        except Exception as error:
            cleanupErrors = await disposeFacetRecords(list(reversed(candidateOrder)))
            if cleanupErrors:
                abortErrors = await self._abort()
                raise ExceptionGroup(
                    "Facet reload activation and cleanup failed", [error, *cleanupErrors, *abortErrors]
                ) from None
            self._phase = "active"
            raise

        previousRecords = [self._facets[candidate.facetId] for candidate in candidateOrder]
        for candidate in candidateOrder:
            self._facets[candidate.facetId] = candidate
        try:
            for candidate in candidateOrder:
                for provision in candidate.provisions:
                    if provision.kind != "singleton":
                        continue
                    if _isLocal(provision.service):
                        self.serviceSlots.bindSingleton(_serviceId(provision.service), provision.implementation)
                    else:
                        provision.replace(self.provider)
            retirementErrors = await disposeFacetRecords(list(reversed(previousRecords)))
            if len(retirementErrors) == 1:
                raise retirementErrors[0]
            if len(retirementErrors) > 1:
                raise ExceptionGroup("Failed to retire replaced facets", retirementErrors)
            for candidate in candidateOrder:
                for provision in candidate.provisions:
                    if provision.kind != "keyed":
                        continue
                    if _isLocal(provision.service):
                        provision.connectLocal(self._localKeyedRegistry())
                    else:
                        provision.connectRemote(self.provider)
        except Exception as error:  # noqa: BLE001
            abortErrors = await self._abort(previousRecords)
            raise ExceptionGroup("Facet reload failed after cutover", [error, *abortErrors]) from None
        self._phase = "active"

    @eager
    async def dispose(self) -> None:
        if self._phase == "dead":
            return
        if self._phase != "active":
            raise RuntimeError(f"Facet host cannot be disposed while {self._phase}")
        errors = await self._terminate()
        if len(errors) == 1:
            raise errors[0]
        if len(errors) > 1:
            raise ExceptionGroup("Failed to dispose facet generation", errors)

    def _validateReplacementProvisions(self, provisions: Sequence[FacetProvision]) -> None:
        for provision in provisions:
            if provision.kind != "singleton" or _isLocal(provision.service):
                continue
            provision.validateReplacement(self.provider)

    async def _resolveExternalServices(self, records: Sequence[FacetRuntime]) -> dict[str, ExternalService]:
        async def catalogue(source: RemoteServiceSource) -> tuple[RemoteServiceSource, Any]:
            return source, await source.catalogue(BACKGROUND_CONTEXT)

        catalogues = await asyncio.gather(*(catalogue(source) for source in self._serviceSources))
        offered: dict[str, tuple[str, RemoteServiceSource]] = {}
        for source, entries in catalogues:
            for entry in entries:
                serviceId, mode = entry["serviceId"], entry["mode"]
                if serviceId in offered:
                    raise RuntimeError(f"Facet host service {serviceId} is offered by more than one source")
                offered[serviceId] = (mode, source)

        local = {reference.serviceId for record in records for reference in record.provides}
        external: dict[str, ExternalService] = {}
        for record in records:
            for requirement in record.requires:
                if requirement.serviceId in local or requirement.serviceId in external:
                    continue
                found = offered.get(requirement.serviceId)
                if found is None:
                    deferred = [source for source in self._serviceSources if source.acceptsUnavailableServices]
                    if len(deferred) > 1:
                        raise RuntimeError(
                            f"Facet host service {requirement.serviceId} has more than one deferred source"
                        )
                    if len(deferred) == 1:
                        found = (requirement.mode, deferred[0])
                if found is not None:
                    external[requirement.serviceId] = ExternalService(requirement.service, found[0], found[1])
        serviceIdsBySource: dict[int, tuple[RemoteServiceSource, list[str]]] = {}
        for serviceId, entry in external.items():
            serviceIdsBySource.setdefault(id(entry.source), (entry.source, []))[1].append(serviceId)
        for source, serviceIds in serviceIdsBySource.values():
            self._sourceBindings[id(source)] = (
                source,
                source.open({
                    "services": [{"id": id_} for id_ in serviceIds],
                    "assertAccess": self._assertServiceTargetAccess,
                    "onError": self._onError,
                }),
            )
        return external

    def _assembleProviders(self) -> None:
        provisions = self._provisions()
        remoteProvisions = [provision for provision in provisions if not _isLocal(provision.service)]
        provider = RemoteServiceProvider([
            {"service": provision.service, "mode": provision.kind} for provision in remoteProvisions
        ])
        internalServices = RemoteServiceBindingImpl({
            "services": [provision.service for provision in remoteProvisions],
            "transport": createLoopbackServiceTransport(provider),
            "assertAccess": self._assertServiceTargetAccess,
            "onError": self._onError,
        })
        localKeyedServices = LocalKeyedServiceRegistry(
            [provision.service for provision in provisions if provision.kind == "keyed" and _isLocal(provision.service)],
            self._onError,
        )
        self._provider = provider
        self._internalServices = internalServices
        self._localKeyedServices = localKeyedServices
        for provision in provisions:
            if provision.kind == "singleton":
                if not _isLocal(provision.service):
                    provision.install(provider)
            elif _isLocal(provision.service):
                provision.connectLocal(localKeyedServices)
            else:
                provision.connectRemote(provider)

    def _bindServices(self, externalServices: Mapping[str, ExternalService]) -> None:
        for provision in self._provisions():
            serviceId = _serviceId(provision.service)
            if provision.kind == "singleton":
                if not self.serviceSlots.hasSingleton(serviceId):
                    continue
                target = (
                    provision.implementation
                    if _isLocal(provision.service)
                    else self._internalServiceBinding().use(provision.service)
                )
                self.serviceSlots.bindSingleton(serviceId, target)
            else:
                self.serviceSlots.bindKeyed(
                    serviceId,
                    self._localKeyedRegistry() if _isLocal(provision.service) else self._internalServiceBinding(),
                )
        for serviceId, entry in externalServices.items():
            binding = self._sourceBindings.get(id(entry.source))
            if binding is None:
                raise RuntimeError(f"Service source for {serviceId} is not open")
            services = binding[1]
            if entry.mode == "singleton":
                self.serviceSlots.bindSingleton(serviceId, services.use(entry.service))
            else:
                self.serviceSlots.bindKeyed(serviceId, services)

    def _provisions(self) -> list[FacetProvision]:
        return [provision for record in self._facets.values() for provision in record.provisions]

    def _localKeyedRegistry(self) -> LocalKeyedServiceRegistry:
        if self._localKeyedServices is None:
            raise RuntimeError("Facet keyed services are not assembled")
        return self._localKeyedServices

    def _internalServiceBinding(self) -> RemoteServices:
        if self._internalServices is None:
            raise RuntimeError("Facet remote services are not assembled")
        return self._internalServices

    async def _disposeServiceBindings(self) -> list[Exception]:
        bindings = [services for _source, services in self._sourceBindings.values()]
        self._sourceBindings.clear()
        if self._internalServices is not None:
            bindings.append(self._internalServices)
        self._internalServices = None
        results = await asyncio.gather(*(services.dispose(BACKGROUND_CONTEXT) for services in bindings), return_exceptions=True)
        return [result for result in results if isinstance(result, Exception)]

    def _assertServiceTargetAccess(self) -> None:
        if self._phase not in ("activating", "active", "reloading", "disposing"):
            raise RuntimeError(f"Facet service targets cannot be used during {self._phase}")

    async def _abort(self, extraRecords: Sequence[FacetRuntime] = ()) -> list[Exception]:
        for record in self._facets.values():
            record.lifecycle.revoke()
        for record in extraRecords:
            record.lifecycle.revoke()
        return await self._terminate(extraRecords)

    async def _terminate(self, extraRecords: Sequence[FacetRuntime] = ()) -> list[Exception]:
        self._phase = "disposing"
        errors = await self._disposeLifecycles()
        errors.extend(await disposeFacetRecords(list(reversed(extraRecords))))
        try:
            if self._localKeyedServices is not None:
                self._localKeyedServices.dispose()
        except Exception as error:  # noqa: BLE001
            errors.append(error)
        self._localKeyedServices = None
        errors.extend(await self._disposeServiceBindings())
        try:
            self.serviceSlots.dispose()
        except Exception as error:  # noqa: BLE001
            errors.append(error)
        try:
            if self._provider is not None:
                self._provider.dispose()
        except Exception as error:  # noqa: BLE001
            errors.append(error)
        self._phase = "dead"
        return errors

    async def _disposeLifecycles(self) -> list[Exception]:
        errors: list[Exception] = []
        order = list(reversed(self._activationOrder if self._activationOrder else list(self._facets)))
        for id_ in order:
            record = self._facets.pop(id_, None)
            if record is None:
                continue
            try:
                await awaitValue(record.lifecycle.dispose())
            except Exception as error:  # noqa: BLE001
                errors.append(error)
        return errors


async def disposeFacetRecords(records: Sequence[FacetRuntime]) -> list[Exception]:
    errors: list[Exception] = []
    for record in records:
        try:
            await record.lifecycle.dispose()
        except Exception as error:  # noqa: BLE001
            errors.append(error)
    return errors


def validateFacets(records: Sequence[FacetRuntime], externalServices: Mapping[str, ExternalService]) -> list[str]:
    # serviceId -> (facetId or None for the host, mode)
    providers: dict[str, tuple[str | None, str]] = {}
    for serviceId, external in externalServices.items():
        providers[serviceId] = (None, external.mode)
    for record in records:
        for provision in record.provides:
            existing = providers.get(provision.serviceId)
            if existing is not None:
                existingFacet, existingMode = existing
                if existingMode != provision.mode:
                    raise RuntimeError(f"Service {provision.serviceId} is provided as both singleton and keyed")
                if existingFacet is None:
                    raise RuntimeError(
                        f"Service {provision.serviceId} is provided by both the host and {record.facetId}"
                    )
                raise RuntimeError(
                    f"Service {provision.serviceId} is provided by both {existingFacet} and {record.facetId}"
                )
            providers[provision.serviceId] = (record.facetId, provision.mode)

    dependencies: dict[str, set[str]] = {record.facetId: set() for record in records}
    # Insertion-ordered, as a ``Set`` is: the activation order must not depend on string hashes.
    dependents: dict[str, dict[str, None]] = {record.facetId: {} for record in records}
    for record in records:
        for requirement in record.requires:
            provider = providers.get(requirement.serviceId)
            if provider is None:
                raise RuntimeError(
                    f"Facet {record.facetId} requires local/{requirement.serviceId}/{requirement.mode}, "
                    "but no facet provides it"
                )
            providerFacet, providerMode = provider
            if providerMode != requirement.mode:
                raise RuntimeError(
                    f"Facet {record.facetId} requires {requirement.serviceId} as {requirement.mode}, "
                    f"but {providerFacet if providerFacet is not None else 'the host'} provides it as {providerMode}"
                )
            if providerFacet is None or providerFacet == record.facetId:
                continue
            dependencies[record.facetId].add(providerFacet)
            dependents[providerFacet][record.facetId] = None
    return topologicalOrder(records, dependencies, dependents)


def topologicalOrder(
    records: Sequence[FacetRuntime], dependencies: Mapping[str, set[str]], dependents: Mapping[str, dict[str, None]]
) -> list[str]:
    remaining = {id_: len(values) for id_, values in dependencies.items()}
    ready = [record.facetId for record in records if remaining.get(record.facetId) == 0]
    order: list[str] = []
    while ready:
        id_ = ready.pop(0)
        order.append(id_)
        for dependent in dependents.get(id_, ()):
            count = remaining[dependent] - 1
            remaining[dependent] = count
            if count == 0:
                ready.append(dependent)
    if len(order) != len(records):
        cycle = [record.facetId for record in records if remaining[record.facetId] > 0]
        raise RuntimeError(f"Facet dependency cycle: {', '.join(cycle)}")
    return order


def recordServiceReference(target: list[FacetServiceReference], service: Any, mode: str) -> None:
    if any(reference.serviceId == _serviceId(service) and reference.mode == mode for reference in target):
        return
    target.append(FacetServiceReference(_serviceId(service), service, mode))


def sameFacetShape(left: FacetRuntime, right: FacetRuntime) -> bool:
    return sameReferences(left.requires, right.requires) and sameReferences(left.provides, right.provides)


def sameReferences(left: Sequence[FacetServiceReference], right: Sequence[FacetServiceReference]) -> bool:
    return len(left) == len(right) and all(
        any(other.serviceId == reference.serviceId and other.mode == reference.mode for other in right)
        for reference in left
    )


__all__ = ["FacetKernel", "disposeFacetRecords"]
