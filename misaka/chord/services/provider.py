"""The service provider: it hosts implementations, answers calls, and publishes state to subscribers.

Translated from pi's ``packages/chord/src/services/provider.ts``.

An implementation is a mapping from member name to member, or an object whose public attributes (names not
beginning with an underscore) are its members. A member is a method (any callable; it takes the call's
arguments and a trailing ``Context`` and may be a coroutine function) or a replicated state. Anything else is
not remotely exposable. Upstream reads own data properties; the two Python forms stand for them.

A service is a ``Service`` or any object or mapping with an ``id`` (and optionally ``local``).
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from misaka.chord.types import (
    Context,
    ServiceCall,
    ServiceCatalogueEntry,
    ServiceInstanceAddress,
    ServiceInstanceSnapshot,
    ServiceMemberSnapshot,
    ServiceMode,
    ServiceProviderUpdate,
    ServiceSubscription,
    ServiceSubscriptionSnapshot,
)
from misaka.utils.collation import localeSortKey
from misaka.utils.js import UNDEFINED, jsonStringify, startTask, toUtf16Units

from .errors import RemoteServiceError
from .handle import AttributeMapping
from .state import serviceDeliveryContext
from .state_internals import ReplicatedStateInternals, getReplicatedStateInternals
from .wire import decodeServiceControlCall


class InstanceMember:
    """``{"kind": "method", "method"}`` or ``{"kind": "state", "state"}``."""

    __slots__ = ("kind", "method", "state")

    def __init__(self, kind: str, method: Callable[..., Any] | None = None, state: ReplicatedStateInternals | None = None) -> None:
        self.kind = kind
        self.method = method
        self.state = state


class ClassifiedRemoteServiceImplementation:
    __slots__ = ("implementation", "members")

    def __init__(self, implementation: Any, members: dict[str, InstanceMember]) -> None:
        self.implementation = implementation
        self.members = members


class ProviderInstance:
    __slots__ = ("active", "address", "implementation", "members", "removeMemberListeners")

    def __init__(
        self,
        address: ServiceInstanceAddress | None,
        implementation: Any,
        members: dict[str, InstanceMember],
    ) -> None:
        self.address = address
        self.implementation = implementation
        self.members = members
        self.removeMemberListeners: list[Callable[[], None]] = []
        self.active = True


class ProviderSubscriber:
    __slots__ = ("active", "buffer", "closed", "draining", "listener", "snapshotSequences", "terminated")

    def __init__(self, listener: Callable[[ServiceProviderUpdate, Context], None]) -> None:
        self.listener = listener
        self.buffer: list[tuple[ServiceProviderUpdate, Context]] = []
        self.snapshotSequences: dict[str, int] = {}
        self.active = False
        self.draining = False
        self.terminated = False
        self.closed = False


class ServiceRegistration:
    __slots__ = ("generations", "instances", "mode", "serviceId", "singleton", "singletonShape", "subscribers")

    def __init__(self, serviceId: str, mode: ServiceMode) -> None:
        self.serviceId = serviceId
        self.mode = mode
        self.singleton: ProviderInstance | None = None
        self.singletonShape: dict[str, str] | None = None
        self.instances: dict[str, ProviderInstance] = {}
        self.generations: dict[str, int] = {}
        self.subscribers: dict[ProviderSubscriber, None] = {}


class _ProviderSubscription:
    def __init__(self, registration: ServiceRegistration, subscriber: ProviderSubscriber, snapshot: ServiceSubscriptionSnapshot) -> None:
        self._registration = registration
        self._subscriber = subscriber
        self.snapshot = snapshot

    def activate(self) -> None:
        subscriber = self._subscriber
        if subscriber.closed or subscriber.active:
            return
        subscriber.active = True
        throwCollectedErrors(drainSubscriber(subscriber), "Failed to activate remote service subscription")

    def close(self, context: Context | None = None) -> None:
        subscriber = self._subscriber
        if subscriber.closed:
            return
        subscriber.closed = True
        subscriber.buffer.clear()
        self._registration.subscribers.pop(subscriber, None)


def _serviceId(service: Any) -> str:
    return service["id"] if isinstance(service, Mapping) else service.id


def _isLocal(service: Any) -> bool:
    local = service.get("local") if isinstance(service, Mapping) else getattr(service, "local", None)
    return local is True

type ServiceUpdatePublisher = Callable[[str, ServiceProviderUpdate, Context], Any]


class RemoteServiceEndpoint:
    """Hosts one provider for one remote consumer and owns that consumer's subscriptions."""

    def __init__(self, provider: RemoteServiceProvider) -> None:
        self._provider = provider
        self._subscriptions: dict[str, ServiceSubscription] = {}
        self._disposed = False

    async def invoke(self, call: ServiceCall, publish: ServiceUpdatePublisher, context: Context) -> Any:
        if self._disposed:
            raise RuntimeError("Remote service endpoint is disposed")
        control = decodeServiceControlCall(call)
        if control is not None and control["type"] == "catalogue":
            return self._provider.catalogue
        if control is not None and control["type"] == "subscribe":
            subscriptionId = control["subscriptionId"]
            if subscriptionId in self._subscriptions:
                raise RuntimeError("Service subscription ID is already active")

            def listener(update: ServiceProviderUpdate, updateContext: Context) -> None:
                result = publish(subscriptionId, update, updateContext)
                if inspect.isawaitable(result):
                    startTask(result).add_done_callback(_ignore)

            subscription = self._provider.subscribe(control["serviceId"], control["mode"], listener)
            self._subscriptions[subscriptionId] = subscription
            subscription.activate()
            return subscription.snapshot
        if control is not None and control["type"] == "unsubscribe":
            existing = self._subscriptions.get(control["subscriptionId"])
            if existing is None:
                raise RuntimeError("Service subscription was not found")
            existing.close()
            del self._subscriptions[control["subscriptionId"]]
            return UNDEFINED
        return await self._provider.invoke(call, context)

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        for subscription in self._subscriptions.values():
            subscription.close()
        self._subscriptions.clear()


def _ignore(task: asyncio.Future[Any]) -> None:
    """``.catch(() => {})``: a failed publication is dropped, and its exception is retrieved."""
    if not task.cancelled():
        task.exception()


class RemoteServiceProvider:
    def __init__(self, entries: Sequence[Any]) -> None:
        definitions: list[tuple[Any, ServiceMode]] = []
        for entry in entries:
            if isinstance(entry, Mapping) and "service" in entry:
                definitions.append((entry["service"], entry["mode"]))
            else:
                definitions.append((entry, "singleton"))
        for service, _mode in definitions:
            if _isLocal(service):
                raise TypeError(f"Local service {_serviceId(service)} cannot be published remotely")
        ids = [_serviceId(service) for service, _mode in definitions]
        if len(set(ids)) != len(ids):
            raise TypeError("Remote service catalogue contains duplicate IDs")
        self._catalogue: list[ServiceCatalogueEntry] = [
            {"serviceId": _serviceId(service), "mode": mode} for service, mode in definitions
        ]
        self._registrations: dict[str, ServiceRegistration] = {}
        self._disposed = False
        for service, mode in definitions:
            self._registrations[_serviceId(service)] = ServiceRegistration(_serviceId(service), mode)

    @property
    def catalogue(self) -> Sequence[ServiceCatalogueEntry]:
        return self._catalogue

    def provide(self, service: Any, implementation: Any) -> None:
        self._assertActive()
        self._assertRemotable(service)
        self._assertAllowed(_serviceId(service))
        registration = self._registration(_serviceId(service), "singleton")
        if registration.singleton is not None:
            raise RemoteServiceError("service_mode_mismatch", f"Remote service {_serviceId(service)} already has a provider")
        classified = classifyRemoteServiceImplementation(registration.serviceId, implementation)
        shape = serviceMemberShape(classified.members)
        self._assertSingletonShape(registration, shape)
        registration.singleton = self._createInstance(registration, classified, None)
        registration.singletonShape = shape

    def withdraw(self, service: Any) -> None:
        """Disconnect one singleton while preserving active subscriptions and remote facades."""
        self._assertActive()
        self._assertRemotable(service)
        self._assertAllowed(_serviceId(service))
        registration = self._registration(_serviceId(service), "singleton")
        previous = registration.singleton
        if previous is None:
            return
        previous.active = False
        for remove in previous.removeMemberListeners:
            remove()
        registration.singleton = None
        self._emit(registration, {"type": "unavailable"})

    def validateReplacement(self, service: Any, implementation: Any) -> None:
        """Check a singleton replacement without changing the active provider."""
        self._assertActive()
        self._assertRemotable(service)
        self._assertAllowed(_serviceId(service))
        registration = self._registration(_serviceId(service), "singleton")
        classified = classifyRemoteServiceImplementation(registration.serviceId, implementation)
        self._assertSingletonShape(registration, serviceMemberShape(classified.members))

    def replace(self, service: Any, implementation: Any) -> None:
        """Replace one singleton without making its stable remote facade unavailable."""
        self._assertActive()
        self._assertRemotable(service)
        self._assertAllowed(_serviceId(service))
        registration = self._registration(_serviceId(service), "singleton")
        classified = classifyRemoteServiceImplementation(registration.serviceId, implementation)
        shape = serviceMemberShape(classified.members)
        self._assertSingletonShape(registration, shape)
        replacement = self._createInstance(registration, classified, None)
        previous = registration.singleton
        if previous is not None:
            previous.active = False
            for remove in previous.removeMemberListeners:
                remove()
        registration.singleton = replacement
        registration.singletonShape = shape
        self._emit(registration, {"type": "replaced", "snapshot": self._snapshotInstance(replacement)})

    def use(self, service: Any) -> Any:
        self._assertActive()
        self._assertRemotable(service)
        self._assertAllowed(_serviceId(service))
        registration = self._registrations.get(_serviceId(service))
        if registration is None or registration.mode != "singleton" or registration.singleton is None:
            raise RemoteServiceError("service_not_found", f"Remote service {_serviceId(service)} has no local provider")
        implementation = registration.singleton.implementation
        return AttributeMapping(implementation) if isinstance(implementation, Mapping) else implementation

    def spawn(self, service: Any, key: str, implementation: Any) -> Callable[[], None]:
        self._assertActive()
        self._assertRemotable(service)
        self._assertAllowed(_serviceId(service))
        if len(key) == 0:
            raise TypeError("Remote service instance key must not be empty")
        registration = self._registration(_serviceId(service), "keyed")
        if key in registration.instances:
            raise RemoteServiceError(
                "service_mode_mismatch",
                f"Remote service {_serviceId(service)} already has a live instance with key {key}",
            )
        generation = registration.generations.get(key, 0) + 1
        registration.generations[key] = generation
        address: ServiceInstanceAddress = {"key": key, "generation": generation}
        classified = classifyRemoteServiceImplementation(registration.serviceId, implementation)
        instance = self._createInstance(registration, classified, address)
        registration.instances[key] = instance
        self._emit(registration, {"type": "spawned", "instance": self._snapshotInstance(instance)})
        closed = False

        def close() -> None:
            nonlocal closed
            if closed:
                return
            closed = True
            if registration.instances.get(key) is not instance:
                return
            instance.active = False
            for remove in instance.removeMemberListeners:
                remove()
            registration.instances.pop(key, None)
            self._emit(registration, {"type": "closed", "instance": address})

        return close

    async def invoke(self, call: ServiceCall, context: Context) -> Any:
        self._assertActive()
        self._assertAllowed(call["serviceId"])
        registration = self._registrations.get(call["serviceId"])
        if registration is None:
            raise RemoteServiceError("service_not_found", f"Unknown remote service {call['serviceId']}")
        instance = self._resolveInstance(registration, call.get("instance"))
        member = instance.members.get(call["member"])
        if member is None:
            raise RemoteServiceError(
                "service_member_not_found", f"Unknown remote service member {call['serviceId']}.{call['member']}"
            )
        if member.kind != "method":
            raise RemoteServiceError(
                "service_member_mismatch", f"Remote service member {call['serviceId']}.{call['member']} is not a method"
            )
        assert member.method is not None
        result = member.method(*call["args"], context)
        if inspect.isawaitable(result):
            result = await result
        return result

    def subscribe(
        self, serviceId: str, mode: ServiceMode, listener: Callable[[ServiceProviderUpdate, Context], None]
    ) -> ServiceSubscription:
        self._assertActive()
        self._assertAllowed(serviceId)
        registration = self._registration(serviceId, mode)
        if registration.mode == "singleton" and registration.singleton is None:
            raise RemoteServiceError("service_not_found", f"Remote service {serviceId} has no provider")
        subscriber = ProviderSubscriber(listener)
        registration.subscribers[subscriber] = None
        snapshot = self._snapshot(registration)
        recordSnapshotSequences(subscriber.snapshotSequences, snapshot["instances"])
        return _ProviderSubscription(registration, subscriber, snapshot)  # type: ignore[return-value]

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        errors: list[Exception] = []
        for registration in self._registrations.values():
            singleton = registration.singleton
            if singleton is not None:
                singleton.active = False
                for remove in singleton.removeMemberListeners:
                    remove()
                registration.singleton = None
                try:
                    self._emit(registration, {"type": "unavailable"})
                except Exception as error:  # noqa: BLE001 - collected
                    errors.append(error)
            for key, instance in list(registration.instances.items()):
                instance.active = False
                for remove in instance.removeMemberListeners:
                    remove()
                registration.instances.pop(key, None)
                try:
                    assert instance.address is not None
                    self._emit(registration, {"type": "closed", "instance": instance.address})
                except Exception as error:  # noqa: BLE001 - collected
                    errors.append(error)
            for subscriber in registration.subscribers:
                if subscriber.active and not subscriber.draining:
                    subscriber.closed = True
                    subscriber.buffer.clear()
                else:
                    subscriber.terminated = True
            registration.subscribers.clear()
        self._registrations.clear()
        if len(errors) == 1:
            raise errors[0]
        if len(errors) > 1:
            raise ExceptionGroup("Failed to dispose remote service provider", errors)

    def _registration(self, serviceId: str, mode: ServiceMode) -> ServiceRegistration:
        registration = self._registrations.get(serviceId)
        if registration is None:
            raise RemoteServiceError("service_not_found", f"Unknown remote service {serviceId}")
        if registration.mode != mode:
            raise RemoteServiceError(
                "service_mode_mismatch", f"Remote service {serviceId} is {registration.mode}, not {mode}"
            )
        return registration

    def _createInstance(
        self,
        registration: ServiceRegistration,
        classified: ClassifiedRemoteServiceImplementation,
        address: ServiceInstanceAddress | None,
    ) -> ProviderInstance:
        instance = ProviderInstance(address, classified.implementation, classified.members)
        for name, member in classified.members.items():
            if member.kind != "state":
                continue
            assert member.state is not None

            def onState(ops: Sequence[Any], sequence: int, context: Context, name: str = name) -> None:
                if not instance.active:
                    return
                update: ServiceProviderUpdate = {"type": "state", "member": name, "sequence": sequence, "ops": ops}
                if address is not None:
                    update["instance"] = address
                self._emit(registration, update, context)

            instance.removeMemberListeners.append(member.state.subscribe(onState))
        return instance

    def _assertSingletonShape(self, registration: ServiceRegistration, replacement: dict[str, str]) -> None:
        current = registration.singletonShape
        if current is None or sameServiceMemberShape(current, replacement):
            return
        raise RemoteServiceError(
            "service_member_mismatch",
            f"Remote service {registration.serviceId} replacement must preserve its member shape",
        )

    def _resolveInstance(self, registration: ServiceRegistration, address: ServiceInstanceAddress | None) -> ProviderInstance:
        if registration.mode == "singleton":
            if address is not None:
                raise RemoteServiceError("service_mode_mismatch", f"Remote service {registration.serviceId} is singleton")
            if registration.singleton is None:
                raise RemoteServiceError("service_not_found", f"Remote service {registration.serviceId} has no provider")
            return registration.singleton
        if address is None:
            raise RemoteServiceError("service_mode_mismatch", f"Remote service {registration.serviceId} is keyed")
        instance = registration.instances.get(address["key"])
        if instance is None:
            raise RemoteServiceError(
                "service_instance_not_found", f"Remote service {registration.serviceId} has no instance {address['key']}"
            )
        if instance.address is None or instance.address["generation"] != address["generation"]:
            raise RemoteServiceError(
                "service_stale_instance", f"Remote service {registration.serviceId} instance {address['key']} is stale"
            )
        return instance

    def _snapshot(self, registration: ServiceRegistration) -> ServiceSubscriptionSnapshot:
        if registration.mode == "singleton":
            instances = [self._snapshotInstance(registration.singleton)] if registration.singleton else []
        else:
            ordered = sorted(registration.instances.values(), key=lambda instance: localeSortKey(instance.address["key"]))  # type: ignore[index]
            instances = [self._snapshotInstance(instance) for instance in ordered]
        return {"serviceId": registration.serviceId, "mode": registration.mode, "instances": instances}

    def _snapshotInstance(self, instance: ProviderInstance) -> ServiceInstanceSnapshot:
        members: list[ServiceMemberSnapshot] = []
        for name, member in instance.members.items():
            if member.kind == "method":
                members.append({"name": name, "kind": "method"})
            else:
                assert member.state is not None
                snapshot = member.state.snapshot()
                members.append(
                    {"name": name, "kind": "state", "sequence": snapshot["sequence"], "ops": [["r", snapshot["value"]]]}
                )
        result: ServiceInstanceSnapshot = {"members": members}
        if instance.address is not None:
            result = {"instance": instance.address, "members": members}
        return result

    def _emit(self, registration: ServiceRegistration, update: ServiceProviderUpdate, context: Context | None = None) -> None:
        if len(registration.subscribers) == 0:
            return
        deliveryContext = context if context is not None else serviceDeliveryContext()
        errors: list[Exception] = []
        subscribers = list(registration.subscribers)
        # Queue for everyone before invoking user code, including reentrant publications.
        for subscriber in subscribers:
            if subscriber.closed or updateCoveredBySnapshot(subscriber.snapshotSequences, update):
                continue
            if len(subscriber.buffer) == 100:
                snapshot = self._snapshot(registration)
                subscriber.buffer.clear()
                subscriber.snapshotSequences.clear()
                recordSnapshotSequences(subscriber.snapshotSequences, snapshot["instances"])
                subscriber.buffer.append(({"type": "reset", "snapshot": snapshot}, deliveryContext))
            else:
                subscriber.buffer.append((update, deliveryContext))
        for subscriber in subscribers:
            errors.extend(drainSubscriber(subscriber))
        throwCollectedErrors(errors, f"Failed to publish remote service {registration.serviceId} update")

    def _assertRemotable(self, service: Any) -> None:
        if _isLocal(service):
            raise RemoteServiceError("service_not_allowed", f"Service {_serviceId(service)} is process-local")

    def _assertAllowed(self, serviceId: str) -> None:
        if serviceId not in self._registrations:
            raise RemoteServiceError("service_not_allowed", f"Remote service {serviceId} is not allowlisted")

    def _assertActive(self) -> None:
        if self._disposed:
            raise RuntimeError("Remote service provider is disposed")


def drainSubscriber(subscriber: ProviderSubscriber) -> list[Exception]:
    if not subscriber.active or subscriber.closed or subscriber.draining:
        return []
    errors: list[Exception] = []
    subscriber.draining = True
    try:
        while not subscriber.closed:
            if not subscriber.buffer:
                break
            update, context = subscriber.buffer.pop(0)
            try:
                subscriber.listener(update, context)
            except Exception as error:  # noqa: BLE001 - collected, as a listener's throw is
                errors.append(error)
    finally:
        subscriber.draining = False
        if subscriber.terminated:
            subscriber.closed = True
    return errors


def recordSnapshotSequences(sequences: dict[str, int], instances: Sequence[ServiceInstanceSnapshot]) -> None:
    for instance in instances:
        for member in instance["members"]:
            if member["kind"] == "state":
                sequences[stateMemberKey(instance.get("instance"), member["name"])] = member["sequence"]  # type: ignore[typeddict-item]


def updateCoveredBySnapshot(sequences: dict[str, int], update: ServiceProviderUpdate) -> bool:
    match update["type"]:
        case "state":
            key = stateMemberKey(update.get("instance"), update["member"])  # type: ignore[typeddict-item]
            sequence = sequences.get(key)
            if sequence is None:
                return False
            if update["sequence"] <= sequence:  # type: ignore[typeddict-item]
                return True
            del sequences[key]
            return False
        case "reset":
            sequences.clear()
            recordSnapshotSequences(sequences, update["snapshot"]["instances"])  # type: ignore[typeddict-item]
            return False
        case "replaced":
            sequences.clear()
            recordSnapshotSequences(sequences, [update["snapshot"]])  # type: ignore[typeddict-item]
            return False
        case "spawned":
            recordSnapshotSequences(sequences, [update["instance"]])  # type: ignore[typeddict-item]
            return False
        case "unavailable":
            sequences.clear()
            return False
        case "closed":
            instance = update["instance"]  # type: ignore[typeddict-item]
            prefix = f"{jsonStringify([instance['key'], instance['generation']])[:-1]},"
            for key in [key for key in sequences if key.startswith(prefix)]:
                del sequences[key]
            return False
    return False


def stateMemberKey(instance: ServiceInstanceAddress | None, member: str) -> str:
    return jsonStringify([member] if instance is None else [instance["key"], instance["generation"], member])


def createRemoteServiceEndpoint(provider: RemoteServiceProvider) -> RemoteServiceEndpoint:
    return RemoteServiceEndpoint(provider)


def validateRemoteServiceImplementation(serviceId: str, implementation: Any) -> None:
    classifyRemoteServiceImplementation(serviceId, implementation)


def classifyRemoteServiceImplementation(serviceId: str, implementation: Any) -> ClassifiedRemoteServiceImplementation:
    if implementation is None or isinstance(implementation, str | bytes | int | float | bool | list | tuple | set):
        raise TypeError(f"Remote service {serviceId} implementation must be an object")
    # MISAKA fork: an implementation is a mapping of members, or an object whose public attributes are its members
    # (`dir`/`getattr`), where upstream reads the own data properties of a plain object.
    if isinstance(implementation, Mapping):
        names = list(implementation)

        def read(name: str) -> Any:
            return implementation[name]
    else:
        names = [name for name in dir(implementation) if not name.startswith("_")]

        def read(name: str) -> Any:
            return getattr(implementation, name)

    members: dict[str, InstanceMember] = {}
    # `Object.keys(...).sort()` orders by UTF-16 code units.
    for name in sorted(names, key=toUtf16Units):
        value = read(name)
        if callable(value):
            members[name] = InstanceMember("method", method=value)
            continue
        state = getReplicatedStateInternals(value)
        if state is not None:
            members[name] = InstanceMember("state", state=state)
            continue
        raise TypeError(f"Remote service member {serviceId}.{name} is not remotely exposable")
    if len(members) == 0:
        raise TypeError(f"Remote service {serviceId} has no members")
    return ClassifiedRemoteServiceImplementation(implementation, members)


def throwCollectedErrors(errors: Sequence[Exception], message: str) -> None:
    if len(errors) == 1:
        raise errors[0]
    if len(errors) > 1:
        raise ExceptionGroup(message, list(errors))


def serviceMemberShape(members: dict[str, InstanceMember]) -> dict[str, str]:
    return {name: member.kind for name, member in members.items()}


def sameServiceMemberShape(left: dict[str, str], right: dict[str, str]) -> bool:
    return len(left) == len(right) and all(right.get(name) == kind for name, kind in left.items())


__all__ = [
    "RemoteServiceEndpoint",
    "RemoteServiceProvider",
    "ServiceUpdatePublisher",
    "createRemoteServiceEndpoint",
    "validateRemoteServiceImplementation",
]
