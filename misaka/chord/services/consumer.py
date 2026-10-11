"""The service consumer: a binding that turns a transport into live service handles.

Translated from pi's ``packages/chord/src/services/consumer.ts``.

MISAKA forks, forced by the language: a service handle is a small object whose attribute reads return the
member views (upstream hands out ``Proxy`` objects); a method view is called with the call's arguments and a
trailing ``Context`` and answers a coroutine, as upstream answers a ``Promise`` (a precondition that upstream
throws synchronously raises when called, one that upstream rejects raises when awaited); a ``Promise`` held in
a field is an ``asyncio`` task.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from misaka.chord.context import BACKGROUND_CONTEXT, awaitWithContext
from misaka.chord.delta import Op
from misaka.chord.types import (
    Context,
    RemoteServiceBindingOptions,
    ServiceInstanceAddress,
    ServiceInstanceSnapshot,
    ServiceMemberSnapshot,
    ServiceProviderUpdate,
    ServiceSubscription,
    ServiceSubscriptionSnapshot,
)
from misaka.utils.js import eager, resolvedPromise, startTask

from .errors import RemoteServiceError
from .handle import ServiceSlot
from .instances import InstanceDirectory
from .state import ReplicatedStateReplica, serviceDeliveryContext

type ErrorReporter = Callable[[Exception], None]


async def _reject(error: BaseException) -> Any:
    raise error


async def _resumed[T](awaitable: Awaitable[T]) -> T:
    """``await promise``: JavaScript resumes a function after a turn of the microtask queue even when the promise
    is settled already, so what else was queued runs first; a coroutine that finishes without suspending would
    resume at once. The transport calls whose ordering the binding guards against are awaited through this."""
    result = await awaitable
    await asyncio.sleep(0)
    return result


async def _settle(value: Any) -> None:
    if inspect.isawaitable(value):
        await _resumed(value)
    else:
        await asyncio.sleep(0)



class RemoteMember:
    """The view of one remote member: call it for a method, read ``value`` or ``subscribe`` for a state."""

    __slots__ = ("_slot",)
    _slot: MemberSlot

    def __init__(self, slot: MemberSlot) -> None:
        object.__setattr__(self, "_slot", slot)

    def __call__(self, *args: Any) -> Awaitable[Any]:
        return self._slot.call(list(args))

    @property
    def value(self) -> Any:
        return self._slot.stateValue()

    def subscribe(self, listener: Callable[..., Any]) -> Callable[[], None]:
        return self._slot.subscribe(listener)

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("A service handle is read-only")


class MemberSlot:
    def __init__(
        self,
        serviceId: str,
        member: str,
        invoke: Callable[[list[Any], Context], Awaitable[Any]],
        isActive: Callable[[], bool],
        assertAccess: Callable[[], None],
        reportError: ErrorReporter,
    ) -> None:
        self._serviceId = serviceId
        self._member = member
        self._invoke = invoke
        self._isActive = isActive
        self._assertAccess = assertAccess
        self._state: ReplicatedStateReplica[Any] = ReplicatedStateReplica(reportError)
        self._kind: str | None = None
        self._expectedKind: str | None = None
        self.value = RemoteMember(self)

    def setDescription(self, kind: str) -> None:
        if self._kind is not None and self._kind != kind:
            raise RuntimeError(f"Remote service member {self._serviceId}.{self._member} changed kind")
        self._kind = kind
        if self._expectedKind is not None and self._expectedKind != kind:
            raise RemoteServiceError(
                "service_member_mismatch",
                f"Remote service member {self._serviceId}.{self._member} is {kind}, not {self._expectedKind}",
            )

    def hydrate(self, sequence: int, ops: Sequence[Op], context: Context) -> None:
        self.setDescription("state")
        self._state.hydrate(sequence, ops, context)

    def update(self, sequence: int, ops: Sequence[Op], context: Context) -> None:
        self.setDescription("state")
        self._state.update(sequence, ops, context)

    def clear(self) -> None:
        self._state.clear()

    def stateValue(self) -> Any:
        self._assertAccess()
        self._expect("state")
        return self._state.value

    def subscribe(self, listener: Callable[..., Any]) -> Callable[[], None]:
        self._assertAccess()
        if not callable(listener):
            raise TypeError("Replicated state subscription listener must be a function")
        self._expect("state")
        return self._state.subscribe(listener)

    def _expect(self, kind: str) -> None:
        if self._expectedKind is not None and self._expectedKind != kind:
            raise RemoteServiceError(
                "service_member_mismatch",
                f"Remote service member {self._serviceId}.{self._member} was used as two different kinds",
            )
        self._expectedKind = kind
        if self._kind is not None and self._kind != kind:
            raise RemoteServiceError(
                "service_member_mismatch",
                f"Remote service member {self._serviceId}.{self._member} is {self._kind}, not {kind}",
            )

    def call(self, args: list[Any]) -> Awaitable[Any]:
        self._assertAccess()
        self._expect("method")
        if not self._isActive():
            return startTask(
                _reject(
                    RemoteServiceError("service_stale_instance", f"Remote service {self._serviceId} binding is closed")
                )
            )
        context = args[-1] if args else None
        if not isContext(context):
            return startTask(
                _reject(
                    RemoteServiceError(
                        "service_invalid_value",
                        f"Remote service method {self._serviceId}.{self._member} requires a trailing Context",
                    )
                )
            )
        # A promise-returning call starts when it is made, awaited or not.
        return startTask(self._invoke(args[:-1], context))  # type: ignore[arg-type]


class ServiceFacadeProxy:
    """The handle a consumer holds: reading an attribute names a member."""

    __slots__ = ("_ServiceFacadeProxy__facade",)

    def __init__(self, facade: ServiceFacade) -> None:
        object.__setattr__(self, "_ServiceFacadeProxy__facade", facade)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return self._ServiceFacadeProxy__facade.slot(name).value

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("A service handle is read-only")


class ServiceFacade:
    def __init__(
        self,
        serviceId: str,
        address: ServiceInstanceAddress | None,
        transport: Any,
        isActive: Callable[[], bool],
        assertAccess: Callable[[], None],
        reportError: ErrorReporter,
    ) -> None:
        self._serviceId = serviceId
        self._address = address
        self._transport = transport
        self._isActive = isActive
        self._assertAccess = assertAccess
        self._reportError = reportError
        self._slots: dict[str, MemberSlot] = {}
        self._descriptions: dict[str, str] = {}
        self.proxy = ServiceFacadeProxy(self)

    def install(self, snapshot: ServiceInstanceSnapshot, context: Context) -> None:
        if not sameAddress(snapshot.get("instance"), self._address):
            raise RuntimeError("Remote service snapshot has the wrong address")
        members = validateMembers(snapshot["members"])
        for name in self._slots:
            if name not in members:
                raise RemoteServiceError(
                    "service_member_not_found", f"Unknown remote service member {self._serviceId}.{name}"
                )
        self._descriptions.clear()
        for name, member in members.items():
            self._descriptions[name] = member["kind"]
        for member in members.values():
            slot = self._slots.get(member["name"])
            if member["kind"] == "state":
                (slot or self.slot(member["name"])).hydrate(member["sequence"], member["ops"], context)  # type: ignore[typeddict-item]
            elif slot is not None:
                slot.setDescription(member["kind"])

    def update(self, member: str, sequence: int, ops: Sequence[Op], context: Context) -> None:
        if self._descriptions.get(member) != "state":
            raise RuntimeError(f"Remote service update targets non-state member {self._serviceId}.{member}")
        self.slot(member).update(sequence, ops, context)

    def clear(self) -> None:
        for slot in self._slots.values():
            slot.clear()

    def slot(self, member: str) -> MemberSlot:
        slot = self._slots.get(member)
        if slot is not None:
            return slot

        def invoke(args: list[Any], context: Context) -> Awaitable[Any]:
            call: dict[str, Any] = {"serviceId": self._serviceId, "member": member, "args": args}
            if self._address is not None:
                call["instance"] = self._address
            return self._transport.invoke(call, context)

        slot = MemberSlot(self._serviceId, member, invoke, self._isActive, self._assertAccess, self._reportError)
        kind = self._descriptions.get(member)
        if kind is not None:
            slot.setDescription(kind)
        self._slots[member] = slot
        return slot


class SingletonBinding:
    __slots__ = ("active", "facade", "revision", "starting", "subscription")

    def __init__(self) -> None:
        self.facade: ServiceFacade | None = None
        self.subscription: ServiceSubscription | None = None
        self.starting: asyncio.Future[None] | None = None
        self.active = True
        self.revision = 0


class KeyedInstance:
    """One live keyed instance: the entry an ``InstanceDirectory`` holds."""

    __slots__ = ("deactivate", "facade", "generation", "key", "service")

    def __init__(
        self, key: str, generation: int, service: Any, facade: ServiceFacade, deactivate: Callable[[], None]
    ) -> None:
        self.key = key
        self.generation = generation
        self.service = service
        self.facade = facade
        self.deactivate = deactivate


class KeyedBinding:
    def __init__(
        self,
        service: Any,
        transport: Any,
        reportError: ErrorReporter,
        assertAccess: Callable[[], None],
        onEmpty: Callable[[], None],
        bound: bool,
    ) -> None:
        self._service = service
        self._transport = transport
        self._reportError = reportError
        self._assertAccess = assertAccess
        self._onEmpty = onEmpty
        self._instances: InstanceDirectory[KeyedInstance] = InstanceDirectory(ready=False, onError=reportError)
        self._subscription: ServiceSubscription | None = None
        self._starting: asyncio.Future[None] | None = None
        self._closed = False
        self._bound = bound
        self._revision = 0

    def observe(self, handler: Callable[[Any, Context], Any]) -> Callable[[], None]:
        if self._closed:
            raise RuntimeError("Remote keyed service binding is closed")
        stopped = False

        def observeInstance(service: Any, context: Context) -> Any:
            slot = ServiceSlot(self._service.id, True)
            slot.bind(service)

            def assertAccess() -> None:
                self._assertAccess()
                signal = context.abortSignal
                if stopped or (signal is not None and signal.aborted):
                    raise RemoteServiceError(
                        "service_stale_instance", f"Remote service {self._service.id} observation is closed"
                    )

            return handler(slot.view(assertAccess), context)

        stop = self._instances.observe(observeInstance)
        if self._bound and self._starting is None:
            revision = self._revision
            starting = startTask(self._start(revision))
            self._starting = starting

            def started(done: asyncio.Future[None]) -> None:
                if done.cancelled():
                    return
                error = done.exception()
                if error is not None and not self._closed and self._revision == revision and self._bound:
                    self._reportError(toError(error))

            starting.add_done_callback(started)

        def stopObserving() -> None:
            nonlocal stopped
            if stopped:
                return
            stopped = True
            stop()
            if self._instances.observerCount == 0:
                self._onEmpty()

        return stopObserving

    async def rebind(self, bound: bool, context: Context) -> None:
        if self._closed:
            return
        self._bound = bound
        self._revision += 1
        revision = self._revision
        await self._reset(context, False)
        if self._closed or self._revision != revision or self._bound != bound:
            return
        if bound and self._instances.observerCount > 0:
            starting = startTask(self._start(revision))
            self._starting = starting
            await asyncio.shield(starting)

    def ready(self) -> asyncio.Future[None]:
        # Shielded: a waiter that gives up must not cancel the start every other waiter shares.
        if self._starting is not None:
            return asyncio.shield(self._starting)
        return resolvedPromise(None)

    @eager
    async def close(self, context: Context) -> None:
        if self._closed:
            return
        self._closed = True
        self._revision += 1
        await self._reset(context, True)
        self._instances.dispose()

    async def _reset(self, context: Context, waitForStarting: bool) -> None:
        self._instances.reset()
        starting = self._starting
        self._starting = None
        subscription = self._subscription
        self._subscription = None

        pending: list[Awaitable[Any]] = []
        if waitForStarting and starting is not None:
            pending.append(_swallow(starting))
        if subscription is not None:
            # Called now, not on a later turn: the transport closes before anything else gets to run.
            pending.append(_settle(subscription.close(context)))
        await _all(pending)

    async def _start(self, revision: int) -> None:
        def onUpdate(update: ServiceProviderUpdate, context: Context) -> None:
            if self._revision == revision:
                self._update(update, context)

        subscription = await _resumed(
            self._transport.subscribe(self._service.id, "keyed", onUpdate, BACKGROUND_CONTEXT)
        )
        if self._closed or not self._bound or self._revision != revision:
            await _settle(subscription.close(BACKGROUND_CONTEXT))
            return
        self._subscription = subscription
        snapshot = subscription.snapshot
        if snapshot["mode"] != "keyed" or snapshot["serviceId"] != self._service.id:
            raise RuntimeError(f"Remote service {self._service.id} returned the wrong keyed snapshot")
        for instanceSnapshot in snapshot["instances"]:
            self._spawn(instanceSnapshot, serviceDeliveryContext())
        subscription.activate()
        self._instances.ready()

    def _update(self, update: ServiceProviderUpdate, context: Context) -> None:
        if self._closed:
            return
        try:
            match update["type"]:
                case "reset":
                    snapshot = update["snapshot"]
                    validateResetSnapshot(snapshot, self._service.id, "keyed")
                    snapshots = {item["instance"]["key"]: item for item in snapshot["instances"]}
                    for instance in list(self._instances.values()):
                        candidate = snapshots.get(instance.key)
                        generation = None if candidate is None else candidate["instance"]["generation"]
                        if generation != instance.generation:
                            self._instances.remove(instance)
                    for item in snapshots.values():
                        existing = self._instances.get(item["instance"]["key"])
                        if existing is None:
                            self._spawn(item, context)
                        else:
                            existing.facade.install(item, context)
                case "unavailable" | "replaced":
                    raise RuntimeError("Keyed service received a singleton lifecycle update")
                case "spawned":
                    self._spawn(update["instance"], context)
                case "closed":
                    closing = self._instances.get(update["instance"]["key"])
                    if closing is not None and closing.generation == update["instance"]["generation"]:
                        self._instances.remove(closing)
                case "state":
                    address = update.get("instance")
                    if address is None:
                        raise RuntimeError("Keyed state update has no instance address")
                    target = self._instances.get(address["key"])
                    if target is None or target.generation != address["generation"]:
                        return
                    target.facade.update(update["member"], update["sequence"], update["ops"], context)
        except Exception as error:  # noqa: BLE001 - reported, as upstream's catch does
            self._reportError(toError(error))

    def _spawn(self, snapshot: ServiceInstanceSnapshot, context: Context) -> None:
        address = snapshot.get("instance")
        if address is None:
            raise RuntimeError("Keyed service instance snapshot has no address")
        active = True

        def isActive() -> bool:
            return active and not self._closed

        facade = ServiceFacade(self._service.id, address, self._transport, isActive, self._assertAccess, self._reportError)
        facade.install(snapshot, context)

        def deactivate() -> None:
            nonlocal active
            active = False
            facade.clear()

        self._instances.replace(KeyedInstance(address["key"], address["generation"], facade.proxy, facade, deactivate))


class RemoteServiceBindingImpl:
    def __init__(self, options: RemoteServiceBindingOptions) -> None:
        self._transport = options["transport"]
        ids = [_serviceId(service) for service in options["services"]]
        if len(set(ids)) != len(ids):
            raise TypeError("Remote service binding has duplicate service IDs")
        self._allowlist = set(ids)
        self._reportError: ErrorReporter = options.get("onError") or (lambda error: None)
        self._assertAccess: Callable[[], None] = options.get("assertAccess") or (lambda: None)
        self._bound = options.get("bound", True)
        self._modes: dict[str, str] = {}
        self._singletons: dict[str, SingletonBinding] = {}
        self._keyed: dict[str, KeyedBinding] = {}
        self._readinessRevision = 0
        self._bindingTransition: asyncio.Future[None] | None = None
        self._disposed = False

    def use(self, service: Any) -> Any:
        self._assertRemotable(service)
        self._assertAvailable(service.id, "singleton")
        binding = self._singletons.get(service.id)
        if binding is not None:
            return binding.facade.proxy  # type: ignore[union-attr]
        binding = SingletonBinding()
        current = binding
        binding.facade = ServiceFacade(
            service.id,
            None,
            self._transport,
            lambda: current.active and not self._disposed and self._bound,
            self._assertHandleAccess,
            self._reportError,
        )
        self._singletons[service.id] = binding
        self._readinessRevision += 1
        if self._bound:
            revision = binding.revision
            starting = startTask(self._startSingleton(service.id, binding, revision))
            binding.starting = starting

            def started(done: asyncio.Future[None]) -> None:
                if done.cancelled():
                    return
                error = done.exception()
                if (
                    error is not None
                    and current.active
                    and current.revision == revision
                    and not self._disposed
                    and self._bound
                ):
                    self._reportError(toError(error))

            starting.add_done_callback(started)
        return binding.facade.proxy

    def observe(self, service: Any, handler: Callable[[Any, Context], Any]) -> Callable[[], None]:
        self._assertRemotable(service)
        self._assertAvailable(service.id, "keyed")
        binding = self._keyed.get(service.id)
        if binding is None:

            def onEmpty() -> None:
                if self._keyed.get(service.id) is not created:
                    return
                del self._keyed[service.id]
                self._readinessRevision += 1
                closing = startTask(created.close(BACKGROUND_CONTEXT))
                closing.add_done_callback(self._reportClose)

            created = KeyedBinding(
                service, self._transport, self._reportError, self._assertHandleAccess, onEmpty, self._bound
            )
            binding = created
            self._keyed[service.id] = binding
            self._readinessRevision += 1
        return binding.observe(handler)

    def _reportClose(self, done: asyncio.Future[None]) -> None:
        if not done.cancelled() and (error := done.exception()) is not None:
            self._reportError(toError(error))

    async def ready(self, context: Context) -> None:
        if self._disposed:
            raise RuntimeError("Remote service binding is disposed")
        while True:
            revision = self._readinessRevision
            starts: list[Awaitable[Any]] = []
            if self._bindingTransition is not None:
                starts.append(self._bindingTransition)
            starts.extend(binding.starting for binding in self._singletons.values() if binding.starting is not None)
            starts.extend(binding.ready() for binding in self._keyed.values())
            await awaitWithContext(_all(starts), context)
            if self._disposed:
                raise RuntimeError("Remote service binding is disposed")
            if revision == self._readinessRevision:
                return

    async def rebind(self, bound: bool, context: Context) -> None:
        if self._disposed:
            raise RuntimeError("Remote service binding is disposed")
        self._bound = bound
        self._readinessRevision += 1
        transitions: list[Awaitable[Any]] = []
        for serviceId, binding in list(self._singletons.items()):
            binding.revision += 1
            assert binding.facade is not None
            binding.facade.clear()
            subscription = binding.subscription
            binding.subscription = None
            starting = startTask(
                self._restartSingleton(serviceId, binding, binding.revision, subscription, bound, context)
            )
            binding.starting = starting
            transitions.append(starting)
        for keyed in list(self._keyed.values()):
            transitions.append(startTask(keyed.rebind(bound, context)))

        async def settleAll() -> None:
            results = await asyncio.gather(*(asyncio.shield(transition) for transition in transitions), return_exceptions=True)
            errors = [result for result in results if isinstance(result, BaseException)]
            if errors:
                raise ExceptionGroup("Failed to rebind services", [toError(error) for error in errors])

        completion = startTask(settleAll())
        self._bindingTransition = completion
        await completion

    async def _restartSingleton(
        self,
        serviceId: str,
        binding: SingletonBinding,
        revision: int,
        subscription: ServiceSubscription | None,
        bound: bool,
        context: Context,
    ) -> None:
        await _settle(None if subscription is None else subscription.close(context))
        if bound:
            await self._startSingleton(serviceId, binding, revision)

    async def dispose(self, context: Context) -> None:
        if self._disposed:
            return
        self._disposed = True
        closes: list[Awaitable[Any]] = []
        for binding in self._singletons.values():
            binding.active = False
            assert binding.facade is not None
            binding.facade.clear()
            if binding.starting is not None:
                closes.append(_swallow(binding.starting))
            if binding.subscription is not None:
                closes.append(_settle(binding.subscription.close(context)))
        for keyed in self._keyed.values():
            closes.append(keyed.close(context))
        self._singletons.clear()
        self._keyed.clear()
        results = await asyncio.gather(*closes, return_exceptions=True)
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise ExceptionGroup("Failed to dispose services", [toError(error) for error in errors])

    async def _startSingleton(self, serviceId: str, binding: SingletonBinding, revision: int) -> None:
        def onUpdate(update: ServiceProviderUpdate, context: Context) -> None:
            if not binding.active or binding.revision != revision:
                return
            facade = binding.facade
            assert facade is not None
            try:
                if update["type"] == "reset":
                    validateResetSnapshot(update["snapshot"], serviceId, "singleton")
                    instances = update["snapshot"]["instances"]
                    if len(instances) == 0:
                        facade.clear()
                    else:
                        facade.install(instances[0], context)
                elif update["type"] == "unavailable":
                    facade.clear()
                elif update["type"] == "replaced":
                    if update["snapshot"].get("instance") is not None:
                        raise RuntimeError("Singleton replacement has an instance address")
                    facade.install(update["snapshot"], context)
                elif update["type"] == "state" and update.get("instance") is None:
                    facade.update(update["member"], update["sequence"], update["ops"], context)
            except Exception as error:  # noqa: BLE001 - reported, as upstream's catch does
                self._reportError(toError(error))

        subscription = await _resumed(
            self._transport.subscribe(serviceId, "singleton", onUpdate, BACKGROUND_CONTEXT)
        )
        if not binding.active or self._disposed or not self._bound or binding.revision != revision:
            await _settle(subscription.close(BACKGROUND_CONTEXT))
            return
        binding.subscription = subscription
        snapshot = subscription.snapshot
        if snapshot["mode"] != "singleton" or snapshot["serviceId"] != serviceId or len(snapshot["instances"]) != 1:
            raise RuntimeError(f"Remote service {serviceId} returned an invalid singleton snapshot")
        assert binding.facade is not None
        binding.facade.install(snapshot["instances"][0], serviceDeliveryContext())
        subscription.activate()

    def _assertHandleAccess(self) -> None:
        if self._disposed:
            raise RuntimeError("Remote service binding is disposed")
        self._assertAccess()

    def _assertRemotable(self, service: Any) -> None:
        if getattr(service, "local", False) is True:
            raise RemoteServiceError("service_not_allowed", f"Service {service.id} is process-local")

    def _assertAvailable(self, serviceId: str, mode: str) -> None:
        if self._disposed:
            raise RuntimeError("Remote service binding is disposed")
        if serviceId not in self._allowlist:
            raise RemoteServiceError("service_not_allowed", f"Remote service {serviceId} is not allowlisted")
        existing = self._modes.get(serviceId)
        if existing is not None and existing != mode:
            raise RemoteServiceError(
                "service_mode_mismatch", f"Remote service {serviceId} is already used as {existing}"
            )
        self._modes[serviceId] = mode


async def _all(awaitables: Sequence[Awaitable[Any]]) -> None:
    """``Promise.all(...).then(() => undefined)``. ``Promise.all`` settles on a later turn, after handlers already
    attached to the promises it joins have run; ``asyncio.gather`` of finished tasks settles at once, so the turn
    is yielded first."""
    await asyncio.sleep(0)
    if awaitables:
        # Shielded: ``Promise.all`` cannot cancel what it joins, so a waiter that is cancelled leaves the joined work running.
        await asyncio.gather(*(asyncio.shield(awaitable) for awaitable in awaitables))


async def _swallow(awaitable: Awaitable[Any]) -> None:
    """``promise.catch(() => {})``."""
    try:
        await asyncio.shield(awaitable)
    except Exception:  # noqa: BLE001, S110
        pass
    except asyncio.CancelledError:
        # A promise that was cancelled is settled for ``.catch(() => {})``; this task's own cancellation is not.
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise


def _serviceId(service: Any) -> str:
    return service["id"] if isinstance(service, dict) else service.id


def validateResetSnapshot(snapshot: ServiceSubscriptionSnapshot, serviceId: str, mode: str) -> None:
    if (
        snapshot["serviceId"] != serviceId
        or snapshot["mode"] != mode
        or (mode == "singleton" and len(snapshot["instances"]) > 1)
    ):
        raise RuntimeError("Remote service reset has the wrong service or mode")
    keys: set[str] = set()
    for instance in snapshot["instances"]:
        hasAddress = instance.get("instance") is not None
        if hasAddress if mode == "singleton" else not hasAddress:
            raise RuntimeError("Remote service reset has an invalid instance address")
        if hasAddress:
            key = instance["instance"]["key"]  # type: ignore[typeddict-item]
            if key in keys:
                raise RuntimeError("Remote service reset repeats an instance key")
            keys.add(key)
        for member in instance["members"]:
            if member["kind"] == "state" and (len(member["ops"]) != 1 or member["ops"][0][0] != "r"):  # type: ignore[typeddict-item]
                raise RuntimeError("Remote service reset must contain full root replacements")


def validateMembers(members: Sequence[ServiceMemberSnapshot]) -> dict[str, ServiceMemberSnapshot]:
    result: dict[str, ServiceMemberSnapshot] = {}
    for member in members:
        if len(member["name"]) == 0 or member["name"] in result:
            raise RuntimeError("Remote service has invalid member descriptions")
        result[member["name"]] = member
    return result


def sameAddress(left: ServiceInstanceAddress | None, right: ServiceInstanceAddress | None) -> bool:
    if left is None or right is None:
        return left is right
    return left["key"] == right["key"] and left["generation"] == right["generation"]


def isContext(value: Any) -> bool:
    # MISAKA fork: a `Context` instance, where upstream accepts any object with `value` and `toString` functions.
    return isinstance(value, Context)


def toError(error: BaseException) -> Exception:
    return error if isinstance(error, Exception) else Exception(str(error))


__all__ = ["RemoteServiceBindingImpl"]
