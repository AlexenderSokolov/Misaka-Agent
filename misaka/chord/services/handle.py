"""Host-owned mutable service targets with consumer-owned guarded views.

Translated from pi's ``packages/chord/src/services/handle.ts``. A view is a dynamic object, not a snapshot:
every member read resolves against the implementation bound at that moment, so a handle taken before a reload
keeps working after it. Upstream builds the views from ``Proxy`` objects; here they are small classes with
``__getattr__`` and ``__call__``.

An implementation is a mapping (member name to member) or an object whose attributes are the members.
Names with two leading underscores are never members, so protocol probes (``__await__``, ``__iter__``) of
the Python runtime and of frameworks do not reach the implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from misaka.utils.js import UNDEFINED


class AttributeMapping:
    """A mapping read by attribute, so a service implementation given as a mapping is used the way an
    object is: ``implementation.read(context)``."""

    __slots__ = ("_AttributeMapping__target",)

    def __init__(self, target: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_AttributeMapping__target", target)

    def __getattr__(self, name: str) -> Any:
        target = self._AttributeMapping__target
        if _isDunder(name) or name not in target:
            raise AttributeError(name)
        return target[name]

    def __getitem__(self, name: str) -> Any:
        return self._AttributeMapping__target[name]

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("A service implementation is read-only through this view")


class _Resolved:
    __slots__ = ("receiver", "value")

    def __init__(self, value: Any, receiver: Any) -> None:
        self.value = value
        self.receiver = receiver


type ValueResolver = Callable[[], _Resolved]


def readMember(target: Any, name: str) -> Any:
    """``Reflect.get(target, name)``: ``UNDEFINED`` for a member the target does not have."""
    if isinstance(target, Mapping):
        return target.get(name, UNDEFINED)
    return getattr(target, name, UNDEFINED)


def isObject(value: Any) -> bool:
    """A JavaScript object or function: anything but a primitive."""
    return not (value is None or value is UNDEFINED or isinstance(value, str | bytes | int | float | bool))


def _isDunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


class ServiceSlot:
    """Host-owned mutable target with consumer-owned guarded views."""

    def __init__(self, serviceId: str, wrapObjects: bool) -> None:
        self._serviceId = serviceId
        self._wrapObjects = wrapObjects
        self._implementation: Any = None

    def view(self, assertAccess: Callable[[], None]) -> Any:
        return _ServiceViewProxy(_ServiceView(self, assertAccess, self._wrapObjects))

    def bind(self, implementation: Any) -> None:
        self._implementation = implementation

    def unbind(self) -> None:
        self._implementation = None

    def resolve(self, name: str, assertAccess: Callable[[], None]) -> _Resolved:
        assertAccess()
        implementation = self._implementation
        if implementation is None:
            raise RuntimeError(f"Service {self._serviceId} is disconnected")
        return _Resolved(readMember(implementation, name), implementation)


class _ServiceView:
    def __init__(self, slot: ServiceSlot, assertAccess: Callable[[], None], wrapObjects: bool) -> None:
        self._slot = slot
        self._assertAccess = assertAccess
        self._wrapObjects = wrapObjects
        self._members: dict[str, ValueView] = {}

    def getMember(self, name: str) -> Any:
        def resolve() -> _Resolved:
            return self._slot.resolve(name, self._assertAccess)

        current = resolve().value
        if not isObject(current) or (not callable(current) and not self._wrapObjects):
            return current
        member = self._members.get(name)
        if member is None:
            member = ValueView(resolve)
            self._members[name] = member
        return member


class _ServiceViewProxy:
    __slots__ = ("_ServiceViewProxy__view",)

    def __init__(self, view: _ServiceView) -> None:
        object.__setattr__(self, "_ServiceViewProxy__view", view)

    def __getattr__(self, name: str) -> Any:
        if _isDunder(name):
            raise AttributeError(name)
        return self._ServiceViewProxy__view.getMember(name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("A service handle is read-only")


class ValueView:
    """A guarded view of one member: calling it calls the member currently bound, and reading an attribute
    reads the current member's attribute (a callable one as a view again)."""

    __slots__ = ("_children", "_resolve")

    def __init__(self, resolve: ValueResolver) -> None:
        object.__setattr__(self, "_resolve", resolve)
        object.__setattr__(self, "_children", {})

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        resolved = self._resolve()
        if not callable(resolved.value):
            raise TypeError("Service member is not callable")
        return resolved.value(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if _isDunder(name):
            raise AttributeError(name)

        def resolve() -> _Resolved:
            parent = self._resolve()
            if not isObject(parent.value):
                raise TypeError("Service member does not have properties")
            return _Resolved(readMember(parent.value, name), parent.value)

        current = resolve().value
        if not callable(current):
            return current
        child = self._children.get(name)
        if child is None:
            child = ValueView(resolve)
            self._children[name] = child
        return child

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("A service handle is read-only")


__all__ = ["AttributeMapping", "ServiceSlot", "ValueView", "isObject", "readMember"]
