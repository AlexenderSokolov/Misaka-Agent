"""The Session kernel: one mutation line, the loaded document tracker cache, and committed publication.

Translated from pi's ``packages/durable/src/session/session.ts``. Public async methods are ``@eager``: calling them
starts the body, as calling a JavaScript async function does, and returns the future of the result. The overloaded
document accessors take their arguments positionally: the owner ID of the token's scope (none for a session
document), the family key, and finally the context (``snapshotAsOf`` takes the entry ``at`` before it).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from misaka.chord import replicatedState
from misaka.chord.context import Context, awaitWithContext, withoutAbortSignal
from misaka.chord.delta import Op, track
from misaka.utils.js import (
    UNDEFINED,
    DOMException,
    awaitValue,
    eager,
    jsIsSafeInteger,
    rejectedPromise,
    startTask,
)

from ..documents import (
    AnyDocDefinition,
    AnyDocToken,
    checkRecordScope,
    checkRecordVersion,
    materializeDocument,
    resolveAddress,
)
from ..errors import StorageRejected
from ..types import ConversationId, Storage
from .observation import RETIREMENT_OPERATIONS, CommittedStateSource, CommittedWatch
from .transaction import LoadedDocument, Transaction, TransactionHost


def createSession(storage: Storage, options: dict[str, Any] | None = None) -> SessionImpl:
    """Open a Session kernel over one storage backend. ``now`` is the wall clock for task times; default ``Date.now``."""
    return SessionImpl(storage, None if options is None else options.get("now"))


class SessionImpl:
    """Session kernel: one mutation line, the loaded document tracker cache, and committed publication.

    Only committed state is observable. Every commit callback, preparation, Storage settlement, adoption, and
    publication enqueue runs while the line is held; listeners run later."""

    def __init__(self, storage: Storage, now: Callable[[], float] | None = None) -> None:
        self._storage = storage
        self._documents: dict[str, LoadedDocument] = {}
        # Sets, as upstream's: a listener subscribed twice is one listener, and one unsubscribe removes it. A dict keeps
        # the insertion order a JavaScript Set iterates in.
        self._commitListeners: dict[Callable[[dict[str, Any], Context], None], None] = {}
        self._closeListeners: dict[Callable[[], None], None] = {}
        self._tail: asyncio.Future[None] | None = None
        self._closing: asyncio.Future[None] | None = None
        self._poison: dict[str, Any] | None = None

        def cached(id: str) -> LoadedDocument | None:
            return self._documents.get(id)

        def install(document: LoadedDocument) -> None:
            self._documents[document.addressId] = document

        def evict(id: str, recordId: int) -> None:
            current = self._documents.get(id)
            if current is not None and current.record["id"] == recordId:
                del self._documents[id]

        self._host = TransactionHost(
            storage,
            (lambda: int(time.time() * 1000)) if now is None else now,
            cached,
            lambda definition, addressId, address, context: self._loadDocument(definition, addressId, address, context),
            install,
            evict,
            lambda tx, record: self.conversationCreated(tx, record),
        )

    def commit(self, change: Callable[[Transaction], Any], context: Context) -> asyncio.Future[Any]:
        return self.commitWith(change, context)

    def commitWith(
        self, change: Callable[[Transaction], Any], context: Context, scope: dict[str, Any] | None = None
    ) -> asyncio.Future[Any]:
        """Internal commit exposing the concrete transaction and its internal operations, such as the reserved-ID root
        bootstrap and task replacement. ``scope`` sets the default ``tx.createTask()`` conversation and the task
        attributed to appended entries."""
        try:
            self._assertUsable()
        except Exception as error:  # noqa: BLE001
            return rejectedPromise(error)
        return self._enqueue(lambda: self._runCommit(change, context, scope))

    def readOnLine(self, job: Callable[[], Awaitable[Any]]) -> asyncio.Future[Any]:
        """Internal: run a read-only job on the mutation line so multi-read derivations observe one committed state."""
        try:
            self._assertUsable()
        except Exception as error:  # noqa: BLE001
            return rejectedPromise(error)

        async def run() -> Any:
            self._assertHealthy()
            return await job()

        return self._enqueue(run)

    async def conversationDocumentOnLine(
        self, token: AnyDocToken, conversationId: ConversationId, context: Context
    ) -> dict[str, Any] | None:
        """Internal: a conversation document's current incarnation and value, for a job already running on the line (see
        ``readOnLine()``). Absent documents are ``None``."""
        definition = token["definition"]
        resolved = resolveAddress(definition, [conversationId, context])
        loaded = await self._loadDocument(definition, resolved.id, resolved.address, context)
        if loaded is None:
            return None
        checkRecordScope(definition, loaded.record)
        checkRecordVersion(definition, loaded.record, loaded.storedVersion)
        return {"record": loaded.record, "version": loaded.valueVersion, "value": loaded.tracker.value}

    @eager
    async def snapshot(self, token: AnyDocToken, *args: Any) -> Any:
        self._assertUsable()
        definition = token["definition"]
        resolved = resolveAddress(definition, args)
        context = args[resolved.nextArgument]
        cached = self._documents.get(resolved.id)
        if cached is not None and cached.valueVersion == definition["version"]:
            loaded: LoadedDocument | None = cached
        else:

            async def load() -> LoadedDocument | None:
                self._assertHealthy()
                return await self._loadDocument(definition, resolved.id, resolved.address, context)

            loaded = await self._enqueue(load)
        if loaded is None:
            return None
        checkRecordScope(definition, loaded.record)
        checkRecordVersion(definition, loaded.record, loaded.storedVersion)
        return loaded.tracker.value

    def documentState(self, token: AnyDocToken, *args: Any) -> asyncio.Future[Any]:
        try:
            self._assertUsable()
            definition = token["definition"]
            resolved = resolveAddress(definition, args)
            context = args[resolved.nextArgument]

            async def attach() -> Any:
                self._assertHealthy()
                loaded = await self._loadDocument(definition, resolved.id, resolved.address, context)
                if loaded is None:
                    return None
                source, detach = self._attachDocument(
                    definition, loaded, lambda value, release: CommittedStateSource(value, release)
                )
                try:
                    return replicatedState(source)
                except Exception:
                    detach()
                    raise

            return self._enqueue(attach)
        except Exception as error:  # noqa: BLE001
            return rejectedPromise(error)

    @eager
    async def watchDoc(self, token: AnyDocToken, *args: Any) -> Any:
        self._assertUsable()
        definition = token["definition"]
        resolved = resolveAddress(definition, args)
        context = args[resolved.nextArgument]
        signal = context.abortSignal
        cancelled = bool(signal.aborted) if signal is not None else False

        def markCancelled() -> None:
            nonlocal cancelled
            cancelled = True

        if signal is not None:
            signal.addEventListener("abort", markCancelled, once=True)
        try:

            async def attach() -> Any:
                self._assertHealthy()
                if cancelled:
                    raise cancellationError(signal)
                loaded = await self._loadDocument(definition, resolved.id, resolved.address, context)
                if cancelled:
                    raise cancellationError(signal)
                if loaded is None:
                    return None
                return self._attachDocument(
                    definition, loaded, lambda value, release: CommittedWatch(value, release)
                )[0]

            watch = await self._enqueue(attach)
            if watch is None:
                return None
            if cancelled:
                watch.cancel()
                raise cancellationError(signal)
            if signal is not None:
                watch.observeCancellation(signal)
            return watch
        finally:
            if signal is not None:
                signal.removeEventListener("abort", markCancelled)

    @eager
    async def snapshotAsOf(self, token: AnyDocToken, *args: Any) -> Any:
        self._assertUsable()
        definition = token["definition"]
        resolved = resolveAddress(definition, args)
        if resolved.address["scope"]["kind"] != "conversation":
            raise TypeError("Session.snapshotAsOf() requires a conversation document")
        conversationId = resolved.address["scope"]["conversationId"]
        at = args[resolved.nextArgument]
        if not jsIsSafeInteger(at):
            raise TypeError("Session.snapshotAsOf() requires an entry ID")
        at = int(at)
        context = args[resolved.nextArgument + 1]

        async def read() -> Any:
            self._assertHealthy()
            storedEntry = await self._storage.entry(conversationId, at, context)
            if storedEntry is None:
                raise RuntimeError(f"Entry {at} is not visible from conversation {conversationId}")
            address = {
                **resolved.address,
                "scope": {"kind": "conversation", "conversationId": storedEntry["entry"]["conversationId"]},
            }
            record = await self._storage.findDocument(address, storedEntry["commitSeq"], context)  # type: ignore[arg-type]
            if record is None:
                return None
            stored = await self._storage.document(record["id"], storedEntry["commitSeq"], context)
            if stored is None:
                raise RuntimeError(f"Historical document {record['id']} ({record['kind']}) cannot be read")
            return materializeDocument(definition, stored)

        return await self._enqueue(read)

    def close(self, context: Context) -> asyncio.Future[None]:
        if self._closing is None:
            cleanup = withoutAbortSignal(context)
            # Seal admission before anything else runs, then stop observers; admitted work settles before Storage closes.
            self._closing = startTask(self._closeSequence(cleanup))
            listeners = list(self._closeListeners)
            self._closeListeners.clear()
            for listener in listeners:
                listener()
        # Shielded: a caller that gives up waiting must not cancel the close the others share.
        return startTask(awaitWithContext(asyncio.shield(self._closing), context))

    async def _closeSequence(self, cleanup: Context) -> None:
        await asyncio.sleep(0)
        await self.beforeClose()

        async def shutdown() -> None:
            self._commitListeners.clear()
            self._documents.clear()
            await self._storage.close(cleanup)

        await self._enqueue(shutdown)

    def conversationCreated(self, tx: Transaction, record: dict[str, Any]) -> Awaitable[None]:
        """Runs inside every transaction that creates or forks a conversation, after the conversation record is staged.
        A plain Session stages nothing; a Harness stages its built-in documents."""

        async def nothing() -> None:
            pass

        return nothing()

    def beforeClose(self) -> Awaitable[None]:
        """Runs after close seals admission and before the line closes Storage; must not reject."""

        async def nothing() -> None:
            pass

        return nothing()

    def subscribeCommits(self, listener: Callable[[dict[str, Any], Context], None]) -> Callable[[], bool]:
        """Register a synchronous post-adoption listener. It must not raise, block, or call Session operations."""
        self._assertUsable()
        self._commitListeners[listener] = None

        def unsubscribe() -> bool:
            return self._commitListeners.pop(listener, UNDEFINED) is not UNDEFINED

        return unsubscribe

    def subscribeClose(self, listener: Callable[[], None]) -> Callable[[], bool]:
        """Register a listener called synchronously when close begins. It must not raise, block, or call Session
        operations."""
        self._assertUsable()
        self._closeListeners[listener] = None

        def unsubscribe() -> bool:
            return self._closeListeners.pop(listener, UNDEFINED) is not UNDEFINED

        return unsubscribe

    def unloadDocuments(self) -> asyncio.Future[None]:
        """Drop every loaded tracker on the mutation line; later access cold-loads from Storage."""

        async def unload() -> None:
            self._documents.clear()

        return self._enqueue(unload)

    async def _runCommit(
        self, change: Callable[[Transaction], Any], context: Context, scope: dict[str, Any] | None
    ) -> Any:
        self._assertHealthy()
        if context.abortSignal is not None:
            context.abortSignal.throwIfAborted()
        tx = Transaction(self._host, context, scope)
        try:
            result = await awaitValue(change(tx))
        except BaseException:
            await tx.settleFailure()
            raise
        writes = await tx.settleSuccess()
        if len(writes) == 0:
            tx.discard()
            return result
        try:
            # Once admitted, caller cancellation does not interrupt Storage settlement.
            seq = await self._storage.commit(writes, withoutAbortSignal(context))
        except BaseException as error:
            tx.discard()
            # Callback errors never reach this branch; StorageRejected alone guarantees that no batch effect committed.
            if not isinstance(error, StorageRejected):
                self._poison = {"error": error}
            raise
        try:
            documents = tx.adopt(seq)
        except BaseException as error:
            # Storage already committed; a failed adoption leaves memory behind durable state.
            self._poison = {"error": error}
            raise
        self._publish(seq, writes, documents, context)
        return result

    def _publish(self, seq: int, writes: list[dict[str, Any]], documents: list[dict[str, Any]], context: Context) -> None:
        if not self._commitListeners:
            return
        changes: list[dict[str, Any]] = []
        for write in writes:
            if write["type"] in ("conversation", "entry", "task", "submission"):
                changes.append(write)
        changes.extend(documents)
        publication = {"seq": seq, "changes": changes}
        for listener in list(self._commitListeners):
            listener(publication, context)

    def _attachDocument(
        self, definition: AnyDocDefinition, loaded: LoadedDocument, create: Callable[[Any, Callable[[], None]], Any]
    ) -> tuple[Any, Callable[[], None]]:
        """Attach an observer to one committed incarnation: check the definition, then forward this incarnation's
        committed changes and close. ``detach`` removes both subscriptions."""
        checkRecordScope(definition, loaded.record)
        checkRecordVersion(definition, loaded.record, loaded.storedVersion)
        unsubscribe: dict[str, Callable[[], None]] = {"commit": lambda: None, "close": lambda: None}

        def detach() -> None:
            unsubscribe["commit"]()
            unsubscribe["close"]()

        observer = create(loaded.tracker.value, detach)
        observed = {"version": loaded.valueVersion}

        def onCommit(publication: dict[str, Any], context: Context) -> None:
            for change in publication["changes"]:
                if change["type"] != "document" or change["record"]["id"] != loaded.record["id"]:
                    continue
                # A document state's frames carry no caller cancellation; a watch observes its own cancellation.
                frameContext = withoutAbortSignal(context) if isinstance(observer, CommittedStateSource) else context
                ops = observedOperations(observed, change)
                # A migration-only base changes nothing for an observer of the new version.
                if len(ops) == 0:
                    continue
                observer.advance(change["value"], ops, frameContext)

        unsubscribe["commit"] = self.subscribeCommits(onCommit)
        unsubscribe["close"] = self.subscribeClose(lambda: observer.closeSession())
        return observer, detach

    async def _loadDocument(
        self, definition: AnyDocDefinition, addressId: str, address: dict[str, Any], context: Context
    ) -> LoadedDocument | None:
        cached = self._documents.get(addressId)
        # A tracker serves only tokens of the version its value was materialized for; others reload from Storage.
        if cached is not None and cached.valueVersion == definition["version"]:
            return cached
        if cached is not None:
            del self._documents[addressId]
        record = await self._storage.findDocument(address, "current", context)  # type: ignore[arg-type]
        if record is None:
            return None
        stored = await self._storage.document(record["id"], "current", context)
        if stored is None:
            raise RuntimeError(f"Current document {record['id']} ({record['kind']}) cannot be read")
        value = materializeDocument(definition, stored)
        loaded = LoadedDocument(
            addressId, stored["record"], stored["version"], definition["version"], stored["deltasSinceBase"], track(value)
        )
        self._documents[addressId] = loaded
        return loaded

    def _enqueue(self, job: Callable[[], Awaitable[Any]]) -> asyncio.Future[Any]:
        run = startTask(self._runAfter(self._tail, job))
        self._tail = startTask(_settle(run))
        # Shielded: a promise cannot be cancelled by whoever waits on it, so a caller that stops waiting leaves the job
        # on the line, and every job behind it, running.
        return asyncio.shield(run)

    @staticmethod
    async def _runAfter(tail: asyncio.Future[None] | None, job: Callable[[], Awaitable[Any]]) -> Any:
        # `.then(job)` never runs the job inline.
        await asyncio.sleep(0)
        if tail is not None:
            await asyncio.shield(tail)
        return await job()

    def _assertUsable(self) -> None:
        if self._closing is not None:
            raise RuntimeError("Session is closed")
        self._assertHealthy()

    def _assertHealthy(self) -> None:
        if self._poison is not None:
            error = RuntimeError("Session is poisoned by a failed commit after storage admission; reopen it")
            error.__cause__ = self._poison["error"]
            raise error


def observedOperations(observed: dict[str, int], change: dict[str, Any]) -> list[Op]:
    """Operations an observer applies for one committed change. An observer hydrated under another definition version
    holds a differently shaped value, so it receives the new value as a root replacement instead of operations for that
    shape."""
    if change["value"] is None:
        return list(RETIREMENT_OPERATIONS)
    if change["version"] == observed["version"]:
        return change["ops"]
    observed["version"] = change["version"]
    return [["r", change["value"]]]


def cancellationError(signal: Any) -> Exception:
    reason = signal.reason
    return reason if isinstance(reason, Exception) else DOMException("The operation was aborted", "AbortError")


async def _settle(future: asyncio.Future[Any]) -> None:
    """A future that finishes when ``future`` settles, whatever it settles with: ``run.then(() => undefined, () => undefined)``."""
    try:
        await asyncio.shield(future)
    except Exception:  # noqa: BLE001, S110
        pass
    except asyncio.CancelledError:
        # A job that ended cancelled is settled for the line; this task's own cancellation is not.
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise


__all__ = ["SessionImpl", "createSession"]
