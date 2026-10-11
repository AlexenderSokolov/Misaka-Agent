"""The durable task scheduler of one Harness.

Translated from pi's ``packages/durable/src/harness/scheduler.ts``. Records, states, outcomes, definitions and
resolutions are plain ``dict`` objects with the upstream keys. A task handler is called ``handler(task, runtime,
context)`` with the stored record as ``task`` and a runtime object (attributes ``taskId``, ``conversationId``,
``signal``, ``models``, ``registry``, ``settings`` and ``hooks``; methods ``agent``, ``env``, ``commit``, ``memo``,
``sleep``, ``watchDoc``, ``snapshot``, ``snapshotAsOf``, ``getTask``, ``waitForTask``, ``outcomes``, ``conversation``,
``entry``, ``context``, ``now`` and ``report``). A ``memo(name, context)`` read answers ``UNDEFINED`` for an absent
memo, since a memo may hold ``None``. Public async methods are ``@eager``; the JavaScript ``void`` calls
(``queueMicrotask(() => void this.#reconcile())``) are ``spawn`` tasks the scheduler keeps alive.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Iterator
from typing import Any

from misaka.chord.context import Context, abortError, awaitWithContext, withAbortSignal
from misaka.chord.json import copyJson
from misaka.utils.js import (
    UNDEFINED,
    AbortController,
    AbortSignal,
    DOMException,
    awaitValue,
    eager,
    errorMessage,
    queueMicrotask,
    rejectedPromise,
    resolvedPromise,
    spawn,
    startTask,
)

from ..types import ConversationId, Storage, TaskId
from .agent import agentHooks
from .context import readContextFrom
from .util import Waiters, closedError, scanAll

SCAN_PAGE_SIZE = 256
MAX_TIMER_DELAY = 2_147_483_647
"""Longest delay ``setTimeout`` supports; longer sleeps wait in several steps."""
LIVE_STATUSES = ("pending", "running", "waiting", "completing")


class _Invocation:
    """One in-memory execution of a task in run or abort mode."""

    def __init__(self, record: dict[str, Any], mode: str, controller: AbortController, context: Context) -> None:
        self.taskId = record["id"]
        self.conversationId = record["conversationId"]
        self.mode = mode
        self.controller = controller
        self.context = context  # Context passed to handlers; cancelled by ``controller``.
        self.watches: dict[Any, None] = {}  # Watches acquired through the runtime; stopped at invocation end.
        self.ended = False
        self.done: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    def finish(self) -> None:
        if not self.done.done():
            self.done.set_result(None)


class _Phase:
    """What a runtime reads for the phase handler it serves: the phase's snapshot and task, and its lazily resolved
    agent."""

    def __init__(self, snapshot: Callable[[], Any], task: Callable[[], dict[str, Any]]) -> None:
        self.snapshot = snapshot
        self.task = task
        self.agent: asyncio.Future[Any] | None = None


class _Reservation:
    def __init__(self, invocation: _Invocation, task: dict[str, Any], snapshot: Any) -> None:
        self.invocation = invocation
        self.task = task
        self.snapshot = snapshot


class _PhaseState:
    """The task definition and registry snapshot a run invocation currently uses, and the replacement definition it
    already reported as unable to take over."""

    def __init__(self, task: dict[str, Any], snapshot: Any) -> None:
        self.task = task
        self.snapshot = snapshot
        self.reported: dict[str, Any] | None = None  # ``{"task": definition or None}`` once reported.


class _Hooks:
    """Dispatches one hook of a task to every matching registered handler, in registry order of the phase snapshot."""

    def __init__(self, scheduler: TaskScheduler, invocation: _Invocation, phase: _Phase, agent: Callable[[], Any]) -> None:
        self._scheduler = scheduler
        self._invocation = invocation
        self._phase = phase
        self._agent = agent

    @eager
    async def each(self, name: str, invoke: Callable[[Callable[..., Any]], Any]) -> None:
        """Call ``invoke`` with each matching handler named ``name``. An ordinary throw from ``invoke`` is reported and
        the next handler runs; once the invocation is signalled, the error propagates."""
        for handlers in agentHooks(await self._agent(), self._phase.task()["definition"]["name"]):
            handler = handlers.get(name) if isinstance(handlers, dict) else getattr(handlers, name, None)
            if not callable(handler):
                continue
            try:
                await awaitValue(invoke(handler))
            except Exception as error:
                if self._invocation.controller.signal.aborted:
                    raise
                self._scheduler._report(error)


def _thrownBy(error: BaseException) -> Exception:
    """What a handler threw. A handler that ends with ``CancelledError`` because something it awaited was cancelled has
    thrown, as awaiting a rejected promise does; cancelling the task that runs it is not a failure and propagates."""
    if isinstance(error, Exception):
        return error
    if isinstance(error, asyncio.CancelledError):
        task = asyncio.current_task()
        if task is None or not task.cancelling():
            return DOMException("The operation was aborted", "AbortError")
    raise error


class TaskRuntimeImpl:
    """Operations of one task invocation. Every operation rejects after the invocation ends; watches acquired through it
    stop at invocation end."""

    def __init__(self, scheduler: TaskScheduler, invocation: _Invocation, phase: _Phase) -> None:
        self._scheduler = scheduler
        self._invocation = invocation
        self._phase = phase
        self.taskId = invocation.taskId
        self.conversationId = invocation.conversationId
        self.signal = invocation.controller.signal
        self.models = scheduler._models
        self.hooks = _Hooks(scheduler, invocation, phase, self._sharedAgent)

    @property
    def settings(self) -> Any:
        return self._scheduler._settings()

    @property
    def registry(self) -> Any:
        return self._phase.snapshot()

    def _sharedAgent(self) -> asyncio.Future[Any]:
        """Resolved at most once per phase handler, at first use, with the invocation's context; fixed for the
        phase."""
        phase = self._phase
        invocation = self._invocation
        if invocation.ended:
            return rejectedPromise(endedError(invocation))
        if phase.agent is None:
            phase.agent = startTask(
                self._scheduler._agent(invocation.conversationId, phase.snapshot(), invocation.context)
            )
            # A caller that stops waiting must not leave the shared resolution's failure unobserved.
            phase.agent.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        # Shielded: a caller that stops waiting must not cancel the resolution the others share.
        return asyncio.shield(phase.agent)

    @eager
    async def agent(self, context: Context) -> Any:
        if self._invocation.ended:
            raise endedError(self._invocation)
        return await awaitWithContext(self._sharedAgent(), context)

    def env(self, context: Context) -> asyncio.Future[Any]:
        return self._scheduler._read(
            self._invocation, lambda: self._scheduler._env(self._invocation.conversationId, context)
        )

    def commit(self, change: Callable[..., Any], context: Context) -> asyncio.Future[None]:
        """Commit on the Session line after rereading the task. Rejects when the task is terminal, the invocation
        ended, the Harness is closing, or, in a run invocation, the task carries an abort mark. A returned state
        replaces the task's state in the same commit; returning nothing leaves it unchanged."""

        async def apply(tx: Any, current: dict[str, Any]) -> None:
            following = await awaitValue(change(tx, current))
            if following is not None:
                await self._scheduler._commitState(tx, self._invocation, current, following)

        return self._scheduler._gated(self._invocation, apply, context)

    def memo(self, name: str, *rest: Any) -> asyncio.Future[Any]:
        """``memo(name, context)`` reads a durable memo of this task; ``memo(name, candidate, context)`` stores
        ``candidate`` unless a memo already exists and returns the durable winner."""
        scheduler = self._scheduler
        invocation = self._invocation
        if len(rest) == 1:

            async def read() -> Any:
                return memoOf(scheduler._live.get(invocation.taskId), name)

            return scheduler._read(invocation, read)
        candidate = rest[0]

        def change(tx: Any, current: dict[str, Any]) -> Any:
            winner = memoOf(current, name)
            if winner is not UNDEFINED:
                return winner
            tx.setTask({**current, "memos": {**(current.get("memos") or {}), name: candidate}})
            return candidate

        return scheduler._gated(invocation, change, rest[1])

    def sleep(self, until: float, context: Context) -> asyncio.Future[None]:
        """Resolve once the Harness clock reaches ``until``; rejects when the invocation or ``context`` is cancelled."""
        return startTask(self._scheduler._sleep(self._invocation, until, context))

    def watchDoc(self, *args: Any) -> asyncio.Future[Any]:
        return startTask(self._scheduler._watchDoc(self._invocation, args))

    def snapshot(self, *args: Any) -> asyncio.Future[Any]:
        return self._scheduler._read(self._invocation, lambda: self._scheduler._session.snapshot(*args))

    def snapshotAsOf(self, *args: Any) -> asyncio.Future[Any]:
        return self._scheduler._read(self._invocation, lambda: self._scheduler._session.snapshotAsOf(*args))

    def getTask(self, id: TaskId, context: Context) -> asyncio.Future[Any]:
        scheduler = self._scheduler
        return scheduler._read(
            self._invocation, lambda: scheduler._session.readOnLine(lambda: scheduler._storage.task(id, context))
        )

    def waitForTask(self, id: TaskId, context: Context) -> asyncio.Future[Any]:
        scheduler = self._scheduler
        return scheduler._read(
            self._invocation,
            lambda: scheduler.waitForTask(id, withAbortSignal(self._invocation.controller.signal, context)),
        )

    def outcomes(self, ids: list[TaskId], context: Context) -> asyncio.Future[Any]:
        scheduler = self._scheduler

        async def read() -> list[dict[str, Any]]:
            outcomes: list[dict[str, Any]] = []
            for id in ids:
                record = await scheduler._storage.task(id, context)
                state = None if record is None else record["state"]
                if state is None or state["status"] != "terminal":
                    raise RuntimeError(f"Task {id} is not terminal")
                outcomes.append(state["outcome"])
            return outcomes

        return scheduler._read(self._invocation, lambda: scheduler._session.readOnLine(read))

    def conversation(self, id: ConversationId, context: Context) -> asyncio.Future[Any]:
        invocation = self._invocation
        scheduler = self._scheduler

        def check() -> None:
            if invocation.ended:
                raise endedError(invocation)

        return scheduler._read(
            invocation, lambda: scheduler._conversation(id, {"signal": invocation.controller.signal, "check": check}, context)
        )

    def entry(self, *args: Any) -> asyncio.Future[Any]:
        """``entry(id, context)``, or ``entry(token, id, context)``: ``None`` when the entry is absent, not visible, or
        has another kind."""
        scheduler = self._scheduler
        invocation = self._invocation
        if len(args) == 2:
            token, id, context = None, args[0], args[1]
        else:
            token, id, context = args[0], args[1], args[2]

        async def read() -> Any:
            found = await scheduler._session.readOnLine(
                lambda: scheduler._storage.entry(invocation.conversationId, id, context)
            )
            entry = None if found is None else found["entry"]
            return entry if token is None or (entry is not None and entry["kind"] == token.kind) else None

        return scheduler._read(invocation, read)

    def context(
        self, conversationId: ConversationId, context: Context, options: dict[str, Any] | None = None
    ) -> asyncio.Future[Any]:
        """Committed raw active transcript and model context, optionally cut off at the visible entry ``at``."""
        scheduler = self._scheduler
        invocation = self._invocation
        at = None if options is None else options.get("at")

        async def read() -> Any:
            kept = scheduler._contexts.get(conversationId)
            result = await readContextFrom(
                scheduler._session,
                scheduler._storage,
                conversationId,
                context,
                at,
                None if kept is None else kept["range"],
            )
            range_ = result["range"]
            # Keep it unless the invocation ended or a concurrent read already kept a newer range.
            kept = scheduler._contexts.get(conversationId)
            if (
                range_ is not None
                and not scheduler._closing
                and not invocation.ended
                and (kept is None or kept["range"]["bounds"]["tail"] <= range_["bounds"]["tail"])
            ):
                # A read of another, idle conversation starts or continues its retention period.
                if not scheduler._idle(conversationId):
                    scheduler._contexts[conversationId] = {"range": range_, "idleSince": None}
                elif scheduler._contextRetentionMs() > 0:
                    scheduler._contexts[conversationId] = {
                        "range": range_,
                        "idleSince": scheduler._now() if kept is None or kept["idleSince"] is None else kept["idleSince"],
                    }
                    scheduler._scheduleExpiry()
            return result["view"]

        return scheduler._read(invocation, read)

    def now(self) -> int:
        if self._invocation.ended:
            raise endedError(self._invocation)
        return self._scheduler._now()

    def report(self, error: BaseException) -> None:
        if self._invocation.ended:
            raise endedError(self._invocation)
        self._scheduler._report(error)


class TaskScheduler:
    """Durable task scheduler of one Harness.

    ``_live`` mirrors every committed non-terminal task record: pending, running, waiting, and completing. The
    synchronous commit listener updates it on the Session line, so code running on the line reads exactly the committed
    state from it.

    Tasks and conversations form one ownership tree (spec section 5.5): a task's parent is its owner task, or its
    conversation; a conversation's parent is its owner task, if any. Walks up that tree decide cascades, idle scopes, and
    whether a task's ordinary owned work is live, which holds its outcome as ``completing`` and delays its abort
    handler.

    Invariant: every task transition is decided and written by one callback serialized on the Session line. That covers
    reservation, marks, runtime commits, finalization, and the synchronous step before each phase, which applies the
    precedence rules and writes a fault or handover. Handlers and joins run off the line. An invocation ends inside the
    step that decides its end, so a runtime commit it queued either lands before that decision or is rejected.

    Options (``dict``): ``session``, ``storage``, ``registry``, ``models``, ``agent(conversationId, snapshot, context)``
    (resolve a conversation's agent against a snapshot; the runtime calls it at most once per phase), ``settings()`` (read
    at each access), ``env(conversationId, context)``, ``now()``, ``report(error)``, ``settleOutcome(tx, record,
    outcome)`` (Harness cleanup staged in the commit that makes an outcome the scheduler wrote itself terminal),
    ``withdrawInputs(tx, conversationId)``, ``conversation(id, binding, context)`` (invocation-bound handle of an
    existing conversation; ``binding`` is ``{"signal", "check"}``) and ``context`` (for scheduler commits and
    invocations; carries no caller cancellation).
    """

    def __init__(self, options: dict[str, Any]) -> None:
        self._session = options["session"]
        self._storage: Storage = options["storage"]
        self._registry = options["registry"]
        self._models = options["models"]
        self._agent = options["agent"]
        self._settings = options["settings"]
        self._env = options["env"]
        self._now = options["now"]
        self._report = options["report"]
        self._settleOutcome = options["settleOutcome"]
        self._withdrawInputs = options["withdrawInputs"]
        self._conversation = options["conversation"]
        self._context: Context = options["context"]
        self._live: dict[TaskId, dict[str, Any]] = {}
        # Context range last read through a task runtime, per conversation: a later read, by any of its tasks, scans only
        # newer entries. ``idleSince`` is the Harness time it was first seen idle. Derived and never persisted; dropped at
        # the first idle check after ``settings.contextRetentionMs`` of idleness, and at close.
        self._contexts: dict[ConversationId, dict[str, Any]] = {}
        # Timer for the earliest expiry of an idle context (see ``_scheduleExpiry``): ``{"at", "timer"}``.
        self._expiry: dict[str, Any] | None = None
        self._invocations: dict[TaskId, _Invocation] = {}
        self._taskWaiters = Waiters()
        self._idleWaiters = Waiters()  # Idle waiters by conversation; ``None`` waits for the whole Harness.
        # Definition whose migration failed per task; retried only once the registry resolves another definition.
        self._failedMigrations: dict[TaskId, dict[str, Any]] = {}
        self._edges: dict[ConversationId, TaskId | None] = {}  # Owner task of each loaded conversation.
        self._conversationOwners: set[TaskId] = set()  # Tasks that own a loaded conversation.
        self._settled: dict[TaskId, dict[str, Any]] = {}  # Ownership fields of terminal tasks that walks pass through.
        # ``failFast`` waiters the next reconcile checks for a failed task in ``on``: at open, when they start waiting,
        # and when one of their tasks fails.
        self._failFastChecks: set[TaskId] = set()
        self._reconcileScheduled = False
        self._cascadePending = False
        self._unsubscribeRegistry: Callable[[], None] = lambda: None
        self._enabled = False
        self._closing = False
        self._dirty = False
        self._draining = False
        self._tasks: set[asyncio.Task[Any]] = set()

    @eager
    async def open(self, context: Context) -> None:
        """Load live tasks and change surviving ``running`` tasks back to ``pending``. Dispatches nothing."""
        self._session.subscribeCommits(lambda publication, commitContext: self._observe(publication))
        self._session.subscribeClose(lambda: self._seal())
        self._unsubscribeRegistry = self._registry.subscribe(lambda: self._kick())

        async def load(tx: Any) -> None:
            # Every table read before the first write.
            scans = []
            for status in LIVE_STATUSES:

                def scanStatus(cursor: Any, status: str = status) -> Any:
                    return tx.scanTasks({"status": status}, SCAN_PAGE_SIZE, cursor)

                scans.append(await scanAll(scanStatus))
            for records in scans:
                for record in records:
                    self._live[record["id"]] = record
                    if record["state"]["status"] == "running":
                        tx.setTask(withState(record, {"status": "pending", "checkpoint": record["state"]["checkpoint"]}))
                    if record["state"]["status"] == "waiting" and record["state"]["policy"] == "failFast":
                        self._failFastChecks.add(record["id"])

        await self._session.commitWith(load, context)
        # Derive abort marks a crash left unapplied below cancelled owners, and finalize held outcomes.
        self._cascadePending = True
        self._scheduleReconcile()

    def resume(self) -> None:
        """Enable scheduling. Idempotent; the kick does nothing once closing."""
        self._enabled = True
        self._kick()

    @eager
    async def join(self) -> None:
        """Wait for every invocation signalled by ``_seal()``. Writes nothing."""
        # Shielded: ``Promise.allSettled`` cannot cancel what it joins, so a join that is cancelled leaves them running.
        await asyncio.gather(
            *[asyncio.shield(invocation.done) for invocation in list(self._invocations.values())], return_exceptions=True
        )

    @eager
    async def abort(self, id: TaskId, context: Context) -> str:
        """Commit the abort mark, or settle a task that no registered definition can take as ``orphaned`` when nothing it
        owns is live, then join the run invocation seen on the line; the commit listener signalled it. The abort
        invocation starts once the task's ordinary owned work is gone. A ``completing`` task is only marked."""

        async def mark(tx: Any) -> dict[str, Any]:
            current = await tx.task(id)
            if current is None:
                raise RuntimeError(f"Task {id} does not exist")
            if current["state"]["status"] == "terminal":
                return {"result": "terminal"}
            invocation = self._invocations.get(id)
            if invocation is None and current["state"]["status"] != "completing":
                await self._loadScopes(False)
                if id not in self._ownedLive():
                    resolution = self._resolve(current, self._registry.snapshot())
                    if resolution["kind"] == "blocked":
                        await self._terminate(tx, current, {"status": "orphaned", "reason": resolution["reason"]})
                        return {"result": "marked"}
            if not current["abortRequested"]:
                tx.setTask({**current, "abortRequested": True})
            return {"result": "marked", "run": invocation if invocation is not None and invocation.mode == "run" else None}

        marked = await self._session.commitWith(mark, context)
        # The commit listener signalled the run; join it.
        if marked.get("run") is not None:
            await awaitWithContext(asyncio.shield(marked["run"].done), context)
        return marked["result"]  # type: ignore[no-any-return]

    @eager
    async def waitForTask(self, id: TaskId, context: Context) -> dict[str, Any]:
        # Check and register on the line so no terminal publication falls between them.
        async def check() -> dict[str, Any]:
            if self._closing:
                raise closedError()
            if id in self._live:
                return {"promise": self._taskWaiters.add(id, context)}
            record = await self._storage.task(id, context)
            if record is None:
                raise RuntimeError(f"Task {id} does not exist")
            return {"promise": resolvedPromise(record)}

        found = await self._session.readOnLine(check)
        return await found["promise"]  # type: ignore[no-any-return]

    def waitForIdle(self, conversationId: ConversationId | None, context: Context) -> asyncio.Future[Any]:
        """Resolve when ordinary traversal from the conversation, or from every ownerless conversation, reaches no live
        non-background task."""
        if self._closing:
            return rejectedPromise(closedError())
        if self._idle(conversationId):
            return resolvedPromise(None)
        self._scheduleReconcile()
        return self._idleWaiters.add(conversationId, context)

    @eager
    async def abortConversation(self, conversationId: ConversationId, background: bool, context: Context) -> None:
        """``Conversation.abort()``: in one commit, withdraw the queued inputs and mark every live non-background task
        that ordinary traversal from the conversation reaches; resolves once the scope is idle. With ``background``,
        traversal crosses background boundaries, and the wait also covers every task it reached."""

        async def mark(tx: Any) -> list[TaskId]:
            queued = await self._loadScopes(True)
            scope = {"conversation": conversationId}
            reached: list[TaskId] = []
            for record in list(self._live.values()):
                if record["background"] and not background:
                    continue
                if self._inScope(parentOf(record), scope, background) is not True:
                    continue
                reached.append(record["id"])
                if not record["abortRequested"]:
                    tx.setTask({**record, "abortRequested": True})
            for id in queued:
                if self._inScope({"conversation": id}, scope, background) is True:
                    await self._withdrawInputs(tx, id)
            return reached

        reached = await self._session.commitWith(mark, context)
        if background:
            for id in reached:
                await self.waitForTask(id, context)
        await self.waitForIdle(conversationId, context)

    # ─── Scheduling ────────────────────────────────────────────────────────

    def _observe(self, publication: dict[str, Any]) -> None:
        updated: list[dict[str, Any]] = []
        failed: list[TaskId] = []
        changed = False
        for change in publication["changes"]:
            if change["type"] != "task":
                continue
            changed = True
            record = change["value"]
            previous = self._live.get(record["id"])
            if failedOutcome(record) and (previous is None or not failedOutcome(previous)):
                failed.append(record["id"])
            if record["state"]["status"] == "terminal":
                self._live.pop(record["id"], None)
                self._failedMigrations.pop(record["id"], None)
                self._failFastChecks.discard(record["id"])
                if record["id"] in self._conversationOwners:
                    self._settled[record["id"]] = nodeOf(record)
                self._taskWaiters.resolve(record["id"], record)
                # Its owner may finalize now.
                self._scheduleReconcile()
                continue
            if record["abortRequested"] and not (previous is not None and previous["abortRequested"] is True):
                self._cascadePending = True
                # Signal a run invocation of the newly marked task; its next step ends it.
                invocation = self._invocations.get(record["id"])
                if invocation is not None and invocation.mode == "run":
                    invocation.controller.abort()
            status = record["state"]["status"]
            if status == "completing" and not (previous is not None and previous["state"]["status"] == "completing"):
                if cancellationIntent(record):
                    self._cascadePending = True
                self._scheduleReconcile()
            if (
                status == "waiting"
                and record["state"]["policy"] == "failFast"
                and not (previous is not None and previous["state"]["status"] == "waiting")
            ):
                self._failFastChecks.add(record["id"])
                self._scheduleReconcile()
            self._live[record["id"]] = record
            updated.append(record)
        for id in failed:
            for record in self._live.values():
                state = record["state"]
                if state["status"] == "waiting" and state["policy"] == "failFast" and id in state["on"]:
                    self._failFastChecks.add(record["id"])
                    self._scheduleReconcile()
        for change in publication["changes"]:
            if change["type"] == "conversation" and change["value"]["id"] not in self._edges:
                owner = change["value"].get("owner")
                self._setEdge(change["value"]["id"], None if owner is None else owner["taskId"])
        for change in publication["changes"]:
            # A queued input below a cancelled owner is withdrawn, even after its cascade.
            if change["type"] != "submission" or change["value"]["status"] != "queued" or change["value"]["type"] != "input":
                continue
            up = {"conversation": change["value"]["conversationId"]}
            if not self._chainKnown(up) or self._belowCancelled(up):
                self._cascadePending = True
        for record in updated:
            # Work created below a cancelled owner, even after its cascade, is aborted too.
            if not self._chainKnown(parentOf(record)):
                self._scheduleReconcile()
            elif not record["background"] and not record["abortRequested"] and self._belowCancelled(parentOf(record)):
                self._cascadePending = True
        # Also retries, with the next commit of any kind, a cascade whose commit failed.
        if self._cascadePending:
            self._scheduleReconcile()
        if not changed:
            return
        self._settleIdle()
        self._kick()

    def _contextRetentionMs(self) -> int:
        """``settings.contextRetentionMs``, or 0 when the host's settings throw. Kept contexts are only a cache, so
        dropping them is safe, while a throw here would escape commit listeners, reconciliation, and the expiry timer."""
        try:
            return self._settings()["contextRetentionMs"]  # type: ignore[no-any-return]
        except Exception as error:  # noqa: BLE001 - reported; the cache is dropped instead
            self._report(error)
            return 0

    def _settleIdle(self) -> None:
        """Resolve idle waiters, and drop each kept context whose conversation has been idle for the retention period."""
        for conversationId in self._idleWaiters.keys():  # noqa: SIM118 - a Waiters, not a dict
            if self._idle(conversationId):
                self._idleWaiters.resolve(conversationId, None)
        now = self._now()
        retention = self._contextRetentionMs()
        for conversationId, kept in list(self._contexts.items()):
            if kept["idleSince"] is not None and now - kept["idleSince"] >= retention:
                del self._contexts[conversationId]
            elif not self._idle(conversationId):
                kept["idleSince"] = None
            elif retention > 0:
                if kept["idleSince"] is None:
                    kept["idleSince"] = now
            else:
                del self._contexts[conversationId]
        self._scheduleExpiry()

    def _scheduleExpiry(self) -> None:
        """Run ``_settleIdle()`` when the earliest idle context expires. The timer never keeps the process alive.
        MISAKA fork: an asyncio timer handle never keeps the event loop running, as an unreferenced Node timer does, so
        upstream's branch for runtimes whose timers cannot be unreferenced (Cloudflare Workers) has no counterpart."""
        retention = self._contextRetentionMs()
        at: int | None = None
        for kept in self._contexts.values():
            if kept["idleSince"] is not None and (at is None or kept["idleSince"] + retention < at):
                at = kept["idleSince"] + retention
        if self._expiry is not None and self._expiry["at"] == at:
            return
        if self._expiry is not None:
            self._expiry["timer"].cancel()
        self._expiry = None
        if at is None or self._closing:
            return

        def fire() -> None:
            self._expiry = None
            self._settleIdle()

        timer = asyncio.get_running_loop().call_later(min(max(0, at - self._now()), MAX_TIMER_DELAY) / 1000, fire)
        self._expiry = {"at": at, "timer": timer}

    # ─── Ownership ───────────────────────────────────────────────────────────

    def _scheduleReconcile(self) -> None:
        if self._reconcileScheduled or self._closing:
            return
        self._reconcileScheduled = True
        queueMicrotask(self._startReconcile)

    def _startReconcile(self) -> None:
        spawn(self._reconcile(), self._tasks)

    def _startDrain(self) -> None:
        spawn(self._drain(), self._tasks)

    async def _reconcile(self) -> None:
        """One commit that applies what committed records imply: abort marks below live owners with cancellation intent
        (spec section 5.4), ``failFast`` marks (spec section 5.5), withdrawn queued inputs below cancelled owners, and
        the final terminal record of every ``completing`` task whose ordinary owned work is gone. The durable records
        are the intent, so this also repairs whatever a crash left unapplied. Resolves idle waiters that the loaded
        edges decide."""
        self._reconcileScheduled = False
        cascade = self._cascadePending
        self._cascadePending = False
        checks = list(self._failFastChecks)
        self._failFastChecks.clear()
        try:

            async def apply(tx: Any) -> None:
                if self._closing:
                    return
                queued = await self._loadScopes(cascade)
                marked: set[TaskId] = set()

                def mark(record: dict[str, Any]) -> None:
                    if record["abortRequested"] or record["id"] in marked:
                        return
                    marked.add(record["id"])
                    tx.setTask({**record, "abortRequested": True})

                # Loading edges can reveal a cancelled owner, so marks are derived on every pass.
                for record in self._live.values():
                    if not record["background"] and self._belowCancelled(parentOf(record)):
                        mark(record)
                for id in checks:
                    waiter = self._live.get(id)
                    if waiter is None or waiter["state"]["status"] != "waiting" or not await self._anyFailed(waiter["state"]["on"]):
                        continue
                    # Every other live task: the failed one keeps its own outcome.
                    for member in waiter["state"]["on"]:
                        other = self._live.get(member)
                        if other is not None and not failedOutcome(other):
                            mark(other)
                for id in queued:
                    if self._belowCancelled({"conversation": id}):
                        await self._withdrawInputs(tx, id)
                await self._finalize(tx)

            await self._session.commitWith(apply, self._context)
        except Exception as error:  # noqa: BLE001 - any pass may have staged marks, so a failed one is retried
            self._cascadePending = True
            for id in checks:
                self._failFastChecks.add(id)
            if not self._closing:
                self._report(error)
        self._settleIdle()

    async def _anyFailed(self, ids: list[TaskId]) -> bool:
        """Whether any of ``ids`` holds or ended with an outcome other than ``completed``."""
        for id in ids:
            record = self._live.get(id)
            if record is None:
                record = await self._storage.task(id, self._context)
            if record is not None and failedOutcome(record):
                return True
        return False

    async def _finalize(self, tx: Any) -> None:
        """Write the terminal record of every ``completing`` task without live ordinary owned work. Finalizing one can
        free its owner, so this repeats over the commit's candidates until nothing changes. A held scheduler outcome
        gets its Harness cleanup here."""
        while True:
            overlay = overlayOf(tx)
            owned = self._ownedLive(overlay)
            done = [
                record
                for record in self._liveRecords(overlay)
                if record["state"]["status"] == "completing" and record["id"] not in owned
            ]
            if len(done) == 0:
                return
            for record in done:
                outcome = record["state"]["outcome"]
                tx.setTask(withState(record, {"status": "terminal", "outcome": outcome}))
                # REMINDER: only the scheduler writes ``faulted`` and ``orphaned`` (spec section 5.4); their cleanup waits
                # for this commit.
                if outcome["status"] == "faulted" or outcome["status"] == "orphaned":
                    await self._settleOutcome(tx, record, outcome)

    async def _loadScopes(self, queued: bool) -> list[ConversationId]:
        """Load the owner chains of every live task and, with ``queued``, of every conversation with queued
        submissions, on the Session line; returns the latter. Reads committed Storage directly, so it may run inside a
        commit callback."""
        for record in list(self._live.values()):
            if not self._chainKnown(parentOf(record)):
                await self._loadChain(parentOf(record))
        if not queued:
            return []
        submissions = await scanAll(
            lambda cursor: self._storage.scanSubmissions({"status": "queued"}, SCAN_PAGE_SIZE, cursor, self._context)
        )
        conversations = list(dict.fromkeys(submission["conversationId"] for submission in submissions))
        for id in conversations:
            await self._loadChain({"conversation": id})
        return conversations

    async def _loadChain(self, start: dict[str, Any], overlay: dict[str, Any] | None = None) -> None:
        """Load the owner edges and task nodes from ``start`` up to its ownerless root."""
        at: dict[str, Any] | None = start
        while at is not None:
            if "task" in at:
                known = self._node(at["task"], overlay)
                if known is None:
                    record = await self._storage.task(at["task"], self._context)
                    if record is None:
                        return
                    known = nodeOf(record)
                    if record["state"]["status"] == "terminal":
                        self._settled[record["id"]] = known
                at = parentOf(known)
            else:
                edge = self._edge(at["conversation"], overlay)
                if edge is UNDEFINED:
                    record = await self._storage.conversation(at["conversation"], self._context)
                    owner = None if record is None else record.get("owner")
                    edge = None if owner is None else owner["taskId"]
                    self._setEdge(at["conversation"], edge)
                at = None if edge is None else {"task": edge}

    def _setEdge(self, conversationId: ConversationId, owner: TaskId | None) -> None:
        self._edges[conversationId] = owner
        if owner is not None:
            self._conversationOwners.add(owner)

    def _edge(self, id: ConversationId, overlay: dict[str, Any] | None = None) -> Any:
        """Owner task of a conversation, ``None`` when ownerless, ``UNDEFINED`` while not loaded."""
        if overlay is not None and id in overlay["edges"]:
            return overlay["edges"][id]
        return self._edges.get(id, UNDEFINED)

    def _node(self, id: TaskId, overlay: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if overlay is not None and overlay["tasks"].get(id) is not None:
            return overlay["tasks"][id]  # type: ignore[no-any-return]
        found = self._live.get(id)
        return found if found is not None else self._settled.get(id)

    def _above(self, start: dict[str, Any], overlay: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        """Walk up from ``start``: owner tasks and conversations, ending at an ownerless root or an edge not loaded
        yet."""
        at: dict[str, Any] | None = start
        while at is not None:
            if "task" in at:
                node = self._node(at["task"], overlay)
                if node is None:
                    yield {"unknown": True}
                    return
                yield {"task": at["task"], "node": node}
                at = parentOf(node)
            else:
                yield {"conversation": at["conversation"]}
                edge = self._edge(at["conversation"], overlay)
                if edge is UNDEFINED:
                    yield {"unknown": True}
                    return
                at = None if edge is None else {"task": edge}

    def _chainKnown(self, start: dict[str, Any], overlay: dict[str, Any] | None = None) -> bool:
        """Whether every owner above ``start`` is loaded."""
        return not any("unknown" in step for step in self._above(start, overlay))

    def _liveRecords(self, overlay: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        """Live records, with the overlay's candidates replacing committed ones; terminal candidates are gone."""
        for record in self._live.values():
            candidate = overlay["tasks"].get(record["id"]) if overlay is not None else None
            if candidate is None:
                candidate = record
            if candidate["state"]["status"] != "terminal":
                yield candidate
        if overlay is None:
            return
        for record in overlay["tasks"].values():
            if record["id"] not in self._live and record["state"]["status"] != "terminal":
                yield record

    def _ownedLive(self, overlay: dict[str, Any] | None = None) -> dict[TaskId, list[TaskId]]:
        """Every task with live ordinary owned work (spec section 5.5), mapped to that work: each live non-background
        task counts for every owner task above it up to and including the first background one. Owner chains must be
        loaded."""
        owned: dict[TaskId, list[TaskId]] = {}
        for record in self._liveRecords(overlay):
            if record["background"]:
                continue
            for step in self._above(parentOf(record), overlay):
                if "unknown" in step:
                    break
                if "task" not in step:
                    continue
                below = owned.get(step["task"])
                if below is None:
                    owned[step["task"]] = [record["id"]]
                else:
                    below.append(record["id"])
                if step["node"]["background"]:
                    break
        return owned

    def _inScope(self, start: dict[str, Any], scope: dict[str, Any], crossBackground: bool = False) -> bool | None:
        """Whether ordinary traversal from ``scope`` reaches ``start``: walking up reaches the scope's conversation, or
        an ownerless one for ``roots``, without crossing a background owner, unless ``crossBackground``. ``None`` while
        an edge is not loaded."""
        for step in self._above(start):
            if "unknown" in step:
                return None
            if "conversation" in step:
                if "conversation" in scope and step["conversation"] == scope["conversation"]:
                    return True
            elif step["node"]["background"] and not crossBackground:
                return False
        return "roots" in scope

    def _belowCancelled(self, start: dict[str, Any]) -> bool:
        """Whether a live owner's cancellation intent reaches ``start``: walking up finds an owner with intent before a
        background owner without it. Terminal owners never cascade (spec section 5.4)."""
        for step in self._above(start):
            if "unknown" in step:
                return False
            if "task" not in step:
                continue
            live = self._live.get(step["task"])
            if live is not None and cancellationIntent(live):
                return True
            if step["node"]["background"]:
                return False
        return False

    def _seal(self) -> None:
        """Close listener: runs synchronously once admission is sealed, before ``join()``."""
        self._closing = True
        self._unsubscribeRegistry()
        error = closedError()
        self._taskWaiters.rejectAll(error)
        self._idleWaiters.rejectAll(error)
        self._contexts.clear()
        if self._expiry is not None:
            self._expiry["timer"].cancel()
        self._expiry = None
        for invocation in list(self._invocations.values()):
            invocation.controller.abort()

    def _kick(self) -> None:
        self._dirty = True
        if self._draining or not self._enabled or self._closing:
            return
        self._draining = True
        # Never commit synchronously from a commit or registry listener.
        queueMicrotask(self._startDrain)

    async def _drain(self) -> None:
        try:
            while self._dirty and self._enabled and not self._closing:
                self._dirty = False
                for reservation in await self._reserve():
                    self._start(reservation)
        except Exception as error:  # noqa: BLE001 - reported unless the Harness is closing
            if not self._closing:
                self._report(error)
        finally:
            self._draining = False
            # A wakeup that arrived during a failed pass still needs its pass.
            if self._dirty:
                self._kick()

    async def _reserve(self) -> list[_Reservation]:
        """Reserve every eligible task in one commit; orphan abort-marked tasks no definition can take."""
        reservations: list[_Reservation] = []
        try:

            async def reserve(tx: Any) -> None:
                if not self._enabled or self._closing:
                    return
                await self._loadScopes(False)
                owned = self._ownedLive()
                # Taken once per pass, and only when some task is a candidate.
                snapshot: Any = None
                for record in list(self._live.values()):
                    if record["id"] in self._invocations or len(self._waitingOn(record, owned)) > 0:
                        continue
                    if record["state"]["status"] == "completing":
                        continue
                    mode = "abort" if record["abortRequested"] else "run"
                    if snapshot is None:
                        snapshot = self._registry.snapshot()
                    resolution = self._resolve(record, snapshot)
                    if resolution["kind"] == "blocked":
                        if mode == "abort":
                            await self._terminate(tx, record, {"status": "orphaned", "reason": resolution["reason"]})
                        continue
                    if resolution["record"] is not record or record["state"]["status"] != "running":
                        tx.setTask(
                            withState(
                                resolution["record"],
                                {"status": "running", "checkpoint": resolution["record"]["state"]["checkpoint"]},
                            )
                        )
                    # Registered on the line, so marks and later reservations see it and close joins it.
                    invocation = self._createInvocation(record, mode)
                    reservations.append(_Reservation(invocation, resolution["task"], snapshot))

            await self._session.commitWith(reserve, self._context)
        except BaseException:
            for reservation in reservations:
                if self._invocations.get(reservation.invocation.taskId) is reservation.invocation:
                    del self._invocations[reservation.invocation.taskId]
                reservation.invocation.finish()
            raise
        return reservations

    def _waitingOn(self, record: dict[str, Any], owned: dict[TaskId, list[TaskId]]) -> list[TaskId]:
        """Live tasks a task waits for before its next invocation: its live ordinary owned work when abort-marked, since
        abort runs bottom-up, otherwise the live part of the ``on`` of a wait."""
        if record["abortRequested"]:
            return owned.get(record["id"], [])
        if record["state"]["status"] != "waiting":
            return []
        return [id for id in record["state"]["on"] if id in self._live]

    def _resolve(self, record: dict[str, Any], snapshot: Any) -> dict[str, Any]:
        """Resolve the record's definition by kind, migrating an older stored version."""
        fit = self._fit(record, snapshot.task(record["kind"]))
        if "reason" in fit:
            return {"kind": "blocked", "reason": fit["reason"]}
        if not fit["migrates"]:
            return {"kind": "ready", "task": fit["task"], "record": record}
        definition = erased(fit["task"])
        try:
            if definition.get("migrate") is None:
                raise missingMigration(record, definition)
            migrated = definition["migrate"](record["input"], record["state"]["checkpoint"], record["version"])
            state = {**record["state"], "checkpoint": copyJson(migrated["checkpoint"])}
            migratedRecord = {**record, "version": definition["version"], "input": copyJson(migrated["input"]), "state": state}
            return {"kind": "ready", "task": fit["task"], "record": migratedRecord}
        except Exception as error:  # noqa: BLE001 - a failing migration blocks the task and is reported
            self._failedMigrations[record["id"]] = {"task": fit["task"], "error": error}
            self._report(error)
            return {"kind": "blocked", "reason": "migration_failed"}

    def _fit(self, record: dict[str, Any], task: dict[str, Any] | None) -> dict[str, Any]:
        if task is None:
            return {"reason": "missing_task"}
        version = task["definition"]["version"]
        if version == record["version"]:
            return {"task": task, "migrates": False}
        if version < record["version"]:
            return {"reason": "task_too_old"}
        failed = self._failedMigrations.get(record["id"])
        if failed is not None and failed["task"] is task:
            return {"reason": "migration_failed", "error": failed["error"]}
        return {"task": task, "migrates": True}

    @eager
    async def inspect(self, snapshot: Any) -> dict[str, Any]:
        """Scheduling state and every live task with its derived state, read on the Session line. Runs no task code: a
        pending migration shows as ``ready`` with ``migrates``, and only a migration the scheduler already tried, or one
        that cannot exist, shows as failed."""
        await self._loadScopes(False)
        owned = self._ownedLive()
        tasks = [{"record": record, "state": self._inspectTask(record, snapshot, owned)} for record in self._live.values()]
        scheduling = "closing" if self._closing else "running" if self._enabled else "paused"
        return {"scheduling": scheduling, "tasks": tasks}

    def _inspectTask(self, record: dict[str, Any], snapshot: Any, owned: dict[TaskId, list[TaskId]]) -> dict[str, Any]:
        if record["id"] in self._invocations:
            return {"kind": "running"}
        if record["state"]["status"] == "completing":
            return {"kind": "completing"}
        on = self._waitingOn(record, owned)
        if len(on) > 0:
            return {"kind": "waiting", "on": on}
        fit = self._fit(record, snapshot.task(record["kind"]))
        if "reason" in fit:
            return {"kind": "blocked", **fit}
        if fit["migrates"] and fit["task"]["definition"].get("migrate") is None:
            return {
                "kind": "blocked",
                "reason": "migration_failed",
                "error": missingMigration(record, erased(fit["task"])),
            }
        return {"kind": "ready", "migrates": fit["migrates"]}

    def _createInvocation(self, record: dict[str, Any], mode: str) -> _Invocation:
        controller = AbortController()
        invocation = _Invocation(record, mode, controller, withAbortSignal(controller.signal, self._context))
        self._invocations[record["id"]] = invocation
        return invocation

    def _start(self, reservation: _Reservation) -> None:
        invocation = reservation.invocation

        async def execute() -> None:
            discarded = False
            try:
                if invocation.mode == "run":
                    await self._run(reservation)
                else:
                    await self._runAbort(reservation)
            except Exception as error:  # noqa: BLE001 - an escaped failure is reported
                self._report(error)
            except GeneratorExit:
                # The coroutine is dropped while a handler is still suspended, as when the loop closes on a crashed
                # Harness. The continuation of a promise that never settles never runs, so nothing here may either.
                discarded = True
                raise
            finally:
                if not discarded:
                    self._end(invocation)
                    invocation.finish()
                    self._kick()

        spawn(execute(), self._tasks)

    async def _run(self, reservation: _Reservation) -> None:
        """Run phase handlers, each preceded by a step that decides on the line whether the invocation continues."""
        invocation = reservation.invocation
        state = _PhaseState(reservation.task, reservation.snapshot)
        phase = _Phase(lambda: state.snapshot, lambda: state.task)
        runtime = TaskRuntimeImpl(self, invocation, phase)
        previous: dict[str, Any] | None = None
        while True:
            def decideWith(tx: Any, running: dict[str, Any], previous: dict[str, Any] | None = previous) -> Any:
                return self._decide(tx, running, previous, state)

            current = await self._step(invocation, decideWith)
            # Close may seal between the decision and dispatch.
            if current is None or self._closing:
                return
            checkpoint = current["state"]["checkpoint"]
            # Each phase handler resolves its agent afresh, at first use.
            phase.agent = None
            try:
                await awaitValue(erased(state.task)["phases"][checkpoint["phase"]](current, runtime, invocation.context))
                previous = {"checkpoint": checkpoint}
            except BaseException as error:  # noqa: BLE001 - an uncaught throw is a fault the next step writes
                previous = {"checkpoint": checkpoint, "failure": {"error": _thrownBy(error)}}

    def _decide(
        self, tx: Any, current: dict[str, Any], previous: dict[str, Any] | None, state: _PhaseState
    ) -> bool | dict[str, Any]:
        """Precedence rules for a run invocation, on the line. Rules 1 (terminal, ``completing``, or ``waiting``) and 2
        (closing) are applied by ``_step``. Returns whether the invocation continues with the next phase."""
        # 3. abort mark: end; a fresh abort invocation starts once the task's ordinary owned work is gone.
        if current["abortRequested"]:
            return False
        if previous is None:
            return True
        # 4. uncaught error.
        if previous.get("failure") is not None:
            return {"fault": previous["failure"]["error"]}
        # 6. no durable progress.
        if jsonEqual(current["state"]["checkpoint"], previous["checkpoint"]):
            message = f"Task {current['kind']} phase {previous['checkpoint']['phase']} returned without durable progress"
            return {"fault": RuntimeError(message)}
        # 5. progress: refresh the snapshot; hand over to a replacement definition that can take the task.
        state.snapshot = self._registry.snapshot()
        following = state.snapshot.task(current["kind"])
        if following is not state.task:
            if following is not None and canReserve(following, current):
                tx.setTask(withState(current, {"status": "pending", "checkpoint": current["state"]["checkpoint"]}))
                return False
            if state.reported is None or state.reported["task"] is not following:
                state.reported = {"task": following}
                cause = "missing_task" if following is None else "incompatible_task"
                error = RuntimeError(f"Task {current['id']} keeps running under its old {current['kind']} definition")
                error.cause = cause  # type: ignore[attr-defined]
                self._report(error)
        return True

    async def _runAbort(self, reservation: _Reservation) -> None:
        """Run the abort handler once; rules 1, 2, and 4 apply, and returning without an outcome faults."""
        invocation = reservation.invocation
        current = self._live.get(invocation.taskId)
        if current is None or self._closing:
            return
        failure: dict[str, Any] | None = None
        try:
            phase = _Phase(lambda: reservation.snapshot, lambda: reservation.task)
            runtime = TaskRuntimeImpl(self, invocation, phase)
            await awaitValue(erased(reservation.task)["abort"](current, runtime, invocation.context))
        except BaseException as error:  # noqa: BLE001 - a failing abort handler faults the task
            failure = {"error": _thrownBy(error)}
        message = f"Abort handler of task {invocation.taskId} returned without a terminal outcome"
        fault = failure["error"] if failure is not None else RuntimeError(message)
        await self._step(invocation, lambda tx, running: {"fault": fault})

    async def _step(self, invocation: _Invocation, decide: Callable[[Any, dict[str, Any]], Any]) -> dict[str, Any] | None:
        """One synchronous decision on the Session line. A task that is no longer running (rule 1: terminal,
        ``completing``, or ``waiting``) or a closing Harness (rule 2) ends the invocation without a write; otherwise
        ``decide`` may stage a write and returns whether the invocation continues. Ending happens inside the callback,
        before a fault's Harness cleanup. A rejected step, such as admission after close, also ends the invocation."""
        try:

            async def decideOnLine(tx: Any) -> dict[str, Any] | None:
                found = self._live.get(invocation.taskId)
                current = found if found is not None and found["state"]["status"] == "running" else None
                decision = decide(tx, current) if current is not None and not self._closing else False
                if decision is True:
                    return current
                self._end(invocation)
                if decision is not False:
                    message = errorMessage(decision["fault"])
                    assert current is not None
                    await self._terminate(tx, current, {"status": "faulted", "error": {"message": message}})
                return None

            return await self._session.commitWith(decideOnLine, self._context)  # type: ignore[no-any-return]
        except Exception as error:  # noqa: BLE001 - a rejected step ends the invocation
            self._end(invocation)
            if not self._closing:
                self._report(error)
            return None

    async def _terminate(self, tx: Any, record: dict[str, Any], outcome: dict[str, Any]) -> None:
        """Write an outcome the scheduler decided. While the task's ordinary owned work is live it holds as
        ``completing`` and its Harness cleanup waits for the final commit (spec section 5.5, rule 4); otherwise it is
        terminal with its cleanup."""
        await self._loadScopes(False)
        if record["id"] in self._ownedLive(overlayOf(tx)):
            tx.setTask(withState(record, {"status": "completing", "outcome": outcome}))
            return
        tx.setTask(withState(record, {"status": "terminal", "outcome": outcome}))
        await self._settleOutcome(tx, record, outcome)

    async def _commitState(self, tx: Any, invocation: _Invocation, current: dict[str, Any], following: dict[str, Any]) -> None:
        """Replace a running task's state with what it committed. A terminal state holds as ``completing`` while
        ordinary owned work is live, judged on the commit's candidates, so work the same commit creates below the task
        counts. A wait is validated first."""
        if following["status"] == "waiting":
            await self._validateWait(tx, invocation, current, following["on"], following["policy"])
        if following["status"] == "terminal":
            overlay = overlayOf(tx)
            await self._loadScopes(False)
            for record in list(overlay["tasks"].values()):
                await self._loadChain(parentOf(record), overlay)
            if current["id"] in self._ownedLive(overlay):
                tx.setTask(withState(current, {"status": "completing", "outcome": following["outcome"]}))
                return
        tx.setTask(withState(current, following))

    async def _validateWait(
        self, tx: Any, invocation: _Invocation, current: dict[str, Any], on: list[TaskId], policy: str
    ) -> None:
        """A wait names existing tasks other than the waiter and its owners, which could never finish first;
        ``failFast`` only tasks the waiter owns. An abort handler cannot wait."""
        if invocation.mode == "abort":
            raise RuntimeError(f"Abort handler of task {current['id']} cannot wait")
        overlay = overlayOf(tx)
        await self._loadChain(parentOf(current))
        owners: set[TaskId] = set()
        for step in self._above(parentOf(current)):
            if "task" in step:
                owners.add(step["task"])
        for id in on:
            if id == current["id"] or id in owners:
                raise RuntimeError(f"Task {current['id']} cannot wait on itself or its owner {id}")
            member = overlay["tasks"].get(id)
            if member is None:
                member = self._live.get(id)
            if member is None:
                member = await self._storage.task(id, self._context)
            if member is None:
                raise RuntimeError(f"Task {id} does not exist")
            if policy == "failFast" and member.get("owner") != current["id"]:
                raise RuntimeError(f"Task {current['id']} can wait failFast only on tasks it owns; {id} is not one")

    def _end(self, invocation: _Invocation) -> None:
        """End an invocation: its runtime operations reject from now on, its signal aborts, its watches stop, and its
        task is free."""
        if invocation.ended:
            return
        invocation.ended = True
        if self._invocations.get(invocation.taskId) is invocation:
            del self._invocations[invocation.taskId]
        for watch in list(invocation.watches):
            spawn(_stopWatch(watch), self._tasks)
        # Pending waits bound to the invocation, such as a tool's waitForTask(), reject with it.
        invocation.controller.abort(endedError(invocation))

    def _idle(self, conversationId: ConversationId | None) -> bool:
        """No live non-background task in the scope; a task whose owner edges are not loaded yet counts as inside."""
        scope = {"roots": True} if conversationId is None else {"conversation": conversationId}
        for record in self._live.values():
            if not record["background"] and self._inScope(parentOf(record), scope) is not False:
                return False
        return True

    # ─── Invocation runtime ──────────────────────────────────────────────────

    @eager
    async def _read(self, invocation: _Invocation, read: Callable[[], Any]) -> Any:
        """Run a committed-state read unless the invocation has ended."""
        if invocation.ended:
            raise endedError(invocation)
        return await read()

    def _gated(self, invocation: _Invocation, change: Callable[[Any, dict[str, Any]], Any], context: Context) -> asyncio.Future[Any]:
        """Commit after rereading the task on the line and gating the invocation."""
        if invocation.ended:
            return rejectedPromise(endedError(invocation))

        async def gated(tx: Any) -> Any:
            if invocation.ended:
                raise endedError(invocation)
            if self._closing:
                raise closedError()
            found = self._live.get(invocation.taskId)
            if found is None:
                raise RuntimeError(f"Task {invocation.taskId} is terminal")
            if found["state"]["status"] != "running":
                raise RuntimeError(f"Task {invocation.taskId} is {found['state']['status']}")
            if invocation.mode == "run" and found["abortRequested"]:
                raise RuntimeError(f"Task {invocation.taskId} has a durable abort mark")
            return await awaitValue(change(tx, found))

        return self._session.commitWith(  # type: ignore[no-any-return]
            gated, context, {"conversationId": invocation.conversationId, "taskId": invocation.taskId}
        )

    async def _sleep(self, invocation: _Invocation, until: float, context: Context) -> None:
        """Wait until the Harness clock reaches ``until``, rechecking it after every timer."""
        if invocation.ended:
            raise endedError(invocation)
        signals = [invocation.controller.signal]
        if context.abortSignal is not None:
            signals.append(context.abortSignal)
        signal = AbortSignal.any(signals)
        while True:
            signal.throwIfAborted()
            remaining = until - self._now()
            if remaining <= 0:
                return
            await delay(min(remaining, MAX_TIMER_DELAY), signal)

    async def _watchDoc(self, invocation: _Invocation, args: tuple[Any, ...]) -> Any:
        if invocation.ended:
            raise endedError(invocation)
        watch = await self._session.watchDoc(*args)
        if watch is None:
            return None
        if invocation.ended:
            spawn(_stopWatch(watch), self._tasks)
            raise endedError(invocation)
        invocation.watches[watch] = None
        watch.closed.add_done_callback(lambda done: invocation.watches.pop(watch, None))
        return watch


async def _stopWatch(watch: Any) -> None:
    await watch.stop()


def cancellationIntent(record: dict[str, Any]) -> bool:
    """A live owner's durable cancellation intent: its abort mark, or a held outcome other than ``completed``."""
    return record["state"]["status"] != "terminal" and (record["abortRequested"] or failedOutcome(record))


def failedOutcome(record: dict[str, Any]) -> bool:
    """Whether the record holds or ends with an outcome other than ``completed``."""
    state = record["state"]
    return state["status"] in ("completing", "terminal") and state["outcome"]["status"] != "completed"


def parentOf(node: dict[str, Any]) -> dict[str, Any]:
    """Where a walk up the ownership tree continues: an owner task, or a conversation."""
    return {"task": node["owner"]} if node.get("owner") is not None else {"conversation": node["conversationId"]}


def nodeOf(record: dict[str, Any]) -> dict[str, Any]:
    """The immutable ownership fields of a task."""
    return {
        "conversationId": record["conversationId"],
        **({} if record.get("owner") is None else {"owner": record["owner"]}),
        "background": record["background"],
    }


def overlayOf(tx: Any) -> dict[str, Any]:
    """Candidate records a commit staged; they override committed records in ownership walks."""
    tasks = {record["id"]: record for record in tx.stagedTasks()}
    edges = {
        record["id"]: (None if record.get("owner") is None else record["owner"]["taskId"])
        for record in tx.stagedConversations()
    }
    return {"tasks": tasks, "edges": edges}


def memoOf(record: dict[str, Any] | None, name: str) -> Any:
    """Own memo entry only; memo names such as ``toString`` must not resolve to inherited properties. ``UNDEFINED`` when
    there is none."""
    memos = None if record is None else record.get("memos")
    return memos[name] if memos is not None and name in memos else UNDEFINED


def missingMigration(record: dict[str, Any], definition: dict[str, Any]) -> Exception:
    return RuntimeError(f"Task {record['kind']} version {definition['version']} has no migration from {record['version']}")


def erased(task: dict[str, Any]) -> dict[str, Any]:
    return task["definition"]  # type: ignore[no-any-return]


def endedError(invocation: _Invocation) -> Exception:
    return RuntimeError(f"Task {invocation.taskId} invocation has ended")


def withState(record: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Replace a live record's state; memos disappear once an outcome is decided."""
    if state["status"] != "terminal" and state["status"] != "completing":
        return {**record, "state": state}
    rest = {key: value for key, value in record.items() if key != "memos"}
    return {**rest, "state": state}


def canReserve(task: dict[str, Any], record: dict[str, Any]) -> bool:
    """Whether a definition can take the task at reservation: same version, or newer with a migration."""
    definition = task["definition"]
    return definition["version"] == record["version"] or (
        definition["version"] > record["version"] and definition.get("migrate") is not None
    )


async def delay(ms: float, signal: AbortSignal) -> None:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[None] = loop.create_future()

    def onAbort() -> None:
        handle.cancel()
        if not future.done():
            future.set_exception(abortError(signal))

    def fire() -> None:
        signal.removeEventListener("abort", onAbort)
        if not future.done():
            future.set_result(None)

    handle = loop.call_later(ms / 1000, fire)
    signal.addEventListener("abort", onAbort, once=True)
    try:
        await future
    finally:
        handle.cancel()
        with contextlib.suppress(Exception):
            signal.removeEventListener("abort", onAbort)


def jsonEqual(left: Any, right: Any) -> bool:
    """Structural equality of two JSON values; object key order is ignored."""
    if left is right:
        return True
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    if not isinstance(left, list | dict) or not isinstance(right, list | dict):
        return left == right and not isinstance(left, list | dict) and not isinstance(right, list | dict)
    if isinstance(left, list) or isinstance(right, list):
        if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
            return False
        return all(jsonEqual(value, right[index]) for index, value in enumerate(left))
    if len(left) != len(right):
        return False
    return all(key in right and jsonEqual(value, right[key]) for key, value in left.items())


__all__ = ["TaskRuntimeImpl", "TaskScheduler", "jsonEqual", "withState"]
