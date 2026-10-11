"""pi durable's ``test/harness-ownership.test.ts``."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from misaka.ai.providers.faux import fauxAssistantMessage, fauxText, fauxToolCall
from misaka.chord.context import withAbortSignal
from misaka.durable import (
    LiveDoc,
    MemoryStorage,
    StorageRejected,
    configure,
    createSession,
    defineDoc,
    defineTask,
    defineTool,
)
from misaka.durable.storage.sqlite.node import openNodeSqliteStorage
from misaka.utils.js import AbortController
from tests.chat_support import chatSetup, openChat, waitFor
from tests.durable_support import context, toMatchObject
from tests.harness_support import addTask, addTool
from tests.task_support import aborted, deferred, openTasks, race, settled

gates: dict[str, Any] = {}
runs: dict[str, int] = {}


def gate(name: str) -> Any:
    """The gate of a named task: a Deferred that resolves with the outcome the task commits."""
    found = gates.get(name)
    if found is None:
        found = deferred()
        gates[name] = found
    return found


async def holdRun(task: Any, runtime: Any, ctx: Any) -> None:
    runs[task["input"]["name"]] = runs.get(task["input"]["name"], 0) + 1
    ending = await race(gate(task["input"]["name"]).promise, aborted(runtime.signal))
    if ending == "completed":
        state: dict[str, Any] = {"status": "terminal", "outcome": {"status": "completed", "result": None}}
    else:
        state = {"status": "terminal", "outcome": {"status": "failed", "error": {"message": "gate failed"}}}
    await runtime.commit(lambda tx, current: state, ctx)


async def holdAbort(task: Any, runtime: Any, ctx: Any) -> None:
    if task["input"].get("slowAbort") is True:
        await gate(f"abort.{task['input']['name']}").promise
    await runtime.commit(lambda tx, current: {"status": "terminal", "outcome": {"status": "aborted"}}, ctx)


# A task that holds until its named gate opens or it is aborted. With ``slowAbort``, its abort handler first waits for
# the gate ``abort.<name>``.
Hold = defineTask(
    {
        "name": "test.hold",
        "version": 1,
        "initial": lambda input: {"phase": "hold"},
        "phases": {"hold": holdRun},
        "abort": holdAbort,
    }
)


async def waiterWait(task: Any, runtime: Any, ctx: Any) -> None:
    checkpoint = {"phase": "done"}
    await runtime.commit(
        lambda tx, current: {"status": "waiting", "checkpoint": checkpoint, "on": task["input"]["on"], "policy": "allSettled"},
        ctx,
    )


async def waiterDone(_task: Any, runtime: Any, ctx: Any) -> None:
    await runtime.commit(
        lambda tx, current: {"status": "terminal", "outcome": {"status": "completed", "result": None}}, ctx
    )


async def waiterAbort(_task: Any, runtime: Any, ctx: Any) -> None:
    await runtime.commit(lambda tx, current: {"status": "terminal", "outcome": {"status": "aborted"}}, ctx)


# Waits on the tasks in its input, then completes; its abort handler ends it ``aborted``.
Waiter = defineTask(
    {
        "name": "test.waiter",
        "version": 1,
        "initial": lambda input: {"phase": "wait"},
        "phases": {"wait": waiterWait, "done": waiterDone},
        "abort": waiterAbort,
    }
)


async def nothing(*_: Any) -> None:
    return None


# Never registered: aborting it can only orphan it.
Unregistered = defineTask(
    {
        "name": "test.unregistered",
        "version": 1,
        "initial": lambda input: {"phase": "hold"},
        "phases": {"hold": nothing},
        "abort": nothing,
    }
)


def openGate(name: str, ending: str) -> None:
    gate(name).resolve(ending)


@pytest.fixture(autouse=True)
def clearGates() -> Iterator[None]:
    yield
    gates.clear()
    runs.clear()


@pytest.fixture
def sqliteDirectory() -> Iterator[str]:
    directory = tempfile.mkdtemp(prefix="pi-durable-ownership-")
    yield directory
    shutil.rmtree(directory, ignore_errors=True)


async def ownedChild(parent: Any, name: str, options: dict[str, Any] | None = None) -> SimpleNamespace:
    """In ``parent``, stage task ``owner`` and a conversation it owns holding task ``inner``, in one commit."""
    options = options or {}

    async def create(tx: Any) -> SimpleNamespace:
        owner = await tx.createTask(
            Hold,
            {"name": name, **({"slowAbort": True} if options.get("slowOwner") is True else {})},
            {"ownership": {"kind": "conversation"}, "background": options.get("background", False)},
        )
        child = await tx.createConversation({"ownership": {"kind": "task", "taskId": owner}})
        inner = await tx.createTask(
            Hold,
            {"name": f"{name}.inner", **({"slowAbort": True} if options.get("slowInner") is True else {})},
            {"ownership": {"kind": "conversation"}, "conversationId": child["id"]},
        )
        return SimpleNamespace(owner=owner, child=child["id"], inner=inner)

    return await parent.commit(create, context)  # type: ignore[no-any-return]


def gated(message: Any) -> SimpleNamespace:
    """A faux response held until ``release`` or cancellation; ``reached`` resolves when the request is sent."""
    reached = deferred()
    release = deferred()

    async def step(_request: Any, options: Any, *_: Any) -> Any:
        reached.resolve()
        await race(release.promise, aborted(options.signal))
        return message

    return SimpleNamespace(step=step, reached=reached.promise, release=lambda: release.resolve())


async def waitUntil(check: Callable[[], Any]) -> None:
    await waitFor(check)


async def busy(conversation: Any, task: int) -> None:
    """Mark a conversation busy with ``task`` standing in for its run, so submissions queue."""

    async def mark(tx: Any) -> None:
        (await tx.doc(LiveDoc, conversation.id))["run"] = {"taskId": task, "inputs": []}

    await conversation.commit(mark, context)


async def status(harness: Any, id: int) -> dict[str, Any]:
    return (await harness.getTask(id, context))["state"]  # type: ignore[no-any-return]


async def openHarness(storage: Any = None) -> SimpleNamespace:
    harness, registry, reports = await openTasks(storage if storage is not None else MemoryStorage(), [Hold, Waiter])
    root = await harness.root(context)
    harness.resume()
    return SimpleNamespace(harness=harness, registry=registry, reports=reports, root=root)


def hasStatus(harness: Any, id: int, expected: str) -> Callable[[], Any]:
    async def check() -> bool:
        return (await status(harness, id))["status"] == expected

    return check


def abortRequested(harness: Any, id: int) -> Callable[[], Any]:
    async def check() -> bool:
        return bool((await harness.getTask(id, context))["abortRequested"])

    return check


async def test_keeps_a_conversation_busy_while_its_owned_foreground_subtree_has_live_work_holding_the_completed_owner():
    opened = await openHarness()
    harness, root = opened.harness, opened.root
    tree = await ownedChild(root, "owner")
    openGate("owner", "completed")
    await waitUntil(hasStatus(harness, tree.owner, "completing"))
    idle = root.waitForIdle(context)
    assert await settled(idle) is False
    assert await settled(harness.waitForIdle(context)) is False
    openGate("owner.inner", "completed")
    await idle
    await harness.waitForIdle(context)
    assert (await status(harness, tree.owner))["outcome"]["status"] == "completed"
    await harness.close(context)


async def test_stops_idle_traversal_at_a_background_owner():
    opened = await openHarness()
    harness, root = opened.harness, opened.root
    tree = await ownedChild(root, "background", {"background": True})
    await root.waitForIdle(context)
    await harness.waitForIdle(context)
    # The background child is its own scope.
    child = await harness.conversation(tree.child, context)
    assert await settled(child.waitForIdle(context)) is False
    await child.abort(context)
    await harness.close(context)


async def test_cascades_an_abort_mark_to_the_owned_foreground_subtree_and_withdraws_its_queued_inputs():
    opened = await openHarness()
    harness, root = opened.harness, opened.root
    tree = await ownedChild(root, "owner")
    nested = await harness.conversation(tree.child, context)
    deeper = await ownedChild(nested, "deeper")
    shielded = await ownedChild(nested, "shielded", {"background": True})
    # Busy conversations queue submissions; the inner task stands in for the child's run.
    await busy(nested, tree.inner)
    queued = (await nested.submit({"type": "input", "content": "later"}, context)).id
    assert await harness.abortTask(tree.owner, context) == "marked"
    for id in (tree.owner, tree.inner, deeper.owner, deeper.inner):
        assert (await harness.waitForTask(id, context))["state"]["outcome"]["status"] == "aborted"
    # A nested background owner is a boundary.
    assert (await status(harness, shielded.owner))["status"] != "terminal"
    assert (await status(harness, shielded.inner))["status"] != "terminal"
    toMatchObject(
        await (await harness.submission(queued, context)).status(context), {"status": "unanswered", "reason": "aborted"}
    )
    await harness.abortTask(shielded.owner, context)
    await harness.close(context)


async def test_cascades_a_failed_owner_but_not_a_completed_one():
    opened = await openHarness()
    harness, root = opened.harness, opened.root
    failed = await ownedChild(root, "failed")
    completedTree = await ownedChild(root, "done")
    openGate("failed", "failed")
    openGate("done", "completed")
    assert (await harness.waitForTask(failed.inner, context))["state"]["outcome"]["status"] == "aborted"
    assert (await harness.waitForTask(failed.owner, context))["state"]["outcome"]["status"] == "failed"
    await waitUntil(hasStatus(harness, completedTree.owner, "completing"))
    assert (await status(harness, completedTree.inner))["status"] != "terminal"
    openGate("done.inner", "completed")
    assert (await harness.waitForTask(completedTree.owner, context))["state"]["outcome"]["status"] == "completed"
    await harness.close(context)


async def test_aborts_a_background_task_directly_with_its_ordinary_subtree():
    opened = await openHarness()
    harness, root = opened.harness, opened.root
    tree = await ownedChild(root, "background", {"background": True})
    await harness.abortTask(tree.owner, context)
    assert (await harness.waitForTask(tree.inner, context))["state"]["outcome"]["status"] == "aborted"
    await harness.close(context)


async def test_aborts_a_conversation_queued_inputs_withdrawn_writes_kept_foreground_work_aborted_background_kept():
    opened = await openHarness()
    harness, root = opened.harness, opened.root
    foreground = await ownedChild(root, "foreground")
    background = await ownedChild(root, "background", {"background": True})
    await busy(root, foreground.owner)
    input = (await root.submit({"type": "input", "content": "later"}, context)).id
    write = (await root.submit({"type": "write", "entry": {"kind": "note"}}, context)).id
    await root.abort(context)
    for id in (foreground.owner, foreground.inner):
        assert (await status(harness, id))["status"] == "terminal"
    assert (await status(harness, background.owner))["status"] != "terminal"
    assert (await status(harness, background.inner))["status"] != "terminal"
    toMatchObject(await (await harness.submission(input, context)).status(context), {"reason": "aborted"})
    assert (await (await harness.submission(write, context)).status(context))["status"] == "queued"
    await harness.abortTask(background.owner, context)
    await harness.close(context)


async def test_aborts_work_created_below_a_held_failed_owner_but_not_below_a_terminal_one():
    opened = await openHarness()
    harness, root = opened.harness, opened.root
    tree = await ownedChild(root, "owner", {"slowInner": True})
    openGate("owner", "failed")
    await waitUntil(abortRequested(harness, tree.inner))
    assert (await status(harness, tree.owner))["status"] == "completing"
    child = await harness.conversation(tree.child, context)

    def create(name: str) -> Any:
        return child.commit(lambda tx: tx.createTask(Hold, {"name": name}, {"ownership": {"kind": "conversation"}}), context)

    during = await create("during")
    assert (await harness.waitForTask(during, context))["state"]["outcome"]["status"] == "aborted"
    openGate("abort.owner.inner", "completed")
    assert (await harness.waitForTask(tree.owner, context))["state"]["outcome"]["status"] == "failed"
    # A terminal owner never cascades: interrogating its conversation runs normally.
    after = await create("after")
    openGate("after", "completed")
    assert (await harness.waitForTask(after, context))["state"]["outcome"]["status"] == "completed"
    await harness.close(context)


async def test_withdraws_queued_inputs_below_the_aborted_task_but_keeps_its_own_conversations_queue_and_queued_writes():
    opened = await openHarness()
    harness, root = opened.harness, opened.root
    tree = await ownedChild(root, "owner")
    child = await harness.conversation(tree.child, context)
    await busy(root, tree.owner)
    await busy(child, tree.inner)
    own = (await root.submit({"type": "input", "content": "own"}, context)).id
    below = (await child.submit({"type": "input", "content": "below"}, context)).id
    write = (await child.submit({"type": "write", "entry": {"kind": "note"}}, context)).id
    await harness.abortTask(tree.owner, context)
    await harness.waitForTask(tree.inner, context)

    async def statusOf(id: int) -> str:
        return (await (await harness.submission(id, context)).status(context))["status"]  # type: ignore[no-any-return]

    assert await statusOf(own) == "queued"
    assert await statusOf(below) == "unanswered"
    assert await statusOf(write) == "queued"
    await harness.close(context)


async def test_keeps_a_nested_background_owners_subtree_when_its_cancelled_background_owner_is_aborted():
    opened = await openHarness()
    harness, root = opened.harness, opened.root
    outer = await ownedChild(root, "outer", {"background": True})
    inner = await ownedChild(await harness.conversation(outer.child, context), "inner", {"background": True})
    await harness.abortTask(outer.owner, context)
    assert (await harness.waitForTask(outer.inner, context))["state"]["outcome"]["status"] == "aborted"
    assert (await status(harness, inner.owner))["status"] != "terminal"
    assert (await status(harness, inner.inner))["status"] != "terminal"
    await harness.abortTask(inner.owner, context)
    await harness.close(context)


async def test_decides_idle_after_reopen_from_owner_edges_it_has_to_load_first(sqliteDirectory):
    path = os.path.join(sqliteDirectory, "session.sqlite")
    opened = await openHarness(await openNodeSqliteStorage(path))
    foreground = await ownedChild(opened.root, "fg")
    deeper = await ownedChild(await opened.harness.conversation(foreground.child, context), "fg2")
    background = await ownedChild(opened.root, "bg", {"background": True})
    openGate("fg", "completed")
    openGate("fg2", "completed")
    # Both owners hold their outcomes while the work below them runs.
    await waitUntil(hasStatus(opened.harness, foreground.owner, "completing"))
    await waitUntil(hasStatus(opened.harness, deeper.owner, "completing"))
    await opened.harness.close(context)

    opened = await openHarness(await openNodeSqliteStorage(path))
    # Two levels below completed foreground owners, the inner tasks keep the root busy.
    assert await settled(opened.root.waitForIdle(context)) is False
    assert await settled(opened.harness.waitForIdle(context)) is False
    openGate("fg.inner", "completed")
    openGate("fg2.inner", "completed")
    # The background subtree does not count.
    await opened.root.waitForIdle(context)
    await opened.harness.waitForIdle(context)
    assert (await status(opened.harness, foreground.owner))["status"] == "terminal"
    assert (await status(opened.harness, background.inner))["status"] != "terminal"
    await opened.harness.abortTask(background.owner, context)
    await opened.harness.close(context)


@pytest.mark.parametrize(
    ("label", "abortRequested", "state"),
    [
        ("a held failed owner", False, {"status": "completing", "outcome": {"status": "failed", "error": {"message": "crash"}}}),
        ("an abort-marked owner", True, {"status": "pending", "checkpoint": {"phase": "hold"}}),
    ],
)
async def test_derives_marks_a_crash_left_unapplied_below_a_cancelled_owner_at_open(
    sqliteDirectory, label, abortRequested, state
):
    path = os.path.join(sqliteDirectory, "session.sqlite")
    # Without a Harness, nothing derives marks: a cancelled owner with a live task below it.
    session = createSession(await openNodeSqliteStorage(path))

    async def build(tx: Any) -> SimpleNamespace:
        root = await tx.createConversation({"ownership": {"kind": "ownerless"}})
        owner = await tx.createTask(
            Hold, {"name": "gone"}, {"ownership": {"kind": "conversation"}, "conversationId": root["id"]}
        )
        child = await tx.createConversation({"ownership": {"kind": "task", "taskId": owner}})
        inner = await tx.createTask(
            Hold, {"name": "orphan"}, {"ownership": {"kind": "conversation"}, "conversationId": child["id"]}
        )
        return SimpleNamespace(owner=owner, inner=inner)

    tree = await session.commit(build, context)

    async def mark(tx: Any) -> None:
        record = await tx.task(tree.owner)
        tx.setTask({**record, "abortRequested": abortRequested, "state": state})

    await session.commit(mark, context)
    await session.close(context)

    opened = await openHarness(await openNodeSqliteStorage(path))
    harness = opened.harness
    assert (await harness.waitForTask(tree.inner, context))["state"]["outcome"]["status"] == "aborted"
    # The owner finishes only after the work below it.
    outcome = (await harness.waitForTask(tree.owner, context))["state"]["outcome"]["status"]
    assert outcome == ("aborted" if abortRequested else "failed")
    await harness.close(context)


async def test_withdraws_an_input_queued_below_a_held_failed_owner_after_its_cascade():
    opened = await openHarness()
    harness, root = opened.harness, opened.root
    tree = await ownedChild(root, "owner", {"slowInner": True})
    child = await harness.conversation(tree.child, context)
    # The inner task stands in for the child's run, which stays busy after the cascade.
    await busy(child, tree.inner)
    openGate("owner", "failed")
    await waitUntil(abortRequested(harness, tree.inner))
    late = await child.submit({"type": "input", "content": "late"}, context)
    toMatchObject(await late.wait(context), {"status": "unanswered", "reason": "aborted"})
    openGate("abort.owner.inner", "completed")
    await harness.waitForTask(tree.owner, context)
    await harness.close(context)


async def test_marks_work_admitted_after_reopen_below_a_cancelled_owner_whose_edge_was_not_loaded(sqliteDirectory):
    path = os.path.join(sqliteDirectory, "session.sqlite")
    opened = await openHarness(await openNodeSqliteStorage(path))
    tree = await ownedChild(opened.root, "owner", {"slowOwner": True})
    openGate("owner.inner", "completed")
    await opened.harness.waitForTask(tree.inner, context)
    # The owner stays live and cancelled in its slow abort handler.
    await opened.harness.abortTask(tree.owner, context)
    await opened.harness.close(context)

    # The child is empty at open, so nothing loads its edge until new work arrives.
    opened = await openHarness(await openNodeSqliteStorage(path))
    child = await opened.harness.conversation(tree.child, context)
    late = await child.commit(
        lambda tx: tx.createTask(Hold, {"name": "late"}, {"ownership": {"kind": "conversation"}}), context
    )
    assert (await opened.harness.waitForTask(late, context))["state"]["outcome"]["status"] == "aborted"
    openGate("abort.owner", "completed")
    assert (await opened.harness.waitForTask(tree.owner, context))["state"]["outcome"]["status"] == "aborted"
    await opened.harness.close(context)


async def test_cascades_from_an_owner_the_scheduler_orphans():
    opened = await openHarness()
    harness, root = opened.harness, opened.root

    async def create(tx: Any) -> SimpleNamespace:
        owner = await tx.createTask(Unregistered, {"name": "unregistered"}, {"ownership": {"kind": "conversation"}})
        child = await tx.createConversation({"ownership": {"kind": "task", "taskId": owner}})
        inner = await tx.createTask(
            Hold, {"name": "below"}, {"ownership": {"kind": "conversation"}, "conversationId": child["id"]}
        )
        return SimpleNamespace(owner=owner, inner=inner)

    tree = await root.commit(create, context)
    assert await harness.abortTask(tree.owner, context) == "marked"
    assert (await harness.waitForTask(tree.owner, context))["state"]["outcome"] == {
        "status": "orphaned",
        "reason": "missing_task",
    }
    assert (await harness.waitForTask(tree.inner, context))["state"]["outcome"]["status"] == "aborted"
    await harness.close(context)


async def test_cancels_only_the_callers_wait_never_the_shared_work():
    opened = await openHarness()
    harness, root = opened.harness, opened.root
    tree = await ownedChild(root, "owner")
    waiting = AbortController()
    idle = root.waitForIdle(withAbortSignal(waiting.signal, context))
    waiting.abort(RuntimeError("stop waiting"))
    with pytest.raises(Exception, match="stop waiting"):
        await idle
    assert (await status(harness, tree.inner))["status"] != "terminal"

    # Cancelling an abort after its commit leaves the marks in place.
    slow = await ownedChild(root, "slow")

    async def reinput(tx: Any) -> None:
        record = await tx.task(slow.owner)
        tx.setTask({**record, "input": {"name": "slow", "slowAbort": True}})

    await root.commit(reinput, context)
    aborting = AbortController()
    abort = root.abort(withAbortSignal(aborting.signal, context))
    await waitUntil(abortRequested(harness, slow.owner))
    aborting.abort(RuntimeError("stop aborting"))
    with pytest.raises(Exception, match="stop aborting"):
        await abort
    openGate("abort.slow", "completed")
    for id in (tree.owner, tree.inner, slow.owner, slow.inner):
        assert (await harness.waitForTask(id, context))["state"]["outcome"]["status"] == "aborted"
    await harness.close(context)


async def test_aborts_a_waiting_child_whose_awaited_task_completes_in_the_commit_that_marks_its_owner():
    opened = await openHarness()
    harness, root = opened.harness, opened.root

    async def create(tx: Any) -> SimpleNamespace:
        owner = await tx.createTask(Hold, {"name": "owner"}, {"ownership": {"kind": "conversation"}})
        child = await tx.createConversation({"ownership": {"kind": "task", "taskId": owner}})
        dependency = await tx.createTask(Hold, {"name": "dependency"}, {"ownership": {"kind": "conversation"}})
        blocked = await tx.createTask(
            Waiter, {"on": [dependency]}, {"ownership": {"kind": "conversation"}, "conversationId": child["id"]}
        )
        return SimpleNamespace(owner=owner, dependency=dependency, blocked=blocked)

    tree = await root.commit(create, context)
    await waitUntil(lambda: runs.get("dependency") == 1 and runs.get("owner") == 1)
    await waitUntil(hasStatus(harness, tree.blocked, "waiting"))

    # One commit completes the dependency and marks the owner.
    async def both(tx: Any) -> None:
        completedDependency = await tx.task(tree.dependency)
        marked = await tx.task(tree.owner)
        tx.setTask(
            {**completedDependency, "state": {"status": "terminal", "outcome": {"status": "completed", "result": None}}}
        )
        tx.setTask({**marked, "abortRequested": True})

    await root.commit(both, context)
    assert (await harness.waitForTask(tree.blocked, context))["state"]["outcome"]["status"] == "aborted"
    await harness.close(context)


async def test_retries_marks_found_through_an_edge_loaded_after_reopen_when_their_commit_is_rejected(sqliteDirectory):
    path = os.path.join(sqliteDirectory, "session.sqlite")
    opened = await openHarness(await openNodeSqliteStorage(path))
    tree = await ownedChild(opened.root, "owner", {"slowOwner": True})
    openGate("owner.inner", "completed")
    await opened.harness.waitForTask(tree.inner, context)
    await opened.harness.abortTask(tree.owner, context)
    await opened.harness.close(context)

    storage = await openNodeSqliteStorage(path)
    rejectMark: int | None = None
    commit = storage.commit

    async def rejecting(writes: Any, ctx: Any) -> Any:
        nonlocal rejectMark
        if any(
            write["type"] == "task" and write["value"]["id"] == rejectMark and write["value"]["abortRequested"]
            for write in writes
        ):
            rejectMark = None
            raise StorageRejected("rejected once")
        return await commit(writes, ctx)

    storage.commit = rejecting  # type: ignore[method-assign, assignment]
    opened = await openHarness(storage)
    child = await opened.harness.conversation(tree.child, context)

    async def createLate(tx: Any) -> int:
        nonlocal rejectMark
        id = await tx.createTask(Hold, {"name": "late"}, {"ownership": {"kind": "conversation"}})
        rejectMark = id
        return id  # type: ignore[no-any-return]

    late = await child.commit(createLate, context)
    await waitUntil(lambda: rejectMark is None)
    # Any later commit, here the reservation of the new task, retries the cascade.
    assert (await opened.harness.waitForTask(late, context))["state"]["outcome"]["status"] == "aborted"
    openGate("abort.owner", "completed")
    await opened.harness.close(context)


async def test_retries_a_cascade_whose_commit_the_storage_rejected():
    rejectMark: int | None = None

    class Rejecting(MemoryStorage):
        async def commit(self, writes: Any, ctx: Any) -> int:  # type: ignore[override]
            nonlocal rejectMark
            marks = any(
                write["type"] == "task" and write["value"]["id"] == rejectMark and write["value"]["abortRequested"]
                for write in writes
            )
            if marks:
                rejectMark = None
                raise StorageRejected("rejected once")
            return await super().commit(writes, ctx)

    opened = await openHarness(Rejecting())
    harness, root, reports = opened.harness, opened.root, opened.reports
    tree = await ownedChild(root, "owner")
    rejectMark = tree.inner
    openGate("owner", "failed")
    await waitUntil(lambda: any(isinstance(error, StorageRejected) for error in reports))
    assert (await status(harness, tree.owner))["status"] == "completing"
    assert (await status(harness, tree.inner))["status"] != "terminal"
    # The next commit retries the cascade.
    await root.commit(lambda tx: tx.appendEntry(root.id, {"kind": "note"}), context)
    assert (await harness.waitForTask(tree.inner, context))["state"]["outcome"]["status"] == "aborted"
    assert (await harness.waitForTask(tree.owner, context))["state"]["outcome"]["status"] == "failed"
    await harness.close(context)


EMPTY_OBJECT = {"type": "object", "properties": {}}


async def test_owned_gives_a_tool_an_invocation_bound_handle_whose_submissions_stay_durable_after_the_call_ends():
    setup = chatSetup()
    handle: Any = None
    missing: Any = object()
    submission: Any = None

    async def execute(_args: Any, api: Any, callContext: Any) -> dict[str, Any]:
        nonlocal handle, missing, submission

        async def create(tx: Any) -> int:
            created = await tx.createConversation({"ownership": {"kind": "task", "taskId": api.taskId}})
            await configure(tx, created["id"], {"model": {"provider": "faux", "modelId": "faux-1"}})
            return created["id"]  # type: ignore[no-any-return]

        child = await api.commit(create, callContext)
        missing = await api.conversation(99_999, callContext)
        handle = await api.conversation(child, callContext)
        submission = await handle.submit({"type": "input", "content": "child task", "requestId": "child"}, callContext)
        settledChild = await submission.wait(callContext)
        return {"content": [{"type": "text", "text": settledChild["status"]}]}

    addTool(
        setup.registry,
        defineTool({"name": "delegate", "description": "Delegates", "parameters": EMPTY_OBJECT, "execute": execute}),
    )
    setup.faux.setResponses(
        [
            fauxAssistantMessage([fauxToolCall("delegate", {}, {"id": "c1"})], {"stopReason": "toolUse"}),
            fauxAssistantMessage([fauxText("child answer")]),
            fauxAssistantMessage([fauxText("done")]),
        ]
    )
    opened = await openChat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    await (await root.submit({"type": "input", "content": "go"}, context)).wait(context)
    assert missing is None
    # The call ended: the handle and its submission reject, while the submission record remains.
    with pytest.raises(Exception, match="invocation has ended"):
        await handle.submit({"type": "input", "content": "again"}, context)
    with pytest.raises(Exception, match="invocation has ended"):
        await handle.waitForIdle(context)
    with pytest.raises(Exception, match="invocation has ended"):
        await handle.abort(context)
    with pytest.raises(Exception, match="invocation has ended"):
        await submission.wait(context)
    # Nothing was admitted or marked after the call ended.
    childConversation = await harness.conversation(handle.id, context)
    childEntries = (await childConversation.entries({}, 100, None, context))["items"]
    assert len([entry for entry in childEntries if entry["kind"] == "pi.user"]) == 1
    childTasks = [task for task in (await harness.inspect(context))["tasks"] if task["record"]["conversationId"] == handle.id]
    assert childTasks == []
    assert (await (await harness.submission(submission.id, context)).status(context))["status"] == "done"
    await harness.close(context)


async def test_owned_rejects_a_handle_operation_queued_on_the_line_when_the_invocation_ends_before_it_runs():
    setup = chatSetup()
    ready = deferred()
    go = deferred()
    queued = deferred()

    async def execute(_args: Any, api: Any, callContext: Any) -> Any:
        async def create(tx: Any) -> int:
            created = await tx.createConversation({"ownership": {"kind": "task", "taskId": api.taskId}})
            return created["id"]  # type: ignore[no-any-return]

        child = await api.commit(create, callContext)
        handle = await api.conversation(child, callContext)
        ready.resolve({"child": child, "taskId": api.taskId})
        await go.promise
        # Called while the invocation is alive, with a context that is not the call's: the handle binds it to the
        # invocation. It reaches the line only after the abort mark.
        submitting = handle.submit({"type": "input", "content": "late"}, context)
        submitting.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        queued.resolve({"submitting": submitting})
        return await aborted(callContext.abortSignal)

    addTool(
        setup.registry,
        defineTool({"name": "delegate", "description": "Delegates late", "parameters": EMPTY_OBJECT, "execute": execute}),
    )
    setup.faux.setResponses([fauxAssistantMessage([fauxToolCall("delegate", {}, {"id": "c1"})], {"stopReason": "toolUse"})])
    opened = await openChat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    await root.submit({"type": "input", "content": "go"}, context)
    started = await ready.promise
    child, taskId = started["child"], started["taskId"]
    # Hold the Session line, queue the abort mark behind it, then let the tool queue its submit.
    release = deferred()

    async def hold(tx: Any) -> None:
        await release.promise

    holding = root.commit(hold, context)
    aborting = harness.abortTask(taskId, context)
    go.resolve()
    submitting = (await queued.promise)["submitting"]
    release.resolve()
    await holding
    await aborting
    with pytest.raises(Exception):  # noqa: B017
        await submitting
    childConversation = await harness.conversation(child, context)
    assert (await childConversation.entries({}, 10, None, context))["items"] == []
    assert [record for record in (await harness.inspect(context))["submissions"] if record["conversationId"] == child] == []
    await harness.close(context)


async def test_owned_aborts_a_subagent_run_with_its_parents_conversation_rejecting_the_waiting_tool():
    setup = chatSetup()
    child: int | None = None
    toolWait = deferred()

    async def execute(_args: Any, api: Any, callContext: Any) -> dict[str, Any]:
        nonlocal child

        async def create(tx: Any) -> int:
            created = await tx.createConversation({"ownership": {"kind": "task", "taskId": api.taskId}})
            await configure(tx, created["id"], {"model": {"provider": "faux", "modelId": "faux-1"}})
            return created["id"]  # type: ignore[no-any-return]

        child = await api.commit(create, callContext)
        submission = await (await api.conversation(child, callContext)).submit(
            {"type": "input", "content": "child task"}, callContext
        )
        try:
            return {"content": [{"type": "text", "text": (await submission.wait(callContext))["status"]}]}
        except Exception as error:
            toolWait.resolve(str(error))
            raise

    addTool(
        setup.registry,
        defineTool({"name": "delegate", "description": "Delegates", "parameters": EMPTY_OBJECT, "execute": execute}),
    )
    childRun = gated(fauxAssistantMessage([fauxText("never")]))
    setup.faux.setResponses(
        [fauxAssistantMessage([fauxToolCall("delegate", {}, {"id": "c1"})], {"stopReason": "toolUse"}), childRun.step]
    )
    opened = await openChat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    input = await root.submit({"type": "input", "content": "go"}, context)
    await childRun.reached
    await root.abort(context)
    toMatchObject(await input.wait(context), {"status": "unanswered", "reason": "aborted"})
    assert await toolWait.promise
    childConversation = await harness.conversation(child, context)
    await childConversation.waitForIdle(context)
    childLive = await harness.snapshot(LiveDoc, child, context)
    assert childLive == {}
    childInputs = [s for s in (await harness.inspect(context))["submissions"] if s["conversationId"] == child]
    assert childInputs == []
    await harness.waitForIdle(context)
    await harness.close(context)


async def test_owned_lets_a_running_tool_abort_its_owned_child_and_continue():
    setup = chatSetup()
    childRun = gated(fauxAssistantMessage([fauxText("never")]))

    async def execute(_args: Any, api: Any, callContext: Any) -> dict[str, Any]:
        async def create(tx: Any) -> int:
            created = await tx.createConversation({"ownership": {"kind": "task", "taskId": api.taskId}})
            await configure(tx, created["id"], {"model": {"provider": "faux", "modelId": "faux-1"}})
            return created["id"]  # type: ignore[no-any-return]

        child = await api.commit(create, callContext)
        handle = await api.conversation(child, callContext)
        submission = await handle.submit({"type": "input", "content": "child task"}, callContext)
        await childRun.reached
        await handle.abort(callContext)
        settledChild = await submission.wait(callContext)
        return {"content": [{"type": "text", "text": f"child {settledChild['status']}"}]}

    addTool(
        setup.registry,
        defineTool(
            {"name": "delegate", "description": "Delegates and cancels", "parameters": EMPTY_OBJECT, "execute": execute}
        ),
    )
    setup.faux.setResponses(
        [
            fauxAssistantMessage([fauxToolCall("delegate", {}, {"id": "c1"})], {"stopReason": "toolUse"}),
            childRun.step,
            fauxAssistantMessage([fauxText("done")]),
        ]
    )
    opened = await openChat(MemoryStorage(), setup)
    harness, root = opened.harness, opened.root
    assert (await (await root.submit({"type": "input", "content": "go"}, context)).wait(context))["status"] == "done"
    await harness.close(context)


async def test_owned_aborts_the_children_of_a_tool_call_that_throws_or_is_interrupted_while_the_run_continues(
    sqliteDirectory,
):
    path = os.path.join(sqliteDirectory, "session.sqlite")
    children: list[int] = []
    toolRunning = deferred()
    setup = chatSetup()
    addTask(setup.registry, Hold)

    async def execute(args: Any, api: Any, callContext: Any) -> Any:
        async def create(tx: Any) -> int:
            created = await tx.createConversation({"ownership": {"kind": "task", "taskId": api.taskId}})
            return await tx.createTask(  # type: ignore[no-any-return]
                Hold,
                {"name": f"child.{api.callId}"},
                {"ownership": {"kind": "conversation"}, "conversationId": created["id"]},
            )

        child = await api.commit(create, callContext)
        children.append(child)
        if args["mode"] == "throw":
            raise RuntimeError("spawn failed")
        toolRunning.resolve()
        return await aborted(callContext.abortSignal)

    addTool(
        setup.registry,
        defineTool(
            {
                "name": "spawn",
                "description": "Starts owned work, then throws or hangs",
                "parameters": {"type": "object", "properties": {"mode": {"type": "string"}}, "required": ["mode"]},
                "execute": execute,
            }
        ),
    )

    def call(mode: str, id: str) -> Any:
        return fauxAssistantMessage([fauxToolCall("spawn", {"mode": mode}, {"id": id})], {"stopReason": "toolUse"})

    setup.faux.setResponses([call("throw", "c1"), fauxAssistantMessage([fauxText("after throw")]), call("hang", "c2")])
    opened = await openChat(await openNodeSqliteStorage(path), setup)
    assert (await (await opened.root.submit({"type": "input", "content": "one"}, context)).wait(context))["status"] == "done"
    assert (await opened.harness.waitForTask(children[0], context))["state"]["outcome"]["status"] == "aborted"

    # The second call hangs until the process stops; on reopen it is interrupted.
    second = (await opened.root.submit({"type": "input", "content": "two"}, context)).id
    await toolRunning.promise
    await opened.harness.close(context)
    setup.faux.setResponses([fauxAssistantMessage([fauxText("after interrupt")])])
    opened = await openChat(await openNodeSqliteStorage(path), setup)
    opened.harness.resume()
    assert (await opened.harness.waitForTask(children[1], context))["state"]["outcome"]["status"] == "aborted"
    assert (await (await opened.harness.submission(second, context)).wait(context))["status"] == "done"
    tools = (await opened.harness.commit(lambda tx: tx.scanTasks({"kind": "pi.tool"}, 10), context))["items"]
    assert [task["state"]["status"] == "terminal" and task["state"]["outcome"]["status"] for task in tools] == [
        "failed",
        "failed",
    ]
    await opened.harness.close(context)


async def test_owned_reruns_a_replay_safe_subagent_tool_after_a_restart_with_the_same_child_and_submission(
    sqliteDirectory,
):
    path = os.path.join(sqliteDirectory, "session.sqlite")
    children: list[int] = []
    setup = chatSetup()

    async def execute(_args: Any, api: Any, callContext: Any) -> dict[str, Any]:
        async def create(tx: Any) -> int:
            existing = (await tx.scanConversations({"ownerTaskId": api.taskId}, 1))["items"]
            if existing:
                return existing[0]["id"]  # type: ignore[no-any-return]
            created = await tx.createConversation({"ownership": {"kind": "task", "taskId": api.taskId}})
            await configure(tx, created["id"], {"model": {"provider": "faux", "modelId": "faux-1"}})
            return created["id"]  # type: ignore[no-any-return]

        child = await api.commit(create, callContext)
        children.append(child)
        request = {"type": "input", "content": "child task", "requestId": f"subagent:{api.taskId}"}
        submission = await (await api.conversation(child, callContext)).submit(request, callContext)
        settledChild = await submission.wait(callContext)
        return {"content": [{"type": "text", "text": settledChild["status"]}]}

    addTool(
        setup.registry,
        defineTool(
            {
                "name": "subagent",
                "description": "Delegates",
                "parameters": EMPTY_OBJECT,
                "replay": "safe",
                "execute": execute,
            }
        ),
    )
    childRun = gated(fauxAssistantMessage([fauxText("never")]))
    setup.faux.setResponses(
        [fauxAssistantMessage([fauxToolCall("subagent", {}, {"id": "c1"})], {"stopReason": "toolUse"}), childRun.step]
    )
    opened = await openChat(await openNodeSqliteStorage(path), setup)
    input = (await opened.root.submit({"type": "input", "content": "go"}, context)).id
    await childRun.reached
    await opened.harness.close(context)

    setup.faux.setResponses([fauxAssistantMessage([fauxText("child answer")]), fauxAssistantMessage([fauxText("done")])])
    opened = await openChat(await openNodeSqliteStorage(path), setup)
    assert (await (await opened.harness.submission(input, context)).wait(context))["status"] == "done"
    assert len(children) == 2
    assert children[1] == children[0]
    child = await opened.harness.conversation(children[0], context)
    entries = (await child.entries({}, 100, None, context))["items"]
    assert len([entry for entry in entries if entry["kind"] == "pi.user"]) == 1
    results = [message for message in (await opened.root.context(context))["messages"] if message["role"] == "toolResult"]
    assert [message["isError"] for message in results] == [False]
    await opened.harness.close(context)


async def test_owned_lets_a_background_supervisor_resubmit_after_a_restart_without_submitting_twice(sqliteDirectory):
    path = os.path.join(sqliteDirectory, "session.sqlite")
    Children = defineDoc(
        {
            "kind": "test.children",
            "version": 1,
            "scope": "conversation",
            "history": "latest",
            "fork": "initial",
            "initial": dict,
        }
    )

    async def run(task: Any, runtime: Any, ctx: Any) -> None:
        id = (await runtime.snapshot(Children, task["input"]["parent"], ctx))["child"]
        child = await runtime.conversation(id, ctx)
        submission = await child.submit({"type": "input", "content": "work", "requestId": "stable"}, ctx)
        settledChild = await submission.wait(ctx)
        if settledChild["status"] != "done" or settledChild["type"] != "input":
            raise RuntimeError(settledChild["status"])
        await runtime.commit(
            lambda tx, current: {
                "status": "terminal",
                "outcome": {"status": "completed", "result": {"answer": settledChild["answer"]}},
            },
            ctx,
        )

    async def abort(_task: Any, runtime: Any, ctx: Any) -> None:
        await runtime.commit(lambda tx, current: {"status": "terminal", "outcome": {"status": "aborted"}}, ctx)

    Supervisor = defineTask(
        {
            "name": "test.supervisor",
            "version": 1,
            "initial": lambda input: {"phase": "run"},
            "phases": {"run": run},
            "abort": abort,
        }
    )
    setup = chatSetup()
    addTask(setup.registry, Supervisor)
    first = gated(fauxAssistantMessage([fauxText("never")]))
    setup.faux.setResponses([first.step, fauxAssistantMessage([fauxText("answer")])])
    opened = await openChat(await openNodeSqliteStorage(path), setup)
    root = opened.root

    async def create(tx: Any) -> SimpleNamespace:
        supervisor = await tx.createTask(
            Supervisor, {"parent": root.id}, {"ownership": {"kind": "conversation"}, "background": True}
        )
        created = await tx.createConversation({"ownership": {"kind": "task", "taskId": supervisor}})
        await configure(tx, created["id"], {"model": {"provider": "faux", "modelId": "faux-1"}})
        (await tx.doc(Children, root.id))["child"] = created["id"]
        return SimpleNamespace(supervisor=supervisor, child=created["id"])

    created = await root.commit(create, context)
    opened.harness.resume()
    # The submission is admitted and its generation requested; then the process stops.
    await first.reached
    await opened.harness.close(context)

    opened = await openChat(await openNodeSqliteStorage(path), setup)
    opened.harness.resume()
    done = await opened.harness.waitForTask(created.supervisor, context)
    assert done["state"]["outcome"]["status"] == "completed"
    childConversation = await opened.harness.conversation(created.child, context)
    entries = (await childConversation.entries({}, 100, None, context))["items"]
    assert len([entry for entry in entries if entry["kind"] == "pi.user"]) == 1
    # The root never waited for the background supervisor.
    await opened.root.waitForIdle(context)
    await opened.harness.close(context)
