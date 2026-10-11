"""pi durable's ``test/chord-guide.test.ts``.

The code of docs/pico-v5-chord-usage.md, run against a real Harness. Keep the two in sync: the guide's blocks are copied
here apart from formatting and the output sinks, which record instead of printing.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from misaka.chord import (
    createFacetHost,
    createRemoteServiceBinding,
    defineFacet,
    defineService,
)
from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.durable import MemoryStorage, defineDoc, defineDocFamily, defineTask
from misaka.utils.js import jsonStringify
from tests.durable_support import context, toMatchObject
from tests.task_support import completed, deferred, eventually, openTasks

logged: list[list[Any]] = []


def log(*values: Any) -> None:
    logged.append(list(values))


# ─── 1. A Session-wide canvas ────────────────────────────────────────────────

CanvasDoc = defineDoc(
    {
        "kind": "app.canvas",
        "version": 1,
        "scope": "session",
        "initial": lambda: {"strokes": []},
        # Store a complete base after at most 99 replayed deltas.
        "checkpointWhen": lambda _value, _ops, info: info["deltasSinceBase"] >= 99,
    }
)

Canvas: Any = defineService("app.canvas")


async def createCanvasFacet(session: Any, context: Any) -> Any:
    # Creation is explicit; observation never writes.
    async def create(tx: Any) -> None:
        await tx.doc(CanvasDoc)

    await session.commit(create, context)
    state = await session.documentState(CanvasDoc, context)
    if state is None:
        raise RuntimeError("canvas was retired during setup")

    def setup(env: Any) -> None:
        env.own(lambda: state.dispose())

        async def addStroke(stroke: dict[str, Any], callContext: Any) -> None:
            async def change(tx: Any) -> None:
                draft = await tx.doc(CanvasDoc)
                draft["strokes"].push(stroke)  # Chord copies the assigned stroke by value.

            await session.commit(change, callContext)

        env.provide(Canvas, {"state": state, "addStroke": addStroke})

    return defineFacet(SimpleNamespace(id="app.canvas/session", setup=setup))


def setupCanvasConsumer(env: Any) -> None:
    canvas = env.use(Canvas)  # Declare now; access only after activation.

    def activate() -> None:
        # subscribe delivers the current hydrated value, then updates.
        env.own(
            canvas.state.subscribe(
                lambda value, _context, delivery: log(
                    delivery["kind"], delivery["sequence"], "retired" if value is None else len(value["strokes"])
                )
            )
        )

    env.onActivate(activate)


CanvasConsumer = defineFacet(SimpleNamespace(id="app.canvas/consumer", setup=setupCanvasConsumer))


async def runCanvasExample(session: Any) -> None:
    callContext = BACKGROUND_CONTEXT
    provider = await createCanvasFacet(session, callContext)
    host = await createFacetHost({"facets": [provider, CanvasConsumer]})
    try:
        await host.services.use(Canvas).addStroke(
            {"color": "black", "points": [{"x": 10, "y": 20}, {"x": 30, "y": 40}]}, callContext
        )
    finally:
        await host.dispose()  # Unsubscribes; does not delete the canvas or close Session.


async def connectCanvas(transport: Any, callContext: Any) -> Any:
    services = createRemoteServiceBinding({"services": [Canvas], "transport": transport})
    canvas = services.use(Canvas)
    stop = canvas.state.subscribe(
        lambda value, _context, delivery: log(
            delivery["kind"], delivery["sequence"], "retired" if value is None else value["strokes"]
        )
    )
    try:
        await services.ready(callContext)  # Initial snapshot installed; not all future updates.
    except BaseException:
        stop()
        await services.dispose(BACKGROUND_CONTEXT)
        raise

    async def detach() -> None:
        stop()
        await services.dispose(BACKGROUND_CONTEXT)

    return detach


async def shutdown(host: Any, detachClients: Any, harness: Any, callContext: Any) -> None:
    await detachClients()
    await host.dispose()
    await harness.close(callContext)


# ─── 2. Conversation-scoped diff reviews ─────────────────────────────────────

ReviewDoc = defineDocFamily(
    {
        "kind": "app.diff-review",
        "version": 1,
        "family": True,
        "scope": "conversation",
        "history": "latest",
        "fork": "current",
        "initial": lambda seed: {"path": seed["path"], "patch": seed["patch"], "comments": []},
        "checkpointWhen": lambda _value, _ops, info: info["deltasSinceBase"] >= 49,
    }
)
DiffReviews: Any = defineService("app.diff-reviews")


def reviewFacet(session: Any, conversationId: int, reviews: list[dict[str, Any]], callContext: Any) -> Any:
    def setup(env: Any) -> None:
        instances = env.provideMany(DiffReviews)

        async def activate() -> None:
            for review in reviews:

                async def create(tx: Any, review: dict[str, Any] = review) -> None:
                    await tx.doc(ReviewDoc, conversationId, review["key"], review["seed"])

                await session.commit(create, callContext)
                state = await session.documentState(ReviewDoc, conversationId, review["key"], callContext)
                if state is None:
                    raise RuntimeError("review was retired during setup")
                env.own(lambda state=state: state.dispose())

                async def identity(*_: Any, review: dict[str, Any] = review) -> dict[str, Any]:
                    return {"conversationId": conversationId, "key": review["key"]}

                async def addComment(comment: dict[str, Any], commentContext: Any, review: dict[str, Any] = review) -> None:
                    async def change(tx: Any) -> None:
                        draft = await tx.doc(ReviewDoc, conversationId, review["key"], review["seed"])
                        draft["comments"].push(comment)  # Chord copies the assigned comment by value.

                    await session.commit(change, commentContext)

                # Chord instance keys route services; they are not numeric document incarnation IDs.
                instances.spawn(
                    jsonStringify([conversationId, review["key"]]),
                    {"state": state, "identity": identity, "addComment": addComment},
                )  # The facet owns spawned service lifetimes automatically.

        env.onActivate(activate)

    return defineFacet(SimpleNamespace(id="app.diff-reviews/session", setup=setup))


# ─── 3. Task-scoped output and a tool/task watch ─────────────────────────────

sent: list[Any] = []
rendered: list[str] = []


async def sendCommittedFrame(value: Any, ops: Any) -> None:
    sent.append({"value": value, "ops": ops})


async def render(text: str) -> None:
    rendered.append(text)


JobOutputDoc = defineDoc(
    {
        "kind": "app.job-output",
        "version": 1,
        "scope": "task",
        "initial": lambda: {"stdout": "", "chunks": 0},
        "checkpointWhen": lambda _value, _ops, info: info["deltasSinceBase"] >= 99,
    }
)


async def appendJobOutput(runtime: Any, chunk: str, callContext: Any) -> None:
    # Read process output outside this callback. The runtime gates the live task.
    async def change(tx: Any, current: Any) -> None:
        draft = await tx.doc(JobOutputDoc, runtime.taskId)
        draft["stdout"] = (draft["stdout"] + chunk)[-50_000:]
        draft["chunks"] = draft["chunks"] + 1

    await runtime.commit(change, callContext)


async def observeJob(api: Any, producerTaskId: int, finished: Any, callContext: Any) -> None:
    watch = await api.watchDoc(JobOutputDoc, producerTaskId, callContext)
    if watch is None:
        return
    log("retired" if watch.value is None else watch.value["stdout"])
    try:

        async def onFrame(value: Any, ops: Any, _context: Any) -> None:
            await sendCommittedFrame(value, ops)
            await render("retired" if value is None else value["stdout"])

        watch.start(onFrame)  # Serialized callbacks never overlap.
        await finished  # Caller-supplied observation lifetime; outside any commit.
    finally:
        await watch.stop()  # Idempotent; prevents another callback from starting.


# ─── Runs ────────────────────────────────────────────────────────────────────


class InProcessSubscription:
    def __init__(self, subscription: Any) -> None:
        self._subscription = subscription
        self.snapshot = subscription.snapshot

    def activate(self) -> None:
        self._subscription.activate()

    def close(self, *_: Any) -> Any:
        return self._subscription.close()


class InProcessTransport:
    """In-process transport to a host's service provider; a real client puts a socket between the two."""

    def __init__(self, host: Any) -> None:
        self._host = host

    def invoke(self, call: Any, callContext: Any) -> Any:
        return self._host.services.invoke(call, callContext)

    async def subscribe(self, serviceId: str, mode: Any, listener: Any, callContext: Any) -> Any:
        return InProcessSubscription(self._host.services.subscribe(serviceId, mode, listener))


async def test_runs_the_canvas_facet_host_a_remote_client_withdrawal_and_detach_before_harness_close():
    logged.clear()
    harness, _registry, _reports = await openTasks(MemoryStorage(), [])
    await runCanvasExample(harness)
    assert logged == [["hydrate", 0, 0], ["update", 1, 1]]
    assert len((await harness.snapshot(CanvasDoc, context))["strokes"]) == 1

    # A worker installs the canvas facet again; a late remote client hydrates the stroke, then follows updates.
    logged.clear()
    host = await createFacetHost({"facets": [await createCanvasFacet(harness, context)]})
    detach = await connectCanvas(InProcessTransport(host), context)
    await host.services.use(Canvas).addStroke({"color": "red", "points": []}, context)
    await eventually(lambda: len(logged) == 2)
    assert [[kind, len(strokes)] for kind, _, strokes in logged] == [["hydrate", 1], ["update", 2]]
    # Record the shutdown order: clients detach and services withdraw while the Harness is still open.
    order: list[str] = []

    class TrackedHarness:
        def __getattr__(self, name: str) -> Any:
            return getattr(harness, name)

        async def close(self, closeContext: Any) -> None:
            order.append("close start")
            await harness.close(closeContext)
            order.append("close end")

    async def disposeHost() -> None:
        order.append("dispose start")
        await host.dispose()
        order.append("dispose end")

    trackedHost = SimpleNamespace(services=host.services, reload=lambda facets: host.reload(facets), dispose=disposeHost)

    async def detachClients() -> None:
        order.append("detach start")
        await detach()
        order.append("detach end")

    await shutdown(trackedHost, detachClients, TrackedHarness(), context)
    assert order == ["detach start", "detach end", "dispose start", "dispose end", "close start", "close end"]
    with pytest.raises(Exception, match="disposed"):
        host.services.use(Canvas)
    with pytest.raises(Exception, match="closed"):
        await harness.commit(lambda tx: None, context)


async def test_runs_the_diff_reviews_with_a_keyed_consumer():
    harness, _registry, _reports = await openTasks(MemoryStorage(), [])
    root = await harness.root(context)
    reviews = [{"key": "review-7", "seed": {"path": "a.ts", "patch": "-old\n+new"}}]
    seen: list[dict[str, Any]] = []

    def setupConsumer(env: Any) -> None:
        async def observe(review: Any, callContext: Any) -> None:
            key = (await review.identity(callContext))["key"]
            review.state.subscribe(
                lambda value, *_: seen.append({"key": key, "comments": -1 if value is None else len(value["comments"])})
            )
            await review.addComment({"id": "c1", "line": 1, "text": "why?"}, callContext)

        env.observe(DiffReviews, observe)

    consumer = defineFacet(SimpleNamespace(id="app.diff-reviews/consumer", setup=setupConsumer))
    host = await createFacetHost({"facets": [reviewFacet(harness, root.id, reviews, context), consumer]})
    await eventually(lambda: any(frame["comments"] == 1 for frame in seen))
    assert seen == [{"key": "review-7", "comments": 0}, {"key": "review-7", "comments": 1}]
    toMatchObject(
        await harness.snapshot(ReviewDoc, root.id, "review-7", context),
        {"path": "a.ts", "comments": [{"id": "c1"}]},
    )
    await host.dispose()
    await harness.close(context)


async def test_runs_the_job_output_watch_until_the_producer_retires_its_document():
    logged.clear()
    sent.clear()
    rendered.clear()
    gate = deferred()

    async def running(task: Any, runtime: Any, ctx: Any) -> None:
        await appendJobOutput(runtime, f"$ {task['input']['command']}\n", ctx)
        await gate.promise
        await appendJobOutput(runtime, "ok\n", ctx)
        await runtime.commit(lambda tx, current: completed(None), ctx)

    async def noAbort(_task: Any, runtime: Any, ctx: Any) -> None:
        return None

    Job = defineTask(
        {
            "name": "app.job",
            "version": 1,
            "initial": lambda input: {"phase": "running"},
            "phases": {"running": running},
            "abort": noAbort,
        }
    )
    harness, _registry, _reports = await openTasks(MemoryStorage(), [Job])
    root = await harness.root(context)
    id = await root.commit(
        lambda tx: tx.createTask(Job, {"command": "make"}, {"ownership": {"kind": "conversation"}}), context
    )
    harness.resume()

    async def firstChunk() -> bool:
        output = await harness.snapshot(JobOutputDoc, id, context)
        return output is not None and output["chunks"] == 1

    await eventually(firstChunk)

    async def waitFinished() -> None:
        await harness.waitForTask(id, context)

    finished = asyncio.ensure_future(waitFinished())
    observing = asyncio.ensure_future(observeJob(harness, id, finished, context))
    await eventually(lambda: len(logged) == 1)
    gate.resolve()
    await observing
    assert logged == [["$ make\n"]]
    # Stopped once the task finished; the retirement frame may or may not have been delivered before.
    assert rendered[:1] == ["$ make\nok\n"]
    assert await harness.watchDoc(JobOutputDoc, id, context) is None
    await harness.close(context)
