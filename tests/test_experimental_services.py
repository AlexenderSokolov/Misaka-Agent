"""pi coding-agent's ``test/experimental-agent-controller.test.ts``, ``test/experimental-transcript-provider.test.ts`` and
``test/experimental-slash-commands.test.ts``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from misaka.ai.providers.faux import fauxAssistantMessage
from misaka.chord import createFacetHost, defineFacet
from misaka.chord.context import BACKGROUND_CONTEXT
from misaka.experimental.services.agent_controller import AgentController
from misaka.experimental.services.agent_controller_provider import createAgentController
from misaka.experimental.services.slash_commands import SlashCommands
from misaka.experimental.services.slash_commands_provider import (
    SlashCommandRegistry,
    createSlashCommandsRuntimeFacet,
)
from misaka.experimental.services.transcript import Transcript
from misaka.experimental.services.transcript_provider import (
    createTranscriptServiceFacet,
)
from tests.chat_support import waitFor
from tests.experimental_durable_support import openFauxConversation, pendingResponse


def facet(id_: str, setup: Any) -> Any:
    return defineFacet(SimpleNamespace(id=id_, setup=setup))


# ─── AgentController service ─────────────────────────────────────────────────


async def test_prompts_the_root_conversation_through_the_service_catalogue():
    opened = await openFauxConversation([fauxAssistantMessage("hello back")])
    host = await createFacetHost(
        {
            "facets": [
                facet(
                    "test-agent-controller",
                    lambda env: env.provide(AgentController, createAgentController(opened.harness, opened.conversation)),
                )
            ]
        }
    )
    try:
        assert list(host.services.catalogue) == [{"serviceId": AgentController.id, "mode": "singleton"}]
        response = await host.services.invoke(
            {"serviceId": AgentController.id, "member": "prompt", "args": [{"message": "hello", "images": None}]},
            BACKGROUND_CONTEXT,
        )
        assert response["accepted"] is True
        assert isinstance(response["operationId"], str)
        assert response["error"] is None
        assert await host.services.invoke(
            {"serviceId": AgentController.id, "member": "waitForPrompt", "args": [response["operationId"]]},
            BACKGROUND_CONTEXT,
        ) == {"status": "done", "text": "hello back", "reason": None}
    finally:
        await host.dispose()
        await opened.close()


async def test_rejects_a_prompt_while_busy_and_queues_steering_and_follow_up_input():
    pending = pendingResponse()
    opened = await openFauxConversation([pending.step])
    controller = createAgentController(opened.harness, opened.conversation)
    try:
        first = await controller.prompt({"message": "first", "images": None}, BACKGROUND_CONTEXT)
        assert first["accepted"] is True
        await pending.reached

        second = await controller.prompt({"message": "second", "images": None}, BACKGROUND_CONTEXT)
        assert second["accepted"] is False
        assert second["operationId"] is None
        assert second["error"]["code"] == "busy"
        assert "busy" in second["error"]["message"]
        steer = await controller.steer({"message": "steer", "images": None}, BACKGROUND_CONTEXT)
        followUp = await controller.followUp({"message": "later", "images": None}, BACKGROUND_CONTEXT)
        assert steer["accepted"] is True and isinstance(steer["entryId"], str) and steer["error"] is None
        assert followUp["accepted"] is True and isinstance(followUp["entryId"], str) and followUp["error"] is None
        inbox = await opened.conversation.viewState(BACKGROUND_CONTEXT)
        try:
            assert [item["mode"] for item in inbox.value["docs"]["pi.inbox"]["items"]] == ["steer", "followUp"]
        finally:
            inbox.dispose()

        assert await controller.cancelQueued(followUp["entryId"], BACKGROUND_CONTEXT) == {"outcome": "cancelled"}
        assert await controller.cancelQueued(followUp["entryId"], BACKGROUND_CONTEXT) == {"outcome": "already_consumed"}
        assert await controller.cancelQueued("999", BACKGROUND_CONTEXT) == {"outcome": "not_found"}
        assert await controller.cancelQueued("not-an-id", BACKGROUND_CONTEXT) == {"outcome": "not_found"}

        await controller.abort(BACKGROUND_CONTEXT)
        view = await opened.conversation.viewState(BACKGROUND_CONTEXT)
        try:
            assert view.value["docs"].get("pi.live", {}).get("run") is None
        finally:
            view.dispose()
        result = await controller.waitForPrompt(first["operationId"], BACKGROUND_CONTEXT)
        assert result["status"] == "unanswered"
        assert result["text"] is None
    finally:
        await opened.close()


async def test_starts_a_compaction_task():
    opened = await openFauxConversation()
    controller = createAgentController(opened.harness, opened.conversation)
    try:
        response = await controller.compact({"customInstructions": "short"}, BACKGROUND_CONTEXT)
        assert response["accepted"] is True
        assert isinstance(response["operationId"], str)
        assert response["error"] is None
    finally:
        await opened.close()


async def test_reports_an_unknown_prompt():
    opened = await openFauxConversation()
    controller = createAgentController(opened.harness, opened.conversation)
    try:
        with pytest.raises(Exception, match="Unknown prompt: 999"):
            await controller.waitForPrompt("999", BACKGROUND_CONTEXT)
        with pytest.raises(Exception, match="Unknown prompt: nope"):
            await controller.waitForPrompt("nope", BACKGROUND_CONTEXT)
    finally:
        await opened.close()


async def test_sends_images_with_the_prompt_text():
    opened = await openFauxConversation([fauxAssistantMessage("seen")])
    controller = createAgentController(opened.harness, opened.conversation)
    try:
        image = {"type": "image", "data": "abcd", "mimeType": "image/png"}
        response = await controller.prompt({"message": "look", "images": [image]}, BACKGROUND_CONTEXT)
        await controller.waitForPrompt(response["operationId"], BACKGROUND_CONTEXT)
        page = await opened.conversation.entries({}, 10, None, BACKGROUND_CONTEXT)
        user = page["items"][-1]["model"][0]
        assert user["content"] == [{"type": "text", "text": "look"}, image]
    finally:
        await opened.close()


# ─── Transcript service ──────────────────────────────────────────────────────


async def test_replicates_the_conversation_view_as_it_changes():
    opened = await openFauxConversation([fauxAssistantMessage("answer")])
    views: list[Any] = []

    def setupConsumer(env: Any) -> None:
        transcript = env.use(Transcript)
        env.onActivate(lambda: env.own(transcript.state.subscribe(lambda value, *_: views.append(value))))

    host = await createFacetHost(
        {
            "facets": [
                await createTranscriptServiceFacet(opened.conversation, BACKGROUND_CONTEXT),
                facet("test-transcript-consumer", setupConsumer),
            ]
        }
    )
    try:
        assert views[0]["entries"] == []
        submission = await opened.conversation.submit({"type": "input", "content": "question"}, BACKGROUND_CONTEXT)
        await submission.wait(BACKGROUND_CONTEXT)
        await waitFor(lambda: [entry["kind"] for entry in views[-1]["entries"]] == ["pi.user", "pi.assistant"])
    finally:
        await host.dispose()
        await opened.close()


# ─── Slash commands ──────────────────────────────────────────────────────────


def names(commands: Any) -> list[str]:
    return [command["name"] for command in commands]


def test_registers_and_removes_contributions():
    registry = SlashCommandRegistry()
    snapshots: list[list[str]] = []
    unsubscribe = registry.subscribe(lambda commands: snapshots.append(names(commands)))
    close = registry.register({"name": "hello", "description": "Hello", "run": lambda *_: None})
    assert names(registry.list()) == ["hello"]
    with pytest.raises(Exception, match="already registered"):
        registry.register({"name": "hello", "run": lambda *_: None})
    close()
    close()
    assert registry.list() == []
    assert snapshots == [[], ["hello"], []]
    unsubscribe()


def test_rejects_invalid_command_names():
    registry = SlashCommandRegistry()
    for name in ("", "Hello", "-x", "a b", ":x"):
        with pytest.raises(TypeError, match="Invalid slash command name"):
            registry.register({"name": name, "run": lambda *_: None})


def test_stages_replacements_until_the_previous_registration_retires():
    registry = SlashCommandRegistry()
    first = {"name": "hello", "description": "First", "run": lambda *_: None}
    second = {"name": "hello", "description": "Second", "run": lambda *_: None}
    closeFirst = registry.register(first)
    closeSecond = registry.replace(second)
    assert registry.list() == [first]

    closeSecond()
    assert registry.list() == [first]
    closeReplacement = registry.replace(second)
    closeFirst()
    assert registry.list() == [second]
    closeReplacement()
    assert registry.list() == []


async def test_tracks_plugin_facet_reload_and_unload():
    registry = SlashCommandRegistry()

    def originalRun(*_: Any) -> None:
        return None

    def helloFacet(description: str, run: Any, failure: Exception | None = None) -> Any:
        def setup(env: Any) -> None:
            commands = env.use(SlashCommands)

            def activate() -> None:
                env.own(commands.replace({"name": "hello", "description": description, "run": run}))
                if failure is not None:
                    raise failure

            env.onActivate(activate)

        return facet("@test/example-hello", setup)

    host = await createFacetHost(
        {"facets": [createSlashCommandsRuntimeFacet(registry), helloFacet("Original", originalRun)]}
    )
    assert names(registry.list()) == ["hello"]

    failure = RuntimeError("replacement failed")
    with pytest.raises(RuntimeError) as raised:
        await host.reload([helloFacet("Failing", lambda *_: None, failure)])
    assert raised.value is failure
    [only] = registry.list()
    assert only["name"] == "hello"
    assert only["run"] is originalRun

    def replacementRun(*_: Any) -> None:
        return None

    await host.reload([helloFacet("Replacement", replacementRun)])
    [replaced] = registry.list()
    assert replaced["name"] == "hello"
    assert replaced["description"] == "Replacement"
    assert replaced["run"] is replacementRun

    await host.dispose()
    assert registry.list() == []


# ─── Server services (MISAKA: pi's test covers them through the remote runtime test) ───────────────────────────


async def test_serves_session_management_to_an_attached_client_and_releases_it_on_dispose():
    from misaka.chord import createServiceCatalogueCall
    from misaka.experimental.services.plugins import PresentationPlugins
    from misaka.experimental.services.server import createExperimentalServerServices
    from misaka.experimental.services.sessions import (
        SessionDirectory,
        SessionManagement,
    )

    sessions: list[dict[str, Any]] = [{"serverId": "server", "sessionId": "a", "createdAt": 1}]
    calls: list[Any] = []

    async def list_(*_: Any) -> list[dict[str, Any]]:
        calls.append("list")
        return list(sessions)

    async def create(options: dict[str, Any], *_: Any) -> dict[str, Any]:
        calls.append(("create", options))
        sessions.append({"serverId": "server", "sessionId": options["id"], "createdAt": 2})
        return sessions[-1]

    async def remove(sessionId: str, *_: Any) -> None:
        calls.append(("remove", sessionId))
        sessions[:] = [session for session in sessions if session["sessionId"] != sessionId]

    async def prepareSessionPlugins(*_: Any) -> dict[str, Any]:
        return {"packagePaths": [], "presentationPlugins": {}}

    async def reloadPresentationPlugins(*_: Any) -> dict[str, Any]:
        return {}

    services = await createExperimentalServerServices(
        {
            "list": list_,
            "create": create,
            "remove": remove,
            "prepareSessionPlugins": prepareSessionPlugins,
            "reloadPresentationPlugins": reloadPresentationPlugins,
        }
    )

    async def prepareSessionRemoval(sessionId: str, _context: Any) -> None:
        calls.append(("prepareSessionRemoval", sessionId))

    attachment = services.host.attachClient(
        SimpleNamespace(prepareSessionRemoval=prepareSessionRemoval), BACKGROUND_CONTEXT
    )

    def publish(*_: Any) -> None:
        return None

    assert await attachment.invokeService(createServiceCatalogueCall(), publish, BACKGROUND_CONTEXT) == [
        {"serviceId": SessionDirectory.id, "mode": "singleton"},
        {"serviceId": SessionManagement.id, "mode": "singleton"},
        {"serviceId": PresentationPlugins.id, "mode": "singleton"},
    ]
    created = await attachment.invokeService(
        {"serviceId": SessionManagement.id, "member": "create", "args": [{"id": "b"}]}, publish, BACKGROUND_CONTEXT
    )
    assert created == {"serverId": "server", "sessionId": "b", "createdAt": 2}
    await attachment.invokeService(
        {"serviceId": SessionManagement.id, "member": "remove", "args": ["a"]}, publish, BACKGROUND_CONTEXT
    )
    # Every change lists the Sessions again; a removal first lets the router drop the Session's attachments.
    assert calls == ["list", ("create", {"id": "b"}), "list", ("prepareSessionRemoval", "a"), ("remove", "a"), "list"]

    await services.dispose()
    with pytest.raises(RuntimeError, match="attachment is released"):
        await attachment.invokeService(createServiceCatalogueCall(), publish, BACKGROUND_CONTEXT)
