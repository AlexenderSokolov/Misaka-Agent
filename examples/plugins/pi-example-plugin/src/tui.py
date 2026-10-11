"""The presentation facet: contributes ``/hello``, which calls the Session worker facet's greeting service."""

from __future__ import annotations

from types import SimpleNamespace

from misaka.chord import defineFacet
from misaka.experimental.plugin import AgentController, PresentationUI, SlashCommands

from .contract import ExampleFacetService


def setup(env):
    example = env.use(ExampleFacetService)
    commands = env.use(SlashCommands)
    controller = env.use(AgentController)
    ui = env.use(PresentationUI)

    async def run(args, context):
        message = "from the TUI" if len(args) == 0 else args
        reply = await example.greet({"name": message}, context)
        ui.showStatus(f"We got this: {reply['message']} Worker activations: {reply['workerActivations']}.", context)
        return await controller.prompt({"message": f"pong: {message}", "images": None}, context)

    def activate():
        env.own(
            commands.replace(
                {
                    "name": "hello",
                    "description": "Call the bundled Session worker facet",
                    "argumentHint": "<name>",
                    "run": run,
                }
            )
        )

    env.onActivate(activate)


# MISAKA fork: upstream's id is ``@earendil-works/pi-example-plugin/tui``; this package's name
# (pyproject.toml) has no npm scope.
default = defineFacet(SimpleNamespace(id="pi-example-plugin/tui", setup=setup))
