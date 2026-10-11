"""The Session worker facet: provides the greeting service."""

from __future__ import annotations

from types import SimpleNamespace

from misaka.chord import defineFacet
from misaka.chord.context import BACKGROUND_CONTEXT

from .contract import ExampleFacetService


def setup(env):
    workerActivations = env.replicatedState({"count": 0})

    async def greet(arguments, *_):
        return {
            "message": f"Hello {arguments['name']} from the bundled Session worker facet!!!",
            "workerActivations": workerActivations.value["count"],
        }

    env.provide(ExampleFacetService, {"workerActivations": workerActivations, "greet": greet})

    def activate():
        def increment(draft):
            draft["count"] += 1

        workerActivations.change(BACKGROUND_CONTEXT, increment)

    env.onActivate(activate)


# MISAKA fork: upstream's id is ``@earendil-works/pi-example-plugin/session``; this package's name
# (pyproject.toml) has no npm scope.
default = defineFacet(SimpleNamespace(id="pi-example-plugin/session", setup=setup))
