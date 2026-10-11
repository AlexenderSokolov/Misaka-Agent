"""The service the example plugin's two facets share.

``ExampleFacetService`` provides ``workerActivations`` (a replicated ``{"count": int}``) and ``greet({"name": str},
context)``, which resolves to ``{"message": str, "workerActivations": int}``.
"""

from __future__ import annotations

from misaka.chord import defineService

ExampleFacetService = defineService("pi.example-plugin.greeting")
