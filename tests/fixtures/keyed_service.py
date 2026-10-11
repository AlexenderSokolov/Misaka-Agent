"""pi coding-agent's ``test/fixtures/keyed-service.ts``.

``KeyedProbe`` is a keyed service whose instances provide the replicated ``state`` (``{"value": str}``), ``replace(value, context)``
and ``wait(context)``.
"""

from __future__ import annotations

from misaka.chord import defineService

KeyedProbe = defineService("test.keyed-probe")
