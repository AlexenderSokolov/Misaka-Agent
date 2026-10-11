"""Translated from pi's ``coding-agent/src/experimental/services/slash-commands.ts``.

A ``SlashCommandCompletion`` is ``{"value", "label", "description"?}``. A ``SlashCommandContribution`` is a mapping with
``name``, ``description``?, ``argumentHint``?, ``getArgumentCompletions(argumentPrefix)``? (a list of completions, ``None``
or an awaitable of either) and ``run(args, context)``, which returns an agent operation or queue response, or ``None``.

The service provides ``register(command)`` (returns a function that removes it), ``replace(command)`` (stages a same-name
replacement while the previous facet generation retires), ``list()`` and ``subscribe(listener)``.
"""

from __future__ import annotations

from misaka.chord import defineService

SlashCommands = defineService("pi.local.slash-commands", local=True)

__all__ = ["SlashCommands"]
