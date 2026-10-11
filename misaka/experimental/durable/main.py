"""Translated from pi's ``coding-agent/src/experimental/durable/main.ts``.

The durable demo's entry point: ``python -m misaka.experimental.durable.main [--continue | -c]``.
"""

from __future__ import annotations

import asyncio
import sys
import traceback

from misaka.experimental.durable.runtime import openDurable
from misaka.experimental.durable.tui import runDurableTui


def parseArgs(argv: list[str]) -> dict[str, bool]:
    continueSession = False
    for arg in argv:
        if arg in ("--continue", "-c"):
            continueSession = True
        else:
            raise RuntimeError(f"Unknown argument: {arg}")
    return {"continueSession": continueSession}


async def run(argv: list[str]) -> None:
    durable = await openDurable(parseArgs(argv))
    try:
        await runDurableTui(durable.view, durable.controller, durable.settings)
    finally:
        await durable.close()


def main() -> None:
    from misaka.cli import bootstrap
    from misaka.config import env as env_file
    from misaka.config import home
    from misaka.config.engine import configure_logging

    bootstrap.install()
    home.ensure()
    configure_logging()
    env_file.load()
    try:
        asyncio.run(run(sys.argv[1:]))
    except Exception:  # noqa: BLE001 - the process reports its failure and exits
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
