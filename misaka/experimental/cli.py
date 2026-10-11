"""Translated from pi's ``coding-agent/src/experimental/cli.ts``.

The development entry: ``python -m misaka.experimental.cli``. With ``PI_EXPERIMENTAL=1`` it runs the experimental ``server`` and
``client`` commands; every other invocation is the regular ``misaka`` command line.

MISAKA fork: pi's entry is a script that sets up the process and calls its ``main``; here the entry sets up the home, the
log and the home's ``.env`` as ``misaka.cli.app.main`` does before it runs an experimental command.
"""

from __future__ import annotations

import asyncio
import sys


def main(argv: list[str] | None = None) -> int | None:
    args = list(sys.argv[1:] if argv is None else argv)
    from misaka.core.experimental import areExperimentalFeaturesEnabled

    if areExperimentalFeaturesEnabled() and args[:1] in (["server"], ["client"]):
        from misaka.cli import bootstrap
        from misaka.config import env as env_file
        from misaka.config import home
        from misaka.config.engine import configure_logging
        from misaka.experimental.commands import runExperimentalCommand

        bootstrap.install()
        home.ensure()
        configure_logging()
        env_file.load()
        return asyncio.run(runExperimentalCommand(args))
    from misaka.__main__ import main as run

    return run(args)  # type: ignore[no-any-return]


if __name__ == "__main__":
    sys.exit(main())
