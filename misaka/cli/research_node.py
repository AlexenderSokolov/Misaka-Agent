"""``python -m misaka.cli.research_node --run-card TASK_ID [--say TEXT]``: a research card's child
process. With ``--say`` it continues the settled card with that message instead.

Node executors use ``misaka research --node`` in the background; a card child needs
no CLI surface beyond this entry. It runs the same entry wiring as the CLI
(``misaka.cli.bootstrap``) before
``misaka.core.research.node`` builds its session.
"""

from __future__ import annotations

import sys

from misaka.cli import bootstrap
from misaka.cli.setup import setupCli
from misaka.core.research import node


def main(argv: list[str] | None = None) -> int:
    setupCli()
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) not in (2, 4) or args[0] != "--run-card" or (len(args) == 4 and args[2] != "--say"):
        sys.exit("usage: python -m misaka.cli.research_node --run-card TASK_ID [--say TEXT]")
    bootstrap.install()
    from misaka.utils import loop_watchdog
    loop_watchdog.configure(f"card-{args[1]}", exit_on_stall=True)
    from misaka.config import env as env_file
    env_file.load()
    return node.main_card(args[1], say=args[3] if len(args) == 4 else None)


if __name__ == "__main__":
    raise SystemExit(main())
