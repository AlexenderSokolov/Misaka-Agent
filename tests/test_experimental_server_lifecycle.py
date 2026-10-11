"""pi coding-agent's ``test/experimental-server-lifecycle.test.ts``.

MISAKA fork: ``ServerLifetime`` takes ``timers`` (``misaka/experimental/timers.py``), where pi's test installs vitest's fake timers.
"""

from __future__ import annotations

from unittest.mock import Mock

from misaka.experimental.server import ServerLifetime
from tests.experimental_support import FakeTimers


def test_holds_a_foreground_server_until_explicit_shutdown():
    timers = FakeTimers()
    retire = Mock()
    lifetime = ServerLifetime(True, timers)
    lifetime.start(retire)
    timers.advance(60_000)
    retire.assert_not_called()
    lifetime.stop()


def test_retires_an_automatic_server_if_no_first_client_arrives():
    timers = FakeTimers()
    retire = Mock()
    lifetime = ServerLifetime(False, timers)
    lifetime.start(retire)
    timers.advance(10_999)
    retire.assert_not_called()
    timers.advance(1)
    retire.assert_called_once()
    lifetime.stop()


def test_requires_both_client_and_worker_demand_to_disappear():
    timers = FakeTimers()
    retire = Mock()
    lifetime = ServerLifetime(False, timers)
    lifetime.start(retire)
    lifetime.setConnectionCount(1)
    lifetime.setWorkerCount(1)
    lifetime.setConnectionCount(0)
    timers.advance(10_000)
    retire.assert_not_called()

    lifetime.setWorkerCount(0)
    timers.advance(999)
    retire.assert_not_called()
    timers.advance(1)
    retire.assert_called_once()
    lifetime.stop()
