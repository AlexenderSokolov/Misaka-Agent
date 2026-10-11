"""Remote execution environments for Durable: an agent's tools run on another machine, usually over SSH, while the
Durable worker, its storage and credentials stay local.

Translated from pi's ``packages/env`` (``@earendil-works/pi-env`` v1.1.0):

- ``misaka.env.daemon``: the daemon that runs on the remote machine. It speaks a framed protocol on stdin and stdout
  (``docs/protocol.md``) and performs file operations and commands there. MISAKA fork: pi's daemon is Rust; this
  one is pure Python (3.9 or later on the remote), shipped as one zip application (``misaka.env.bundle``).
- ``RemoteExecutionEnv``: a Durable ``ExecutionEnv`` that talks to the daemon through a ``Connection``. Its results
  match ``NodeExecutionEnv`` running on the remote machine; only error messages may differ (``docs/semantics.md``).

Like pi's package, nothing in MISAKA's stable entry points uses it; it is reachable only from the experimental side.
"""

from .connection import (
    Connection,
    ConnectionOptions,
    RemoteError,
    RemoteInfo,
    Reply,
    RequestOptions,
    isConnectionLost,
)
from .remote_env import RemoteExecutionEnv, RemoteExecutionEnvOptions
from .ssh import (
    HostKeyChangedError,
    HostKeyUnknownError,
    RemotePlatform,
    SshConnectOptions,
    SshError,
    SshTarget,
    acceptHostKey,
    connectSsh,
    deployDaemon,
    detectPlatform,
    forgetHostKey,
    packagedDaemon,
    scanHostKey,
    sshArguments,
    sshConnection,
)
from .watch import RemoteWatcher, RemoteWatchOptions

__all__ = [
    "Connection",
    "ConnectionOptions",
    "HostKeyChangedError",
    "HostKeyUnknownError",
    "RemoteError",
    "RemoteExecutionEnv",
    "RemoteExecutionEnvOptions",
    "RemoteInfo",
    "RemotePlatform",
    "RemoteWatchOptions",
    "RemoteWatcher",
    "Reply",
    "RequestOptions",
    "SshConnectOptions",
    "SshError",
    "SshTarget",
    "acceptHostKey",
    "connectSsh",
    "deployDaemon",
    "detectPlatform",
    "forgetHostKey",
    "isConnectionLost",
    "packagedDaemon",
    "scanHostKey",
    "sshArguments",
    "sshConnection",
]
