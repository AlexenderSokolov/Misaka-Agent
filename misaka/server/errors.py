"""Host and lifecycle errors that can safely cross the protocol boundary.

Translated from pi's ``packages/server/src/errors.ts``.
"""

from __future__ import annotations

from misaka.chord.services.errors import RemoteServiceErrorCode

type ServerOperationErrorCode = (
    RemoteServiceErrorCode
    | str  # "wrong_server" | "session_not_found" | "session_ambiguous" | "session_not_attached" | "server_draining"
)

INTERNAL_SERVER_ERROR_MESSAGE = "Internal server error"


class ServerError(Exception):
    """A host or lifecycle error that can safely cross the protocol boundary."""

    def __init__(self, code: ServerOperationErrorCode, message: str) -> None:
        super().__init__(message)
        self.name = "ServerError"
        self.message = message
        self.code = code


class WrongServerError(ServerError):
    def __init__(self) -> None:
        super().__init__("wrong_server", "Request was addressed to another server")
        self.name = "WrongServerError"


class SessionNotFoundError(ServerError):
    def __init__(self, message: str = "Session was not found") -> None:
        super().__init__("session_not_found", message)
        self.name = "SessionNotFoundError"


class SessionAmbiguousError(ServerError):
    def __init__(self) -> None:
        super().__init__("session_ambiguous", "Session ID matches more than one session")
        self.name = "SessionAmbiguousError"


class SessionNotAttachedError(ServerError):
    def __init__(self) -> None:
        super().__init__("session_not_attached", "Session is not attached to this client")
        self.name = "SessionNotAttachedError"


class ServerDrainingError(ServerError):
    def __init__(self) -> None:
        super().__init__("server_draining", "Server is draining")
        self.name = "ServerDrainingError"


__all__ = [
    "INTERNAL_SERVER_ERROR_MESSAGE",
    "ServerDrainingError",
    "ServerError",
    "ServerOperationErrorCode",
    "SessionAmbiguousError",
    "SessionNotAttachedError",
    "SessionNotFoundError",
    "WrongServerError",
]
