"""Translated from pi's ``packages/client/src/errors.ts``."""

from __future__ import annotations

from typing import Any

from misaka.protocol import ProtocolError, ProtocolErrorCode


class ServerError(Exception):
    def __init__(self, error: ProtocolError) -> None:
        super().__init__(error["message"])
        self.name = "ServerError"
        self.message: str = error["message"]
        self.code: ProtocolErrorCode = error["code"]


class DisconnectedError(Exception):
    def __init__(self, message: str = "Client is disconnected", cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.name = "DisconnectedError"
        self.message = message
        if cause is not None:
            self.__cause__ = cause
        self.cause = cause


class ClientDisposedError(Exception):
    def __init__(self) -> None:
        super().__init__("Client is disposed")
        self.name = "ClientDisposedError"
        self.message = "Client is disposed"


def toError(error: BaseException | Any) -> Exception:
    return error if isinstance(error, Exception) else Exception(str(error))


def toDisconnectedError(error: BaseException | Any) -> DisconnectedError:
    cause = toError(error)
    return cause if isinstance(cause, DisconnectedError) else DisconnectedError(str(cause), cause)


__all__ = ["ClientDisposedError", "DisconnectedError", "ServerError", "toDisconnectedError", "toError"]
