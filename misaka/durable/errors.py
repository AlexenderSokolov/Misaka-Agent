"""Translated from pi's ``packages/durable/src/errors.ts``."""

from __future__ import annotations


class ReadAfterWrite(Exception):
    """A transaction read a table after its first table write. Read every required row before writing."""

    def __init__(self, method: str) -> None:
        super().__init__(f"Tx.{method}() cannot read tables after the first table write")
        self.name = "ReadAfterWrite"


class StorageRejected(Exception):
    """Storage rejected a batch before any durable effect; the owning Session may continue safely."""

    def __init__(self, message: str, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.name = "StorageRejected"
        self.cause = cause
        if cause is not None:
            self.__cause__ = cause


class ConversationBusy(Exception):
    """A submission reached a busy conversation and was not admitted."""

    def __init__(self, conversationId: int) -> None:
        super().__init__(f"Conversation {conversationId} is busy")
        self.name = "ConversationBusy"
        self.conversationId = conversationId


__all__ = ["ConversationBusy", "ReadAfterWrite", "StorageRejected"]
