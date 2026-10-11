"""Translated from pi's ``packages/chord/src/services/errors.ts``."""

from __future__ import annotations

from typing import Any, Literal

type RemoteServiceErrorCode = Literal[
    "service_not_allowed",
    "service_not_found",
    "service_mode_mismatch",
    "service_member_not_found",
    "service_member_mismatch",
    "service_instance_not_found",
    "service_stale_instance",
    "service_invalid_value",
]

REMOTE_SERVICE_ERROR_CODES: tuple[RemoteServiceErrorCode, ...] = (
    "service_not_allowed",
    "service_not_found",
    "service_mode_mismatch",
    "service_member_not_found",
    "service_member_mismatch",
    "service_instance_not_found",
    "service_stale_instance",
    "service_invalid_value",
)


def isRemoteServiceErrorCode(value: Any) -> bool:
    return isinstance(value, str) and value in REMOTE_SERVICE_ERROR_CODES


class RemoteServiceError(Exception):
    def __init__(self, code: RemoteServiceErrorCode, message: str) -> None:
        super().__init__(message)
        self.name = "RemoteServiceError"
        self.message = message
        self.code = code


__all__ = ["REMOTE_SERVICE_ERROR_CODES", "RemoteServiceError", "RemoteServiceErrorCode", "isRemoteServiceErrorCode"]
