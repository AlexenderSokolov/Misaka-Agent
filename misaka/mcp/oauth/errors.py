"""The errors of the MCP OAuth flow.

Translated from pi's ``packages/mcp/src/oauth/errors.ts``.
"""

from __future__ import annotations

from misaka.utils.js import UNDEFINED, jsonStringify


class OAuthError(Exception):
    def __init__(self, code: str, message: str, errorUri: str | None = None) -> None:
        super().__init__(message or code)
        self.message = message or code
        self.name = "OAuthError"
        self.code = code
        self.errorUri = errorUri


class OAuthIssuerMismatchError(Exception):
    def __init__(self, expected: str, received: str | None) -> None:
        message = (
            # `JSON.stringify(undefined)` is undefined, which the template prints as "undefined".
            f"OAuth issuer mismatch: expected {'undefined' if expected is UNDEFINED else jsonStringify(expected)}, "
            f"received {'none' if received is None else jsonStringify(received)}"
        )
        super().__init__(message)
        self.message = message
        self.name = "OAuthIssuerMismatchError"
        self.expected = expected
        # `None` when an authorization response lacks the `iss` parameter its server promised (RFC 9207).
        self.received = received


class OAuthInsecureEndpointError(Exception):
    def __init__(self, endpoint: str) -> None:
        message = f"Refusing to send OAuth credentials to non-HTTPS endpoint {endpoint}"
        super().__init__(message)
        self.message = message
        self.name = "OAuthInsecureEndpointError"
        self.endpoint = endpoint


class OAuthRegistrationError(Exception):
    def __init__(self, status: int, body: str) -> None:
        message = f"OAuth dynamic client registration failed with status {status}: {body}"
        super().__init__(message)
        self.message = message
        self.name = "OAuthRegistrationError"
        self.status = status
        self.body = body


class McpOAuthAuthorizationRequiredError(Exception):
    def __init__(self) -> None:
        message = "MCP OAuth authorization requires user interaction"
        super().__init__(message)
        self.message = message
        self.name = "McpOAuthAuthorizationRequiredError"


__all__ = [
    "McpOAuthAuthorizationRequiredError",
    "OAuthError",
    "OAuthInsecureEndpointError",
    "OAuthIssuerMismatchError",
    "OAuthRegistrationError",
]
