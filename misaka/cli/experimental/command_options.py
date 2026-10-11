"""Translated from pi's ``coding-agent/src/cli/experimental/command-options.ts``.

An auth input is ``{"type": "token", "token": ...}`` or ``{"type": "file", "path": ...}``; a transport address is
``{"transport": "unix", "path": ...}`` or ``{"transport": "radius", "serverId": ...}``.

MISAKA fork: Node's WHATWG ``URL`` has no counterpart in the Python standard library, so ``_parseUrl`` reads the pieces of
a URL that ``parseTransportAddress`` inspects (scheme, authority, path, query, fragment) and ``_unixHref`` re-serializes
a ``unix:`` URL the way ``URL.href`` does for the cases the address check can tell apart (percent-encoding of the path
and removal of dot segments).
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote, unquote

from misaka.cli.experimental.command import (
    ParsedCommandInput,
    stringOption,
    valueOption,
)
from misaka.protocol import isServerId
from misaka.utils.js import toWellFormed

authTokenOption = stringOption("--auth-token")
authTokenFileOption = stringOption("--auth-token-file")


def parseAuthInput(options: dict[str, str | None]) -> dict[str, Any]:
    authToken, authTokenFile = options.get("authToken"), options.get("authTokenFile")
    if authToken is not None and authTokenFile is not None:
        return {"errors": ["--auth-token and --auth-token-file are mutually exclusive"]}
    if authToken is not None:
        return {"auth": {"type": "token", "token": authToken}, "errors": []}
    if authTokenFile is not None:
        return {"auth": {"type": "file", "path": authTokenFile}, "errors": []}
    return {"errors": []}


_SCHEME = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*):")
_AUTHORITY_FORM = re.compile(r"//([^/?#]*)([^?#]*)(\?[^#]*)?(#.*)?", re.DOTALL)
_OPAQUE_FORM = re.compile(r"([^?#]*)(\?[^#]*)?(#.*)?", re.DOTALL)
_PORT = re.compile(r"[0-9]*")
# Characters the URL standard percent-encodes in a path, besides everything above U+007E.
_PATH_ENCODE_SET = frozenset(' "#<>?^`{}')
# The forbidden host code points: a host holding one is not a host.
_FORBIDDEN_HOST = frozenset("\0\t\n\r #/:<>?@[\\]^|")
_SPECIAL_SCHEMES = frozenset(("http:", "https:", "ws:", "wss:", "ftp:", "file:"))
# What the URL parser removes from both ends of its input: C0 controls and the space.
_STRIPPED = "".join(chr(code) for code in range(0x21))


def _invalidHost(hostname: str) -> bool:
    return any(character in _FORBIDDEN_HOST or ord(character) < 0x20 or ord(character) == 0x7F for character in hostname)


def _parseUrl(value: str) -> dict[str, str]:
    """The parts of ``new URL(value)`` that ``parseTransportAddress`` reads; raises ``ValueError`` where it would throw."""
    # The URL parser reads a well-formed string, drops leading and trailing C0 controls and spaces, and removes tabs and newlines.
    text = toWellFormed(value).strip(_STRIPPED).translate({9: None, 10: None, 13: None})
    scheme = _SCHEME.match(text)
    if scheme is None:
        raise ValueError("Invalid URL")
    protocol = f"{scheme.group(1).lower()}:"
    rest = text[scheme.end() :]
    if protocol not in ("radius:", "unix:"):
        # MISAKA fork: the host of a special scheme is checked for the code points that make it invalid, and no further.
        if protocol in _SPECIAL_SCHEMES and rest[:2] in ("//", "\\\\", "/\\", "\\/"):
            hostPort = re.split(r"[/\\?#]", rest.lstrip("/\\"), maxsplit=1)[0].rpartition("@")[2]
            host = hostPort[: hostPort.find("]") + 1] if hostPort.startswith("[") else hostPort.partition(":")[0]
            if (host == "" and protocol != "file:") or (not host.startswith("[") and _invalidHost(host)):
                raise ValueError("Invalid URL")
        return {"protocol": protocol}
    authority = _AUTHORITY_FORM.fullmatch(rest)
    if authority is not None:
        hostPort, pathname, search, hash = authority.group(1), authority.group(2), authority.group(3), authority.group(4)
        userInfo, hasUserInfo, hostPort = hostPort.rpartition("@")
        username, _colon, password = userInfo.partition(":")
        if hostPort.startswith("["):
            closing = hostPort.find("]")
            if closing == -1 or (hostPort[closing + 1 :] != "" and not hostPort[closing + 1 :].startswith(":")):
                raise ValueError("Invalid URL")
            hostname, hasPort, port = hostPort[: closing + 1], hostPort[closing + 1 : closing + 2], hostPort[closing + 2 :]
        else:
            hostname, hasPort, port = hostPort.partition(":")
            # An opaque host is percent-encoded where it is not ASCII.
            if _invalidHost(hostname):
                raise ValueError("Invalid URL")
            hostname = "".join(quote(character, safe="") if ord(character) > 0x7E else character for character in hostname)
        if _PORT.fullmatch(port) is None:
            raise ValueError("Invalid URL")
        # An empty host cannot carry credentials or a port.
        if hostname == "" and (hasUserInfo or hasPort):
            raise ValueError("Invalid URL")
    else:
        opaque = _OPAQUE_FORM.fullmatch(rest)
        assert opaque is not None
        pathname, search, hash = opaque.group(1), opaque.group(2), opaque.group(3)
        username = password = hostname = port = ""
    return {
        "protocol": protocol,
        "username": username,
        "password": password,
        "hostname": hostname,
        "port": port,
        "pathname": pathname,
        # ``URL.search`` and ``URL.hash`` read as empty when only the delimiter is present.
        "search": "" if search in (None, "?") else search or "",
        "hash": "" if hash in (None, "#") else hash or "",
    }


def _encodePathSegment(segment: str) -> str:
    return "".join(
        quote(character, safe="") if character in _PATH_ENCODE_SET or ord(character) < 0x20 or ord(character) > 0x7E else character
        for character in segment
    )


def _unixHref(pathname: str) -> str:
    """``new URL("unix://" + pathname).href`` for a URL without an authority: dot segments are resolved and the path
    percent-encoded."""
    if not pathname.startswith("/"):
        return f"unix:{pathname}"
    output: list[str] = []
    segments = pathname[1:].split("/")
    for index, segment in enumerate(segments):
        last = index == len(segments) - 1
        lowered = segment.lower()
        if lowered in ("..", ".%2e", "%2e.", "%2e%2e"):
            if output:
                output.pop()
            if last:
                output.append("")
        elif lowered in (".", "%2e"):
            if last:
                output.append("")
        else:
            output.append(_encodePathSegment(segment))
    return "unix:///" + "/".join(output) if output else "unix:///"


def _decodeUriComponent(text: str) -> str:
    if re.search(r"%(?![0-9A-Fa-f]{2})", text):
        raise ValueError("URI malformed")
    return unquote(text, errors="strict")


def parseTransportAddress(value: str) -> dict[str, Any]:
    try:
        url = _parseUrl(value)
    except ValueError:
        return {"error": f'Invalid --connect address "{value}"'}
    if url["protocol"] == "radius:":
        if (
            url["username"]
            or url["password"]
            or url["port"]
            or (url["pathname"] != "" and url["pathname"] != "/")
            or url["search"]
            or url["hash"]
            or value != f"radius://{url['hostname']}{url['pathname']}"
        ):
            return {"error": f'Invalid --connect address "{value}"'}
        serverId = url["hostname"]
        if not isServerId(serverId):
            return {"error": "Radius transport address requires a lowercase UUIDv4 server ID"}
        return {"address": {"transport": "radius", "serverId": serverId}}
    if url["protocol"] != "unix:":
        return {"error": f'Unsupported --connect transport "{url["protocol"]}"'}
    if url["hostname"] or url["port"] or url["username"] or url["password"]:
        return {"error": "Unix transport address must not include an authority"}
    if (
        not value.startswith("unix:///")
        or value.startswith("unix:////")
        or "?" in value
        or "#" in value
        or _unixHref(url["pathname"]) != value
    ):
        return {"error": f'Invalid --connect address "{value}"'}
    try:
        path = _decodeUriComponent(url["pathname"])
    except ValueError:
        return {"error": f'Invalid --connect address "{value}"'}
    if "\0" in path:
        return {"error": f'Invalid --connect address "{value}"'}
    if not path.startswith("/"):
        return {"error": "Unix transport address requires an absolute path"}
    return {"address": {"transport": "unix", "path": path}}


def _parseConnect(value: str) -> dict[str, Any]:
    result = parseTransportAddress(value)
    if result.get("address") is not None:
        return {"ok": True, "value": result["address"]}
    return {"ok": False, "error": result.get("error") or f'Invalid --connect address "{value}"'}


connectOption = valueOption("--connect", _parseConnect)


def parseAuth(input: ParsedCommandInput) -> dict[str, Any]:
    return parseAuthInput({"authToken": input.value(authTokenOption), "authTokenFile": input.value(authTokenFileOption)})


def unsupportedOptions(command: str, input: ParsedCommandInput) -> list[str]:
    if len(input.remainingArgs) == 0:
        return []
    return [f"The experimental {command} command does not support existing CLI options yet"]
