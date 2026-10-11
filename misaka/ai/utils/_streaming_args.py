"""MISAKA fork: a tool call's arguments while they stream in, kept raw and parsed sparingly.

pi re-parses the whole accumulated buffer with ``parseStreamingJson`` on every fragment; in Python that cost 31 s of
loop-thread CPU for one 38 KB argument, so adapters keep the raw text here and parse it on a schedule, and the final
arguments are parsed strictly (``finish``).
"""

from __future__ import annotations

from typing import Any

from misaka.ai.utils.json_parse import parseJsonWithRepair, parseStreamingJson

STREAMING_PARSE_THRESHOLD_BYTES = 2048


class StreamingArgs:
    """A tool call's arguments while they stream in: kept raw, parsed sparingly.

    Providers deliver tool arguments as JSON fragments, and every adapter used to hand
    ``parseStreamingJson`` the *whole* accumulated buffer again on every fragment.
    That parse is up to four full scans of the buffer (``JSON.parse``, the ``repairJson``
    pass, then ``partial-json`` on the raw and on the repaired text), so a 38KB Write argument
    arriving in 20-character fragments was parsed ~1950 times over growing prefixes --
    measured at 31s of loop-thread CPU for one tool call.

    The raw string is the durable half and is always exact: the fragments are appended
    verbatim, and ``finish()`` parses the complete buffer at the end of the block, which
    is the value the tool actually runs with. ``arguments`` is the *live view* in
    between, refreshed often enough for the UI and rarely enough to stay cheap.
    """

    __slots__ = ("_parsed_length", "_raw", "_value")

    def __init__(self, initial: str = "") -> None:
        self._raw = initial
        self._parsed_length = -1  # nothing parsed yet: -1 so an initial buffer still earns one
        self._value: Any = {}

    @property
    def raw(self) -> str:
        """Every fragment seen so far, concatenated and unaltered."""
        return self._raw

    def append(self, delta: str) -> None:
        if delta:
            self._raw += delta

    @property
    def arguments(self) -> Any:
        """The last parsed value, re-parsed first if enough new bytes have arrived."""
        if self._is_stale():
            self._parse()
        return self._value

    def finish(self) -> Any:
        """MISAKA fork: final arguments require a complete object, never prefix recovery.

        Keep the existing complete-string escape repair and empty/no-argument convention.
        Incremental UI previews still use the permissive parser in arguments.
        """
        value = parseJsonWithRepair(self._raw) if self._raw.strip() else {}
        if not isinstance(value, dict):
            raise ValueError("Final tool arguments must be a JSON object")  # noqa: TRY004 - malformed wire value, same contract as JSONDecodeError
        self._value = value
        self._parsed_length = len(self._raw)
        return self._value

    def finish_into(self, tool_call) -> None:
        """Keep malformed calls in the batch, but return an error receipt instead of executing."""
        try:
            tool_call.arguments = self.finish()
            tool_call.argumentsError = None
        except (ValueError, RecursionError):
            tool_call.arguments = {}
            tool_call.argumentsError = "Invalid final tool arguments; resend a complete JSON object."

    def _is_stale(self) -> bool:
        total = len(self._raw)
        pending = total - self._parsed_length
        if pending <= 0:
            return False
        if total <= STREAMING_PARSE_THRESHOLD_BYTES:
            # Parsing a buffer this small costs microseconds, and the TUI renders tool
            # arguments while they stream -- a tool call under the threshold (nearly all
            # of them) must keep showing its path the moment it is spelled out.
            return True
        # Above it, wait for a fixed slice *and* for a fraction of what is already there.
        # The fixed slice alone would still parse len/slice prefixes of average len/2,
        # which is the same quadratic with a smaller constant; the fraction is what keeps
        # the total work proportional to the argument's length.
        return pending >= max(STREAMING_PARSE_THRESHOLD_BYTES, total // 8)

    def _parse(self) -> None:
        self._value = parseStreamingJson(self._raw)
        self._parsed_length = len(self._raw)


__all__ = ["STREAMING_PARSE_THRESHOLD_BYTES", "StreamingArgs"]
