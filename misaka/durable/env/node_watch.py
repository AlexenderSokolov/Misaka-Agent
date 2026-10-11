"""Ported from: pi packages/durable/src/env/node-watch.ts @ v1.1.0

MISAKA fork (one place, ``_fsWatch`` and ``_statfsType`` at the end of the module): Node's ``fs.watch`` and
``fs.statfs`` have no standard-library counterpart. ``_fsWatch`` watches one path with inotify (ctypes) on Linux and
Android and with a kqueue vnode filter on macOS; elsewhere it raises ``ENOSYS``, which is upstream's "unsupported" path
(the watcher switches to polling). macOS gets one watcher per directory, as Linux does, because a kqueue watch is not
recursive (upstream uses one recursive FSEvents watcher per target there). Windows polls by default upstream as here.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import hashlib
import os
import stat as statModule
import sys
import time
from collections.abc import Callable
from typing import Any, Literal, NotRequired, TypedDict

from misaka.chord.context import Context

from . import FileError, WatchChange, WatchTarget


class NodeWatchOptions(TypedDict, total=False):
    """How ``NodeExecutionEnv`` watches.

    - ``mode``: force a mode. By default ``polling`` is chosen on Windows, where native watchers keep directories open
      and so block renaming their parents, and for file systems that do not report remote changes.
    - ``pollIntervalMs``: interval between snapshots in ``polling`` mode; default 2000 ms.
    - ``maxDirectories``: most directories one watcher covers; default 10,000.
    """

    mode: Literal["native", "polling"]
    pollIntervalMs: float
    maxDirectories: int


class _ResolvedTarget:
    __slots__ = ("hidden", "names", "path", "recursive")

    def __init__(self, path: str, recursive: bool, hidden: bool, names: set[str]) -> None:
        self.path = path
        self.recursive = recursive
        self.hidden = hidden
        self.names = names


class _Entry(TypedDict):
    """What a snapshot remembers of one path. Directories and ancestors are compared by identity only."""

    kind: Literal["file", "directory", "symlink", "other"]
    dev: int
    ino: int
    size: int
    mtimeMs: float
    hash: NotRequired[str]


type _Snapshot = dict[str, _Entry]


class _Installed:
    """A native watcher and the identity of what it was installed on: a path replaced by another file needs a new
    one."""

    __slots__ = ("dev", "ino", "watcher")

    def __init__(self, watcher: _NativeWatcher, dev: int, ino: int) -> None:
        self.watcher = watcher
        self.dev = dev
        self.ino = ino


class _Scan:
    """A scan: the snapshot, and targets that are symbolic links to files, whose files need their own watchers."""

    __slots__ = ("linkedFiles", "snapshot")

    def __init__(self, snapshot: _Snapshot, linkedFiles: set[str]) -> None:
        self.snapshot = snapshot
        self.linkedFiles = linkedFiles


DEBOUNCE_MS = 50
# On macOS, `fs.watch` returns before libuv's FSEvents stream is live, and changes in between are never reported
# (about 2% of changes made right after the call, none after 100 ms). A rescan this long after installing watchers
# catches them.
FSEVENTS_SETTLE_MS = 500
DEFAULT_POLL_MS = 2000
DEFAULT_MAX_DIRECTORIES = 10_000
# In polling mode, recently modified small files are also compared by content: a second write within the file system's
# timestamp granularity can keep size and modification time.
HASH_MAX_BYTES = 256 * 1024
HASH_RECENT_MS = 5000

# Linux `statfs` magic numbers of file systems that accept watches but do not report changes made elsewhere.
UNRELIABLE_FILE_SYSTEMS = {
    0x6969,  # NFS
    0x517B,  # SMB
    0xFF534D42,  # CIFS
    0xFE534D42,  # SMB2
    0x65735546,  # FUSE (sshfs, Android shared storage)
    0x01021997,  # 9P (WSL2 Windows drives)
    0x0BD00BD0,  # Lustre
    0x47504653,  # GPFS
    0x00C36400,  # Ceph
    0x5346414F,  # OpenAFS
    0x6B414653,  # kAFS
    0x5DCA2DF5,  # sdcardfs
}


class _BudgetExceeded(Exception):
    pass


def _isNodeError(error: Any) -> bool:
    return isinstance(error, OSError) and error.errno is not None


def _isDenied(error: Any) -> bool:
    return _isNodeError(error) and error.errno in (errno.EACCES, errno.EPERM)


def _entryOf(stats: os.stat_result, hash: str | None = None) -> _Entry:
    mode = stats.st_mode
    kind: Literal["file", "directory", "symlink", "other"] = (
        "file"
        if statModule.S_ISREG(mode)
        else "directory"
        if statModule.S_ISDIR(mode)
        else "symlink"
        if statModule.S_ISLNK(mode)
        else "other"
    )
    if (
        kind == "directory"
        and sys.platform == "win32"
        and getattr(stats, "st_reparse_tag", 0) == statModule.IO_REPARSE_TAG_MOUNT_POINT  # type: ignore[attr-defined]
    ):
        kind = "symlink"  # MISAKA fork: libuv reports a junction as a symbolic link (as fileInfoFromStats does).
    entry: _Entry = {
        "kind": kind,
        "dev": stats.st_dev,
        "ino": stats.st_ino,
        "size": 0 if kind == "directory" else stats.st_size,
        "mtimeMs": 0 if kind == "directory" else stats.st_mtime_ns / 1e6,
    }
    if hash is not None:
        entry["hash"] = hash
    return entry


def _sameEntry(a: _Entry, b: _Entry) -> bool:
    return (
        a["kind"] == b["kind"]
        and a["dev"] == b["dev"]
        and a["ino"] == b["ino"]
        and a["size"] == b["size"]
        and a["mtimeMs"] == b["mtimeMs"]
        and a.get("hash") == b.get("hash")
    )


def _ancestorsOf(path: str) -> list[str]:
    """Ancestors of ``path``, nearest first, up to the root."""
    result: list[str] = []
    current = os.path.dirname(path)
    while True:
        result.append(current)
        if os.path.dirname(current) == current:
            return result
        current = os.path.dirname(current)


def _isWithin(path: str, ancestor: str) -> bool:
    """Whether ``path`` is ``ancestor`` or below it."""
    return path == ancestor or path.startswith(ancestor if ancestor.endswith(os.sep) else ancestor + os.sep)


def _excluded(target: _ResolvedTarget, name: str) -> bool:
    return (target.hidden and name.startswith(".")) or name in target.names


def _hashFile(path: str) -> str:
    with open(path, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()


async def _lstatOrNone(path: str) -> os.stat_result | None:
    try:
        return await asyncio.to_thread(os.lstat, path)
    except Exception:  # noqa: BLE001 - `lstat(path).catch(() => undefined)`
        return None


class NodeFileWatcher:
    """Watches by snapshots: native events only trigger a debounced rescan, and changes are the difference between
    snapshots plus the event paths. A replaced file, a renamed or recreated ancestor, or a directory created with its
    contents therefore never depends on which events an operating system sends. New directories get their watchers
    before they are scanned again, so nothing written into them before the watcher existed is missed."""

    def __init__(
        self,
        targets: list[_ResolvedTarget],
        onChange: Callable[[WatchChange], None],
        mode: Literal["native", "polling"],
        options: NodeWatchOptions,
    ) -> None:
        self._targets = targets
        self._onChange = onChange
        self._mode: Literal["native", "polling"] = mode
        pollIntervalMs = options.get("pollIntervalMs")
        self._pollIntervalMs = DEFAULT_POLL_MS if pollIntervalMs is None else pollIntervalMs
        maxDirectories = options.get("maxDirectories")
        self._maxDirectories = DEFAULT_MAX_DIRECTORIES if maxDirectories is None else maxDirectories
        self._watchers: dict[str, _Installed] = {}
        self._events: dict[str, None] = {}
        self._snapshot: _Snapshot = {}
        self._timer: asyncio.TimerHandle | None = None
        self._settleTimer: asyncio.TimerHandle | None = None
        self._running: asyncio.Task[None] | None = None
        self._dirty = False
        self._closed = False

    @staticmethod
    async def open(
        targets: list[WatchTarget],
        resolvePath: Callable[[str], str],
        onChange: Callable[[WatchChange], None],
        options: NodeWatchOptions,
    ) -> NodeFileWatcher:
        """Establish coverage: watchers first, then the snapshot later changes are compared with."""
        resolved = [
            _ResolvedTarget(
                resolvePath(target["path"]),
                target.get("recursive") is True,
                (target.get("exclude") or {}).get("hidden") is True,
                set((target.get("exclude") or {}).get("names") or []),
            )
            for target in targets
        ]
        # Windows refuses to rename a directory while another directory below it is open, and a native watcher keeps
        # every watched directory open: watching would break renames of their parents, so Windows polls.
        forced = options.get("mode")
        mode: Literal["native", "polling"] = (
            forced
            if forced is not None
            else "polling"
            if sys.platform == "win32" or await _anyUnreliable([target.path for target in resolved])
            else "native"
        )
        watcher = NodeFileWatcher(resolved, onChange, mode, options)
        try:
            await watcher._sync(False)
        except BaseException as error:
            watcher._stop()
            if isinstance(error, _BudgetExceeded):
                raise FileError("invalid", str(error)) from None
            raise
        watcher._schedulePoll()
        return watcher

    @property
    def mode(self) -> Literal["native", "polling"]:
        return self._mode

    async def close(self, _context: Context) -> None:
        if self._closed:
            return
        self._stop()
        running = self._running
        if running is not None:
            with contextlib.suppress(Exception):
                await asyncio.shield(running)

    def _stop(self) -> None:
        self._closed = True
        if self._timer is not None:
            self._timer.cancel()
        self._timer = None
        if self._settleTimer is not None:
            self._settleTimer.cancel()
        self._settleTimer = None
        for installed in self._watchers.values():
            installed.watcher.close()
        self._watchers.clear()

    def _deliver(self, change: WatchChange) -> None:
        if self._closed:
            return
        try:
            self._onChange(change)
        except Exception:  # noqa: BLE001, S110 - A throwing callback must not stop watching.
            pass

    def _schedulePoll(self) -> None:
        if self._closed or self._mode != "polling":
            return

        def fire() -> None:
            self._timer = None
            self._flush().add_done_callback(lambda _task: self._schedulePoll())

        self._timer = asyncio.get_running_loop().call_later(self._pollIntervalMs / 1000, fire)

    def _scheduleFlush(self) -> None:
        if self._closed or self._timer is not None:
            return

        def fire() -> None:
            self._timer = None
            self._flush()

        self._timer = asyncio.get_running_loop().call_later(DEBOUNCE_MS / 1000, fire)

    def _scheduleSettle(self) -> None:
        if self._closed:
            return
        if self._settleTimer is not None:
            self._settleTimer.cancel()

        def fire() -> None:
            self._settleTimer = None
            self._flush()

        self._settleTimer = asyncio.get_running_loop().call_later(FSEVENTS_SETTLE_MS / 1000, fire)

    def _flush(self) -> asyncio.Task[None]:
        if self._running is not None:
            self._dirty = True
            return self._running

        async def run() -> None:
            try:
                while True:
                    self._dirty = False
                    events = list(self._events)
                    self._events.clear()
                    changed = await self._sync(True)
                    for path in events:
                        changed[path] = None
                    if len(changed) > 0:
                        self._deliver({"paths": sorted(changed)})
                    if not (self._dirty and not self._closed):
                        break
            except Exception as error:  # noqa: BLE001
                message = str(error)
                fileError = (
                    error
                    if isinstance(error, FileError)
                    else FileError("permission_denied" if _isDenied(error) else "invalid", message)
                )
                self._deliver({"error": fileError})
                self._stop()
            finally:
                self._running = None

        self._running = asyncio.get_running_loop().create_task(run())
        return self._running

    async def _sync(self, report: bool) -> dict[str, None]:
        """Rescan, report differences, and install watchers for new directories, rescanning until none are new."""
        changed: dict[str, None] = {}
        round_ = 0
        while round_ < 10 and not self._closed:
            scan = await self._scan()
            # Closed during the scan: installing watchers now would leak them.
            if self._closed:
                break
            next_ = scan.snapshot
            if report:
                for path in self._diff(self._snapshot, next_):
                    changed[path] = None
            self._snapshot = next_
            if self._mode == "polling" or not self._reconcileWatchers(scan):
                break
            if sys.platform == "darwin":
                self._scheduleSettle()
            # Something written into a new directory before its watcher existed shows up in the next round.
            report = True
            round_ += 1
        return changed

    def _diff(self, previous: _Snapshot, next_: _Snapshot) -> list[str]:
        changed: list[str] = []
        for path, entry in next_.items():
            before = previous.get(path)
            if before is None or not _sameEntry(before, entry):
                changed.append(self._reported(path))
        for path in previous:
            if path not in next_:
                changed.append(self._reported(path))
        return changed

    def _reported(self, path: str) -> str:
        """An ancestor that changed identity moved every target below it; report those targets."""
        if any(_isWithin(path, target.path) for target in self._targets):
            return path
        return next((target.path for target in self._targets if _isWithin(target.path, path)), path)

    async def _scan(self) -> _Scan:
        snapshot: _Snapshot = {}
        linkedFiles: set[str] = set()
        # Directories are counted once, but traversed once per target: overlapping targets differ in recursion and
        # exclusions, and an entry recorded for one target must still be descended into for another.
        counted: set[str] = set()
        traversed: set[str] = set()
        # Kinds of listed entries, not following links: a target that links to a directory is recorded by `stat`, but
        # symbolic links below a recursive target are not followed.
        listed: dict[str, str] = {}

        def countDirectory(path: str) -> None:
            counted.add(path)
            if len(counted) > self._maxDirectories:
                raise _BudgetExceeded(f"Watched paths exceed {self._maxDirectories} directories")

        async def record(path: str, stats: os.stat_result) -> None:
            hash: str | None = None
            if (
                self._mode == "polling"
                and statModule.S_ISREG(stats.st_mode)
                and stats.st_size <= HASH_MAX_BYTES
                and time.time() * 1000 - stats.st_mtime_ns / 1e6 < HASH_RECENT_MS
            ):
                try:
                    hash = await asyncio.to_thread(_hashFile, path)
                except Exception:  # noqa: BLE001
                    hash = None
            snapshot[path] = _entryOf(stats, hash)

        async def scanDirectory(index: int, target: _ResolvedTarget, directory: str) -> None:
            key = f"{index}\0{directory}"
            if key in traversed:
                return
            traversed.add(key)
            try:
                names = await asyncio.to_thread(os.listdir, directory)
            except Exception as error:
                # The watched directory itself must be readable; below it, unreadable directories are skipped.
                if directory == target.path and _isDenied(error):
                    raise
                if _isNodeError(error) and error.errno in (errno.ENOENT, errno.EACCES, errno.EPERM, errno.ENOTDIR):  # type: ignore[attr-defined]
                    return
                raise
            for name in names:
                if _excluded(target, name):
                    continue
                path = f"{directory if directory.endswith(os.sep) else directory + os.sep}{name}"
                kind = listed.get(path)
                if kind is None:
                    stats = await _lstatOrNone(path)
                    if stats is None:
                        continue
                    kind = _entryOf(stats)["kind"]
                    listed[path] = kind
                    # A target's own entry (following links) wins over its listing by another target.
                    if path not in snapshot:
                        await record(path, stats)
                if target.recursive and kind == "directory":
                    countDirectory(path)
                    await scanDirectory(index, target, path)

        for index, target in enumerate(self._targets):
            for ancestor in _ancestorsOf(target.path):
                if ancestor in snapshot:
                    continue
                ancestorStats = await _lstatOrNone(ancestor)
                # Identity only: an ancestor's own timestamps change with every unrelated sibling.
                if ancestorStats is not None:
                    snapshot[ancestor] = {**_entryOf(ancestorStats), "size": 0, "mtimeMs": 0}  # type: ignore[typeddict-item]
            # The target itself may be a symbolic link to what is watched; follow it. A missing target is watched for its
            # creation; one that cannot be reached for lack of permission fails.
            try:
                stats = await asyncio.to_thread(os.stat, target.path)
            except Exception as error:
                if _isDenied(error):
                    raise
                continue
            await record(target.path, stats)
            if statModule.S_ISREG(stats.st_mode):
                linkStats = await _lstatOrNone(target.path)
                if linkStats is not None and statModule.S_ISLNK(linkStats.st_mode):
                    linkedFiles.add(target.path)
            if statModule.S_ISDIR(stats.st_mode):
                countDirectory(target.path)
                await scanDirectory(index, target, target.path)
        return _Scan(snapshot, linkedFiles)

    def _reconcileWatchers(self, scan: _Scan) -> bool:
        """Watch every existing ancestor of each target, each target directory, each target that is a symbolic link to
        a file (changes to that file are not events of the link's directory), and on Linux each directory below a
        recursive target (elsewhere one recursive watcher per target). Returns whether a watcher was added."""
        snapshot, linkedFiles = scan.snapshot, scan.linkedFiles
        wanted: dict[str, bool] = {}
        for path in linkedFiles:
            wanted[path] = False
        # MISAKA fork: macOS watches per directory too (a kqueue watch is not recursive; see the module docstring).
        perDirectory = sys.platform.startswith("linux") or sys.platform == "android" or sys.platform == "darwin"
        for target in self._targets:
            for ancestor in _ancestorsOf(target.path):
                entry = snapshot.get(ancestor)
                if entry is not None and entry["kind"] == "directory" and ancestor not in wanted:
                    wanted[ancestor] = False
            targetEntry = snapshot.get(target.path)
            if targetEntry is None or targetEntry["kind"] != "directory":
                continue
            recursiveRoot = target.recursive and not perDirectory
            wanted[target.path] = recursiveRoot
            if target.recursive and perDirectory:
                for path, entry in snapshot.items():
                    if entry["kind"] == "directory" and path != target.path and _isWithin(path, target.path):
                        wanted[path] = False
        for path, installed in list(self._watchers.items()):
            entry = snapshot.get(path)
            # Gone, or replaced: a watcher follows the directory it was installed on, not the path.
            if path not in wanted or entry is None or entry["dev"] != installed.dev or entry["ino"] != installed.ino:
                installed.watcher.close()
                del self._watchers[path]
        added = False
        for path, recursive in wanted.items():
            entry = snapshot.get(path)
            if path in self._watchers or entry is None:
                continue
            linked = path in linkedFiles
            try:
                holder: list[_NativeWatcher] = []

                def onError(path: str = path, holder: list[_NativeWatcher] = holder) -> None:
                    # A watched directory that disappears or fails: rescan, which also replaces the watcher.
                    watcher = holder[0]
                    watcher.close()
                    installed = self._watchers.get(path)
                    if installed is not None and installed.watcher is watcher:
                        del self._watchers[path]
                    self._scheduleFlush()

                def listener(filename: str | None, path: str = path, linked: bool = linked) -> None:
                    if linked:
                        self._onLinkedFileEvent(path)
                    else:
                        self._onEvent(path, filename)

                watcher = _fsWatch(path, recursive, listener, onError)
                holder.append(watcher)
                self._watchers[path] = _Installed(watcher, entry["dev"], entry["ino"])
                added = True
            except Exception as error:  # noqa: BLE001 - what is not an errno error is ignored, as upstream does
                # Out of watches or unsupported: compare snapshots from now on, and say coverage was uncertain.
                if _isNodeError(error) and error.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):  # type: ignore[attr-defined]
                    self._switchToPolling()
                    return False
        return added

    def _switchToPolling(self) -> None:
        if self._mode == "polling":
            return
        self._mode = "polling"
        for installed in self._watchers.values():
            installed.watcher.close()
        self._watchers.clear()
        self._deliver({"overflow": True})
        if self._timer is not None:
            self._timer.cancel()
        self._timer = None
        self._schedulePoll()

    def _onEvent(self, directory: str, filename: str | None) -> None:
        if self._closed:
            return
        path = (
            directory
            if filename is None
            else f"{directory if directory.endswith(os.sep) else directory + os.sep}{filename}"
        )
        # Events about unrelated siblings of an ancestor, or about excluded entries, are ignored.
        relevant = self._inScope(path)
        if relevant:
            self._events[self._reported(path)] = None
        if relevant or filename is None:
            self._scheduleFlush()

    def _onLinkedFileEvent(self, target: str) -> None:
        """The file a target links to changed: report the target."""
        if self._closed:
            return
        self._events[target] = None
        self._scheduleFlush()

    def _inScope(self, path: str) -> bool:
        for target in self._targets:
            if _isWithin(target.path, path):
                return True
            if not _isWithin(path, target.path) or path == target.path:
                continue
            components = os.path.relpath(path, target.path).split(os.sep)
            if not target.recursive and len(components) > 1:
                continue
            if any(_excluded(target, name) for name in components):
                continue
            return True
        return False


async def _anyUnreliable(paths: list[str]) -> bool:
    """Whether any path, or its nearest existing ancestor, is on a file system that does not report remote changes."""
    if not sys.platform.startswith("linux") and sys.platform != "android":
        return False
    for path in paths:
        for candidate in [path, *_ancestorsOf(path)]:
            fsType = await asyncio.to_thread(_statfsType, candidate)
            if fsType is None:
                continue
            if fsType in UNRELIABLE_FILE_SYSTEMS:
                return True
            break
    return False


# --- MISAKA fork: the native primitives (fs.watch, fs.statfs) --------------------------------------------------------


class _NativeWatcher:
    """What ``_fsWatch`` returns: ``close()`` stops it (idempotent)."""

    def __init__(self, closer: Callable[[], None]) -> None:
        self._closer: Callable[[], None] | None = closer

    def close(self) -> None:
        closer, self._closer = self._closer, None
        if closer is not None:
            closer()


def _statfsType(path: str) -> int | None:
    """``statfs(path).type`` on Linux, or ``None`` when it fails (``statfs(candidate).catch(() => undefined)``)."""
    import ctypes

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        # struct statfs begins with f_type (__fsword_t, a long); 256 bytes covers the whole structure.
        buffer = (ctypes.c_long * 32)()
        if libc.statfs(os.fsencode(path), ctypes.byref(buffer)) != 0:
            return None
        return buffer[0] & 0xFFFFFFFF
    except (OSError, AttributeError):
        return None


# inotify(7)
_IN_MODIFY, _IN_ATTRIB, _IN_CLOSE_WRITE, _IN_MOVED_FROM, _IN_MOVED_TO = 0x2, 0x4, 0x8, 0x40, 0x80
_IN_CREATE, _IN_DELETE, _IN_DELETE_SELF, _IN_MOVE_SELF, _IN_IGNORED = 0x100, 0x200, 0x400, 0x800, 0x8000
_IN_NONBLOCK, _IN_CLOEXEC = 0o4000, 0o2000000
_INOTIFY_MASK = (
    _IN_MODIFY | _IN_ATTRIB | _IN_CLOSE_WRITE | _IN_MOVED_FROM | _IN_MOVED_TO | _IN_CREATE | _IN_DELETE
    | _IN_DELETE_SELF | _IN_MOVE_SELF
)


def _fsWatch(
    path: str,
    recursive: bool,
    listener: Callable[[str | None], None],
    onError: Callable[[], None],
) -> _NativeWatcher:
    """``fs.watch(path, {recursive, persistent: false}, listener)`` with its ``error`` event as ``onError``. Raises an
    ``OSError`` as ``fs.watch`` throws: ``ENOENT``/``EACCES``/``EPERM`` for the path, anything else (``ENOSPC`` out of
    watches, ``ENOSYS`` unsupported) makes the caller switch to polling."""
    if recursive:
        raise OSError(errno.ENOSYS, "recursive native watching is not available", path)
    loop = asyncio.get_running_loop()
    if sys.platform.startswith("linux") or sys.platform == "android":
        return _inotifyWatch(loop, path, listener, onError)
    if sys.platform == "darwin":
        return _kqueueWatch(loop, path, listener, onError)
    raise OSError(errno.ENOSYS, "native file watching is not available", path)


def _inotifyWatch(
    loop: asyncio.AbstractEventLoop,
    path: str,
    listener: Callable[[str | None], None],
    onError: Callable[[], None],
) -> _NativeWatcher:
    import ctypes
    import struct

    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(_IN_NONBLOCK | _IN_CLOEXEC)
    if fd < 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), path)
    if libc.inotify_add_watch(fd, os.fsencode(path), _INOTIFY_MASK) < 0:
        number = ctypes.get_errno()
        os.close(fd)
        raise OSError(number, os.strerror(number), path)
    header = struct.Struct("iIII")

    def readable() -> None:
        try:
            data = os.read(fd, 64 * 1024)
        except BlockingIOError:
            return
        except OSError:
            onError()
            return
        offset = 0
        failed = False
        while offset + header.size <= len(data):
            _wd, mask, _cookie, length = header.unpack_from(data, offset)
            raw = data[offset + header.size : offset + header.size + length]
            offset += header.size + length
            if mask & (_IN_IGNORED | _IN_DELETE_SELF | _IN_MOVE_SELF):
                failed = True
                continue
            name = raw.split(b"\0", 1)[0]
            listener(os.fsdecode(name) if name else None)
        if failed:
            onError()

    loop.add_reader(fd, readable)

    def close() -> None:
        with contextlib.suppress(Exception):
            loop.remove_reader(fd)
        with contextlib.suppress(OSError):
            os.close(fd)

    return _NativeWatcher(close)


def _kqueueWatch(
    loop: asyncio.AbstractEventLoop,
    path: str,
    listener: Callable[[str | None], None],
    onError: Callable[[], None],
) -> _NativeWatcher:
    import select

    fd = os.open(path, getattr(os, "O_EVTONLY", 0x8000))
    kq = select.kqueue()  # type: ignore[attr-defined]
    fflags = (
        select.KQ_NOTE_WRITE  # type: ignore[attr-defined]
        | select.KQ_NOTE_DELETE  # type: ignore[attr-defined]
        | select.KQ_NOTE_RENAME  # type: ignore[attr-defined]
        | select.KQ_NOTE_EXTEND  # type: ignore[attr-defined]
        | select.KQ_NOTE_ATTRIB  # type: ignore[attr-defined]
        | select.KQ_NOTE_LINK  # type: ignore[attr-defined]
        | select.KQ_NOTE_REVOKE  # type: ignore[attr-defined]
    )
    try:
        kq.control(
            [
                select.kevent(  # type: ignore[attr-defined]
                    fd,
                    filter=select.KQ_FILTER_VNODE,  # type: ignore[attr-defined]
                    flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,  # type: ignore[attr-defined]
                    fflags=fflags,
                )
            ],
            0,
        )
    except OSError:
        kq.close()
        os.close(fd)
        raise
    gone = select.KQ_NOTE_DELETE | select.KQ_NOTE_RENAME | select.KQ_NOTE_REVOKE  # type: ignore[attr-defined]

    def readable() -> None:
        try:
            events = kq.control(None, 16, 0)
        except OSError:
            onError()
            return
        if any(event.fflags & gone for event in events):
            onError()
            return
        # A vnode event names no entry: the directory as a whole changed.
        listener(None)

    loop.add_reader(kq.fileno(), readable)

    def close() -> None:
        with contextlib.suppress(Exception):
            loop.remove_reader(kq.fileno())
        with contextlib.suppress(OSError):
            kq.close()
        with contextlib.suppress(OSError):
            os.close(fd)

    return _NativeWatcher(close)


__all__ = ["NodeFileWatcher", "NodeWatchOptions"]
