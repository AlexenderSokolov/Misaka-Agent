"""Credential persistence and resolution for coding-agent providers."""

from __future__ import annotations

import asyncio
import copy
import math
import os
import random
import stat
import threading
import time
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout
from pydantic import ValidationError

from misaka.ai.auth.types import (
    ApiKeyCredential as RuntimeApiKeyCredential,
)
from misaka.ai.auth.types import (
    AuthOperationOptions,
    Credential,
    CredentialInfo,
)
from misaka.ai.auth.types import (
    OAuthCredential as RuntimeOAuthCredential,
)
from misaka.ai.utils.sleep import sleep as abortable_sleep
from misaka.config import get_auth_path
from misaka.core.resolve_config_value import resolveConfigValue
from misaka.utils import atomic
from misaka.utils.abort import raceWithAbortSignal
from misaka.utils.js import (
    addAbortListener,
    isNumber,
    jsonParse,
    jsonStringify,
    startTask,
    throwIfAborted,
)
from misaka.utils.paths import (
    file_revision_from_stat,
    getFileRevision,
    normalizePath,
)
from misaka.utils.text import stripBom
from misaka.utils.values import signal_aborted

type ApiKeyCredential = dict[str, Any]
type OAuthCredential = dict[str, Any]
type AuthCredential = ApiKeyCredential | OAuthCredential
type AuthStorageData = dict[str, AuthCredential]
type FileRevision = tuple[int, int, int, int, int]


def _file_revision_from_stat(value: os.stat_result) -> FileRevision:
    # The same revision getFileRevision reads by path, or the two never match on Windows.
    return file_revision_from_stat(value)


@dataclass(slots=True)
class LockResult:
    result: Any
    next: str | None = None
    publish: Callable[[], None] | None = None


@dataclass(slots=True)
class _AuthFileReload:
    task: asyncio.Task[tuple[AuthStorageData, FileRevision | None]]
    readers: int = 0


_auth_file_reloads: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, _AuthFileReload]
] = weakref.WeakKeyDictionary()
_auth_file_reloads_lock = threading.Lock()


def _once_callback(callback: Callable[[], None]) -> Callable[[], None]:
    lock = threading.Lock()
    called = False

    def call() -> None:
        nonlocal called
        with lock:
            if called:
                return
            called = True
        callback()

    return call


class AuthStorageBackend:
    def withLock(self, fn: Callable[[str | None], LockResult]) -> Any:  # pragma: no cover - protocol-like
        raise NotImplementedError

    async def withLockAsync(
        self,
        fn: Callable[[str | None], Awaitable[LockResult]],
        options: AuthOperationOptions | None = None,
    ) -> Any:  # pragma: no cover
        raise NotImplementedError


class FileAuthStorageBackend(AuthStorageBackend):
    def __init__(self, authPath: str | None = None):
        self.authPath = normalizePath(authPath or get_auth_path())
        self._current_revision: FileRevision | None = None

    def _lock_path(self) -> str:
        return f"{self.authPath}.lock"

    def ensureParentDir(self) -> None:
        parent_dir = Path(self.authPath).parent
        if parent_dir.exists():
            return
        parent_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _target(self) -> str:
        """The file the credentials live in. ``auth.json`` may be a symlink (a dotfiles repo),
        followed once here; every open and write then works on the file it names, so a write
        replaces that file and leaves the link in place. A second hard link is allowed too, but
        the atomic replace gives the target a new inode, which detaches the other name."""
        return os.path.realpath(self.authPath)

    @staticmethod
    def _validate_file_stat(value: os.stat_result) -> None:
        if not stat.S_ISREG(value.st_mode):
            raise RuntimeError("Auth storage must be a regular file")
        if hasattr(os, "getuid") and value.st_uid != os.getuid():
            raise RuntimeError("Auth storage must be owned by the current user")
        if value.st_size == 0:
            raise RuntimeError("Auth storage is empty; restore or remove it before retrying")

    def _open_existing(self) -> int:
        target = self._target()
        before = os.lstat(target)
        self._validate_file_stat(before)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(target, flags)
        try:
            opened = os.fstat(fd)
            self._validate_file_stat(opened)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise RuntimeError("Auth storage changed while it was being opened")
            return fd
        except BaseException:
            os.close(fd)
            raise

    def ensureFileExists(self) -> None:
        # The mode applies only on creation so administrator-managed modes and ACLs remain intact.
        try:
            os.close(self._open_existing())
            return
        except FileNotFoundError:
            pass

        target = self._target()
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        try:
            fd = os.open(target, flags, 0o600)
        except FileExistsError:                            # another creator won; trust it only after validation
            os.close(self._open_existing())
            return

        created = None
        try:
            created = os.fstat(fd)
            if os.name != "nt" and hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
            payload = memoryview(b"{}")
            while payload:
                written = os.write(fd, payload)
                if written <= 0:
                    raise OSError("Could not initialize auth storage")
                payload = payload[written:]
            os.fsync(fd)
            os.close(fd)
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                current = os.lstat(target)
                if created is not None and (current.st_dev, current.st_ino) == (created.st_dev, created.st_ino):
                    os.unlink(target)
            except OSError:
                pass
            raise

    def _read_file(self) -> tuple[str, FileRevision]:
        fd = self._open_existing()
        before = _file_revision_from_stat(os.fstat(fd))
        with os.fdopen(fd, encoding="utf-8-sig") as handle:
            content = handle.read()
            after_stat = os.fstat(handle.fileno())
            self._validate_file_stat(after_stat)
            after = _file_revision_from_stat(after_stat)
            if after != before:
                raise RuntimeError("Auth storage changed while it was being read")
            return content, after

    def _create_lock(self) -> FileLock:
        return FileLock(self._lock_path(), timeout=0)

    def _lock_signature(self) -> tuple[int, int] | None:
        try:
            stat_result = os.stat(self._lock_path())
        except OSError:
            return None
        return (stat_result.st_dev, stat_result.st_ino)

    def _assert_lock_uncompromised(self, expected_signature: tuple[int, int] | None) -> None:
        if self._lock_signature() != expected_signature:
            raise RuntimeError("Auth storage lock was compromised")

    def acquireLockSyncWithRetry(self) -> FileLock:
        max_attempts = 10
        delay_ms = 20
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            lock = self._create_lock()
            try:
                lock.acquire(timeout=0)
                return lock
            except Timeout as error:
                last_error = error
                if attempt == max_attempts:
                    raise
                time.sleep(delay_ms / 1000)

        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to acquire auth storage lock")

    async def _acquire_lock_async_with_retry(
        self, options: AuthOperationOptions | None = None
    ) -> FileLock:
        """``acquireLockAsync``: retried with backoff until 30 s have passed, as long as another process holds it."""
        staleMs = 30_000
        maxDelayMs = 2_000
        deadline = time.monotonic() * 1000 + staleMs
        retry = 0
        while True:
            _throw_if_aborted(options)
            lock = self._create_lock()
            try:
                lock.acquire(timeout=0)
            except Timeout:
                _throw_if_aborted(options)
                remainingMs = deadline - time.monotonic() * 1000
                if remainingMs <= 0:
                    raise
                baseDelayMs = min(10 * 2**retry, maxDelayMs / 2)
                retry += 1
                delayMs = min(round(baseDelayMs * (1 + random.random())), remainingMs)
                await abortable_sleep(delayMs, options.signal if options is not None else None)
                continue
            try:
                _throw_if_aborted(options)
            except BaseException:
                lock.release()
                raise
            return lock

    def withLock(self, fn: Callable[[str | None], LockResult]) -> Any:
        self.ensureParentDir()
        lock = self.acquireLockSyncWithRetry()
        try:
            self.ensureFileExists()
            current, self._current_revision = self._read_file()
            outcome = fn(current)
            if outcome.next is not None:
                atomic.write_text(self._target(), outcome.next)
            if outcome.publish is not None:
                outcome.publish()
            return outcome.result
        finally:
            self._current_revision = None
            lock.release()

    async def withLockAsync(
        self,
        fn: Callable[[str | None], Awaitable[LockResult]],
        options: AuthOperationOptions | None = None,
    ) -> Any:
        _throw_if_aborted(options)
        self.ensureParentDir()
        lock = await self._acquire_lock_async_with_retry(options)
        expected_signature = self._lock_signature()
        try:
            self._assert_lock_uncompromised(expected_signature)
            _throw_if_aborted(options)
            self.ensureFileExists()
            self._assert_lock_uncompromised(expected_signature)
            _throw_if_aborted(options)
            current, self._current_revision = self._read_file()
            _throw_if_aborted(options)
            outcome = await fn(current)
            self._assert_lock_uncompromised(expected_signature)
            _throw_if_aborted(options)
            if outcome.next is not None:
                atomic.write_text(self._target(), outcome.next)
            self._assert_lock_uncompromised(expected_signature)
            if outcome.publish is not None:
                outcome.publish()
            return outcome.result
        finally:
            self._current_revision = None
            try:
                lock.release()
            except Exception:  # noqa: BLE001, S110 - releasing an already-released lock is fine
                pass


class InMemoryAuthStorageBackend(AuthStorageBackend):
    def __init__(self) -> None:
        self.value: str | None = None
        self._asyncChain: asyncio.Future[Any] | None = None

    def withLock(self, fn: Callable[[str | None], LockResult]) -> Any:
        outcome = fn(self.value)
        if outcome.next is not None:
            self.value = outcome.next
        if outcome.publish is not None:
            outcome.publish()
        return outcome.result

    async def withLockAsync(
        self,
        fn: Callable[[str | None], Awaitable[LockResult]],
        options: AuthOperationOptions | None = None,
    ) -> Any:
        previous = self._asyncChain

        async def operation() -> Any:
            # `await previous.catch(() => {})`: the chain moves on whether the previous operation failed or not.
            if previous is not None and not previous.done():
                await asyncio.wait([previous])
            _throw_if_aborted(options)
            outcome = await fn(self.value)
            _throw_if_aborted(options)
            if outcome.next is not None:
                self.value = outcome.next
            if outcome.publish is not None:
                outcome.publish()
            return outcome.result

        task = startTask(operation())
        self._asyncChain = task
        _observe(task)
        return await raceWithAbortSignal(task, options.signal if options is not None else None)


def _observe(task: asyncio.Future[Any]) -> None:
    task.add_done_callback(lambda finished: finished.exception() if not finished.cancelled() else None)

async def _with_lock_async(
    backend: AuthStorageBackend,
    fn: Callable[[str | None], Awaitable[LockResult]],
    options: AuthOperationOptions | None,
) -> Any:
    if options is None:
        return await backend.withLockAsync(fn)
    return await backend.withLockAsync(fn, options)


def _coerce_storage_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {str(index): item for index, item in enumerate(value)}
    if isinstance(value, str):
        return {str(index): item for index, item in enumerate(value)}
    return {}


def oauth_account_key(provider: str, account: str | None = None) -> str:
    """Named grants share the existing atomic auth store and its refresh lock."""
    if account is None:
        return provider
    if not account or len(account) > 64 or any(not (c.isascii() and (c.isalnum() or c in "-_.")) for c in account):
        raise ValueError("OAuth account names take 1–64 ASCII letters, digits, dots, underscores or hyphens")
    return f"{provider}:{account}"


class AuthStorage:
    def __init__(self, storage: AuthStorageBackend):
        self.data: Any = {}
        self.loadError: Exception | None = None
        self.storage = storage
        self._auth_path = storage.authPath if isinstance(storage, FileAuthStorageBackend) else None
        self._file_revision: FileRevision | None = None
        self._read_state_lock = threading.Lock()
        self._read_state_version = 0
        self._read_attempt_generation = 0
        self._read_applied_generation = 0
        self.reload()

    @classmethod
    def create(cls, authPath: str | None = None) -> AuthStorage:
        return cls(FileAuthStorageBackend(authPath))

    @classmethod
    def fromStorage(cls, storage: AuthStorageBackend) -> AuthStorage:
        return cls(storage)

    @classmethod
    def inMemory(cls, data: AuthStorageData | None = None) -> AuthStorage:
        storage = InMemoryAuthStorageBackend()
        storage.withLock(lambda _current: LockResult(result=None, next=jsonStringify(data or {}, 2)))
        return cls.fromStorage(storage)

    def parseStorageData(self, content: str | None) -> AuthStorageData:
        if not content:
            return {}
        return jsonParse(stripBom(content))

    def reload(self) -> None:
        content: str | None = None
        revision: FileRevision | None = None
        generation = (
            self._reserve_read_generation() if self._auth_path is not None else 0
        )

        def capture(current: str | None) -> LockResult:
            nonlocal content, revision
            content = current
            revision = (
                self.storage._current_revision
                if isinstance(self.storage, FileAuthStorageBackend)
                else None
            )
            return LockResult(result=None)

        try:
            self.storage.withLock(capture)
            data = self.parseStorageData(content)
            if self._auth_path is None:
                self.data = data
                self.loadError = None
            else:
                self._publish_file_snapshot(data, revision, generation)
        except Exception as error:  # noqa: BLE001
            self._record_file_load_error(error, generation)

    def _reserve_read_generation(self) -> int:
        with self._read_state_lock:
            self._read_attempt_generation += 1
            return self._read_attempt_generation

    def _record_file_load_error(self, error: Exception, generation: int) -> bool:
        if self._auth_path is None:
            self.loadError = error
            return True
        with self._read_state_lock:
            if generation < self._read_applied_generation:
                return False
            self._read_applied_generation = generation
            if self.loadError is error and self._file_revision is None:
                return True
            self.loadError = error
            self._file_revision = None
            self._read_state_version += 1
            return True

    def _publish_file_snapshot(
        self,
        data: AuthStorageData,
        revision: FileRevision | None,
        generation: int,
    ) -> AuthStorageData:
        """Publish only a snapshot that still matches the path's current revision."""
        if self._auth_path is None:  # pragma: no cover - file reloads only
            self.data = data
            return data

        with self._read_state_lock:
            if generation < self._read_applied_generation:
                return self.data
            current_revision = getFileRevision(self._auth_path)
            self._read_applied_generation = generation
            if revision != current_revision:
                if current_revision is not None and self._file_revision == current_revision:
                    return self.data
                self._file_revision = None
                return data
            if (
                self.data is data
                and self._file_revision == revision
                and self.loadError is None
            ):
                return self.data
            self.data = data
            self._file_revision = revision
            self.loadError = None
            self._read_state_version += 1
            return data

    def _replace_cached_data(self, data: Any) -> None:
        with self._read_state_lock:
            self._read_attempt_generation += 1
            self._read_applied_generation = self._read_attempt_generation
            self._read_state_version += 1
            self.data = data
            self._file_revision = None
            self.loadError = None

    async def reloadFromStorageAsync(
        self,
    ) -> tuple[AuthStorageData, FileRevision | None]:
        async def capture(current: str | None) -> LockResult:
            data = self.parseStorageData(current)
            revision = (
                self.storage._current_revision
                if isinstance(self.storage, FileAuthStorageBackend)
                else None
            )
            return LockResult(result=(data, revision))

        return await self.storage.withLockAsync(capture)

    async def _join_file_reload(
        self, signal: Any = None
    ) -> tuple[AuthStorageData, FileRevision | None]:
        if self._auth_path is None:  # pragma: no cover - guarded by readLatestData
            return self.data, None

        loop = asyncio.get_running_loop()
        with _auth_file_reloads_lock:
            reloads = _auth_file_reloads.setdefault(loop, {})
            reload = reloads.get(self._auth_path)
            if reload is None:
                reload = _AuthFileReload(loop.create_task(self.reloadFromStorageAsync()))
                reload.task.add_done_callback(
                    lambda task: task.exception() if not task.cancelled() else None
                )
                reloads[self._auth_path] = reload
            reload.readers += 1

        left = False

        def leave() -> None:
            nonlocal left
            if left:
                return
            left = True
            cancel_reload = False
            with _auth_file_reloads_lock:
                reload.readers -= 1
                reloads = _auth_file_reloads.get(loop)
                if reload.readers == 0 and reloads is not None and reloads.get(self._auth_path) is reload:
                    reloads.pop(self._auth_path, None)
                    if not reloads:
                        _auth_file_reloads.pop(loop, None)
                    cancel_reload = not reload.task.done()
            if cancel_reload:
                reload.task.cancel()

        # MISAKA fork: pi's reader leaves in the `finally` of its await, which on an abort runs as a microtask, before
        # any timer -- so the last reader stops the reload before its lock retry can fire again. asyncio has no
        # microtasks; the reader leaves the moment its signal aborts instead.
        removeAbortListener = addAbortListener(signal, leave) if signal is not None else None
        try:
            return await raceWithAbortSignal(asyncio.shield(reload.task), signal)
        finally:
            if removeAbortListener is not None:
                removeAbortListener()
            leave()

    async def readLatestData(
        self, options: AuthOperationOptions | None = None
    ) -> AuthStorageData:
        """Return the latest safe file snapshot without polling or watching in the background."""
        _throw_if_aborted(options)
        if self._auth_path is None:
            return self.data

        generation = self._reserve_read_generation()
        revision = getFileRevision(self._auth_path)
        if revision is not None:
            with self._read_state_lock:
                if revision == self._file_revision and revision == getFileRevision(
                    self._auth_path
                ):
                    self._read_applied_generation = max(
                        self._read_applied_generation, generation
                    )
                    return self.data

        try:
            data, revision = await self._join_file_reload(
                options.signal if options is not None else None
            )
        except Exception as error:  # A signal-bearing read reports storage errors.
            if options is not None and signal_aborted(options.signal):
                raise
            self._record_file_load_error(error, generation)
            if options is not None and options.signal is not None:
                raise
            return self.data

        data = self._publish_file_snapshot(data, revision, generation)
        _throw_if_aborted(options)
        return data

    # CredentialStore.

    async def read(
        self, providerId: str, options: AuthOperationOptions | None = None
    ) -> Credential | None:
        _throw_if_aborted(options)
        credential = _to_runtime_credential(
            _coerce_storage_object(await self.readLatestData(options)).get(providerId)
        )
        _throw_if_aborted(options)
        return credential

    async def list(
        self, options: AuthOperationOptions | None = None
    ) -> list[CredentialInfo]:
        _throw_if_aborted(options)
        latest = await self.readLatestData(options)
        # pi lists every record by its type, readable or not, so it can still be logged out.
        entries = [
            CredentialInfo(providerId=providerId, type=value["type"])
            for providerId, value in _coerce_storage_object(latest).items()
            if isinstance(value, dict) and value.get("type") in ("api_key", "oauth")
        ]
        _throw_if_aborted(options)
        return entries

    async def modify(
        self,
        providerId: str,
        fn: Callable[[Credential | None], Awaitable[Credential | None]],
        options: AuthOperationOptions | None = None,
    ) -> Credential | None:
        _throw_if_aborted(options)
        publish: Callable[[], None] | None = None

        async def update(current: str | None) -> LockResult:
            nonlocal publish
            current_data = _coerce_storage_object(self.parseStorageData(current))
            produced = await fn(
                _to_runtime_credential(
                    current_data.get(providerId), resolve_api_key=False
                )
            )
            _throw_if_aborted(options)
            if produced is None:
                publish = _once_callback(
                    lambda: self._replace_cached_data(current_data)
                )
                return LockResult(
                    result=_to_runtime_credential(
                        current_data.get(providerId), resolve_api_key=False
                    ),
                    publish=publish,
                )
            latest_data = {
                **current_data,
                providerId: _storedCredential(produced),
            }
            publish = _once_callback(
                lambda: self._replace_cached_data(latest_data)
            )
            return LockResult(
                result=produced,
                next=jsonStringify(latest_data, 2),
                publish=publish,
            )

        result = await _with_lock_async(self.storage, update, options)
        if publish is not None:
            publish()
        return result

    async def delete(
        self, providerId: str, options: AuthOperationOptions | None = None
    ) -> None:
        _throw_if_aborted(options)
        publish: Callable[[], None] | None = None

        async def remove(current: str | None) -> LockResult:
            nonlocal publish
            latest_data = _coerce_storage_object(self.parseStorageData(current))
            latest_data.pop(providerId, None)
            _throw_if_aborted(options)

            publish = _once_callback(lambda: self._replace_cached_data(latest_data))
            return LockResult(
                result=None,
                next=jsonStringify(latest_data, 2),
                publish=publish,
            )

        await _with_lock_async(self.storage, remove, options)
        if publish is not None:
            publish()


def _throw_if_aborted(options: AuthOperationOptions | None) -> None:
    if options is not None:
        throwIfAborted(options.signal)
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        raise asyncio.CancelledError


def _storedCredential(credential: Credential) -> AuthCredential:
    """The record ``JSON.stringify`` writes for a credential: the fields it was given, none it was left without."""
    return {"type": credential.type, **credential.model_dump(exclude_unset=True)}

def _to_runtime_credential(
    value: Any, *, resolve_api_key: bool = True
) -> Credential | None:
    if not isinstance(value, dict):
        return None
    # MISAKA fork: pi hands the stored object on as it is and its readers take the fields
    # they know. The credential models here check them, so an api_key record keeps only the
    # fields they declare, and a record they still cannot read is no credential -- not an
    # error that stops every provider's availability, or the login that would replace it.
    try:
        if value.get("type") == "api_key":
            credential = RuntimeApiKeyCredential.model_validate(
                {key: item for key, item in value.items() if key in RuntimeApiKeyCredential.model_fields}
            )
            if credential.key is None or not resolve_api_key:
                return credential
            # pi auth-storage.ts `read`: an unresolvable key reads as no key, so the provider
            # reports unconfigured instead of failing every availability check.
            return credential.model_copy(update={"key": resolveConfigValue(credential.key, credential.env)})
        if value.get("type") == "oauth":
            return RuntimeOAuthCredential.model_validate(value)
    except ValidationError:
        return None
    return None



class ReadOnlyAuthStorage:
    """A CredentialStore that reads auth.json once and never writes it (``ReadOnlyAuthStorage``)."""

    def __init__(self, authPath: str | None = None) -> None:
        self._authPath = FileAuthStorageBackend(authPath).authPath
        self._data: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        if self._data is not None:
            return self._data
        try:
            with open(self._authPath, encoding="utf-8", errors="replace") as file:
                parsed = jsonParse(stripBom(file.read()))
        except FileNotFoundError:
            self._data = {}
            return self._data
        except Exception as error:
            raise RuntimeError(f"Failed to read auth.json: {error}") from error
        if not isinstance(parsed, dict):
            raise RuntimeError("Invalid auth.json: expected an object")  # noqa: TRY004 - upstream throws Error
        for providerId, credential in parsed.items():
            if not isinstance(credential, dict):
                raise RuntimeError(f'Invalid auth.json credential for provider "{providerId}"')  # noqa: TRY004
            if credential.get("type") == "api_key":
                env = credential.get("env")
                validKey = "key" not in credential or isinstance(credential["key"], str)
                validEnv = "env" not in credential or (
                    isinstance(env, dict) and all(isinstance(entry, str) for entry in env.values())
                )
                if validKey and validEnv:
                    continue
            elif (
                credential.get("type") == "oauth"
                and isinstance(credential.get("access"), str)
                and isinstance(credential.get("refresh"), str)
                and isNumber(credential.get("expires"))
                and math.isfinite(credential["expires"])
            ):
                continue
            raise RuntimeError(f'Invalid auth.json credential for provider "{providerId}"')
        self._data = parsed
        return self._data

    async def read(self, providerId: str, options: AuthOperationOptions | None = None) -> Credential | None:
        _throw_if_aborted(options)
        credential = self._load().get(providerId)
        _throw_if_aborted(options)
        return _to_runtime_credential(copy.deepcopy(credential)) if credential else None

    async def list(self, options: AuthOperationOptions | None = None) -> list[CredentialInfo]:
        _throw_if_aborted(options)
        credentials = [
            CredentialInfo(providerId=providerId, type=credential["type"]) for providerId, credential in self._load().items()
        ]
        _throw_if_aborted(options)
        return credentials

    async def modify(
        self,
        providerId: str,
        fn: Callable[[Credential | None], Awaitable[Credential | None]],
        options: AuthOperationOptions | None = None,
    ) -> Credential | None:
        raise RuntimeError("Read-only credential storage cannot modify auth.json")

    async def delete(self, providerId: str, options: AuthOperationOptions | None = None) -> None:
        raise RuntimeError("Read-only credential storage cannot modify auth.json")


def readStoredCredential(providerId: str, authPath: str | None = None) -> AuthCredential | None:
    """One-off synchronous read of a stored credential from an auth.json file, without instantiating a store or
    resolving configured key values."""
    try:
        with open(FileAuthStorageBackend(authPath).authPath, encoding="utf-8", errors="replace") as file:
            data = jsonParse(stripBom(file.read()))
        return data.get(providerId) if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001 - upstream `catch { return undefined; }`
        return None


__all__ = [
    "ApiKeyCredential",
    "AuthCredential",
    "AuthStorage",
    "AuthStorageBackend",
    "AuthStorageData",
    "FileAuthStorageBackend",
    "InMemoryAuthStorageBackend",
    "OAuthCredential",
    "ReadOnlyAuthStorage",
    "readStoredCredential",
]
