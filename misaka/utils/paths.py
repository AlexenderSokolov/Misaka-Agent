"""Path normalization helpers shared across coding-agent modules."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

UNICODE_SPACES = re.compile(r"[\u00A0\u2000-\u200A\u202F\u205F\u3000]")
_WINDOWS_SHELL_DRIVE = re.compile(r"^/(?:mnt/|cygdrive/)?([a-zA-Z])(?:/(.*))?$")


def canonicalizePath(path: str) -> str:
    # `strict=` has existed since 3.10 and this package requires >=3.12, so there is no
    # interpreter that answers this call with TypeError; the only failure is the OSError
    # a missing or unresolvable path raises, and that answer is the path unchanged.
    try:
        return os.path.realpath(path, strict=True)
    except OSError:
        return path


def getFileRevision(path: str) -> tuple[int, int, int, int, int] | None:
    """Identity of the bytes currently at ``path`` (pi paths.ts:36-42).

    ``os.stat`` follows symlinks and imposes no shape filters, on purpose. An earlier
    revision of this helper used ``os.lstat`` plus ``S_ISREG``/``st_nlink == 1``/
    ``st_size > 0`` guards; each of those made a perfectly ordinary credential file
    (``~/.misaka/auth.json`` symlinked into a dotfiles repo, hard-linked by a backup
    tool, or momentarily zero bytes mid-write) return ``None`` forever, which silently
    disabled the concurrent-edit detection in ``core/auth_storage.py``. ``None`` here
    means only "cannot stat it".
    """
    try:
        stats = os.stat(path)
    except OSError:
        return None
    return file_revision_from_stat(stats)


def file_revision_from_stat(stats: os.stat_result) -> tuple[int, int, int, int, int]:
    """The revision ``getFileRevision`` reads, from a ``stat`` or an ``fstat`` alike.

    On Windows CPython (3.12+) fills ``st_ctime`` with the creation time from ``os.stat`` but
    with the last metadata change from ``os.fstat``, so once a file has been renamed into place
    the two never agree; the creation time both report stands in there.
    """
    changed = stats.st_birthtime_ns if os.name == "nt" else stats.st_ctime_ns
    return stats.st_dev, stats.st_ino, stats.st_size, stats.st_mtime_ns, changed


def isLocalPath(value: str) -> bool:
    """Whether a source names a local file, not a package, URL, or ``builtin:`` extension."""
    trimmed = value.strip()
    return not trimmed.startswith(("npm:", "git:", "github:", "http:", "https:", "ssh:", "builtin:"))


def normalizeWindowsShellPath(file_path: str) -> str:
    """Git Bash, MSYS, Cygwin and WSL drive paths in the form native Windows APIs accept (pi paths.ts:66-73)."""
    if not file_path.startswith("/") or file_path.startswith("//") or "\\" in file_path:
        return file_path
    match = _WINDOWS_SHELL_DRIVE.match(file_path)
    if match is None:
        return file_path
    suffix = (match[2] or "").replace("/", "\\")
    return f"{match[1].upper()}:\\{suffix}"


def normalizePath(
    value: str,
    *,
    trim: bool = False,
    expand_tilde: bool = True,
    home_dir: str | None = None,
    strip_at_prefix: bool = False,
    normalize_unicode_spaces: bool = False,
) -> str:
    normalized = value.strip() if trim else value
    if normalize_unicode_spaces:
        normalized = UNICODE_SPACES.sub(" ", normalized)
    if strip_at_prefix and normalized.startswith("@"):
        normalized = normalized[1:]
    if os.name == "nt":                       # on POSIX /mnt/c and /c are real directories
        normalized = normalizeWindowsShellPath(normalized)

    if expand_tilde:
        home = home_dir or str(Path.home())
        if normalized == "~":
            return home
        if normalized.startswith("~/") or (os.name == "nt" and normalized.startswith("~\\")):
            return os.path.join(home, normalized[2:])

    if normalized.startswith("file://"):
        parsed = urlparse(normalized)
        return url2pathname(unquote(f"{parsed.netloc}{parsed.path}" if parsed.netloc else parsed.path))

    return normalized


def resolvePath(value: str, base_dir: str | None = None, **options: object) -> str:
    normalized = normalizePath(value, **options)
    normalized_base_dir = normalizePath(base_dir or os.getcwd())
    if os.path.isabs(normalized):
        return os.path.abspath(normalized)
    return os.path.abspath(os.path.join(normalized_base_dir, normalized))


def getCwdRelativePath(file_path: str, cwd: str) -> str | None:
    resolved_cwd = resolvePath(cwd)
    resolved_path = resolvePath(file_path, resolved_cwd)
    try:
        relative_path = os.path.relpath(resolved_path, resolved_cwd)
    except ValueError:
        # Windows: path and cwd sit on different drives. Node's `relative()` hands back an
        # absolute path here, which pi then rejects via `isAbsolute` — so "not inside cwd"
        # is the answer, not an exception thrown through a caller that only wanted a label.
        return None
    is_inside_cwd = relative_path == "." or (
        relative_path != ".." and not relative_path.startswith(f"..{os.sep}") and not os.path.isabs(relative_path)
    )
    return relative_path if is_inside_cwd else None


def formatPathRelativeToCwdOrAbsolute(file_path: str, cwd: str) -> str:
    absolute_path = resolvePath(file_path, cwd)
    display = getCwdRelativePath(absolute_path, cwd) or absolute_path
    return display.replace(os.sep, "/")


def markPathIgnoredByCloudSync(path: str) -> None:
    if sys_platform() == "darwin":
        attrs = ["com.dropbox.ignored", "com.apple.fileprovider.ignore#P"]
        for attr in attrs:
            _best_effort_run(["xattr", "-w", attr, "1", path])
    elif sys_platform() == "linux":
        _best_effort_run(["setfattr", "-n", "user.com.dropbox.ignored", "-v", "1", path])


def _best_effort_run(command: list[str]) -> None:
    try:
        subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return


def sys_platform() -> str:
    if os.name == "nt":
        return "win32"
    return os.uname().sysname.lower()


def posix_relpath(path: str | os.PathLike[str], start: str | os.PathLike[str]) -> str:
    """``os.path.relpath`` with ``/`` between the parts on every platform: the form a relative
    path takes in a board row, a project file or a prompt, so what Windows writes reads the
    same everywhere (the grep and find tools already report paths this way)."""
    return os.path.relpath(path, start).replace(os.sep, "/")



__all__ = [
    "canonicalizePath",
    "file_revision_from_stat",
    "formatPathRelativeToCwdOrAbsolute",
    "getCwdRelativePath",
    "getFileRevision",
    "isLocalPath",
    "markPathIgnoredByCloudSync",
    "normalizePath",
    "posix_relpath",
    "resolvePath",
]
