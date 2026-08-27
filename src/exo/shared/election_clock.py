import fcntl
import os
import re
import sys
import threading
from pathlib import Path
from typing import Protocol

from filelock import FileLock

_CLOCK_PATTERN = re.compile(rb"(?:0|[1-9][0-9]*)")
_MAX_CLOCK = 2**63 - 1
_LOCK_TIMEOUT_SECONDS = 10


class ElectionClock(Protocol):
    def observe(self, clock: int) -> None: ...

    def mint(self, observed: int) -> int: ...


class InMemoryElectionClock:
    def __init__(self, clock: int = 0) -> None:
        self._clock = clock

    def observe(self, clock: int) -> None:
        _validate_clock(clock)
        self._clock = max(self._clock, clock)

    def mint(self, observed: int) -> int:
        _validate_clock(observed)
        next_clock = max(self._clock, observed) + 1
        _validate_clock(next_clock)
        self._clock = next_clock
        return next_clock


class DurableElectionClock:
    """Crash-safe, process-safe high-water mark for election clocks."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file_lock = FileLock(
            f"{path}.lock",
            timeout=_LOCK_TIMEOUT_SECONDS,
        )
        self._thread_lock = threading.Lock()

    def observe(self, clock: int) -> None:
        _validate_clock(clock)
        with self._thread_lock, self._file_lock:
            current = self._read_unlocked()
            if clock > current:
                self._write_unlocked(clock)

    def mint(self, observed: int) -> int:
        _validate_clock(observed)
        with self._thread_lock, self._file_lock:
            current = self._read_unlocked()
            next_clock = max(current, observed) + 1
            _validate_clock(next_clock)
            self._write_unlocked(next_clock)
            return next_clock

    def _read_unlocked(self) -> int:
        try:
            raw = self._path.read_bytes()
        except FileNotFoundError:
            return 0
        if _CLOCK_PATTERN.fullmatch(raw) is None:
            raise ValueError(f"Corrupt election clock file: {self._path}")
        clock = int(raw)
        _validate_clock(clock)
        return clock

    def _write_unlocked(self, clock: int) -> None:
        payload = str(clock).encode("ascii")
        temporary = self._path.with_name(f".{self._path.name}.tmp")
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                offset = 0
                while offset < len(payload):
                    offset += os.write(fd, payload[offset:])
                _sync_file(fd)
            finally:
                os.close(fd)

            os.replace(temporary, self._path)
            directory_fd = os.open(self._path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)


def _validate_clock(clock: int) -> None:
    if clock < 0 or clock > _MAX_CLOCK:
        raise ValueError(f"Election clock is outside the supported range: {clock}")


def _sync_file(fd: int) -> None:
    if sys.platform == "darwin" and hasattr(fcntl, "F_FULLFSYNC"):
        try:
            fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
            return
        except OSError:
            pass
    os.fsync(fd)
