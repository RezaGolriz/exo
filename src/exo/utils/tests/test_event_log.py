import os
from pathlib import Path
from threading import Event, Thread

import pytest

from exo.shared.types.events import TestEvent
from exo.utils.disk_event_log import DiskEventLog


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    return tmp_path / "event_log"


def test_append_and_read_back(log_dir: Path):
    log = DiskEventLog(log_dir)
    events = [TestEvent() for _ in range(5)]
    for e in events:
        log.append(e)

    assert len(log) == 5

    result = list(log.read_all())
    assert len(result) == 5
    for original, restored in zip(events, result, strict=True):
        assert original.event_id == restored.event_id

    log.close()


def test_read_range(log_dir: Path):
    log = DiskEventLog(log_dir)
    events = [TestEvent() for _ in range(10)]
    for e in events:
        log.append(e)

    result = list(log.read_range(3, 7))
    assert len(result) == 4
    for i, restored in enumerate(result):
        assert events[3 + i].event_id == restored.event_id

    log.close()


def test_read_range_bounds(log_dir: Path):
    log = DiskEventLog(log_dir)
    events = [TestEvent() for _ in range(3)]
    for e in events:
        log.append(e)

    # Start beyond count
    assert list(log.read_range(5, 10)) == []
    # Negative start
    assert list(log.read_range(-1, 2)) == []
    # End beyond count is clamped
    result = list(log.read_range(1, 100))
    assert len(result) == 2

    log.close()


def test_empty_log(log_dir: Path):
    log = DiskEventLog(log_dir)
    assert len(log) == 0
    assert list(log.read_all()) == []
    assert list(log.read_range(0, 10)) == []
    log.close()


def _archives(log_dir: Path) -> list[Path]:
    return sorted(log_dir.glob("events.*.bin.zst"))


def test_rotation_on_close(log_dir: Path):
    log = DiskEventLog(log_dir)
    log.append(TestEvent())
    log.close()

    active = log_dir / "events.bin"
    assert not active.exists()

    archives = _archives(log_dir)
    assert len(archives) == 1
    assert archives[0].stat().st_size > 0


def test_rotation_on_construction_with_stale_file(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "events.bin").write_bytes(b"stale data")

    log = DiskEventLog(log_dir)
    archives = _archives(log_dir)
    assert len(archives) == 1
    assert archives[0].exists()
    assert len(log) == 0

    log.close()


def test_empty_log_no_archive(log_dir: Path):
    """Closing an empty log should not leave an archive."""
    log = DiskEventLog(log_dir)
    log.close()

    active = log_dir / "events.bin"

    assert not active.exists()
    assert _archives(log_dir) == []


def test_close_is_idempotent(log_dir: Path):
    log = DiskEventLog(log_dir)
    log.append(TestEvent())
    log.close()
    archive = _archives(log_dir)
    log.close()  # should not raise

    assert _archives(log_dir) == archive


@pytest.mark.parametrize("has_events", [False, True])
def test_stale_close_preserves_successor_active_file(
    log_dir: Path, has_events: bool
) -> None:
    stale = DiskEventLog(log_dir)
    if has_events:
        stale.append(TestEvent())

    active = log_dir / "events.bin"
    active.unlink()
    active.write_bytes(b"successor")
    successor_identity = active.stat()

    stale.close()

    current_identity = active.stat()
    assert (current_identity.st_dev, current_identity.st_ino) == (
        successor_identity.st_dev,
        successor_identity.st_ino,
    )
    assert active.read_bytes() == b"successor"


def test_second_close_preserves_successor_active_file(log_dir: Path) -> None:
    stale = DiskEventLog(log_dir)
    stale.append(TestEvent())
    stale.close()
    active = log_dir / "events.bin"
    active.write_bytes(b"successor")

    stale.close()

    assert active.read_bytes() == b"successor"


def test_overlapping_constructor_preserves_successor_active_file(log_dir: Path) -> None:
    stale = DiskEventLog(log_dir)
    stale.append(TestEvent())
    successor = DiskEventLog(log_dir)
    successor.append(TestEvent())
    active = log_dir / "events.bin"
    successor_identity = active.stat()

    stale.close()

    current_identity = active.stat()
    assert (current_identity.st_dev, current_identity.st_ino) == (
        successor_identity.st_dev,
        successor_identity.st_ino,
    )
    assert len(list(successor.read_all())) == 1
    successor.close()


def test_constructor_waits_for_close_path_mutation(
    log_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = DiskEventLog(log_dir)
    stale.append(TestEvent())
    close_entered = Event()
    allow_close = Event()
    successor_created = Event()
    real_rotate = DiskEventLog._rotate  # pyright: ignore[reportPrivateUsage]

    def blocked_rotate(source: Path, directory: Path) -> None:
        close_entered.set()
        assert allow_close.wait(timeout=5)
        real_rotate(source, directory)

    monkeypatch.setattr(DiskEventLog, "_rotate", staticmethod(blocked_rotate))

    close_thread = Thread(target=stale.close)
    close_thread.start()
    assert close_entered.wait(timeout=5)

    successor: list[DiskEventLog] = []

    def construct_successor() -> None:
        successor.append(DiskEventLog(log_dir))
        successor_created.set()

    constructor_thread = Thread(target=construct_successor)
    constructor_thread.start()
    assert not successor_created.wait(timeout=0.1)

    allow_close.set()
    close_thread.join(timeout=5)
    constructor_thread.join(timeout=5)

    assert not close_thread.is_alive()
    assert not constructor_thread.is_alive()
    assert successor_created.is_set()
    assert (log_dir / "events.bin").exists()
    successor[0].close()


def test_close_stat_error_still_closes_file(
    log_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = DiskEventLog(log_dir)
    real_stat = Path.stat

    def deny_active_stat(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if path == log_dir / "events.bin":
            raise PermissionError("denied")
        return real_stat(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", deny_active_stat)

    log.close()

    assert log._file.closed  # pyright: ignore[reportPrivateUsage]


def test_successive_sessions(log_dir: Path):
    """Simulate two master sessions: both archives should be kept."""
    log1 = DiskEventLog(log_dir)
    log1.append(TestEvent())
    log1.close()

    first_archive = _archives(log_dir)[-1]

    log2 = DiskEventLog(log_dir)
    log2.append(TestEvent())
    log2.append(TestEvent())
    log2.close()

    # Session 1 archive shifted to slot 2, session 2 in slot 1
    second_archive = _archives(log_dir)[-1]
    should_be_first_archive = _archives(log_dir)[-2]

    assert first_archive.exists()
    assert second_archive.exists()
    assert first_archive != second_archive
    assert should_be_first_archive == first_archive


def test_rotation_keeps_at_most_5_archives(log_dir: Path):
    """After 7 sessions, only the 5 most recent archives should remain."""
    all_archives: list[Path] = []
    for _ in range(7):
        log = DiskEventLog(log_dir)
        log.append(TestEvent())
        log.close()
        all_archives.append(_archives(log_dir)[-1])

    for old in all_archives[:2]:
        assert not old.exists()
    for recent in all_archives[2:]:
        assert recent.exists()
