from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace

import pytest

from exo.utils import disk_usage


def _usage_400(_path: object) -> SimpleNamespace:
    return SimpleNamespace(total=1_000, used=600, free=400)


def _usage_200(_path: object) -> SimpleNamespace:
    return SimpleNamespace(total=1_000, used=800, free=200)


def _capacity_650(_path: Path) -> int:
    return 650


def _capacity_1500(_path: Path) -> int:
    return 1_500


def _no_capacity(_path: Path) -> None:
    return None


def _failed_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(returncode=1, stdout="")


def test_filesystem_capacity_uses_standard_free_space_off_macos(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(disk_usage.sys, "platform", "linux")
    monkeypatch.setattr(
        disk_usage.shutil,
        "disk_usage",
        _usage_400,
    )

    assert disk_usage.filesystem_capacity(tmp_path) == (1_000, 400)


def test_filesystem_capacity_uses_macos_important_usage_capacity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(disk_usage.sys, "platform", "darwin")
    monkeypatch.setattr(
        disk_usage.shutil,
        "disk_usage",
        _usage_200,
    )
    monkeypatch.setattr(
        disk_usage,
        "_macos_important_usage_capacity",
        _capacity_650,
    )

    assert disk_usage.filesystem_capacity(tmp_path) == (1_000, 650)


def test_filesystem_capacity_clamps_invalid_macos_capacity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        disk_usage.shutil,
        "disk_usage",
        _usage_200,
    )
    monkeypatch.setattr(
        disk_usage,
        "_macos_important_usage_capacity",
        _capacity_1500,
    )

    assert disk_usage.filesystem_capacity(tmp_path) == (1_000, 1_000)


def test_filesystem_capacity_uses_nearest_existing_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[Path] = []

    def fake_disk_usage(path: Path) -> SimpleNamespace:
        observed.append(path)
        return SimpleNamespace(total=1_000, used=600, free=400)

    monkeypatch.setattr(disk_usage.shutil, "disk_usage", fake_disk_usage)
    monkeypatch.setattr(
        disk_usage,
        "_macos_important_usage_capacity",
        _no_capacity,
    )

    missing = tmp_path / "models" / "nested"
    assert disk_usage.filesystem_capacity(missing) == (1_000, 400)
    assert observed == [tmp_path]


def test_macos_capacity_falls_back_when_osascript_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(disk_usage.sys, "platform", "darwin")
    monkeypatch.setattr(
        disk_usage.subprocess,
        "run",
        _failed_run,
    )

    assert disk_usage._macos_important_usage_capacity(tmp_path) is None  # pyright: ignore[reportPrivateUsage]


def test_macos_capacity_success_is_cached_for_same_volume_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = 0

    def successful_run(*_args: object, **_kwargs: object) -> CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return CompletedProcess(args=[], returncode=0, stdout="650\n", stderr="")

    monkeypatch.setattr(disk_usage.sys, "platform", "darwin")
    monkeypatch.setattr(disk_usage.shutil, "disk_usage", _usage_200)
    monkeypatch.setattr(disk_usage.subprocess, "run", successful_run)
    disk_usage._cached_macos_capacity.cache_clear()  # pyright: ignore[reportPrivateUsage]

    assert disk_usage.filesystem_capacity(tmp_path) == (1_000, 650)
    assert disk_usage.filesystem_capacity(tmp_path) == (1_000, 650)
    assert calls == 1
