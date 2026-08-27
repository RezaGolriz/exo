import json
import shutil
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

_IMPORTANT_USAGE_SCRIPT = """
ObjC.import("Foundation");
const url = $.NSURL.fileURLWithPath(PATH_PLACEHOLDER);
const value = Ref();
const error = Ref();
const ok = url.getResourceValueForKeyError(
    value,
    $.NSURLVolumeAvailableCapacityForImportantUsageKey,
    error
);
if (!ok) {
    throw new Error(ObjC.unwrap(error[0].localizedDescription));
}
ObjC.unwrap(value[0]).toString();
"""
_CAPACITY_CACHE_SECONDS = 5


def _nearest_existing_path(path: Path) -> Path:
    candidate = path.expanduser()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _macos_important_usage_capacity(path: Path) -> int | None:
    """Return Finder-style available capacity, including reclaimable APFS space."""
    if sys.platform != "darwin":
        return None

    script = _IMPORTANT_USAGE_SCRIPT.replace(
        "PATH_PLACEHOLDER", json.dumps(str(path.resolve()))
    )
    try:
        result = subprocess.run(
            ["/usr/bin/osascript", "-l", "JavaScript", "-e", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        capacity = int(result.stdout.strip())
    except ValueError:
        return None
    return capacity if capacity >= 0 else None


@lru_cache(maxsize=32)
def _cached_macos_capacity(path: str, time_bucket: int) -> int | None:
    del time_bucket
    return _macos_important_usage_capacity(Path(path))


def filesystem_capacity(path: Path) -> tuple[int, int]:
    """Return total and safely available bytes for the filesystem containing path."""
    existing_path = _nearest_existing_path(path)
    usage = shutil.disk_usage(existing_path)
    available = usage.free
    important_capacity = _cached_macos_capacity(
        str(existing_path.resolve()), int(time.monotonic() / _CAPACITY_CACHE_SECONDS)
    )
    if important_capacity is not None:
        available = important_capacity
    return usage.total, min(max(available, 0), usage.total)
