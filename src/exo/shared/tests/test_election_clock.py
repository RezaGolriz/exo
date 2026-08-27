import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import pytest

from exo.shared.election_clock import DurableElectionClock


def _mint(store: DurableElectionClock) -> int:
    return store.mint(0)


def _mint_path(path: str) -> int:
    return DurableElectionClock(Path(path)).mint(0)


def test_first_boot_mints_clock_one(tmp_path: Path) -> None:
    store = DurableElectionClock(tmp_path / "election_clock")

    assert store.mint(observed=0) == 1


def test_restart_mints_above_highest_observed_clock(tmp_path: Path) -> None:
    path = tmp_path / "election_clock"
    first_process = DurableElectionClock(path)
    first_process.observe(41)

    restarted_process = DurableElectionClock(path)

    assert restarted_process.mint(observed=0) == 42
    assert path.read_text() == "42"


@pytest.mark.parametrize("corrupt", [b"", b" 7", b"07", b"-1", b"garbage"])
def test_corrupt_clock_fails_closed(tmp_path: Path, corrupt: bytes) -> None:
    path = tmp_path / "election_clock"
    path.write_bytes(corrupt)
    store = DurableElectionClock(path)

    with pytest.raises(ValueError, match="Corrupt election clock file"):
        store.mint(observed=0)


def test_concurrent_stores_mint_unique_clocks(tmp_path: Path) -> None:
    path = tmp_path / "election_clock"
    stores = [DurableElectionClock(path) for _ in range(8)]

    with ThreadPoolExecutor(max_workers=len(stores)) as executor:
        minted = list(executor.map(_mint, stores))

    assert sorted(minted) == list(range(1, len(stores) + 1))
    assert path.read_text() == str(len(stores))


def test_concurrent_processes_mint_unique_clocks(tmp_path: Path) -> None:
    path = tmp_path / "election_clock"
    process_count = 4

    with ProcessPoolExecutor(
        max_workers=process_count,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        minted = list(executor.map(_mint_path, [str(path)] * process_count))

    assert sorted(minted) == list(range(1, process_count + 1))
    assert path.read_text() == str(process_count)
