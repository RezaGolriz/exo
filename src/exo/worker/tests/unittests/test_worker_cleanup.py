from typing import cast

from exo.shared.types.worker.runners import RunnerId
from exo.worker.main import (
    _shutdown_runners,  # pyright: ignore[reportPrivateUsage]
)
from exo.worker.runner.supervisor import RunnerSupervisor


class _MutatingRunner:
    def __init__(
        self,
        runner_id: RunnerId,
        runners: dict[RunnerId, RunnerSupervisor],
        stopped: list[RunnerId],
    ) -> None:
        self._runner_id = runner_id
        self._runners = runners
        self._stopped = stopped

    def shutdown(self) -> None:
        self._stopped.append(self._runner_id)
        self._runners.pop(self._runner_id, None)


class _FailingRunner(_MutatingRunner):
    def shutdown(self) -> None:
        super().shutdown()
        raise RuntimeError("shutdown failed")


def test_runner_cleanup_uses_snapshot_when_shutdown_mutates_mapping() -> None:
    runners: dict[RunnerId, RunnerSupervisor] = {}
    stopped: list[RunnerId] = []
    runner_ids = [RunnerId("one"), RunnerId("two"), RunnerId("three")]
    for runner_id in runner_ids:
        runners[runner_id] = cast(
            RunnerSupervisor,
            cast(object, _MutatingRunner(runner_id, runners, stopped)),
        )

    _shutdown_runners(runners)

    assert set(stopped) == set(runner_ids)
    assert runners == {}


def test_runner_cleanup_continues_after_one_shutdown_raises() -> None:
    runners: dict[RunnerId, RunnerSupervisor] = {}
    stopped: list[RunnerId] = []
    runner_ids = [RunnerId("one"), RunnerId("two"), RunnerId("three")]
    for runner_id in runner_ids:
        runner_type = (
            _FailingRunner if runner_id == RunnerId("two") else _MutatingRunner
        )
        runners[runner_id] = cast(
            RunnerSupervisor,
            cast(object, runner_type(runner_id, runners, stopped)),
        )

    _shutdown_runners(runners)

    assert set(stopped) == set(runner_ids)
    assert runners == {}
