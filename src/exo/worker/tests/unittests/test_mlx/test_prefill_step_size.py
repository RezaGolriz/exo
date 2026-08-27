# pyright: reportAny=false, reportUnknownLambdaType=false
# pyright: reportUnknownArgumentType=false, reportInvalidCast=false
# pyright: reportPrivateUsage=false

import pytest

from exo.worker.engines.mlx.constants import (
    DEFAULT_PREFILL_STEP_SIZE,
    PREFILL_STEP_ENV,
    _parse_prefill_step_size,
)


def test_prefill_step_size_preserves_compatible_default() -> None:
    assert _parse_prefill_step_size(None) == DEFAULT_PREFILL_STEP_SIZE == 4096


def test_prefill_step_size_accepts_positive_override() -> None:
    assert _parse_prefill_step_size("1024") == 1024


@pytest.mark.parametrize("value", ["0", "-1", "invalid", "1.5"])
def test_prefill_step_size_rejects_invalid_override(value: str) -> None:
    with pytest.raises(ValueError, match=PREFILL_STEP_ENV):
        _parse_prefill_step_size(value)
