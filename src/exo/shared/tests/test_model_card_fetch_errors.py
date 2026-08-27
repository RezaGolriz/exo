import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from exo.download import download_utils
from exo.shared.models import model_cards
from exo.shared.models.model_cards import ModelCard, ModelCardFetchError
from exo.shared.types.common import ModelId


def _model_info_with_weights(_model_id: ModelId) -> SimpleNamespace:
    return SimpleNamespace(safetensors=SimpleNamespace(total=123_456))


def _model_info_without_weights(_model_id: ModelId) -> SimpleNamespace:
    return SimpleNamespace(safetensors=None)


def _mock_model_files(
    monkeypatch: pytest.MonkeyPatch,
    target_dir: Path,
    *,
    config: dict[str, Any] | None,
    index: dict[str, Any] | None,
) -> None:
    async def resolve(_model_id: ModelId) -> Path:
        return target_dir

    async def download(
        _model_id: ModelId,
        _revision: str,
        path: str,
        _target_dir: Path,
        *_args: object,
        **_kwargs: object,
    ) -> Path:
        data = config if path == "config.json" else index
        if data is None:
            raise FileNotFoundError()
        output = target_dir / path
        output.write_text(json.dumps(data))
        return output

    monkeypatch.setattr(download_utils, "resolve_model_dir", resolve)
    monkeypatch.setattr(download_utils, "download_file_with_retry", download)


@pytest.mark.anyio
async def test_missing_config_has_nonempty_actionable_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _mock_model_files(monkeypatch, tmp_path, config=None, index=None)

    with pytest.raises(ModelCardFetchError) as error:
        await ModelCard.fetch_from_hf(ModelId("missing/model"))

    assert error.value.category == "invalid_repository"
    assert "missing/model" in str(error.value)
    assert "config.json" in str(error.value)


@pytest.mark.anyio
async def test_single_safetensors_file_uses_hub_size_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _mock_model_files(
        monkeypatch,
        tmp_path,
        config={
            "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 2,
            "hidden_size": 64,
        },
        index=None,
    )
    monkeypatch.setattr(
        model_cards,
        "model_info",
        _model_info_with_weights,
    )

    card = await ModelCard.fetch_from_hf(ModelId("single/file-model"))

    assert card.storage_size.in_bytes == 123_456


@pytest.mark.anyio
async def test_missing_weight_metadata_is_categorized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _mock_model_files(
        monkeypatch,
        tmp_path,
        config={
            "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 2,
            "hidden_size": 64,
        },
        index=None,
    )
    monkeypatch.setattr(
        model_cards,
        "model_info",
        _model_info_without_weights,
    )

    with pytest.raises(ModelCardFetchError) as error:
        await ModelCard.fetch_from_hf(ModelId("missing/weights"))

    assert error.value.category == "missing_metadata"
    assert "storage size" in str(error.value)
