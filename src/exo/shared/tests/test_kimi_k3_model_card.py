from pathlib import Path

import pytest

from exo.shared.models import model_cards
from exo.shared.models.model_cards import (
    ConfigData,
    ModelCard,
    ModelTask,
    VisionCardConfig,
)
from exo.shared.types.backends import Backend
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory


def test_kimi_k3_text_architecture_supports_tensor() -> None:
    config = ConfigData.model_validate(
        {
            "architectures": ["KimiK3ForConditionalGeneration"],
            "vision_config": {"model_type": "kimi_k3_vision", "image_token_id": 1},
            "text_config": {
                "architectures": ["KimiLinearForCausalLM"],
                "hidden_size": 7168,
                "num_hidden_layers": 93,
                "num_key_value_heads": 96,
                "max_position_embeddings": 1_048_576,
            },
        },
        context={"model_id": "moonshotai/Kimi-K3"},
    )

    assert config.supports_tensor is True
    assert config.layer_count == 93


def test_vision_autodetection_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    detected = VisionCardConfig(image_token_id=1, model_type="test_vision")

    def detect_vision(_: ModelId) -> VisionCardConfig:
        return detected

    monkeypatch.setattr(model_cards, "detect_vision_from_config", detect_vision)

    card = ModelCard(
        model_id=ModelId("test/text-only"),
        storage_size=Memory.from_bytes(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=True,
        autodetect_vision=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )

    assert card.vision is None


@pytest.mark.asyncio
async def test_builtin_kimi_k3_card_is_tensor_only_and_text_only() -> None:
    path = (
        Path(__file__).parents[4]
        / "resources/inference_model_cards/moonshotai--Kimi-K3.toml"
    )
    card = await ModelCard.load_from_path(path)  # type: ignore[arg-type]

    assert card.model_id == ModelId("moonshotai/Kimi-K3")
    assert card.supports_tensor is True
    assert card.supports_pipeline is False
    assert card.autodetect_vision is False
    assert card.trust_remote_code is True
    assert card.vision is None
    assert card.storage_size.in_bytes == 1_560_860_324_864
    assert card.context_length == 1_048_576
