import pytest
from anyio import Path

from exo.shared.models.model_cards import ConfigData, ModelCard


def test_glm_moe_dsa_custom_card_does_not_claim_unsupported_tensor_sharding() -> None:
    config = ConfigData.model_validate(
        {
            "architectures": ["GlmMoeDsaForCausalLM"],
            "model_type": "glm_moe_dsa",
            "num_hidden_layers": 78,
            "hidden_size": 6144,
            "num_key_value_heads": 64,
            "max_position_embeddings": 1_048_576,
        }
    )

    assert config.supports_tensor is False


@pytest.mark.anyio
async def test_glm52_catalog_card_declares_supported_pipeline_metadata() -> None:
    card_path = (
        Path(__file__).parents[4]
        / "resources/inference_model_cards/mlx-community--GLM-5.2-mxfp4.toml"
    )
    card = await ModelCard.load_from_path(card_path)

    assert card.family == "glm"
    assert card.quantization == "mxfp4"
    assert card.base_model == "GLM-5.2"
    assert card.supports_pipeline is True
    assert card.supports_tensor is False
    assert card.storage_size.in_bytes == 395_094_087_168
    assert card.reasoning_dialect == "post_last_user"
    assert card.sampling_defaults.temperature == 1.0
    assert card.sampling_defaults.top_p == 0.95
