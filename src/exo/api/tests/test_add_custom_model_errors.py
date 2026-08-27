# pyright: reportAny=false
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.main import API
from exo.shared.models.model_cards import ModelCard, ModelCardFetchError
from exo.shared.types.common import ModelId


def _client() -> TestClient:
    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]
    app.post("/models/add")(api.add_custom_model)
    return TestClient(app)


def test_add_model_surfaces_categorized_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail(_model_id: ModelId) -> ModelCard:
        raise ModelCardFetchError("missing_metadata", "weights are unavailable")

    monkeypatch.setattr(ModelCard, "fetch_from_hf", staticmethod(fail))
    response = _client().post("/models/add", json={"model_id": "some/model"})

    assert response.status_code == 400
    data: dict[str, Any] = response.json()
    assert data["error"]["message"] == "weights are unavailable"


def test_add_model_never_returns_empty_generic_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail(_model_id: ModelId) -> ModelCard:
        raise RuntimeError()

    monkeypatch.setattr(ModelCard, "fetch_from_hf", staticmethod(fail))
    response = _client().post("/models/add", json={"model_id": "some/model"})

    message: str = response.json()["error"]["message"]
    assert response.status_code == 400
    assert "some/model" in message
    assert "RuntimeError" in message
