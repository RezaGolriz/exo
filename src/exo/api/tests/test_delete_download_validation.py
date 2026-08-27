# pyright: reportUnusedFunction=false, reportAny=false

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from exo.api.main import API
from exo.shared.types.common import ModelId, NodeId


def _make_api() -> Any:
    api = object.__new__(API)
    api._send_download = AsyncMock()  # pyright: ignore[reportPrivateUsage]
    return api


def _make_test_client() -> tuple[TestClient, Any]:
    api = _make_api()
    app = FastAPI()
    app.delete("/download/{node_id}/{model_id:path}")(api.delete_download)
    return TestClient(app), api


@pytest.mark.parametrize("model_id", [".", "..", "org/../model", "model\\.."])
async def test_delete_download_rejects_unsafe_model_id(model_id: str) -> None:
    api = _make_api()

    with pytest.raises(HTTPException) as exc_info:
        await api.delete_download(NodeId("node"), ModelId(model_id))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid model id"
    api._send_download.assert_not_awaited()


async def test_delete_download_dispatches_valid_model_id() -> None:
    api = _make_api()

    response = await api.delete_download(NodeId("node"), ModelId("org/model"))

    assert response.command_id
    api._send_download.assert_awaited_once()


def test_encoded_traversal_returns_400_without_dispatch() -> None:
    client, api = _make_test_client()

    response = client.delete("/download/node/%2e%2e")

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid model id"}
    api._send_download.assert_not_awaited()
