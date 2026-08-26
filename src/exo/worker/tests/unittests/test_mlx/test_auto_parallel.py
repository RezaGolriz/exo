import json
import multiprocessing as mp
import os
import tempfile
from typing import Any

import mlx.core as mx
import mlx.nn as mlx_nn
import pytest

from exo.worker.engines.mlx.auto_parallel import (
    CustomMlxLayer,
    PipelineFirstLayer,
    PipelineLastLayer,
    patch_pipeline_model,
    tensor_auto_parallel,
)
from exo.worker.tests.unittests.test_mlx.conftest import MockLayer


def run_pipeline_device(
    rank: int,
    world_size: int,
    hostfile_path: str,
    result_queue: Any,  # pyright: ignore[reportAny]
) -> None:
    import os

    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)

    class MockLayerInner(mlx_nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.custom_attr = "test_value"

        def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
            return x * 2

    class MockModel(mlx_nn.Module):
        def __init__(self, layers: list[mlx_nn.Module]) -> None:
            super().__init__()
            self.layers = layers

        def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
            for layer in self.layers:
                x = layer(x, *args, **kwargs)
            return x

    try:
        group = mx.distributed.init(backend="ring", strict=True)

        mock = MockLayerInner()
        first = PipelineFirstLayer(mock, r=rank, group=group)
        composed = PipelineLastLayer(first, r=rank, s=world_size, group=group)

        # Wrap in a mock model, then wrap in PipelineParallelModel for all_gather
        inner_model = MockModel([composed])
        model = patch_pipeline_model(inner_model, group)

        x = mx.ones((1, 4))
        result = model(x)
        mx.eval(result)
        success = result.shape == x.shape
        result_queue.put((rank, success, result))  # pyright: ignore[reportAny]
    except Exception as e:
        result_queue.put((rank, False, str(e)))  # pyright: ignore[reportAny]


def test_single_wrapper_delegates_attributes() -> None:
    mock = MockLayer()
    wrapped = CustomMlxLayer(mock)

    assert wrapped.custom_attr == "test_value"  # type: ignore[attr-defined]
    assert wrapped.use_sliding is True  # type: ignore[attr-defined]


def test_composed_wrappers_delegate_attributes() -> None:
    mock = MockLayer()
    group = mx.distributed.init()

    first = PipelineFirstLayer(mock, r=0, group=group)
    composed = PipelineLastLayer(first, r=0, s=1, group=group)

    assert composed.custom_attr == "test_value"  # type: ignore[attr-defined]
    assert composed.use_sliding is True  # type: ignore[attr-defined]


def test_missing_attribute_raises() -> None:
    mock = MockLayer()
    wrapped = CustomMlxLayer(mock)

    with pytest.raises(AttributeError):
        _ = wrapped.nonexistent_attr  # type: ignore[attr-defined]


def test_kimi_k3_uses_model_owned_tensor_sharding() -> None:
    class Model(mlx_nn.Module):
        model_type = "kimi_k3"

        def __init__(self) -> None:
            super().__init__()
            self.layers = [mlx_nn.Identity()]
            self.shard_calls = 0

        def shard(self, group: mx.distributed.Group) -> None:
            assert group.size() == 1
            self.shard_calls += 1

        def __call__(self, x: mx.array) -> mx.array:
            return x

    Model.__module__ = "mlx_lm.models.kimi_k3"
    model = Model()
    group = mx.distributed.init()

    responses = list(tensor_auto_parallel(model, group))

    assert model.shard_calls == 1
    assert len(responses) == 1
    assert responses[0].layers_loaded == 1
    assert responses[0].total == 1


def test_installed_kimi_k3_runs_through_exo_tensor_path() -> None:
    from mlx_lm.models.cache import ArraysCache, KVCache
    from mlx_lm.models.kimi_k3 import Model, ModelArgs

    args = ModelArgs(
        text_config={
            "model_type": "kimi_linear",
            "vocab_size": 128,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 96,
            "rms_norm_eps": 1e-5,
            "hidden_act": "situ",
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "linear_attn_config": {
                "kda_layers": [1],
                "full_attn_layers": [2],
                "num_heads": 2,
                "head_dim": 32,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
                "use_full_rank_gate": True,
            },
            "num_experts": 4,
            "moe_intermediate_size": 32,
            "q_lora_rank": 24,
            "kv_lora_rank": 16,
            "qk_nope_head_dim": 16,
            "qk_rope_head_dim": 8,
            "v_head_dim": 16,
            "mla_use_nope": True,
            "mla_use_output_gate": True,
            "num_experts_per_token": 2,
            "num_shared_experts": 1,
            "first_k_dense_replace": 1,
            "routed_expert_hidden_size": 32,
            "latent_moe_use_norm": True,
            "attn_res_block_size": 2,
        },
    )
    mx.random.seed(17)
    model = Model(args)
    tokens = mx.array([[2, 7, 11, 19]], dtype=mx.int32)
    expected = model(tokens)
    mx.eval(expected)

    original_call = Model.__call__
    try:
        responses = list(tensor_auto_parallel(model, mx.distributed.init()))
        assert Model.__call__ is not original_call
        actual = model(tokens)
        cache = model.make_cache()
        assert isinstance(cache[0], ArraysCache)
        assert isinstance(cache[1], KVCache)
        prefilled = model(tokens[:, :3], cache=cache)
        decoded = model(tokens[:, 3:4], cache=cache)
        mx.eval(actual, prefilled, decoded)
        assert decoded.shape == expected[:, -1:].shape
        assert cache[1].offset == tokens.shape[1]
    finally:
        Model.__call__ = original_call

    assert [(response.layers_loaded, response.total) for response in responses] == [
        (2, 2)
    ]
    assert mx.allclose(actual, expected, rtol=1e-5, atol=1e-5).item()
    assert mx.allclose(prefilled, expected[:, :3], rtol=1e-5, atol=1e-5).item()
    assert mx.allclose(decoded, expected[:, -1:], rtol=1e-5, atol=1e-5).item()


def test_composed_call_works() -> None:
    ctx = mp.get_context("spawn")

    world_size = 2
    base_port = 29500

    hosts = [f"127.0.0.1:{base_port + i}" for i in range(world_size)]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(hosts, f)
        hostfile_path = f.name

    try:
        result_queue: Any = ctx.Queue()

        processes: list[Any] = []
        for rank in range(world_size):
            p = ctx.Process(
                target=run_pipeline_device,
                args=(rank, world_size, hostfile_path, result_queue),
            )
            p.start()
            processes.append(p)

        for p in processes:  # pyright: ignore[reportAny]
            p.join(timeout=10)  # pyright: ignore[reportAny]

        results: dict[int, Any] = {}
        errors: dict[int, str] = {}
        while not result_queue.empty():  # pyright: ignore[reportAny]
            rank, success, value = result_queue.get()  # pyright: ignore[reportAny]
            if success:
                results[rank] = value
            else:
                errors[rank] = value

        assert len(results) == world_size, (
            f"Expected {world_size} results, got {len(results)}. Errors: {errors}"
        )

        for rank in range(world_size):
            assert rank in results, (
                f"Device {rank} failed: {errors.get(rank, 'unknown')}"
            )
            result_array = results[rank]
            # Both devices see the final result (4.0) after all_gather
            assert (result_array == 4.0).all(), (
                f"Device {rank}: expected 4.0, got {result_array}"
            )
    finally:
        os.unlink(hostfile_path)
