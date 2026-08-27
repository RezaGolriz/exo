from exo.api.main import (
    _resolved_instance_shape,  # pyright: ignore[reportPrivateUsage]
)
from exo.shared.types.common import NodeId
from exo.shared.types.worker.instances import InstanceMeta, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import PipelineShardMetadata, Sharding


def test_preview_reports_single_node_coercion_as_pipeline_ring() -> None:
    runner_id = RunnerId("runner")
    node_id = NodeId("node")
    instance = MlxRingInstance.model_construct(
        instance_id="instance",
        shard_assignments=ShardAssignments.model_construct(
            model_id="org/model",
            runner_to_shard={
                runner_id: PipelineShardMetadata.model_construct(
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=1,
                    n_layers=1,
                )
            },
            node_to_runner={node_id: runner_id},
        ),
        hosts_by_node={},
        ephemeral_port=1234,
    )

    sharding, instance_meta = _resolved_instance_shape(instance)

    assert sharding is Sharding.Pipeline
    assert instance_meta is InstanceMeta.MlxRing
