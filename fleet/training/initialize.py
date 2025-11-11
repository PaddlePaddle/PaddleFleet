
import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from datetime import timedelta
from fleet.core import parallel_state as ps 


def intialize_fleet(strategy: fleet.DistributedStrategy):

    fleet.init(is_collective=True, strategy=strategy)
    hcg = fleet.get_hybrid_communicate_group()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    ps.intialize_model_parallel(hcg)
    print(f"fleet initialize successfully: {rank=} {world_size=}")


