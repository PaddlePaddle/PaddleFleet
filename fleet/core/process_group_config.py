from dataclasses import dataclass, field, fields
from functools import partial
from typing import List, Optional

import paddle

from fleet.core import parallel_state


@dataclass
class ProcessGroupCollection:
    """Unified process group collection for transformer model parallelism, gradient communication,
     and finalization.

    Fields use init=False and must be set after instance creation.

    Args:
        # Model Parallelism Groups
        tp: Tensor parallel process group
        pp: Pipeline parallel process group
        cp: Context parallel process group
        ep: Expert model parallel group
        dp: Data parallel process group
        dp_cp: Data and context parallel group
        expt_dp: Expert data parallel group

    Example:
        # Create instance and set needed process groups
        pgs = ProcessGroupCollection()
        pgs.tp = tp_group
        pgs.pp = pp_group
        pgs.dp = dp_group

        # Pass to model components
        model = TransformerModel(..., pg_collection=pgs)
        ddp_model = DistributedDataParallel(..., pg_collection=pgs)
        finalize_model_grads(..., pg_collection=pgs)
    """

    # _TENSOR_MODEL_PARALLEL_GROUP
    tp: paddle.distributed.communication.group.Group = field(init=False)

    # _PIPELINE_MODEL_PARALLEL_GROUP
    pp: paddle.distributed.communication.group.Group = field(init=False)

    # _CONTEXT_PARALLEL_GROUP
    cp: paddle.distributed.communication.group.Group = field(init=False)

    # _EXPERT_MODEL_PARALLEL_GROUP
    ep: paddle.distributed.communication.group.Group = field(init=False)

    # _DATA_PARALLEL_GROUP
    dp: paddle.distributed.communication.group.Group = field(init=False)

    # _DATA_PARALLEL_GROUP_WITH_CP
    dp_cp: paddle.distributed.communication.group.Group = field(init=False)

    # _EXPERT_DATA_PARALLEL_GROUP
    expt_dp: paddle.distributed.communication.group.Group = field(init=False)

    def __init__(self, **kwargs):
        for key in kwargs:
            if key in [field.name for field in fields(self)]:
                setattr(self, key, kwargs[key])
            else:
                raise ValueError(f"Unknown attribute: {key}")

    
    @classmethod
    def use_mpu_process_groups(cls, required_pgs: Optional[List[str]] = None):
        """
        Use the default process groups from parallel_state.

        Args:
            required_pgs (List[str], optional): List of process group names to initialize.
                If None, pull all default process groups. Each string should correspond to
                one of the dataclass process group attributes.
        """
        # Get all available process groups
        all_pgs = {field.name for field in fields(cls)}

        # If no specific process groups requested, use all
        if required_pgs is None:
            required_pgs = list(all_pgs)

        # Validate requested process groups
        invalid_pgs = [pg for pg in required_pgs if pg not in all_pgs]
        if invalid_pgs:
            raise ValueError(f"Invalid process groups requested: {invalid_pgs}")

        # Mapping of attribute names to their initialization functions
        pg_to_func = {
            'tp': partial(parallel_state.get_tensor_model_parallel_group, check_initialized=False),
            'pp': partial(
                parallel_state.get_pipeline_model_parallel_group, check_initialized=False
            ),
            'cp': partial(parallel_state.get_context_parallel_group, check_initialized=False),
            'ep': partial(parallel_state.get_expert_model_parallel_group, check_initialized=False),
            'dp': parallel_state.get_data_parallel_group,
            'dp_cp': partial(parallel_state.get_data_parallel_group, with_context_parallel=True),
            'expt_dp': partial(
                parallel_state.get_expert_data_parallel_group, check_initialized=False
            ),
        }

        assert all(
            pg in pg_to_func for pg in required_pgs
        ), f"Initialization function for process group not defined for all \
        ProcessGroupCollection fields"

        # Build initialization dict by calling appropriate parallel_state get_foo_group
        init_dict = {pg: pg_to_func[pg]() for pg in required_pgs}

        return cls(**init_dict)
