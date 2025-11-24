# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import paddle
from paddle import Tensor

from paddlefleet.parallel_state import get_tensor_model_parallel_world_size

# from paddlefleet.dist_checkpointing.mapping import ShardedStateDict
from paddlefleet.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.transformer.layer import FleetLayer
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.utils import get_logger

# from paddlefleet.utils import make_tp_sharded_tensor_for_checkpoint

logger = get_logger(__name__)


class LanguageLayer(FleetLayer):
    """Base language Layer that has common helper functions used across GPT, BERT etc.

    Args:
        config (TransformerConfig): Input transformer config for the model
        pg_collection (ProcessGroupCollection): Model communication process groups
    """

    def __init__(
        # self, config: TransformerConfig, pg_collection: Optional[ProcessGroupCollection] = None
        self,
        config: TransformerConfig,
        pg_collection=None,
    ) -> None:
        super().__init__(config=config)
        self._set_attention_backend()
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.cp_group = pg_collection.cp
        self.pp_group = pg_collection.pp
        """
        assert hasattr(self.pg_collection, "embd"), (
            "pg_collection must have a embd. In previous version, it used default "
            "`parallel_state.default_embedding_ranks` to create the process group."
            "If you are using the default process group, please use"
            "`parallel_state.get_embedding_group()` "
            "If you don't need embd_group, you need to explicitly set it to None."
        )
        self.embd_group = pg_collection.embd
        """
        self.vp_stage = None
        self.vp_size = self.config.virtual_pipeline_model_parallel_size

        self.ignored_index = -100

        self.enable_parallel_cross_entropy = (
            paddle.distributed.is_initialized()
            and get_tensor_model_parallel_world_size() > 1
            and config.parallel_output
        )

        if (
            self.enable_parallel_cross_entropy
        ):  # and False: # and lm_head is distributed
            self.loss_func = (
                paddle.distributed.fleet.meta_parallel.ParallelCrossEntropy()
            )
        else:
            self.loss_func = paddle.nn.CrossEntropyLoss(
                reduction="none",
            )

    def _is_in_embd_group(self):
        if self.embd_group is None:
            return False
        if paddle.distributed.get_rank() in self.embd_group.ranks():
            if paddle.distributed.get_rank() == self.embd_group.ranks()[0]:
                return is_vp_first_stage(
                    self.vp_stage, self.vp_size
                ) and is_pp_first_stage(self.pp_group)
            elif paddle.distributed.get_rank() == self.embd_group.ranks()[-1]:
                return is_vp_last_stage(
                    self.vp_stage, self.vp_size
                ) and is_pp_last_stage(self.pp_group)
            else:
                return True
        return False

    # pylint: disable=line-too-long
    def _set_attention_backend(self):
        """Set attention backend

        Transformer engine works based on optout. By default all three attention backend flags are set to 1. So if the user chooses a particular attention backend we set the other two to 0. If the user chooses local, we set all 3 TE env variables to 0.
        """
        pass

    def compute_language_model_loss(
        self, labels: Tensor, logits: Tensor
    ) -> Tensor:
        """Computes the language model loss (Cross entropy across vocabulary)

        Args:
            labels (Tensor): The labels of dimension [batch size, seq length]
            logits (Tensor): The final logits returned by the output layer of the transformer model

        Returns:
            Tensor: Loss tensor of dimensions [batch size, sequence_length]
        """
        # TODO(pkuzyc): check the difference between vocab_parallel_cross_entropy
        # and paddle.nn.CrossEntropy, and use vocab_parallel_cross_entropy as loss func.
        loss = self.loss_func(logits.cast("float32"), labels)
        # loss = tensor_parallel.vocab_parallel_cross_entropy(
        #     logits.cast("float32"), labels
        # )

        lossmask = labels != self.ignored_index
        if (~lossmask).all():  # empty span
            logger.warning(
                f"encounter empty span when calculate loss, ignored_index={self.ignored_index}"
            )
            loss = paddle.mean(loss) * 0.0
        else:
            lossmask = lossmask.reshape([-1]).cast(paddle.float32)
            loss = paddle.sum(
                loss.cast(paddle.float32).reshape([-1]) * lossmask
            )
            loss = loss / lossmask.sum()

        return loss

    def setup_embeddings_and_output_layer(self) -> None:
        """Sets up embedding layer in first stage and output layer in last stage.

        This function initializes word embeddings in the final stage when we are
        using pipeline parallelism and sharing word embeddings, and sets up param
        attributes on the embedding and output layers.
        """

        # Set `is_embedding_or_output_parameter` attribute.
        if self.pre_process:
            self.embedding.word_embeddings.weight.is_embedding_or_output_parameter = True
        if self.post_process and self.output_layer.weight is not None:
            self.output_layer.weight.is_embedding_or_output_parameter = True

        # If share_embeddings_and_output_weights is True, we need to maintain duplicated
        # embedding weights in post processing stage. If use Multi-Token Prediction (MTP),
        # we also need to maintain duplicated embedding weights in mtp process stage.
        # So we need to copy embedding weights from pre processing stage as initial parameters
        # in these cases.
        if not self.share_embeddings_and_output_weights and not getattr(
            self.config, "num_nextn_predict_layers", 0
        ):
            return

        if self.config.pipeline_model_parallel_size == 1:
            # Zero out wgrad if sharing embeddings between two layers on same
            # pipeline stage to make sure grad accumulation into main_grad is
            # correct and does not include garbage values (e.g., from paddle.empty).
            self.shared_embedding_or_output_weight().zero_out_wgrad = True
            return

        if (
            is_vp_first_stage(self.vp_stage, self.vp_size)
            and is_pp_first_stage(self.pp_group)
            and self.pre_process
            and not self.post_process
        ):
            self.shared_embedding_or_output_weight().shared_embedding = True

        if (
            self.post_process or getattr(self, "mtp_process", False)
        ) and not self.pre_process:
            assert not (
                is_vp_first_stage(self.vp_stage, self.vp_size)
                and is_pp_first_stage(self.pp_group)
            )
            # set weights of the duplicated embedding to 0 here,
            # then copy weights from pre processing stage using all_reduce below.
            weight = self.shared_embedding_or_output_weight()
            weight.data.fill_(0)
            weight.shared = True
            weight.shared_embedding = True

        # Parameters are shared between the word embeddings layers, and the
        # heads at the end of the model. In a pipelined setup with more than
        # one stage, the initial embedding layer and the head are on different
        # workers, so we do the following:
        # 1. Create a second copy of word_embeddings on the last stage, with
        #    initial parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that
        #    the two copies of word_embeddings start off with the same
        #    parameter values.
        # 3. In the training loop, before an all-reduce between the grads of
        #    the two word_embeddings layers to ensure that every applied weight
        #    update is the same on both stages.

        # Ensure that first and last stages have the same initial parameter
        # values.
        if paddle.distributed.is_initialized():
            if self._is_in_embd_group():
                weight = self.shared_embedding_or_output_weight()
                paddle.distributed.all_reduce(weight, group=self.embd_group)

        elif not getattr(LanguageLayer, "embedding_warning_printed", False):
            logger.warning(
                "Distributed processes aren't initialized, so the output layer "
                "is not initialized with weights from the word embeddings. "
                "If you are just manipulating a model this is fine, but "
                "this needs to be handled manually. If you are training "
                "something is definitely wrong."
            )
            LanguageLayer.embedding_warning_printed = True

    def shared_embedding_or_output_weight(self) -> Tensor:
        """Gets the emedding weight or output logit weights when share embedding and output weights set to True.

        Returns:
            Tensor: During pre processing it returns the input embeddings weight while during post processing it returns the final output layers weight
        """
        if self.pre_process:
            return self.embedding.word_embeddings.weight
        elif self.post_process:
            return self.output_layer.weight
        return None

    # def sharded_state_dict(
    #    self,
    #    prefix: str = '',
    #    sharded_offsets: Tuple[Tuple[int, int, int]] = (),
    #    metadata: Optional[dict] = None,
    # ) -> ShardedStateDict:
    #    """Sharded state dict implementation that handles the output layer weights tying.

    #    Args:
    #        prefix (str): Module name prefix.
    #        sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
    #        metadata (Optional[Dict]): metadata controlling sharded state dict creation.

    #    Returns:
    #        ShardedStateDict: sharded state dict for the LanguageModel
    #    """
    #    assert not sharded_offsets, "Unexpected sharded offsets"
    #    sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)

    #    first_stage_word_emb_key = f'{prefix}embedding.word_embeddings.weight'
    #    output_layer_weight_key = f'{prefix}output_layer.weight'
    #    output_layer_bias_key = f'{prefix}output_layer.bias'

    #    if self.share_embeddings_and_output_weights:
    #        self.tie_embeddings_and_output_weights_state_dict(
    #            sharded_state_dict, output_layer_weight_key, first_stage_word_emb_key
    #        )
    #    elif self.post_process:
    #        # Make sure the output layer follows the embeddings padding logic
    #        sharded_state_dict[output_layer_weight_key].allow_shape_mismatch = True

    #    # Regardless of sharing the output weights with embeddings, we must handle the bias padding
    #    if self.post_process and output_layer_bias_key in sharded_state_dict:
    #        sharded_state_dict[output_layer_bias_key].allow_shape_mismatch = True

    #    return sharded_state_dict

    # def tie_embeddings_and_output_weights_state_dict(
    #    self,
    #    sharded_state_dict: ShardedStateDict,
    #    output_layer_weight_key: str,
    #    first_stage_word_emb_key: str,
    # ) -> None:
    #    """Ties the embedding and output weights in a given sharded state dict.

    #    Args:
    #        sharded_state_dict (ShardedStateDict): state dict with the weight to tie
    #        output_layer_weight_key (str): key of the output layer weight in the state dict.
    #            This entry will be replaced with a tied version
    #        first_stage_word_emb_key (str): this must be the same as the
    #            ShardedTensor.key of the first stage word embeddings.

    #    Returns: None, acts in-place
    #    """
    #    if not self.post_process:
    #        # No output layer
    #        assert output_layer_weight_key not in sharded_state_dict, sharded_state_dict.keys()
    #        return

    #    if self.pre_process:
    #        # Output layer is equivalent to the embedding already
    #        return

    #    # If use Multi-Token Prediction (MTP), we need maintain both embedding layer and output
    #    # layer in mtp process stage. In this case, if share_embeddings_and_output_weights is True,
    #    # the shared weights will be stored in embedding layer, and output layer will not have
    #    # any weight.
    #    if getattr(self, 'mtp_process', False):
    #        # No output layer
    #        assert output_layer_weight_key not in sharded_state_dict, sharded_state_dict.keys()
    #        return

    #    # Replace the default output layer with a one sharing the weights with the embedding
    #    del sharded_state_dict[output_layer_weight_key]
    #    tensor = self.shared_embedding_or_output_weight()
    #    last_stage_word_emb_replica_id = (
    #        1,  # copy of first stage embedding
    #        0,
    #        parallel_state.get_data_parallel_rank(with_context_parallel=True),
    #    )

    #    sharded_state_dict[output_layer_weight_key] = make_tp_sharded_tensor_for_checkpoint(
    #        tensor=tensor,
    #        key=first_stage_word_emb_key,
    #        replica_id=last_stage_word_emb_replica_id,
    #        allow_shape_mismatch=True,
    #    )
