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

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import paddle
from paddle import Tensor

from paddlefleet import parallel_state, tensor_parallel
from paddlefleet.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from paddlefleet.models.common.embeddings.rotary_pos_embedding import (
    RotaryEmbedding,
)
from paddlefleet.models.common.language_layer.language_layer import (
    LanguageLayer,
)
from paddlefleet.transformer.enums import ModelType
from paddlefleet.transformer.multi_token_prediction import (
    MTPLossAutoScaler,
    MTPLossLoggingHelper,
    MultiTokenPredictionBlock,
    roll_tensor,
)
from paddlefleet.transformer.transformer_block import TransformerBlock

if TYPE_CHECKING:
    from paddlefleet.packed_seq_params import PackedSeqParams
    from paddlefleet.process_groups_config import ProcessGroupCollection
    from paddlefleet.transformer.spec_utils import LayerSpec
    from paddlefleet.transformer.transformer_config import TransformerConfig


class GPTModel(LanguageLayer):
    """GPT Transformer language model.

    Args:
        config (TransformerConfig):
            Transformer config
        transformer_layer_spec (ModuleSpec):
            Specifies module to use for transformer layers
        vocab_size (int):
            Vocabulary size
        max_sequence_length (int):
            maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional):
            Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional):
            Include an output layer (used with pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool, optional):
            Defaults to False.
        parallel_output (bool, optional):
            Do not gather the outputs, keep them split across tensor
            parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool, optional):
            When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional):
            Position embedding type.. Defaults to 'learned_absolute'.
        rotary_percent (float, optional):
            Percent of rotary dimension to use for rotary position embeddings.
            Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional):
            Base period for rotary position embeddings. Ignored unless
            position_embedding_type is 'rope'.
            Defaults to 10000.
        rope_scaling (bool, optional): Toggle RoPE scaling.
        rope_scaling_factor (float): RoPE scaling factor. Default 8.
        scatter_embedding_sequence_parallel (bool, optional):
            Whether embeddings should be scattered across sequence parallel
            region or not. Defaults to True.
        seq_len_interpolation_factor (Optional[float], optional):
            scale of linearly interpolating RoPE for longer sequences.
            The value must be a float larger than 1.0. Defaults to None.
        pg_collection (ProcessGroupCollection): Model communication process groups
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: LayerSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal[
            "learned_absolute", "rope", "mrope", "yarn", "none"
        ] = "learned_absolute",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: float | None = None,
        mtp_block_spec: LayerSpec | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        vp_stage: int | None = None,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection)

        # if has_config_logger_enabled(config):
        #    log_config_to_disk(config, locals(), prefix=type(self).__name__)

        self.transformer_layer_spec: LayerSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = (
            share_embeddings_and_output_weights
        )
        self.vp_stage = vp_stage

        if hasattr(self.config, "position_embedding_type"):
            self.position_embedding_type = self.config.position_embedding_type
        else:
            self.position_embedding_type = position_embedding_type

        self.model_type = ModelType.encoder_or_decoder

        self.max_position_embeddings = max_sequence_length
        self.rotary_percent = rotary_percent

        if hasattr(self.config, "rotary_base"):
            self.rotary_base = self.config.rotary_base
        else:
            self.rotary_base = rotary_base
        self.rotary_scaling = rope_scaling
        self.mtp_block_spec = mtp_block_spec
        self.mtp_process = mtp_block_spec is not None

        if self.pre_process or self.mtp_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
                scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
                tp_group=self.pg_collection.tp,
            )

        if (
            self.position_embedding_type == "rope"
            and not self.config.multi_latent_attention
        ):
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                rope_scaling=rope_scaling,
                rope_scaling_factor=rope_scaling_factor,
                cp_group=self.pg_collection.cp,
            )

        # elif self.position_embedding_type == 'yarn':
        #    self.rotary_pos_emb = YarnRotaryEmbedding(
        #        kv_channels=self.config.kv_channels,
        #        rotary_percent=rotary_percent,
        #        rotary_interleaved=self.config.rotary_interleaved,
        #        seq_len_interpolation_factor=seq_len_interpolation_factor,
        #        rotary_base=rotary_base,
        #        scaling_factor=getattr(self.config, "yarn_rotary_scaling_factor"),
        #        original_max_position_embeddings=getattr(
        #            self.config, "yarn_original_max_position_embeddings"
        #        ),
        #        beta_fast=getattr(self.config, "yarn_beta_fast"),
        #        beta_slow=getattr(self.config, "yarn_beta_slow"),
        #        mscale=getattr(self.config, "yarn_mscale"),
        #        mscale_all_dim=getattr(self.config, "yarn_mscale_all_dim"),
        #        correction_range_round_to_int=getattr(
        #            self.config, "yarn_correction_range_round_to_int"
        #        ),
        #        use_cpu_initialization=self.config.use_cpu_initialization,
        #    )
        # elif self.position_embedding_type == 'mrope' and not self.config.multi_latent_attention:
        #    self.rotary_pos_emb = MultimodalRotaryEmbedding(
        #        kv_channels=self.config.kv_channels,
        #        rotary_percent=rotary_percent,
        #        rotary_interleaved=self.config.rotary_interleaved,
        #        seq_len_interpolation_factor=seq_len_interpolation_factor,
        #        rotary_base=rotary_base,
        #    )
        #    self.mrope_section = self.config.mrope_section
        #    assert (
        #        self.mrope_section is not None
        #    ), "mrope require mrope_section setting, but we got None from TransformerConfig"

        # Cache for RoPE tensors which do not change between iterations.
        self.rotary_pos_emb_cache = {}

        # Transformer.
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            pg_collection=self.pg_collection,
            vp_stage=vp_stage,
        )

        if self.mtp_process:
            self.mtp = MultiTokenPredictionBlock(
                config=self.config, spec=self.mtp_block_spec, vp_stage=vp_stage
            )

        # Output
        if self.post_process:
            if self.config.defer_embedding_wgrad_compute:
                # The embedding activation buffer preserves a reference to the input activations
                # of the final embedding projection layer GEMM. It will hold the activations for
                # all the micro-batches of a global batch for the last pipeline stage. Once we are
                # done with all the back props for all the microbatches for the last pipeline stage,
                # it will be in the pipeline flush stage. During this pipeline flush we use the
                # input activations stored in embedding activation buffer and gradient outputs
                # stored in gradient buffer to calculate the weight gradients for the embedding
                # final linear layer.
                self.embedding_activation_buffer = []
                self.grad_output_buffer = []
            else:
                self.embedding_activation_buffer = None
                self.grad_output_buffer = None

            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
                tp_group=self.pg_collection.tp,
            )

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See fleet.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, (
            "input_tensor should only be length 1 for gpt/bert"
        )
        self.decoder.set_input_tensor(input_tensor[0])

    def _preprocess(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        decoder_input: Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """Preprocesses inputs for the transformer decoder.

        Applies embeddings to input tokens, or uses `decoder_input` from a previous
        pipeline stage. Also sets up rotary positional embeddings.
        """

        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(
                input_ids=input_ids, position_ids=position_ids
            )
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        rotary_pos_cos = None
        rotary_pos_sin = None
        # this is used to store combined cos/sin embeddings, exclusively for flash infer rope
        rotary_pos_cos_sin = None

        if (
            self.position_embedding_type == "rope"
            and not self.config.multi_latent_attention
        ):
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                self.decoder, decoder_input, self.config, packed_seq_params
            )
            rotary_pos_emb = self.rotary_pos_emb(
                rotary_seq_len,
                packed_seq=packed_seq_params is not None
                and packed_seq_params.qkv_format == "thd",
            )
        # elif self.position_embedding_type == 'yarn':
        #    if self.training or not self.config.flash_decode:
        #        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
        #            inference_context, self.decoder, decoder_input, self.config, packed_seq_params
        #        )
        #        rotary_pos_emb, _ = self.rotary_pos_emb(rotary_seq_len)
        #    else:
        #        raise NotImplementedError(
        #            "Flash decoding uses precomputed cos and sin for RoPE, not implemented in "
        #            "YarnRotaryEmbedding yet."
        #        )
        # elif self.position_embedding_type == 'mrope' and not self.config.multi_latent_attention:
        #    if self.training or not self.config.flash_decode:
        #        rotary_pos_emb = self.rotary_pos_emb(position_ids, self.mrope_section)
        #    else:
        #        # Flash decoding uses precomputed cos and sin for RoPE
        #        raise NotImplementedError(
        #            "Flash decoding uses precomputed cos and sin for RoPE, not implemented in "
        #            "MultimodalRotaryEmbedding yet."
        #        )

        sequence_len_offset = None

        preproc_output = (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        )
        if rotary_pos_cos_sin is not None:
            # only in the case of flashinfer fused rope will we
            # return this extra tensor
            # this is for backwards compatibility with
            # legacy unit tests, which break if you
            # return a 6 tuple instead of 5.
            preproc_output += (rotary_pos_cos_sin,)

        return preproc_output

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor = None,
        attention_mask: Tensor = None,
        attn_mask_startend_row_indices: Tensor = None,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict | None = None,
        runtime_gather_output: bool | None = None,
        *,
        loss_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units

        Args:
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
        """

        if position_ids is None:
            batch_size, seq_length = input_ids.shape[0], input_ids.shape[1]
            position_ids = paddle.arange(seq_length, dtype="int64").expand(
                (batch_size, seq_length)
            )
        if attention_mask is None:
            attention_mask = paddle.ones_like(input_ids, dtype="int64")

        preproc_output = self._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=decoder_input,
            packed_seq_params=packed_seq_params,
        )

        (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        ) = preproc_output[:5]

        rotary_pos_cos_sin = (
            preproc_output[5] if len(preproc_output) == 6 else None
        )

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            **(extra_block_kwargs or {}),
        )

        return self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            mtp_in_postprocess=self.mtp_process,
            loss_mask=loss_mask,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            runtime_gather_output=runtime_gather_output,
            extra_block_kwargs=extra_block_kwargs,
        )

    def _postprocess(
        self,
        hidden_states,
        input_ids,
        position_ids,
        labels,
        rotary_pos_emb,
        rotary_pos_cos,
        rotary_pos_sin,
        mtp_in_postprocess=None,
        loss_mask=None,
        decoder_input=None,
        attention_mask=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        runtime_gather_output=None,
        extra_block_kwargs=None,
    ):
        """Postprocesses decoder hidden states to generate logits or compute loss.

        Applies Multi-Token Prediction if enabled, generates output logits through
        the output layer, and computes language model loss when labels are provided.
        """

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        if mtp_in_postprocess:
            hidden_states = self.mtp(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                embedding=self.embedding,
                **(extra_block_kwargs or {}),
            )

        if not self.post_process:
            return hidden_states

        if self.mtp_process:
            mtp_labels = labels.clone()
            hidden_states_list = paddle.chunk(
                hidden_states, 1 + self.config.mtp_num_layers, axis=0
            )
            hidden_states = hidden_states_list[0]
            if loss_mask is None:
                # if loss_mask is not provided, use all ones as loss_mask
                loss_mask = paddle.ones_like(mtp_labels)
            for mtp_layer_number in range(self.config.mtp_num_layers):
                # output
                mtp_logits, _ = self.output_layer(
                    hidden_states_list[mtp_layer_number + 1],
                    weight=output_weight,
                    # runtime_gather_output=runtime_gather_output,
                )
                # Calc loss for the current Multi-Token Prediction (MTP) layers.
                mtp_labels, _ = roll_tensor(
                    mtp_labels, shifts=-1, dims=-1, cp_group=self.cp_group
                )
                loss_mask, num_tokens = roll_tensor(
                    loss_mask, shifts=-1, dims=-1, cp_group=self.cp_group
                )
                mtp_loss = self.compute_language_model_loss(
                    mtp_labels, mtp_logits
                )
                mtp_loss = loss_mask * mtp_loss
                if self.training:
                    MTPLossLoggingHelper.save_loss_to_tracker(
                        paddle.sum(mtp_loss) / num_tokens,
                        mtp_layer_number,
                        self.config.mtp_num_layers,
                        avg_group=parallel_state.get_data_parallel_group(
                            with_context_parallel=True
                        ),
                    )
                mtp_loss_scale = (
                    self.config.mtp_loss_scaling_factor
                    / self.config.mtp_num_layers
                )
                if self.config.calculate_per_token_loss:
                    hidden_states = MTPLossAutoScaler.apply(
                        hidden_states, mtp_loss_scale * mtp_loss
                    )
                else:
                    hidden_states = MTPLossAutoScaler.apply(
                        hidden_states, mtp_loss_scale * mtp_loss / num_tokens
                    )
        sequence_parallel_override = False

        logits, _ = self.output_layer(
            hidden_states,
            weight=output_weight,
            # runtime_gather_output=runtime_gather_output,
        )

        loss = self.compute_language_model_loss(labels, logits)
        outputs = {"loss": loss, "logits": logits}
        return outputs

    def shared_embedding_or_output_weight(self) -> Tensor:
        """Gets the embedding weight or output logit weights when share input embedding and
        output weights set to True or when use Multi-Token Prediction (MTP) feature.

        Returns:
            Tensor: During pre processing or MTP process it returns the input embeddings weight.
            Otherwise, during post processing it returns the final output layers weight.
        """
        if self.pre_process or self.mtp_process:
            # Multi-Token Prediction (MTP) need both embedding layer and output layer.
            # So there will be both embedding layer and output layer in the mtp process stage.
            # In this case, if share_embeddings_and_output_weights is True, the shared weights
            # will be stored in embedding layer, and output layer will not have any weight.
            assert hasattr(self, "embedding"), (
                "embedding is needed in this pipeline stage, but it is not initialized."
            )
            return self.embedding.word_embeddings.weight.T
        elif self.post_process:
            return self.output_layer.weight
        return None

    def build_schedule_plan(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict | None = None,
        runtime_gather_output: bool | None = None,
        loss_mask: Tensor | None = None,
    ):
        """Builds a computation schedule plan for the model.

        This function creates a schedule plan for a model chunk, including
        preprocessing, transformer layers, and postprocessing.
        The schedule plan is used to optimize computation and memory usage
        in distributed environments.

        Args:
            input_ids (Tensor): Input token IDs.
            position_ids (Tensor): Position IDs.
            attention_mask (Tensor): Attention mask.
            decoder_input (Tensor, optional): Decoder input tensor. Defaults to None.
            labels (Tensor, optional): Labels for loss computation. Defaults to None.
            packed_seq_params (PackedSeqParams, optional):
                Parameters for packed sequences. Defaults to None.
            extra_block_kwargs (dict, optional):
                Additional keyword arguments for blocks. Defaults to None.
            runtime_gather_output (Optional[bool], optional):
                Whether to gather output at runtime. Defaults to None.
            loss_mask (Optional[Tensor], optional): Loss mask. Defaults to None.

        Returns:
            TransformerModelChunkSchedulePlan: The model chunk schedule plan.
        """

        from ..common.model_chunk_schedule_plan import (
            TransformerModelChunkSchedulePlan,
        )

        return TransformerModelChunkSchedulePlan(
            self,
            input_ids,
            position_ids,
            attention_mask,
            decoder_input,
            labels,
            packed_seq_params,
            extra_block_kwargs,
            runtime_gather_output,
            loss_mask,
        )

    def get_hardware_flops(self):
        return 989e3
