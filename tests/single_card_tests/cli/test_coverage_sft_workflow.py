# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
"""Single card tests for the SFT CLI workflow helpers.

``paddlefleet.cli.train.sft.workflow`` owns three things worth pinning
without a training job: the model-reproduction observation callback that
writes the loss / env receipts, the small offline validators the CLI
calls before it builds a dataset, and
``apply_glm_moe_dsa_training_contract``, which projects CLI and YAML
semantics onto the Fleet provider config.

Every case drives the production helper directly with stubbed
collaborators: namespaces for the argument dataclasses and tiny
``paddle.nn.Layer`` trees for the model. Nothing starts a process group
and nothing loads a checkpoint, so this stays a single-card test.
"""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import paddle

from paddlefleet.cli.train.sft import workflow as wf

# Every environment key the callback reads. Tests clear the whole set so a
# variable leaking in from the runner cannot flip a branch.
_REPRO_ENV_KEYS = (
    "MODEL_REPRO_ENV_PATH",
    "MODEL_REPRO_LOSS_PATH",
    "MODEL_REPRO_MODEL_CONFIG_PATH",
    "MODEL_REPRO_MODEL_ID",
    "MODEL_REPRO_MODEL_REVISION",
    "MRK_INVOCATION_ID",
)


def _patch_env(test, **values):
    """Install ``values`` and drop every other receipt environment key."""
    environ = {
        key: value
        for key, value in os.environ.items()
        if key not in _REPRO_ENV_KEYS
    }
    environ.update(values)
    patcher = mock.patch.dict(os.environ, environ, clear=True)
    patcher.start()
    test.addCleanup(patcher.stop)


def _tmpdir(test):
    """Return a directory that disappears when the test finishes."""
    holder = tempfile.TemporaryDirectory()
    test.addCleanup(holder.cleanup)
    return Path(holder.name)


class TokenizerLoadingTests(unittest.TestCase):
    """The processor stays on the weights path; the tokenizer may not."""

    def _patch_loaders(self, processor_result):
        calls = {}
        tokenizer = SimpleNamespace(name="tokenizer")

        def load_tokenizer(path):
            calls["tokenizer"] = path
            return tokenizer

        def load_processor(path, use_fast=None):
            calls["processor"] = (path, use_fast)
            if isinstance(processor_result, Exception):
                raise processor_result
            return processor_result

        patcher = mock.patch.multiple(
            wf,
            AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
            AutoProcessor=SimpleNamespace(from_pretrained=load_processor),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return calls, tokenizer

    def test_the_tokenizer_path_wins_over_the_weights_path(self):
        processor = SimpleNamespace(name="processor")
        calls, tokenizer = self._patch_loaders(processor)
        loaded_tokenizer, loaded_processor = wf.load_tokenizer_and_processor(
            SimpleNamespace(
                tokenizer_name_or_path="/tok", model_name_or_path="/model"
            ),
            SimpleNamespace(processor_use_fast=True),
        )
        self.assertIs(loaded_tokenizer, tokenizer)
        self.assertIs(loaded_processor, processor)
        self.assertEqual(calls["tokenizer"], "/tok")
        self.assertEqual(calls["processor"], ("/model", True))

    def test_the_weights_path_is_used_when_no_tokenizer_is_configured(self):
        processor = SimpleNamespace(name="processor")
        calls, _ = self._patch_loaders(processor)
        wf.load_tokenizer_and_processor(
            SimpleNamespace(
                tokenizer_name_or_path=None, model_name_or_path="/model"
            ),
            SimpleNamespace(processor_use_fast=False),
        )
        self.assertEqual(calls["tokenizer"], "/model")
        self.assertEqual(calls["processor"], ("/model", False))

    def test_an_independent_tokenizer_may_serve_as_the_processor(self):
        _, tokenizer = self._patch_loaders(OSError("no processor files"))
        loaded_tokenizer, loaded_processor = wf.load_tokenizer_and_processor(
            SimpleNamespace(
                tokenizer_name_or_path="/tok", model_name_or_path="/model"
            ),
            SimpleNamespace(processor_use_fast=False),
        )
        self.assertIs(loaded_tokenizer, tokenizer)
        self.assertIs(loaded_processor, tokenizer)

    def test_a_shared_path_keeps_the_processor_failure(self):
        self._patch_loaders(ValueError("broken processor"))
        with self.assertRaises(ValueError):
            wf.load_tokenizer_and_processor(
                SimpleNamespace(
                    tokenizer_name_or_path="/model",
                    model_name_or_path="/model",
                ),
                SimpleNamespace(processor_use_fast=False),
            )


class FinalHfExportTests(unittest.TestCase):
    """The final HF export runs regardless of the save_to_hf cadence."""

    def test_tensor_parallel_runs_merge_the_shards(self):
        calls = []
        trainer = SimpleNamespace(
            save_model=lambda **kwargs: calls.append(kwargs)
        )
        self.assertTrue(
            wf.save_final_hf_model_if_requested(
                trainer, SimpleNamespace(tensor_model_parallel_size=4)
            )
        )
        self.assertEqual(
            calls,
            [{"merge_tensor_parallel": True, "last_fc_to_hf": True}],
        )

    def test_single_card_runs_do_not_merge(self):
        calls = []
        trainer = SimpleNamespace(
            save_model=lambda **kwargs: calls.append(kwargs)
        )
        wf.save_final_hf_model_if_requested(
            trainer, SimpleNamespace(tensor_model_parallel_size=1)
        )
        self.assertEqual(
            calls,
            [{"merge_tensor_parallel": False, "last_fc_to_hf": True}],
        )


def _sequence(length=4, **overrides):
    """A pretokenized row as ``create_indexed_dataset`` yields it."""
    values = list(range(length))
    fields = {
        "token_ids": list(values),
        "position_ids": list(values),
        "labels": list(values),
        "num_examples": 1,
    }
    fields.update(overrides)
    return wf.TextSequence(**fields)


class PretokenizedValidationTests(unittest.TestCase):
    """The offline validator refuses anything the trainer cannot pack."""

    def test_a_valid_dataset_passes(self):
        self.assertIsNone(
            wf.validate_pretokenized_offline_dataset([[_sequence()]], 4)
        )

    def test_empty_datasets_are_rejected(self):
        for dataset in (None, []):
            with self.subTest(dataset=dataset):
                with self.assertRaises(ValueError) as caught:
                    wf.validate_pretokenized_offline_dataset(dataset, 4)
                self.assertIn("at least one row", str(caught.exception))

    def test_rows_must_hold_exactly_one_sequence(self):
        with self.assertRaises(ValueError) as caught:
            wf.validate_pretokenized_offline_dataset(
                [[_sequence(), _sequence()]], 4
            )
        self.assertIn("exactly one TextSequence", str(caught.exception))

    def test_bare_sequences_are_rejected(self):
        with self.assertRaises(ValueError) as caught:
            wf.validate_pretokenized_offline_dataset([_sequence()], 4)
        self.assertIn("exactly one TextSequence", str(caught.exception))

    def test_a_length_mismatch_names_the_field(self):
        with self.assertRaises(ValueError) as caught:
            wf.validate_pretokenized_offline_dataset(
                [[_sequence(labels=[0, 1])]], 4
            )
        self.assertIn(
            "pretokenized labels length 2 != 4", str(caught.exception)
        )

    def test_non_integer_values_are_rejected(self):
        with self.assertRaises(TypeError) as caught:
            wf.validate_pretokenized_offline_dataset(
                [[_sequence(position_ids=[0, 1, 2, 3.5])]], 4
            )
        self.assertIn(
            "pretokenized position_ids must contain integer values",
            str(caught.exception),
        )


def _dsa_config(**overrides):
    """A provider config double for the GLM MoE DSA contract."""
    values = {
        "model_type": "glm_moe_dsa",
        "mtp_num_layers": 0,
        "mtp_loss_scaling_factor": 0.3,
        "persist_layer_norm": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _dsa_training_args(**overrides):
    values = {
        "use_accuracy_compatible": False,
        "num_nextn_predict_layers": 0,
        "mtp_num_layers": 0,
        "mtp_loss_scaling_factor": None,
        "fp32_residual_connection": True,
        "moe_token_dispatcher_type": "flex",
        "moe_router_bias_update_rate": 0.002,
        "moe_expert_fusion": None,
        "bias_activation_fusion": None,
        "overlap_p2p_comm": None,
        "batch_p2p_comm": None,
        "variable_seq_lengths": None,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "sequence_parallel": False,
        "expert_tensor_model_parallel_size": -1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _dsa_model_args(**overrides):
    values = {"mtp_attention_flexible": False, "persist_layer_norm": None}
    values.update(overrides)
    return SimpleNamespace(**values)


def _dsa_data_args(**overrides):
    values = {"pretokenized_dataset": False}
    values.update(overrides)
    return SimpleNamespace(**values)


class GlmMoeDsaContractTests(unittest.TestCase):
    """CLI semantics reach the Fleet provider config, or fail closed."""

    def _apply(self, config, training, model=None, data=None):
        wf.apply_glm_moe_dsa_training_contract(
            config,
            training,
            model or _dsa_model_args(),
            data or _dsa_data_args(),
        )

    def test_other_model_types_are_left_alone(self):
        config = _dsa_config(model_type="qwen2_moe")
        self._apply(config, _dsa_training_args(num_nextn_predict_layers=1))
        self.assertFalse(hasattr(config, "use_accuracy_compatible"))
        self.assertEqual(config.mtp_num_layers, 0)
        self.assertFalse(hasattr(config, "mtp_enabled"))

    def test_the_accuracy_target_is_canonicalized(self):
        config = _dsa_config()
        # A YAML ``true`` can arrive as a truthy string.
        self._apply(config, _dsa_training_args(use_accuracy_compatible="true"))
        self.assertEqual(config.use_accuracy_compatible, "megatron")
        other = _dsa_config()
        self._apply(other, _dsa_training_args(use_accuracy_compatible="false"))
        self.assertFalse(other.use_accuracy_compatible)

    def test_the_mtp_depth_is_mirrored_and_the_legacy_key_dropped(self):
        config = _dsa_config()
        training = _dsa_training_args(num_nextn_predict_layers=1)
        self._apply(config, training)
        self.assertEqual(config.num_nextn_predict_layers, 1)
        self.assertEqual(training.num_nextn_predict_layers, 1)
        self.assertTrue(config.mtp_enabled)
        # TransformerConfig rejects a non-zero ``mtp_num_layers``.
        self.assertEqual(training.mtp_num_layers, 0)
        self.assertFalse(hasattr(config, "mtp_num_layers"))

    def test_an_explicit_mtp_num_layers_sets_the_depth(self):
        config = _dsa_config()
        training = _dsa_training_args(mtp_num_layers=2)
        self._apply(config, training)
        self.assertEqual(config.num_nextn_predict_layers, 2)
        self.assertEqual(training.num_nextn_predict_layers, 2)
        self.assertTrue(config.mtp_enabled)

    def test_a_zero_depth_leaves_mtp_disabled(self):
        config = _dsa_config()
        self._apply(config, _dsa_training_args())
        self.assertEqual(config.num_nextn_predict_layers, 0)
        self.assertFalse(config.mtp_enabled)

    def test_conflicting_mtp_depths_are_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self._apply(
                _dsa_config(),
                _dsa_training_args(
                    num_nextn_predict_layers=1, mtp_num_layers=2
                ),
            )
        self.assertIn("MTP depth mismatch", str(caught.exception))

    def test_the_mtp_loss_weight_is_taken_as_a_float(self):
        config = _dsa_config()
        self._apply(
            config,
            _dsa_training_args(
                num_nextn_predict_layers=1, mtp_loss_scaling_factor="0.25"
            ),
        )
        self.assertEqual(config.mtp_loss_scaling_factor, 0.25)
        self.assertIsInstance(config.mtp_loss_scaling_factor, float)

    def test_the_provider_default_mtp_loss_weight_survives(self):
        config = _dsa_config(mtp_loss_scaling_factor=0.3)
        self._apply(config, _dsa_training_args(num_nextn_predict_layers=1))
        self.assertEqual(config.mtp_loss_scaling_factor, 0.3)

    def test_pretokenized_mtp_requires_flexible_attention(self):
        with self.assertRaises(ValueError) as caught:
            self._apply(
                _dsa_config(),
                _dsa_training_args(num_nextn_predict_layers=1),
                data=_dsa_data_args(pretokenized_dataset=True),
            )
        self.assertIn("mtp_attention_flexible=true", str(caught.exception))

    def test_pretokenized_mtp_passes_with_flexible_attention(self):
        config = _dsa_config()
        self._apply(
            config,
            _dsa_training_args(num_nextn_predict_layers=1),
            model=_dsa_model_args(mtp_attention_flexible=True),
            data=_dsa_data_args(pretokenized_dataset=True),
        )
        self.assertEqual(config.num_nextn_predict_layers, 1)

    def test_moe_and_residual_semantics_are_copied(self):
        config = _dsa_config()
        self._apply(
            config,
            _dsa_training_args(
                fp32_residual_connection=False,
                moe_token_dispatcher_type="flex",
                moe_router_bias_update_rate=0.002,
                moe_expert_fusion=1,
            ),
        )
        self.assertFalse(config.fp32_residual_connection)
        self.assertEqual(config.moe_token_dispatcher_type, "flex")
        self.assertEqual(config.moe_router_bias_update_rate, 0.002)
        self.assertIs(config.moe_expert_fusion, True)

    def test_absent_moe_fields_fall_back_to_the_documented_defaults(self):
        config = _dsa_config()
        training = _dsa_training_args()
        del training.moe_token_dispatcher_type
        del training.moe_router_bias_update_rate
        del training.moe_expert_fusion
        self._apply(config, training)
        self.assertEqual(config.moe_token_dispatcher_type, "alltoall")
        self.assertEqual(config.moe_router_bias_update_rate, 0.001)
        self.assertFalse(hasattr(config, "moe_expert_fusion"))

    def test_only_configured_fusion_flags_reach_the_config(self):
        config = _dsa_config()
        self._apply(
            config,
            _dsa_training_args(
                bias_activation_fusion=True,
                overlap_p2p_comm=False,
                batch_p2p_comm=None,
                variable_seq_lengths=True,
            ),
        )
        self.assertTrue(config.bias_activation_fusion)
        self.assertFalse(config.overlap_p2p_comm)
        self.assertTrue(config.variable_seq_lengths)
        # ``None`` means "keep the provider default".
        self.assertFalse(hasattr(config, "batch_p2p_comm"))

    def test_parallel_degrees_are_clamped_to_at_least_one(self):
        config = _dsa_config()
        self._apply(
            config,
            _dsa_training_args(
                tensor_model_parallel_size=-1,
                pipeline_model_parallel_size=2,
                context_parallel_size=0,
                expert_model_parallel_size=8,
                sequence_parallel=1,
            ),
        )
        self.assertEqual(config.tensor_model_parallel_size, 1)
        self.assertEqual(config.pipeline_model_parallel_size, 2)
        self.assertEqual(config.context_parallel_size, 1)
        self.assertEqual(config.expert_model_parallel_size, 8)
        self.assertIs(config.sequence_parallel, True)

    def test_an_unset_expert_tensor_degree_becomes_one(self):
        config = _dsa_config()
        self._apply(
            config,
            _dsa_training_args(expert_tensor_model_parallel_size=-1),
        )
        self.assertEqual(config.expert_tensor_parallel_size, 1)

    def test_a_configured_expert_tensor_degree_is_kept(self):
        config = _dsa_config()
        self._apply(
            config, _dsa_training_args(expert_tensor_model_parallel_size=4)
        )
        self.assertEqual(config.expert_tensor_parallel_size, 4)

    def test_a_zero_expert_tensor_degree_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self._apply(
                _dsa_config(),
                _dsa_training_args(expert_tensor_model_parallel_size=0),
            )
        self.assertIn(
            "expert_tensor_model_parallel_size must be -1 or at least 1",
            str(caught.exception),
        )

    def test_persist_layer_norm_is_only_overridden_when_configured(self):
        config = _dsa_config(persist_layer_norm=False)
        self._apply(
            config,
            _dsa_training_args(),
            model=_dsa_model_args(persist_layer_norm=True),
        )
        self.assertTrue(config.persist_layer_norm)
        untouched = _dsa_config(persist_layer_norm=False)
        self._apply(untouched, _dsa_training_args())
        self.assertFalse(untouched.persist_layer_norm)


class _LayeredModel(paddle.nn.Layer):
    """``model.layers.<index>`` names, which the MTP freeze regex reads."""

    def __init__(self, count):
        super().__init__()
        self.model = paddle.nn.Layer()
        self.model.layers = paddle.nn.LayerList(
            [paddle.nn.Linear(2, 2) for _ in range(count)]
        )

    def forward(self, value):
        for layer in self.model.layers:
            value = layer(value)
        return value


class FreezeExceptMtpTests(unittest.TestCase):
    """Only the trailing MTP layers keep gradients."""

    def test_base_layers_are_frozen_and_mtp_layers_are_not(self):
        model = _LayeredModel(3)
        wf.freeze_param_except_mtp(
            model,
            SimpleNamespace(num_hidden_layers=2, num_nextn_predict_layers=1),
        )
        frozen = {
            name: param.stop_gradient
            for name, param in model.state_dict().items()
        }
        self.assertTrue(frozen["model.layers.0.weight"])
        self.assertTrue(frozen["model.layers.0.bias"])
        self.assertTrue(frozen["model.layers.1.weight"])
        self.assertFalse(frozen["model.layers.2.weight"])
        self.assertFalse(frozen["model.layers.2.bias"])

    def test_the_depth_falls_back_to_mtp_num_layers(self):
        model = _LayeredModel(3)
        wf.freeze_param_except_mtp(
            model,
            SimpleNamespace(
                num_hidden_layers=1,
                num_nextn_predict_layers=0,
                mtp_num_layers=2,
            ),
        )
        frozen = {
            name: param.stop_gradient
            for name, param in model.state_dict().items()
        }
        self.assertTrue(frozen["model.layers.0.weight"])
        self.assertFalse(frozen["model.layers.1.weight"])
        self.assertFalse(frozen["model.layers.2.weight"])

    def test_without_an_mtp_depth_every_parameter_is_frozen(self):
        model = _LayeredModel(2)
        wf.freeze_param_except_mtp(
            model,
            SimpleNamespace(num_hidden_layers=2, num_nextn_predict_layers=0),
        )
        self.assertTrue(
            all(param.stop_gradient for param in model.parameters())
        )


if __name__ == "__main__":
    unittest.main()


class _RunSftConfig(SimpleNamespace):
    """Provider config double; ``run_sft`` asks for the text sub-config."""

    def get_text_config(self):
        return self


class _StubTrainer:
    """Record the trainer wiring instead of building a real trainer."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.saved = []
        self.trained = []
        self.metrics = []
        self.state_saved = False
        self.optimizer_parameters = None
        _StubTrainer.instances.append(self)

    def set_optimizer_grouped_parameters(self, parameters):
        self.optimizer_parameters = parameters

    def train(self, resume_from_checkpoint=None):
        self.trained.append(resume_from_checkpoint)
        return SimpleNamespace(metrics={"train_runtime": 2.0})

    def save_model(self, **kwargs):
        self.saved.append(kwargs)

    def log_metrics(self, split, metrics):
        self.metrics.append(("log", split, metrics))

    def save_metrics(self, split, metrics):
        self.metrics.append(("save", split, metrics))

    def save_state(self):
        self.state_saved = True


def _sft_model_args(**overrides):
    values = {
        "model_name_or_path": "/models/glm52",
        "tokenizer_name_or_path": "/models/glm52-tokenizer",
        "download_hub": None,
        "copy_custom_file_list": None,
        "use_fast_layer_norm": False,
        "pp_seg_method": "layer:GlmMoeDsaDecoderLayer",
        "_attn_implementation": "eager",
        "lora": False,
        "moe_logging": False,
        "persist_layer_norm": None,
        "mtp_attention_flexible": True,
        "stage": "SFT",
        "continue_training": True,
        "neftune": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _sft_data_args(**overrides):
    values = {
        "max_seq_len": 4,
        "processor_use_fast": False,
        "new_special_tokens_path": None,
        "num_samples_each_epoch": 1,
        "random_shuffle": False,
        "greedy_intokens": False,
        "packing": False,
        "mix_strategy": "concat",
        "encode_one_turn": False,
        "use_template": True,
        "truncate_packing": False,
        "template_backend": "fleet",
        "split_multi_turn": False,
        "dataset_type": "offline",
        "binpacking": False,
        "packing_interval": 1,
        "packed_idx_cache_dir": None,
        "template": "glm5_2",
        "truncation_strategy": "right",
        "skip_warmup": True,
        "warmup_only_rank0": True,
        "make_offline_data": False,
        "eval_with_do_generation": False,
        "input_dir": "/data/glm52",
        "pretokenized_dataset": True,
        "pretokenized_pad_token_id": None,
        "padding_free": False,
        "train_dataset_path": None,
        "train_dataset_prob": None,
        "train_dataset_type": None,
        "eval_dataset_path": None,
        "eval_dataset_prob": None,
        "eval_dataset_type": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _sft_training_args(output_dir, **overrides):
    values = {
        "print_config": lambda *args, **kwargs: None,
        "pre_alloc_memory": 0,
        "device": "gpu",
        "seed": 42,
        "local_rank": -1,
        "world_size": 1,
        "fp16": False,
        "bf16": True,
        "fp16_opt_level": "O2",
        "output_dir": output_dir,
        "do_train": True,
        "do_eval": False,
        "overwrite_output_dir": True,
        "resume_from_checkpoint": None,
        "weight_quantize_algo": None,
        "pad_token_id": None,
        "prediction_loss_only": False,
        "use_expert_parallel": False,
        "enable_auto_parallel": False,
        "freeze_config": "",
        "compute_type": "bf16",
        "dataset_world_size": 1,
        "dataset_rank": 0,
        "dataloader_num_workers": 0,
        "dataset_num_proc": 1,
        "convert_from_hf": False,
        "load_via_cpu": False,
        "load_checkpoint_format": "unified",
        "autotuner_benchmark": False,
        "should_load_dataset": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "global_batch_size": 1,
        "max_steps": 2,
        "decay_steps": 2,
        "num_train_epochs": 1,
        "save_strategy": wf.IntervalStrategy.STEPS,
        "evaluation_strategy": wf.IntervalStrategy.STEPS,
        "logging_strategy": wf.IntervalStrategy.STEPS,
        "train_mtp_only": False,
        # GLM MoE DSA contract inputs.
        "use_accuracy_compatible": True,
        "num_nextn_predict_layers": 0,
        "mtp_num_layers": 0,
        "mtp_loss_scaling_factor": None,
        "fp32_residual_connection": True,
        "moe_token_dispatcher_type": "alltoall",
        "moe_router_bias_update_rate": 0.001,
        "moe_expert_fusion": None,
        "bias_activation_fusion": None,
        "overlap_p2p_comm": None,
        "batch_p2p_comm": None,
        "variable_seq_lengths": None,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "data_parallel_size": 1,
        "sequence_parallel": False,
        "expert_tensor_model_parallel_size": -1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class RunSftWiringTests(unittest.TestCase):
    """``run_sft`` plumbing, exercised with every collaborator stubbed."""

    def setUp(self):
        _patch_env(self)
        _StubTrainer.instances = []
        self.record = SimpleNamespace(
            configs=[],
            seeds=[],
            devices=[],
            flags=[],
            llm_configs=[],
            weights=[],
            special_tokens=[],
            indexed=[],
            templates=[],
        )
        self.model_config = _RunSftConfig(
            model_type="glm_moe_dsa",
            architectures=["GlmMoeDsaForCausalLM"],
            tie_word_embeddings=False,
            quantization_config=SimpleNamespace(
                is_weight_quantize=lambda: False
            ),
            mtp_num_layers=0,
        )
        self.model = SimpleNamespace(is_fleet=False, parameters=lambda: [])
        self.tokenizer = SimpleNamespace(
            pad_token_id=None, eos_token_id=7, chat_template=None
        )
        self.processor = SimpleNamespace(name="processor")
        self.dataset = [[_sequence(length=4)]]
        self.output_dir = str(_tmpdir(self))
        patcher = mock.patch.multiple(
            wf,
            AutoConfig=SimpleNamespace(from_pretrained=self._auto_config),
            QuantizationConfig=SimpleNamespace(from_dict=self._quantization),
            LlmMetaConfig=SimpleNamespace(set_llm_config=self._llm_config),
            AutoModelForCausalLM=SimpleNamespace(
                from_pretrained=self._from_pretrained,
                from_config=self._from_config,
            ),
            AutoTokenizer=SimpleNamespace(from_pretrained=self._tokenizer),
            AutoProcessor=SimpleNamespace(from_pretrained=self._processor),
            add_new_special_tokens=self._special_tokens,
            create_indexed_dataset=self._indexed_dataset,
            get_template_and_fix_tokenizer=self._template,
            set_random_seed=self._seed,
            set_seed=self._seed,
            SFTTrainer=_StubTrainer,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        for name, handler in (
            ("set_device", self._set_device),
            ("set_flags", self._set_flags),
        ):
            device_patcher = mock.patch.object(paddle, name, handler)
            device_patcher.start()
            self.addCleanup(device_patcher.stop)

    # -- stubs ---------------------------------------------------------
    def _auto_config(self, path, **kwargs):
        self.record.configs.append((path, kwargs))
        return self.model_config

    def _quantization(self, payload):
        return SimpleNamespace(is_weight_quantize=lambda: False)

    def _llm_config(self, config, training_args):
        self.record.llm_configs.append((config, training_args))

    def _from_pretrained(self, path, **kwargs):
        self.record.weights.append((path, kwargs))
        return self.model

    def _from_config(self, config, **kwargs):
        self.record.weights.append((None, kwargs))
        return self.model

    def _tokenizer(self, path):
        return self.tokenizer

    def _processor(self, path, use_fast=None):
        return self.processor

    def _special_tokens(self, tokenizer, path):
        self.record.special_tokens.append((tokenizer, path))

    def _indexed_dataset(self, **kwargs):
        prefix = kwargs["data_file_prefix"]
        self.record.indexed.append(prefix)
        if prefix.endswith("eval"):
            raise FileNotFoundError(prefix)
        return self.dataset

    def _template(self, dataset_config):
        self.record.templates.append(dict(dataset_config))
        return SimpleNamespace(name="template")

    def _seed(self, **kwargs):
        self.record.seeds.append(kwargs)

    def _set_device(self, device):
        self.record.devices.append(device)

    def _set_flags(self, flags):
        self.record.flags.append(flags)

    def _run(
        self,
        model_args=None,
        data_args=None,
        training_args=None,
        generating_args=None,
    ):
        return wf.run_sft(
            model_args or _sft_model_args(),
            data_args or _sft_data_args(),
            generating_args or SimpleNamespace(enable_thinking=True),
            training_args or _sft_training_args(self.output_dir),
        )

    # -- tests ---------------------------------------------------------
    def test_an_unknown_attention_implementation_fails_closed(self):
        with self.assertRaises(ValueError) as caught:
            self._run(
                model_args=_sft_model_args(
                    _attn_implementation="does_not_exist"
                ),
                training_args=_sft_training_args(
                    self.output_dir, num_nextn_predict_layers=1
                ),
            )
        self.assertIn("Invalid _attn_implementation", str(caught.exception))
        # The GLM contract and the kernel flag run before the model is built.
        self.assertEqual(self.model_config.use_accuracy_compatible, "megatron")
        self.assertEqual(self.model_config.num_nextn_predict_layers, 1)
        self.assertTrue(self.model_config.mtp_enabled)
        self.assertEqual(
            self.record.flags,
            [{"FLAGS_use_accuracy_compatible_kernel": True}],
        )
        self.assertEqual(self.record.devices, ["gpu"])
        self.assertEqual(self.record.seeds, [{"seed_": 42}, {"seed": 42}])
        self.assertEqual(self.record.configs[0][0], "/models/glm52")
        self.assertEqual(self.record.configs[0][1]["dtype"], "bfloat16")
        self.assertEqual(len(self.record.llm_configs), 1)
        self.assertIs(self.record.llm_configs[0][0], self.model_config)
        # The model is never constructed on this path.
        self.assertEqual(self.record.weights, [])

    def test_accuracy_compatible_off_clears_the_kernel_flag(self):
        with self.assertRaises(ValueError):
            self._run(
                model_args=_sft_model_args(
                    _attn_implementation="does_not_exist"
                ),
                training_args=_sft_training_args(
                    self.output_dir, use_accuracy_compatible=False
                ),
            )
        self.assertFalse(self.model_config.use_accuracy_compatible)
        self.assertEqual(
            self.record.flags,
            [{"FLAGS_use_accuracy_compatible_kernel": False}],
        )

    def test_the_glm5_2_template_receives_the_thinking_switch(self):
        with self.assertRaises(ValueError) as caught:
            self._run(
                data_args=_sft_data_args(template_backend="custom"),
                training_args=_sft_training_args(
                    self.output_dir, num_nextn_predict_layers=1
                ),
                generating_args=SimpleNamespace(enable_thinking=True),
            )
        # The offline pretokenized MTP path fails closed without a pad id.
        self.assertIn(
            "pretokenized_pad_token_id is required", str(caught.exception)
        )
        self.assertEqual(len(self.record.templates), 1)
        dataset_config = self.record.templates[0]
        self.assertEqual(dataset_config["template"], "glm5_2")
        self.assertTrue(dataset_config["enable_thinking"])
        self.assertIs(dataset_config["tokenizer"], self.tokenizer)
        self.assertIs(dataset_config["processor"], self.processor)
        self.assertEqual(dataset_config["dtype"], "bfloat16")
        # The tokenizer went through the CLI loader and the pad fallback.
        self.assertEqual(self.tokenizer.pad_token_id, 7)
        self.assertEqual(self.record.special_tokens, [(self.tokenizer, None)])
        self.assertEqual(self.record.indexed, ["/data/glm52/train"])
        self.assertEqual(self.record.weights[0][0], "/models/glm52")

    def test_other_templates_keep_their_registered_thinking_default(self):
        with self.assertRaises(ValueError):
            self._run(
                data_args=_sft_data_args(
                    template="qwen3_vl", template_backend="custom"
                ),
                training_args=_sft_training_args(
                    self.output_dir, num_nextn_predict_layers=1
                ),
                generating_args=SimpleNamespace(enable_thinking=False),
            )
        self.assertEqual(self.record.templates[0]["template"], "qwen3_vl")
        self.assertNotIn("enable_thinking", self.record.templates[0])

    def test_a_validated_pretokenized_dataset_reaches_the_eval_shard(self):
        with self.assertRaises(FileNotFoundError):
            self._run(
                data_args=_sft_data_args(pretokenized_pad_token_id=0),
                training_args=_sft_training_args(
                    self.output_dir, num_nextn_predict_layers=1, do_eval=True
                ),
            )
        self.assertEqual(
            self.record.indexed,
            ["/data/glm52/train", "/data/glm52/eval"],
        )

    def test_a_malformed_pretokenized_row_stops_the_run(self):
        self.dataset = [[_sequence(length=3)]]
        with self.assertRaises(ValueError) as caught:
            self._run(
                data_args=_sft_data_args(pretokenized_pad_token_id=0),
                training_args=_sft_training_args(
                    self.output_dir, num_nextn_predict_layers=1
                ),
            )
        self.assertIn(
            "pretokenized token_ids length 3 != 4", str(caught.exception)
        )
        self.assertEqual(_StubTrainer.instances, [])

    def test_an_autotuner_benchmark_skips_the_final_export(self):
        self._run(
            data_args=_sft_data_args(pretokenized_pad_token_id=0),
            training_args=_sft_training_args(
                self.output_dir,
                num_nextn_predict_layers=1,
                autotuner_benchmark=True,
            ),
        )
        trainer = _StubTrainer.instances[0]
        self.assertEqual(trainer.trained, [None])
        self.assertEqual(trainer.saved, [])
        self.assertFalse(trainer.state_saved)


if __name__ == "__main__":
    unittest.main()
