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
"""Single card tests for ``paddlefleet.trainer.checkpoint_export``.

The helpers under test are pure JSON / filesystem code: no distributed
group, no device and no downloaded checkpoint. Tiny safetensors
containers are built inside a temporary directory, so the header
parser, the duplicate-name guard and the HF cadence path resolver all
run against real bytes instead of mocks.
"""

import json
import os
import struct
import tempfile
import unittest

from paddlefleet.trainer.checkpoint_export import (
    HF_CHECKPOINT_PREFIX,
    assert_unique_safetensors_names,
    collect_safetensors_names,
    hf_export_provenance,
    iter_safetensors_files,
    resolve_hf_checkpoint_dir,
    write_tiny_safetensors,
)

ONE_F32 = ("F32", [1], b"\x00\x00\x00\x00")


def _write_raw_container(path, header, payload=b""):
    """Write a safetensors container from an arbitrary header object."""
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((-len(raw)) % 8)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as stream:
        stream.write(struct.pack("<Q", len(raw)))
        stream.write(raw)
        stream.write(payload)
    return raw


def _read_bytes(path):
    with open(path, "rb") as stream:
        return stream.read()


class _OpaqueDtype:
    """Stands in for a framework dtype: not JSON serializable by itself."""

    def __str__(self):
        return "paddle.bfloat16"


class _ProviderConfig:
    """Provider-shaped config that exposes only a subset of the fields.

    The snapshot contract is "record what the provider actually has and
    name what it does not", so the missing fields matter as much as the
    present ones. Values cover every branch of the serializer: scalars,
    ``None``, tuple, nested dict with a non-string key, and an opaque
    object.
    """

    def __init__(self):
        self.num_hidden_layers = 2
        self.num_attention_heads = 8
        self.num_key_value_heads = 2
        self.hidden_size = 64
        self.multi_latent_attention = True
        self.mtp_loss_scaling_factor = 0.1
        self.params_dtype = "bfloat16"
        self.dtype = _OpaqueDtype()
        self.q_lora_rank = None
        self.dsa_indexer_types = ("full", "shared")
        self.indexer_types = {0: "full", 1: ["shared", "full"]}
        self.dsa_index_topk = 2048


class HfExportProvenanceTests(unittest.TestCase):
    def test_schema_stage_and_provider_identity(self):
        snapshot = hf_export_provenance(
            _ProviderConfig(), {"aoa_statements": []}, "out", 7
        )
        self.assertEqual(snapshot["schema"], "paddlefleet-hf-export/v1")
        self.assertEqual(snapshot["stage"], "prepared")
        self.assertEqual(
            snapshot["provider_class"],
            f"{_ProviderConfig.__module__}._ProviderConfig",
        )
        self.assertEqual(snapshot["global_step"], 7)

    def test_output_dir_is_absolute_and_step_defaults_to_none(self):
        snapshot = hf_export_provenance(
            _ProviderConfig(), None, os.path.join("results", "hf")
        )
        self.assertTrue(os.path.isabs(snapshot["output_dir"]))
        self.assertEqual(
            snapshot["output_dir"],
            os.path.abspath(os.path.join("results", "hf")),
        )
        self.assertIsNone(snapshot["global_step"])
        self.assertIsNone(snapshot["aoa_config"])

    def test_present_and_missing_fields_partition_the_snapshot(self):
        config = _ProviderConfig()
        snapshot = hf_export_provenance(config, None, "out")
        values = snapshot["provider_config"]
        missing = snapshot["missing_provider_fields"]

        self.assertIn("hidden_size", values)
        self.assertIn("num_hidden_layers", values)
        self.assertIn("using_sonic_moe", missing)
        self.assertIn("n_routed_experts", missing)
        self.assertEqual(set(values) & set(missing), set())
        self.assertEqual(len(missing), len(set(missing)))
        for field in values:
            self.assertTrue(hasattr(config, field), field)
        for field in missing:
            self.assertFalse(hasattr(config, field), field)

    def test_values_are_json_safe_without_losing_scalars(self):
        snapshot = hf_export_provenance(
            _ProviderConfig(), {"aoa_statements": ["a -> b"]}, "out", 3
        )
        values = snapshot["provider_config"]

        self.assertIs(values["multi_latent_attention"], True)
        self.assertIsNone(values["q_lora_rank"])
        self.assertEqual(values["mtp_loss_scaling_factor"], 0.1)
        self.assertEqual(values["params_dtype"], "bfloat16")
        self.assertEqual(values["hidden_size"], 64)
        # Opaque objects degrade to str, tuples become lists and dict
        # keys are coerced to str so the snapshot stays JSON round-trippable.
        self.assertEqual(values["dtype"], "paddle.bfloat16")
        self.assertEqual(values["dsa_indexer_types"], ["full", "shared"])
        self.assertEqual(
            values["indexer_types"],
            {"0": "full", "1": ["shared", "full"]},
        )
        self.assertEqual(snapshot["aoa_config"], {"aoa_statements": ["a -> b"]})
        self.assertEqual(json.loads(json.dumps(snapshot)), snapshot)

    def test_snapshot_never_reports_completion(self):
        # The snapshot describes export preparation only; a caller must
        # not be able to read success out of it.
        snapshot = hf_export_provenance(_ProviderConfig(), None, "out", 1)
        self.assertNotIn("status", snapshot)
        self.assertNotIn("weights_loaded", snapshot)
        self.assertEqual(
            sorted(snapshot),
            [
                "aoa_config",
                "global_step",
                "missing_provider_fields",
                "output_dir",
                "provider_class",
                "provider_config",
                "schema",
                "stage",
            ],
        )


class ResolveHfCheckpointDirTests(unittest.TestCase):
    def test_default_nests_cadence_under_output_dir(self):
        self.assertEqual(HF_CHECKPOINT_PREFIX, "hf_checkpoint")
        path = resolve_hf_checkpoint_dir(os.path.join("run", "ckpt"), 5)
        self.assertEqual(path, os.path.join("run", "ckpt", "hf_checkpoint-5"))

    def test_opt_in_override_moves_the_cadence_root(self):
        path = resolve_hf_checkpoint_dir(
            os.path.join("run", "ckpt"),
            5,
            save_hf_output_dir=os.path.join("run", "oracle"),
        )
        self.assertEqual(path, os.path.join("run", "oracle", "hf_checkpoint-5"))
        self.assertFalse(path.startswith(os.path.join("run", "ckpt")))

    def test_empty_override_falls_back_to_output_dir(self):
        # Falsy override must not produce a relative "hf_checkpoint-1".
        path = resolve_hf_checkpoint_dir("run", 1, save_hf_output_dir="")
        self.assertEqual(path, os.path.join("run", "hf_checkpoint-1"))

    def test_prefix_is_overridable_and_step_is_coerced_to_int(self):
        self.assertEqual(
            resolve_hf_checkpoint_dir("run", "12", prefix="snapshot"),
            os.path.join("run", "snapshot-12"),
        )
        self.assertEqual(
            resolve_hf_checkpoint_dir("run", 12.9),
            os.path.join("run", "hf_checkpoint-12"),
        )


class WriteTinySafetensorsTests(unittest.TestCase):
    def test_container_layout_is_parsable_and_offsets_are_contiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "model.safetensors")
            write_tiny_safetensors(
                path,
                {
                    "a.weight": ("F32", [1], b"\x01\x02\x03\x04"),
                    "b.weight": ("F32", [2], b"\x05" * 8),
                },
            )
            blob = _read_bytes(path)

        header_length = struct.unpack("<Q", blob[:8])[0]
        header = json.loads(blob[8 : 8 + header_length])
        payload = blob[8 + header_length :]

        self.assertEqual(header_length % 8, 0)
        self.assertEqual(sorted(header), ["a.weight", "b.weight"])
        self.assertEqual(header["a.weight"]["dtype"], "F32")
        self.assertEqual(header["a.weight"]["shape"], [1])
        self.assertEqual(header["a.weight"]["data_offsets"], [0, 4])
        self.assertEqual(header["b.weight"]["shape"], [2])
        self.assertEqual(header["b.weight"]["data_offsets"], [4, 12])
        self.assertEqual(payload, b"\x01\x02\x03\x04" + b"\x05" * 8)

    def test_writer_creates_missing_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a", "b", "model.safetensors")
            write_tiny_safetensors(path, {"w": ONE_F32})
            self.assertTrue(os.path.isfile(path))
            self.assertEqual(collect_safetensors_names(tmp), ["w"])


class CollectSafetensorsNamesTests(unittest.TestCase):
    def test_walk_is_recursive_and_ignores_other_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_tiny_safetensors(
                os.path.join(tmp, "model-00001.safetensors"), {"top": ONE_F32}
            )
            write_tiny_safetensors(
                os.path.join(tmp, "shard", "model-00002.safetensors"),
                {"nested": ONE_F32},
            )
            with open(os.path.join(tmp, "config.json"), "w") as stream:
                stream.write("{}")
            with open(
                os.path.join(tmp, "model.safetensors.index.json"), "w"
            ) as stream:
                stream.write("{}")

            files = sorted(
                os.path.relpath(path, tmp)
                for path in iter_safetensors_files(tmp)
            )
            self.assertEqual(
                files,
                [
                    "model-00001.safetensors",
                    os.path.join("shard", "model-00002.safetensors"),
                ],
            )
            self.assertEqual(
                sorted(collect_safetensors_names(tmp)), ["nested", "top"]
            )

    def test_metadata_entry_is_not_a_tensor_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_raw_container(
                os.path.join(tmp, "model.safetensors"),
                {
                    "__metadata__": {"format": "pt"},
                    "w": {
                        "dtype": "F32",
                        "shape": [1],
                        "data_offsets": [0, 4],
                    },
                },
                b"\x00\x00\x00\x00",
            )
            self.assertEqual(collect_safetensors_names(tmp), ["w"])

    def test_truncated_length_prefix_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "short.safetensors")
            with open(path, "wb") as stream:
                stream.write(b"\x00\x01\x02")
            with self.assertRaisesRegex(
                ValueError, "truncated safetensors header"
            ):
                collect_safetensors_names(tmp)

    def test_header_shorter_than_declared_length_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cut.safetensors")
            body = b'{"w": {}}'
            with open(path, "wb") as stream:
                stream.write(struct.pack("<Q", len(body) + 64))
                stream.write(body)
            with self.assertRaisesRegex(
                ValueError, "truncated safetensors header"
            ):
                collect_safetensors_names(tmp)

    def test_non_object_header_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_raw_container(os.path.join(tmp, "list.safetensors"), ["w"])
            with self.assertRaisesRegex(
                ValueError, "safetensors header is not an object"
            ):
                collect_safetensors_names(tmp)


class AssertUniqueSafetensorsNamesTests(unittest.TestCase):
    def test_disjoint_shards_are_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_tiny_safetensors(
                os.path.join(tmp, "model-00001-of-00002.safetensors"),
                {"model.layers.0.mlp.up_proj.weight": ONE_F32},
            )
            write_tiny_safetensors(
                os.path.join(tmp, "model-00002-of-00002.safetensors"),
                {"model.layers.0.mlp.down_proj.weight": ONE_F32},
            )
            assert_unique_safetensors_names(tmp)

    def test_nested_cadence_copy_breaks_the_oracle_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = {"model.layers.3.mlp.gate.weight": ONE_F32}
            write_tiny_safetensors(
                os.path.join(tmp, "model-00001-of-00001.safetensors"), payload
            )
            write_tiny_safetensors(
                os.path.join(
                    tmp,
                    f"{HF_CHECKPOINT_PREFIX}-5",
                    "model-00001-of-00001.safetensors",
                ),
                payload,
            )
            with self.assertRaisesRegex(
                ValueError, "invalid or duplicate tensor name"
            ):
                assert_unique_safetensors_names(tmp)

    def test_empty_tensor_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_tiny_safetensors(
                os.path.join(tmp, "model.safetensors"), {"": ONE_F32}
            )
            with self.assertRaisesRegex(
                ValueError, "invalid or duplicate tensor name"
            ):
                assert_unique_safetensors_names(tmp)


if __name__ == "__main__":
    unittest.main()
