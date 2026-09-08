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

"""Run the production KV split/alignment block with explicit TP shard adapters."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import paddle


class TestIEEEMLAKVLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (
            Path(__file__).resolve().parents[3]
            / "src/paddlefleet/transformer/multi_latent_attention.py"
        )
        tree = ast.parse(source.read_text())
        layer = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "MLASelfAttention"
        )
        method = next(
            n
            for n in layer.body
            if getattr(n, "name", None) == "get_query_key_value_tensors"
        )
        blocks = [
            n
            for n in method.body
            if isinstance(n, ast.If)
            and ast.unparse(n.test).startswith("kv_combined.size(-1)")
        ]
        assert len(blocks) == 1
        cls.block = compile(
            ast.Module(body=blocks, type_ignores=[]), str(source), "exec"
        )
        constructor = next(
            n for n in layer.body if getattr(n, "name", None) == "__init__"
        )
        controls = [
            n
            for n in ast.walk(constructor)
            if isinstance(n, ast.If)
            and "output_size_per_partition" in ast.unparse(n.test)
        ]
        assert len(controls) == 1
        cls.up_layout = compile(
            ast.Module(body=controls, type_ignores=[]), str(source), "exec"
        )

    def equal(self, a, b):
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(
            a.cast("float32").numpy().tobytes(),
            b.cast("float32").numpy().tobytes(),
        )

    def run_layout(
        self, *, feature_sharded=False, ieee=True, tp=2, sp=True, mqa=False
    ):
        paddle.seed(516)
        full = paddle.randn([6, 1, 12]).cast("bfloat16")
        local = full[:, :, :6] if feature_sharded else full[:3]
        calls = []
        group = SimpleNamespace(nranks=tp)

        def gather_features(x):
            calls.append("gather_features")
            return paddle.concat([x, full[:, :, 6:]], axis=-1)

        def gather_positions(x, group):
            calls.append("gather_positions")
            return paddle.concat([x, full[3:, :, 8:]], axis=0)

        def scatter_sequence(x, group):
            calls.append("scatter_sequence")
            return x[:3]

        ns = {
            "paddle": paddle,
            "kv_combined": local,
            "self": SimpleNamespace(
                kv_lora_rank=8,
                qk_rope_head_dim=4,
                config=SimpleNamespace(sequence_parallel=sp),
                pg_collection=SimpleNamespace(tp=group),
                mqa_latent=mqa,
            ),
            "ieee_kernel_enabled": lambda: ieee,
            "get_pg_size": lambda group: group.nranks,
            "gather_from_tensor_model_parallel_region": gather_features,
            "gather_from_sequence_parallel_region": gather_positions,
            "scatter_to_sequence_parallel_region": scatter_sequence,
        }
        exec(self.block, ns)
        return full, ns["kv_compressed"], ns["k_pos_emb"], calls

    def test_kv_up_gathers_only_sequence_local_down_output(self):
        for ieee, partition, expected in [
            (True, None, True),
            (True, 6, False),
            (True, 12, True),
            (False, 6, True),
        ]:
            down = SimpleNamespace()
            if partition is not None:
                down.output_size_per_partition = partition
            up = SimpleNamespace(
                sequence_parallel=True,
                allreduce_dgrad=False,
                world_size=2,
                disable_grad_reduce=False,
            )
            layer = SimpleNamespace(
                config=SimpleNamespace(
                    use_accuracy_compatible=True, sequence_parallel=True
                ),
                kv_a_proj_with_mqa=down,
                kv_b_proj=up,
                qk_rope_head_dim=4,
            )
            exec(
                self.up_layout,
                {
                    "ieee_kernel_enabled": lambda: ieee,
                    "self": layer,
                    "kv_lora_rank": 8,
                },
            )
            self.assertEqual(up.sequence_parallel, expected)
            self.assertEqual(up.allreduce_dgrad, not expected)

    def test_replicated_down_projection_keeps_kv_local_and_rope_full(self):
        full, kv, rope, calls = self.run_layout()
        self.equal(kv, full[:3, :, :8])
        self.equal(rope, full[:, :, 8:])
        self.assertEqual(calls, ["gather_positions"])

    def test_partitioned_down_projection_keeps_upstream_full_sequence(self):
        full, kv, rope, calls = self.run_layout(feature_sharded=True)
        self.equal(kv, full[:, :, :8])
        self.equal(rope, full[:, :, 8:])
        self.assertEqual(calls, ["gather_features"])

    def test_default_tp1_nonsp_and_new_mqa_modes_keep_local_positions(self):
        for args in [{"ieee": False}, {"tp": 1}, {"sp": False}, {"mqa": True}]:
            with self.subTest(**args):
                full, kv, rope, calls = self.run_layout(**args)
                self.equal(kv, full[:3, :, :8])
                self.equal(rope, full[:3, :, 8:])
                self.assertEqual(calls, [])


if __name__ == "__main__":
    paddle.set_device("gpu:0")
    unittest.main()
