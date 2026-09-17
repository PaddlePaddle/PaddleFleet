# GLM52 EP2 / TP1 / PP2 accuracy case

Runs PaddleFleet and Megatron independently for 100 steps on four GPUs with
EP2, TP1, PP2 and sequence parallelism disabled. The model is an official GLM-5.2
subset: 3 dense layers, 1 MoE layer, 1 MTP layer and 16 experts.

## Prerequisites

Use the consolidated GLM-5.2 implementation from PaddleFleet #1961.
This case creates private `venv/paddle` and `venv/torch` environments below
`GLM52_EP2_TP1_PP2`. It uses the current CI PaddleFleet and FleetOps wheel paths,
Paddle `3.4.0.post20260907` with CUDA 12.9, Torch `2.12.1+cu129`, mcore-bridge
`1.6.1`, and Transformers `5.12.1` on the Torch side. The reference Megatron
`40456374` and Swift `baa4ba73` wheels are verified against `reference_wheels.txt`.
The bridge version retains the CPU RoPE frequency initialization of the aligned
baseline. The shared environments, installer and other model settings remain
unchanged. GLM52 runs before legacy environment setup so an unrelated setup
failure does not prevent its own precision check.

CI supplies `PADDLEFLEET_WHEEL_PATH` and `PADDLEFLEET_OPS_WHEEL_PATH` from the
current build. Both are required when creating the private Paddle environment.
For preconfigured environments, set `GLM52_VENV_ROOT` to the parent of `paddle/`
and `torch/`. `GLM52_TORCH_VENV` can select a separate Torch path. Explicit
`GLM52_VENV_ROOT` skips dependency installation on both sides.

The case builds DeepEP `17cfb817bccec3a9c247013360cc550c2bac441e` and
fast-hadamard-transform `f134af63deb2df17e1171a9ec1ea4a7d8604d5ca` against its
own Torch installation, without reusing binary build caches. `deep_ep_cuda12.patch` retains the full NVSHMEM interface and omits only the
CUDA13-specific optional NVLink-utilization scheduling hint on CUDA12. Bridge 1.6.1 requires TE import dependencies, so TE 2.17.1 is installed,
while the accuracy configuration continues to disable TE computation.

The build uses CUDA 12.9 from the image when available, otherwise checksum-pinned
NVIDIA CUDA 12.9.1 components under the case environment. Compiler variables stay
in the build subprocess; no system CUDA or driver installation occurs. Both
environment setup scripts run `uv pip check`.

Paddle declares cuBLAS `12.9.0.13`, while the aligned reference uses
`12.9.1.4`. The Paddle installer first checks the unmodified dependency set,
then explicitly installs `nvidia-cublas-cu12==12.9.1.4` in its private environment.
Paddle preloads this package directly, so changing `LD_LIBRARY_PATH` alone does
not select the required runtime. The final dependency check retains and reports
this one known exact-pin conflict in `venv/paddle/glm52-dependency-check.log`;
any other incompatibility fails setup. Package metadata is not rewritten.
This override is limited to this accuracy case and is not a general Paddle
compatibility claim.

The default model cache is
`/home/.cache/PaddleFormers/GLM-5.2-BF16-minimal`.
Use the uploaded [PaddleFormers/GLM-5.2-BF16-minimal model](https://git.aistudio.baidu.com/PaddleFormers/GLM-5.2-BF16-minimal.git),
revision `685a4b891d004a64bd2f1fef320deeffe162af02`. Populate the cache using
AI Studio's Git LFS download procedure, including config, both safetensors
shards, their index, tokenizer files, generation config, license and manifests.
A single `model.safetensors` is also supported; the native loaders validate the
selected checkpoint and its indexed shards.

The source is `zai-org/GLM-5.2@b4734de4facf877f85769a911abafc5283eab3d9`.
Both uploaded shards preserve all 187 tensors from the original extracted
checkpoint (`53ad1565fe8173420db4b3ef53db3bddef0afda4b43f859aacf16826700916c2`)
without changing names, shapes, dtypes or bytes. `upload_manifest.json` records
the individual file checksums. CI cache provisioning and matching companion
builds are still required; adding this case does not provision them.

Data reuses the existing MinimaxV2.5_EP2 cache:
`/home/.cache/PaddleFormers/MiniMax-V2.5-bf16_2EP/alignment_paddle.jsonl` and
`alignment_torch.jsonl`. Local overrides are `GLM52_MODEL_DIR`,
`GLM52_TOKENIZER_DIR`, and `GLM52_DATA_DIR`.

## Run

```bash
bash scripts/alignment_model_accuracy/GLM52_EP2_TP1_PP2/run_alignment.sh
```

The shared runner invokes this case before legacy environment setup. Existing case
scripts, loss comparison and cleanup remain unchanged. GLM52 retains its own
logs, raw loss, environment records and checkpoints under
`GLM52_EP2_TP1_PP2/results/<ALIGNMENT_RUN_TAG>/`, including failed runs.
The two framework scripts use the same tag when launched by run_alignment.sh.
The case logs SHA256 fingerprints of its model, tokenizer and data files before
training, and prints native environment, input and loss receipts on exit. These
records remain in CI logs after container cleanup. Reporting does not change
the training or comparison exit status.

The case-local comparator requires complete native artifacts with consecutive
steps 1–100, finite values and bitwise identical main training loss. Missing or
truncated results fail. This does not establish equality of all intermediate
tensors or MTP loss. Local success does not substitute for a run on CI hardware.

The Paddle YAML sets `hybrid_parallel_expert_grad_scale: 1.0`: EP2/TP1 deferred
token normalization must not apply the automatic TP/EP scale of 0.5 first.

Both YAML files select numerical alignment with `use_accuracy_compatible: true`.
On the Torch side this selects reproducible gradient clipping when `clip_grad > 0`.
The Torch YAML also explicitly enables the existing `norm_accuracy_compatible`,
`router_accuracy_compatible` and `dsa_accuracy_compatible` options in
`megatron_extra_kwargs` for GLM52 reference numerics. These options retain their
existing names and defaults in Megatron-Core.
PaddleFleet applies the corresponding Paddle backend flag before constructing
the GLM model. Model layers receive the mode through configuration or explicit
function arguments; no `MODEL_REPRO_IEEE_KERNEL` or FP32 accumulator override is
required. Both sides set `bias_activation_fusion: false` in YAML. Pipeline
communication settings also come from the parsed Paddle YAML.
