# GLM52 EP2 / TP1 / PP2 accuracy case

Runs PaddleFormers and Megatron independently for 100 steps on four GPUs with
EP2, TP1, PP2 and sequence parallelism disabled. The model is an official GLM-5.2
subset: 3 dense layers, 1 MoE layer, 1 MTP layer and 16 experts.

## Prerequisites

Use the GLM-5.2 implementations from PaddleFleet #1961, PaddleFormers #4956,
PFCCLab/Megatron-LM #4 and PFCCLab/ms-swift #3. Before these changes reach the
standard wheels, prepare matching environments explicitly and set
`GLM52_VENV_ROOT` to their parent directory (`paddle/` and `torch/`). This case
does not change the shared dependency installer or other models' environments.

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

The shared runner invokes this case after its existing cases. Existing case
scripts, loss comparison and cleanup remain unchanged. GLM52 retains its own
logs, raw loss, environment records and checkpoints under
`GLM52_EP2_TP1_PP2/results/<ALIGNMENT_RUN_TAG>/`, including failed runs.
The two framework scripts use the same tag when launched by run_alignment.sh.

The case-local comparator requires complete native artifacts with consecutive
steps 1–100, finite values and bitwise identical main training loss. Missing or
truncated results fail. This does not establish equality of all intermediate
tensors or MTP loss. Local success does not substitute for a run on CI hardware.

The Paddle YAML sets `hybrid_parallel_expert_grad_scale: 1.0`: EP2/TP1 deferred
token normalization must not apply the automatic TP/EP scale of 0.5 first.

Both YAML files select numerical alignment with `use_accuracy_compatible: true`.
PaddleFormers applies the corresponding Paddle backend flag before constructing
the GLM model. Model layers receive the mode through configuration or explicit
function arguments; no `MODEL_REPRO_IEEE_KERNEL` or FP32 accumulator override is
required. Both sides set `bias_activation_fusion: false` in YAML. Pipeline
communication settings also come from the parsed Paddle YAML.
