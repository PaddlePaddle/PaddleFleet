import copy
import datetime
import json
import logging
import os
import random
import re
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
from arguments import parse_arguments
from data import (
    TestItemDataset,
    load_data,
)
from model_utils import AnthropicModel, OpenAIModel, TgiVllmModel, load_LLM
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_test(args, model, dataset, test_file, demo_file):
    logger.info(
        f"running test on {dataset} with test {test_file} and demo {demo_file}"
    )
    # dataset specific changes tag
    tag = args.tag
    if dataset == "popqa":
        tag += f"_pop{args.popularity_threshold}"

    test_name = os.path.splitext(os.path.basename(test_file))[0]
    output_filename = f"{dataset}_{tag}_{test_name}_in{args.input_max_length}_size{args.max_test_samples}_shots{args.shots}_samp{args.do_sample}max{args.generation_max_length}min{args.generation_min_length}t{args.temperature}p{args.top_p}_chat{args.use_chat_template}_{args.seed}"
    output_path = os.path.join(args.output_dir, f"{output_filename}.json")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    should_skip = None  # placeholder
    if (
        rank == 0
        and os.path.exists(output_path)
        and not args.overwrite
        and not args.debug
    ):
        should_skip = True
        # if os.path.exists(output_path):
        #    with open(output_path, 'r') as f:
        #        tmp = json.load(f)
        # if 'ttft_ms' in tmp['averaged_metrics']:
        #    should_skip = True

    # 让 rank 0 的 should_skip 传播给所有进程
    should_skip = [should_skip]
    dist.broadcast_object_list(should_skip, src=0)
    should_skip = should_skip[0]

    if should_skip:
        logger.info(f"{output_path} already exists, skipping...")
        return output_path  # 所有进程都 return

    random.seed(args.seed)
    data = load_data(args, dataset, test_file, demo_file)
    logger.info(f"loaded {len(data['data'])} samples from {dataset}")

    # 做数据并行切分
    data["data"] = data["data"].shard(num_shards=world_size, index=rank)
    logger.info(
        f"[Rank {rank}] shard data {len(data['data'])} samples from {dataset}"
    )

    dataloader = DataLoader(
        TestItemDataset(data, model, model.tokenizer),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=args.num_workers if not args.debug else 0,
    )

    # we first prepare all inputs and then run the evaluation in batch
    # the dataloader is a bit of an overkill here, but it makes it easier to switch back to iterative instead of batch eval
    metrics = defaultdict(list)
    all_inputs = []
    all_input_texts = []
    for idx, inputs in enumerate(
        tqdm(dataloader, desc=f"[Rank {rank}] Preparing inputs")
    ):
        inputs, input_text = inputs[0]
        if args.count_tokens:
            # count_tokens is only available for models that tokenizes the input
            metrics["input_len"].append(inputs.input_ids.shape[1])
            continue
        all_inputs.append(inputs)
        all_input_texts.append(input_text)

    # HY: for the thinking mode, we add additional 32k tokens to allow models to generate thinking process
    if args.thinking:
        args.generation_max_length += 32768
        args.input_max_length += 32768
        model.max_length = args.input_max_length
        model.generation_max_length = args.generation_max_length
        args.stop_new_line = False
        logger.info(
            "thinking mode, adding 32k tokens to generation and input max length, also disabling stop_new_line"
        )

    logger.info("Running generation...")
    start_time = time.time()
    # generate all outputs
    if isinstance(model, (OpenAIModel, AnthropicModel)) and (
        not isinstance(model, TgiVllmModel)
    ):
        # using the batch API makes it cheaper and faster
        logger.info(
            "Using the OpenAI/Anthropic batch API by default, if you want to use the iterative API, please change the code"
        )
        all_outputs = model.generate_batch(
            all_inputs,
            batch_file=output_path + ".batch",
            desc=f"[Rank {rank}] {dataset}",
        )
    else:
        all_outputs = model.generate_batch(
            all_inputs,
            desc=f"[Rank {rank}] {dataset}",
            record_ttft_ms=args.record_ttft_ms,
            record_e2e_ms=args.record_e2e_ms,
            record_attn_ms=args.record_attn_ms,
            method=args.method,
        )
    end_time = time.time()

    # then we do all the postprocessing + evaluation
    results = []
    for idx, output in enumerate(all_outputs):
        test_item = data["data"][idx]
        input_text = all_input_texts[idx]

        if output is None:
            logger.info(
                f"skipping example {idx + 1} because the model returned None"
            )
            continue

        # If we do not use the chat template, then we are doing completion, and for the sake of parsing, we want to prepend the system prompt to the input.
        # For example, since we are autocompleting "Answer:"" in the input, then we should prepend the system prompt to the output as well.
        # This requires some coordination from the dataset preprocessing
        if not args.use_chat_template:
            prepend_text = data["system_template"].format(**test_item)
            output["output"] = prepend_text + output["output"]

        if args.thinking:
            matches = re.search(
                r"(.*</think>)(.*)", output["output"], flags=re.DOTALL
            )
            if matches:
                output["output"] = matches.group(2).strip()
                output["thoughts"] = matches.group(1).strip()

        mets, others = data["post_process"](output, test_item)
        output.update({**others, **mets})
        for k, v in mets.items():
            metrics[k].append(v)

        metrics["input_len"].append(output["input_len"])
        metrics["output_len"].append(output["output_len"])

        if args.record_ttft_ms:
            metrics["ttft_ms"].append(output["ttft_ms"])
        if args.record_e2e_ms:
            metrics["e2e_ms"].append(output["e2e_ms"])
        if args.record_attn_ms:
            metrics["attn_ms"].append(output["attn_ms"])
            metrics["estimate_func_ms"].append(output["estimate_func_ms"])
        if output.get("sparse_ratio") is not None:
            metrics["sparse_ratios"].append(float(output["sparse_ratio"]))

        result = {**test_item, **output}
        result.pop("context", None)
        result.pop("input_ids", None)
        if input_text is None:
            input_text = result["input_text"]
        results.append(result)

        # print out some examples, we also limit how much we print out since it can get really long
        # if idx < 5 or args.debug:
        #    logger.info(f"Example {idx+1}: ")
        #    logger.info(f"Decoder inputs:\n{input_text}\n")

        #    logger.info(f"Input length: {output['input_len']}")
        #    # currently we hardcode somethings to print out, but you may change these to print out other things
        #    logger.info(f"Question: {test_item['question'] if 'question' in test_item else ''}")
        #    logger.info(f"Answer: {test_item['answer'] if 'answer' in test_item else ''}")
        #    logger.info(f"Output: {output['output']}")
        #    logger.info(f"Parsed output: {output['parsed_output']}")
        #    logger.info(f"Metrics: {mets}")

        if not getattr(model, "uses_paddle_worker", False):
            for name, module in model.model.named_modules():
                if name.split(".")[-1] == "self_attn":
                    if (
                        hasattr(module, "sparse_ratio")
                        and module.sparse_ratio is not None
                    ):
                        if hasattr(module.sparse_ratio, "item"):  # torch.Tensor
                            metrics["sparse_ratios"].append(
                                module.sparse_ratio.item()
                            )
                        else:
                            metrics["sparse_ratios"].append(
                                float(module.sparse_ratio)
                            )
                        module.sparse_ratio = None

        if args.debug:
            import pdb  # noqa: T100

            pdb.set_trace()  # noqa: T100

    if args.record_ttft_ms and len(metrics["ttft_ms"]) > 2:
        metrics["ttft_ms"] = metrics["ttft_ms"][2:]
    if args.record_e2e_ms and len(metrics["e2e_ms"]) > 2:
        metrics["e2e_ms"] = metrics["e2e_ms"][2:]
    if args.record_attn_ms and len(metrics["attn_ms"]) > 2:
        metrics["attn_ms"] = metrics["attn_ms"][2:]
        metrics["estimate_func_ms"] = metrics["estimate_func_ms"][2:]

    if "alce" in dataset:
        local_data = copy.deepcopy(results)

    # 数据并行结果汇总
    gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, results)
    results = []
    for res in gathered_results:
        results.extend(res)

    gathered_metrics = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_metrics, metrics)
    metrics = defaultdict(list)
    for mets in gathered_metrics:
        for k, v in mets.items():
            metrics[k].extend(v)

    mem_usage = None
    if getattr(model, "uses_paddle_worker", False):
        local_mem_usage = getattr(model, "_max_memory_usage", None)
        gathered_mem_usage = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_mem_usage, local_mem_usage)
        valid_mem_usage = [
            item for item in gathered_mem_usage if item is not None
        ]
        if valid_mem_usage:
            mem_usage = sum(valid_mem_usage)
        if mem_usage is not None:
            logger.info(
                f"Paddle worker memory usage: {mem_usage / 1000**3:.02f} GB"
            )
    elif not args.no_cuda and torch.cuda.is_available():
        mem_usage = sum(
            [
                torch.cuda.max_memory_allocated(i)
                for i in range(torch.cuda.device_count())
            ]
        )
        logger.info(f"Memory usage: {mem_usage / 1000**3:.02f} GB")
    logger.info(f"Total time: {end_time - start_time:.02f} s")
    logger.info(
        f"Throughput: {len(results) / (end_time - start_time):.02f} samples/s"
    )

    if args.count_tokens:
        logger.info(
            f"----{dataset}----\nAverage input length: {np.mean(metrics['input_len']):.02f}, std input length: {np.std(metrics['input_len']):.02f}, max input length: {max(metrics['input_len'])}, min input length: {min(metrics['input_len'])}\n----returning----"
        )
        return output_path

    if len(results) == 0:
        logger.error(
            "No results to evaluate, something went wrong, returning..."
        )
        return output_path

    if "alce" in dataset:
        # import nltk
        # logger.info(f"[Rank {rank}] download punkt_tab...")
        # nltk.download('punkt_tab')
        logger.info(f"[Rank {rank}] running eval_alce.py...")

        from eval_alce import (
            compute_autoais,
            compute_claims,
            compute_len,
            compute_qa,
            compute_qampari_f1,
            compute_rouge,
            compute_str_em,
        )
        from utils import remove_citations

        data = copy.deepcopy(results)
        if "nocite" not in dataset:
            args.alce_citations = True
        if "qampari" in output_path:
            args.alce_no_rouge = True
            args.alce_qa = False
            args.alce_mauve = False
            args.alce_decontext = False
            qampari = True
        else:
            qampari = False

        if getattr(model, "uses_paddle_worker", False) and (
            args.alce_qa or args.alce_citations or args.alce_claims_nli
        ):
            logger.info(
                f"[Rank {rank}] releasing Paddle worker cache before torch-based ALCE evaluation"
            )
            try:
                model.release_cache()
            except Exception as exc:
                logger.warning(
                    f"[Rank {rank}] failed to release Paddle worker cache: {exc}"
                )

        # HY: If you want to run the full ALCE evaluation, you should uncomment the following lines
        # In HELMET, we don't use the MAUVE scores.
        # if "asqa" in dataset:
        #     args.alce_mauve = True
        # elif "eli5" in dataset:
        #     args.alce_mauve = True
        #     args.alce_claims_nli = True

        # Truncate by newline and remove on the fly search result
        # logger.warning("We remove all the pre/appended space/newlines and we truncate the answer by the first newline.")
        logger.warning(
            f"[Rank {rank}] We remove all the pre/appended space/newlines and replace newlines with spaces."
        )
        logger.warning(
            f"[Rank {rank}] We replace any on the fly search result to standard bracket citation format."
        )
        for i in range(len(data)):
            data[i]["output"] = re.sub(r"\n+", " ", data[i]["output"])
            data[i]["output"] = data[i]["output"].replace("<|im_end|>", "")

        for i in range(len(local_data)):
            local_data[i]["output"] = re.sub(
                r"\n+", " ", local_data[i]["output"]
            )
            local_data[i]["output"] = local_data[i]["output"].replace(
                "<|im_end|>", ""
            )

        # Remove all citations for all non-AutoAIS evaluation
        normalized_data = copy.deepcopy(data)
        local_normalized_data = copy.deepcopy(local_data)
        for i in range(len(normalized_data)):
            normalized_data[i]["output"] = remove_citations(
                normalized_data[i]["output"]
            )
        for i in range(len(local_normalized_data)):
            local_normalized_data[i]["output"] = remove_citations(
                local_normalized_data[i]["output"]
            )

        alce_metrics = {}
        logger.warning(f"[Rank {rank}] compute_len")
        alce_metrics["length"] = compute_len(normalized_data)
        logger.warning(f"[Rank {rank}] compute_str_em")
        alce_metrics["str_em"], alce_metrics["str_hit"] = compute_str_em(
            local_normalized_data, allgather=True
        )
        if qampari:
            logger.warning(f"[Rank {rank}] compute_qampari_f1")
            alce_metrics.update(
                compute_qampari_f1(
                    local_normalized_data, cot=args.alce_cot, allgather=True
                )
            )
        if not args.alce_no_rouge:
            logger.warning(f"[Rank {rank}] compute_rouge")
            alce_metrics["rougeLsum"] = compute_rouge(
                local_normalized_data, allgather=True
            )
        if args.alce_qa:
            logger.warning(f"[Rank {rank}] compute_qa")
            alce_metrics.update(
                compute_qa(
                    local_normalized_data,
                    allgather=True,
                    qa_model_name_or_path=args.qa_model_name_or_path,
                )
            )
        # if args.alce_mauve:
        #    # not support dist eval
        #    if rank == 0:
        #        alce_metrics['mauve'] = compute_mauve(normalized_data)
        if args.alce_citations:
            logger.warning(
                f"[Rank {rank}] compute_autoais {args.autoais_model_name_or_path}"
            )
            alce_metrics.update(
                compute_autoais(
                    local_data,
                    qampari=qampari,
                    at_most_citations=args.alce_at_most_citations,
                    allgather=True,
                    autoais_model_name_or_path=args.autoais_model_name_or_path,
                )
            )
        if args.alce_claims_nli:
            logger.warning(
                f"[Rank {rank}] compute_claims {args.autoais_model_name_or_path}"
            )
            alce_metrics["claims_nli"] = compute_claims(
                local_normalized_data,
                allgather=True,
                autoais_model_name_or_path=args.autoais_model_name_or_path,
            )

    averaged_metrics = {
        k: np.mean(v) * (100 if "_len" not in k and "_ms" not in k else 1)
        for k, v in metrics.items()
    }

    logger.info("Averaged metrics:")
    for k, v in averaged_metrics.items():
        logger.info(f"{k}: {v:.02f}")

    output = {
        "args": args.__dict__,
        "metrics": metrics,
        "averaged_metrics": averaged_metrics,
        "throughput": len(results) / (end_time - start_time),
    }
    if args.save_outputs:
        output["data"] = results

    if "alce" in dataset:
        output["alce_metrics"] = alce_metrics

    if mem_usage is not None:
        output["memory_usage"] = mem_usage

    if rank == 0:
        if args.output_dir is not None:
            with open(output_path, "w") as f:
                json.dump(output, f, indent=4)
            # this makes it easier to parse results, but alce uses a different evaluation script
            if "alce" not in dataset:
                with open(output_path + ".score", "w") as f:
                    json.dump(output["averaged_metrics"], f, indent=4)
            else:
                with open(output_path + ".score", "w") as f:
                    if "sparse_ratios" in averaged_metrics:
                        alce_metrics["sparse_ratios"] = averaged_metrics[
                            "sparse_ratios"
                        ]
                    if "ttft_ms" in averaged_metrics:
                        alce_metrics["ttft_ms"] = averaged_metrics["ttft_ms"]
                    if "e2e_ms" in averaged_metrics:
                        alce_metrics["e2e_ms"] = averaged_metrics["e2e_ms"]
                    if "attn_ms" in averaged_metrics:
                        alce_metrics["attn_ms"] = averaged_metrics["attn_ms"]
                    if "estimate_func_ms" in averaged_metrics:
                        alce_metrics["estimate_func_ms"] = averaged_metrics[
                            "estimate_func_ms"
                        ]
                    json.dump(alce_metrics, f, indent=4)
            logger.info(f"done, results are written to {output_path}")

    # if world_size > 1:
    #    dist.barrier()

    # 整理碎片
    if (
        not getattr(model, "uses_paddle_worker", False)
        and torch.cuda.is_available()
    ):
        torch.cuda.empty_cache()
    return output_path


def main():
    args = parse_arguments()

    logger.info(f"Arguments: {args}")
    assert args.model_name_or_path is not None
    os.makedirs(args.output_dir, exist_ok=True)

    os.environ["NCCL_BLOCKING_WAIT"] = "0"  # not to enforce timeout

    # Parent HELMET process keeps torch/HF data and judge flow. Use gloo by
    # default so the tested model owns the CUDA context inside the Paddle worker.
    parent_backend = os.environ.get("HELMET_PARENT_DIST_BACKEND", "gloo")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    dist.init_process_group(
        backend=parent_backend, timeout=datetime.timedelta(seconds=6000000)
    )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if parent_backend == "nccl":
        torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    datasets = args.datasets.split(",")
    test_files = args.test_files.split(",")
    demo_files = args.demo_files.split(",")
    max_lengths = (
        ([int(args.input_max_length)] * len(datasets))
        if isinstance(args.input_max_length, int)
        or len(args.input_max_length.split(",")) == 1
        else [int(l) for l in args.input_max_length.split(",")]
    )
    gen_lengths = (
        ([int(args.generation_max_length)] * len(datasets))
        if isinstance(args.generation_max_length, int)
        or len(args.generation_max_length.split(",")) == 1
        else [int(l) for l in args.generation_max_length.split(",")]
    )
    assert len(test_files) == len(demo_files)

    args.input_max_length = max(max_lengths)
    model = load_LLM(args)

    if getattr(model, "uses_paddle_worker", False):
        logger.info(f"[Rank {rank}] Paddle worker handles attention patching")
    elif args.method != "full":
        raise RuntimeError(
            "Legacy Torch sparse attention patching is not included in this release. "
            "Use the Paddle worker path for rrattn evaluation."
        )
    else:
        logger.info(
            f"[Rank {rank}] Running full attention or API/serving evaluation without attention patching"
        )

    try:
        logger.info(f"[Rank {rank}] {len(datasets)} dataset: {datasets}")
        for dataset, test_file, demo_file, max_length, gen_length in zip(
            datasets, test_files, demo_files, max_lengths, gen_lengths
        ):
            args.datasets = dataset
            args.test_files = test_file
            args.demo_files = demo_file
            args.input_max_length = max_length
            args.generation_max_length = gen_length
            model.max_length = max_length
            model.generation_max_length = gen_length

            try:
                output_path = run_test(
                    args, model, dataset, test_file, demo_file
                )
            except Exception as e:
                logger.exception(e)
                logger.error(f"Error in {dataset}, aborting all ranks...")
                if (
                    dist.is_initialized()
                    and dist.get_world_size() > 1
                    and hasattr(dist, "abort")
                ):
                    dist.abort()
                raise
    finally:
        if hasattr(model, "close"):
            model.close()

    if world_size > 1:
        dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
