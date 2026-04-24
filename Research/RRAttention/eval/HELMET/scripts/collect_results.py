import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import yaml
from tabulate import tabulate
from tqdm import tqdm

dataset_to_metrics = {
    "json_kv": "substring_exact_match",
    "nq": "substring_exact_match",
    "popqa": "substring_exact_match",
    "triviaqa": "substring_exact_match",
    "hotpotqa": "substring_exact_match",
    "narrativeqa": ["gpt-4-score"],
    "msmarco_rerank_psg": "NDCG@10",
    "trec_coarse": "exact_match",
    "trec_fine": "exact_match",
    "banking77": "exact_match",
    "clinic150": "exact_match",
    "nlu": "exact_match",
    "qmsum": "rougeL_recall",
    "multi_lexsum": ["gpt-4-f1"],
    "ruler_niah_s_1": "ruler_recall",
    "ruler_niah_s_2": "ruler_recall",
    "ruler_niah_s_3": "ruler_recall",
    "ruler_niah_mk_1": "ruler_recall",
    "ruler_niah_mk_2": "ruler_recall",
    "ruler_niah_mk_3": "ruler_recall",
    "ruler_niah_mq": "ruler_recall",
    "ruler_niah_mv": "ruler_recall",
    "ruler_fwe": "ruler_recall",
    "ruler_cwe": "ruler_recall",
    "ruler_vt": "ruler_recall",
    "ruler_qa_1": "substring_exact_match",
    "ruler_qa_2": "substring_exact_match",
    "infbench_qa": ["rougeL_f1"],
    "infbench_choice": ["exact_match"],
    "infbench_sum": ["gpt-4-f1"],
    "alce_asqa": ["str_em", "citation_rec", "citation_prec"],
    "alce_qampari": ["qampari_rec_top5", "citation_rec", "citation_prec"],
}

speed_metrics = [
    "sparse_ratios",
    "e2e_ms",
    "ttft_ms",
    "attn_ms",
    "estimate_func_ms",
]

dataset_to_metrics = {
    k: [v] if isinstance(v, str) else v for k, v in dataset_to_metrics.items()
}
custom_avgs = {
    "Recall": [
        "json_kv substring_exact_match",
        "ruler_niah_mk_2 ruler_recall",
        "ruler_niah_mk_3 ruler_recall",
        "ruler_niah_mv ruler_recall",
    ],
    "RAG": [
        "nq substring_exact_match",
        "hotpotqa substring_exact_match",
        "popqa substring_exact_match",
        "triviaqa substring_exact_match",
    ],
    "ICL": [
        "trec_coarse exact_match",
        "trec_fine exact_match",
        "banking77 exact_match",
        "clinic150 exact_match",
        "nlu exact_match",
    ],
    "Cite": [
        "alce_asqa str_em",
        "alce_asqa citation_rec",
        "alce_asqa citation_prec",
        "alce_qampari qampari_rec_top5",
        "alce_qampari citation_rec",
        "alce_qampari citation_prec",
    ],
    "Re-rank": [
        "msmarco_rerank_psg NDCG@10",
    ],
    # "LongQA": ['narrativeqa gpt-4-score', 'infbench_qa rougeL_f1', 'infbench_choice exact_match', ],
    "LongQA": [
        "infbench_qa rougeL_f1",
        "infbench_choice exact_match",
    ],
    # "Summ": ['infbench_sum gpt-4-f1', 'multi_lexsum gpt-4-f1', ],
    # "RULER": ['ruler_niah_s_1 ruler_recall', 'ruler_niah_s_2 ruler_recall', 'ruler_niah_s_3 ruler_recall', 'ruler_niah_mk_1 ruler_recall', 'ruler_niah_mk_2 ruler_recall', 'ruler_niah_mk_3 ruler_recall', 'ruler_niah_mq ruler_recall', 'ruler_niah_mv ruler_recall', 'ruler_cwe ruler_recall', 'ruler_fwe ruler_recall', 'ruler_vt ruler_recall', 'ruler_qa_1 substring_exact_match', 'ruler_qa_2 substring_exact_match'],
    # "Ours": ['Recall', 'RAG', 'ICL', 'Cite', 'Re-rank', 'LongQA', 'Summ'],
    "Avg.": ["Recall", "RAG", "ICL", "Cite", "Re-rank", "LongQA"],
}


@dataclass
class arguments:
    tag: str = "v1"
    input_max_length: int = 131072
    generation_max_length: int = 100
    generation_min_length: int = 0
    max_test_samples: int = 100
    shots: int = 2
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    use_chat_template: bool = False
    seed: int = 42
    test_name: str = ""
    dataset: str = "nq"
    output_dir: str = "output"
    popularity_threshold: float = 3

    category: str = "synthetic"

    def update(self, new):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_path(self, return_speed=False):
        tag = self.tag
        path = os.path.join(
            self.output_dir,
            f"{self.dataset}_{tag}_{self.test_name}_in{self.input_max_length}_size{self.max_test_samples}_shots{self.shots}_samp{self.do_sample}max{self.generation_max_length}min{self.generation_min_length}t{self.temperature}p{self.top_p}_chat{self.use_chat_template}_{self.seed}.json",
        )
        if return_speed:
            return path.replace(".json", ".json.speed")

        if os.path.exists(path.replace(".json", "-gpt4eval_o.json")):
            return path.replace(".json", "-gpt4eval_o.json")
        if "alce" in self.dataset:
            return path.replace(".json", ".json.score")

        if os.path.exists(path + ".score"):
            return path + ".score"
        return path

    def get_metric_name(self):
        for d, m in dataset_to_metrics.items():
            if d in self.dataset:
                return d, m
        return None

    def get_averaged_metric(self):
        path = self.get_path()
        print(path)
        if not os.path.exists(path):
            print("path doesn't exist")
            return None
        with open(path) as f:
            results = json.load(f)

        _, metric = self.get_metric_name()

        if path.endswith(".score"):
            if any(m not in results for m in metric):
                print("metric doesn't exist")
                return None
            s = {m: results[m] for m in metric}
            for m in speed_metrics:
                if m in results:
                    s[m] = results[m]
        else:
            if any(m not in results["averaged_metrics"] for m in metric):
                print("metric doesn't exist")
                return None
            s = {m: results["averaged_metrics"][m] for m in metric}
            for m in speed_metrics:
                if m in results["averaged_metrics"]:
                    s[m] = results["averaged_metrics"][m]

        # Scale metrics, but ignore sparse_ratios
        s = {
            m: v
            * (100 if m == "gpt-4-f1" else 1)
            * (100 / 3 if m == "gpt-4-score" else 1)
            if (m != "sparse_ratios" and "_ms" not in m)
            else v
            for m, v in s.items()
        }
        print("found scores:", s)
        return s

    def get_metric_by_depth(self):
        path = self.get_path()
        path = path.replace(".score", "")
        print(path)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            results = json.load(f)

        output = []
        _, metric = self.get_metric_name()
        metric = metric[0]
        keys = ["depth", "k", metric]
        for d in results["data"]:
            o = {}
            for key in keys:
                if key == "k" and "ctxs" in d:
                    d["k"] = len(d["ctxs"])
                if key not in d:
                    print("no", key)
                    return None
                o[key] = d[key]
            o["metric"] = o.pop(metric)
            output.append(o)

        df = pd.DataFrame(output)
        dfs = df.groupby(list(output[0].keys())[:-1]).mean().reset_index()

        return dfs.to_dict("records")


if __name__ == "__main__":
    # comment out the models you don't want to include, or add the new ones
    models_configs = [
        {
            "model": "gpt-4-0125-preview",
            "use_chat_template": True,
            "training_length": 128000,
        },
        {
            "model": "gpt-4o-mini-2024-07-18",
            "use_chat_template": True,
            "training_length": 128000,
        },
        {
            "model": "gpt-4o-2024-05-13",
            "use_chat_template": True,
            "training_length": 128000,
        },
        {
            "model": "gpt-4o-2024-08-06",
            "use_chat_template": True,
            "training_length": 128000,
        },
        {
            "model": "claude-3-5-sonnet-20240620",
            "use_chat_template": True,
            "training_length": 200000,
        },
        {
            "model": "gemini-1.5-flash-001",
            "use_chat_template": True,
            "training_length": 1048576,
        },
        {
            "model": "gemini-1.5-pro-001",
            "use_chat_template": True,
            "training_length": 2097152,
        },
        # llama 2 based models
        {
            "model": "Llama-2-7B-32K",
            "use_chat_template": False,
            "training_length": 32768,
        },
        {"model": "Llama-2-7B-32K-Instruct", "training_length": 32768},
        {
            "model": "llama-2-7b-80k",
            "use_chat_template": False,
            "training_length": 80000,
        },
        {
            "model": "Yarn-Llama-2-7b-64k",
            "use_chat_template": False,
            "training_length": 65536,
        },
        {
            "model": "Yarn-Llama-2-7b-128k",
            "use_chat_template": False,
            "training_length": 131072,
        },
        # llama 3 models
        {
            "model": "Meta-Llama-3-8B",
            "use_chat_template": False,
            "training_length": 8192,
        },
        {"model": "Meta-Llama-3-8B-Instruct", "training_length": 8192},
        {
            "model": "Meta-Llama-3-8B-Theta16M",
            "use_chat_template": False,
            "training_length": 8192,
        },
        {"model": "Meta-Llama-3-8B-Instruct-Theta16M", "training_length": 8192},
        {
            "model": "Meta-Llama-3-70B-Theta16M",
            "use_chat_template": False,
            "training_length": 8192,
        },
        {
            "model": "Meta-Llama-3-70B-Instruct-Theta16M",
            "training_length": 8192,
        },
        {
            "model": "Llama-3.1-8B",
            "use_chat_template": False,
            "training_length": 131072,
        },
        {"model": "Llama-3.1-8B-Instruct", "training_length": 131072},
        {
            "model": "Llama-3.1-70B",
            "use_chat_template": False,
            "training_length": 131072,
        },
        {"model": "Llama-3.1-70B-Instruct", "training_length": 131072},
        {"model": "Llama-3.3-70B-Instruct", "training_length": 131072},
        {
            "model": "Llama-3.2-1B",
            "use_chat_template": False,
            "training_length": 131072,
        },
        {"model": "Llama-3.2-1B-Instruct", "training_length": 131072},
        {
            "model": "Llama-3.2-3B",
            "use_chat_template": False,
            "training_length": 131072,
        },
        {"model": "Llama-3.2-3B-Instruct", "training_length": 131072},
        # mistral models
        {
            "model": "Mistral-7B-v0.1",
            "use_chat_template": False,
            "training_length": 8192,
        },
        {"model": "Mistral-7B-Instruct-v0.1", "training_length": 8192},
        {"model": "Mistral-7B-Instruct-v0.2", "training_length": 32768},
        {
            "model": "Mistral-7B-v0.3",
            "use_chat_template": False,
            "training_length": 32768,
        },
        {"model": "Mistral-7B-Instruct-v0.3", "training_length": 32768},
        {"model": "Ministral-8B-Instruct-2410", "training_length": 131072},
        {
            "model": "Mistral-Nemo-Base-2407",
            "use_chat_template": False,
            "training_length": 128000,
        },
        {"model": "Mistral-Nemo-Instruct-2407", "training_length": 128000},
        {"model": "MegaBeam-Mistral-7B-512k", "training_length": 524288},
        # yi models
        {
            "model": "Yi-6B-200K",
            "use_chat_template": False,
            "training_length": 200000,
        },
        {
            "model": "Yi-9B-200K",
            "use_chat_template": False,
            "training_length": 200000,
        },
        {
            "model": "Yi-34B-200K",
            "use_chat_template": False,
            "training_length": 200000,
        },
        {
            "model": "Yi-1.5-9B-32K",
            "use_chat_template": False,
            "training_length": 32768,
        },
        # phi models
        {"model": "Phi-3-mini-128k-instruct", "training_length": 131072},
        {"model": "Phi-3-small-128k-instruct", "training_length": 131072},
        {"model": "Phi-3-medium-128k-instruct", "training_length": 131072},
        {"model": "Phi-3.5-mini-instruct", "training_length": 131072},
        # qwen models
        {
            "model": "Qwen2-7B",
            "use_chat_template": False,
            "training_length": 32768,
        },
        {"model": "Qwen2-7B-Instruct", "training_length": 32768},
        {
            "model": "Qwen2-57B-A14B",
            "use_chat_template": False,
            "training_length": 32768,
        },
        {"model": "Qwen2-57B-A14B-Instruct", "training_length": 32768},
        {
            "model": "Qwen2.5-1.5B",
            "use_chat_template": False,
            "training_length": 32768,
        },
        {"model": "Qwen2.5-1.5B-Instruct", "training_length": 32768},
        {
            "model": "Qwen2.5-3B",
            "use_chat_template": False,
            "training_length": 32768,
        },
        {"model": "Qwen2.5-3B-Instruct", "training_length": 32768},
        {
            "model": "Qwen2.5-7B",
            "use_chat_template": False,
            "training_length": 131072,
        },
        {"model": "Qwen2.5-7B-Instruct", "training_length": 131072},
        {"model": "Qwen2.5-72B-Instruct", "training_length": 131072},
        # prolong
        {
            "model": "Llama-3-8B-ProLong-512k-Instruct",
            "training_length": 524288,
        },
        # gemma 2 models
        {
            "model": "gemma-2-9b",
            "use_chat_template": False,
            "training_length": 8192,
        },
        {"model": "gemma-2-9b-it", "training_length": 8192},
        {"model": "gemma-2-9b-it-Theta320K", "training_length": 8192},
        {
            "model": "gemma-2-27b",
            "use_chat_template": False,
            "training_length": 8192,
        },
        {"model": "gemma-2-27b-it", "training_length": 8192},
        {"model": "gemma-2-27b-it-Theta320K", "training_length": 8192},
        # others
        {"model": "c4ai-command-r-v01", "training_length": 131072},
        {
            "model": "Jamba-v0.1",
            "use_chat_template": False,
            "training_length": 262144,
        },
        {"model": "AI21-Jamba-1.5-Mini", "training_length": 262144},
    ]

    models_configs = [
        {
            "model": "Llama-3.1-8B",
            "use_chat_template": False,
            "training_length": 131072,
        },
        {"model": "Llama-3.1-8B-Instruct", "training_length": 131072},
        {
            "model": "DeepSeek-R1-Distill-Llama-8B",
            "training_length": 131072,
            "do_sample": True,
            "temperature": 0.6,
        },
        {
            "model": "Qwen2-7B",
            "use_chat_template": False,
            "training_length": 32768,
        },
        {"model": "Qwen2-7B-Instruct", "training_length": 32768},
        {
            "model": "DeepSeek-R1-Distill-Qwen-7B",
            "training_length": 131072,
            "do_sample": True,
            "temperature": 0.6,
        },
    ]

    models_configs = [
        {"model": "Meta-Llama-3.1-8B-Instruct/full", "training_length": 131072},
        {
            "model": "Meta-Llama-3.1-8B-Instruct/rrattn_0.9",
            "training_length": 131072,
        },
        {
            "model": "Meta-Llama-3.1-8B-Instruct/rrattn_0.95",
            "training_length": 131072,
        },
        {
            "model": "Meta-Llama-3.1-8B-Instruct/rrattn_0.99",
            "training_length": 131072,
        },
        {
            "model": "Meta-Llama-3.1-8B-Instruct/flex_0.9",
            "training_length": 131072,
        },
        {
            "model": "Meta-Llama-3.1-8B-Instruct/flex_0.95",
            "training_length": 131072,
        },
        {
            "model": "Meta-Llama-3.1-8B-Instruct/flex_0.99",
            "training_length": 131072,
        },
        {
            "model": "Meta-Llama-3.1-8B-Instruct/xattn_0.9",
            "training_length": 131072,
        },
        {
            "model": "Meta-Llama-3.1-8B-Instruct/xattn_0.95",
            "training_length": 131072,
        },
        {
            "model": "Meta-Llama-3.1-8B-Instruct/xattn_0.99",
            "training_length": 131072,
        },
        # {"model": "Meta-Llama-3.1-8B-Instruct/rrattn_0.9_s4", "training_length": 131072},
        # {"model": "Meta-Llama-3.1-8B-Instruct/rrattn_0.95_s4", "training_length": 131072},
        # {"model": "Meta-Llama-3.1-8B-Instruct/rrattn_0.99_s4", "training_length": 131072},
        # {"model": "Meta-Llama-3.1-8B-Instruct/rrattn_0.9_s16", "training_length": 131072},
        # {"model": "Meta-Llama-3.1-8B-Instruct/rrattn_0.95_s16", "training_length": 131072},
        # {"model": "Meta-Llama-3.1-8B-Instruct/rrattn_0.99_s16", "training_length": 131072},
        # {"model": "Meta-Llama-3.1-8B-Instruct/rrattn_0.9_s32", "training_length": 131072},
        # {"model": "Meta-Llama-3.1-8B-Instruct/rrattn_0.95_s32", "training_length": 131072},
        # {"model": "Meta-Llama-3.1-8B-Instruct/rrattn_0.99_s32", "training_length": 131072},
        {"model": "Qwen2.5-7B-Instruct/full", "training_length": 131072},
        {"model": "Qwen2.5-7B-Instruct/rrattn_0.9", "training_length": 131072},
        {"model": "Qwen2.5-7B-Instruct/rrattn_0.95", "training_length": 131072},
        {"model": "Qwen2.5-7B-Instruct/rrattn_0.99", "training_length": 131072},
        {"model": "Qwen2.5-7B-Instruct/flex_0.9", "training_length": 131072},
        {"model": "Qwen2.5-7B-Instruct/flex_0.95", "training_length": 131072},
        {"model": "Qwen2.5-7B-Instruct/flex_0.99", "training_length": 131072},
        {"model": "Qwen2.5-7B-Instruct/xattn_0.9", "training_length": 131072},
        {"model": "Qwen2.5-7B-Instruct/xattn_0.95", "training_length": 131072},
        {"model": "Qwen2.5-7B-Instruct/xattn_0.99", "training_length": 131072},
        {"model": "ERNIE-4.5-21B-A3B-PT/full", "training_length": 131072},
        {"model": "ERNIE-4.5-21B-A3B-PT/rrattn_0.9", "training_length": 131072},
        {
            "model": "ERNIE-4.5-21B-A3B-PT/rrattn_0.95",
            "training_length": 131072,
        },
        {
            "model": "ERNIE-4.5-21B-A3B-PT/rrattn_0.99",
            "training_length": 131072,
        },
    ]

    # set your configs here, only include the ones that you ran
    config_files = [
        "configs/recall.yaml",
        "configs/recall_short.yaml",
        "configs/rag.yaml",
        "configs/rag_short.yaml",
        "configs/longqa.yaml",
        "configs/longqa_short.yaml",
        "configs/summ.yaml",
        "configs/summ_short.yaml",
        "configs/rerank.yaml",
        "configs/rerank_short.yaml",
        "configs/icl.yaml",
        "configs/icl_short.yaml",
        "configs/cite.yaml",
        "configs/cite_short.yaml",
        "configs/ruler.yaml",
        "configs/ruler_short.yaml",
    ]

    config_files = [
        "configs/recall.yaml",
        "configs/recall_short.yaml",
        "configs/rag.yaml",
        "configs/rag_short.yaml",
        "configs/longqa.yaml",
        "configs/longqa_short.yaml",
        "configs/rerank.yaml",
        "configs/rerank_short.yaml",
        "configs/icl.yaml",
        "configs/icl_short.yaml",
        "configs/cite.yaml",
        "configs/cite_short.yaml",
    ]

    tags = ["v1"]

    dataset_configs = []
    for file in config_files:
        c = yaml.safe_load(open(file))

        if isinstance(c["generation_max_length"], int):
            c["generation_max_length"] = ",".join(
                [str(c["generation_max_length"])]
                * len(c["datasets"].split(","))
            )
        for d, t, l, g in zip(
            c["datasets"].split(","),
            c["test_files"].split(","),
            c["input_max_length"].split(","),
            c["generation_max_length"].split(","),
        ):
            dataset_configs.append(
                {
                    "dataset": d,
                    "test_name": os.path.basename(os.path.splitext(t)[0]),
                    "input_max_length": int(l),
                    "generation_max_length": int(g),
                    "max_test_samples": c["max_test_samples"],
                    "use_chat_template": c["use_chat_template"],
                    "shots": c["shots"],
                }
            )
    print(dataset_configs)

    failed_paths = []
    df = []
    for model in tqdm(models_configs):
        for tag in tags:
            args = arguments()
            # args.output_dir = f"output_bak/{model['model']}"
            args.output_dir = f"output/{model['model']}"

            args.tag = tag
            for dataset in dataset_configs:
                args.update(dataset)
                args.update(model)

                metric = args.get_averaged_metric()
                dsimple, mnames = args.get_metric_name()

                if metric is None:
                    failed_paths.append(args.get_path())
                    continue

                # Extract sparse_ratios from the metric dict, default to np.nan if not found
                speed_metrics_dict = {}
                for m in speed_metrics:
                    speed_metrics_dict[m] = metric.pop(m, np.nan)

                for k, m in metric.items():
                    df.append(
                        {
                            **asdict(args),
                            **model,
                            "metric name": k,
                            "metric": m,
                            "dataset_simple": dsimple + " " + k,
                            "test_data": f"{args.dataset}-{args.test_name}-{args.input_max_length}",
                            **speed_metrics_dict,
                        }
                    )

    all_df = pd.DataFrame(df)

    lf_df = all_df.pivot_table(
        index=[
            "input_max_length",
            "model",
            "tag",
        ],
        columns="dataset_simple",
        values="metric",
        sort=False,
    )
    lf_df = lf_df.reset_index()

    for m in speed_metrics:
        avg_tmp_df = (
            all_df.dropna(subset=[m])
            .groupby(["input_max_length", "model", "tag"])[m]
            .mean()
            .reset_index()
        )
        lf_df = pd.merge(
            lf_df,
            avg_tmp_df,
            on=["input_max_length", "model", "tag"],
            how="left",
        )

    for k, v in custom_avgs.items():
        # Handle missing columns gracefully when calculating custom averages
        valid_cols = [col for col in v if col in lf_df.columns]
        if valid_cols:
            lf_df[k] = lf_df[valid_cols].mean(axis=1)
        else:
            lf_df[k] = np.nan

    # Markdown 输出
    # print(lf_df.to_markdown(index=False))

    cols_to_show = [
        "input_max_length",
        "model",
        "tag",
        *speed_metrics,
        "Avg.",
        "Recall",
        "RAG",
        "ICL",
        "Cite",
        "Re-rank",
        "LongQA",
    ]
    cols_to_float = [
        *speed_metrics,
        "Avg.",
        "Recall",
        "RAG",
        "ICL",
        "Cite",
        "Re-rank",
        "LongQA",
    ]

    # lf_df = lf_df.sort_values('input_max_length', ascending=True)

    # 1. 找出input_max_length的升序顺序
    length_order = sorted(lf_df["input_max_length"].unique())

    # 2. 在每一个input_max_length下，model的顺序保持原始顺序
    model_order = lf_df["model"].drop_duplicates().tolist()

    # 3. 用input_max_length和model做多级排序，并用model_order自定义排序
    lf_df["model_order"] = pd.Categorical(
        lf_df["model"], categories=model_order, ordered=True
    )
    lf_df["length_order"] = pd.Categorical(
        lf_df["input_max_length"], categories=length_order, ordered=True
    )
    lf_df = lf_df.sort_values(["length_order", "model_order"]).drop(
        ["model_order", "length_order"], axis=1
    )

    lf_df[cols_to_float] = lf_df[cols_to_float].map(
        lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
    )
    # print(tabulate(lf_df[cols_to_show], headers=cols_to_show, tablefmt='pipe', showindex=False, floatfmt='.2f'))

    df_show = lf_df[cols_to_show]

    sep_line = []
    for idx, col in enumerate(df_show.columns):
        # Calculate max length needed for column, considering header and data
        max_len = max(df_show[col].astype(str).map(len).max(), len(str(col)))

        # Add padding to match tabulate's default padding (1 space on each side)
        max_len += 2

        # Manually adjust for first/last column if tabulate does something special (usually not needed with 'pipe')
        # if idx == 0:
        #     max_len += 1

        sep_line.append("-" * max_len)

    df_sorted = df_show.reset_index(drop=True)
    # 生成带分割线的数据列表
    rows = []
    prev_length = None
    for _, row in df_show.iterrows():
        if prev_length is not None and row["input_max_length"] != prev_length:
            # 插入一行分隔
            rows.append(sep_line)
        rows.append(list(row))
        prev_length = row["input_max_length"]

    # 输出
    colalign = ["right"] * 1 + ["left"] * (len(cols_to_show) - 1)
    print(
        tabulate(
            rows,
            headers=cols_to_show,
            tablefmt="pipe",
            showindex=False,
            colalign=colalign,
        )
    )

    # print(lf_df.to_csv(index=False))

    # print("Warning, failed to get the following paths, make sure that these are correct or the printed results will not be accurate:", failed_paths)
    # import pdb; pdb.set_trace()
