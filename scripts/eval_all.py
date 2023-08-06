import configparser
import subprocess
import pandas as pd
import re
from cpuinfo import get_cpu_info
import psutil
import torch
import argparse


def parse_output(output: str):
    block_lat = 0
    for line in reversed(output.splitlines()):
        m = re.search(r"BLK AVG: ([\d\.]+) ms", line)
        if m:
            block_lat = float(m[1])
            break

    throughput = .0
    for line in reversed(output.splitlines()):
        m = re.search(r", (\d+) tokens/sec\.", line)
        if m:
            throughput = int(m[1])
            break
    
    peak_mem_encoder = peak_mem_decoder = 0
    for line in output.splitlines():
        m = re.search(r"MEM usage: (\d+) (\d+)", line)
        if m:
            peak_mem_encoder = int(m[1])
            peak_mem_decoder = int(m[2])
            break

    max_active_experts = 0
    for line in output.splitlines():
        m = re.search(r"Max active experts: (\d+)", line)
        if m:
            max_active_experts = int(m[1])
            break
    
    return block_lat, throughput, peak_mem_encoder, peak_mem_decoder, max_active_experts


def profile_config(cpp_config, model, method, batch_size):
    iterations = 4
    exp_name = f"{model}_{method}_{batch_size}"
    print(f"Running {exp_name}")
    if method == "GPU-only":
        encoder_fetcher_mode = "0"
        decoder_fetcher_mode = "0"
    elif method == "Pre-gated":
        encoder_fetcher_mode = "1"
        decoder_fetcher_mode = "2"
    elif method == "DeepSpeed":
        encoder_fetcher_mode = "1"
        decoder_fetcher_mode = "1"
    elif method == "SE-MoE":
        encoder_fetcher_mode = "1"
        decoder_fetcher_mode = "2"
        iterations = 1

    cpp_config["default"] = {
        "arena_size": "21474836480",
        "encoder_fetcher_mode": encoder_fetcher_mode,
        "decoder_fetcher_mode": decoder_fetcher_mode,
        "profiling": "1",
        "detailed_timing": "0",
        "offload_path": "/data/ft/{}/".format(model),
        "disk_offload": "0",
        "load_from_cpp": "1",
        "use_cache": "0",
        "quant_mode": "0",
        "vocab_size": "32128",
        "fetch_all": str(int(method == "SE-MoE")),
        "forced_num_experts": "0",
    }

    with open("/workspace/FasterTransformer/cpp_config.ini", "w") as fp:
        cpp_config.write(fp)

    command = (
        f"python /workspace/FasterTransformer/examples/pytorch/t5/perf_benchmark.py "
        f"--batch_size {batch_size} "
        f"--beam_width 4 "
        f"--seq_len 256 "
        f"--data_type fp32 "
        f"--test_time 3 "
        f"--sampling_topk 1 "
        f"--model_type Megatron-DeepSpeed "
        f"--ckpt_path /data/ft/{model}/ "
        f"--model t5-base "
        f"--duration 0 "
        f"--iterations {iterations} "
    )

    print(command)

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        cwd="/workspace/FasterTransformer/build"
    )

    with open(f"/workspace/FasterTransformer/logs/{exp_name}.log", "w") as fp:
        fp.write(result.stdout)

    block_lat, throughput, peak_mem_encoder, peak_mem_decoder, max_active_experts = parse_output(result.stdout)

    if "base" in model:
        size_per_expert = 18874368
    elif "large" in model:
        size_per_expert = 33554432

    total_experts = int(re.search(r"\d+", model)[0])
    arena_size = 21474836480

    if method == "Pre-gated":
        used_buffer = 2 * max_active_experts
    elif method == "DeepSpeed":
        used_buffer = max_active_experts
    elif method == "GPU-only":
        used_buffer = 0
    elif method == "SE-MoE":
        used_buffer = 2 * total_experts

    peak_mem = peak_mem_decoder - arena_size - size_per_expert * (2 * total_experts - used_buffer)
    print(
        f"BLK AVG: {block_lat} ms, "
        f"throughput: {throughput} tokens/sec, "
        f"peak_mem_encoder: {peak_mem_encoder}, "
        f"peak_mem_decoder: {peak_mem_decoder}, "
        f"max_active_experts: {max_active_experts}, "
        f"peak_mem: {peak_mem}"
    )

    return {
        "block_lat": block_lat,
        "throughput": throughput,
        "peak_mem": peak_mem,
        "max_active_expert": max_active_experts,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--re_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    models = [
        "switch-base-8",
        # "switch-base-16",
        # "switch-base-32",
        "switch-base-64",
        # "switch-base-128",
        "switch-base-256",
        "switch-large-128",
    ]
    batch_sizes = [
        1,
        # 2,
        # 4,
        8,
        16,
    ]
    methods = [
        # "GPU-only",
        "Pre-gated",
        # "DeepSpeed",
        # "SE-MoE",
    ]
    metrics = [
        "peak_mem",
        "max_active_expert",
    ]

    cpp_config = configparser.ConfigParser()
    cpp_config.read("/workspace/FasterTransformer/cpp_config.ini")

    hardware_infos = {
        "CPU": [get_cpu_info()["brand_raw"]] * len(batch_sizes),
        "RAM (GB)": [int(psutil.virtual_memory().total / 1024 / 1024 / 1024)] * len(batch_sizes),
        "GPU": [torch.cuda.get_device_name()] * len(batch_sizes),
    }

    results = {}
    for metric in metrics:
        results[metric] = {"bs": batch_sizes}
        results[metric].update(hardware_infos)

    if not args.re_run:
        for method in methods:
            for model in models:
                records = []

                for batch_size in batch_sizes:
                    records.append(profile_config(cpp_config, model, method, batch_size))

                for metric, result in results.items():
                    result["{}/{}".format(model, method)] = [record[metric] for record in records]

                # Generate CSV after each model and method runned
                for metric, result in results.items():
                    df = pd.DataFrame.from_dict(result)
                    df.to_csv(f"{metric}s.csv", index=False)

    else:
        dfs = {metric: pd.read_csv(f"/workspace/FasterTransformer/performance_data/{metric}s.csv") for metric in metrics}
        models = [
            "switch-base-8",
            # "switch-base-16",
            # "switch-base-32",
            # "switch-base-64",
            # "switch-base-128",
            # "switch-base-256",
            # "switch-large-128",
        ]
        batch_sizes = [
            1,
            2,
            4,
            8,
            16,
        ]
        methods = [
            "GPU-only",
            "Pre-gated",
            "DeepSpeed",
            "SE-MoE",
        ]
        rerun_configs = [
            (model, method, batch_size)
            for model in models
            for method in methods
            for batch_size in batch_sizes
        ]
        for model, method, batch_size in rerun_configs:
            row_idx = batch_sizes.index(batch_size)
            col_idx = "{}/{}".format(model, method)
            record = profile_config(cpp_config, model, method, batch_size)
            for metric, df in dfs.items():
                df.loc[row_idx, col_idx] = record[metric]
                df.to_csv(f"{metric}s.csv", index=False)


if __name__ == "__main__":
    main()
