import configparser
import subprocess
import pandas as pd
import re
from cpuinfo import get_cpu_info
import psutil
import torch


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
    
    return block_lat, throughput


def main():
    models = [
        "switch-base-8",
        "switch-base-16",
        "switch-base-32",
        "switch-base-64",
        "switch-base-128",
        "switch-base-256",
        "switch-large-128",
    ]
    batch_sizes = [
        1,
        2,
        4,
        8,
        16,
        32,
    ]
    methods = ["GPU-only", "Pre-gated", "DeepSpeed", "SE-MoE"]

    cpp_config = configparser.ConfigParser()
    cpp_config.read("/workspace/FasterTransformer/cpp_config.ini")

    block_lats = {"bs": batch_sizes}
    throughputs = {"bs": batch_sizes}
    hardware_infos = {
        "CPU": [get_cpu_info()["brand_raw"]] * len(batch_sizes),
        "RAM (GB)": [int(psutil.virtual_memory().total / 1024 / 1024 / 1024)] * len(batch_sizes),
        "GPU": [torch.cuda.get_device_name()] * len(batch_sizes),
    }
    block_lats.update(hardware_infos)
    throughputs.update(hardware_infos)

    for method in methods:
        for model in models:
            _block_lats = []
            _throughputs = []

            for batch_size in batch_sizes:
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

                block_lat, throughput = parse_output(result.stdout)
                print(f"BLK AVG: {block_lat} ms, throughput: {throughput} tokens/sec")
                _block_lats.append(block_lat)
                _throughputs.append(throughput)

            block_lats["{}/{}".format(model, method)] = _block_lats
            throughputs["{}/{}".format(model, method)] = _throughputs

            # Generate CSV after each model and method runned
            df = pd.DataFrame.from_dict(block_lats)
            df.to_csv("block_lats.csv", index=False)

            df = pd.DataFrame.from_dict(throughputs)
            df.to_csv("throughputs.csv", index=False)


if __name__ == "__main__":
    main()
