export CUDA_VISIBLE_DEVICES=1

python /workspace/FasterTransformer/examples/pytorch/t5/perf_benchmark.py \
        --batch_size 2 \
        --beam_width 4 \
        --seq_len 256 \
        --data_type fp32 \
        --test_time 3 \
        --sampling_topk 1 \
        --model_type Megatron-DeepSpeed \
        --ckpt_path /data/ft/switch-base-128-orig/ \
        --model t5-base \
        --duration 0 \
        --iterations 4
