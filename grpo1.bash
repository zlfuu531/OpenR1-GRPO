#!/bin/bash

# 激活 Conda 环境
#conda activate openr1

# 设置使用的 GPU
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# 运行训练命令
accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml