#!/bin/bash

# 激活环境
#source ~/anaconda3/etc/profile.d/conda.sh
#conda activate openr1

# 设置 Swanlab 环境变量（如果需要）
#export NCCL_P2P_LEVEL=NVL
#export SWANLAB_PROJECT="grpo_project"

# 环境变量设置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 启动训练并将日志实时保存到文件

accelerate launch \
    --config_file /vepfs-d-data/q-caiyue/zlf/open-r1/recipes/accelerate_configs/zero2.yaml \
    --num_processes 7 \
    --main_process_port 23333 \
    /vepfs-d-data/q-caiyue/zlf/open-r1/src/open_r1/grpo_copy.py \
    --config /vepfs-d-data/q-caiyue/zlf/open-r1/recipes/Qwen2.5-7B-Instruct/grpo/config_demov5.yaml


# 训练结束后取消激活环境
#conda deactivate 