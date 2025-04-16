
conda activate llama_factory

CUDA_VISIBLE_DEVICES=0,1 vllm serve /vepfs-d-data/q-caiyue/zlf/amodel/lora_old_v1_60 --host 0.0.0.0 --port 8001 --gpu-memory-utilization 0.9 --max-model-len 32768 --tensor-parallel-size 2 --served-model-name "LORA_OLD" 

CUDA_VISIBLE_DEVICES=2,3 vllm serve /vepfs-d-data/q-caiyue/zlf/amodel/lora_new_v1_62 --host 0.0.0.0 --port 8002 --gpu-memory-utilization 0.9 --max-model-len 32768 --tensor-parallel-size 2 --served-model-name "LORA_NEW" 

CUDA_VISIBLE_DEVICES=4,5 vllm serve /vepfs-d-data/q-caiyue/zlf/open-r1/lora_old_v1_60/checkpoint-50 --host 0.0.0.0 --port 8003 --gpu-memory-utilization 0.9 --max-model-len 32768 --tensor-parallel-size 2 --served-model-name "GRPOV1" 

CUDA_VISIBLE_DEVICES=6,7 vllm serve /vepfs-d-data/q-caiyue/zlf/open-r1/lora_old_60_grpov1check50_v4/checkpoint-40 --host 0.0.0.0 --port 8004 --gpu-memory-utilization 0.9 --max-model-len 32768 --tensor-parallel-size 2 --served-model-name "GRPOV2"



            "http://0.0.0.0:8001/v1": "LORA_48_V1_50",
            "http://0.0.0.0:8002/v1": "LORA_48_V1_60",
            "http://0.0.0.0:8003/v1": "LORA_48_V1_66",
            "http://0.0.0.0:8004/v1": "LORA_48_V2_60",
            "http://0.0.0.0:8005/v1": "LORA_48_V2_70",
            "http://0.0.0.0:8006/v1": "LORA_48_V2_80",
            "http://0.0.0.0:8007/v1": "LORA_48_V3_90",
            "http://0.0.0.0:8008/v1": "LORA_48_V3_130",
            "http://0.0.0.0:8009/v1": "LORA_48_V4_70",
            "http://0.0.0.0:8010/v1": "LORA_48_V4_80",
            "http://0.0.0.0:8011/v1": "LORA_48_V4_90",
            "http://0.0.0.0:8012/v1": "LORA_48_V4_110",
            "http://0.0.0.0:8013/v1": "LORA_48_V4_100"

# ====== v1 系列 ======
CUDA_VISIBLE_DEVICES=0,1 vllm serve /vepfs-d-data/q-caiyue/zlf/amodel/lora_48_v1_50 --host 0.0.0.0 --port 8001 --gpu-memory-utilization 0.9 --max-model-len 32768 --tensor-parallel-size 2 --served-model-name "LORA_48_V1_50"

CUDA_VISIBLE_DEVICES=2,3 vllm serve /vepfs-d-data/q-caiyue/zlf/amodel/lora_48_v1_60 --host 0.0.0.0 --port 8002 --gpu-memory-utilization 0.9 --max-model-len 32768 --tensor-parallel-size 2 --served-model-name "LORA_48_V1_60"

CUDA_VISIBLE_DEVICES=4,5 vllm serve /vepfs-d-data/q-caiyue/zlf/amodel/lora_48_v1_66 --host 0.0.0.0 --port 8003 --gpu-memory-utilization 0.9 --max-model-len 32768 --tensor-parallel-size 2 --served-model-name "LORA_48_V1_66"

# ====== v2 系列 ======
CUDA_VISIBLE_DEVICES=6,7 vllm serve /vepfs-d-data/q-caiyue/zlf/amodel/lora_48_v2_60 --host 0.0.0.0 --port 8004 --gpu-memory-utilization 0.9 --max-model-len 32768 --tensor-parallel-size 2 --served-model-name "LORA_48_V2_60"

CUDA_VISIBLE_DEVICES=0,1 vllm serve /vepfs-d-data/q-caiyue/zlf/amodel/lora_48_v2_70 --host 0.0.0.0 --port 8005 --gpu-memory-utilization 0.9 --max-model-len 32768 --tensor-parallel-size 2 --served-model-name "LORA_48_V2_70"

CUDA_VISIBLE_DEVICES=2,3 vllm serve /vepfs-d-data/q-caiyue/zlf/amodel/lora_48_v2_80 --host 0.0.0.0 --port 8006 --gpu-memory-utilization 0.9 --max-model-len 32768 --tensor-parallel-size 2 --served-model-name "LORA_48_V2_80"

# ====== v3 系列 ======
CUDA_VISIBLE_DEVICES=4,5 vllm serve /vepfs-d-data/q-caiyue/zlf/amodel/lora_48_v3_90 --host 0.0.0.0 --port 8007 --gpu-memory-utilization 0.9 --max-model-len 32768 --tensor-parallel-size 2 --served-model-name "LORA_48_V3_90"

CUDA_VISIBLE_DEVICES=6,7 vllm serve /vepfs-d-data/q-caiyue/zlf/amodel/lora_48_v3_130 --host 0.0.0.0 --port 8008 --gpu-memory-utilization 0.9 --max-model-len 32768 --tensor-parallel-size 2 --served-model-name "LORA_48_V3_130"

# ====== v4 系列 ======
CUDA_VISIBLE_DEVICES=0,1 vllm serve /vepfs-d-data/q-caiyue/zlf/amodel/lora_48_v4_70 --host 0.0.0.0 --port 8009 --gpu-memory-utilization 0.9 --max-model-len 32768 --tensor-parallel-size 2 --served-model-name "LORA_48_V4_70"

CUDA_VISIBLE_DEVICES=2,3 vllm serve /vepfs-d-data/q-caiyue/zlf/amodel/lora_48_v4_80 --host 0.0.0.0 --port 8010 --gpu-memory-utilization 0.9 --max-model-len 32768 --tensor-parallel-size 2 --served-model-name "LORA_48_V4_80"

CUDA_VISIBLE_DEVICES=4,5 vllm serve /vepfs-d-data/q-caiyue/zlf/amodel/lora_48_v4_90 --host 0.0.0.0 --port 8011 --gpu-memory-utilization 0.9 --max-model-len 32768 --tensor-parallel-size 2 --served-model-name "LORA_48_V4_90"

CUDA_VISIBLE_DEVICES=6,7 vllm serve /vepfs-d-data/q-caiyue/zlf/amodel/lora_48_v4_110 --host 0.0.0.0 --port 8012 --gpu-memory-utilization 0.9 --max-model-len 32768 --tensor-parallel-size 2 --served-model-name "LORA_48_V4_110"

# ====== y4 系列 ======
CUDA_VISIBLE_DEVICES=0,1 vllm serve /vepfs-d-data/q-caiyue/zlf/amodel/lora_48_v4_100 --host 0.0.0.0 --port 8013 --gpu-memory-utilization 0.9 --max-model-len 32768 --tensor-parallel-size 2 --served-model-name "LORA_48_V4_100"





            "http://0.0.0.0:8014/v1": "LORA_48_V1_20",
            "http://0.0.0.0:8015/v1": "LORA_48_V1_30",
            "http://0.0.0.0:8016/v1": "LORA_48_V1_40",
            "http://0.0.0.0:8017/v1": "LORA_48_V1_50",
# ====== v1 系列新增检查点（8卡版）======

# checkpoint-20（使用GPU 0,1）
CUDA_VISIBLE_DEVICES=0,1 vllm serve /vepfs-d-data/q-caiyue/zlf/open-r1/grpo_48_all_v1/checkpoint-20 --host 0.0.0.0 --port 8014 --gpu-memory-utilization 0.9 --max-model-len 32768 --tensor-parallel-size 2 --served-model-name "LORA_48_V1_20"

# checkpoint-30（使用GPU 2,3）
CUDA_VISIBLE_DEVICES=2,3 vllm serve /vepfs-d-data/q-caiyue/zlf/open-r1/grpo_48_all_v1/checkpoint-30 --host 0.0.0.0 --port 8015 --gpu-memory-utilization 0.9 --max-model-len 32768 --tensor-parallel-size 2 --served-model-name "LORA_48_V1_30"

# checkpoint-40（使用GPU 4,5）
CUDA_VISIBLE_DEVICES=4,5 vllm serve /vepfs-d-data/q-caiyue/zlf/open-r1/grpo_48_all_v1/checkpoint-40 --host 0.0.0.0 --port 8016 --gpu-memory-utilization 0.9 --max-model-len 32768 --tensor-parallel-size 2 --served-model-name "LORA_48_V1_40"

# checkpoint-50（使用GPU 6,7）
CUDA_VISIBLE_DEVICES=6,7 vllm serve /vepfs-d-data/q-caiyue/zlf/open-r1/grpo_48_all_v1/checkpoint-50 --host 0.0.0.0 --port 8017 --gpu-memory-utilization 0.9 --max-model-len 32768 --tensor-parallel-size 2 --served-model-name "LORA_48_V1_50"  





# ====== v3 系列新增检查点（8卡版）======
            "http://0.0.0.0:8018/v1": "LORA_48_V3_20",
            "http://0.0.0.0:8019/v1": "LORA_48_V3_30",
            "http://0.0.0.0:8020/v1": "LORA_48_V3_40",
            "http://0.0.0.0:8021/v1": "LORA_48_V3_50",


CUDA_VISIBLE_DEVICES=0,1 vllm serve /vepfs-d-data/q-caiyue/zlf/open-r1/grpo_48_all_v3/checkpoint-20 --host 0.0.0.0 --port 8018 --gpu-memory-utilization 0.9 --max-model-len 32768 --tensor-parallel-size 2 --served-model-name "LORA_48_V3_20"

CUDA_VISIBLE_DEVICES=2,3 vllm serve /vepfs-d-data/q-caiyue/zlf/open-r1/grpo_48_all_v3/checkpoint-30 --host 0.0.0.0 --port 8019 --gpu-memory-utilization 0.9 --max-model-len 32768 --tensor-parallel-size 2 --served-model-name "LORA_48_V3_30"

CUDA_VISIBLE_DEVICES=4,5 vllm serve /vepfs-d-data/q-caiyue/zlf/open-r1/grpo_48_all_v3/checkpoint-40 --host 0.0.0.0 --port 8020 --gpu-memory-utilization 0.9 --max-model-len 32768 --tensor-parallel-size 2 --served-model-name "LORA_48_V3_40"

CUDA_VISIBLE_DEVICES=6,7 vllm serve /vepfs-d-data/q-caiyue/zlf/open-r1/grpo_48_all_v3/checkpoint-50 --host 0.0.0.0 --port 8021 --gpu-memory-utilization 0.9 --max-model-len 32768 --tensor-parallel-size 2 --served-model-name "LORA_48_V3_50"







