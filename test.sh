accelerate launch \
    --config_file recipes/accelerate_configs/zero3.yaml \     
    --num_processes 3 \ 
    --main_process_port 29501 \    
    src/open_r1/grpo.py \       #算法代码文件选择grpo or ppo
    --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml  #参数设置文件    