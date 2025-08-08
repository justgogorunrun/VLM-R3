#!/bin/bash

# The latest vllm==0.7.3 is required for this script: pip3 install vllm==0.7.3
# The latest transformers is required too, install by: pip install git+https://github.com/huggingface/transformers.git@a40f1ac602fe900281722254c52ce3773f28eb0e

TIMESTAMP=$(date +"%m%d_%H%M")

export API_KEY="your_api_key_here"  # Update with your actual API key
export API_BASE_URL="your_api_base_url" # for example "https://dashscope.aliyuncs.com/compatible-mode/v1"
export DEBUG_MODE="true"
export LOG_PATH="./log/vllm_run_acc_${TIMESTAMP}.txt"
export LOG_PATH_FORMAT="./log/vllm_run_format_${TIMESTAMP}.txt"
export LOG_PATH_REPEAT="./log/vllm_run_repeat_${TIMESTAMP}.txt"

model_path="path/to/your/model"  # Update this to your model path
dataset_path="path/to/your/dataset.json"  # Update this to your dataset path
output_dir="./output/rgrpo_vllm_Qwen2.5-VL-${TIMESTAMP}/"
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi
RUN_NAME="rgrpo_vllm_Qwen2.5-VL-${TIMESTAMP}"

DS_CONFIG="./src/r1-v/local_scripts/zero1_no_optimizer.json"  # Note that other zero setting would meet bugs related to vllm at current stage.
# NOTE: you are expected to use X + 1 cards for X training proc and 1 vLLM proc 
# e.g., the visible devices should be 0,1,2,3,4 for 5 cards, and  --nproc_per_node="4"

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" torchrun \
    --nproc_per_node="5" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/r1-v/src/open_r1/rgrpo.py \
    --use_vllm true \
    --output_dir $output_dir \
    --model_name_or_path $model_path \
    --dataset_name $dataset_path \
    --max_prompt_length 5120 \
    --max_completion_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --bf16 true \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --min_pixels 3136 \
    --max_pixels 3211264 \
    --num_train_epochs 2 \
    --run_name ${RUN_NAME} \
    --save_steps 50 \
    --save_total_limit 3 \
    --report_to tensorboard \
    --temperature 1.0 \
    --num_generations 5 \
    --vllm_device "cuda:5" \
    --vllm_gpu_memory_utilization 0.9 \
    --deepspeed ${DS_CONFIG} \
    --reward_funcs accuracy format repeat length \
    --save_only_model True \
    --beta 0.0
