# VLM-R3
This is the official code for [VLM-R $^3$: Region Recognition, Reasoning, and Refinement for Enhanced Multimodal Chain-of-Thought](https://arxiv.org/abs/2505.16192).

## Setup

```bash
conda create -n vlmr3 python=3.11 
conda activate vlmr3
bash setup.sh
```

## Training
### SFT
Download the [SFT dataset](https://www.modelscope.cn/datasets/LittleHenry/VLM-R3-sft-rl-v1/resolve/master/dataset.zip) and put it under `qwen_vl_finetune/dataset/`.
```bash
cd qwen_vl_finetune
bash sft_7b.sh
```
### GRPO
We use Qwen2.5 72B API as judge model. Please set:
```bash
export API_KEY="your_api_key_here" 
export API_BASE_URL="your_api_base_url" # "https://dashscope.aliyuncs.com/compatible-mode/v1" for example
```
Download the [RL dataset](https://www.modelscope.cn/datasets/LittleHenry/VLM-R3-sft-rl-v1/resolve/master/data.zip) and put it under `data/`.
```bash
bash run_rgrpo_vllm.sh
```

- Note: Please Keep the per_device_train_batch_size to 1 for now, as there is a revealed bug about batched training.


## Resources
### Model
- Huggingface: [VLM-R3-7b-rl-v1](https://huggingface.co/lh-hyr/VLM-R3-7b-rl-v1)

## Evaluation
For the following benchmarks, you can use our provided standalone scripts.
```bash
cd src/eval
```
- [V*](https://huggingface.co/datasets/craigwu/vstar_bench)
```bash
python test_vstar_r3.py
```
- [MMVP](https://huggingface.co/datasets/MMVP/MMVP)
```bash
python test_mmvp_r3.py
```
- [CVBench](https://huggingface.co/datasets/nyu-visionx/CV-Bench)
```bash
python test_cvbench_r3.py
```
  
For other benchmarks like ScienceQA, HR-Bench, and MME-RealWorld, we use [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). To run these evaluations, please follow these steps:

1.  Replace the `lmms_eval/models/vllm.py` file in your `lmms-eval` code base with our provided `src/eval/vllm.py`.
2.  When launching the evaluation script, choose `vllm` as the model for evaluation.
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m lmms_eval \
    --model vllm \
    --model_args model_version=path/to/ckpt,tensor_parallel_size=1,gpu_memory_utilization=0.9 \
    --tasks mmerealworld_lite \
    --batch_size 16 \
    --log_samples \
    --log_samples_suffix vllm \
    --output_path ./logs_rl
```

## Acknowledgements

Thanks to [R1-V](https://github.com/StarsfieldAI/R1-V)(our initial code base), [QwenVL](https://github.com/QwenLM/Qwen2.5-VL)(our base model) and [Open-R1-Multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal).


## Citation

```bib
@article{jiang2025vlm,
  title={VLM-R $\^{} 3$: Region Recognition, Reasoning, and Refinement for Enhanced Multimodal Chain-of-Thought},
  author={Jiang, Chaoya and Heng, Yongrui and Ye, Wei and Yang, Han and Xu, Haiyang and Yan, Ming and Zhang, Ji and Huang, Fei and Zhang, Shikun},
  journal={arXiv preprint arXiv:2505.16192},
  year={2025}
}
```



