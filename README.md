# VLM-R3: Region Recognition, Reasoning, and Refinement for Enhanced Multimodal Chain-of-Thought
This is the official code for [VLM-R $^3$: Region Recognition, Reasoning, and Refinement for Enhanced Multimodal Chain-of-Thought](https://arxiv.org/abs/2505.16192).

## Setup

```bash
conda create -n vlmr3 python=3.11 
conda activate vlmr3
bash setup.sh
```

## Training
### SFT
```bash
cd qwen_vl_finetune
bash sft_7b.sh
```
### GRPO

```bash
bash run_rgrpo_vllm.sh
```

- Note: Please Keep the per_device_train_batch_size to 1 for now, as there is a revealed bug about batched training.



## Resources
### Model
- ModelScope: [VLM-R3-7b-rl-v1](https://www.modelscope.cn/models/LittleHenry/VLM-R3-7b-rl-v1)
- Huggingface: 

### Dataset
coming soon

## Evaluation
```bash
cd src/eval
python test_vstar_r3.py

python test_mmvp_r3.py

python test_cvbench_r3.py
```

For other benchmarks such as HR-Bench, MME-RealWorld, and ScienceQA, we use [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). To run these evaluations, please follow these steps:

1.  Replace the `lmms_eval/models/vllm.py` file in your `lmms-eval` installation with our provided `src/eval/vllm_r3_lmm-eval.py`.
2.  When launching the evaluation script, choose `vllm` as the model for evaluation.


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



