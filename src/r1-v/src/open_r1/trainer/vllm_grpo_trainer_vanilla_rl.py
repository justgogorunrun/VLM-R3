# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tqdm import tqdm
from PIL import Image
import json
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info, smart_resize
import torch
import os  
import re  
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from vllm import LLM, SamplingParams
vllm_available = True
import random
random.seed(42) 
import torch.multiprocessing as mp


def get_img_crop(img, bbox):
    """
    Crop the image based on the bounding box coordinates.
    bbox: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    # Ensure the coordinates are within the image bounds
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    # Handle different image types
    if isinstance(img, np.ndarray):  # NumPy array (OpenCV image)
        x2 = min(img.shape[1]-1, int(x2))
        y2 = min(img.shape[0]-1, int(y2))
        if x1 >= x2 or y1 >= y2:
            print(f"## DEBUG: Invalid bbox: {bbox}")
            return None
        return img[y1:y2+1, x1:x2+1]
    else:  # Assume PIL Image
        width, height = img.size
        x2 = min(width-1, int(x2))
        y2 = min(height-1, int(y2))
        if x1 >= x2 or y1 >= y2:
            print(f"## DEBUG: Invalid bbox: {bbox}")
            return None
        return img.crop((x1, y1, x2+1, y2+1))  # PIL crop takes (left, top, right, bottom)

def bbox_transform(bbox, w, h, w1, h1):
    """
    Transform bbox coordinates from original image size to new image size.
    bbox: [x1, y1, x2, y2]
    h, w: original image height and width
    h1, w1: new image height and width
    """
    x1, y1, x2, y2 = bbox
    x1 = int(x1 * w1 / w)
    x2 = int(x2 * w1 / w)
    y1 = int(y1 * h1 / h)
    y2 = int(y2 * h1 / h)
    return [x1, y1, x2, y2]


class AgentVLMVLLM:
    """
    A class that encapsulates the multimodal interleaved reasoning process
    and manages multi-turn dialogues.
    """
    
    def __init__(self, model=None, 
                    processor=None,
                    temp_dir="./agent_crops_tmp",
                    device="cuda",
                    min_pixels=4*28*28,
                    max_pixels=2048*28*28,
                    temperature=1.0,
                    max_tokens_once=2048,
                    crop_min_pixels=4*28*28,
                    crop_max_pixels=1024*28*28):
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)
        self.sample_id = 0
        self.cnt_tmp_img = 0
        
        self.prompt = """\nYou need to first think about the reasoning process in your mind, and then provide the answer. When thinking, you should call the "crop" tool (format: {"bbox_2d": [x1, y1, x2, y2]}) to focus on the key areas in the image. The reasoning process and the answer are included in the <think> </think> and <answer> </answer> tags respectively."""
        # Constants
        self.delta_img = '<|vision_start|><|image_pad|><|vision_end|>'
        self.img_crop_path = os.path.join(self.temp_dir, "{device}_sample_{sample}_crop_{cnt}.png")
        
        # Stopping criteria for generation
        # self.stop_pattern = r'```json\s*(.*?)\s*```'
        self.stop_pattern = r'\{.*?\}'

        # Initialize models
        self.model = model
        self.processor = processor
        self.temperature = temperature
        self.max_tokens_once = max_tokens_once
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens_once,
            stop=["]}","]\n}","] }", "</answer>"],
        )
        self.device = device
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.crop_min_pixels = crop_min_pixels
        self.crop_max_pixels = crop_max_pixels

        for filename in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
                continue    
    
    def process(self, image_path, question, max_iterations=8, sample_id=None):
        """
        Process an image with a question using multimodal interleaved reasoning.
        """
        self.cnt_tmp_img = 0
        if sample_id is not None:
            self.sample_id = sample_id
        else:
            self.sample_id += 1
        
        img_crop_list = []
        response_list = []
        
        # Load original image 
        raw_img = Image.open(image_path).convert("RGB")
        width, height = raw_img.size
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=28,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,  
        )
        real_input_img = raw_img.resize((resized_width, resized_height))
        question += self.prompt
        
        # Initialize messages
        text_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question},
                ],
            }
        ]  
        img_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                ],
            }
        ]
        
        full_response = ""
        crop_flag = False
        
        prompt_ids_len = None

        text = self.processor.apply_chat_template(
            text_messages, tokenize=False, add_generation_prompt=True if len(text_messages)==1 else False
        )
        if text.endswith("<|im_end|>\n"):
            text = text[:-len("<|im_end|>\n")] + "\n"
        
        # Process inputs
        image_inputs, video_inputs = process_vision_info(img_messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        if prompt_ids_len is None:
            prompt_ids_len = len(inputs['input_ids'][0])

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        llm_input = {
            "prompt": text,
            "multi_modal_data": mm_data,
        }
        generated_ids = inputs['input_ids'][0].cpu().tolist()

        # if len(inputs['input_ids'][0]) >= 4096:
        #     print(f"##DEBUG:  Exceeded maximum input length")
        #     break
        
        # Generate next part of response
        outputs = self.model.generate([llm_input], sampling_params=self.sampling_params)
        generated_ids = generated_ids + list(outputs[0].outputs[0].token_ids)
        
        # Extract and decode the newly generated content
        generated_ids_trimmed = outputs[0].outputs[0].token_ids
        generated_text = outputs[0].outputs[0].text 
        if "<answer>" in generated_text and "</answer>" not in generated_text:
            generated_text += "</answer>\n"
        # print("##DEBUG: Generated text:", generated_text)
        
        # Check if we have a final answer
        if generated_ids_trimmed[-1] == self.processor.tokenizer.eos_token_id or "</answer>" in generated_text:
            # Add to the full response
            full_response += generated_text
            response_list.append(generated_text)
            if len(text_messages) == 1:
                text_messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": response_list[-1]},
                        ],
                    },
                )
            else:
                text_messages[-1]['content'].append(
                    {"type": "text", "text": response_list[-1]},
                )


        return response_list, img_crop_list, full_response, img_messages, text_messages


import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from accelerate.utils.other import is_compiled_module
from accelerate.utils import broadcast_object_list, gather, gather_object
import torch
import torch.utils.data
import transformers
import warnings
from unittest.mock import patch
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)
from trl.import_utils import is_vllm_available

from trl.models import (
    create_reference_model,
    prepare_deepspeed,
    unwrap_model_for_generation,
)
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad
from trl import GRPOTrainer

import copy

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams


if is_wandb_available():
    import swanlab as wandb
import torch.nn as nn
from torch.utils.data import Sampler

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


from typing import Any, Callable, Optional, Union, Sized
class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility.
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count

class Qwen2VLRGRPOVLLMTrainer(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        # qwen2-vl related params
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):

        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if (
                isinstance(torch_dtype, torch.dtype)
                or torch_dtype == "auto"
                or torch_dtype is None
            ):
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False
                if args.gradient_checkpointing
                else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model, **model_init_kwargs
                )
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model, **model_init_kwargs
                )
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(
                    model, **model_init_kwargs
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        self.vision_modules_keywords = ["visual"]# by hyr
        freeze_vision_modules = True  
        if freeze_vision_modules:  
            print("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.vision_modules_keywords):
                    p.requires_grad = False
            print("Freezing MLP modules...")
            for n, p in model.visual.merger.named_parameters(): 
                p.requires_grad = False

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
            elif "Qwen2.5-VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(
                    model_id, **model_init_kwargs
                )
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        # Processing class
        if processing_class is None:
            if "Qwen2" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                # if "Qwen" in model_id:
                processing_class.image_processor.max_pixels = max_pixels
                processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(
                    model.config._name_or_path, padding_side="left"
                )
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError(
                    "The number of reward processing classes must match the number of reward functions."
                )

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(
                        reward_func.config._name_or_path
                    )
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = (
                        reward_processing_class.eos_token
                    )
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = (
            args.max_completion_length
        )  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1,  # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)
        self.use_vllm = args.use_vllm

        # 当前设为定值1
        self.num_iterations = 1
        
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False
        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [
            n_gen
            for n_gen in range(2, global_batch_size + 1)
            if (global_batch_size) % n_gen == 0
        ]

        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [
                n_gen
                for n_gen in range(2, global_batch_size + 1)
                if (global_batch_size) % n_gen == 0
            ]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                if vllm_device == "auto":
                    vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
                # Check that the requested device is available
                # if (
                #     vllm_device.split(":")[0] == "cuda"
                #     and int(vllm_device.split(":")[1]) >= torch.cuda.device_count()
                # ):
                #     raise ValueError(
                #         f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                #         "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                #         "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                #         f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                #     )
                # Check that the requested device is not also used for training
                if vllm_device in {
                    f"cuda:{idx}" for idx in range(self.accelerator.num_processes)
                }:
                    warnings.warn(
                        f"The requested device {vllm_device} is also used for training. This may lead to unexpected "
                        "behavior. It is recommended to use a dedicated device for vLLM."
                    )
                # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                # setting (profiling_patch).
                world_size_patch = patch(
                    "torch.distributed.get_world_size", return_value=1
                )
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
                    return_value=None,
                )
                with world_size_patch, profiling_patch:
                    print("vllm is running on: ", vllm_device)
                    self.llm = LLM(
                        model=model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        dtype=torch.bfloat16,
                        limit_mm_per_prompt={"image": 16, "video": 10},
                        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # This is particularly useful here because we generate completions from the same prompts.
                        enable_prefix_caching=True,
                        enforce_eager=True,
                        # Ensure that training and inference use the same processor for images. ## ！！！
                        mm_processor_kwargs=(
                            {
                                "max_pixels": max_pixels,
                                "min_pixels": min_pixels,
                            }
                        ),
                        max_model_len=args.max_prompt_length + args.max_completion_length*8,
                    )
                self.sampling_params = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=self.max_completion_length,
                )

            self._last_loaded_step = (
                0  # tag to avoid useless loading during grad accumulation
            )

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            raise ValueError(
                "Qwen2VLGRPOVLLMTrainer only supports vllm generation, please set --use_vllm True"
            )

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(
                    reward_func, evaluation_mode=True
                )
        
        self.epsilon_low = 0.2
        self.epsilon_high = 0.28

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # We need a custom sampler that samples the same prompt multiple times
    # def _get_train_sampler(self):
    #     return RepeatRandomSampler(self.train_dataset, self.num_generations)
    def _get_train_sampler(self) -> Sampler:
        """Returns a sampler that ensures proper data sampling for GRPO training."""
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=42,
        )

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(
        self,
        model,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        logits_to_keep,
    ):
        pixel_values = pixel_values.to(model.device)
        image_grid_thw = image_grid_thw.to(device=model.device)
        logits = model(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        ).logits  # (B, L, V)
        logits = logits[
            :, :-1, :
        ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[
            :, -logits_to_keep:
        ]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        logits = logits[:, -logits_to_keep:]
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(
                log_probs, dim=1, index=input_ids_row.unsqueeze(1)
            ).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]   # x["prompt"]= [{'text': None, 'type': 'image'}, {'text': 'Please provide the bounding box coordinate of the region this sentence describes: ora...s. Output the final answer in JSON format.', 'type': 'text'}]
    
        img_path = [input["image_path"][0] for input in inputs]  
        problem = [input['problem'] for input in inputs]
        # print(f"## DEBUG: problem: {problem}")

        # if self.max_prompt_length is not None:  # not support yet
        #     prompt_ids = prompt_ids[:, -self.max_prompt_length :]
        #     prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                with unwrap_model_for_generation(
                    self.model,
                    self.accelerator,
                    gather_deepspeed3_params=False,  # TODO: fix this, self.args.ds3_gather_for_generation,
                ) as unwrapped_model:
                    if is_compiled_module(unwrapped_model):
                        state_dict = unwrapped_model._orig_mod.state_dict()
                    else:
                        state_dict = unwrapped_model.state_dict()
                if self.accelerator.is_main_process:
                    llm_model = (
                        self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    )
                    llm_model.load_weights(state_dict.items())
                    print("## DEBUG: update vLLM !!!")
                self._last_loaded_step = self.state.global_step
                
            self.accelerator.wait_for_everyone()
            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_img_path = gather_object(img_path)  
            # print("## DEBUG: all_img_path\n", all_img_path)
            all_problem = gather_object(problem)

            if self.accelerator.is_main_process:
                res = []
                agent = AgentVLMVLLM(self.llm, self.processing_class, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
                # agent.set_model(self.llm, self.processing_class)
                for img_path, problem in zip(all_img_path, all_problem):
                    _, _, _, img_messages, text_messages = agent.process(img_path, problem)
                    res.append({
                        "img_messages": img_messages,
                        "text_messages": text_messages,
                    })
            else:
                res = [None] * len(all_img_path)

            # print("## DEBUG: res\n",res)
            res = broadcast_object_list(res, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(inputs),
                (self.accelerator.process_index + 1) * len(inputs),
            )

            res = res[process_slice]
            torch.cuda.empty_cache()  
            res = res[0]  # only support per_device_batch_size==1
            self.accelerator.wait_for_everyone()

            # Convert numpy arrays back to tensors and place on the correct device]
            text_messages = res['text_messages']
            img_messages = res['img_messages']
            image_inputs, video_inputs = process_vision_info(img_messages)
            all_text = self.processing_class.apply_chat_template(
                text_messages, tokenize=False, add_generation_prompt=False # False!!!
            )
            mutimodal_inputs = self.processing_class(
                text=[all_text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            ).to(device)
            prompt_completion_ids = mutimodal_inputs["input_ids"]  # (B==1, P+C)

            prompt_text = self.processing_class.apply_chat_template(
                text_messages[:1], tokenize=False, add_generation_prompt=True
            )
            prompt_ids = self.processing_class(
                text=[prompt_text],
                images=image_inputs[:1],
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            ).to(device)["input_ids"]  # (B==1, P)
            prompt_length = prompt_ids.shape[1]  # (B==1, P)
            completion_ids = prompt_completion_ids[:, prompt_length:]  # (B==1, L)
            # prompt_ids = prompt_completion_ids[:, :prompt_length]  # (B==1, L)

            # self.accelerator.wait_for_everyone()
            print(f"## DEBUG: generate_returned_result.shape: {prompt_completion_ids.shape}")
            prompt_mask = mutimodal_inputs["attention_mask"][:, :prompt_length]  # (B, P)

            completion_mask = mutimodal_inputs["attention_mask"][:, prompt_length:]  # (B, C) # by hyr


        # # below are the same with yifan's code
        # # Mask everything after the first EOS token 
        # is_eos = completion_ids == self.processing_class.eos_token_id
        # device = self.accelerator.device
        # eos_idx = torch.full(
        #     (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        # )
        # eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        # sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
        #     is_eos.size(0), -1
        # )
        # completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()


        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        # pixel_values = prompt_inputs["pixel_values"].repeat_interleave(
        #     self.num_generations, dim=0
        # )

        pixel_values = mutimodal_inputs["pixel_values"]
        # [None].repeat_interleave(self.num_generations, dim=0)
        # pixel_values = pixel_values.view(-1, pixel_values.shape[-1])

        image_grid_thw = mutimodal_inputs["image_grid_thw"]

        logits_to_keep = completion_ids.size(1)

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    pixel_values,
                    image_grid_thw,
                    logits_to_keep,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        pixel_values,
                        image_grid_thw,
                        logits_to_keep,
                    )

        # Decode the generated completions
        completions = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = [
                [{"role": "assistant", "content": completion}]
                for completion in completions
            ]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device) # [per_device_train_batch_size, types of reward_funcs]
        acc_r = None
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):  
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else: 
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # No need to duplicate prompts as we're not generating multiple completions per prompt
                        # reward_kwargs[key].extend([example[key]] * self.num_generations)
                        reward_kwargs[key].extend([example[key]])
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                if "accuracy" in reward_func.__name__:
                    acc_r = output_reward_func
                if "repeat" in reward_func.__name__:
                    print(f"## DEBUG: acc_r: {acc_r}")
                    print(f"## DEBUG: repeat_r: {output_reward_func}")
                    output_reward_func = [
                        r if a == 1.0 else 0.0 for r, a in zip(output_reward_func, acc_r)
                    ]
                    print(f"## DEBUG: repeat_r_new: {output_reward_func}")
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        # for i, reward_func in enumerate(self.reward_funcs):
              
        # print(f"## DEBUG: rewards_per_func.shape: {rewards_per_func.shape}") # [per_device_train_batch_size, num_reward_funcs]
        # Gather rewards across processes
        rewards_per_func = self.accelerator.gather(rewards_per_func) # [per_device_train_batch_size*num_processes, num_reward_funcs]
        
        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)   # [per_device_train_batch_size*num_processes, 1]
        
        # Compute grouped-wise rewards
        # Each group consists of num_generations completions for the same prompt
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        print(f"## DEBUG: advantages min: {advantages.min()}, max: {advantages.max()}")

        # Get only the local slice of advantages
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # print(f"## DEBUG: advantages.shape: {advantages.shape}") # [1]

        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(
                reward_per_func[i].item()
            )

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model
        

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(
            model,
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            logits_to_keep,
        )

        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps)
            - 1
        )
        advantages = inputs["advantages"]
        
        # # x - x.detach() allows for preserving gradients from x
        # per_token_loss = torch.exp(
        #     per_token_logps - per_token_logps.detach()
        # ) * advantages.unsqueeze(1)
        # per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its computation
        # and use per_token_logps.detach() instead

        # old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        old_per_token_logps = per_token_logps.detach()

        # Compute the policy ratio and clipped version
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # mask image tokens in response  add by hyr
        special_tokens = torch.tensor([151655, 151652, 151653], device=completion_ids.device)
        special_tokens_mask = torch.zeros_like(completion_ids, dtype=torch.bool)
        for token in special_tokens:
            special_tokens_mask = special_tokens_mask | (completion_ids == token)
        completion_mask = completion_mask.clone()  
        completion_mask[special_tokens_mask] = 0
        
        # Add KL penalty if beta > 0
        if self.beta > 0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            per_token_loss = per_token_loss + self.beta * per_token_kl

            # Log KL divergence
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # Compute final loss
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log clip ratio
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

        completion_length = (
            self.accelerator.gather_for_metrics(completion_mask.sum(1))
            .float()
            .mean()
            .item()
        )
        self._metrics["completion_length"].append(completion_length)

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics.items()
        }  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()