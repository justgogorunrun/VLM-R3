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

    
    def _extract_fields(self, text):
        """
        Extracts JSON-like content from the text.
        Args:
            text: Input text containing JSON-like content
        Returns:
            Parsed JSON data or None if no valid JSON found
        """
        pattern = r'\{.*?\}'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            last_match = matches[-1]
        else:
            return None
        json_content = last_match
        try:
            json_data = json.loads(json_content)
        except json.JSONDecodeError:
            print("Invalid JSON format")
            return None
        return json_data

    
    def _crop_and_save_region(self, img, raw_img, bbox, scale_factor=None):
        """
        Crop and save a region from the raw image.
        Args:
            img: Processed image for the VL model
            raw_img: Original PIL image
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            scale_factor: Optional scaling factor
            
        Returns:
            Actual scale factor used
        """
        if not bbox:
            return None
        
        bbox = bbox_transform(bbox, img.size[0], img.size[1], raw_img.size[0], raw_img.size[1])
        
        # Get original image dimensions
        orig_width, orig_height = raw_img.size
        orig_area = orig_width * orig_height
        
        # Crop the region
        cropped_img = get_img_crop(raw_img, bbox)
        if cropped_img is None:
            print(f"## DEBUG: Invalid bbox: {bbox}")
            return None
        
        actual_scale = 1.0
        crop_width, crop_height = cropped_img.size
        bbox_area = crop_width * crop_height
        r = bbox_area / orig_area  
        if r < 0.125:
            actual_scale = 2.0
        elif r >= 0.5:
            actual_scale = 1.0
        else:
            actual_scale = 2.0 - (r - 0.125) / 0.375  
        crop_height = int(actual_scale * crop_height)  
        crop_width = int(actual_scale * crop_width)    

        h1, w1 = smart_resize(
            crop_height,
            crop_width,
            factor=28,
            min_pixels=self.crop_min_pixels,
            max_pixels=self.crop_max_pixels,
        )
        cropped_img = cropped_img.resize((w1, h1), Image.LANCZOS)
        
        self.cnt_tmp_img += 1
        device = getattr(self.model, 'device', self.device)  
        output_path = self.img_crop_path.format(device=str(device).replace(':',"_"),sample=self.sample_id, cnt=self.cnt_tmp_img)
        cropped_img.save(output_path)
        print(f"## DEBUG: Saved cropped image to {output_path} (scale={actual_scale})")
        
        return actual_scale
    
    
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

        iterations = 0
        while iterations < max_iterations:
            iterations += 1
            
            # Add cropped image to messages if available
            if self.cnt_tmp_img > 0 and crop_flag:
                crop_flag = False
                device = getattr(self.model, 'device', self.device)  
                add_img_path = self.img_crop_path.format(device=str(device).replace(':',"_"), sample=self.sample_id, cnt=self.cnt_tmp_img)
                add_img_message = {
                    "type": "image",
                    "image": add_img_path
                }
                img_crop_list.append(add_img_path)
                img_messages[0]['content'].append(add_img_message)
                text_messages[-1]['content'].append(add_img_message)
            
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
                break

            if "bbox_2d" in generated_text:
                generated_text += "]}\n"

            # Extract operation from JSON
            op = self._extract_fields(generated_text)
            print(f"##DEBUG: Extracted operation: {op}")
            if op:
                try:
                    actual_scale = self._crop_and_save_region(real_input_img, raw_img, op["bbox_2d"])
                    # Update the generated text with actual scale
                    matches = re.findall(self.stop_pattern, generated_text, re.DOTALL)
                    if matches:
                        op_str = matches[-1]
                        if "```json" in generated_text:
                            generated_text += "```\n"
                    crop_flag = True
                except Exception as e:
                    print(f"##DEBUG: - Error processing op: {e}")
                    generated_text += "Invalid operation!"
                    
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

        # Check if we hit the iteration limit
        if iterations >= max_iterations:
            print('*' * 80)
            print(f"## ERROR: {self.device} Exceeded maximum iterations")
            print('*' * 80)
        return response_list, img_crop_list, full_response, img_messages, text_messages

    

import asyncio
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import BytesIO
from multiprocessing import cpu_count
from typing import List, Optional, Tuple, Union

import numpy as np
from accelerate import Accelerator, DistributedType
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

NUM_SECONDS_TO_SLEEP = 5

try:
    from vllm import LLM, SamplingParams
except ImportError:
    vllm = None


@register_model("vllm")
class VLLM(lmms):
    def __init__(
        self,
        model_version: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        batch_size: int = 1,
        max_frame_num: int = 32,
        threads: int = 16,  # Threads to use for decoding visuals
        trust_remote_code: Optional[bool] = True,
        chat_template: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Manually set a image token for GPT4V so that we can search for it
        # and split the text and image
        # Here we just use the same token as llava for convenient
        self.model_version = model_version
        self.max_frame_num = max_frame_num
        self.threads = threads
        self.chat_template = chat_template

        # Convert any string arguments that start with { and end with } to dictionaries
        for key, value in kwargs.items():
            if isinstance(value, str) and value.strip().startswith('{') and value.strip().endswith('}'):
                try:
                    kwargs[key] = json.loads(value)
                except json.JSONDecodeError:
                    eval_logger.warning(f"Failed to parse JSON-like string for argument '{key}': {value}")

        self.min_pixels = 4*28*28
        self.max_pixels = 2048*28*28*4
        # Set up vllm client
        self.client = LLM(
            model=self.model_version,
            # tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            limit_mm_per_prompt={"image": 16, "video": 10},
            mm_processor_kwargs=(
                {
                    "max_pixels": 2048*28*28*4,
                    "min_pixels": 4*28*28,
                }
            ),
            max_model_len=16384*4,
            # **kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_version, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device
        self.batch_size_per_gpu = int(batch_size)

    # Function to encode the image
    def encode_image(self, image: Union[Image.Image, str]):
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy()

        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()

        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    # # Function to encode the video
    # def encode_video(self, video_path):
    #     vr = VideoReader(video_path, ctx=cpu(0))
    #     total_frame_num = len(vr)
    #     uniform_sampled_frames = np.linspace(0, total_frame_num - 1, self.max_frame_num, dtype=int)

    #     # Ensure the last frame is included
    #     if total_frame_num - 1 not in uniform_sampled_frames:
    #         uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)

    #     frame_idx = uniform_sampled_frames.tolist()
    #     frames = vr.get_batch(frame_idx).asnumpy()

    #     base64_frames = []
    #     for frame in frames:
    #         img = Image.fromarray(frame)
    #         output_buffer = BytesIO()
    #         img.save(output_buffer, format="PNG")
    #         byte_data = output_buffer.getvalue()
    #         base64_str = base64.b64encode(byte_data).decode("utf-8")
    #         base64_frames.append(base64_str)

    #     return base64_frames

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        batch_size = self.batch_size_per_gpu
        batched_requests = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]
        for batch_requests in batched_requests:
            
            # batched_messages = []
            response_text = []
            for idx in range(len(batch_requests)):
                contexts, gen_kwargs, doc_to_visual, doc_id, task, split = batch_requests[idx].arguments
                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 1024
                if gen_kwargs["max_new_tokens"] > 4096:
                    gen_kwargs["max_new_tokens"] = 4096
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0
                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = 0.95
                    
                agent = AgentVLMVLLM(self.client, self.processor, min_pixels=self.min_pixels, max_pixels=self.max_pixels,temp_dir="./agent_crops_tmp_rl_1",temperature=gen_kwargs["temperature"], max_tokens_once=gen_kwargs["max_new_tokens"], device=self.device)
                # params = {
                #     "temperature": gen_kwargs["temperature"],
                #     # "max_tokens": gen_kwargs["max_new_tokens"],
                #     "max_tokens": 1024, 
                #     "top_p": gen_kwargs["top_p"],
                # }
                # sampling_params = SamplingParams(**params) 

                visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]

                visuals = self.flatten(visuals)
                image_path = None
                for visual in visuals:
                    if isinstance(visual, str) and (".jpg" in visual or ".jpeg" in visual or ".png" in visual or ".gif" in visual or ".bmp" in visual or ".tiff" in visual or ".webp" in visual):
                        image_path = visual
                    elif isinstance(visual, Image.Image):
                        # 暂存
                        image_path = ".tmp_rl_1.png"
                        visual.save(image_path)
                    break
                if len(visuals)>1:
                    print("##DEBUG:  Found multiple images, using the first one")
                
                question = contexts
                _, _, full_response, img_messages, text_messages = agent.process(image_path, question, 12)

                content_matches = re.findall(r'<answer>(.*?)</answer>', full_response, re.DOTALL)
                response = content_matches[-1].strip() if content_matches else full_response.strip().split('\n')[-1] # by hyr
                response_text.append(response)
            # if self.chat_template is not None:
            #     with open(self.chat_template, "r") as f:
            #         chat_template = f.read()
            #     response = self.client.chat(sampling_params=sampling_params, messages=batched_messages, chat_template=chat_template)
            # else:
            #     response = self.client.chat(sampling_params=sampling_params, messages=batched_messages)
            # response_text = [o.outputs[0].text for o in response]

            assert len(response_text) == len(batch_requests)
            res.extend(response_text)
            pbar.update(len(batch_requests))

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "GPT4V not support"

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
