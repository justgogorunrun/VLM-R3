"""Direct evaluation script for Video-Holmes without relying on vLLM.

This script mirrors the evaluation logic in ``evaluate1.py`` but uses the
native inference interfaces for every supported model family so that video
pre-processing options such as the number of sampled frames and the maximum
pixel budget per frame can be controlled explicitly.  The main motivation is
to avoid the ``value too large`` runtime error that may happen when letting
vLLM implicitly decide how to decode videos.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

# ---------------------------------------------------------------------------
# Video preprocessing utilities
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class VideoDecodeConfig:
    """Configuration controlling how a video is decoded."""

    num_frames: int = 32
    max_pixels: Optional[int] = 320 * 360
    use_thumbnail: bool = True
    image_size: int = 448


def build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: Iterable[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    *,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> List[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images: List[Image.Image] = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _resize_to_max_pixels(image: Image.Image, max_pixels: Optional[int]) -> Image.Image:
    if max_pixels is None:
        return image
    width, height = image.size
    pixels = width * height
    if pixels <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / pixels)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return image.resize((new_width, new_height), resample=Image.BICUBIC)


def sample_video_frames(
    video_path: str,
    *,
    config: VideoDecodeConfig,
    use_internvl_dynamic_tiles: bool = False,
) -> Tuple[List[torch.Tensor], List[int]]:
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    start_idx, end_idx = 0, max_frame
    seg_size = float(end_idx - start_idx) / config.num_frames
    indices = [
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(config.num_frames)
    ]

    transform = build_transform(config.image_size)
    pixel_values_list: List[torch.Tensor] = []
    num_patches_list: List[int] = []
    for index in indices:
        index = min(max(index, 0), max_frame)
        frame = Image.fromarray(vr[index].asnumpy()).convert("RGB")
        frame = _resize_to_max_pixels(frame, config.max_pixels)
        if use_internvl_dynamic_tiles:
            tiles = dynamic_preprocess(
                frame,
                image_size=config.image_size,
                use_thumbnail=config.use_thumbnail,
                max_num=1,
            )
            pixel_values = torch.stack([transform(tile) for tile in tiles])
        else:
            pixel_values = transform(frame).unsqueeze(0)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    stacked = torch.cat(pixel_values_list, dim=0)
    return stacked, num_patches_list


def decode_video_for_qwen(
    video_path: str,
    config: VideoDecodeConfig,
) -> List[Image.Image]:
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    start_idx, end_idx = 0, max_frame
    seg_size = float(end_idx - start_idx) / config.num_frames
    indices = [
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(config.num_frames)
    ]
    frames: List[Image.Image] = []
    for index in indices:
        index = min(max(index, 0), max_frame)
        frame = Image.fromarray(vr[index].asnumpy()).convert("RGB")
        frame = _resize_to_max_pixels(frame, config.max_pixels)
        frames.append(frame)
    return frames


# ---------------------------------------------------------------------------
# Model preparation helpers
# ---------------------------------------------------------------------------


def prepare_internvl_25_8b(model_path: Optional[str]):
    path = model_path or "OpenGVLab/InternVL2_5-8B"
    device_map = _split_internvl_layers("InternVL2_5-8B")
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    return model, tokenizer, generation_config


def _split_internvl_layers(model_name: str) -> Dict[str, int]:
    device_map: Dict[str, int] = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        "InternVL2_5-1B": 24,
        "InternVL2_5-2B": 24,
        "InternVL2_5-4B": 36,
        "InternVL2_5-8B": 32,
        "InternVL2_5-26B": 48,
        "InternVL2_5-38B": 64,
        "InternVL2_5-78B": 80,
    }[model_name]
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    layout = [num_layers_per_gpu] * world_size
    layout[0] = math.ceil(layout[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(layout):
        for _ in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map.update(
        {
            "vision_model": 0,
            "mlp1": 0,
            "language_model.model.tok_embeddings": 0,
            "language_model.model.embed_tokens": 0,
            "language_model.output": 0,
            "language_model.model.norm": 0,
            "language_model.model.rotary_emb": 0,
            "language_model.lm_head": 0,
            f"language_model.model.layers.{num_layers - 1}": 0,
        }
    )
    return device_map


def prepare_internvl_3(model_path: Optional[str]):
    path = model_path or "OpenGVLab/InternVL3-8B"

    def split_model(model_id: str) -> Dict[str, int]:
        device_map: Dict[str, int] = {}
        world_size = torch.cuda.device_count()
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        num_layers = config.llm_config.num_hidden_layers
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        layout = [num_layers_per_gpu] * world_size
        layout[0] = math.ceil(layout[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(layout):
            for _ in range(num_layer):
                device_map[f"language_model.model.layers.{layer_cnt}"] = i
                layer_cnt += 1
        device_map.update(
            {
                "vision_model": 0,
                "mlp1": 0,
                "language_model.model.tok_embeddings": 0,
                "language_model.model.embed_tokens": 0,
                "language_model.output": 0,
                "language_model.model.norm": 0,
                "language_model.model.rotary_emb": 0,
                "language_model.lm_head": 0,
                f"language_model.model.layers.{num_layers - 1}": 0,
            }
        )
        return device_map

    device_map = split_model(path)
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    return model, tokenizer, generation_config


QWEN_MODEL_PATHS = {
    "Qwen2.5-VL-7B": "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen2.5-VL-32B": "Qwen/Qwen2.5-VL-32B-Instruct",
    "VideoChat-R1": "OpenGVLab/VideoChat-R1_7B",
    "Video-R1": "Video-R1/Video-R1-7B",
    "Qwen3-VL-8B": "Qwen/Qwen3-VL-8B-Instruct",
}


def prepare_qwen_family_direct(model_name: str, model_path: Optional[str]):
    path = model_path or QWEN_MODEL_PATHS.get(model_name)
    if path is None:
        raise ValueError(f"Unsupported Qwen-family model: {model_name}")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer
    return model, processor


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def predict_internvl(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    generation_config: Dict,
    question_prompt: str,
    video_path: str,
    config: VideoDecodeConfig,
) -> str:
    pixel_values, num_patches_list = sample_video_frames(
        video_path, config=config, use_internvl_dynamic_tiles=True
    )
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    video_prefix = "".join(
        [f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))]
    )
    question = video_prefix + question_prompt
    response, _ = model.chat(
        tokenizer,
        pixel_values,
        question,
        generation_config,
        num_patches_list=num_patches_list,
        history=None,
        return_history=True,
    )
    return response


def predict_qwen_family_direct(
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    question_prompt: str,
    video_path: str,
    config: VideoDecodeConfig,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.8,
) -> str:
    frames = decode_video_for_qwen(video_path, config)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": question_prompt},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[prompt], videos=[frames], return_tensors="pt")
    first_param = next(model.parameters())
    device = first_param.device
    model_dtype = first_param.dtype
    for key, value in list(inputs.items()):
        if isinstance(value, torch.Tensor):
            if torch.is_floating_point(value):
                inputs[key] = value.to(device, dtype=model_dtype)
            else:
                inputs[key] = value.to(device)
    generation_outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    if isinstance(generation_outputs, tuple):
        generation_ids = generation_outputs[0]
    else:
        generation_ids = generation_outputs
    response = processor.batch_decode(
        generation_ids, skip_special_tokens=True
    )[0]
    return response


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def normalise_choice(predicted_answer: str) -> Tuple[str, str]:
    think_pattern = r"<think>\s*(.*?)\s*</think>"
    try:
        think_matches = re.findall(think_pattern, predicted_answer, re.DOTALL)
    except Exception:
        think_matches = []
    thinking = think_matches[-1].strip() if think_matches else "WRONG"

    answer_pattern = r"<answer>\s*(.*?)\s*</answer>"
    try:
        answer_matches = re.findall(answer_pattern, predicted_answer, re.DOTALL)
    except Exception:
        answer_matches = []
    choice_block = answer_matches[-1].strip() if answer_matches else predicted_answer

    for option in ["A", "B", "C", "D", "E", "F"]:
        if re.search(rf"\b{option}\b", choice_block):
            return option, thinking
    return "WRONG", thinking


def evaluate(
    video_path: str,
    json_file_path: str,
    output_path: str,
    model_name: str,
    model_path: Optional[str],
    config: VideoDecodeConfig,
):
    if model_name == "InternVL2.5-8B":
        model, tokenizer, generation_config = prepare_internvl_25_8b(model_path)
        model_family = "internvl"
    elif model_name == "InternVL3-8B":
        model, tokenizer, generation_config = prepare_internvl_3(model_path)
        model_family = "internvl"
    elif model_name in QWEN_MODEL_PATHS:
        model, processor = prepare_qwen_family_direct(model_name, model_path)
        model_family = "qwen"
    elif "gemini" in model_name.lower():
        raise ValueError("Gemini models require API access and are not supported here.")
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    os.makedirs(output_path, exist_ok=True)
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    question_types = ["SR", "IMC", "TCI", "TA", "MHR", "PAR", "CTI"]
    correct_counts = defaultdict(int)
    total_counts = defaultdict(int)

    output_process: List[Dict] = []
    json_file_output = os.path.join(output_path, f"Results-{model_name}.json")

    processed_ids = set()
    if os.path.exists(json_file_output):
        with open(json_file_output, "r", encoding="utf-8") as f:
            existing = json.load(f)
        for item in existing:
            qid = item.get("Question ID")
            if qid:
                processed_ids.add(qid)
        output_process.extend(existing)

    for item in tqdm(data, desc="Evaluating", ncols=100):
        question_id = item.get("Question ID")
        if question_id in processed_ids:
            continue
        explanation = item.get("Explanation")
        question_type = item.get("Question Type")
        question = item.get("Question")
        options = item.get("Options") or {}
        options_text = ", ".join([
            f"{key}: {value}" for key, value in options.items()
        ])
        correct_answer = item.get("Answer")
        video_id = item.get("video ID")
        video_file = os.path.join(video_path, f"{video_id}.mp4")

        question_prompt = (
            "Based on the given video, reason and answer the single-choice question. "
            "Provide your reasoning between the <think> and </think> tags, and then "
            "give your final answer between the <answer> and </answer> tags. The "
            f"question is: {question}. The options are: {options_text}. Your answer:"
        )

        try:
            if model_family == "internvl":
                predicted = predict_internvl(
                    model,
                    tokenizer,
                    generation_config,
                    question_prompt,
                    video_file,
                    config,
                )
            elif model_family == "qwen":
                predicted = predict_qwen_family_direct(
                    model,
                    processor,
                    question_prompt,
                    video_file,
                    config,
                )
            else:
                raise RuntimeError("Unexpected model family")
        except Exception as exc:  # pragma: no cover - logging for debugging
            print(f"Failed to process {question_id}: {exc}")
            continue

        print(f"Prediction for {question_id}: {predicted}")
        predicted_choice, thinking = normalise_choice(predicted)

        if predicted_choice == correct_answer:
            correct_counts[question_type] += 1
        total_counts[question_type] += 1

        output_process.append(
            {
                "video ID": video_id,
                "Question ID": question_id,
                "Question Type": question_type,
                "Question": question,
                "Options": options,
                "GT": correct_answer,
                "Explanation": explanation,
                "Predicr Answer": predicted_choice,
                "Thinking": thinking,
                "Correct": predicted_choice == correct_answer,
            }
        )

        with open(json_file_output, "w", encoding="utf-8") as f:
            json.dump(output_process, f, indent=2, ensure_ascii=False)

    accuracies = {
        q_type: (correct_counts[q_type] / total_counts[q_type] if total_counts[q_type] > 0 else 0)
        for q_type in question_types
    }
    total_correct = sum(correct_counts.values())
    total_questions = sum(total_counts.values())
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0

    csv_file = os.path.join(output_path, f"Results-{model_name}.csv")
    with open(csv_file, "w", encoding="utf-8", newline="") as csvfile:
        fieldnames = question_types + ["Overall Accuracy", "Total Questions"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        row = {q_type: f"{accuracies[q_type]:.2f}" for q_type in question_types}
        row["Overall Accuracy"] = f"{overall_accuracy:.2f}"
        row["Total Questions"] = total_questions
        writer.writerow(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Direct Video-Holmes evaluation")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="Results/")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--max_pixels", type=int, default=320 * 360)
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--disable_thumbnail", action="store_true")
    args = parser.parse_args()

    config = VideoDecodeConfig(
        num_frames=args.num_frames,
        max_pixels=args.max_pixels,
        image_size=args.image_size,
        use_thumbnail=not args.disable_thumbnail,
    )

    evaluate(
        args.video_path,
        args.benchmark,
        args.output_path,
        args.model_name,
        args.model_path,
        config,
    )


if __name__ == "__main__":
    main()
