#!/usr/bin/env python3
"""Interactive & batch demo for the V*R-R3 visual reasoning agent.

- 保留单样本（图像/视频->网格->推理->grounding）流程；
- 新增 JSON/JSONL 批处理，显示进度条（tqdm），逐条保存可视化与模型输出；
- 将 full_response 与 <answer>...</answer> 中间文本(model_answer)保存至 outputs.json；
- 若原始 JSON 含 "answer"（如 'A'/'B'/'C'/'D'），计算整体准确率；
- 输出文件名包含 videoID / question_id 等，保证不同样本间不冲突。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Dict, Any, Optional

import av
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch._dynamo.config
from transformers import AutoProcessor
from tqdm import tqdm
from pathlib import Path

# 确保仓库根路径在 sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.model.r3_vllm import AgentVLMVLLM, bbox_transform  # noqa: E402
from qwen_vl_utils import smart_resize  # noqa: E402
from vllm import LLM
# import torch
# torch._dynamo.config.suppress_errors = True

DEFAULT_NUM_FRAMES = 256 #16
GRID_ROWS = 16 #4
GRID_COLS = 16 #4

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".mpg", ".mpeg"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
ALL_MEDIA_EXTS = VIDEO_EXTS | IMAGE_EXTS


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDE]", s):
        return ""

    matches = re.search(r"[ABCDE]", s)
    if matches is None:
        return ""
    return matches[0]

def extract_characters_regex_longvu(pred):
    # 这是longvu的 处理
    pred = pred.replace("Answer", "")

    letters = ["A", "B", "C", "D", "E"]

    pred_answer = re.findall("[\(\ \[]*([A-E])[\)\.\ \]]*", pred)

    if pred_answer:
        pred_answer = pred_answer[0].strip()
        pred_answer = pred_answer.strip("()")
    if pred_answer in letters:
        pred_idx = letters.index(pred_answer)
        pred = letters[pred_idx]
    else:
        print("pred_answer: ", pred_answer, " pred: ", pred, flush=True)
        pred_idx = 2
        pred = letters[pred_idx]

    return pred





def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive & batch demo for V*R-R3")

    # 模型与推理相关
    p.add_argument("--model-path", type=Path, default="/remote-home/zhangkc/VLM-R3-7b-rl-v1",
                   help="Path to the pretrained checkpoint compatible with vLLM.")
    p.add_argument("--device", type=str, default="cuda", help="Device string (e.g., 'cuda', 'cuda:0').")
    p.add_argument("--device-num", type=int, default=1, help="The number of gpus.")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    p.add_argument("--min-pixels", type=int, default=32 * 28 * 28)
    p.add_argument("--max-pixels", type=int, default=8192 * 28 * 28)
    p.add_argument("--crop-min-pixels", type=int, default=32 * 28 * 28)
    p.add_argument("--crop-max-pixels", type=int, default=4096 * 28 * 28)
    p.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES,
                   help="Number of frames uniformly sampled for the grid.")
    p.add_argument("--max-iterations", type=int, default=8, help="Max reasoning turns.")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.95)

    # I/O 与批处理
    p.add_argument("--input-path", type=Path, default=None,
                   help="Path to a single image or video file (legacy single-sample mode).")
    p.add_argument("--input-json", type=Path, default=None,
                   help="Path to a JSON/JSONL file for batch processing.")
    p.add_argument("--media-root", type=Path, default=None,
                   help="Root dir to search media files referenced by JSON items (via videoID/video_id/etc.).")
    p.add_argument("--output-dir", type=Path, default=Path("demo_outputs"),
                   help="Directory to store mosaics, visualisations and outputs.json.")
    p.add_argument("--question", type=str, default=None,
                   help="Question in single-sample mode. Batch mode uses item['question'].")

    args = p.parse_args()

    # 至少有一种输入
    if args.input_json is None and args.input_path is None:
        p.error("Please provide either --input-json for batch mode or --input-path for single-sample mode.")

    return args


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def sample_frames_uniformly(frames: Sequence[Image.Image], count: int) -> List[Image.Image]:
    if not frames:
        raise ValueError("No frames decoded from video.")
    if len(frames) == count:
        return list(frames)
    indices = np.linspace(0, len(frames) - 1, count, dtype=int)
    return [frames[idx] for idx in indices]


def decode_video_frames(video_path: Path) -> List[Image.Image]:
    container = av.open(str(video_path))
    frames: List[Image.Image] = []
    try:
        for frame in container.decode(video=0):
            frames.append(frame.to_image())
    finally:
        container.close()
    return frames


def create_video_grid(video_path: Path, output_path: Path, num_frames: int) -> Path:
    if GRID_ROWS * GRID_COLS != num_frames:
        raise ValueError(
            f"Grid configuration {GRID_ROWS}x{GRID_COLS} requires exactly {GRID_ROWS * GRID_COLS} frames, "
            f"but num_frames={num_frames}."
        )
    frames = decode_video_frames(video_path)
    sampled_frames = sample_frames_uniformly(frames, num_frames)

    widths = {img.width for img in sampled_frames}
    heights = {img.height for img in sampled_frames}
    if len(widths) != 1 or len(heights) != 1:
        raise ValueError("Decoded frames must share the same resolution for grid composition.")

    fw = sampled_frames[0].width
    fh = sampled_frames[0].height
    grid_image = Image.new("RGB", (fw * GRID_COLS, fh * GRID_ROWS))

    for idx, frame in enumerate(sampled_frames):
        row = idx // GRID_COLS
        col = idx % GRID_COLS
        grid_image.paste(frame, (col * fw, row * fh))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid_image.save(output_path)
    return output_path


def extract_grounding_boxes(full_response: str) -> List[List[float]]:
    pattern = re.compile(r"\{[^{}]*\"bbox_2d\"\s*:\s*\[([^\]]+)\][^{}]*\}")
    boxes: List[List[float]] = []
    for match in pattern.finditer(full_response):
        coords_text = match.group(1)
        try:
            coords = [float(x.strip()) for x in coords_text.split(",")]
        except ValueError:
            continue
        if len(coords) == 4:
            boxes.append(coords)
    return boxes


def extract_answer_text(full_response: str) -> Optional[str]:
    # 抓取 <answer> ... </answer> 的内容；若无，则回退到简单启发式
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", full_response, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # 回退：尝试捕获 "Answer: X" / "答案：X" 等
    m2 = re.search(r"(?:^|\n)\s*(?:Answer|答案)\s*[:：]\s*(.*)", full_response, re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    return None


def draw_grounding_steps(
    image_path: Path,
    boxes: Iterable[Sequence[float]],
    resized_size: tuple[int, int],
    output_dir: Path,
    stem_prefix: str,
) -> List[Path]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    resized_width, resized_height = resized_size

    colours = [
        "#ff6b6b", "#f7b731", "#45aaf2", "#26de81",
        "#a55eea", "#fd9644", "#2bcbba", "#778ca3",
    ]

    annotated_paths: List[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, bbox in enumerate(boxes, start=1):
        mapped_bbox = bbox_transform(bbox, resized_width, resized_height, width, height)
        colour = colours[(idx - 1) % len(colours)]

        annotated = image.copy()
        drawer = ImageDraw.Draw(annotated)
        drawer.rectangle(mapped_bbox, outline=colour, width=max(2, min(width, height) // 200))
        drawer.text((mapped_bbox[0] + 4, mapped_bbox[1] + 4), f"Step {idx}", fill=colour)

        out = output_dir / f"{stem_prefix}_grounding_step_{idx:02d}.png"
        annotated.save(out)
        annotated_paths.append(out)

    image.close()
    return annotated_paths


def smart_slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)[:200]


def pick_id_parts(item: Dict[str, Any]) -> Dict[str, str]:
    """从条目中提取用于命名与检索的关键字段。"""
    parts = {
        "video_id": str(item.get("video_id", "")),
        "videoID": str(item.get("videoID", "")),
        "question_id": str(item.get("question_id", "")),
        "domain": str(item.get("domain", "")),
        "sub_category": str(item.get("sub_category", "")),
    }
    return {k: smart_slug(v) for k, v in parts.items() if v}


def resolve_media_path(item: Dict[str, Any], media_root: Optional[Path]) -> Optional[Path]:
    """尽力根据条目信息在 media_root 下找到本地媒体文件。
    支持字段：'path'/'file'/'image'/'video'（直接给路径）；
    否则用 videoID/video_id/filename 做基名在 media_root 中递归搜索。
    """
    v = item.get("videoID")
    print(v)
    video_path = "/remote-home/zhangkc/data/zhangkc/video-mme-bench/data/" + v + ".mp4"
    # video_path = media_root + v + ".mp4"
    
    return video_path
    # 直接路径字段
    # for key in ("path", "file", "image", "video"):
    #     v = item.get(key)
    #     if isinstance(v, str) and v:
    #         p = Path(v)
    #         if p.exists():
    #             return p
    #         if media_root:
    #             p2 = media_root / v
    #             if p2.exists():
    #                 return p2

    # if not media_root or not media_root.exists():
    #     return None

    # # 基名候选
    # candidates = []
    # for key in ("videoID", "video_id", "filename", "id"):
    #     vv = item.get(key)
    #     if isinstance(vv, str) and vv:
    #         candidates.append(Path(vv).stem)

    # # 递归搜索
    # for base in candidates:
    #     # 1) 完整文件名匹配
    #     for ext in ALL_MEDIA_EXTS:
    #         for p in media_root.rglob(f"**/{base}{ext}"):
    #             return p
    #     # 2) 任何以 base 开头的媒体
    #     for ext in ALL_MEDIA_EXTS:
    #         for p in media_root.rglob(f"**/{base}*{ext}"):
    #             return p

    # return None


def options_to_map(options: List[str]) -> Dict[str, str]:
    """将 ['A. 1.', 'B. 3.'] 解析为 {'A':'1', 'B':'3'}。"""
    m: Dict[str, str] = {}
    for opt in options:
        if not isinstance(opt, str):
            continue
        # 允许 'A. xxx' / 'A) xxx' / 'A xxx'
        mm = re.match(r"\s*([A-Da-d])\s*[\.\)\-:]*\s*(.+)\s*$", opt)
        if mm:
            m[mm.group(1).upper()] = mm.group(2).strip()
    return m


def normalize_text(x: str) -> str:
    x = x.lower().strip()
    x = re.sub(r"[\s\.\,\;\:\!\?]+", " ", x)
    x = x.replace("’", "'").replace("“", '"').replace("”", '"')
    return x


def extract_first_int(s: str) -> Optional[int]:
    m = re.search(r"-?\d+", s)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    return None


def pick_predicted_letter(model_answer: Optional[str], options_map: Dict[str, str]) -> Optional[str]:
    """将 model_answer 映射为选项字母 A/B/C/D（若可能）。"""
    if not model_answer:
        return None

    ans = model_answer.strip()

    # 1) 直接字母
    m = re.match(r"^\s*([A-Da-d])\b", ans)
    if m:
        return m.group(1).upper()

    # 2) 包含 'option D' / '选项D'
    m2 = re.search(r"option\s*([A-Da-d])", ans, re.IGNORECASE)
    if m2:
        return m2.group(1).upper()

    # 3) 与选项内容匹配（文本归一化后包含/相等）
    norm_ans = normalize_text(ans)
    for k, v in options_map.items():
        if normalize_text(v) == norm_ans:
            return k
    for k, v in options_map.items():
        if normalize_text(v) in norm_ans or norm_ans in normalize_text(v):
            return k

    # 4) 数值匹配（适合 Counting 问题）
    vi = extract_first_int(ans)
    if vi is not None:
        for k, v in options_map.items():
            v_int = extract_first_int(v)
            if v_int is not None and v_int == vi:
                return k

    return None


@dataclass
class RunResult:
    meta: Dict[str, Any]
    base_image_path: Optional[str]
    grid_path: Optional[str]
    grounding_paths: List[str]
    full_response: str
    model_answer: Optional[str]
    predicted_option: Optional[str]
    gt_option: Optional[str]
    is_correct: Optional[bool]
    error: Optional[str]


def run_single_item(
    agent: AgentVLMVLLM,
    processor: AutoProcessor,
    llm: LLM,
    args: argparse.Namespace,
    item: Dict[str, Any],
) -> RunResult:
    """对单条 JSON 数据执行原有流程。"""
    id_parts = pick_id_parts(item)
    stem_bits = [id_parts.get("videoID"), id_parts.get("video_id"), id_parts.get("question_id")]
    stem_prefix = "__".join([b for b in stem_bits if b]) or "sample"

    media_path = resolve_media_path(item, args.media_root)
    question = item.get("question") or args.question
    options = item.get("options") # 字符串列表， 将其添加在问题后面
    # # formatted_options = " ".join([f"({chr(65 + i)}) {option}" for i, option in enumerate(options)])
    # formatted_options = " ".join(["{option} " for i, option in enumerate(options)])
    # 格式化选项
    formatted_options = " ".join(options)  # 直接将选项列表连接成一个字符串

    # 将格式化后的选项附加到问题后面
    question = question + " " + formatted_options
    if not question:
        raise ValueError("No question provided for this item.")
    # if not media_path or not media_path.exists():
    #     raise FileNotFoundError("Media file not found for item (try --media-root).")

    # media_path = media_path.resolve()
    out_item_dir = args.output_dir / smart_slug(stem_prefix)
    out_item_dir.mkdir(parents=True, exist_ok=True)

    print(media_path)
    if isinstance(media_path, str):
        media_path = Path(media_path)
    # 1) 准备输入图像（若是视频则拼接网格）
    if media_path: # is_video(media_path):
        grid = out_item_dir / f"{media_path.stem}__{stem_prefix}__grid_{args.num_frames}.png"
        image_path = create_video_grid(media_path, grid, args.num_frames)
        grid_path = str(image_path)
    else:
        image_path = media_path
        grid_path = None

    # 2) 运行 agent
    _, _, full_response, _, _ = agent.process(
        str(image_path),
        str(question),
        max_iterations=args.max_iterations,
    )

    # 3) 解析 grounding & 可视化
    boxes = extract_grounding_boxes(full_response)
    grounding_paths: List[str] = []
    if boxes:
        base_image = Image.open(image_path)
        resized_h, resized_w = smart_resize(
            base_image.height,
            base_image.width,
            factor=28,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        base_image.close()

        grounding_paths = [str(p) for p in draw_grounding_steps(
            image_path=image_path,
            boxes=boxes,
            resized_size=(resized_w, resized_h),
            output_dir=out_item_dir / "grounding",
            stem_prefix=media_path.stem + "__" + stem_prefix,
        )]

    # 4) 提取答案 与 选项映射
    model_answer = extract_answer_text(full_response)
    options_map = options_to_map(item.get("options") or [])
    pred_letter = pick_predicted_letter(model_answer, options_map) if options_map else None

    gt = item.get("answer")
    gt_letter = gt.strip().upper() if isinstance(gt, str) and gt.strip() else None
    
    # outputs_extra_longva = extract_characters_regex(outputs)
    # outputs_extra_longvu = extract_characters_regex_longvu(outputs)
    
    is_correct = (pred_letter == gt_letter) if (pred_letter and gt_letter) else None

    return RunResult(
        meta={
            "question_id": item.get("question_id"),
            "video_id": item.get("video_id"),
            "videoID": item.get("videoID"),
            "domain": item.get("domain"),
            "sub_category": item.get("sub_category"),
            "url": item.get("url"),
            "question": question,
            "options": item.get("options"),
        },
        base_image_path=str(image_path),
        grid_path=grid_path,
        grounding_paths=grounding_paths,
        full_response=full_response,
        model_answer=model_answer,
        predicted_option=pred_letter,
        gt_option=gt_letter,
        is_correct=is_correct,
        error=None,
    )


def load_json_items(path: Path) -> List[Dict[str, Any]]:
    """支持 .json（数组或单对象）与 .jsonl（每行一个对象）。"""
    if path.suffix.lower() == ".jsonl":
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return [data]
    else:
        raise ValueError("Unsupported JSON structure (expect list or object).")


def save_outputs_and_report(results: List[RunResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "outputs.json"

    serializable = []
    total = len(results)
    have_gt = 0
    correct = 0

    for r in results:
        serializable.append({
            "meta": r.meta,
            "base_image_path": r.base_image_path,
            "grid_path": r.grid_path,
            "grounding_paths": r.grounding_paths,
            "full_response": r.full_response,
            "model_answer": r.model_answer,
            "predicted_option": r.predicted_option,
            "gt_option": r.gt_option,
            "is_correct": r.is_correct,
            "error": r.error,
        })
        if r.gt_option is not None:
            have_gt += 1
            if r.is_correct is True:
                correct += 1

    # 整体准确率（分母=总条目数；未答/解析失败计为错误）
    overall_acc = (correct / total) if total > 0 else 0.0

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_items": total,
            "with_ground_truth": have_gt,
            "num_correct": correct,
            "overall_accuracy": overall_acc,
            "results": serializable,
        }, f, ensure_ascii=False, indent=2)

    print("\n===== 汇总结果 =====")
    print(f"总样本数: {total}")
    print(f"含GT样本数: {have_gt}")
    print(f"预测正确数: {correct}")
    print(f"整体准确率: {overall_acc:.4f}")
    print(f"明细保存至: {out_path}")


def build_agent(args: argparse.Namespace) -> tuple[LLM, AutoProcessor, AgentVLMVLLM]:
    print("Loading processor and vLLM model...")
    processor = AutoProcessor.from_pretrained(
        str(args.model_path),
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    llm = LLM(
        model=str(args.model_path),
        device=args.device,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.device_num, #2  
        dtype=torch.bfloat16,
        limit_mm_per_prompt={"image": 16, "video": 0},  # 禁止原生视频输入
        mm_processor_kwargs={
            "max_pixels": args.max_pixels,
            "min_pixels": args.min_pixels,
        },
        max_model_len=8192 * 4,
    )
    agent = AgentVLMVLLM(
        model=llm,
        processor=processor,
        temp_dir=str(args.output_dir / "crops"),
        device=args.device,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        temperature=args.temperature,
        crop_min_pixels=args.crop_min_pixels,
        crop_max_pixels=args.crop_max_pixels,
    )
    return llm, processor, agent


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 构建模型（一次）
    llm, processor, agent = build_agent(args)

    # 单样本模式（兼容旧用法）
    if args.input_json is None and args.input_path is not None:
        if not args.input_path.exists():
            raise FileNotFoundError(f"Input path not found: {args.input_path}")

        question = args.question or input("请输入你的问题: ")

        # 若为视频，先做网格
        if is_video(args.input_path):
            mosaic_path = args.output_dir / f"{args.input_path.stem}_grid_{args.num_frames}.png"
            print(f"Decoding video and creating {GRID_ROWS}x{GRID_COLS} grid with {args.num_frames} frames...")
            image_path = create_video_grid(args.input_path, mosaic_path, args.num_frames)
            print(f"Video grid saved to: {image_path}")
        else:
            image_path = args.input_path

        print("Running agent reasoning...")
        _, _, full_response, _, _ = agent.process(
            str(image_path), question, max_iterations=args.max_iterations
        )

        print("\n===== 模型输出 =====")
        print(full_response)

        boxes = extract_grounding_boxes(full_response)
        if not boxes:
            print("\n未检测到任何grounding区域。")
        else:
            print(f"\n检测到 {len(boxes)} 个grounding步骤，正在生成可视化...")
            base_image = Image.open(image_path)
            resized_h, resized_w = smart_resize(
                base_image.height, base_image.width, factor=28,
                min_pixels=args.min_pixels, max_pixels=args.max_pixels
            )
            base_image.close()
            stem_prefix = args.input_path.stem
            annotated_paths = draw_grounding_steps(
                image_path=image_path,
                boxes=boxes,
                resized_size=(resized_w, resized_h),
                output_dir=args.output_dir / "grounding",
                stem_prefix=stem_prefix,
            )
            print("Grounding 可视化已保存：")
            for p in annotated_paths:
                print(f" - {p}")

        # 保存单条 outputs.json
        model_answer = extract_answer_text(full_response)
        save_outputs_and_report([RunResult(
            meta={"question": question, "single_input": str(args.input_path)},
            base_image_path=str(image_path),
            grid_path=str(image_path) if is_video(args.input_path) else None,
            grounding_paths=[str(x) for x in (annotated_paths if boxes else [])],
            full_response=full_response,
            model_answer=model_answer,
            predicted_option=None,
            gt_option=None,
            is_correct=None,
            error=None,
        )], args.output_dir)
        return

    # 批处理模式
    items = load_json_items(args.input_json)
    results: List[RunResult] = []
    correct = 0

    pbar = tqdm(items, desc="Processing items", unit="item")
    for item in pbar:
        try:
            r = run_single_item(agent, processor, llm, args, item)
        except Exception as e:
            # 记录错误但不中断整体
            id_parts = pick_id_parts(item)
            r = RunResult(
                meta={
                    "question_id": item.get("question_id"),
                    "video_id": item.get("video_id"),
                    "videoID": item.get("videoID"),
                    "question": item.get("question"),
                },
                base_image_path=None,
                grid_path=None,
                grounding_paths=[],
                full_response="",
                model_answer=None,
                predicted_option=None,
                gt_option=str(item.get("answer")).strip().upper() if item.get("answer") else None,
                is_correct=False if item.get("answer") else None,
                error=str(e),
            )

        results.append(r)
        if r.is_correct is True:
            correct += 1
        # 在进度条上动态显示当前正确/总数
        pbar.set_postfix_str(f"correct={correct}/{len(results)}")

    # 保存总结果并打印准确率
    save_outputs_and_report(results, args.output_dir)


if __name__ == "__main__":
    main()
