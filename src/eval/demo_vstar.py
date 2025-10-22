diff --git a/src/scripts/demo_vstar_r3.py b/src/scripts/demo_vstar_r3.py
new file mode 100644
index 0000000000000000000000000000000000000000..55ec86e73ce520a55a54cf4544c33b0d75eb187c
--- /dev/null
+++ b/src/scripts/demo_vstar_r3.py
@@ -0,0 +1,351 @@
+#!/usr/bin/env python3
+"""Interactive demo for the V*R-R3 visual reasoning agent.
+
+This script mirrors the behaviour exercised in ``test_vstar_r3.py`` and adds
+utilities for end users to query the model with either a single image or a
+video.  When a video is provided, the script uniformly samples 64 frames and
+stitches them into an 8x8 grid without resizing the individual frames.  The
+resulting image (either the original image or the video grid) is then fed to
+``AgentVLMVLLM`` for multi-step reasoning.  Each grounding operation produced
+by the model is visualised as a bounding box overlay on the input image to help
+understand the step-by-step reasoning process.
+
+Example usage::
+
+    python src/scripts/demo_vstar_r3.py \
+        --model-path /path/to/model \
+        --input-path /path/to/image_or_video \
+        --question "Where is the red car located?"
+
+The full response (including ``<think>`` and ``<answer>`` segments) will be
+printed to the console, and the grounding visualisations will be saved in the
+output directory.
+"""
+
+from __future__ import annotations
+
+import argparse
+import re
+import sys
+from pathlib import Path
+from typing import Iterable, List, Sequence
+
+import av
+import numpy as np
+from PIL import Image, ImageDraw
+import torch
+from transformers import AutoProcessor
+from vllm import LLM
+
+# Ensure the repository root is on the Python path so that ``src`` can be
+# imported when this script is executed directly.
+REPO_ROOT = Path(__file__).resolve().parents[2]
+if str(REPO_ROOT) not in sys.path:
+    sys.path.append(str(REPO_ROOT))
+
+from src.model.r3_vllm import AgentVLMVLLM, bbox_transform  # noqa: E402
+from qwen_vl_utils import smart_resize  # noqa: E402
+
+
+DEFAULT_NUM_FRAMES = 64
+GRID_ROWS = 8
+GRID_COLS = 8
+
+
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(description="Interactive demo for V*R-R3")
+    parser.add_argument(
+        "--model-path",
+        type=Path,
+        required=True,
+        help="Path to the pretrained checkpoint compatible with vLLM.",
+    )
+    parser.add_argument(
+        "--input-path",
+        type=Path,
+        required=True,
+        help="Path to an image or a video file.",
+    )
+    parser.add_argument(
+        "--question",
+        type=str,
+        default=None,
+        help="Question to ask about the visual input.  If omitted, you will be prompted interactively.",
+    )
+    parser.add_argument(
+        "--output-dir",
+        type=Path,
+        default=Path("demo_outputs"),
+        help="Directory used to store intermediate mosaics and grounding visualisations.",
+    )
+    parser.add_argument(
+        "--device",
+        type=str,
+        default="cuda",
+        help="Device string passed to vLLM (e.g. 'cuda', 'cuda:0').",
+    )
+    parser.add_argument(
+        "--temperature",
+        type=float,
+        default=0.0,
+        help="Sampling temperature for generation.",
+    )
+    parser.add_argument(
+        "--min-pixels",
+        type=int,
+        default=32 * 28 * 28,
+        help="Lower bound on pixel count used by the processor.",
+    )
+    parser.add_argument(
+        "--max-pixels",
+        type=int,
+        default=8192 * 28 * 28,
+        help="Upper bound on pixel count used by the processor.",
+    )
+    parser.add_argument(
+        "--crop-min-pixels",
+        type=int,
+        default=32 * 28 * 28,
+        help="Lower bound on pixel count for cropped regions.",
+    )
+    parser.add_argument(
+        "--crop-max-pixels",
+        type=int,
+        default=4096 * 28 * 28,
+        help="Upper bound on pixel count for cropped regions.",
+    )
+    parser.add_argument(
+        "--num-frames",
+        type=int,
+        default=DEFAULT_NUM_FRAMES,
+        help="Number of frames uniformly sampled from the video (must be 64 for the default 8x8 grid).",
+    )
+    parser.add_argument(
+        "--max-iterations",
+        type=int,
+        default=8,
+        help="Maximum reasoning turns performed by the agent.",
+    )
+    parser.add_argument(
+        "--gpu-memory-utilization",
+        type=float,
+        default=0.9,
+        help="Memory utilisation hint passed to vLLM.",
+    )
+    return parser.parse_args()
+
+
+def is_video(path: Path) -> bool:
+    return path.suffix.lower() in {
+        ".mp4",
+        ".mov",
+        ".avi",
+        ".mkv",
+        ".webm",
+        ".flv",
+        ".wmv",
+        ".mpg",
+        ".mpeg",
+    }
+
+
+def sample_frames_uniformly(frames: Sequence[Image.Image], count: int) -> List[Image.Image]:
+    if not frames:
+        raise ValueError("No frames decoded from video.")
+
+    if len(frames) == count:
+        return list(frames)
+
+    indices = np.linspace(0, len(frames) - 1, count, dtype=int)
+    return [frames[idx] for idx in indices]
+
+
+def decode_video_frames(video_path: Path) -> List[Image.Image]:
+    container = av.open(str(video_path))
+    frames: List[Image.Image] = []
+    try:
+        for frame in container.decode(video=0):
+            frames.append(frame.to_image())
+    finally:
+        container.close()
+    return frames
+
+
+def create_video_grid(video_path: Path, output_path: Path, num_frames: int) -> Path:
+    if GRID_ROWS * GRID_COLS != num_frames:
+        raise ValueError(
+            f"Grid configuration {GRID_ROWS}x{GRID_COLS} requires exactly {GRID_ROWS * GRID_COLS} frames, "
+            f"but num_frames={num_frames}."
+        )
+
+    frames = decode_video_frames(video_path)
+    sampled_frames = sample_frames_uniformly(frames, num_frames)
+
+    # Ensure all frames share the same resolution (video decoders typically
+    # guarantee this, but we validate explicitly).
+    widths = {img.width for img in sampled_frames}
+    heights = {img.height for img in sampled_frames}
+    if len(widths) != 1 or len(heights) != 1:
+        raise ValueError("Decoded frames must share the same resolution for grid composition.")
+
+    frame_width = sampled_frames[0].width
+    frame_height = sampled_frames[0].height
+    grid_image = Image.new("RGB", (frame_width * GRID_COLS, frame_height * GRID_ROWS))
+
+    for idx, frame in enumerate(sampled_frames):
+        row = idx // GRID_COLS
+        col = idx % GRID_COLS
+        grid_image.paste(frame, (col * frame_width, row * frame_height))
+
+    output_path.parent.mkdir(parents=True, exist_ok=True)
+    grid_image.save(output_path)
+    return output_path
+
+
+def extract_grounding_boxes(full_response: str) -> List[List[float]]:
+    pattern = re.compile(r"\{[^{}]*\"bbox_2d\"\s*:\s*\[([^\]]+)\][^{}]*\}")
+    boxes: List[List[float]] = []
+    for match in pattern.finditer(full_response):
+        coords_text = match.group(1)
+        try:
+            coords = [float(x.strip()) for x in coords_text.split(",")]
+        except ValueError:
+            continue
+        if len(coords) == 4:
+            boxes.append(coords)
+    return boxes
+
+
+def draw_grounding_steps(
+    image_path: Path,
+    boxes: Iterable[Sequence[float]],
+    resized_size: tuple[int, int],
+    output_dir: Path,
+) -> List[Path]:
+    image = Image.open(image_path).convert("RGB")
+    width, height = image.size
+    resized_width, resized_height = resized_size
+
+    colours = [
+        "#ff6b6b",
+        "#f7b731",
+        "#45aaf2",
+        "#26de81",
+        "#a55eea",
+        "#fd9644",
+        "#2bcbba",
+        "#778ca3",
+    ]
+
+    annotated_paths: List[Path] = []
+    output_dir.mkdir(parents=True, exist_ok=True)
+
+    for idx, bbox in enumerate(boxes, start=1):
+        mapped_bbox = bbox_transform(bbox, resized_width, resized_height, width, height)
+        colour = colours[(idx - 1) % len(colours)]
+
+        annotated = image.copy()
+        drawer = ImageDraw.Draw(annotated)
+        drawer.rectangle(mapped_bbox, outline=colour, width=max(2, min(width, height) // 200))
+        label = f"Step {idx}"
+        text_position = (mapped_bbox[0] + 4, mapped_bbox[1] + 4)
+        drawer.text(text_position, label, fill=colour)
+
+        output_path = output_dir / f"grounding_step_{idx:02d}.png"
+        annotated.save(output_path)
+        annotated_paths.append(output_path)
+
+    return annotated_paths
+
+
+def main() -> None:
+    args = parse_args()
+
+    if not args.input_path.exists():
+        raise FileNotFoundError(f"Input path not found: {args.input_path}")
+
+    args.output_dir.mkdir(parents=True, exist_ok=True)
+
+    question = args.question or input("请输入你的问题: ")
+
+    if is_video(args.input_path):
+        mosaic_path = args.output_dir / f"{args.input_path.stem}_grid.png"
+        print(f"Decoding video and creating {GRID_ROWS}x{GRID_COLS} grid with {args.num_frames} frames...")
+        image_path = create_video_grid(args.input_path, mosaic_path, args.num_frames)
+        print(f"Video grid saved to: {image_path}")
+    else:
+        image_path = args.input_path
+
+    print("Loading processor and vLLM model...")
+    processor = AutoProcessor.from_pretrained(
+        str(args.model_path),
+        min_pixels=args.min_pixels,
+        max_pixels=args.max_pixels,
+    )
+
+    llm = LLM(
+        model=str(args.model_path),
+        device=args.device,
+        gpu_memory_utilization=args.gpu_memory_utilization,
+        dtype=torch.bfloat16,
+        limit_mm_per_prompt={"image": 16, "video": 10},
+        mm_processor_kwargs={
+            "max_pixels": args.max_pixels,
+            "min_pixels": args.min_pixels,
+        },
+        max_model_len=8192 * 4,
+    )
+
+    agent = AgentVLMVLLM(
+        model=llm,
+        processor=processor,
+        temp_dir=str(args.output_dir / "crops"),
+        device=args.device,
+        min_pixels=args.min_pixels,
+        max_pixels=args.max_pixels,
+        temperature=args.temperature,
+        crop_min_pixels=args.crop_min_pixels,
+        crop_max_pixels=args.crop_max_pixels,
+    )
+
+    print("Running agent reasoning...")
+    _, _, full_response, _, _ = agent.process(
+        str(image_path),
+        question,
+        max_iterations=args.max_iterations,
+    )
+
+    print("\n===== 模型输出 =====")
+    print(full_response)
+
+    boxes = extract_grounding_boxes(full_response)
+    if not boxes:
+        print("\n未检测到任何grounding区域。")
+        return
+
+    print(f"\n检测到 {len(boxes)} 个grounding步骤，正在生成可视化...")
+    base_image = Image.open(image_path)
+    resized_height, resized_width = smart_resize(
+        base_image.height,
+        base_image.width,
+        factor=28,
+        min_pixels=args.min_pixels,
+        max_pixels=args.max_pixels,
+    )
+    base_image.close()
+
+    annotated_paths = draw_grounding_steps(
+        image_path=image_path,
+        boxes=boxes,
+        resized_size=(resized_width, resized_height),
+        output_dir=args.output_dir / "grounding",
+    )
+
+    print("Grounding 可视化已保存：")
+    for path in annotated_paths:
+        print(f" - {path}")
+
+
+if __name__ == "__main__":
+    main()
+
