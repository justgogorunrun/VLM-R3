"""Utilities for parsing and visualising reasoning traces produced by AgentVLMVLLM.

Shared helper functions are defined here to keep the notebook-based demo and the
web visualisation builder in sync.  The helpers focus on identifying bounding
boxes emitted by the agent and on packaging reasoning trajectories into a
serialisable structure that can easily be consumed by front-end code.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

BBox = List[float]

# Matches the innermost JSON objects that contain a ``"bbox_2d"`` field.
BBOX_PATTERN = re.compile(r"\{[^{}]*\"bbox_2d\"\s*:\s*\[([^\]]+)\][^{}]*\}")


@dataclass
class ReasoningStep:
    """Container describing a single reasoning step."""

    index: int
    text: str
    bbox: Optional[BBox]


def extract_grounding_boxes(full_response: str) -> List[BBox]:
    """Extract all bounding boxes that appear in the full response string."""

    boxes: List[BBox] = []
    for match in BBOX_PATTERN.finditer(full_response):
        coords_text = match.group(1)
        coords: List[float] = []
        try:
            coords = [float(x.strip()) for x in coords_text.split(",")]
        except ValueError:
            continue
        if len(coords) == 4:
            boxes.append(coords)
    return boxes


def extract_answer_text(full_response: str) -> Optional[str]:
    """Return the text enclosed by ``<answer> ... </answer>`` if present."""

    m = re.search(r"<answer>\s*(.*?)\s*</answer>", full_response, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    # Fallback to simple "Answer: xxx" style patterns.
    m2 = re.search(r"(?:^|\n)\s*(?:Answer|答案)\s*[:：]\s*(.*)", full_response, re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    return None


def _candidate_json_objects(text: str) -> List[dict]:
    """Return JSON objects embedded in *text* in the order they appear."""

    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    objects: List[dict] = []
    for candidate in matches:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        objects.append(parsed)
    return objects


def parse_reasoning_steps(chunks: Iterable[str]) -> List[ReasoningStep]:
    """Parse reasoning *chunks* produced by :meth:`AgentVLMVLLM.process`."""

    steps: List[ReasoningStep] = []
    for idx, chunk in enumerate(chunks, start=1):
        bbox: Optional[BBox] = None
        for obj in reversed(_candidate_json_objects(chunk)):
            bbox_candidate = obj.get("bbox_2d")
            if isinstance(bbox_candidate, Sequence) and len(bbox_candidate) == 4:
                try:
                    bbox = [float(v) for v in bbox_candidate]
                except (TypeError, ValueError):
                    bbox = None
                if bbox is not None:
                    break
        steps.append(ReasoningStep(index=idx, text=chunk.strip(), bbox=bbox))
    return steps


def remap_bbox(bbox: Sequence[float], src_size: Sequence[float], dst_size: Sequence[float]) -> BBox:
    """Map *bbox* from *src_size* to *dst_size* coordinate space."""

    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    if src_w == 0 or src_h == 0:
        return [0.0, 0.0, 0.0, 0.0]
    scale_x = dst_w / src_w
    scale_y = dst_h / src_h
    x1, y1, x2, y2 = bbox
    return [
        round(x1 * scale_x, 2),
        round(y1 * scale_y, 2),
        round(x2 * scale_x, 2),
        round(y2 * scale_y, 2),
    ]


def describe_run_for_export(
    *,
    question: str,
    media_path: Path,
    base_image_path: Path,
    base_size: Sequence[int],
    resized_size: Sequence[int],
    response_chunks: Iterable[str],
    full_response: str,
    media_type: str,
) -> dict:
    """Create a serialisable payload that summarises a reasoning run."""

    base_width, base_height = base_size
    resized_width, resized_height = resized_size
    steps = parse_reasoning_steps(response_chunks)
    remapped_steps = []
    for step in steps:
        mapped_bbox = None
        if step.bbox is not None:
            mapped_bbox = remap_bbox(step.bbox, (resized_width, resized_height), (base_width, base_height))
        remapped_steps.append(
            {
                "index": step.index,
                "text": step.text,
                "bbox": mapped_bbox,
            }
        )

    return {
        "question": question,
        "media_type": media_type,
        "media_path": str(media_path),
        "base_image": {
            "path": str(base_image_path),
            "width": base_width,
            "height": base_height,
        },
        "steps": remapped_steps,
        "full_response": full_response,
        "answer": extract_answer_text(full_response),
    }
