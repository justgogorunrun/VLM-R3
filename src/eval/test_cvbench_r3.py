
from datetime import datetime
from openai import OpenAI
import sys
from pathlib import Path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))
from src.model.r3_vllm import AgentVLMVLLM
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

# ======================
# Configuration
# ======================

DATA_DIR = "/path/to/CV-Bench/"
TEST_FILE = "/path/to/CV-Bench/test_2d.jsonl"
MODEL_PATH = "/path/to/model/checkpoint"
TEMP_DIR = "./crops_cvbench_test"
VLLM_DEVICE = "cuda:0"

SEED = 42
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
API_BASE_URL = os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
VERIFY_MODEL_NAME = "qwen2.5-72b-instruct"

TEMPERATURE = 0.0
MIN_PIXELS = 32*28*28
MAX_PIXELS = 8192*28*28
CROP_MIN_PIXELS = 32*28*28
CROP_MAX_PIXELS = 4096*28*28

# ======================
# Initialization
# ======================
random.seed(SEED)
processor = AutoProcessor.from_pretrained(MODEL_PATH, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)


llm = LLM(
    model=MODEL_PATH,
    device=VLLM_DEVICE,
    gpu_memory_utilization=0.9,
    dtype=torch.bfloat16,
    limit_mm_per_prompt={"image": 16, "video": 10},
    mm_processor_kwargs={
        "max_pixels": MAX_PIXELS,
        "min_pixels": MIN_PIXELS,
    },
    max_model_len=8192 * 4,
)

processor = AutoProcessor.from_pretrained(
    MODEL_PATH, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
agent = AgentVLMVLLM(
    model=llm,
    processor=processor,
    temp_dir=TEMP_DIR,
    device=VLLM_DEVICE,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
    temperature=TEMPERATURE,
    crop_min_pixels=CROP_MIN_PIXELS,
    crop_max_pixels=CROP_MAX_PIXELS,
)


client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
def evaluate_answer_similarity(student_answer, ground_truth, problem):
    """Use llm to evaluate answer similarity."""
    try:
        response = client.chat.completions.create(
            model=VERIFY_MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a evaluation expert. Compare the student's answer with the correct answer. Output ONLY '1.0' if the student's answer matches the correct answer in meaning, or '0.0' if the student's answer does not contain a correct answer. No other output is allowed.
                    Question: {problem}\nStudent's answer: {student_answer}\nCorrect answer: {ground_truth}\nOutput only 1.0 or 0.0:"""
                }
            ],
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        if "1.0" not in result and "0.0" not in result:
            print(f"Unexpected response from GPT: {result}")
            result = 0.0
        return float(result)
    
    except Exception as e:
        print(f"Error in GPT evaluation: {e}")
        # If API call fails, fall back to simple text matching
        return 1.0 if student_answer ==ground_truth else 0.0
def acc_verifier(content, sol, problem,  **kwargs):
    # Extract answer from content if it has think/answer tags
    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
    
    # Extract answer from content if it has think/answer tags
    content_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
    student_answer = content_matches[-1].strip() if content_matches else content.strip().split('\n')[-1] # by hyr

    gt_list = ground_truth.split('/')
    gt_list = [item.strip() for item in gt_list]
    for gt in gt_list:
        print(f"## DEBUG: student answer: {student_answer}, ground truth: {gt}")
        if gt.lower() == student_answer.lower():
            return 1.0
        elif len(gt)==1 and len(student_answer)==1 and gt.lower() != student_answer.lower():
            return 0.0
    problem = problem.replace("Answer the question with Yes or No.", "")
    problem = problem.replace("Answer the question using a single word or phrase.", "")
    problem = problem.replace("Answer with the option's letter from the given choices.", "")
    problem = problem.strip()
    return evaluate_answer_similarity(student_answer, ground_truth, problem)




# ======================
# Load Dataset
# ======================
with open(TEST_FILE, "r") as f:
    lines = f.readlines()

random.shuffle(lines)
print(f"Total samples: {len(lines)}")

# ======================
# Evaluation Loop
# ======================
correct_num = 0
total_num = 0

with open(f"test_cvbench_log_{timestamp}.txt", "w") as log_file:
    log_file.write(f"\n----- Evaluation Start at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -----\n")
    for cnt, sample_str in enumerate(lines):
        try:
            sample = json.loads(sample_str)
            prompt = sample["prompt"]
            filename = os.path.join(DATA_DIR, sample["filename"])
            answer = sample["answer"]

            image = Image.open(filename)

            log_file.write(f"Sample {cnt}:\nPrompt: {prompt}\nImage path: {filename}\nAnswer: {answer}\nResponse:\n")
            print(f"Sample {cnt}:\nPrompt: {prompt}\nImage path: {filename}\nAnswer: {answer}\nResponse:\n")

            _, _, full_response, _, _ = agent.process(filename, prompt)

            log_file.write(full_response + "\n")
            print(full_response + "\n")

            r1 = acc_verifier(full_response, answer, prompt)
            if r1 == 1.0:
                correct_num += 1

            log_file.write("-" * 80 + "\n")
            print("-" * 80 + "\n")

            total_num += 1
            print(f"Current accuracy: {correct_num / total_num:.4f}")

        except Exception as e:
            log_file.write(f"Error processing sample {cnt}: {e}\n")
    log_file.write("model_path: {}\n".format(MODEL_PATH))
    log_file.write(f"Final accuracy: {correct_num / total_num:.4f}\n")

print(f"Final accuracy: {correct_num / total_num:.4f}")

