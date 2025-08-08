import re
class EvalAIAnswerProcessor:
    """
    Processes an answer similar to Eval AI
        copied from
        https://github.com/facebookresearch/mmf/blob/c46b3b3391275b4181567db80943473a89ab98ab/pythia/tasks/processors.py#L897
    """

    CONTRACTIONS = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        "Im": "I'm",
        "Ive": "I've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "let's": "let's",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }

    NUMBER_MAP = {
        "none": "0",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    ARTICLES = ["a", "an", "the"]
    PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
    COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
    PUNCTUATIONS = [
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    ]

    def __init__(self, *args, **kwargs):
        pass

    def word_tokenize(self, word):
        word = word.lower()
        word = word.replace(",", "").replace("?", "").replace("'s", " 's")
        return word.strip()

    def process_punctuation(self, in_text):
        out_text = in_text
        for p in self.PUNCTUATIONS:
            if (p + " " in in_text or " " + p in in_text) or (re.search(self.COMMA_STRIP, in_text) is not None):
                out_text = out_text.replace(p, "")
            else:
                out_text = out_text.replace(p, " ")
        out_text = self.PERIOD_STRIP.sub("", out_text, re.UNICODE)
        return out_text

    def process_digit_article(self, in_text):
        out_text = []
        temp_text = in_text.lower().split()
        for word in temp_text:
            word = self.NUMBER_MAP.setdefault(word, word)
            if word not in self.ARTICLES:
                out_text.append(word)
            else:
                pass
        for word_id, word in enumerate(out_text):
            if word in self.CONTRACTIONS:
                out_text[word_id] = self.CONTRACTIONS[word]
        out_text = " ".join(out_text)
        return out_text

    def __call__(self, item):
        item = self.word_tokenize(item)
        item = item.replace("\n", " ").replace("\t", " ").strip()
        item = self.process_punctuation(item)
        item = self.process_digit_article(item)
        return item
eval_ai_processor = EvalAIAnswerProcessor()


import pathlib
import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer, Qwen2VLGRPOVLLMTrainerModified,Qwen2VLGRPOVLLMTrainerVanillaRL
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
import json
from datasets import Dataset
import random
import numpy as np
import torch
random.seed(42) 

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

from openai import OpenAI
client = OpenAI(
    api_key="sk-e797527fabb94987b072cda08ace813e",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
) # by hyr

def accuracy_reward_old(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards

def evaluate_answer_similarity_sk(student_answer, ground_truth):
    if student_answer is None or student_answer == "" or student_answer.lower().startswith("$country") or "administrative_area" in student_answer.lower():
        print(f"Student answer: {student_answer}, Ground truth: {ground_truth}, Reward: 0.0")
        return 0.0
    ground_truth = ground_truth.strip()
    student_answer = student_answer.strip()
    ground_truth = ground_truth.replace("$", "")
    student_answer = student_answer.replace("$", "")
    # tmp = student_answer.split(',')
    # if len(tmp) != 2:
    #     return 0.0
    try:
        response = client.chat.completions.create(
            model="qwen2.5-72b-instruct",# by hyr
            # model="qwen-plus", 
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                { "role": "user",
                    "content": f"""Compare the student's answer with the correct solution in the format 'country,administrative_area'.
                    Rules:
                    - Output "1.0" if both country and administrative area are correct
                    - Output "0.5" if only one of the country and administrative area is correct
                    - Output "0.0" if neither the country nor the administrative region is correct, or if the student's answer is null
                    
                    Student's answer: {student_answer}\nCorrect solution: {ground_truth}
                    Output only 1.0, 0.5 or 0.0:"""
                }
            ],
            temperature=0
        )
        result = response.choices[0].message.content.strip()
        # print("## DEBUG: llm reward result : ", result)
        if "1.0" in result:
            reward = 1.0
        elif "0.5" in result:
            reward = 0.5
        elif "0.0" in result:
            reward = 0.0
        else:
            print(f"Unexpected response from GPT: {result}")
            reward = 0.0  # Changed default to 0.0 for unexpected responses
    except Exception as e:
        print(f"Error in GPT evaluation: {e}")
        # Fall back to basic string matching
        if student_answer == ground_truth:
            reward = 1.0
        else:
            # Parse and compare parts individually
            try:
                student_parts = student_answer.split(',')
                truth_parts = ground_truth.split(',')
                if len(student_parts) == 2 and len(truth_parts) == 2:
                    matches = sum(1 for i in range(2) if student_parts[i] == truth_parts[i])
                    reward = matches * 0.5
                else:
                    reward = 0.0
            except:
                reward = 0.0
    if reward == 0.0:
        reward = 0.1  # 只要不是空值就给0.1
    print(f"Student answer: {student_answer}, Ground truth: {ground_truth}, Reward: {reward}")
    return reward
    
def llm_reward_sk(content, sol, **kwargs):
    # Extract answer from content if it has think/answer tags
    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
    
    # Extract answer from content if it has think/answer tags
    content_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
    # student_answer = content_matches[-1].strip() if content_matches else content.strip()
    if content_matches:
        student_answer = content_matches[-1].strip()
        r =  evaluate_answer_similarity_sk(student_answer, ground_truth)
    else:
        r = 0.0
    return r

def sw_acc_reward(content, sol):
    # Extract answer from content if it has think/answer tags
    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
    
    # Extract answer from content if it has think/answer tags
    content_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
    # student_answer = content_matches[-1].strip() if content_matches else content.strip()
    if content_matches:
        student_answer = content_matches[-1].strip()
        student_answer = student_answer.replace("$", "").lower()
        ground_truth = ground_truth.replace("$", "").lower()
        gt_country = ground_truth.split(',')[0].strip()
        gt_area = ground_truth.split(',')[1].split('/')
        gt_area = [area.strip() for area in gt_area]
        r = 0.0
        tmp = student_answer.split(',')
        if len(tmp) >=1:
            if tmp[0].strip() == gt_country:
                r += 0.5
        if len(tmp) >= 2:
            if tmp[1].strip() in gt_area:
                r += 1
        print(f"## DEBUG: student answer: {student_answer}, ground truth: {ground_truth}, reward: {min(r, 1.0)}")
    else:
        r = 0.0
    r = max(0.0, min(r, 1.0))
    return r

from open_r1.utils.math import compute_score
def math_reward(content, sol, **kwargs):
    content = clean_text(content)
    sol = clean_text(sol)
    return compute_score(content, sol)
def clean_text(text, exclue_chars=['\n', '\r']):
    # Extract content between <answer> and </answer> if present
    answer_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_matches:
        # Use the last match
        text = answer_matches[-1]
    
    for char in exclue_chars:
        if char in ['\n', '\r']:
            # If there is a space before the newline, remove the newline
            text = re.sub(r'(?<=\s)' + re.escape(char), '', text)
            # If there is no space before the newline, replace it with a space
            text = re.sub(r'(?<!\s)' + re.escape(char), ' ', text)
        else:
            text = text.replace(char, ' ')
    
    # Remove leading and trailing spaces and convert to lowercase
    return text.strip().rstrip('.').lower()

cnt_llm = 0
def evaluate_answer_similarity(student_answer, ground_truth, problem):
    """Use llm to evaluate answer similarity."""
    global cnt_llm
    try:
        response = client.chat.completions.create(
            model="qwen2.5-72b-instruct",
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
        cnt_llm += 1
        print(f"## DEBUG: llm call times : {cnt_llm}")
        return float(result)
    
    except Exception as e:
        print(f"Error in GPT evaluation: {e}")
        # If API call fails, fall back to simple text matching
        return 1.0 if student_answer ==ground_truth else 0.0

def llm_reward(content, sol, problem, **kwargs):
    # Extract answer from content if it has think/answer tags
    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
    
    # Extract answer from content if it has think/answer tags
    content_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
    student_answer = content_matches[-1].strip() if content_matches else content.strip()
    return evaluate_answer_similarity(student_answer, ground_truth, problem)

def textvqa_reward(content, sol, problem,  **kwargs):
    # Extract answer from content if it has think/answer tags
    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
    
    # Extract answer from content if it has think/answer tags
    content_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
    student_answer = content_matches[-1].strip() if content_matches else content.strip().split('\n')[-1] # by hyr

    gt_list = ground_truth.split('/')
    gt_list = [item.strip().lower() for item in gt_list]
    student_answer = student_answer.lower()
    for gt in gt_list:
        gt = eval_ai_processor(gt)
        student_answer = eval_ai_processor(student_answer)
        print(f"## DEBUG: student answer: {student_answer}, ground truth: {gt}")
        if gt == student_answer:
            return 1.0
        if gt=="yes" or gt=="no":
            if gt in student_answer and (not("yes" in student_answer and "no" in student_answer)) and len(student_answer)<48:
                return 1.0
    return evaluate_answer_similarity(student_answer, ground_truth, problem)

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
        if gt == student_answer:
            return 1.0
        elif len(gt)==1 and len(student_answer)==1 and gt.lower() != student_answer.lower():
            return 0.0
    return evaluate_answer_similarity(student_answer, ground_truth, problem)

def accuracy_reward(completions, solution, source, problem,  **kwargs):
    """Reward function that checks if the completion is correct using symbolic verification, exact string matching, or fuzzy matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol, sc, pro in zip(contents, solution, source, problem):
        if sc == "MMK12":
            reward = math_reward(content, sol)
        elif sc == "seekworld":
            reward = sw_acc_reward(content, sol)
        elif sc == "textvqa":
            reward = textvqa_reward(content, sol, pro)
        else:
            reward = acc_verifier(content, sol, pro)
        # # 加一点随机扰动,正太分布
        # reward += random.gauss(0, 0.0001)
        reward = max(0.0, min(reward, 1.0))
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                f.write(f"------------- {current_time} Accuracy reward: {reward}  Source: {sc} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards

# 修改的format函数, 只检查<answer>标签
def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r".*?<answer>.*?</answer>\s*" 
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    rewards = [0.5 if match else 0.0 for match in matches]

    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH_FORMAT")
        # local_rank = int(os.getenv("LOCAL_RANK", 0))
        for i, (content, reward) in enumerate(zip(completion_contents, rewards)):
            if rewards[i] == 0.5:
                continue
            with open(log_path, "a", encoding="utf-8") as f:
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                f.write(f"------------- {current_time} Format reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
    return rewards

# def format_reward(completions, **kwargs):
#     """Reward function that checks if the completion has a specific format."""
#     pattern = r"<think>.*?</think>\s*<answer>.*?</answer>\s*"
#     completion_contents = [completion[0]["content"] for completion in completions]
#     rewards = []
#     for content in completion_contents:
#         # 初始化奖励为0
#         reward = 0.0
#         # 检查基本格式
#         match = re.fullmatch(pattern, content, re.DOTALL)
#         if match:
#             # 检查标签数量
#             if (content.count("<think>") == 1 and 
#                 content.count("</think>") == 1 and 
#                 content.count("<answer>") == 1 and 
#                 content.count("</answer>") == 1):
#                 # 检查标签顺序
#                 think_start = content.find("<think>")
#                 think_end = content.find("</think>")
#                 answer_start = content.find("<answer>")
#                 answer_end = content.find("</answer>")
#                 if think_start < think_end < answer_start < answer_end:
#                     reward = 0.5
#         rewards.append(reward)
#         # 记录格式不正确的完成
#         if os.getenv("DEBUG_MODE") == "true" and reward < 0.5:
#             log_path = os.getenv("LOG_PATH_FORMAT")
#             with open(log_path, "a", encoding="utf-8") as f:
#                 current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
#                 f.write(f"------------- {current_time} Format reward: {reward} -------------\n")
#                 f.write(f"Content: {content}\n")
#     return rewards


# def length_reward_old(completions, **kwargs):
#     """thinking process length reward function, 每个字符奖励0.002, 至多0.25"""
#     completion_contents = [completion[0]["content"] for completion in completions]
#     rewards = []
#     for content in completion_contents:
#         # Extract the content before <answer> tag
#         # content = re.sub(r'<answer>.*?</answer>', '', content, flags=re.DOTALL)
#         content = content.split("<answer>")[0]
#         # Remove leading and trailing whitespace
#         content = re.sub(r'user\n|assistant\n|[\n\r]', '', content)
#         # Calculate the length of the content
#         length = len(content)
#         # Calculate the reward based on the length
#         reward = min(0.25, length * 0.002)
#         reward = max(0.0, reward)
#         rewards.append(reward)
#     print("## DEBUG: length rewards: ", rewards)
#     return rewards


def length_reward(completions, **kwargs):
    """
    思考过程长度奖励函数，按以下步骤处理：
    1. 提取<think>...</think>之间的内容
    2. 以{"bbox_2d":[...]}为间隔分段
    3. 计算所有段中最短段的长度，乘以0.01作为奖励（上限0.25）
    """
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content in completion_contents:
        # # 1. 提取<think>...</think>之间的内容
        # think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        # if think_match:
        #     thinking = think_match.group(1).strip()
        # else:
        #     # 如果没有<think>标签，则使用<answer>标签之前的内容
        #     thinking = content.split("<answer>")[0].strip()
        thinking = content.split("<answer>")[0].strip().replace("<think>","").replace("</think>", "")
        
        # 去除空格、换行、回车等无意义字符
        thinking = re.sub(r'\s+', ' ', thinking).strip()
        
        # 2. 以{"bbox_2d":[...]}为间隔分段
        pattern_bbox = r'\{"bbox_2d":\s*\[[^\]]*\]\}'
        segments = re.split(pattern_bbox, thinking)
        
        # 如果没有匹配到bbox，则segments就是[原字符串]
        # 如果匹配到了，则可能包含空字符串

        # 除了首尾只添加非空分段
        cleaned_segments = []
        ll = len(segments)
        for i, segment in enumerate(segments):
            segment = segment.strip()
            if segment or i == 0 or i == ll - 1: 
                cleaned_segments.append(segment)
        
        # 3. 找到所有段中的最小长度，如果没有有效段则长度为0
        if cleaned_segments:
            min_length = min(len(segment) for segment in cleaned_segments)
        else:
            min_length = 0
        
        # 计算奖励，每个字符0.01，最高0.25
        # reward = min(0.25, min_length * 0.01)
        # 计算奖励，每个字符0.01，最高0.10
        reward = min(0.25, min_length * 0.01)
        reward = max(0.0, reward)
        rewards.append(reward)
    
    if os.getenv("DEBUG_MODE") == "true":
        print(f"## DEBUG: length rewards: {rewards}")
    
    return rewards


import json
import re
from datetime import datetime
def repeat_reward(completions, solution, source, problem,  **kwargs):
    """
    奖励函数：检查生成内容中是否有重复的JSON匹配项
    """
    rewards = []
    pattern_json = r"\{.*?\}"
    contents = [completion[0]["content"] for completion in completions]
    for content, sol, sc, pro in zip(contents, solution, source, problem):

        json_matches = re.findall(pattern_json, content, re.DOTALL)
        bonus = 0.0
        json_all_success = True
        setjson = set()
        cntjson = 0
        for jstr in json_matches:
            if "bbox" not in jstr:
                continue
            try:
                js = json.loads(jstr)
                # 解析成功，检查是否重复
                # a = js["scale"]
                b = js["bbox_2d"]
                setjson.add(json.dumps(js, sort_keys=True))
                cntjson += 1
            except Exception as e:
                print(f"Error parsing JSON: {e}")
                json_all_success = False
                break

        if json_all_success:
            bonus = min(1,len(setjson))
            final_score = 0.25*bonus   
        else:
            final_score = 0    # json解析失败
        # final_score -= 0.25*(cntjson-len(setjson)) # 严惩json重复

        if final_score < 0.25:
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH_REPEAT")
                # local_rank = int(os.getenv("LOCAL_RANK", 0))
                with open(log_path, "a", encoding="utf-8") as f:
                    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                    f.write(f"------------- {current_time} repeat reward: {final_score} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
            
        rewards.append(final_score)
        
    return rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "repeat": repeat_reward,
    "length": length_reward,
}


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    data_files = script_args.dataset_name.split(":")
    all_data = []
    for data_file in data_files:
        # data_file 是json文件
        with open(data_file, 'r') as f:
            data = json.load(f)
            for t in data:
                item = {}
                item['image_path'] = [t['image_path']]
                item['problem'] = t["question"]
                item['solution'] = t["answer"]
                item['source'] = t["source"]
                all_data.append(item) # {'image_path': ['/data/jcy/data/data/coco/train2014/COCO_train2014_000000581857.jpg'], 'problem': 'Please provide the bounding box coordinate of the region this sentence describes: the lady with the blue shirt.', 'solution': '[103.93, 299.99, 238.15, 477.41]', 'accu_reward_method': 'default'}

    dataset = Dataset.from_list(all_data)

    def make_conversation_from_jsonl(example):
        if 'image_path' in example and example['image_path'] is not None:
            assert all(os.path.exists(p) for p in example['image_path']), f"Image paths do not exist: {example['image_path']}"
            # Don't load image here, just store the path
            return {
                'image_path': [p for p in example['image_path']],  # Store path instead of loaded image
                'problem': example['problem'],
                'solution': f"<answer> {example['solution']} </answer>",
                'source': example['source'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        *({'type': 'image', 'text': None} for _ in range(len(example['image_path']))),
                        {'type': 'text', 'text': example['problem']}
                    ]
                }]
            }
        
    dataset = dataset.map(make_conversation_from_jsonl, num_proc=8)

    
    trainer_cls = Qwen2VLGRPOVLLMTrainerVanillaRL
    print("using: ", trainer_cls)

    # Split dataset for validation if requested
    splits = {'train': dataset}        
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=splits['train'],
        eval_dataset=splits.get('validation') if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    # trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
