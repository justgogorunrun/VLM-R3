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
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLRGRPOVLLMTrainer
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
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

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
        elif sc == "textvqa":
            reward = textvqa_reward(content, sol, pro)
        else:
            reward = acc_verifier(content, sol, pro)
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

def length_reward(completions, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content in completion_contents:
        thinking = content.split("<answer>")[0].strip().replace("<think>","").replace("</think>", "")

        thinking = re.sub(r'\s+', ' ', thinking).strip()

        pattern_bbox = r'\{"bbox_2d":\s*\[[^\]]*\]\}'
        segments = re.split(pattern_bbox, thinking)
        
        cleaned_segments = []
        ll = len(segments)
        for i, segment in enumerate(segments):
            segment = segment.strip()
            if segment or i == 0 or i == ll - 1: 
                cleaned_segments.append(segment)
        
        if cleaned_segments:
            min_length = min(len(segment) for segment in cleaned_segments)
        else:
            min_length = 0

        reward = min(0.10, min_length * 0.01)
        reward = max(0.0, reward)
        rewards.append(reward)
    
    if os.getenv("DEBUG_MODE") == "true":
        print(f"## DEBUG: length rewards: {rewards}")
    
    return rewards


import json
import re
from datetime import datetime
def repeat_reward(completions, solution, source, problem,  **kwargs):
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
                b = js["bbox_2d"]
                setjson.add(json.dumps(js, sort_keys=True))
                cntjson += 1
            except Exception as e:
                print(f"Error parsing JSON: {e}")
                json_all_success = False
                break

        if json_all_success:
            bonus = min(1,len(setjson))
            final_score = 0.5*bonus   
        else:
            final_score = 0    

        if final_score < 0.5:
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
        with open(data_file, 'r') as f:
            data = json.load(f)
            for t in data:
                item = {}
                item['image_path'] = [t['image_path']]
                item['problem'] = t["question"]
                item['solution'] = t["answer"]
                item['source'] = t["source"]
                all_data.append(item)

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

    
    # trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    trainer_cls = Qwen2VLRGRPOVLLMTrainer
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
