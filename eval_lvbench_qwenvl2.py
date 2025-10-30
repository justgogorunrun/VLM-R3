import argparse
import gc
import sys
sys.path.append('/remote-home/zhangkc/data/zhangkc/Qwen2-VL-Finetune')
# sys.path.append('/remote-home/share/haohh/a100_data/data2/temp_zc/LongVA/longva')
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration, HfArgumentParser, Qwen2VLForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from accelerate import Accelerator
import glob
import numpy as np
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
from pathlib import Path
import random
import json
from datasets import load_dataset

from torch.utils.data import Dataset

from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np
import torchvision.transforms as T
from torchvision import transforms
from video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from torchvision.transforms.functional import InterpolationMode
import re, subprocess

# # 设置可见gpu为3
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'


SEED = 24242424
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

prompt_templates = {
    "mistral": {
        "preprompt": "<s>[INST]",
        "postprompt": " [/INST]"
    },
    "vicuna": {
        "preprompt": "<s>A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER:",
        "postprompt": "ASSISTANT:"
    },
    "llama3": {
        "preprompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
        "postprompt": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "qwen2": {
        "preprompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
        "postprompt": "<|im_end|>\n<|im_start|>assistant\n",
    }, 
    "yi": {
        "preprompt": "<|im_start|>system\nAnswer the questions.<|im_end|>\n<|im_start|>user\n",
        "postprompt": "<|im_end|>\n<|im_start|>assistant\n",
    },
}  #postprompt是后缀，preprompt表示是总prompt的前缀  # 后面这个"<|im_end|>\n<|im_start|>assistant\n" 有什么用
# \nAnswer the question using a single word or phrase.
# The color of the bottle cap is
# answer = "Yellow"


def safe_tokenize(tokenizer, text):
    tokenized = tokenizer.encode(text, return_tensors="pt")
    if tokenizer.bos_token != None and len(tokenized) > 0 and tokenized[0, 0] == tokenizer.bos_token_id:
        tokenized = tokenized[:, 1:]
    return tokenized

# 下面这个函数的作用是将answer_embeds拼接到input_embeds的后面，然后将input_embeds的长度补齐到accelerator.num_processes的倍数，然后将input_embeds分成accelerator.num_processes份，然后将每一份输入到模型中，然后将模型的输出
# 拼接起来，然后将拼接起来的输出再拼接起来，然后将拼接起来的输出和answer_ids进行比较，如果相等则返回1，否则返回0。
# answer = "more bet"
def eval_forward(accelerator, model, input_embeds, answer_embeds, pad_id, answer_ids, tokenizer):
    # first append answer_embeds to input_embeds
    prompt_length = input_embeds.shape[1]
    labels_length = answer_embeds.shape[1]
    input_embeds = torch.cat([input_embeds, answer_embeds], dim=1)
    # second pad input_embeds to the multiple of accelerator.num_processes
    pad_tensor = torch.tensor(
        [pad_id]
        * (
            (accelerator.num_processes * 2)
            - input_embeds.shape[1] % (accelerator.num_processes * 2)
        )
    ).unsqueeze(0).unsqueeze(-1).expand(-1, -1, input_embeds.shape[-1]).to(accelerator.device)  # 这个向量用于
    input_embeds = torch.cat([input_embeds, pad_tensor], dim=1)
    position_ids = (
        torch.arange(input_embeds.shape[1]).unsqueeze(0).expand(input_embeds.shape[0], -1)
    ).to(accelerator.device)
    print("accelerator的device是：", accelerator.device)
    accelerator.print(input_embeds.shape)
    prepared = prepare_seq_parallel_inputs(
        "zigzag_ring_attn",
        input_embeds,
        position_ids,
        None,
        accelerator.process_index,
        accelerator.num_processes,
        accelerator.device,
    )  # 
    local_input_embeds = prepared["local_input_ids"]
    local_position_ids = prepared["local_position_ids"]
    with torch.inference_mode():
        logits = model(
            inputs_embeds=local_input_embeds,
            position_ids=local_position_ids,
            use_cache=False,
        ).logits
        pred = logits.argmax(dim=-1)

    # gather all logits using accelerator.gather
    def undo_extract_local(gathered_value, world_size, dim=1):
        value_chunks = gathered_value.chunk(2 * world_size, dim=dim)
        reordered_chunks = [None] * (2 * world_size)
        for i in range(world_size):
            reordered_chunks[i] = value_chunks[i * 2]
            reordered_chunks[2 * world_size - i - 1] = value_chunks[i * 2 + 1]
        return torch.cat(reordered_chunks, dim=dim)

    correct = False

    gathered_logits = accelerator.gather(pred.squeeze(0)).unsqueeze(0)
    # undo extract local on the gathered logits
    pred = undo_extract_local(gathered_logits, accelerator.num_processes)
    pred = pred[:, prompt_length - 1 : prompt_length + labels_length - 1]
    # check if the logits are correct, extract argmax id
    # compare the predicted_ids with the labels
    correct = (pred == answer_ids.to(accelerator.device)).all()
    if  accelerator.is_main_process:
        print(
            "Predicted: ",
            tokenizer.decode(pred.squeeze().tolist()),
            "Answer: ",
            tokenizer.decode(answer_ids.squeeze().tolist()),
        )
        # print id as well
        print(
            "Predicted: ",
            pred.squeeze().tolist(),
            "Answer: ",
            answer_ids.squeeze().tolist(),
        )
    return int(correct)



def inference(args):
    model = args.model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        model_max_length=sys.maxsize,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    accelerator = Accelerator(
        mixed_precision="bf16",
        
    )
    kwargs = {"rope_theta": args.rope_theta} if args.rope_theta is not None else {}
    if "qwen2" in args.model.lower() or "longva" in args.model.lower():
        model = Qwen2ForCausalLM_RingAttn.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
            device_map=accelerator.device,
            **kwargs,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
            device_map=accelerator.device,
            **kwargs,
        )
    tokenizer.pad_token = tokenizer.eos_token
    # remember to remove <s>
    accelerator.print("Preparing Haystack...")
    #haystack_embeddings = load_haystack(args, accelerator)
    assert len(haystack_embeddings) >= args.max_frame_num, "Haystack embeddings are not enough. Max frame {} is not found. Currently only {} frames.".format(args.max_frame_num, len(haystack_embeddings))
    haystack_embeddings = haystack_embeddings[:args.max_frame_num].to(accelerator.device)
    prompt = prompt_templates[args.prompt_template]
    preprompt_embeddings = load_text_embeddings(prompt["preprompt"], tokenizer, model, accelerator, args.replace_double_newline)
    postprompt_embeddings = load_text_embeddings(prompt["postprompt"], tokenizer, model, accelerator, args.replace_double_newline)
    
    needle_dataset = load_dataset(args.needle_dataset)["test"]
    answer_embedding_list = []
    answer_id_list = []
    needle_embedding_list = []
    question_embeding_list = []
    for index, instance in enumerate(needle_dataset):
        answer = instance["answer"]
        question = instance["question"]
        needle_embedding_list.append(torch.load(args.needle_embedding_dir + f"/{index}.pt", map_location="cpu").to(torch.bfloat16).to(accelerator.device))
        answer_embedding_list.append(load_text_embeddings(answer, tokenizer, model, accelerator))
        answer_id_list.append(safe_tokenize(tokenizer, answer))
        question_embeding_list.append(load_text_embeddings(question, tokenizer, model, accelerator))
        
    accelerator.print("Starting Evaluation...")
    model = accelerator.prepare(model)
    model.gradient_checkpointing_enable()
    all_accuries = []
    for num_frames in tqdm(
        range(
            args.min_frame_num, args.max_frame_num + 1, args.frame_interval
        )
    ):
        for depth in np.arange(0, 1 + args.depth_interval, args.depth_interval):
            accuracies = []
            for question_embedding, needle_embedding, answer_embedding, answer_id in zip(question_embeding_list, needle_embedding_list, answer_embedding_list, answer_id_list):
                query_frame_idx = int(depth * num_frames)
                input_frames = torch.cat([haystack_embeddings[:query_frame_idx],needle_embedding.unsqueeze(0), haystack_embeddings[query_frame_idx:num_frames]], dim=0).view(-1, haystack_embeddings.shape[-1]).unsqueeze(0)
                input_emebds = torch.cat([preprompt_embeddings, input_frames,question_embedding, postprompt_embeddings], dim=1)
                correct = eval_forward(
                    accelerator, model, input_emebds, answer_embedding, tokenizer.pad_token_id, answer_id, tokenizer
                )
                gc.collect()
                torch.cuda.empty_cache()
                if accelerator.is_main_process:
                    accuracies.append(correct)
            if accelerator.is_main_process:
                result = {
                    "Num. Frame": num_frames,
                    "Frame Depth": round(depth * 100, -1),
                    "Score": sum(accuracies) / len(accuracies),
                }
                accelerator.print(result)
                all_accuries.append(result)
    if accelerator.is_main_process:
        model_name = args.model.split("/")[-1]
        os.makedirs(f"{args.output_path}/{model_name}", exist_ok=True)
        # save all_accuries as json
        with open(f"{args.output_path}/{model_name}/all_accuracies.json", "w") as f:
            json.dump(all_accuries, f, indent=4)
    return all_accuries, accelerator


def plot(args,  all_accuries):
    df = pd.DataFrame(all_accuries)
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", ["#F0496E", "#EBB839", "#9ad5b3"]
    )

    pivot_table = pd.pivot_table(
        df,
        values="Score",
        index=["Frame Depth", "Num. Frame"],
        aggfunc="mean",
    ).reset_index()  # This will aggregate
    pivot_table = pivot_table.pivot(
        index="Frame Depth", columns="Num. Frame", values="Score"
    )
    # Create the heatmap with better aesthetics
    plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    ax = sns.heatmap(
        pivot_table,
        # annot=True,
        fmt="g",
        vmin=0,
        vmax=1,
        linecolor='white',
        linewidths=1.5, 
        cmap=cmap,
        cbar_kws={"label": "Score"},
    )
    
    # Set the color bar label font size
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(14)
    cbar.ax.tick_params(labelsize=14)

    
    # Define the formatter function
    def thousands_formatter(x, pos):
        if x >= 1000:
            return f'{x/1000:.1f}K'
        return f'{x}'

    context_lengths = pivot_table.columns
    formatted_context_lengths = [thousands_formatter(x, None) for x in context_lengths]

    # More aesthetics
    plt.xlabel("Num. of Frames", fontsize=14)  # X-axis label
    plt.ylabel("Depth Percent", fontsize=14)  # Y-axis label
    plt.xticks(ticks=[i + 0.5 for i in range(len(context_lengths))], labels=formatted_context_lengths, rotation=45, fontsize=14)
    # plt.xticks(rotation=45, fontsize=14)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0, fontsize=14)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area
    # save
    model_name = args.model.split("/")[-1]

    plt.savefig(f"{args.output_path}/{model_name}/heatmap.png")
    # calculate average accuracy
    average_accuracy = df["Score"].mean()
    print(f"Average Accuracy: {average_accuracy}")
    # save as txt
    with open(f"{args.output_path}/{model_name}/avg_accuracy.txt", "w") as f:
        f.write(f"Average Accuracy: {average_accuracy}\n")

# 以上函数都是原本longva为了测试 needle数据集的程序。 现在自己要测试videomme，所以修改和增删inference函数和加载数据集类如下。 这个类 copy from videochat2.
class MME_dataset(Dataset):
    def __init__(self, data_prefix, anno_path, num_segments=16, resolution=224, max_subtitle_len=4096):
        self.data_prefix = data_prefix
        with open(anno_path, 'r') as f:
            self.data_list = json.load(f)
            
        self.num_segments = num_segments
        self.max_subtitle_len = max_subtitle_len
        
        # transform   进行图像预处理，依次对图象进行缩放、裁剪、堆叠、标准化等操作
        crop_size = resolution
        scale_size = resolution
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        self.transform = T.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std) 
        ])
    
    def __str__(self):
        task_dict = {}
        total = 0
        for data in self.data_list:
            if data['duration_category'] not in ans_dict:
                task_dict[data['duration_category']] = {}
            for q in data['questions']:
                if q['task_type'] not in ans_dict[data['duration_category']]:
                    ans_dict[data['duration_category']][q['task_type']] = 0
                ans_dict[data['duration_category']][q['task_type']] += 1
                total += 1

        res = f"There are {len(self.data_list)} videos.\n"
        res += f"There are {total} QAs.\n"
        for k, v in task_dict.items():
            res += f"------{k}------\n"
            for kk, vv in task_dict.items():
                res += f"{kk}: {vv}\n"
                
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices

    def read_frame(self, video_path, bound=None):
        video_path = os.path.join(video_path, str(self.num_segments))
        
        if os.path.exists(video_path):
            frame_list = [p for p in os.listdir(video_path)]
        else:
            raise Exception
            
        images_group = list()
        
        for frame_name in frame_list:
            img = Image.open(os.path.join(video_path, frame_name))
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs
    
    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())  #fromarray函数将numpy数组转换为PIL图像
            #     # 归一化到 [0, 1]
            # image_array = vr[frame_index].asnumpy()
            # image_normalized = (image_array - image_array.min()) / (image_array.max() - image_array.min())

            # # 转换为 PIL 图像
            # image_pil = Image.fromarray((image_normalized * 255).astype(np.uint8))
            # images_group.append(image_pil)

            images_group.append(img)
        
        # # 为了试一下没有视频输入的情况  所以强行用随机值给下面的images——group 随机赋值
        # images_group = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(len(images_group))]
        
        return images_group
        # torch_imgs = self.transform(images_group)  # 好像longva要求的视频输出直接是 asnumpy 所以不进行这个转换了
        
        # return torch_imgs

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer = f"({answer}) {data['options'][ord(answer) - ord('A')][3:]}"
        for idx, c in enumerate(data['options']):
            cur_choice, cur_text = c[0], c[3:]
            question += f"({cur_choice}) {cur_text}\n"
        question = question.rstrip()
        return question, answer

    def __getitem__(self, idx):
        video_name = self.data_list[idx]['url'].split("watch?v=")[1]
        video_path = os.path.join(self.data_prefix, "data", video_name)   #  ！ 这个地方要根据自己的设置更改   # 因为作者把视频存储成16帧了 所以原来这个地方是frames，现在这个地方更改的是适合自己的
        video_path = video_path + '.mp4'
        # We store the videos with only 16 or 32 frames for testing,
        # since directly reading the whold videos cost a lot of time.
        # You can also read the whole video via self.read_video(video_path)
        torch_imgs = self.read_video(video_path)  # 读取视频，从中采帧，然后利用transform进行图像预处理（裁剪、归一化）
        duration_category = self.data_list[idx]['duration']
        qa_list = []
        #print(self.data_list[idx], idx)  #  {'video_id': '001', 'duration': 'short',  'url': 'https://www.youtube.com/watch?v=fFjv93ACGo8', 'videoID': 'fFjv93ACGo8'.}就是每一行..
        
        """ for qa in self.data_list[idx]:  # qa好像就是self.data_list[idx]  应该是个字典  是的，如上
            qa_list.append(self.qa_template(qa)) """

        """ same_video = []
        for video in self.data_list:
            video_id = video['video_id']
            if video_id == self.data_list[idx]['video_id'] and video not in same_video:
                
                same_video.append(video)

        for qa in same_video:  # qa好像就是self.data_list[idx]  应该是个字典  是的，如上
            qa_list.append(self.qa_template(qa)) """
        qa_list.append(self.qa_template(self.data_list[idx]))  
        subtitle = ""
        try:
            subtitle_path = os.path.join(self.data_prefix, "subtitle_vtt", video_name + ".vtt")   # 原来的程序为".vtt" 应该是支持网页视频字幕对应的 但是原数据集文件是 .srt文件 
            if os.path.exists(subtitle_path):
                subtitle = read_vtt_and_concatenate(subtitle_path, model.mistral_tokenizer, self.max_subtitle_len)
        except Exception:
            subtitle = ""
            print(f"Error for {subtitle_path}")
            
        return {
            'subtitle': subtitle,
            'video': torch_imgs, 
            'qa_list': qa_list,
            'duration_category': duration_category,
            'video_id': video_name
        }
    

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

def load_video(video_path, num_segments=8, is_fps = False, return_msg=False, resolution=224):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    
    # 根据is_fps 判断的hi否决定采取 1fps采样， 如果参数为True，则采用1fps采样，否则采用均匀采样
    if is_fps:
        # 按原视频帧率1s1帧采样
        # 计算帧数：
        # 计算采样间隔，即每sample_rate秒采样一帧
        sample_rate = 16
        interval = int(vr.get_avg_fps() * sample_rate)

        # 生成采样帧的索引
        frame_indices = np.arange(0, num_frames, interval)
        
    else:
        # # 均匀采样
        # frame_indices = get_index(num_frames, num_segments)
        # 直接均匀取帧、
        frame_indices = np.linspace(0, num_frames - 1, num_segments, dtype=int)

    # # transform
    # crop_size = resolution
    # scale_size = resolution
    # input_mean = [0.48145466, 0.4578275, 0.40821073]
    # input_std = [0.26862954, 0.26130258, 0.27577711]

    # transform = T.Compose([
    #     GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
    #     GroupCenterCrop(crop_size),
    #     Stack(),
    #     ToTorchFormatTensor(),
    #     GroupNormalize(input_mean, input_std) 
    # ])

    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy())
        images_group.append(img)
    return images_group
    # torch_imgs = transform(images_group)
    # if return_msg:
    #     fps = float(vr.get_avg_fps())
    #     sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
    #     # " " should be added in the start and end
    #     msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
    #     return torch_imgs, msg
    # else:
    #     return torch_imgs

    


def inference_lvbench(args):
    # 参照readme文件中快速调用的 代码改造 而来
    accelerator = Accelerator(
        mixed_precision="bf16",
        
    )
    DEFAULT_IMAGE_TOKEN = "<image>"
    device_map="cuda"
    max_frames_num = args.max_frame_num # you can change this to several thousands so long you GPU memory can handle it :)
    gen_kwargs = {"do_sample": True, "temperature": 0.5, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": 1024}
    
    # you can also set the device map to auto to accomodate more frames
    # tokenizer, model, image_processor, _ = load_pretrained_model(args.model, None, "llava-qwen-Evolutive", device_map="cuda") #-Evolutive
    bnb_model_from_pretrained_args = {} 
    if args.model_name == "qwenvl2":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",   #  "sdpa"
                **bnb_model_from_pretrained_args
            )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",   #  "sdpa"
                **bnb_model_from_pretrained_args
            )
    
    processor = AutoProcessor.from_pretrained(args.model)
    # 设置 任务模型（暂定）
    # model.task = "video_train_caption"
    # model.task = "video_train_qa_with_text"
    model.task = "video_eval_qa_with_text"
    model= model.to(device_map)
    # processor = processor.to(device_map)
    #model.task = "video_train_qa"
    
    # 设置 当前最适合lvbench的 筛选 tokens的阈值参数. 如果不设置的话这个地方的值默认是0.495。 根据在videomme上 测试的最好模型参数， 这个值对于lvbench偏大
    # model.Residual_causalEnhancedModual.config.keep_threshold = 0.488
    
    
    # word_ids = {w: tokenizer.encode(w, add_special_tokens=False)[0] for w in {"yes", "no"}}
    # print(word_ids) # 得到几个选项字母的token id
    # # 打印一下模型结构和参数大小看一下是否有明显错误.  最新训练的保存的参数没有问题
    print(model)
    # for name, param in model.named_parameters():
    #     print(name, param.size(), param, "\n")

    
    prompt = prompt_templates[args.prompt_template]
    videochat2_prompt = "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n"
    videochat2_question_prompt="\nOnly give the best option."
    # # 上面那些是从videochat2中copy的， 下面是从作者自己构建的评估库 的imml的 task 的untils文件中复制过来的
    lmmseval_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
    
    my_prompt = "The global and local features of each video frame will be provided to you later in chronological order. Carefully watch the video."
    if args.my_prompt:
        root_prompt = prompt["preprompt"] + my_prompt + lmmseval_prompt 
    else:
        #root_prompt = prompt["preprompt"] + lmmseval_prompt  
        root_prompt = prompt["preprompt"] + videochat2_prompt + videochat2_question_prompt

    root_prompt = my_prompt + "Respond with only the letter (A, B, C, or D) of the correct option.\n" # 感觉好像得加上这个 回车符号，比不加acc高些？
    print("当前测试下的root_prompt是：", root_prompt)
    
    # 加载视频数据集
        #  加载问题文件  修改路径 
    video_path_root = "/remote-home/zhangkc/data/temp_zc/LVBench/data/videos/00000"  # 这个路径是视频的路径
    anno_path =  "/remote-home/zhangkc/data/temp_zc/LVBench/data/video_info.meta.json" #"your_data_path/Video-MME.json"
    
    
    # 因为这个benchmark在下载的时候把原来的视频的名字变化了 所以中间要多一个找对应视频名字的过程  对应关系就在00000.json文件中
    with open("/remote-home/zhangkc/data/temp_zc/LVBench/data/videos/00000.json", "r") as f:
        video_name_find = json.load(f)

    # You can find the csv files in https://github.com/egoschema/EgoSchema/blob/main/questions.json  # 在全部上
    with open("/remote-home/zhangkc/data/temp_zc/LVBench/data/video_info.meta.json", "r") as f:
        full_data = json.load(f)

    full_lvbench = []
    for data in full_data:
        video_name_orin = data['key']
        for item in video_name_find:
            
            if video_name_orin == item['url'].split('watch?v=')[-1]:
                video_name_new = item['key']
                break
        video = video_name_new + "_h264"+ '.mp4'
        
        for item in data["qa"]:
            question_orin = item["question"]
            question = question_orin.split("\n")[0]
            question_format = f"Question: {question.capitalize()}\nOptions:\n"
            #print(item["question"].split("?\n"))
            if "?\n" in item["question"]:
                question_format += item["question"].split("?\n")[1]
            else:
                # 根据第一次"\n" 符号出现的位置分割item["question"]为两部分
                question_format += item["question"].split("\n", 1)[1]

            #question_format += item["question"].split("?")[1]
            item["question"] = question_format
        
        
        full_lvbench.append({
            'key': data['key'],
            "type": data["type"],
            'video': video,
            "QA": data["qa"]
        })

    

    pred_full = []
    pred_simple = []
    # import pdb;pdb.set_trace()  遍历每一个视频 分别输出答案并最后输出需要的文件
    for idx, example in enumerate(tqdm(full_lvbench)):

        if not os.path.exists(os.path.join("/remote-home/zhangkc/data/temp_zc/LVBench/data/videos/00000", example['video'])):   
            # with open("/remote-home/zhangkc/data/temp_zc/LVBench/data/noexist_videos.txt", "a") as f:
            #     f.write(example['video'] + '\n')
            continue
        
        # question = example['question']
        # sample_set["Q"] = question
        # sample_set["video_name"] = video_path
        # duration_category = example['duration']
        video_path = os.path.join("/remote-home/zhangkc/data/temp_zc/LVBench/data/videos/00000", example['video'])
        print(video_path, os.path.exists(video_path)) 
    
    
    num_frame = max_frames_num
    resolution = 224
    
    
    output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    # 如果路径不存在，创建文件
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        password = "!@#$%^+123456"
        # command = f"sudo -S chmod 777 {args.output_dir}"
        command = f"sudo -u dbcloud_admin -S chmod -R 777 {args.output_dir}"
        # subprocess.run(command, input=f"{password}\n", text=True, shell=True)
        with subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, text=True) as proc:
            proc.communicate(input="!@#$%^+123456\n")
    if not os.path.exists(answers_file):
        password = "!@#$%^+123456"
        # # command = f"sudo -S chmod 777 {answers_file}"
        # create_command = f"sudo -u dbcloud_admin touch {answers_file}"
        # subprocess.run(create_command, input=f"{password}\n", text=True, shell=True)
        # command = f"sudo -u dbcloud_admin -S chmod 777 {answers_file}"
        # subprocess.run(command, input=f"{password}\n", text=True, shell=True)

        # 创建文件的命令
        create_command = f"sudo -u dbcloud_admin touch {answers_file}"
        # 改变文件权限的命令
        chmod_command = f"sudo -u dbcloud_admin chmod -R 777 {args.output_dir}"

        # 执行创建文件的命令
        with subprocess.Popen(create_command, shell=True, stdin=subprocess.PIPE, text=True) as proc:
            proc.communicate(input="!@#$%^+123456\n")

        # 执行改变文件权限的命令
        with subprocess.Popen(chmod_command, shell=True, stdin=subprocess.PIPE, text=True) as proc:
            proc.communicate(input="!@#$%^+123456\n")
        
        # 如果文件不存在，则创建一个空文件
        with open(answers_file, 'w') as file:
            pass  # 这里不需要写入任何内容，只是为了创建文件
    print(f"文件 {answers_file} 已创建。")
    ans_file = open(answers_file, "w")
    
    correct = 0
    total = 0
    res_list = []
    acc_dict = {}
    answer_embedding_list = []
    answer_id_list = []
    video_embedding_list = []
    question_embeding_list = []
    accuracies = []
    for idx, example in enumerate(tqdm(full_lvbench)):
        example['video'] = example['video'].split(".")[0] + '.mp4'
        print(example['video'])
        vid_path = os.path.join("/remote-home/zhangkc/data/temp_zc/LVBench/data/videos/00000", example['video']) # 后面直接是Event-Bench-Videos/youtube
        print(vid_path)
        if not os.path.exists(vid_path):   
            # with open("/remote-home/zhangkc/data/temp_zc/LVBench/data/noexist_videos.txt", "a") as f:
            #     f.write(example['video'] + '\n')
            continue
        
        # # 根据超参数决定是否直接1fps读取视频
        # video  = load_video(vid_path, num_segments=num_frame, is_fps=args.fps, return_msg=True)
        
        
        
        
        # TC, H, W = video.shape
        # video = video.reshape(1, TC//3, 3, H, W).to("cuda")   
        # video = video.squeeze(0)   
        
        qa_idx = 0
        #   因为一个视频对应着多个问题 所以这里要循环 依次解答
        for idx, item in enumerate(example['QA']):
            simple_pred = {}
            question = item['question'] 
            answer = item["answer"]
            
            
            
            
            final_prompt = root_prompt + "\n" + question  + "\n"
            
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": vid_path,
                            "max_pixels": args.max_pixels,
                            # "fps": 1.0,
                            "max_frames": num_frame, # 这个和读取视频有关的 都在 smart_nframes 函数里面 可以通过指定这两个参数影响取帧.(qwen-vl-utils 库的vision_process.py 文件中)
                        },
                        {
                            "type": "text",
                            "text": final_prompt
                        }
                    ]
                }
            ]

            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            video_kwargs = {}
            image_inputs, video_inputs = process_vision_info(messages) # 这里不需要返回video_kwargs  # , return_video_kwargs=True

            inputs = processor(
                    text=[prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    # fps=fps, # 这个参数是专门针对视频设计的 
                    padding=True,
                    return_tensors="pt",
                    **video_kwargs,
                )
            inputs = inputs.to("cuda")
                
            
            with torch.inference_mode():
                if "with_text" in model.task:
                    outputs = model.generate(**inputs, max_new_tokens=256) #, question=question ,k=8 
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
                    ]
                    # output_text = processor.batch_decode(
                    #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    # )  #generate(input_ids, images=[video_tensor],  modalities=["video"], question=question ,k=8,  **gen_kwargs)
                else:
                    outputs = model.generate(**inputs, max_new_tokens=256)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)
                    ]
                    

                
            output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    ) 
            #如果是list就取第一个
            if isinstance(output_text, list):
                output_text = output_text[0]
            
            print(question ,"\n" , "output:", output_text, "\n", "answer:", answer)
            
            outputs = output_text
            
            # candidates = example["candidates"]
            # # 找到answer在candidates中的位置
            # answer_index = candidates.index(answer)
            
            # # 转换成大写字母
            # char_answer = chr(ord('A') + answer_index)
            # print(char_answer)
            
            # 按照统一标准进行对比处理
            # 首先对输出做正则匹配处理. 以下是longva的处理
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

                if len(s.split()) > 10 and not re.search("[ABCD]", s):
                    return ""

                matches = re.search(r"[ABCD]", s)
                if matches is None:
                    return ""
                return matches[0]
            
            def extract_characters_regex_longvu(pred):
                # 这是longvu的 处理
                pred = pred.replace("Answer", "")

                letters = ["A", "B", "C", "D"]

                pred_answer = re.findall("[\(\ \[]*([A-D])[\)\.\ \]]*", pred)

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


            outputs_extra_longva = extract_characters_regex(outputs)
            outputs_extra_longvu = extract_characters_regex_longvu(outputs)
            if outputs == answer or outputs in answer or answer in outputs or outputs_extra_longva in answer or  outputs_extra_longvu in answer:
                accuracies.append(1)
            else:
                accuracies.append(0)
            score = sum(accuracies) / len(accuracies)
            print("运行到当前的score:", score)

            outputs = outputs.strip()
            # new_example = {}
            # new_example['q_id'] = example["q_id"]
            # new_example['question'] = question
            # new_example['gt'] = answer
            # new_example['pred'] = outputs
            # new_example["task"] = example["task"]
            # new_example["candidates"] = example["candidates"]
            example['QA'][idx]["pred"] = outputs
            ans_file.write(json.dumps(example, ensure_ascii=False) + "\n")
            ans_file.flush()

            # ans_file.write(json.dumps(new_example, ensure_ascii=False) + "\n")
            # ans_file.flush()
    ans_file.close()
        
    
def main(args):
    if args.plot_only:
        # load all_accuracies from json
        model_name = args.model.split("/")[-1]
        with open(f"{args.output_path}/{model_name}/all_accuracies.json", "r") as f:
            all_accuracies = json.load(f)
        plot(args, all_accuracies)
    else:
        inference_lvbench(args)
        


if __name__ == "__main__":
    args = argparse.ArgumentParser() # data/temp_zc/LongVA/longva/checkpoints/my_train_total_model_finetune2_lr5e-4_addprompt copy/finetune_mymethod_llava_video_part_1.5B_addprompt/checkpoint-1400
    args.add_argument("--model", type=str, default="/remote-home/zhangkc/data/temp_zc/Qwen2-VL-7B-Instruct")  #/data/temp_zc/haohh_file/checkpoint-41800 试一下自己训练的qwen2-1.5b的模型 /data/temp_zc/LongVA-7B 原本是预训练模型/data/temp_zc/LongVA-7B /data/temp_zc/LongVA/longva/checkpoints/finetune /data/temp_zc/LongVA/longva/checkpoints/my_train2/finetune_mymethod/checkpoint-200
    args.add_argument("--model_name", type=str, default="qwenvl2")
    args.add_argument("--my_prompt", default=False, action="store_true")
    args.add_argument("--max_frame_num", type=int, default=128)
    args.add_argument("--max_pixels", type=int, default=448*448)
    args.add_argument("--fps",  type=bool, default=False)
    args.add_argument("--needle_dataset", type=str, default="lmms-lab/v_niah_needles")
    args.add_argument("--min_frame_num", type=int, default=20)
    args.add_argument("--frame_interval", type=int, default=20)
    args.add_argument("--output_dir", type=str, default="/remote-home/zhangkc/StreamQwen2-VL-Finetune/eval/longvideo_understanding/") # longva7b_output 是复现的训练好的原始模型论文结果 longva_qwen1.5b_eventbench  longva7b_eventbench
    args.add_argument("--output_name", type=str, default="qwenvl27b_lvbench_128frames_rebuttle")
    args.add_argument("--depth_interval", type=float, default=0.1)
    args.add_argument("--num_samples", type=int, default=1)
    args.add_argument("--rope_theta", type=float, default=None)
    
    args.add_argument("--prompt_template", default = "qwen2", type=str)
    args.add_argument("--replace_double_newline", action="store_true")
    args.add_argument("--plot_only", default= False ,action="store_true")
    
    main(args.parse_args())
