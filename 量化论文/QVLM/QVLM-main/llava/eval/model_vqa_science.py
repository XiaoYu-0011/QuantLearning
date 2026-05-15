import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, AutoConfig, LlamaForCausalLM

from PIL import Image
import math
import random
import numpy as np

# =============================================================================
# 【关键量化导入】QuantAct - 量化激活值模块
# 来源：bitsandbytes 库的量化工具
# 作用：在模型推理前进行校准，收集激活值的动态范围，用于后续的量化计算
# =============================================================================
from bitsandbytes.quantization_utils.quant_modules import QuantAct


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

# =============================================================================
# 量化校准函数
# =============================================================================
# 目的：在正式推理前，使用少量校准样本（默认 8 张图像）来收集激活值的统计信息
# 原理：量化模型需要将浮点激活值映射到低比特整数，需要预先知道激活值的动态范围
# 流程：
#   1. 设置 QuantAct 模块为校准模式 (calibrate=True)
#   2. 运行前向传播，收集激活值的 min/max 值
#   3. 搜索最优的量化参数
#   4. 关闭校准模式，保存量化参数供后续推理使用
# =============================================================================
def run_calibrate(args, tokenizer, model, image_processor):
    print('\n==> start calibrate')

    # 【量化步骤 1】遍历模型所有模块，将 QuantAct 模块设置为校准模式
    # 在校准模式下，QuantAct 会记录经过它的激活值的动态范围（min/max）
    for name, module in model.named_modules():
        if isinstance(module, QuantAct):
            module.set_calibrate(calibrate=True)

    # 加载校准集问题（训练集样本）
    questions = json.load(open(os.path.expanduser(args.question_file_calibrate), "r"))
    # 设置随机种子并打乱顺序，确保校准样本的随机性
    np.random.seed(0)
    np.random.shuffle(questions)
    # 校准样本数量：16 * num_chunks
    num_of_sample = 16 * args.num_chunks
    # 校准图像数量：8 张（经验值，足够收集有代表性的激活值分布）
    calibrate_images = 8
    search_flag = 0  # 搜索状态标志：0=初始校准，1=已收集完 8 张图，2=搜索完成
    questions = get_chunk(questions[:num_of_sample], args.num_chunks, args.chunk_idx)

    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        question = line['conversations'][0]
        gt_ans = line["conversations"][1]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs

        if 'image' in line:
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder_calibrate, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            images = image_tensor.unsqueeze(0).half().cuda()
            image_sizes = [image.size]
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            images = None
            image_sizes = None

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        if search_flag > 0 and images == None:
            # no image
            print("skip")
            continue

        # 【量化步骤 2】在推理模式下运行前向传播
        # torch.inference_mode() 禁用梯度计算，节省内存并加速推理
        # 在 calibrate=True 模式下运行此步骤，QuantAct 会记录激活值的动态范围
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                # image_sizes=image_sizes,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        # 校准流程控制：
        # search_flag=0: 初始校准阶段，收集激活值范围
        # search_flag=1: 处理完 8 张校准图像后，进入搜索阶段
        # search_flag=2: 搜索完成，校准结束
        if search_flag == 1:
            search_flag += 1
        elif search_flag == 2:
            # Finish search twice - 完成两次搜索后退出
            break
        if i == calibrate_images-1:
            # 【量化步骤 3】处理完 8 张校准图像后，启动量化参数搜索
            # set_search(True) 让 QuantAct 基于收集的激活值范围，搜索最优的量化参数（如缩放因子、零点）
            for name, module in model.named_modules():
                if isinstance(module, QuantAct):
                    module.set_search(search=True)
            print('==> searching!')
            search_flag += 1

    # 【量化步骤 4】校准结束，关闭校准模式
    # 此时 QuantAct 模块已经保存了量化参数，后续推理将使用这些参数进行量化计算
    for name, module in model.named_modules():
        if isinstance(module, QuantAct):
            module.set_calibrate(calibrate=False)
    print('==> end calibrate')


def eval_model(args):
    # Model
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    # =============================================================================
    # 【关键量化步骤】加载 4bit 量化模型
    # =============================================================================
    # load_pretrained_model 函数根据 args.load_4bit 参数决定是否以 4bit 精度加载模型
    # 4bit 量化使用 bitsandbytes 库，将模型权重从 FP32 量化为 4bit 整数
    # 优点：显存占用减少约 8 倍（相比 FP32），可部署在消费级 GPU 上
    # 注意：此时模型权重已量化，但激活值仍为 FP16，需要后续校准来确定激活值的量化参数
    # =============================================================================
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)
    print(model)

    # =============================================================================
    # 【关键量化步骤】执行量化校准
    # =============================================================================
    # 4bit 量化模型的激活值需要特殊处理：
    # - 权重量化是静态的（加载时已确定）
    # - 激活值量化需要动态校准，因为激活值的分布依赖于输入数据
    # run_calibrate 函数使用 8 张校准样本来收集激活值的统计信息，确定量化参数
    # =============================================================================
    # calibrate
    run_calibrate(args, tokenizer, model, image_processor)
    
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        question = line['conversations'][0]
        gt_ans = line["conversations"][1]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs

        if 'image' in line:
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            images = image_tensor.unsqueeze(0).half().cuda()
            image_sizes = [image.size]
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            images = None
            image_sizes = None

        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                # image_sizes=image_sizes,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,)
                # stopping_criteria=[stopping_criteria])

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--image-folder-calibrate", type=str, default="")
    parser.add_argument("--question-file-calibrate", type=str, default="tables/question_train.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    # ==========================================================================
    # 【关键量化参数】加载 4bit/8bit 量化模型
    # ==========================================================================
    # --load-8bit: 使用 8bit 精度加载模型权重
    # --load-4bit: 使用 4bit 精度加载模型权重（推荐，显存占用更低）
    # 这两个参数互斥，通常只使用其中一个
    # 量化加载由 bitsandbytes 库处理，支持 NF4（正态浮点 4bit）和 FP4 格式
    # ==========================================================================
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()

    eval_model(args)
