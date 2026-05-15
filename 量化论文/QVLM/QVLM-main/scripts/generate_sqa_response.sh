#!/bin/bash
# =============================================================================
# ScienceQA 数据集推理脚本（带量化校准）
# =============================================================================
# 功能：使用 4bit 量化的 LLaVA 模型对 ScienceQA 测试集进行推理生成答案
# 特点：在正式推理前，会先使用训练集样本进行量化校准，收集激活值的统计信息
# =============================================================================

# 设置只使用第一块 GPU (GPU 0)
CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_science \
    # 模型路径 - 需要替换为实际的量化模型路径
    --model-path <model-path> \
    # 测试集问题文件 - ScienceQA 测试集的 JSON 格式问题
    --question-file /<path>/ScienceQA/data/scienceqa/llava_test_QCM-LEPA.json \
    # 测试集图像目录 - 包含测试样本对应的图片
    --image-folder /<path>/ScienceQA/data/scienceqa/images/test \
    # 校准集问题文件 - 使用训练集样本进行量化校准（收集激活值分布）
    --question-file-calibrate /<path>/ScienceQA/data/scienceqa/llava_train_QCM-LEPA.json \
    # 校准集图像目录 - 训练集图片用于校准
    --image-folder-calibrate /<path>/ScienceQA/data/scienceqa/images/train \
    # 输出答案文件路径 - 生成的模型预测结果将保存为 JSONL 格式
    --answers-file /<path>/LLaVA/results/ScienceQA/LLaVA-vicuna-7B-v1.3-4bit.jsonl \
    # 对话模式 - 使用 LLaVA v1 版本的对话模板
    --conv-mode llava_v1 \
    # 【关键量化参数】加载 4bit 量化模型
    # 启用此选项后，模型权重将以 4bit 精度加载，大幅减少显存占用
    # 实际量化加载逻辑在 load_pretrained_model() 函数中处理
    --load-4bit