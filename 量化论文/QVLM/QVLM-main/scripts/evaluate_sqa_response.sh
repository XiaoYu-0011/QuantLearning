#!/bin/bash
# =============================================================================
# ScienceQA 答案评估脚本
# =============================================================================
# 功能：读取模型生成的答案文件，解析预测结果，计算准确率
# 评估维度：
#   1. 整体准确率
#   2. 按学科分类：自然科学 (NAT)、社会科学 (SCO)、语言 (LAN)
#   3. 按上下文类型：纯文本 (TXT)、含图像 (IMG)、无上下文 (NO)
#   4. 按年级分类：低年级 G1(1-6 年级)、高年级 G7(7-12 年级)
# =============================================================================

# 设置只使用第一块 GPU (GPU 0)
# 注意：此脚本主要进行 CPU 计算（JSON 解析、准确率统计），GPU 设置可能不是必需的
CUDA_VISIBLE_DEVICES=0 python ../llava/eval/eval_science_qa.py \
    # ScienceQA 数据集基础目录 - 包含 problems.json 和 pid_splits.json 等文件
    --base-dir /<path>/ScienceQA/data/scienceqa \
    # 模型预测结果文件 - generate_sqa_response.sh 脚本生成的 JSONL 格式答案
    --result-file /<path>/LLaVA/results/ScienceQA/LLaVA-vicuna-7B-v1.3-4bit.jsonl \
    # 输出详细结果文件 - 包含每道题的详细分析（正确/错误列表）
    --output-file /<path>/LLaVA/results/ScienceQA/test_llava-7b_output.json \
    # 输出统计结果文件 - 包含各维度的准确率统计
    --output-result /<path>/LLaVA/results/ScienceQA/test_llava-7b_result.json 