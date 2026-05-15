# ARCQuant 创新实验指南

## 项目概述

在 ARCQuant（W4A4 PTQ 量化框架）基础上做三个创新实验。详细实现方案见 `docs/` 目录。

## 硬件要求

- **SM 120+ GPU**（Blackwell: RTX 5090 / B100 / B200），CUDA 12.8+
- A100/H100 无法编译 NVFP4 kernel（CMakeLists 写死 `CMAKE_CUDA_ARCHITECTURES 120a`）

## 代码与模型

- 代码: `ARCQuant/`
- 模型: `/workspace/model/Llama-3.1-8B-Instruct`（hidden=4096, intermediate=14336, 32层）

## Phase 1: 基线复现

```bash
# 1. 环境
cd /workspace/yst_ARCQuant_test
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -r ARCQuant/requirements.txt && uv pip install pybind11 scikit-learn

# 2. 编译 kernel
cd ARCQuant/kernels && bash remake.sh

# 3. 预处理
cd /workspace/yst_ARCQuant_test/ARCQuant
python reorder_indices.py --model /workspace/model/Llama-3.1-8B-Instruct --samples 128 --seqlen 2048 --act_sort_metric max

# 4. 评估基线 PPL
python model/main.py /workspace/model/Llama-3.1-8B-Instruct --act_sort_metric max --dataset wikitext2 --eval_ppl --quant_type NVFP4
```

## Phase 2-4: 创新实验

详细方案见：
- @docs/innovation_1_dp_budget.md - Layer-level S 预算分配（DP 优化）
- @docs/innovation_2_gptq.md - GPTQ 权重量化集成
- @docs/innovation_3_kv_cache.md - 残差补偿扩展到 KV Cache

## 关键代码位置

| 功能 | 文件 | 行号 |
|------|------|------|
| S 计算（阈值法） | utilize.py | 547-549 |
| 通道重排索引 | utilize.py | 22-64 |
| 激活统计收集 | utilize.py | 80-310 |
| 权重量化（NVFP4） | model/qLinearLayer.py | 25-28, 55 |
| 权重量化（fake） | model/quantize.py | 287-312 |
| 激活量化（fake） | model/quantize.py | 314-343 |
| KV Cache 量化 | model/qLlamaLayer.py | 311-312, 338-339 |
| 模型装配 | model/model_utils.py | 20-48 |
| PPL 评估 | model/eval.py | 14-86 |

## avg_bits 公式

```
avg_bits = 4.5 × (in_features + select_num) / in_features
```
