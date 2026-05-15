#*
# @file Different utility functions
# Copyright (c) Yaohui Cai, Zhewei Yao, Zhen Dong, Amir Gholami
# All rights reserved.
# This file is part of ZeroQ repository.
#
# ZeroQ is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ZeroQ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ZeroQ repository.  If not, see <http://www.gnu.org/licenses/>.
#*

# =============================================================================
# Q-VLM 核心量化模块：QuantAct - 激活值量化器
# =============================================================================
#
# Q-VLM 论文核心创新点:
# ─────────────────────────────────────────────────────────────────────────────
# 1. 分离式量化策略 (Separated Quantization Strategy)
#    - LLaMA 语言层：使用 channel-wise 量化 (per-channel)
#    - CLIP 视觉层：使用 row-wise 量化 (per-row)
#    原因：视觉和语言特征的激活值分布特性不同
#
# 2. 基于熵的校准策略 (Entropy-based Calibration)
#    - 使用香农熵 (Shannon Entropy) 衡量量化前后激活分布的差异
#    - 使用 DED (Distribution Entropy Distance) 衡量层间分布变化
#
# 3. 两阶段校准流程 (Two-stage Calibration)
#    Stage 1: 收集激活值范围 (calibrate=True)
#             使用少量校准样本 (默认 8 张图像) 收集激活值的 min/max
#    Stage 2: 搜索最优量化参数 (search=True)
#             在收集的范围内搜索最优的量化参数，最小化 L0.5 损失
#
# 4. 跨模态误差补偿 (Cross-Modal Error Compensation)
#    - 在 CLIP 层量化搜索时引入熵损失权重
#    - 公式：score = lploss + entropy_weight * entropyloss
#            其中 entropyloss = mean(llama_entropy)
# ─────────────────────────────────────────────────────────────────────────────
#
# 关键数学公式:
# ─────────────────────────────────────────────────────────────────────────────
# 公式 (4) - 香农熵:
#   H(X) = -Σ p(x) * log(p(x))
#   在代码中：cal_entropy(attn) = -mean(sum(attn * log(attn), dim=1))
#
# 公式 (5) - 分布熵距离 (DED):
#   D(k, k+1) = -Σᵢⱼ p(x_q⁽ᵏ⁾, x_q⁽ᵏ⁺¹⁾) * log(p(x_q⁽ᵏ⁺¹⁾ | x_q⁽ᵏ⁾))
#   在代码中：compute_DED(p_k, p_k1)
#
# 公式 (6) - 量化参数搜索:
#   min_{scale,zero} L_p(Q(x; scale, zero), x)  其中 p=0.5
#   在代码中：lp_loss(activ_tmp, inputs_calibrate, p=0.5)
# ─────────────────────────────────────────────────────────────────────────────
# =============================================================================

import torch
import time
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from .quant_utils import *
import sys

# 全局变量：用于存储层间熵值和分布，支持跨层信息传递
last_layer_entropy = 0
last_layer_distribution = torch.Tensor(np.zeros([1,100,4096])).cuda()
llama_entropy = []
llama_distribution = []

class QuantAct(Module):
    """
    【Q-VLM 核心模块】激活值量化器

    Q-VLM 的创新设计:
    ─────────────────────────────────────────────────────────────────────────
    1. 区分 LLaMA 层和 CLIP 层，使用不同的量化策略:
       - LLaMA 层 (llama_layer=True): channel-wise 量化，维度为 4096 (hidden_size)
       - CLIP 层 (llama_layer=False): row-wise 量化，维度为 257 (visual tokens)

    2. 支持校准模式 (calibrate) 和搜索模式 (search):
       - calibrate=True: 收集激活值的动态范围 (min/max)
       - search=True: 在收集的范围内搜索最优量化参数

    3. 使用熵作为量化质量的度量:
       - 对 LLaMA 层，计算每层激活的香农熵
       - 使用 DED 衡量相邻层之间的分布变化
    ─────────────────────────────────────────────────────────────────────────

    参数:
    activation_bit: 激活值量化位数，Q-VLM 中通常使用 16bit 或更低
    running_stat: 是否使用运行统计量 (moving average)，Q-VLM 未使用
    input_dim: 输入特征维度，LLaMA 为 4096，CLIP 为 257
    llama_layer: 是否为 LLaMA 语言层，True 表示语言层，False 表示视觉层
    count_block: 块索引，用于熵策略判断
    count_layer: 层索引，用于熵策略判断
    """
    def __init__(self,
                 activation_bit=16,
                 # full_precision_flag=False,
                 running_stat=False,
                 # beta=0.9,
                 input_dim=4096,
                 llama_layer=True,
                 count_block=1,
                 count_layer=1):
        """
        初始化量化参数
        """
        super(QuantAct, self).__init__()
        self.activation_bit = activation_bit
        self.momentum = 0.99
        # self.full_precision_flag = full_precision_flag
        self.running_stat = running_stat
        self.llama_layer = llama_layer

        self.init_range = 6.  # 初始化范围，经验值
        self.dim = input_dim
        self.count_block = count_block
        self.count_layer = count_layer
        self.search_flag = True
        self.sample_num = 0
        self.last_entropy = 0
        self.first_search = True

        # 【Q-VLM 创新点】为 LLaMA 层和 CLIP 层分别维护独立的量化范围
        if self.llama_layer == True:
            # ========== LLaMA 语言层量化参数 ==========
            # channel-wise 量化，每个 channel 独立的量化范围
            # shape: [4096] 对应 LLaMA 的 hidden_size
            self.llama_range_min = torch.Tensor(-self.init_range * np.zeros(self.dim)).cuda()
            self.llama_range_max = torch.Tensor(self.init_range * np.zeros(self.dim)).cuda()
        else:
            # ========== CLIP 视觉层量化参数 ==========
            # row-wise 量化，每行独立的量化范围
            # shape: [257] 对应 CLIP 的 visual tokens 数量 (1 cls + 256 patch)
            # 注意：CLIP ViT-L/14 的 sequence length 为 257
            CLIP_row_dim = 257 # v1.3
            # CLIP_row_dim = 577 # v1.5 (position_embedding): Embedding(577, 1024)
            self.CLIP_range_min = torch.Tensor(-self.init_range * np.zeros(CLIP_row_dim)).cuda()
            self.CLIP_range_max = torch.Tensor(self.init_range * np.zeros(CLIP_row_dim)).cuda()


        self.group_num = 8

        # 使用非对称量化函数
        self.act_function = AsymmetricQuantFunction.apply
        self._calibrate = False  # 校准模式标志
        self.search = False      # 搜索模式标志

    def set_calibrate(self, calibrate=True):
        self._calibrate = calibrate

    def set_search(self, search=True):
        self.search = search
    
    def quantization(self, inputs, quantization_min, quantization_max):
        scale, zero_point = asymmetric_linear_quantization_params(
            self.activation_bit, quantization_min , quantization_max
        )
        # print(inputs.shape[-1], scale.shape[0], inputs.shape[-1]==scale.shape[0])
        if inputs.shape[-1] == scale.shape[0]:
            # print(inputs.shape, scale.shape)  # torch.Size([8, 638, 4096]) torch.Size([4096])
            new_quant_x = torch.round(scale * inputs - zero_point)
            n = 2**(self.activation_bit - 1)
            new_quant_x_1 = 0.5 * ((-new_quant_x - n).abs() - (new_quant_x - (n - 1)).abs() - 1)
            quant_act = (new_quant_x_1 + zero_point) / scale
            return quant_act
        else:
            new_quant_x = torch.round(scale * inputs.transpose(1,-1) - zero_point)
            n = 2**(self.activation_bit - 1)
            new_quant_x_1 = 0.5 * ((-new_quant_x - n).abs() - (new_quant_x - (n - 1)).abs() - 1)
            quant_act = (new_quant_x_1 + zero_point) / scale
            return quant_act.transpose(1,-1)

    
    def compute_DED(self, p_k, p_k1):
        """
        【Q-VLM 创新点】计算分布熵距离 (Distribution Entropy Distance, DED)

        对应 Q-VLM 论文公式 (4):
        ─────────────────────────────────────────────────────────────────────────
        D(k, k+1) = -Σᵢⱼ p(x_q⁽ᵏ⁾, x_q⁽ᵏ⁺¹⁾) * log(p(x_q⁽ᵏ⁺¹⁾ | x_q⁽ᵏ⁾))

        其中:
        - x_q⁽ᵏ⁾: 第 k 层量化后的激活值
        - x_q⁽ᵏ⁺¹⁾: 第 k+1 层量化后的激活值
        - p(x_q⁽ᵏ⁾, x_q⁽ᵏ⁺¹⁾): 联合概率分布
        - p(x_q⁽ᵏ⁺¹⁾ | x_q⁽ᵏ⁾): 条件概率分布

        DED 的物理意义:
        - 衡量相邻两层之间激活分布的差异程度
        - DED 越大，说明量化导致的分布变化越剧烈
        - Q-VLM 使用 DED 作为量化质量的度量标准
        ─────────────────────────────────────────────────────────────────────────

        参数:
        p_k: 第 k 层激活分布，shape: [batch, seq_len, hidden_dim]
        p_k1: 第 k+1 层激活分布，shape: [batch, seq_len, hidden_dim]

        返回:
        ded: 分布熵距离的标量值

        Q-VLM 中的关键应用:
        - 在 quant_modules.py 第 236 行调用：last_layer_entropy = self.compute_DED(...)
        - 用于判断是否需要搜索更优的量化参数
        """
        # Step 1: L1 归一化，将激活值转换为概率分布
        # 公式：p_normalized = p / Σ|p|
        p_k = F.normalize(p_k, p=1, dim=1)
        p_k1 = F.normalize(p_k1, p=1, dim=1)

        # Step 2: 计算联合概率分布 p(x, y) ≈ p(x) * p(y)
        # 这里使用元素级乘积近似联合分布
        joint_p = p_k * p_k1
        joint_p = joint_p / joint_p.sum(dim=1, keepdim=True)  # 重新归一化

        # Step 3: 计算条件概率分布 p(y|x) = p(y) / p(x)
        # 加 1e-5 防止除零
        condition_p = p_k1 / (p_k + 1e-5)
        condition_p = condition_p / condition_p.sum(dim=1, keepdim=True)  # 重新归一化

        # Step 4: 计算 DED = -E[log p(y|x)] = -Σ p(x,y) * log p(y|x)
        # 这就是互信息 I(X;Y) 的负值形式
        return -1 * torch.sum(joint_p * torch.log(condition_p + 1e-5), dim=1).mean()

    def cal_entropy(self, attn):
        """
        【Q-VLM 核心函数】计算香农熵 (Shannon Entropy)

        对应 Q-VLM 论文公式 (4):
        ─────────────────────────────────────────────────────────────────────────
        H(X) = -Σ p(x) * log(p(x))

        在代码中的实现:
        H(X) = -mean(sum(attn * log(attn), dim=1))

        香农熵的物理意义:
        - 衡量激活分布的不确定性/信息量
        - 熵值越高，表示分布越均匀，信息量越大
        - 熵值越低，表示分布越集中，信息量越小

        Q-VLM 使用熵的两个场景:
        1. 单层熵：衡量量化前后单激活分布的变化 (本函数)
        2. 层间熵：使用 DED 衡量相邻层之间的分布传递 (compute_DED)
        ─────────────────────────────────────────────────────────────────────────

        参数:
        attn: 注意力激活张量，shape: [batch, seq_len, hidden_dim]
              通常是 attention weights 或激活值的绝对值

        返回:
        entropy: 香农熵的标量值

        Q-VLM 中的关键应用:
        - 在 quant_modules.py 第 234 行：last_layer_entropy = self.cal_entropy(quant_act.abs())
        - 在 quant_modules.py 第 273 行：entropy = self.cal_entropy(quant_act.abs()).item()
        - 用于跨模态误差补偿：score = lploss + entropyweight * entropyloss (第 308 行)
        """
        # Step 1: L2 归一化，将激活值转换为单位向量
        # 目的：消除幅值影响，只关注分布形状
        attn = torch.nn.functional.normalize(attn, dim=1)

        # Step 2: 计算香农熵 H(X) = -Σ p(x) * log(p(x))
        # 注意：attn * torch.log(attn+1e-7) 逐项计算 p(x)*log(p(x))
        #      torch.sum(..., dim=1) 对序列维度求和得到每个样本的熵
        #      .mean() 对 batch 维度求平均得到标量熵值
        # 加 1e-7 防止 log(0) 导致 NaN
        return -1 * torch.sum((attn * torch.log(attn+1e-7)), dim=1).mean()

    def search_strategy_judge(self):
        """
        【Q-VLM 创新点】基于熵的搜索策略判断

        Q-VLM 的核心洞察:
        ─────────────────────────────────────────────────────────────────────────
        问题：传统 PTQ 方法对所有层使用统一的量化策略，忽略了层间敏感度差异

        Q-VLM 的观察:
        - 某些层对量化误差更敏感，需要更精细的参数搜索
        - 某些层对量化误差不敏感，可以直接使用收集的量化范围

        解决方案：使用熵作为敏感度度量
        - 如果当前层的熵 >= 平均熵，说明该层信息量大，需要搜索
        - 如果当前层的熵 < 平均熵，说明该层信息量小，可以跳过搜索
        ─────────────────────────────────────────────────────────────────────────

        判断逻辑:
        ─────────────────────────────────────────────────────────────────────────
        search_flag = (last_layer_entropy >= mean(llama_entropy)) OR (count_block % 3 == 1)

        其中:
        - last_layer_entropy: 上一层的熵值 (由 cal_entropy 或 compute_DED 计算)
        - llama_entropy: 所有 LLaMA 层的熵值列表
        - count_block % 3 == 1: 每 3 个 block 强制搜索一次，防止误差累积

        特殊情况:
        - 第一层 (block=1, layer=1) 始终搜索，因为它是误差传递的起点
        - 第一个样本始终搜索，用于建立熵值基准
        ─────────────────────────────────────────────────────────────────────────

        返回:
        search_flag: True 表示需要搜索最优量化参数，False 表示使用收集的范围即可

        Q-VLM 中的关键应用:
        - 在 quant_modules.py 第 221 行调用：self.search_flag = self.search_strategy_judge()
        - 决定第 223-229 行是否执行范围更新操作
        """
        self.sample_num += 1
        global last_layer_entropy, llama_entropy

        # 判断条件 1: 熵比较
        # 如果上一层的熵 >= 所有层的平均熵，说明该层敏感，需要搜索
        if last_layer_entropy >= np.mean(llama_entropy) or self.count_block % 3 == 1:
            search_flag = True
        else:
            # 否则该层不敏感，可以跳过搜索
            search_flag = False

        # 判断条件 2: 边界情况处理
        # 第一层 (block=1, layer=1) 始终搜索，因为它是误差传递的起点
        # 第一个样本始终搜索，用于建立熵值基准
        if (self.count_block == 1 and self.count_layer == 1) or self.sample_num <= 1:
            search_flag = True
            llama_entropy = []  # 重置熵值列表，开始新的校准轮次

        return search_flag

    def calibrate_quantization(self, inputs, init_min=-6, init_max=6):
        """
        【Q-VLM 核心函数】校准量化参数 - 收集激活值范围并执行量化

        这是 Q-VLM 两阶段校准流程的核心实现:
        ─────────────────────────────────────────────────────────────────────────
        Stage 1: 收集激活值范围 (calibrate=True 模式)
            - 使用少量校准样本 (默认 8 张图像)
            - 收集每层激活值的 min/max 动态范围
            - 存储在 llama_range_min/max 或 CLIP_range_min/max 中

        Stage 2: 搜索最优量化参数 (search=True 模式)
            - 在收集的范围内搜索最优的 scale 和 zero_point
            - 最小化 L0.5 损失：min L_p(Q(x), x) where p=0.5
            - 对 CLIP 层引入熵损失权重进行跨模态补偿
        ─────────────────────────────────────────────────────────────────────────

        Q-VLM 的分离式量化策略:
        ─────────────────────────────────────────────────────────────────────────
        | 层类型 | 量化维度 | 范围变量 | 维度大小 | 物理意义          |
        |--------|----------|----------|----------|-------------------|
        | LLaMA  | channel  | llama_range_* | 4096   | hidden_size       |
        | CLIP   | row      | CLIP_range_*    | 257    | visual tokens     |
        ─────────────────────────────────────────────────────────────────────────

        LLaMA 层 (llama_layer=True):
        - 使用 channel-wise 量化，每个 channel 独立的量化范围
        - x_min = torch.min(inputs, dim=1)[0] - 沿序列维度取最小值
        - x_max = torch.max(inputs, dim=1)[0] - 沿序列维度取最大值
        - shape: [4096] 对应 hidden_size

        CLIP 层 (llama_layer=False):
        - 使用 row-wise 量化，每行独立的量化范围
        - x_min = torch.min(inputs, dim=-1)[0] - 沿特征维度取最小值
        - x_max = torch.max(inputs, dim=-1)[0] - 沿特征维度取最大值
        - shape: [257] 对应 visual tokens (1 cls + 256 patch)

        跨模态误差补偿 (CLIP 层特有):
        ─────────────────────────────────────────────────────────────────────────
        score = lploss + entropyweight * entropyloss

        其中:
        - lploss = (activ_tmp - inputs_calibrate).abs().pow(0.5).mean()
        - entropyloss = mean(llama_entropy) - LLaMA 层的平均熵
        - entropyweight = 0.01 - 熵损失权重

        物理意义:
        - 在 CLIP 层量化搜索时，不仅考虑重构误差 (lploss)
        - 还考虑对 LLaMA 层的影响 (entropyloss)
        - 实现跨模态的误差传递建模
        ─────────────────────────────────────────────────────────────────────────

        参数:
        inputs: 输入激活张量，shape: [batch, seq_len, hidden_dim]
        init_min, init_max: 初始量化范围 (未使用，保留接口)

        返回:
        quant_act: 量化后的激活张量

        Q-VLM 中的关键调用:
        - 在 forward() 第 271 行：quant_act = self.calibrate_quantization(inputs_calibrate)
        - 在 forward() 第 316 行：quant_act = self.calibrate_quantization(inputs_calibrate)
        """
        if self.llama_layer == True:
            # ========== LLaMA 语言层量化校准 ==========
            # Step 1: 判断是否需要搜索最优参数
            # 基于熵的搜索策略：敏感层搜索，不敏感层直接使用收集的范围
            self.search_flag = self.search_strategy_judge()

            if self.search_flag:
                # Step 2: 收集激活值范围 (channel-wise)
                # 沿序列维度 (dim=1) 取每个 channel 的最小/最大值
                # inputs shape: [batch, seq_len, 4096]
                # x_min, x_max shape: [4096]
                x_min = torch.min(inputs, dim=1)[0].squeeze(dim=0)
                x_max = torch.max(inputs, dim=1)[0].squeeze(dim=0)

                # Step 3: 更新量化范围 (in-place 操作，节省显存)
                # 取当前范围与新收集范围的交集
                # 公式：range_min = min(range_min, x_min)
                #      range_max = max(range_max, x_max)
                self.llama_range_min += -self.llama_range_min + torch.min(self.llama_range_min, x_min)
                self.llama_range_max += -self.llama_range_max + torch.max(self.llama_range_max, x_max)

            # Step 4: 执行量化
            quant_act = self.quantization(inputs, self.llama_range_min, self.llama_range_max)

            # Step 5: 计算熵值，用于搜索策略判断
            # 第 1 层和第 7 层使用香农熵，其他层使用 DED
            global last_layer_entropy, last_layer_distribution
            if self.count_layer == 1 or self.count_layer == 7:
                # 香农熵：H(X) = -Σ p(x) * log(p(x))
                last_layer_entropy = self.cal_entropy(quant_act.abs())
            else:
                # DED: D(k,k+1) = -Σᵢⱼ p(x_q⁽ᵏ⁾, x_q⁽ᵏ⁺¹⁾) * log(p(x_q⁽ᵏ⁺¹⁾ | x_q⁽ᵏ⁾))
                last_layer_entropy = self.compute_DED(last_layer_distribution, quant_act.abs())
            last_layer_distribution = quant_act.abs()  # 保存当前分布供下一层使用

            # Step 6: 记录熵值到全局列表
            # 用于 search_strategy_judge 中的熵比较
            if not np.isnan(last_layer_entropy.item()):
                llama_entropy.append(last_layer_entropy.item())

            return quant_act
        else:
            # ========== CLIP 视觉层量化校准 ==========
            # row-wise 量化：沿特征维度取每行的最小/最大值
            # inputs shape: [batch, 257, hidden_dim]
            # x_min, x_max shape: [257]
            x_min = torch.min(inputs, dim=-1)[0].squeeze(dim=0)
            x_max = torch.max(inputs, dim=-1)[0].squeeze(dim=0)

            # 更新量化范围 (in-place 操作)
            self.CLIP_range_min += -self.CLIP_range_min + torch.min(self.CLIP_range_min, x_min)
            self.CLIP_range_max += -self.CLIP_range_max + torch.max(self.CLIP_range_max, x_max)

            # 执行量化
            quant_act = self.quantization(inputs, self.CLIP_range_min, self.CLIP_range_max)
            return quant_act
    
    def forward(self, x):
        """
        【Q-VLM 核心函数】前向传播 - 执行激活值量化

        这是 QuantAct 模块的主入口，支持三种工作模式:
        ─────────────────────────────────────────────────────────────────────────
        模式 1: 校准模式 (self._calibrate=True, self.search=False)
            - 仅收集激活值范围，不执行参数搜索
            - 用于两阶段校准的 Stage 1

        模式 2: 搜索模式 (self._calibrate=True, self.search=True)
            - 在收集的范围内搜索最优量化参数
            - 最小化 L0.5 损失或跨模态复合损失
            - 用于两阶段校准的 Stage 2

        模式 3: 推理模式 (self._calibrate=False)
            - 使用校准/搜索得到的固定量化参数
            - 用于实际推理和评估
        ─────────────────────────────────────────────────────────────────────────

        Q-VLM 的两阶段校准流程 (在 model_vqa_science.py 中调用):
        ─────────────────────────────────────────────────────────────────────────
        # Stage 1: 收集激活范围
        for name, module in model.named_modules():
            if isinstance(module, QuantAct):
                module.set_calibrate(calibrate=True)  # 第 356 行
        model(inputs)  # 前向传播收集范围

        # Stage 2: 搜索最优参数
        for name, module in model.named_modules():
            if isinstance(module, QuantAct):
                module.set_search(search=True)  # 第 106 行
        model(inputs)  # 前向传播搜索参数
        ─────────────────────────────────────────────────────────────────────────

        搜索策略详解:
        ─────────────────────────────────────────────────────────────────────────
        LLaMA 层 (第 281-295 行):
            - 搜索范围：7 个候选 (aa in range(7))
            - 候选缩放：new_max = llama_range_max * (1.0 - aa * 0.1)
                        new_min = llama_range_min * (1.0 - aa * 0.1)
            - 损失函数：L0.5 损失 score = lp_loss(activ_tmp, inputs_calibrate, p=0.5)
            - 选择策略：贪心选择 loss 最小的候选

        CLIP 层 (第 297-314 行):
            - 搜索范围：3 个候选 (aa in range(3))
            - 候选缩放：new_max = CLIP_range_max * (1.0 - aa * 0.001)
                        new_min = CLIP_range_min * (1.0 - aa * 0.001)
            - 损失函数：复合损失 score = lploss + entropyweight * entropyloss
            - 跨模态补偿：entropyloss = mean(llama_entropy)
            - 选择策略：贪心选择 score 最小的候选
        ─────────────────────────────────────────────────────────────────────────

        参数:
        x: 输入激活张量，shape: [batch, seq_len, hidden_dim]

        返回:
        quant_act: 量化后的激活张量

        Q-VLM 中的关键应用:
        - 在模型前向传播时被自动调用
        - 每次 forward 都会对激活值进行量化
        """
        percentile = 0.9997  # 分位数阈值 (未使用，保留)
        inputs_calibrate = x.data  # 获取输入数据的副本

        # ========== 校准模式 (Calibration Mode) ==========
        if self._calibrate:
            # 特殊情况：单个 token 直接返回 (可能是 [CLS] 或特殊 token)
            if inputs_calibrate.shape[1] == 1:
                return x
            else:
                global llama_entropy, llama_distribution

                # ----- Stage 1: 首次搜索 - 收集熵值基准 -----
                # 仅在 search=True 且 first_search=True 时执行一次
                if self.search and self.first_search:
                    self.first_search = False
                    if self.llama_layer:
                        # LLaMA 层：收集激活值并计算熵
                        quant_act = self.calibrate_quantization(inputs_calibrate)
                        llama_distribution.append(quant_act)
                        entropy = self.cal_entropy(quant_act.abs()).item()
                        if not np.isnan(entropy):
                            llama_entropy.append(entropy)
                    else:
                        # CLIP 层：直接返回量化结果
                        quant_act = self.calibrate_quantization(inputs_calibrate)
                        return quant_act

                # ----- Stage 2: LLaMA 层参数搜索 -----
                # 搜索最优的 llama_range_min/max，最小化 L0.5 损失
                elif self.search and self.llama_layer == True and self.first_search == False:
                    best_score = 1e+10  # 初始化最优损失
                    best_max = self.llama_range_max  # 初始化最优范围
                    best_min = self.llama_range_min

                    # 贪心搜索：尝试 7 个不同的缩放比例
                    for aa in range(7):
                        # 候选范围：原范围的 100%, 90%, 80%, ..., 40%
                        new_max = self.llama_range_max * (1.0 - (aa * 0.1))
                        new_min = self.llama_range_min * (1.0 - (aa * 0.1))

                        # 执行量化
                        activ_tmp = self.quantization(inputs_calibrate, new_min, new_max)

                        # 计算 L0.5 损失
                        # Q-VLM 关键创新：使用 p=0.5 而非 p=2，对异常值更鲁棒
                        score = lp_loss(activ_tmp, inputs_calibrate, p=0.5, reduction='all')

                        # 更新最优解
                        if score < best_score:
                            best_max = new_max
                            best_min = new_min
                            best_score = score

                    # 保存最优量化参数
                    self.llama_range_max = best_max
                    self.llama_range_min = best_min

                # ----- Stage 2: CLIP 层参数搜索 (带跨模态补偿) -----
                # 搜索最优的 CLIP_range_min/max，最小化复合损失
                elif self.search and self.llama_layer == False and self.first_search == False:
                    best_score = 1e+10  # 初始化最优损失
                    best_max = self.CLIP_range_max  # 初始化最优范围
                    best_min = self.CLIP_range_min

                    # 获取 LLaMA 层的平均熵作为跨模态误差度量
                    entropyloss = np.mean(llama_entropy)
                    entropyweight = 0.01  # 熵损失权重

                    # 贪心搜索：尝试 3 个不同的缩放比例 (更精细，步长 0.001)
                    for aa in range(3):
                        # 候选范围：原范围的 100%, 99.9%, 99.8%
                        new_max = self.CLIP_range_max * (1.0 - (aa * 0.001))
                        new_min = self.CLIP_range_min * (1.0 - (aa * 0.001))

                        # 执行量化
                        activ_tmp = self.quantization(inputs_calibrate, new_min, new_max)

                        # 计算复合损失
                        # lploss: 重构误差 (L0.5 损失)
                        lploss = (activ_tmp - inputs_calibrate).abs().pow(0.5).mean()
                        # score: 重构误差 + 跨模态误差补偿
                        score = lploss + entropyweight * entropyloss

                        # 更新最优解
                        if score < best_score:
                            best_max = new_max
                            best_min = new_min
                            best_score = score

                    # 保存最优量化参数
                    self.CLIP_range_max = best_max
                    self.CLIP_range_min = best_min

                # ----- 推理模式：直接使用保存的量化参数 -----
                else:
                    quant_act = self.calibrate_quantization(inputs_calibrate)
                    return quant_act

        # ========== 推理模式 (Inference Mode) ==========
        # self._calibrate=False 时使用固定的量化参数

        # 特殊情况：单个 token 的 row-wise 量化
        if inputs_calibrate.shape[1] == 1:
            # 将当前激活与历史范围合并，动态更新量化范围
            activation_catrange_min = torch.cat([self.activation_range_min.unsqueeze(dim=0), inputs_calibrate.squeeze(dim=0)], dim=0)
            activation_catrange_max = torch.cat([self.activation_range_max.unsqueeze(dim=0), inputs_calibrate.squeeze(dim=0)], dim=0)

            self.activation_range_min = torch.min(activation_catrange_min, dim=0)[0].squeeze(dim=0)
            self.activation_range_max = torch.max(activation_catrange_max, dim=0)[0].squeeze(dim=0)
            quant_act = self.quantization(x, self.activation_range_min, self.activation_range_max)

            return quant_act
        else:
            # 正常情况：根据层类型选择量化策略
            if self.llama_layer == True:
                # ===== LLaMA 语言层：channel-wise 量化 =====
                # 特殊情况：维度不是 4096 或第 4 层，使用独立范围
                if self.dim != 4096 or self.count_layer == 4:
                    # 为特殊层计算独立的量化范围
                    self.llama_range_min1 = torch.min(inputs_calibrate, dim=1)[0].squeeze(dim=0)
                    self.llama_range_max1 = torch.max(inputs_calibrate, dim=1)[0].squeeze(dim=0)

                    quant_act = self.quantization(x, self.llama_range_min1, self.llama_range_max1)
                    self.activation_range_min = self.llama_range_min1
                    self.activation_range_max = self.llama_range_max1
                else:
                    # 使用校准/搜索得到的标准范围
                    quant_act = self.quantization(x, self.llama_range_min, self.llama_range_max)
                    self.activation_range_min = self.llama_range_min
                    self.activation_range_max = self.llama_range_max

                return quant_act
            else:
                # ===== CLIP 视觉层：row-wise 量化 =====
                quant_act = self.quantization(x, self.CLIP_range_min, self.CLIP_range_max)
                return quant_act


def calibrate(model, loader, device):
    """
    【Q-VLM 核心函数】校准流程入口 - Stage 1: 收集激活值范围

    Q-VLM 两阶段校准流程的 Stage 1 实现:
    ─────────────────────────────────────────────────────────────────────────
    目标：使用少量校准样本 (默认 8 张图像) 收集模型各层的激活值范围

    输入：
    - model: 待量化的 VLM 模型 (LLaVA)
    - loader: 校准数据 DataLoader，包含校准样本
    - device: GPU 设备 ID

    输出：
    - 校准后的模型，其 QuantAct 模块已保存各层的激活范围
      (llama_range_min/max, CLIP_range_min/max)
    ─────────────────────────────────────────────────────────────────────────

    校准流程详解:
    ─────────────────────────────────────────────────────────────────────────
    Step 1: 设置校准模式 (第 661-663 行)
        for name, module in model.named_modules():
            if isinstance(module, QuantAct):
                module.set_calibrate(calibrate=True)

        作用：将所有 QuantAct 模块设置为校准模式，
             在 forward 过程中收集激活值的 min/max 并保存到范围变量中

    Step 2: 准备校准数据 (第 664-671 行)
        - 从 DataLoader 中读取 32 个样本 (4*8)
        - 为什么是 8 张？Q-VLM 论文发现少量样本即可估计激活范围
        - 使用 4*8-1 = 31 次循环，加上初始的 1 个，共 32 个样本

    Step 3: 执行前向传播 (第 672-673 行)
        with torch.no_grad():
            model(inputs)

        作用：在校准模式下运行前向传播，
             触发 QuantAct.forward() → calibrate_quantization()
             收集各层激活范围到 llama_range_min/max 和 CLIP_range_min/max

    Step 4: 关闭校准模式 (第 674-676 行)
        for name, module in model.named_modules():
            if isinstance(module, QuantAct):
                module.set_calibrate(calibrate=False)

        作用：校准完成后关闭校准模式，准备进入 Stage 2 (搜索模式)
    ─────────────────────────────────────────────────────────────────────────

    关键设计决策:
    ─────────────────────────────────────────────────────────────────────────
    1. 为什么使用 8 张图像？
       - Q-VLM 论文发现，激活值范围的估计不需要大量样本
       - 8 张图像已能覆盖大部分激活值的动态范围
       - 更多样本带来的收益递减，但计算成本线性增长

    2. 为什么使用 torch.no_grad()？
       - 校准阶段不需要反向传播
       - 节省显存和计算资源
       - 加快校准速度

    3. 为什么校准后要关闭校准模式？
       - 校准模式用于收集范围，搜索模式用于优化参数
       - 两种模式互斥，需要分别执行
    ─────────────────────────────────────────────────────────────────────────

    参数:
    model: 待量化的 VLM 模型 (LLaVA)，已插入 QuantAct 模块
    loader: PyTorch DataLoader，提供校准样本
            在 model_vqa_science.py 中由 ScienceQADataset 构建
    device: GPU 设备 ID，用于指定在哪个 GPU 上执行校准

    返回:
    model: 校准后的模型，QuantAct 模块已保存各层激活范围

    Q-VLM 中的调用位置:
    - 在 model_vqa_science.py 第 84 行调用:
      model = calibrate(model, val_loader, device)
    """
    print('\n==> start calibrate')

    # Step 1: 设置所有 QuantAct 模块为校准模式
    # 校准模式下，forward 会收集激活值的 min/max 范围
    for name, module in model.named_modules():
        if isinstance(module, QuantAct):
            module.set_calibrate(calibrate=True)

    # Step 2: 准备校准数据 - 从 DataLoader 中读取 32 个样本
    # 从 loader 中读取第一个批次作为初始样本
    inputs = next(iter(loader))
    # use 1 gpu to calibrate (单 GPU 校准)
    inputs = inputs[0].cuda(device, non_blocking=True)

    # 读取剩余 31 个批次，拼接成一个大 batch
    for i in range(4*8-1):  # 4*8-1 = 31 次循环
        inputs1 = next(iter(loader))
        # inputs1, _= next(iter(loader))
        inputs1 = inputs1[0].to(device, non_blocking=True)
        inputs = torch.cat((inputs, inputs1), 0)  # 沿 batch 维度拼接

    # Step 3: 执行前向传播，收集激活范围
    # torch.no_grad() 禁用梯度计算，节省显存
    with torch.no_grad():
        model(inputs)

    # Step 4: 关闭校准模式
    # 校准完成后关闭校准模式，准备进入 Stage 2 (搜索模式)
    for name, module in model.named_modules():
        if isinstance(module, QuantAct):
            module.set_calibrate(calibrate=False)

    print('==> end calibrate')
    return model


def find_scale_by_percentile_min(x, percentile=0.9999):
    """
    【辅助函数】基于分位数查找量化范围下界

    功能：使用分位数方法找到激活值的最小值（下界）
    用于异常值过滤：忽略极端小的异常值，使用统计上更稳健的分位数

    参数:
    x: 输入张量
    percentile: 分位数阈值，默认 0.9999
                表示忽略最小的 0.01% 的极端值

    返回:
    第 (1-percentile) 分位数的值作为下界
    """
    x_cpu = x.flatten().detach().cpu().numpy()
    max_k = int(x_cpu.size * (1 - percentile))
    # print(max_k)
    return np.partition(x_cpu, max_k)[max_k]

def find_scale_by_percentile_max(x, percentile=0.9999):
    """
    【辅助函数】基于分位数查找量化范围上界

    功能：使用分位数方法找到激活值的最大值（上界）
    用于异常值过滤：忽略极端大的异常值，使用统计上更稳健的分位数

    参数:
    x: 输入张量
    percentile: 分位数阈值，默认 0.9999
                表示使用第 99.99% 分位数作为上界，忽略 0.01% 的极端大值

    返回:
    第 percentile 分位数的值作为上界

    数学原理:
    ─────────────────────────────────────────────────────────────────────
    np.partition(x, k) 将数组重新排列，使得:
    - 前 k 个元素是最小的 k 个元素
    - 第 k 个元素是第 k 小的元素
    - 后面的元素是剩余的元素

    例如：percentile=0.9999, x.size=10000
         max_k = 10000 * 0.9999 = 9999
         np.partition(x, 9999)[9999] = 第 9999 小的元素 = 第 99.99% 分位数
    ─────────────────────────────────────────────────────────────────────
    """
    x_cpu = x.flatten().detach().cpu().numpy()
    max_k = int(x_cpu.size * percentile)
    # print(max_k)
    return np.partition(x_cpu, max_k)[max_k]
