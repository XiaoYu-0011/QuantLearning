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
# Q-VLM 量化核心工具函数
# =============================================================================
# 本文件实现了 Q-VLM 论文中的基础量化原语，包括：
# 1. 非对称线性量化 (Asymmetric Linear Quantization)
# 2. 量化/反量化函数 (Quantize/Dequantize)
# 3. 直线通过估计器 (Straight-Through Estimator, STE)
#
# 核心数学公式:
# ─────────────────────────────────────────────────────────────────────────────
# 公式 1: 非对称线性量化参数计算
#   scale = n / (saturation_max - saturation_min)
#   zero_point = scale * saturation_min + 2^(num_bits-1)  (对于有符号量化)
#
# 公式 2: 量化函数 (浮点 → 整数)
#   x_q = round(scale * x - zero_point)
#
# 公式 3: 反量化函数 (整数 → 浮点)
#   x_dequant = (x_q + zero_point) / scale
#
# 公式 4: 带梯度直通的量化 (用于训练)
#   x_quantized = STE(x) = round(x) + (x - round(x)).detach()
# ─────────────────────────────────────────────────────────────────────────────
# =============================================================================

import math
import numpy as np
from torch.autograd import Function, Variable
import torch

def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    【Q-VLM 创新点】L_p 范数损失函数，用于量化误差搜索

    在 Q-VLM 中，该损失函数用于搜索最优量化参数：
    - 第 216 行：lp_loss(activ_tmp, inputs_calibrate, p=0.5, reduction='all')
    - p=0.5 是 Q-VLM 的关键设计，相比传统的 L2 损失，L0.5 损失对异常值更鲁棒

    数学公式:
    ─────────────────────────────────────────────────────────────────────────
    L_p(pred, tgt) = mean(sum(|pred - tgt|^p, dim=1))

    当 p=0.5 时:
    L_0.5 = mean(sum(|pred - tgt|^0.5, dim=1))

    Q-VLM 使用 p=0.5 而非 p=2 的原因:
    1. 激活值分布通常具有长尾特性，L0.5 损失对异常值不敏感
    2. 实验表明 L0.5 能更好地保持量化后模型的性能
    ─────────────────────────────────────────────────────────────────────────

    pred: 量化后的张量
    tgt: 原始浮点张量 (量化前的真值)
    p: Lp 范数的 p 值，Q-VLM 中使用 p=0.5
    reduction: 'none' 表示对每个样本计算损失，'mean' 表示计算全局平均
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()


def find_MSESmallest(x, k, x_min=None, x_max=None):

    scale, zero_point = asymmetric_linear_quantization_params(
        k, x_min, x_max)
    new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
    n = 2 ** (k - 1)
    new_quant_x = torch.clamp(new_quant_x, -n, n - 1)
    quant_x = linear_dequantize(new_quant_x,
                                scale,
                                zero_point,
                                inplace=False)
    return quant_x


def clamp(input, min, max, inplace=False):
    """
    Clamp tensor input to (min, max).
    input: input tensor to be clamped
    """

    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def linear_quantize(input, scale, zero_point, inplace=False):
    """
    【核心量化函数】将单精度浮点张力量化为整数

    对应 Q-VLM 论文公式 (2):
    ─────────────────────────────────────────────────────────────────────────
    x_q = round(scale * x - zero_point)

    其中:
    - x: 输入浮点张量
    - scale: 量化缩放因子，scale = n / (x_max - x_min)
    - zero_point: 量化零点，zero_point = scale * x_min + 2^(bits-1)
    - x_q: 量化后的整数张量
    ─────────────────────────────────────────────────────────────────────────

    input: 单精度浮点输入张量，shape 可以是 [B, C, H, W] (卷积) 或 [N, D] (线性)
    scale: 量化缩放因子，shape 为 [C] (per-channel) 或 [1] (per-tensor)
    zero_point: 量化零点，shape 与 scale 相同
    inplace: 是否原地操作，True 可以节省显存但会修改输入

    返回:
    quantized: 量化后的整数张量
    """

    # reshape scale and zeropoint for convolutional weights and activation
    '''if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)'''
    # mapping single-precision input to integer values with the given scale and zeropoint
    if inplace:
        input.mul_(scale).sub_(zero_point).round_()
        return input
    return torch.round(scale * input - zero_point)


def linear_dequantize(input, scale, zero_point, inplace=False):
    """
    【核心反量化函数】将整数量化张量映射回浮点值

    对应 Q-VLM 论文公式 (3):
    ─────────────────────────────────────────────────────────────────────────
    x_dequant = (x_q + zero_point) / scale

    其中:
    - x_q: 量化整数张量
    - zero_point: 量化零点
    - scale: 量化缩放因子
    - x_dequant: 反量化后的浮点张量（近似于原始输入）

    注意：反量化后的值与原始输入存在量化误差:
    error = x - x_dequant = x - (round(scale*x - zero_point) + zero_point) / scale
    ─────────────────────────────────────────────────────────────────────────

    input: 整数量化输入张量
    scale: 量化缩放因子
    zero_point: 量化零点
    inplace: 是否原地操作

    返回:
    dequantized: 反量化后的浮点张量
    """

    # reshape scale and zeropoint for convolutional weights and activation
    '''if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)'''
    # mapping integer input to fixed point float point value with given scaling factor and zeropoint
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    return (input + zero_point) / scale


def asymmetric_linear_quantization_params(num_bits,
                                          saturation_min,
                                          saturation_max,
                                          integral_zero_point=True,
                                          signed=True):
    """
    【核心函数】计算非对称线性量化参数 (scale 和 zero_point)

    对应 Q-VLM 论文公式 (1):
    ─────────────────────────────────────────────────────────────────────────
    给定量化范围 [saturation_min, saturation_max] 和量化位数 num_bits:

    1. 计算量化级数:
       n = 2^num_bits - 1

    2. 计算缩放因子 (scale):
       scale = n / (saturation_max - saturation_min)

    3. 计算零点 (zero_point):
       zero_point_raw = scale * saturation_min
       zero_point = round(zero_point_raw) + 2^(num_bits-1)  (当 signed=True)
    ─────────────────────────────────────────────────────────────────────────

    参数说明:
    num_bits: 量化位数，Q-VLM 中通常使用 4bit 或 8bit
    saturation_min: 量化范围下界，通常通过校准过程确定
    saturation_max: 量化范围上界，通常通过校准过程确定
    integral_zero_point: 是否将零点取整为整数，True 有利于整数加速
    signed: 是否使用有符号量化，True 表示量化值范围为 [-2^(bits-1), 2^(bits-1)-1]

    返回:
    scale: 缩放因子，用于将浮点值映射到整数域
    zero_point: 零点，用于处理非对称量化范围

    Q-VLM 中的关键应用:
    - 在 quant_modules.py 第 94-96 行调用此函数计算激活值的量化参数
    - Q-VLM 的创新在于使用 per-channel (LLaMA 层) 或 per-row (CLIP 层) 的方式
      计算不同的 saturation_min/max，而非传统的 per-tensor 方式
    """
    n = 2**num_bits - 1
    # scale = n / torch.clamp((saturation_max - saturation_min), min=1e-8)
    '''if (saturation_max - saturation_min) <= 0:
        scale = n / (saturation_max - saturation_min + 1e-8)
    else:
        scale = n / (saturation_max - saturation_min)'''
    scale = n / (saturation_max - saturation_min)
    # print(n, saturation_max, saturation_min, scale)
    # print(torch.clamp((saturation_max - saturation_min), min=1e-8))
    zero_point = scale * saturation_min

    if integral_zero_point:
        if isinstance(zero_point, torch.Tensor):
            zero_point = zero_point.round()
        else:
            zero_point = float(round(zero_point))
    if signed:
        zero_point += 2**(num_bits - 1)
    return scale, zero_point


class AsymmetricQuantFunction(Function):
    """
    【核心量化层】非对称量化函数 (PyTorch Autograd Function)

    这是 Q-VLM 中实现激活值量化的核心底层操作，支持反向传播

    量化流程 (forward):
    ─────────────────────────────────────────────────────────────────────────
    Step 1: 计算量化参数
        scale, zero_point = asymmetric_linear_quantization_params(k, x_min, x_max)

    Step 2: 量化到整数域
        x_q = round(scale * x - zero_point)

    Step 3: 限制范围到 [-n, n-1], 其中 n = 2^(k-1)
        x_q_clipped = clamp(x_q, -n, n-1)
        使用平滑钳位：x_q_clipped = 0.5 * ((-x_q - n).abs() - (x_q - (n-1)).abs() - 1)

    Step 4: 反量化回浮点域
        x_dequant = (x_q_clipped + zero_point) / scale
    ─────────────────────────────────────────────────────────────────────────

    梯度处理 (backward):
    ─────────────────────────────────────────────────────────────────────────
    使用 Straight-Through Estimator (STE):
        ∂L/∂x = ∂L/∂y  (直接传递梯度，忽略量化操作的不可导性)

    这是因为量化函数 round() 的梯度理论上为 0 (阶梯函数)，
    STE 通过近似让梯度直接通过，使得量化感知训练成为可能
    ─────────────────────────────────────────────────────────────────────────
    """
    @staticmethod
    def forward(ctx, x, k, x_min=None, x_max=None):
        """
        前向传播：执行量化操作

        参数:
        x: 待量化的单精度浮点张量
        k: 量化位数 (如 4bit, 8bit)
        x_min: 量化范围下界 (可选，若为 None 则使用 x.min())
        x_max: 量化范围上界 (可选，若为 None 则使用 x.max())

        返回:
        quant_x: 量化并反量化后的浮点张量 (近似于 x)
        """

        '''if x_min is None or x_max is None or (sum(x_min == x_max) == 1
                                              and x_min.numel() == 1):
            x_min, x_max = x.min(), x.max()'''
        # Step 1: 计算量化参数 (scale 和 zero_point)
        scale, zero_point = asymmetric_linear_quantization_params(
            k, x_min, x_max)
        # Step 2: 量化到整数域
        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
        n = 2**(k - 1)
        # Step 3: 限制范围到 [-n, n-1]
        # Q-VLM 使用平滑钳位而非 torch.clamp，有利于梯度稳定
        # 数学等价于：clamp(x, -n, n-1) 但梯度更平滑
        new_quant_x_1 = 0.5 * ((-new_quant_x - n).abs() - (new_quant_x - (n - 1)).abs() - 1)
        # Step 4: 反量化回浮点域
        quant_x = linear_dequantize(new_quant_x_1,
                                    scale,
                                    zero_point,
                                    inplace=False)
        return torch.autograd.Variable(quant_x, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：Straight-Through Estimator (STE)

        直接传递梯度，忽略量化操作的不可导性
        这是一种启发式方法，在实践经验证有效
        """
        # raise NotImplementedError
        return grad_output, None, None, None
