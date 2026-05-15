# Q-VLM 量化核心创新点解析

> **论文**: Q-VLM: Post-training Quantization for Large Vision-Language Models (NeurIPS'24)
> **代码位置**: `D:\量化学习\QVLM\QVLM-main\custom_bitsandbytes`

---

## 量化流程总览

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  FP16 权重   │───>│ 4-bit 权重量化│───>│ 激活值量化   │───>│ 混合精度 MatMul│───>│  FP16 输出   │
│             │    │ NF4/FP4      │    │ QuantAct    │    │ INT8 GEMM    │    │             │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘    └─────────────┘
                          ↓                    ↓                    ↓
                     分块量化              熵指导搜索           异常值补偿
                     blocksize=64         DED 度量              阈值检测
```

---

## 核心创新点 1: NF4 4-bit 权重量化

### NF4 数据类型设计

NF4 (Normal Float 4) 是专为神经网络权重分布设计的 4-bit 数据类型。其设计理念是：
- 假设权重服从标准正态分布 N(0,1)
- 每个量化 bin 在正态分布下具有**相等的面积**
- 最终归一化到 [-1, 1] 范围

### 16 个量化值

```python
NF4 值 = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
     0.07958029955625534,  0.16093020141124725,  0.24611230194568634,
     0.33791524171829224,  0.44070982933044434,  0.5626170039176941,
     0.7229568362236023,   1.0
]
```

### 代码实现位置

```python
# custom_bitsandbytes/bitsandbytes/functional.py:717-765
def get_4bit_type(typename, device=None, blocksize=64):
    if typename == 'nf4':
        # QLoRA 论文中的 NF4 数据类型
        data = [-1.0, -0.6961928009986877, ...]
    elif typename == 'fp4':
        # FP4: 2 位指数 +1 位尾数
        data = [0, 0.0625, 8.0, 12.0, 4.0, 6.0, 2.0, 3.0, ...]
    return data.to(device)
```

---

## 核心创新点 2: 分块量化 (Block-wise Quantization)

### 设计动机

全局限化使用单个缩放因子无法适应权重矩阵中不同区域的分布变化。Q-VLM 采用细粒度的分块量化：

- **blocksize = 64**: 每 64 个参数独立量化
- **嵌套量化**: 对缩放因子本身再进行量化压缩

### 量化流程

```python
# custom_bitsandbytes/bitsandbytes/functional.py:775-853
def quantize_4bit(A, absmax=None, out=None, blocksize=64, compress_statistics=False):
    # 1. 计算分块数量
    n = A.numel()
    blocks = n // blocksize
    blocks += 1 if n % blocksize > 0 else 0

    # 2. 为每块计算 absmax
    absmax = torch.zeros((blocks,), device=A.device, dtype=torch.float32)

    # 3. CUDA 核函数量化
    lib.cquantize_blockwise_fp32_nf4(code, A, absmax, out, blocksize, n)

    # 4. 可选：嵌套量化压缩统计信息
    if compress_statistics:
        offset = absmax.mean()
        absmax -= offset
        qabsmax, state2 = quantize_blockwise(absmax, blocksize=256)
        state = [qabsmax, input_shape, A.dtype, blocksize, [offset, state2], quant_type]
    else:
        state = [absmax, input_shape, A.dtype, blocksize, None, quant_type]

    return out, state
```

### 内存节省

| 组件 | 原始 | 量化后 | 压缩率 |
|------|------|--------|--------|
| 权重 | 2 bytes/param | 0.5 bytes/param | 75% |
| absmax | - | 4 bytes/64 params | ~6% |
| 嵌套 absmax | - | 4 bytes/256 blocks | ~1.5% |

---

## 核心创新点 3: 激活值动态量化 (QuantAct)

### 基于熵的自适应搜索

Q-VLM 提出使用**激活熵**和**分布差异度量 (DED)** 来指导量化参数搜索：

```python
# custom_bitsandbytes/bitsandbytes/quantization_utils/quant_modules.py:127-144
def cal_entropy(self, attn):
    """计算激活熵"""
    attn = torch.nn.functional.normalize(attn, dim=1)
    return -1 * torch.sum((attn * torch.log(attn+1e-7)), dim=1).mean()

def compute_DED(self, p_k, p_k1):
    """计算分布差异度量 D(k, k+1)"""
    joint_p = p_k * p_k1
    condition_p = p_k1 / (p_k + 1e-5)
    return -1 * torch.sum(joint_p * torch.log(condition_p + 1e-5), dim=1).mean()

def search_strategy_judge(self):
    """基于熵的搜索策略判断"""
    if last_layer_entropy >= np.mean(llama_entropy):
        search_flag = True   # 需要搜索
    else:
        search_flag = False  # 跳过搜索
```

### 搜索策略

```
┌─────────────────────────────────────────────────────────────────┐
│                    QuantAct 搜索流程                            │
├─────────────────────────────────────────────────────────────────┤
│  1. Calibrate 阶段：收集激活范围                                │
│         ↓                                                       │
│  2. 计算当前层激活熵                                            │
│         ↓                                                       │
│  3. 与历史熵值比较 (DED 度量)                                    │
│         ↓                                                       │
│  4. 判断是否需要搜索：                                          │
│     - 熵值上升 → 需要搜索                                        │
│     - 熵值稳定 → 跳过搜索                                        │
│         ↓                                                       │
│  5. LP Loss 优化：尝试不同范围，选择最小损失                      │
└─────────────────────────────────────────────────────────────────┘
```

### 代码实现

```python
# custom_bitsandbytes/bitsandbytes/quantization_utils/quant_modules.py:146-179
def calibrate_quantization(self, inputs, init_min=-6, init_max=6):
    if self.llama_layer == True:
        # LLaMA 层：使用熵指导的搜索策略
        self.search_flag = self.search_strategy_judge()

        if self.search_flag:
            # 更新量化范围
            x_min = torch.min(inputs, dim=1)[0].squeeze(dim=0)
            x_max = torch.max(inputs, dim=1)[0].squeeze(dim=0)
            self.llama_range_min += -self.llama_range_min + torch.min(self.llama_range_min, x_min)
            self.llama_range_max += -self.llama_range_max + torch.max(self.llama_range_max, x_max)

        quant_act = self.quantization(inputs, self.llama_range_min, self.llama_range_max)

        # 更新熵统计
        if self.count_layer == 1 or self.count_layer == 7:
            last_layer_entropy = self.cal_entropy(quant_act.abs())
        else:
            last_layer_entropy = self.compute_DED(last_layer_distribution, quant_act.abs())

        return quant_act

    else:
        # CLIP 层：行级搜索
        x_min = torch.min(inputs, dim=-1)[0].squeeze(dim=0)
        x_max = torch.max(inputs, dim=-1)[0].squeeze(dim=0)
        self.CLIP_range_min += -self.CLIP_range_min + torch.min(self.CLIP_range_min, x_min)
        self.CLIP_range_max += -self.CLIP_range_max + torch.max(self.CLIP_range_max, x_max)
        quant_act = self.quantization(inputs, self.CLIP_range_min, self.CLIP_range_max)
        return quant_act
```

---

## 核心创新点 4: 混合精度矩阵乘法 (MatMulLt)

### Double Quantization 流程

```
输入 A (FP16)
    ↓
┌─────────────────────────────────────┐
│ Double Quantization                 │
│  - CA: INT8 量化激活                 │
│  - SCA: 行缩放因子                   │
│  - coo_tensorA: 异常值索引           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 异常值处理                          │
│  - 阈值检测：threshold > 0           │
│  - 异常值维度使用 FP16 独立计算        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ INT8 GEMM (iGEMM)                   │
│  - C32A × CxB → out32              │
│  - Turing/Ampere 格式优化            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Dequantization                      │
│  - out = out32 × SCA × SCB         │
│  - bias 融合                        │
└─────────────────────────────────────┘
    ↓
输出 (FP16)
```

### 代码实现

```python
# custom_bitsandbytes/bitsandbytes/autograd/_functions.py:291-440
class MatMul8bitLt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, out=None, bias=None, state=MatmulLtState):
        # 1. 激活值 Double Quantization
        CA, CAt, SCA, SCAt, coo_tensorA = F.double_quant(
            A.to(torch.float16), threshold=state.threshold
        )

        # 2. 权重变换 (Turing/Ampere 格式)
        if using_igemmlt:
            state.CxB, state.SB = F.transform(CB, to_order=formatB)

        # 3. INT8 GEMM
        C32A, SA = F.transform(CA, "col32")
        out32, Sout32 = F.igemmlt(C32A, state.CxB, SA, state.SB)

        # 4. 反量化得到 FP16 输出
        output = F.mm_dequant(out32, Sout32, SCA, state.SCB, bias=bias)

        # 5. 异常值补偿计算
        if coo_tensorA is not None and subA is not None:
            output += torch.matmul(subA, state.subB)

        return output
```

---

## 量化核心公式

### 非对称线性量化

```
量化：  x_q = round(s × x - z)

反量化：x̂ = (x_q + z) / s

其中:
  s = (2^n - 1) / (max - min)  # 缩放因子
  z = s × min                   # 零点
  n = 量化位数 (4 或 8)
```

### 分块量化

```
每块独立计算缩放因子:
  s_i = (2^n - 1) / max(|x_i|)

其中 x_i 是第 i 块的权重
```

### 激活熵

```
H(X) = -Σ p(x) × log(p(x))

用于衡量激活分布的复杂度，指导量化参数搜索
```

---

## custom_bitsandbytes 代码结构

```
custom_bitsandbytes/
├── bitsandbytes/
│   ├── nn/
│   │   ├── modules.py              # Linear4bit, Linear8bitLt 层定义
│   │   ├── triton_based_modules.py # Triton 实现的层
│   │   └── __init__.py
│   ├── autograd/
│   │   └── _functions.py           # MatMul8bitLt, MatMul4Bit 自动微分
│   ├── quantization_utils/
│   │   ├── quant_modules.py        # QuantAct 激活量化
│   │   └── quant_utils.py          # 量化工具函数
│   ├── triton/
│   │   ├── quantize_rowwise.py     # 行向量量化
│   │   ├── dequantize_rowwise.py   # 行向量反量化
│   │   └── int8_matmul_*.py        # INT8 矩阵乘法
│   ├── functional.py               # 量化/反量化核函数
│   ├── utils.py                    # 工具函数
│   └── __init__.py
└── csrc/
    ├── kernels.cu                  # CUDA 核函数
    └── ops.cu                      # 操作封装
```

### 关键文件行数参考

| 文件 | 功能 | 关键行号 |
|------|------|----------|
| `nn/modules.py` | Linear4bit 层 | 206-315 |
| `nn/modules.py` | Linear8bitLt 层 | 423-512 |
| `functional.py` | 4-bit 量化 | 775-853 |
| `functional.py` | 4-bit 反量化 | 861-936 |
| `autograd/_functions.py` | MatMul8bitLt | 291-491 |
| `autograd/_functions.py` | MatMul4Bit | 494-549 |
| `quantization_utils/quant_modules.py` | QuantAct | 36-277 |

---

## 内存节省效果

| 模型组件 | FP16 | Q-VLM (4-bit) | 节省比例 |
|----------|------|---------------|----------|
| 权重存储 | 2 bytes/param | 0.5 bytes/param | 75% |
| 激活缓存 | FP16 | INT4/INT8 | 50-75% |
| 优化器状态 | 8 bytes/param | 2 bytes/param | 75% |
| **总计 (7B 模型)** | **~14 GB** | **~5 GB** | **~64%** |

---

## 关键创新总结

1. **NF4 权重量化**: 专为 LLM 权重分布设计的 4-bit 数据类型，在极低比特下保持模型精度

2. **分块量化**: blocksize=64 的细粒度分块，配合嵌套量化压缩，减少量化误差

3. **动态激活量化**: 基于熵的自适应范围搜索，根据层间分布变化动态调整量化参数

4. **混合精度 MatMul**: INT8 GEMM + 异常值 FP16 补偿，平衡速度和精度

---

## 参考资源

- **论文**: [Q-VLM: Post-training Quantization for Large Vision-Language Models](https://arxiv.org/abs/2410.08119)
- **代码仓库**: https://github.com/ChangyuanWang17/QVLM
- **QLoRA 论文**: https://arxiv.org/abs/2305.14314
- **bitsandbytes**: https://github.com/bitsandbytes-foundation/bitsandbytes

---

*文档生成时间：2026-03-23*
