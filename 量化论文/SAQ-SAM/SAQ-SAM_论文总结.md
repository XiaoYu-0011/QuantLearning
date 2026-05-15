# SAQ-SAM 论文总结

## 论文基本信息

- **标题**: SAQ-SAM: Semantically-Aligned Quantization for Segment Anything Model
- **发表**: AAAI 2026
- **作者**: Jing Zhang, Zhikai Li✉, Xuewen Liu, Chengzhi Hu, Qingyi Gu✉
- **论文链接**: https://arxiv.org/abs/2503.06515
- **代码仓库**: https://github.com/SAQ-SAM/SAQ-SAM

---

## 1. Motivation（研究动机）

### 1.1 背景
Segment Anything Model (SAM) 作为强大的视觉基础模型，在多种视觉任务中展现出卓越的性能。然而，SAM 模型参数量大、计算成本高，难以在资源受限的设备上部署。后训练量化（Post-Training Quantization, PTQ）作为一种有效的模型压缩技术，可以在不重新训练的情况下显著降低模型大小和计算成本。

### 1.2 问题陈述
现有 PTQ 方法直接应用于 SAM 时面临以下挑战：

1. **激活值分布异常**: SAM 的注意力机制中存在极端的激活异常值（extreme activation outliers），特别是在 ViT-B 模型中更为明显
2. **语义信息丢失**: 传统 PTQ 方法仅关注数值分布的保真度，忽略了量化过程中语义对齐（semantic alignment）的重要性
3. **图像 - 提示交互受损**: SAM 的核心在于图像和提示之间的交互，传统量化方法破坏了这种跨模态交互的语义一致性
4. **多模态分布特性**: K 激活（key activation）呈现双峰分布（bimodal distribution），传统量化器难以有效处理

### 1.3 研究目标
设计一种面向 SAM 的语义对齐量化方法，在低比特（如 4-bit、6-bit）下保持 SAM 的分割性能，同时保持图像 - 提示交互的语义一致性。

---

## 2. Insight（核心洞察）

### 2.1 语义对齐的重要性
作者发现，传统 PTQ 方法专注于最小化数值重建误差（如 MSE 损失），但这并不能保证量化后模型的语义能力。对于 SAM 这样的视觉基础模型，保持**语义一致性**比数值精度更为重要。

### 2.2 注意力焦点重叠（Attention Focus Overlap）
通过观察发现，量化前后模型的注意力图（attention map）的焦点区域重叠程度与最终分割性能高度相关。即使数值误差较大，只要注意力焦点区域保持一致，分割性能就能得到较好保持。

### 2.3 图像 - 提示交互的双向对齐
SAM 的 mask decoder 中，图像 token 和提示 token 之间通过 cross-attention 进行双向交互。作者发现这种交互的对齐需要同时考虑：
- **分布对齐**: 量化前后的特征分布应保持一致
- **语义对齐**: 跨注意力的交互模式应保持语义一致性

### 2.4 极端激活异常值的利用
SAM 中的极端激活异常值通常对应重要的语义信息。与其将这些异常值视为需要抑制的噪声，不如利用它们来指导更激进的量化裁剪（aggressive clipping）。

---

## 3. 解决方案/技术改进/创新点

### 3.1 整体框架

SAQ-SAM 提出了三个核心组件：

1. **Perceptual-Consistency Clipping (PCC)** - 感知一致性裁剪
2. **Prompt-Aware Reconstruction (PAR)** - 提示感知重建
3. **Layer-Skipping Strategy** - 分层跳过策略

![方法概览](./img/method.png)

### 3.2 Perceptual-Consistency Clipping (PCC)

**动机**: 利用注意力焦点重叠来指导激活裁剪范围的确定，在保留语义能力的同时允许更激进的裁剪。

**核心思想**:
- 使用注意力图的 IoU（Intersection over Union）作为量化前后语义一致性的度量
- 通过搜索最优裁剪百分比，最大化注意力焦点的重叠度

**技术实现**:

```python
class AttentionOverlapLoss(nn.Module):
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold  # 阈值，用于确定高注意力区域

    def forward(self, attn1, attn2):
        # 获取每个 batch 中的最大值
        max_vals1 = attn1.amax(dim=-1, keepdim=True)
        max_vals2 = attn2.amax(dim=-1, keepdim=True)

        # 生成高注意力区域 mask
        mask1 = (attn1 > self.threshold * max_vals1).float()
        mask2 = (attn2 > self.threshold * max_vals2).float()

        # 计算 IoU
        intersection = torch.min(mask1, mask2)
        union = torch.max(mask1, mask2)
        iou = intersection.sum(dim=-1) / (union.sum(dim=-1) + eps)

        return (1 - iou).mean()  # IoU 损失
```

**搜索过程**:
```python
pct_list = [0.85, 0.87, 0.88, ..., 0.99999]  # 裁剪百分比列表
for pct in pct_list:
    calib_sz_across_pct(act_ori, pct)  # 使用当前百分比校准
    q_ATM = self.get_ATM(qkv)  # 获取量化后的注意力图
    score = AOL_loss(q_ATM, self.ori_ATM)  # 计算 IoU 损失
    if score < best_score:
        best_score = score
        best_pct = pct  # 记录最优百分比
```

**关键观察**: PCC 主要对 SAM-B 有效，因为极端激活异常值主要出现在 ViT-B 中，但对 SAM-L 和 SAM-H 也有一定的性能提升。

### 3.3 Prompt-Aware Reconstruction (PAR)

**动机**: 传统重建方法仅关注 encoder 输出的重建，忽略了 mask decoder 中图像 - 提示交互的重要性。

**核心思想**:
- 在 encoder 量化过程中，引入 mask decoder 中的 cross-attention 交互作为监督信号
- 通过最小化量化前后交互响应的差异，实现语义对齐

**技术实现**:

1. **Cross Image Token (CIT) 提取**:
```python
def get_CIT(image_embedding, image_pe, sparse_prompt_embed, dense_prompt_embed, mask_decoder):
    CA_block = mask_decoder.transformer
    # 通过 cross-attention block 获取交互后的 image token
    mask_decoder.predict_calib_recon(...)
    dc_out = data_saver_pe.output_store[1]  # Cross image token
    return dc_out
```

2. **PAR 损失函数**:
```python
class LossFunction:
    def __call__(self, pred, tgt, pred_attention=None, target_attention=None):
        # 输出重建损失
        rec_loss = lp_loss(pred, tgt, p=self.p)

        # PAR: 注意力交互相似性损失
        CAM_sim_loss = lp_loss(pred_attention, target_attention, p=2.0, sum_dim=-1)

        # 总损失
        total_loss = self.args.beta * rec_loss + self.args.alpha * CAM_sim_loss

        return total_loss
```

**重建流程**:
```python
def reconstruction_IE(args, model, fp_model, module, ...):
    # 获取量化输入和 FP 输出
    quant_inp = save_inp_oup_data_en(model.predictor.model, module, image_data, ...)
    fp_inp, fp_oup = save_inp_oup_data_en(fp_model.predictor.model, fp_module, image_data, ...)

    # 保存 FP interaction response target
    image_tokens_fp = transform_image_token(fp_oup, fp_model.predictor.model.image_encoder.neck)
    fp_dc_out = save_CIT_list(image_tokens_fp, image_pes, ...)

    # 优化过程
    for i in range(config.iters_CAMS):
        # 前向传播
        cur_quant_oup = module_ddp(cur_inp)
        cur_image_tokens_q = transform_image_token(cur_quant_oup, ...)
        cur_dc_out = get_CIT(cur_image_tokens_q, ...)  # 量化 interaction response

        # 计算损失
        err = loss_func(cur_quant_oup, cur_fp_oup, cur_dc_out, fp_dc_out[idx])
        err.backward()
```

### 3.4 分层重建策略

SAQ-SAM 对整个模型进行分层重建，包括：

1. **Encoder 重建 (CAMS)**: 使用 PAR 损失对 image encoder 进行逐层/逐 stage 重建
2. **Decoder 重建 (CLM)**: 使用类似的策略对 mask decoder 进行重建

```python
def recon_model(args, model, fp_model, cali_data, ...):
    # Encoder 重建
    if args.recon_encoder == 'CAMS':
        _recon_model_IE(args, model.predictor.model.image_encoder, ...)

    # Decoder 重建
    if args.recon_decoder == 'CLM':
        _recon_model_MD(args, model.predictor.model.mask_decoder, ...)
```

### 3.5 特殊量化器设计

**针对 Softmax 输出**: 使用自适应粒度量化（Adaptive Granularity Quantization, AGQ）
```python
softmax_a_config = update_specialized_quantizer_config(a_qconfig, 'softmax')
# quantizer: AdaptiveGranularityQuantize
# observer: LogAvgMSEFastObserver
```

**针对双峰 K 激活**: 使用符号感知量化（Sign-aware Quantization）
```python
sign_a_config = update_specialized_quantizer_config(a_qconfig, 'bimodal')
# quantizer: LSQSignFakeQuantize
# observer: SignAvgMSEFastObserver
```

**双峰分布整合 (BIG)**:
```python
def bimodal_adjust(model, logger):
    for name, m in model.named_modules():
        if isinstance(m, QuantDecoderOurAttentionBlock) and 'token_to_image' in name:
            if m.k_post_act_fake_quantize.is_bimodal:
                # 调整权重符号以适应双峰分布
                sign = m.k_post_act_fake_quantize.sign
                m.k_proj.module.weight.mul_(sign.unsqueeze(1))
                m.k_proj.module.bias.mul_(sign)
```

### 3.6 量化配置

```yaml
# config_SA_66.yaml (W6A6 配置示例)
encoder_a_qconfig:
    quantizer: LSQFakeQuantize
    observer: AvgMSEFastObserver
    bit: 6
    symmetric: False
    ch_axis: -1

encoder_w_qconfig:
    quantizer: AdaRoundFakeQuantize
    observer: MSEObserver
    bit: 6
    symmetric: False
    ch_axis: 0

ptq4sam:
    BIG: True         # 启用双峰分布整合
    AGQ: True         # 启用自适应粒度量化
    global_num: 128   # 双峰检测参数
    threshold: 0.5    # PCC 阈值
```

---

## 4. 实验结果

### 4.1 实验设置
- **数据集**: COCO (实例分割)、DIOR-R (旋转目标检测)、LoveDA (语义分割)
- **基准模型**: SAM-B/H + 不同检测器 (YOLOX, Faster R-CNN, H-Deformable-DETR, DINO)
- **量化配置**: W4A4, W6A6, W8A8

### 4.2 主要结果

| 方法 | SAM-B (W6A6) | SAM-B (W4A4) |
|------|-------------|-------------|
| PTQ4SAM (Baseline) | XX.X mAP | XX.X mAP |
| SAQ-SAM* (PCC only) | XX.X mAP | XX.X mAP |
| SAQ-SAM (Full) | **+11.7% mAP** | **+XX% mAP** |

注：SAQ-SAM*表示仅使用 PCC 的重建方法，不带 PAR 重建。

### 4.3 关键发现

1. **PCC 的有效性**: 即使不使用重建，仅 PCC 就能显著提升低比特量化的性能
2. **PAR 的贡献**: PAR 重建在 encoder 量化中带来额外的性能提升，特别是在极低比特下
3. **模型规模影响**: PCC 对 SAM-B 效果最明显，对 SAM-L/H 也有一定提升
4. **跨任务泛化**: 在多种下游任务上均有一致的性能提升

---

## 5. 对多模态大模型量化的思考

### 5.1 语义对齐 > 数值精度

SAQ-SAM 的核心贡献在于提出了**语义对齐量化**的范式。这对其他多模态大模型的量化有重要启示：

1. **任务导向的量化目标**: 量化不应仅追求数值重建精度，而应关注任务相关的语义指标
2. **注意力作为语义代理**: 对于基于 attention 的模型，注意力图的重叠是衡量语义一致性的有效代理
3. **跨模态交互保护**: 多模态模型的核心是模态间的交互，量化时应特别保护这些交互机制

### 5.2 针对架构特性的量化设计

SAQ-SAM 的成功部分归功于针对 SAM 架构特性的定制化设计：

1. **双峰分布处理**: K 激活的双峰分布是 SAM 的特性，设计专门的量化器（LSQSignFakeQuantize）非常有效
2. **异常值利用**: 将极端激活异常值视为语义信号而非噪声，利用它们指导裁剪
3. **分组件量化**: encoder 和 decoder 采用不同的量化策略，encoder 使用 PAR 重建，decoder 使用 CLM 重建

### 5.3 对其他多模态模型的启示

| SAM 特性 | 其他多模态模型对应 | 可能的量化策略 |
|---------|------------------|---------------|
| 图像 -提示交互 | 视觉 - 语言交互 (CLIP, Flamingo) | 跨模态注意力保护 |
| 双峰 K 激活 | 文本 token 的稀疏激活 (LLM) | 混合精度/分组量化 |
| 极端异常值 | 模态特定特征 (LLaVA) | 异常值感知裁剪 |
| Cross-Attention | 多模态融合层 | 交互感知重建 |

### 5.4 未来研究方向

1. **统一的多模态量化框架**: 将 SAQ-SAM 的语义对齐思想扩展到其他多模态模型
2. **自动化量化策略搜索**: 基于模型架构自动搜索最优量化配置
3. **极低比特量化**: 探索 2-bit/3-bit 下的语义对齐方法
4. **端到端重建**: 将多模态交互作为一个整体进行重建，而非分组件

---

## 6. 借鉴与应用

### 6.1 技术借鉴

#### 可迁移的技术组件

1. **注意力重叠损失 (AOL Loss)**
   - 适用于任何基于 attention 的模型
   - 可作为量化过程中的正则化项
   - 实现简单，计算开销小

2. **交互感知重建**
   - 适用于有跨模态交互的模型
   - 可以扩展到其他多模态融合场景
   - 关键是选择合适的交互表示作为监督信号

3. **双峰分布检测与处理**
   - 适用于激活分布呈现多峰特性的场景
   - 可以结合混合精度量化
   - 需要针对具体模型分析激活分布

#### 代码级借鉴

```python
# AOL Loss 可以直接复用
class AttentionOverlapLoss(nn.Module):
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, attn1, attn2):
        max_vals1 = attn1.amax(dim=-1, keepdim=True)
        max_vals2 = attn2.amax(dim=-1, keepdim=True)
        mask1 = (attn1 > self.threshold * max_vals1).float()
        mask2 = (attn2 > self.threshold * max_vals2).float()
        intersection = torch.min(mask1, mask2)
        union = torch.max(mask1, mask2)
        iou = intersection.sum(dim=-1) / (union.sum(dim=-1) + eps)
        return (1 - iou).mean()
```

### 6.2 方法借鉴

#### 对于视觉 - 语言模型 (VLM) 量化

1. **跨模态注意力保护**
   - VLM 中的 cross-attention 类似于 SAM 的图像 - 提示交互
   - 可以使用类似 PAR 的重建策略
   - 监督信号可以是融合后的文本/视觉表示

2. **文本 token 的特殊处理**
   - 文本 token 的嵌入分布可能与视觉 token 不同
   - 可以考虑分组量化或混合精度

3. **语义一致性验证**
   - 使用 VQA 或其他下游任务作为验证指标
   - 而不仅仅是数值重建精度

#### 对于 LLM 量化

1. **激活分布分析**
   - 类似 SAQ-SAM 分析 K 激活的双峰分布
   - 针对特定分布设计量化器

2. **注意力感知裁剪**
   - 使用注意力重叠作为裁剪指导
   - 保护重要的注意力模式

### 6.3 实验设计借鉴

1. **多任务评估**: 在多个下游任务上评估量化效果
2. **消融研究**: 分离各个组件的贡献 (PCC vs PAR)
3. **跨模型验证**: 在不同规模的模型上验证方法

---

## 7. 代码结构分析

### 7.1 核心文件结构

```
SAQ-SAM/
├── ptq4sam/
│   ├── model/
│   │   └── quant_model.py          # 量化模型定义，特殊模块
│   ├── quantization/
│   │   ├── quantized_module.py     # 量化模块基类
│   │   ├── fake_quant.py          # 伪量化实现
│   │   ├── observer.py            # 观察器实现
│   │   └── state.py               # 量化状态管理
│   ├── quant_rep/
│   │   └── quant_modules_rep.py   # 重建相关量化模块
│   └── solver/
│       ├── test_quant_SAQ_m.py    # 主量化脚本
│       ├── recon.py               # 重建逻辑
│       └── utils.py               # 工具函数
├── exp/
│   ├── config_SA_66.yaml          # 量化配置
│   └── config_SA_44.yaml
└── projects/
    └── instance_segment_anything/  # SAM 实现
```

### 7.2 量化流程

```
1. 模型量化化 (quantize_model)
   ├── Encoder 量化 (QuantImageEncoderOurViT)
   └── Decoder 量化 (QuantDecoderOurTwoWayAttentionBlock)

2. 校准 (calibrate)
   ├── 激活校准 (enable_calibration_woquantization)
   ├── PCC 校准 (FFC 选项)
   └── 权重校准 (enable_calibration_with_quantization)

3. 重建 (recon_model)
   ├── Encoder 重建 (reconstruction_IE)
   └── Decoder 重建 (reconstruction_MD)

4. 推理评估
```

### 7.3 关键类继承关系

```
nn.Module
├── QuantizedModule
│   ├── QuantizedLayer
│   ├── QuantizedBlock
│   │   ├── QuantEncoderAttentionBlock
│   │   └── QuantDecoderOurAttentionBlock
│   ├── QuantizedTransformerLayer
│   │   └── QuantDecoderOurTwoWayAttentionBlock
│   └── QuantizedTransformerStage
│       └── Stage
└── QuantImageEncoderOurViT
```

---

## 8. 局限性与改进方向

### 8.1 当前局限

1. **重建时间较长**: 分层重建需要大量迭代，训练时间较长
2. **超参数敏感**: PCC 阈值、PAR 权重等超参数需要调优
3. **GPU 内存需求**: 需要双 GPU 进行重建 (FP 模型 + 量化模型)
4. **代码 Bug**: README 中提到 PTQ4SAM 基线存在 dropout 概率未重置的 bug

### 8.2 改进方向

1. **加速重建**: 探索更高效的重建策略，如梯度缓存、并行重建
2. **自适应超参数**: 基于模型特性自动确定超参数
3. **单 GPU 重建**: 优化内存使用，支持单 GPU 重建
4. **更广泛的验证**: 在更多多模态模型上验证方法

---

## 9. 总结

SAQ-SAM 提出了一种面向 SAM 的语义对齐量化方法，主要贡献包括：

1. **提出了语义对齐量化的新范式**: 从数值精度转向语义一致性
2. **设计了 PCC 和 PAR 两个核心技术**: 分别处理激活裁剪和特征重建
3. **针对 SAM 架构的定制化设计**: 处理双峰分布、极端异常值等特殊问题
4. **实验验证**: 在多个任务和模型规模上验证了有效性

对于多模态大模型量化的研究，SAQ-SAM 提供了以下启示：

- 语义对齐比数值精度更重要
- 跨模态交互需要特别保护
- 架构特性驱动的量化设计更有效
- 注意力机制可以作为语义一致性的代理

这些思想和技术可以迁移到其他多模态模型的量化中，具有广泛的应用前景。

---

## 附录：核心命令

### 运行量化

```bash
# SAQ-SAM* (仅 PCC)
python ptq4sam/solver/test_quant_SAQ_m.py \
--quant-encoder --quant-decoder \
--config='./projects/configs/yolox/yolo_l-sam-vit-b.py' \
--q_config='./exp/config_SA_66.yaml' \
--FFC \
--recon_encoder='no' --recon_decoder='no'

# SAQ-SAM (完整方法，包含 PAR 重建)
python ptq4sam/solver/test_quant_SAQ_m.py \
--quant-encoder --quant-decoder \
--config='./projects/configs/yolox/yolo_l-sam-vit-b.py' \
--q_config='./exp/config_SA_66.yaml' \
--FFC \
--recon_encoder='CAMS' --recon_decoder='CLM' --CAM_loss='PAR'
```

---

*文档生成时间: 2026-03-25*
*基于 SAQ-SAM 论文 (arXiv:2503.06515) 和代码仓库分析*
