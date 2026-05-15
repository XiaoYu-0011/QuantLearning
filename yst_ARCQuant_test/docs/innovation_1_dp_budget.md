# ARCQuant 创新实验 - Layer-level S 预算分配 (DP)

## 原理

当前代码中每层独立用阈值 `τ = 0.125 × max(activation)` 计算 S（残差通道数），各层互不干扰。这意味着：
- 不敏感的层可能分了过多 S（浪费计算量）
- 敏感的层可能 S 不够（精度损失大）

改进：给定总 S 预算（= 所有层 S 之和），用动态规划找最优分配。

## 要修改的文件

**只改一个文件：`utilize.py`**

在 `search_select_proportions()` 函数（第 445-578 行）之后，添加新函数。

## 实现步骤

### Step 1: 在 `utilize.py` 末尾添加误差评估函数

```python
@torch.no_grad()
def compute_layer_quant_error(layer_activations, reorder_idx, select_num, quant_type='NVFP4'):
    """
    计算给定 select_num 下某层的量化误差。
    layer_activations: [num_tokens, hidden_dim] 的激活张量
    reorder_idx: 该层的通道重排索引
    select_num: 残差通道数（必须是 64 的倍数）
    返回: 均方量化误差
    """
    x = layer_activations.float()
    idx = reorder_idx.to(x.device).to(torch.int32)
    x_reordered = x[:, idx]
    
    in_features = x.shape[-1]
    
    if quant_type == 'NVFP4':
        quantize_func = quantize_nvfp4_tensor
    elif quant_type == 'MXFP4':
        quantize_func = quantize_mxfp4_tensor
    elif quant_type == 'HiF4':
        quantize_func = quantize_hif4_tensor
    else:
        quantize_func = quantize_int4_tensor
    
    # 主量化
    q_x = quantize_func(x_reordered)
    
    if select_num == 0:
        error = x_reordered - q_x
    else:
        # 残差量化（对最后 select_num 个通道）
        residual = x_reordered - q_x
        outlier_residual = residual[:, -select_num:]
        q_residual = quantize_func(outlier_residual)
        
        error = residual.clone()
        error[:, -select_num:] = outlier_residual - q_residual
    
    mse = (error ** 2).mean().item()
    return mse
```

### Step 2: 在 `utilize.py` 末尾添加 DP 搜索函数

```python
@torch.no_grad()
def search_select_proportions_dp(model, dataloader, device_, seqlen, reorder_index, 
                                  budget_ratio=1.0, quant_type='NVFP4'):
    """
    用动态规划优化各层 S 分配。
    
    budget_ratio: 总 S 预算相对于原始阈值法总 S 的比例。
                  1.0 = 同样的总 S，但重新分配
                  0.8 = 减少 20% 总 S，但通过优化分配保持精度
    """
    # Step 1: 先用原始方法获取每层的激活和 baseline select_num
    baseline_select_nums, baseline_avg_bits = search_select_proportions(
        model, dataloader, device_, seqlen, reorder_index
    )
    
    # Step 2: 收集每层每个线性层的激活
    nsamples = len(dataloader)
    device = device_
    
    layers = model.model.layers
    if hasattr(model.model, "embed_tokens"):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    
    cache = {'attention_mask': None, 'position_ids': None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            cache['inps'] = inp
            cache['attention_mask'] = kwargs.get('attention_mask')
            cache['position_ids'] = kwargs.get('position_ids')
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    if isinstance(dataloader, list):
        dataloader = torch.stack(dataloader, dim=0).squeeze(1)
    try:
        model(dataloader.to(device))
    except ValueError:
        pass
    layers[0] = layers[0].module
    if hasattr(model.model, "embed_tokens"):
        model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()
    
    inps = cache['inps']
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    
    # Step 3: 构建误差表 error_table[layer_name][s] = mse
    error_table = {}
    layer_names_ordered = []
    in_features_map = {}
    
    print("Building error table for DP optimization...")
    for i in tqdm(range(len(layers)), desc="Profiling layers"):
        layer = layers[i].to(device)
        
        act_scales = {}
        hooks = []
        layer_prefix = f"layers.{i}"
        
        for name, m in layer.named_modules():
            if isinstance(m, nn.Linear):
                full_name = f"{layer_prefix}.{name}"
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(
                            lambda m, x, y, name, d: d.__setitem__(
                                name + ".input", x[0] if isinstance(x, tuple) else x
                            ),
                            name=full_name, d=act_scales
                        )
                    )
                )
        
        inps = inps.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)
        
        with torch.no_grad():
            inps = layer(inps, attention_mask=attention_mask, position_ids=position_ids)[0]
        
        for name, activation in act_scales.items():
            if 'output' in name:
                continue
            if name not in reorder_index:
                continue
            
            activation = activation.reshape(-1, activation.shape[-1]).contiguous()
            in_features = activation.shape[-1]
            idx = reorder_index[name].to(device).to(torch.int32)
            
            max_s = min(in_features, baseline_select_nums.get(name, 0) * 3)
            s_candidates = list(range(0, max_s + 64, 64))
            
            errors = {}
            for s in s_candidates:
                if s > in_features:
                    break
                mse = compute_layer_quant_error(activation, idx, s, quant_type)
                errors[s] = mse
            
            error_table[name] = errors
            layer_names_ordered.append(name)
            in_features_map[name] = in_features
        
        for h in hooks:
            h.remove()
        del act_scales, hooks
        layer = layer.cpu()
        gc.collect()
        torch.cuda.empty_cache()
    
    # Step 4: 计算总预算
    total_baseline_s = sum(baseline_select_nums.values())
    total_budget = int(total_baseline_s * budget_ratio)
    total_budget = (total_budget // 64) * 64
    
    print(f"Baseline total S: {total_baseline_s}")
    print(f"DP budget (ratio={budget_ratio}): {total_budget}")
    
    # Step 5: 动态规划
    unit = 64
    B = total_budget // unit
    N = len(layer_names_ordered)
    
    INF = float('inf')
    dp = [[INF] * (B + 1) for _ in range(N + 1)]
    choice = [[0] * (B + 1) for _ in range(N + 1)]
    dp[0][0] = 0.0
    
    for i in range(N):
        name = layer_names_ordered[i]
        errors = error_table[name]
        
        for b in range(B + 1):
            if dp[i][b] == INF:
                continue
            for s_val, mse in errors.items():
                s_units = s_val // unit
                new_b = b + s_units
                if new_b <= B:
                    new_cost = dp[i][b] + mse
                    if new_cost < dp[i + 1][new_b]:
                        dp[i + 1][new_b] = new_cost
                        choice[i + 1][new_b] = s_val
    
    best_b = min(range(B + 1), key=lambda b: dp[N][b])
    
    # 回溯最优分配
    optimized_select_nums = {}
    optimized_avg_bits = {}
    b = best_b
    for i in range(N, 0, -1):
        name = layer_names_ordered[i - 1]
        s = choice[i][b]
        optimized_select_nums[name] = s
        in_feat = in_features_map[name]
        optimized_avg_bits[name] = 4.5 * (in_feat + s) / in_feat
        b -= s // unit
    
    # 打印对比
    print("\n=== DP Optimization Results ===")
    print(f"{'Layer':<60} {'Baseline S':>10} {'DP S':>10} {'Delta':>8}")
    print("-" * 90)
    total_dp_s = 0
    for name in layer_names_ordered:
        bs = baseline_select_nums.get(name, 0)
        ds = optimized_select_nums.get(name, 0)
        total_dp_s += ds
        delta = ds - bs
        sign = "+" if delta > 0 else ""
        print(f"{name:<60} {bs:>10} {ds:>10} {sign}{delta:>7}")
    
    print(f"\nTotal S: baseline={total_baseline_s}, DP={total_dp_s}")
    
    total_elements = sum(in_features_map.values())
    total_bits = sum(4.5 * (in_features_map[n] + optimized_select_nums[n]) for n in layer_names_ordered)
    print(f"Average bits: {total_bits / total_elements:.2f}")
    
    return optimized_select_nums, optimized_avg_bits
```

### Step 3: 创建 `reorder_indices_dp.py`（ARCQuant 根目录）

```python
"""
Layer-level S budget allocation using Dynamic Programming.
python reorder_indices_dp.py --model /PATH/TO/MODEL --samples 128 --seqlen 2048 --act_sort_metric max --budget_ratio 1.0
"""
from datasets import load_dataset
import torch.nn as nn
import gc
from utilize import *
import torch
from collections import defaultdict
import functools
from typing import List
import time
import argparse
import math
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="path of the hf model")
parser.add_argument("--dataset", type=str, default="wikitext2", choices=["wikitext2", "c4", "humaneval", "pile"])
parser.add_argument("--act_sort_metric", type=str, default="max")
parser.add_argument("--samples", type=int, default=128)
parser.add_argument("--seqlen", type=int, default=2048)
parser.add_argument("--budget_ratio", type=float, default=1.0,
                    help="S budget ratio vs baseline. 1.0=same total, 0.8=20% less")
parser.add_argument("--quant_type", type=str, default="NVFP4", choices=["NVFP4", "MXFP4", "INT4", "HiF4"])
args = parser.parse_args()

DATASET_LOADERS = {"wikitext2": get_wikitext2, "c4": get_c4, "pile": get_pile, "humaneval": get_humaneval}

def main():
    model, enc = load_model(args.model)
    path = args.model.rstrip('/')
    model_name = path.split('/')[-1]
    os.makedirs("./saved", exist_ok=True)

    get_dataset = DATASET_LOADERS[args.dataset]
    dataset_name = args.dataset.lower()
    
    act_scales_filename = f'./saved/{model_name.lower()}_act_scales_{dataset_name}_{args.act_sort_metric}.pt'
    if not os.path.exists(act_scales_filename):
        dataloader, _ = get_dataset(nsamples=args.samples, seed=0, seqlen=args.seqlen, tokenizer=enc)
        act_scales = get_act_stats(model, dataloader, "cuda:0", metric=args.act_sort_metric, seqlen=args.seqlen)
        torch.save(act_scales, act_scales_filename)
    else:
        act_scales = torch.load(act_scales_filename)
    
    reorder_index = get_reorder_index(model, act_scales, metric=args.act_sort_metric)
    
    _, inps = get_dataset(nsamples=32, seed=0, tokenizer=enc, seqlen=args.seqlen)
    select_num, average_bits = search_select_proportions_dp(
        model, inps, "cuda", args.seqlen, reorder_index,
        budget_ratio=args.budget_ratio, quant_type=args.quant_type
    )
    
    suffix = f"_dp_br{args.budget_ratio}"
    torch.save(reorder_index, f'./saved/{model_name.lower()}_reorder_index_{dataset_name}_{args.act_sort_metric}.pt')
    torch.save(select_num, f'./saved/{model_name.lower()}_select_num_{dataset_name}_{args.act_sort_metric}{suffix}.pt')
    torch.save(average_bits, f'./saved/{model_name.lower()}_average_bits_{dataset_name}_{args.act_sort_metric}{suffix}.pt')

if __name__ == "__main__":
    main()
```

### Step 4: 评估

```bash
# 将 DP 结果复制为标准文件名，然后正常评估
cp saved/*_select_num_*_dp_br1.0.pt saved/llama-3.1-8b-instruct_select_num_wikitext2_max.pt
python model/main.py /workspace/model/Llama-3.1-8B-Instruct --act_sort_metric max --dataset wikitext2 --eval_ppl --quant_type NVFP4
```

## 实验设计

| 实验 | budget_ratio | 含义 | 期望 PPL 变化 |
|------|-------------|------|--------------|
| baseline | 原始阈值法 | 各层独立算 S | 基线 |
| dp_1.0 | 1.0 | 同样总 S，DP 重分配 | -0.1~-0.3 |
| dp_0.8 | 0.8 | 总 S 减少 20%，DP 优化 | 接近基线 |
