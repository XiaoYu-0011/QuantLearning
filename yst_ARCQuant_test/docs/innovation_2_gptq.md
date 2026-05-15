# ARCQuant 创新实验 - GPTQ 权重量化集成

## 原理

ARCQuant 目前用 RTN（Round-To-Nearest）量化权重——直接四舍五入到最近的 FP4 值。GPTQ 通过 Hessian 矩阵逐列补偿量化误差，精度更高。

## 实现思路

1. 安装 auto-gptq: `pip install auto-gptq`
2. 修改 `model/qLinearLayer.py` 的 `QLinearLayer.__init__`
3. 或更简单：用 GPTQ 算法先优化权重，再做 NVFP4 打包

## 关键修改点

### 文件: `model/qLinearLayer.py`

```python
# 原始代码（第 55 行）:
self.W, self.scale_w, self.scale = NVFP4_reorder_quantize_w(
    originalLayer.weight.data, reorder_index.to(torch.int16).cuda(), select_num
)

# 改为先做 GPTQ 权重优化，再做 NVFP4 打包
# 需要传入 Hessian 矩阵 H（从 calibration 数据中获取）
```

### GPTQ 核心算法（需要实现）

```python
def gptq_quantize_weight(W, H, quantize_func, blocksize=128):
    """
    GPTQ 逐列量化 + 误差补偿。
    W: [out_features, in_features] 原始权重
    H: [in_features, in_features] Hessian 矩阵 (X^T X / nsamples)
    quantize_func: 量化函数 (如 quantize_nvfp4_tensor)
    """
    W = W.clone().float()
    n_cols = W.shape[1]
    
    # Cholesky 分解 Hessian
    H = torch.linalg.cholesky(H + 1e-5 * torch.eye(n_cols, device=H.device))
    H_inv = torch.linalg.inv(H)
    
    for i in range(0, n_cols, blocksize):
        j = min(i + blocksize, n_cols)
        W_block = W[:, i:j].clone()
        Q_block = quantize_func(W_block)
        err = W_block - Q_block
        
        # 误差补偿到后续列
        W[:, j:] -= err @ H_inv[i:j, j:]
        W[:, i:j] = Q_block
    
    return W
```

### 需要收集 Hessian 矩阵

在 `reorder_indices.py` 的 calibration 阶段，对每个线性层收集 `H = X^T X / nsamples`：

```python
def stat_hessian_hook(m, x, y, name, hessian_dict, nsamples):
    if isinstance(x, tuple):
        x = x[0]
    x = x.reshape(-1, x.shape[-1]).float()
    H = x.t() @ x / nsamples
    if name in hessian_dict:
        hessian_dict[name] += H
    else:
        hessian_dict[name] = H
```

## 注意事项

- GPTQ 需要每层的 Hessian 矩阵，需要在 calibration 阶段一起收集
- GPTQ 是逐列量化+补偿，与 ARCQuant 的通道重排可能有交互效应
- 建议先在 fake quantization 模式下验证 GPTQ+ARCQuant 的组合效果
- 重排后的权重需要用重排后的 Hessian 做 GPTQ
