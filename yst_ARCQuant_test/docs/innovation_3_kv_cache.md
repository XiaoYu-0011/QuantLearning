# ARCQuant 创新实验 - 残差补偿扩展到 KV Cache

## 原理

当前 KV Cache 用简单的 INT4 group quantization（`quantize_int_group`，group_size=64）。可以借鉴 ARCQuant 的残差通道思想，对 KV Cache 中的离群通道也做双重量化。

## 关键修改点

### 文件: `model/qLlamaLayer.py`，`QLlamaAttention.forward()` 方法

```python
# 当前代码（第 311-312 行）:
if self.q_kv_cache:
    key_states = quantize_int_group(key_states, nbits=4, group_size=64)

# 改为残差补偿版本:
if self.q_kv_cache:
    key_states = quantize_kv_with_residual(key_states, nbits=4, group_size=64, 
                                            residual_channels=self.kv_residual_channels)
```

同样修改 value_states（第 338-339 行）。

### 文件: `model/quantize.py`，添加新函数

```python
@torch.no_grad()
def quantize_kv_with_residual(kv, nbits, group_size, residual_channels=0):
    """对 KV Cache 做残差补偿量化"""
    if residual_channels == 0:
        return quantize_int_group(kv, nbits, group_size)
    
    # kv shape: [batch, num_heads, seq_len, head_dim]
    q_kv = quantize_int_group(kv, nbits, group_size)
    
    residual = kv - q_kv
    
    # 选择残差最大的 residual_channels 个通道
    channel_importance = residual.abs().mean(dim=(0, 1, 2))
    _, top_channels = channel_importance.topk(residual_channels)
    
    residual_selected = residual[..., top_channels]
    q_residual = quantize_int_group(residual_selected, nbits, group_size)
    
    result = q_kv.clone()
    result[..., top_channels] = result[..., top_channels] + q_residual
    
    return result
```

## 注意事项

- KV Cache 的 head_dim 通常只有 128，残差通道数不能太多（建议 8-16）
- 这会增加 KV Cache 的存储开销，需要权衡
- 对长序列场景（大 seq_len）收益更大
- 需要在 `QLlamaAttention.__init__` 中添加 `self.kv_residual_channels` 参数
