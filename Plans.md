# SGLang HiCache Benchmark — Plans.md

作成日: 2026-05-13

---

## Phase 1: 环境准备

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 1.1 | 确认模型文件存在 `/sgl-workspace/model/Qwen3-14B` | `ls` 输出包含 config.json | - | cc:完了 |
| 1.2 | 确认数据集就绪 ShareGPT (`/sgl-workspace/datasets/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json`) | 文件存在且大小 > 100MB | - | cc:完了 |
| 1.3 | 确认 SGLang 可启动 `python -m sglang.launch_server --help` | 命令不报错 | 1.1 | cc:完了 |

## Phase 2: 对比测试 — 四种缓存策略

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 2.1 | **基线**：禁用缓存 (`--disable-radix-cache`)，跑 `bench_multiturn.py`，保存结果 | JSONL 结果文件生成 | 1.3 | cc:完了 |
| 2.2 | **FCFS**：先进先出调度 (`--schedule-policy fcfs`)，跑测试，保存结果 | JSONL 结果文件生成 | 2.1 | cc:完了 |
| 2.3 | **长前缀匹配**：默认调度（不加缓存参数），跑测试，保存结果 | JSONL 结果文件生成 | 2.2 | cc:完了 |
| 2.4 | **分层缓存**：启用 hicache (`--enable-hierarchical-cache`)，跑测试，保存结果 | JSONL 结果文件生成 | 2.3 | cc:完了 (rate=1 成功, rate>=2 D状态挂起) |

## Phase 3: 结果汇总与报告

| Task | 内容 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 3.1 | 从 4 份 JSONL 提取关键指标（throughput, cache_hit_rate, avg_ttft, avg_latency, p99_latency） | Python 脚本成功解析所有文件 | 2.4 | cc:完了 |
| 3.2 | 生成对比表格和 Markdown 报告保存到 `/sgl-workspace/sglang/benchmark/hicache/RESULTS.md` | 文件存在且包含 4 组策略对比数据 | 3.1 | cc:完了 |

---

## 测试参数

- **模型**: `/sgl-workspace/model/Qwen3-14B`
- **数据集**: `/sgl-workspace/datasets/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json`
- **端口**: 30000（每次测试前 flush_cache）
- **请求率**: `[1, 2, 4, 8]`（从低到高，每种策略跑 4 档）
- **客户端数**: 64
- **轮数**: 3
- **输入长度**: 512
- **输出长度**: 64

## 结果保存路径

- 原始 JSONL: `/sgl-workspace/sglang/benchmark/hicache/results/`
- 对比报告: `/sgl-workspace/sglang/benchmark/hicache/RESULTS.md`
