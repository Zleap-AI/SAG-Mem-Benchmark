# 论文运行入口

[项目首页](../README.zh-CN.md) · [实验指南](../reproduction/README.md)

SAG-Mem 与 SAG-Mem-Lite 的统一入口是 `reproduction/run.py`。它负责方法配置、完整实验、恢复、预检查及固定证据重放。

```bash
python reproduction/run.py --help
```

## 入口索引

| 入口 | 用途 |
|---|---|
| `reproduction/run.py` | 两种方法的统一运行入口 |
| `reproduction/experiments/ablations/prepare.py` | 准备时间呈现和无扩展控制输入 |
| `reproduction/experiments/ablations/run.py` | 消融评测 |
| `reproduction/experiments/latency/run.py` | 串行配对延迟 |
| `reproduction/experiments/lite/verify.py` | 已完成运行的来源核验 |
| `reproduction/experiments/lite/canonical_summary.py` | Qwen3 Lite 用量与费用统计 |
| `reproduction/analysis/aggregate.py` | 结果汇总 |

## 共享工具

`run_benchmark.py` 提供统一入口使用的数据、Reader/Judge 和结果处理参数。方法运行请使用 `reproduction/run.py`，以加载对应的检索配置。

查看运行进度：

```bash
python scripts/monitor_benchmark.py \
  --output reproduction/experiments/main/results/locomo/full/run.jsonl
```

仅重新汇总已有结果：

```bash
python scripts/run_benchmark.py --metrics-only \
  --output reproduction/experiments/main/results/locomo/full/run.jsonl
```

这两个命令读取本地结果，不调用模型。
