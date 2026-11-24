# SAG-Mem 实验指南

[项目首页](../README.zh-CN.md) · [English](../README.md)

本目录提供 SAG-Mem 与 SAG-Mem-Lite 在 LoCoMo 和 LongMemEval-S 上的运行、重放、消融、费用统计和结果核验。论文结果、方法流程和首次安装步骤见项目首页。

## 运行入口

在仓库根目录执行：

```bash
python reproduction/run.py --help
```

| 方法 | 初始候选数 | 扩展轮数 | Reader 分块数 |
|---|---:|---:|---:|
| `sag-mem` | 50 | 1 | 20 |
| `sag-mem-lite` | 100 | 2 | 20 |

两种方法均在会话内保存完整消息，以 512 BGE token 为软目标。`--messages-per-chunk` 默认值为 `3`，每块最多三条完整消息；设置为 `0` 可取消消息数限制。SAG-Mem 使用 JEV 评分；Lite 使用嵌入相似度，默认配置为 `qwen3`，也支持 `bge`。

## 环境与数据

```bash
python -m pip install -r reproduction/requirements.txt
test -f .env || cp reproduction/.env.example .env
```

在根目录 `.env` 中配置服务密钥、Reader/Judge 模型和请求参数。将数据放到以下位置，或用 `--data-path` 指定已有文件：

```text
reproduction/data/
├── locomo10.json
└── longmemeval_s_cleaned.json
```

[下载命令与文件校验值](../README.zh-CN.md#数据集)

预检查无需密钥，不调用模型：

```bash
python reproduction/run.py --method sag-mem --dataset locomo --limit 2 --preflight
python reproduction/run.py --method sag-mem-lite --dataset longmemeval --limit 2 --preflight
```

## 主实验

以下命令会调用模型。先在 `.env` 中配置 Qwen Reader/Judge：

```bash
for dataset in locomo longmemeval; do
  python reproduction/run.py --method sag-mem --dataset "$dataset" \
    --model qwen --build-concurrency 2 --resume
  python reproduction/run.py --method sag-mem-lite --dataset "$dataset" \
    --model qwen --embedding-profile qwen3 --build-concurrency 2 --resume
done
```

SAG-Mem 默认输出到 `reproduction/experiments/main/results/<dataset>/full/run.jsonl`，Lite 输出到 `reproduction/experiments/lite/results/<dataset>/qwen/run.jsonl`。

- `--output`：结果文件。
- `--index-root`：索引目录，默认为 `reproduction/indexes`。
- `--cache-root`：缓存目录，默认位于索引目录下的 `cache/`。
- `--resume`：在相同配置下跳过成功问题并重试失败问题。
- `--limit`：限制问题数，适合先运行小样本。

## Reader/Judge 重放

将 `.env` 中的 Reader/Judge 模型和供应商改为目标配置，再通过 `--replay-from` 指定已完成的源结果。

```bash
python reproduction/run.py --method sag-mem --dataset locomo --model deepseek --resume \
  --replay-from reproduction/experiments/main/results/locomo/full/run.jsonl
```

重放复用检索结果和完整证据，只调用 Reader/Judge。`--model` 选择实验分组与默认输出目录，实际服务配置从 `.env` 读取。完整 DeepSeek 配置和两数据集命令见[输出与恢复](../README.zh-CN.md#输出与恢复)。

## 消融、延迟与费用

| 实验 | 说明 |
|---|---|
| [时间呈现与无扩展控制](experiments/ablations/README.md) | 从主实验结果准备配对输入，再评测 Reader/Judge |
| [串行延迟](experiments/latency/README.md) | 使用预建索引，分别测量检索与 Reader 时间 |
| [Lite 配置与统计](experiments/lite/README.md) | 嵌入配置、重放、token 和费用统计 |

完成两个数据集、两组 Reader/Judge 运行后，汇总结果：

```bash
python reproduction/analysis/aggregate.py --require-all --data-root reproduction/data
```

汇总文件为 `reproduction/analysis/final_results.json` 与 `question_metrics.json`。输入结果应保留对应 manifest、索引和来源记录。

`top5/10/20_session_recall` 与 `top5/10/20_message_recall` 分别统计前 K 个分块覆盖的正确会话和消息比例，重复来源只计一次。没有有效标注时记录为空，不计入该指标的平均值。LongMemEval 的消息标注来自 `has_answer`。系统成本排除 Judge，缓存用量可用时按缓存单价计算。

## 核验与测试

核验已完成的主实验结果：

```bash
python -m reproduction.experiments.lite.verify \
  reproduction/experiments/main/results/locomo/full/run.jsonl
```

运行论文流程测试：

```bash
python -m pytest -q reproduction
```

测试使用合成输入和模型测试响应，覆盖建索引、检索、评分、缓存、重放、消融、延迟和汇总。核验过程检查数据标识、问题覆盖、来源哈希以及 Reader 上下文。
