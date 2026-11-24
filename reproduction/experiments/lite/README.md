# SAG-Mem-Lite

[项目首页](../../../README.zh-CN.md) · [实验指南](../../README.md)

SAG-Mem-Lite 在原始对话分块上执行 BM25 检索和词法扩展，以问题与分块的余弦相似度为候选评分。默认每次检索最多 100 个新候选、最多扩展两轮、最终返回 Top-20，并发为 20。

## 分块与嵌入

在同一会话内按 512 BGE token 软目标合并完整消息。`--messages-per-chunk` 默认值为 `3`，每块最多三条完整消息；设置为 `0` 可取消消息数限制。单条超长消息保持完整，选中证据后再添加时间标注。

| 配置 | 模型 | 维度 | 输入窗口 |
|---|---|---:|---:|
| `qwen3` | `qwen/qwen3-embedding-8b` | 4,096 | 32,768 token |
| `bge` | `BAAI/bge-large-en-v1.5` | 1,024 | 512 token |

`--embedding-profile` 选择配置，默认 `qwen3`。`--embedding-overflow error` 对超过编码窗口的输入报错；BGE 可选择 `prefix`，只编码可容纳的原文前缀，保存和交给 Reader 的内容仍为完整分块。BGE 服务端模型名称可通过 `EMBEDDING_MODEL_NAME` 设置。

## 运行

在根目录 `.env` 配置 Reader/Judge 和 API 服务后，从仓库根目录执行。以下命令会调用模型：

```bash
for dataset in locomo longmemeval; do
  python reproduction/run.py --method sag-mem-lite --dataset "$dataset" \
    --model qwen --embedding-profile qwen3 \
    --candidate-k 100 --max-hops 2 --top-k 20 --concurrency 20 --resume
done
```

使用 BGE：

```bash
python reproduction/run.py --method sag-mem-lite --dataset locomo \
  --embedding-profile bge --embedding-overflow prefix \
  --output reproduction/experiments/lite/bge/locomo/run.jsonl
```

默认 Qwen 输出为 `reproduction/experiments/lite/results/<dataset>/qwen/run.jsonl`。索引、文档向量与检索缓存按数据和配置标识复用。

## 重放

先在 `.env` 配置目标 Reader/Judge 模型和供应商，再执行：

```bash
python reproduction/run.py --method sag-mem-lite --dataset locomo --model deepseek --resume \
  --replay-from reproduction/experiments/lite/results/locomo/qwen/run.jsonl
```

重放仅调用 Reader/Judge，保留原有证据与顺序，不重新检索或计算嵌入。

## 结果核验与费用

```bash
python -m reproduction.experiments.lite.verify \
  reproduction/experiments/lite/results/locomo/qwen/run.jsonl

python -m reproduction.experiments.lite.canonical_summary \
  reproduction/experiments/lite/results/locomo/qwen/run.jsonl
```

核验检查问题覆盖、原文、Reader 上下文和缓存哈希。`canonical_summary` 按运行记录中的编码器统计 token、输入窗口和索引大小；价格从 `prices.json` 读取。系统成本包含一次构建、逐题检索和 Reader，Judge 费用单列。存在输入缓存用量时使用缓存单价，缺少缓存记录时按普通输入单价计算。请求日志另行保留包括重试在内的实际用量和费用。
