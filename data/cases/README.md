# 案例缓存目录（不随仓库分发）

本目录用于存放三个案例的统一缓存：

```
pacs.npz     约 217 MB   7 类 × 4 风格（Photo/Art/Cartoon/Sketch）
organ.npz    约  95 MB   11 类 × 3 成像平面（轴位/冠状位/矢状位）
digits.npz   约 168 MB   10 类 × 3 来源（MNIST/EMNIST-digits/SVHN）
```

生成方式（会自动下载公开数据，共约 3 GB）：

```powershell
python scripts\prepare_data.py all      # 需要网络；PACS 的 parquet 解析需要 pyarrow
python scripts\dataset_stats.py         # 生成 results/dataset_stats.json（表 1 用）
```

每个 npz 的字段：

| 字段 | 含义 |
|---|---|
| `images` | `uint8 [N, h, w, 3]`，灰度图已复制为 3 通道 |
| `labels` | `int64 [N]`，跨域统一的类别编号 `0..C-1` |
| `domains` | `int64 [N]`，域索引，顺序见同文件内的 `domain_names` |
| `splits` | `int64 [N]`，`0=train, 1=val, 2=test` |

> 说明：仓库**已提交** `results/` 中的全部逐轮实验结果，因此不生成该缓存也能离线复现
> 全部数字与图表（见上级 README 的"快速复现"）。
