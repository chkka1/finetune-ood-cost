# 配对跨域微调实验（配套代码）

本仓库是一套**配对跨域实验**代码，用于测量"领域微调"在域内与域外的不同后果：

- 同一批类别（域内域外共享标签空间）；
- 同一个评估分类器（都在源域训练集特征上重新拟合，微调后丢弃训练头）；
- 同一批随机种子 + 逐轮记录 → 剂量-反应曲线；
- 三档微调强度（lr=1e-4 / 3e-4 / 1e-3）使用**完全相同的轮数预算**（PACS 40 轮、OrganMNIST 24 轮），
  排除训练时长这一混杂因素，使“强度”可以单独作为自变量考察（数字案例只跑了 3e-4 一档）。

这样设计的目的，是让**域内增益与域外代价在同一量纲下可比**，避免"两侧在不同类别集/不同协议上
测得"导致的口径问题。三个公开案例：PACS（7 类 × 4 风格）、MedMNIST 器官数据（11 类 × 3 成像平面）、
多来源手写数字（10 类 × MNIST/EMNIST/SVHN）。

> **本仓库只包含代码、实验结果与图表，不含任何论文稿件或作者信息。**
> 相关成果正式发表后可在本文件末尾补充引用信息。

---

## 1. 目录结构

```
.
├── scripts/
│   ├── prepare_data.py     下载三个案例并统一缓存为 npz（含抽样可视化校验）
│   ├── dataset_stats.py    统计案例构成 → results/dataset_stats.json
│   ├── run_experiment.py   核心：微调 + 逐轮跨域评估 + 特征距离诊断
│   ├── analyze.py          汇总与统计检验 → results/analysis.json + analysis_summary.md
│   └── plot_figures.py     按期刊规范出图（图 1~图 6；PNG 600 dpi + PDF 矢量 + TIFF）
├── results/                已提交的真实实验结果（CSV/JSON，约 70 KB）
├── figures/                六张结果图（fig1~fig6）+ 数据抽样确认图
├── data/cases/             空目录：npz 案例缓存（约 480 MB，需生成，不随仓库分发）
├── requirements.txt
├── reproduce.ps1           一键复现
├── LICENSE                 MIT（第三方数据条款见第 5 节）
└── .gitignore
```

---

## 2. 环境

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# 若需 GPU：按 https://pytorch.org 选择对应 CUDA 版本的 torch/torchvision 安装命令
```

`pyarrow` 只在解析 PACS 的 parquet 镜像时需要；不做数据重下载可不装。
实测环境：Python 3.9、torch 2.5.1+cu121、torchvision 0.20.1、NVIDIA RTX 4060 Laptop (8 GB)。
CPU 也能跑完整流程，只是实验部分会慢很多。

---

## 3. 复现方式

### 3.1 快速复现（离线，约 1 分钟；不需要下载数据）

已提交 `results/` 中的**全部逐轮原始结果**，因此一步即可重现每个数字与全部结果图：

```powershell
.\reproduce.ps1            # 等价于 analyze.py（统计 + 调用 plot_figures.py 出图）
```

产出：
- `results/analysis_summary.md`、`results/analysis.json`（全部关键数字与统计量）
- `figures/fig1_overview.png`（各案例在各域上的精度变化总览，热力图）
- `figures/fig2_dose.png`（剂量-反应曲线，误差棒；三档强度用点线/虚线/实线区分）
- `figures/fig3_doseheat.png`（剂量-反应热力图：行为域、列为轮数）
- `figures/fig4_lrdose.png`（学习率剂量-反应：末轮精度变化 vs 学习率）
- `figures/fig5_summary.png`（域内/域外精度变化对比，分组条形）
- `figures/fig6_gap.png`（域外精度变化 vs 特征距离变化，散点 + 拟合）

每张图同时导出三种格式：`png`（600 dpi 位图）、`pdf`（矢量，字体以 TrueType 嵌入）、
`tif`（600 dpi LZW 压缩），便于按期刊要求取用。

### 3.2 完整复现（重新下载数据 + 重跑全部实验，约 1 小时）

```powershell
.\reproduce.ps1 -Full
```

`-Full` 会依次执行：`prepare_data.py all`（下载约 3 GB）→ `dataset_stats.py` →
7 组实验（PACS 弱/强/更弱、OrganMNIST 弱/强/更弱、Digits 弱）→ `analyze.py`。
也可单独执行，例如只跑某一个案例：

```powershell
$env:PYTHONIOENCODING='utf-8'
python scripts\run_experiment.py --case pacs --size 128 --epochs 40 --eval-epochs 1 2 4 8 16 24 32 40 `
    --seeds 0 1 2 --lr 1e-3 --tag _strong
python scripts\analyze.py
```

`run_experiment.py` 的输出 CSV 是**逐行追加**的：中断后重跑会自动跳过已有 (seed, epoch)，
不会重复计算。

### 3.3 图表规范（plot_figures.py）

出图按中文核心与 SCI 论文的通用制图约定实现，便于直接投稿：

| 规范项 | 实现方式 |
|---|---|
| 字体 | 拉丁字母与数字用 Times New Roman，中文自动回退黑体（`font.family` 列表式逐字回退） |
| 坐标轴 | 四边闭合、刻度朝内、开启次刻度；不使用网格线 |
| 误差 | 误差棒为 ±1 标准差（3 个随机种子）；不叠加阴影带 |
| 配色 | **全部图均为彩色**，采用色盲友好的 Okabe-Ito 配色：图 2/图 3/图 4/图 5 的颜色表示“域”（域内统一黑色作参照，每个目标域一种颜色 + 一种标记）；图 6 的颜色表示“案例”；图 1、图 3 为红-蓝发散色标（每格标注数值） |
| 线型 | 实线＝强微调、虚线＝弱微调；域身份同时由颜色/灰度与标记形状编码 |
| 子图 | 用 (a)(b)(c) 统一编号；同一图内共享一个图例 |
| 尺寸 | 单栏图约 8.9 cm、双栏图约 17.5 cm，字号按最终印刷尺寸设定（缩放后仍 ≥7 pt） |
| 导出 | 同时输出 PNG（600 dpi）、PDF（矢量，TrueType 嵌入）、TIFF（LZW） |
| 文字 | 全中文标注；统计量写作 `r = …`、`P < …`、`n = …`，不使用与中文字体混排的数学公式排版 |

论文中的表格采用**三线表**（仅顶线、表头线、底线，无竖线），表题在表上方居中，
数字列右对齐以便按位读数。

---

## 4. 核对点（复现后应得到的关键数字）

| 项 | 值 |
|---|---|
| PACS 强微调（lr=1e-3，40 轮） | 域内 −2.67 pp；Art −16.00、Cartoon −5.00、Sketch −7.67 |
| PACS 更弱微调（lr=1e-4，40 轮） | 域内 +0.67 pp；Art −4.33、Cartoon −6.00、Sketch −4.67 |
| PACS 弱微调（lr=3e-4，40 轮） | 域内 −0.67 pp；Art −8.67、Cartoon −1.33、Sketch +1.67 |
| OrganMNIST 强微调（24 轮） | 域内 +10.00；冠状位 +17.67、矢状位 −0.67 |
| OrganMNIST 弱微调（24 轮） | 域内 +10.33；冠状位 +16.67、矢状位 −1.33 |
| Digits 弱微调（16 轮） | 域内 +4.00；EMNIST −13.33、SVHN +2.33 |
| OrganMNIST 更弱微调（lr=1e-4，24 轮） | 域内 +9.33；冠状位 +15.00、矢状位 −0.67 |
| 域外精度变化 vs 特征距离变化 | 总体 r=−0.19（p=3.9e−4，n=354）；案例内 PACS −0.34、器官 −0.69、数字 −0.96 |

任一数字不符，请检查 `results/*.csv` 是否被改动。

---

## 5. 数据来源与许可

| 数据 | 来源 | 说明 |
|---|---|---|
| PACS | Li 等，ICCV 2017（`flwrlabs/pacs` HuggingFace 镜像，单 parquet） | 仅用于学术研究；引用请引原文 |
| Organ{A,C,S}MNIST | MedMNIST v2（Yang 等，Scientific Data 2023；Zenodo record 10519652） | CC BY 4.0 |
| MNIST | LeCun 等（OSSCI 镜像） | 学术用途 |
| EMNIST（digits 划分） | Cohen 等，IJCNN 2017（NIST） | 学术用途 |
| SVHN | Netzer 等，NIPS Workshop 2011（Stanford） | 学术用途 |

脚本只负责下载与缓存，**数据本身不随本仓库分发**。二次使用请遵循各数据集原始条款。
本仓库代码采用 MIT 许可，见 [`LICENSE`](LICENSE)。

---

## 6. 脚本设计上的两个要点（便于审阅）

1. **公平比较**：微调阶段挂线性头训练；评估阶段丢弃该头，改为在源域训练集特征上
   **重新拟合岭回归分类器**（λ=1）并用于所有域。微调前后唯一变化的是特征表示。
2. **诊断量无需标签**：MMD / Fréchet 距离只用到目标域的**无标签**样本，因此"微调前先量距离"
   这一判据在真实部署中是可执行的。

---

## 7. 常见问题

- **显存不足**：三个实验并行跑时每个约 1.6 GB；`run_experiment.py` 可用 `--batch` 调小。
- **PACS 解析失败**：确认已装 `pyarrow`；若镜像地址失效，可改用 PACS 官方来源后
  把 `prepare_data.py:prep_pacs()` 的读取部分替换为对应格式（其余流程不变）。
- **USPS 未纳入**：其镜像证书过期导致 torchvision 下载失败，`prepare_data.py` 已保留解析入口，
  手工下载 `usps.bz2`/`usps.t.bz2` 后即可启用（本项目的数字案例用 MNIST/EMNIST/SVHN 三源）。

---

## 引用信息

相关成果发表后在此补充题录信息（当前仅提供代码）。
