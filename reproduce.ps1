# 一键复现：配对跨域微调实验（代码与结果，不含论文正文）
#
#   .\reproduce.ps1          快速复现（离线，约 1 分钟）：由已提交的 results/ 重算全部图与统计
#   .\reproduce.ps1 -Full    完整复现（GPU，数小时）：重新下载数据 + 重跑全部实验
#
# 环境：先按 README 建好虚拟环境并安装 requirements.txt；
#       也可用 $env:PYTHON 指定解释器，或把解释器放在 .venv\Scripts\python.exe。

param([switch]$Full)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"

function Get-Python {
    if ($env:PYTHON) { return $env:PYTHON }
    $local = Join-Path $root ".venv\Scripts\python.exe"
    if (Test-Path $local) { return $local }
    return "python"
}
$py = Get-Python
Write-Host "使用解释器: $py" -ForegroundColor Cyan

function Invoke-Step([string]$label, [string[]]$argv) {
    Write-Host "`n=== $label ===" -ForegroundColor Green
    & $py @argv
    if ($LASTEXITCODE -ne 0) { throw "步骤失败: $label（退出码 $LASTEXITCODE）" }
}

$S5 = @("0", "1", "2", "3", "4")          # 主实验 5 个随机种子
$S3 = @("0", "1", "2")                    # 稳健性对照 3 个种子
$E40 = @("1", "2", "4", "8", "16", "24", "32", "40")
$E24 = @("1", "2", "4", "8", "16", "24")
$E16 = @("1", "2", "4", "8", "16")

if ($Full) {
    Invoke-Step "01 下载并缓存三个案例（约 3 GB）" @("$root\scripts\prepare_data.py", "all")
    Invoke-Step "02 统计案例构成" @("$root\scripts\dataset_stats.py")

    # ---- 主实验：三案例 × 三档强度 × 5 种子（各档轮数预算相同）----
    Invoke-Step "03 PACS 弱微调（lr=3e-4，40 轮）" @("$root\scripts\run_experiment.py",
        "--case", "pacs", "--size", "128", "--epochs", "40", "--eval-epochs") + $E40 +
        @("--seeds") + $S5 + @("--lr", "3e-4")
    Invoke-Step "04 PACS 强微调（lr=1e-3，40 轮）" @("$root\scripts\run_experiment.py",
        "--case", "pacs", "--size", "128", "--epochs", "40", "--eval-epochs") + $E40 +
        @("--seeds") + $S5 + @("--lr", "1e-3", "--tag", "_strong")
    Invoke-Step "05 PACS 更弱微调（lr=1e-4，40 轮）" @("$root\scripts\run_experiment.py",
        "--case", "pacs", "--size", "128", "--epochs", "40", "--eval-epochs") + $E40 +
        @("--seeds") + $S5 + @("--lr", "1e-4", "--tag", "_lr1e4")
    Invoke-Step "06 OrganMNIST 弱微调（24 轮）" @("$root\scripts\run_experiment.py",
        "--case", "organ", "--size", "64", "--epochs", "24", "--eval-epochs") + $E24 +
        @("--seeds") + $S5 + @("--batch", "128", "--lr", "3e-4", "--max-train", "12000")
    Invoke-Step "07 OrganMNIST 强微调（24 轮）" @("$root\scripts\run_experiment.py",
        "--case", "organ", "--size", "64", "--epochs", "24", "--eval-epochs") + $E24 +
        @("--seeds") + $S5 + @("--batch", "128", "--lr", "1e-3", "--max-train", "12000", "--tag", "_strong")
    Invoke-Step "08 OrganMNIST 更弱微调（lr=1e-4，24 轮）" @("$root\scripts\run_experiment.py",
        "--case", "organ", "--size", "64", "--epochs", "24", "--eval-epochs") + $E24 +
        @("--seeds") + $S5 + @("--batch", "128", "--lr", "1e-4", "--max-train", "12000", "--tag", "_lr1e4")
    Invoke-Step "09 Digits 弱微调（16 轮）" @("$root\scripts\run_experiment.py",
        "--case", "digits", "--size", "64", "--epochs", "16", "--eval-epochs") + $E16 +
        @("--seeds") + $S5 + @("--batch", "128", "--lr", "3e-4", "--max-train", "12000")
    Invoke-Step "10 Digits 强微调（16 轮）" @("$root\scripts\run_experiment.py",
        "--case", "digits", "--size", "64", "--epochs", "16", "--eval-epochs") + $E16 +
        @("--seeds") + $S5 + @("--batch", "128", "--lr", "1e-3", "--max-train", "12000", "--tag", "_strong")
    Invoke-Step "11 Digits 更弱微调（lr=1e-4，16 轮）" @("$root\scripts\run_experiment.py",
        "--case", "digits", "--size", "64", "--epochs", "16", "--eval-epochs") + $E16 +
        @("--seeds") + $S5 + @("--batch", "128", "--lr", "1e-4", "--max-train", "12000", "--tag", "_lr1e4")

    # ---- 稳健性对照：分辨率与主干（PACS，弱微调 16 轮 × 3 种子）----
    Invoke-Step "12 分辨率对照 ResNet-18@224（b32）" @("$root\scripts\run_experiment.py",
        "--case", "pacs", "--size", "224", "--backbone", "resnet18", "--epochs", "16",
        "--eval-epochs") + $E16 + @("--seeds") + $S3 + @("--lr", "3e-4", "--batch", "32", "--tag", "_r224")
    Invoke-Step "13 批大小对照 ResNet-18@224（b16）" @("$root\scripts\run_experiment.py",
        "--case", "pacs", "--size", "224", "--backbone", "resnet18", "--epochs", "16",
        "--eval-epochs") + $E16 + @("--seeds") + $S3 + @("--lr", "3e-4", "--batch", "16", "--tag", "_r224b16")
    Invoke-Step "14 架构对照 ViT-B/16@224（lr=3e-4, b16）" @("$root\scripts\run_experiment.py",
        "--case", "pacs", "--size", "224", "--backbone", "vit_b_16", "--epochs", "16",
        "--eval-epochs") + $E16 + @("--seeds") + $S3 + @("--lr", "3e-4", "--batch", "16", "--tag", "_v224")
    Invoke-Step "15 学习率对照 ViT-B/16@224（lr=1e-4, b32）" @("$root\scripts\run_experiment.py",
        "--case", "pacs", "--size", "224", "--backbone", "vit_b_16", "--epochs", "16",
        "--eval-epochs") + $E16 + @("--seeds") + $S3 + @("--lr", "1e-4", "--batch", "32", "--tag", "_v224lr4")
} else {
    Write-Host "快速模式：使用已提交的 results/ 离线重算（不下载数据、不训练）" -ForegroundColor Yellow
}

# ---- 分析与出图（全部离线，仅依赖 results/*.csv）----
Invoke-Step "16 汇总统计与主图（fig1~fig4）" @("$root\scripts\analyze.py")
Invoke-Step "17 补充统计：置信区间、配对检验、判据离线验证" @("$root\scripts\extra_stats.py")
Invoke-Step "18 距离-代价模型拟合（表 5）" @("$root\scripts\theory_fit.py")
Invoke-Step "19 跨案例检验：无量纲剂量、留一外推（表 6）" @("$root\scripts\cross_case_model.py")
Invoke-Step "20 模型适用范围图（fig5）" @("$root\scripts\make_fig5_scope.py")
Invoke-Step "21 架构与分辨率对照（表 7）" @("$root\scripts\robustness_check.py")

Write-Host "`n完成。产出：" -ForegroundColor Green
Write-Host "  $root\figures\fig1_doseheat / fig2_lrdose / fig3_summary / fig4_gap / fig5_scope（png+pdf+tif）"
Write-Host "  $root\results\analysis.json, analysis_summary.md, extra_stats.json,"
Write-Host "                theory_fit.json, cross_case.json, robustness.json"
