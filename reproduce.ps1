# 一键复现：配对跨域微调实验（代码与结果，不含论文正文）
#
#   .\reproduce.ps1          快速复现（离线，约 1 分钟）：由已提交的 results/ 重算三张图与统计
#   .\reproduce.ps1 -Full    完整复现（约 1 小时）：重新下载数据 + 重跑全部实验
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

if ($Full) {
    Invoke-Step "1/10 下载并缓存三个案例（约 3 GB）" @("$root\scripts\prepare_data.py", "all")
    Invoke-Step "2/10 统计案例构成" @("$root\scripts\dataset_stats.py")
    Invoke-Step "3/10 PACS 弱微调（lr=3e-4，40 轮，与强微调同预算）" @("$root\scripts\run_experiment.py",
        "--case", "pacs", "--size", "128", "--epochs", "40",
        "--eval-epochs", "1", "2", "4", "8", "16", "24", "32", "40",
        "--seeds", "0", "1", "2", "--lr", "3e-4")
    Invoke-Step "4/10 PACS 强微调（lr=1e-3，40 轮）" @("$root\scripts\run_experiment.py",
        "--case", "pacs", "--size", "128", "--epochs", "40",
        "--eval-epochs", "1", "2", "4", "8", "16", "24", "32", "40",
        "--seeds", "0", "1", "2", "--lr", "1e-3", "--tag", "_strong")
    Invoke-Step "5/10 OrganMNIST 弱微调（24 轮，与强微调同预算）" @("$root\scripts\run_experiment.py",
        "--case", "organ", "--size", "64", "--epochs", "24",
        "--eval-epochs", "1", "2", "4", "8", "16", "24", "--seeds", "0", "1", "2",
        "--batch", "128", "--lr", "3e-4", "--max-train", "12000")
    Invoke-Step "6/10 OrganMNIST 强微调（24 轮）" @("$root\scripts\run_experiment.py",
        "--case", "organ", "--size", "64", "--epochs", "24",
        "--eval-epochs", "1", "2", "4", "8", "16", "24", "--seeds", "0", "1", "2",
        "--batch", "128", "--lr", "1e-3", "--max-train", "12000", "--tag", "_strong")
    Invoke-Step "7/10 Digits 弱微调（16 轮）" @("$root\scripts\run_experiment.py",
        "--case", "digits", "--size", "64", "--epochs", "16",
        "--eval-epochs", "1", "2", "4", "8", "16", "--seeds", "0", "1", "2",
        "--batch", "128", "--lr", "3e-4", "--max-train", "12000")
    Invoke-Step "8/10 PACS 更弱微调（lr=1e-4，40 轮）" @("$root\scripts\run_experiment.py",
        "--case", "pacs", "--size", "128", "--epochs", "40",
        "--eval-epochs", "1", "2", "4", "8", "16", "24", "32", "40",
        "--seeds", "0", "1", "2", "--lr", "1e-4", "--tag", "_lr1e4")
    Invoke-Step "9/10 OrganMNIST 更弱微调（lr=1e-4，24 轮）" @("$root\scripts\run_experiment.py",
        "--case", "organ", "--size", "64", "--epochs", "24",
        "--eval-epochs", "1", "2", "4", "8", "16", "24", "--seeds", "0", "1", "2",
        "--batch", "128", "--lr", "1e-4", "--max-train", "12000", "--tag", "_lr1e4")
} else {
    Write-Host "快速模式：使用已提交的 results/ 离线重算（不下载数据、不训练）" -ForegroundColor Yellow
}

Invoke-Step "汇总统计与出图（fig2 / fig3 / fig4）" @("$root\scripts\analyze.py")

Write-Host "`n完成。产出：" -ForegroundColor Green
Write-Host "  $root\figures\fig2_dose.png, fig3_gap.png, fig4_summary.png"
Write-Host "  $root\results\analysis_summary.md, analysis.json"
