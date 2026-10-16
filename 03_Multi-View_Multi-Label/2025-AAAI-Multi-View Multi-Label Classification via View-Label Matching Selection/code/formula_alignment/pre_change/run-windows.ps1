# Windows 启动入口：默认用 CUDA 环境短跑 Emotions，也可指定轮数和折数。
# 修正版完整运行示例：.\run-windows.ps1 -Epochs 80 -Folds 5
param(
    [int]$Epochs = 2,
    [int]$Folds = 2,
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [string]$PythonPath = '',
    [string]$OutputDirectory = '',
    [double]$Beta = 1.0,
    [double]$DevFraction = 0.2,
    [int]$FoldIndex = 0,
    [double]$LearningRate = 0.001
)
$ErrorActionPreference = 'Stop'
if (-not $PythonPath) {
    # 按目标设备选择已有 Python 环境；传入 -PythonPath 时使用指定路径。
    $PythonPath = if ($Device -eq 'cuda') {
        'D:\python-venv\cuda\.venv\Scripts\python.exe'
    } else {
        'D:\python-venv\deeplearn.venv\Scripts\python.exe'
    }
}
if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Python environment missing: $PythonPath. Supply -PythonPath with an installed environment."
}
# 图表只导出图片，避免训练结束后弹出窗口阻塞终端。
$env:MPLBACKEND = 'Agg'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
# 无论从哪个目录启动，都在代码目录解析相对数据和模型路径。
Push-Location $PSScriptRoot
try {
    # 每次运行单独保存，绝不覆盖旧六折模型和历史结果。
    if (-not $OutputDirectory) {
        $OutputDirectory = './results_corrected_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
    }
    $runArguments = @('-u', 'demo.py', '--device', $Device, '--dataset', 'emotions.mat',
        '--data_root', './data', '--k_fold', $Folds, '--end_epochs', $Epochs,
        '--output_dir', $OutputDirectory, '--beta', $Beta, '--dev_fraction', $DevFraction,
        '--lr', $LearningRate)
    # 指定 FoldIndex 时只执行该折，Folds 仍表示外层划分数。
    if ($FoldIndex -gt 0) { $runArguments += @('--fold_index', $FoldIndex) }
    # 直接显示终端输出，不额外生成日志文件。
    & $PythonPath @runArguments
    if ($LASTEXITCODE -ne 0) { throw "VAMS failed. Check the error shown above." }
} finally {
    # 正常完成或发生错误时，都恢复用户原来的工作目录。
    Pop-Location
}
