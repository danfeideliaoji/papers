param(
    [int]$Epochs = 2,
    [int]$Folds = 2,
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [string]$PythonPath = '',
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
if (-not $PythonPath) {
    $PythonPath = if ($Device -eq 'cuda') {
        'D:\python-venv\cuda\.venv\Scripts\python.exe'
    } else {
        'D:\python-venv\deeplearn.venv\Scripts\python.exe'
    }
}
if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Python environment missing: $PythonPath. Supply -PythonPath with an installed environment."
}
$env:MPLBACKEND = 'Agg'
$env:PYTHONUTF8 = '1'
Push-Location $PSScriptRoot
try {
    $checkpointDirectory = if ($Device -eq 'cuda') { './Model_Best_GPU' } else { './Model_Best' }
    $runArguments = @('-u', 'demo.py', '--device', $Device, '--dataset', 'emotions.mat',
        '--data_root', './data', '--k_fold', $Folds, '--end_epochs', $Epochs,
        '--checkpoint_dir', $checkpointDirectory)
    if (-not $Resume) { $runArguments += '--fresh_start' }
    & $PythonPath @runArguments | Tee-Object -FilePath "run_windows_$Device.log"
    if ($LASTEXITCODE -ne 0) { throw "VAMS failed. See run_windows_$Device.log." }
} finally {
    Pop-Location
}
