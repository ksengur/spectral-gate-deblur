# Second training distribution: the four factorial arms trained on Gaussian + motion kernels (--train-kernels mix).
# Runs are named runs_cond/<arm>_mixk_s<seed>; restartable (skips runs with eval.csv / eval_shift.csv).
# Usage: .\run_mixk.ps1 [-Seeds 0] [-Arms unrolled,unrolled_emb]
param(
    [int[]]$Seeds = @(0),
    [string[]]$Arms = @("unrolled", "unrolled_emb", "unrolled_both", "unrolled_blind"),
    [string]$Python = "python",
    [string]$Train = "data\DIV2K\DIV2K_train_HR",
    [string]$Valid = "data\DIV2K\DIV2K_valid_HR",
    [int]$Iters = 50000
)
Set-Location (Split-Path -Parent $PSScriptRoot)
foreach ($seed in $Seeds) {
    foreach ($arm in $Arms) {
        $run = "runs_cond/${arm}_mixk_s$seed"
        if (-not (Test-Path "$run/eval.csv")) {
            Write-Output "=== $run  $(Get-Date -Format 'yyyy-MM-dd HH:mm') ==="
            & $Python -u train_2x2.py --train-dir $Train --out $run --train-order mix --cond $arm --seed $seed `
                --iters $Iters --train-kernels mix
            if ($LASTEXITCODE -ne 0) { Write-Output "train failed: $run"; continue }
            & $Python -u eval_2x2.py --run $run --test-dir $Valid
        }
        if (-not (Test-Path "$run/eval_shift.csv")) {
            & $Python -u eval_shift.py --runs $run --test-dir $Valid
        }
    }
}
Write-Output "ALL DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
