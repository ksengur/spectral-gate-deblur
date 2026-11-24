# Factorial unrolled study (embedding vs data term): 4 arms for the given seeds.
# Same command on every machine; skips runs that already have eval.csv, so it can be restarted.
# Usage: .\run_factorial.ps1 -Train data\DIV2K\DIV2K_train_HR -Valid data\DIV2K\DIV2K_valid_HR -Seeds 1
#        [-Python C:\path\to\python.exe] [-Arms unrolled,unrolled_emb]
param(
    [Parameter(Mandatory = $true)][string]$Train,
    [Parameter(Mandatory = $true)][string]$Valid,
    [int[]]$Seeds = @(1),
    [string[]]$Arms = @("unrolled", "unrolled_emb", "unrolled_both", "unrolled_blind"),
    [string]$Python = "python",
    [int]$Iters = 50000
)
Set-Location (Split-Path -Parent $PSScriptRoot)
foreach ($seed in $Seeds) {
    foreach ($arm in $Arms) {
        $run = "runs_cond/${arm}_s$seed"
        if (Test-Path "$run/eval.csv") { Write-Output "skip $run (done)"; continue }
        Write-Output "=== $run  $(Get-Date -Format 'yyyy-MM-dd HH:mm') ==="
        & $Python train_2x2.py --train-dir $Train --out $run --train-order mix --cond $arm --seed $seed --iters $Iters
        if ($LASTEXITCODE -ne 0) { Write-Output "train failed: $run"; continue }
        & $Python eval_2x2.py --run $run --test-dir $Valid
    }
}
# blind reference needed for paired comparisons on this seed
foreach ($seed in $Seeds) {
    if (-not (Test-Path "runs/mix_none_s$seed/eval.csv")) {
        Write-Output "NOTE: runs/mix_none_s$seed is missing on this machine; copy it from the main machine for analysis."
    }
}
