# Sensitivity of the two-sided gate to the Wiener weight of the nu score (fallback unchanged, lambda = 300 s^2).
param([string]$Python = "python")
Set-Location (Split-Path -Parent $PSScriptRoot)
$runs = @()
foreach ($s in 0, 1, 2) { foreach ($a in "unrolled", "unrolled_emb") { $runs += "runs_cond/${a}_s$s" } }
foreach ($lam in 100, 900) {
    Write-Output "=== lambda $lam $(Get-Date -Format 'HH:mm') ==="
    & $Python -u gate4.py --runs $runs --train-dir data\DIV2K\DIV2K_train_HR --test-dir data\DIV2K\DIV2K_valid_HR `
        --nu-alpha $lam --out-root "outputs/gate4_lam$lam"
}
Write-Output "ALL DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
