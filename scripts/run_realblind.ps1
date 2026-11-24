# Real Levin captures with blind noise estimates (MAD, Immerkaer) instead of the oracle sigma: networks + gate.
# Outputs outputs/gate4_real_<est>[_mixk]/<run>/gate4_real.csv
param([string]$Python = "python")
Set-Location (Split-Path -Parent $PSScriptRoot)
$gauss = @()
foreach ($s in 0, 1, 2) { foreach ($a in "unrolled", "unrolled_emb", "unrolled_both", "unrolled_blind") { $gauss += "runs_cond/${a}_s$s" } }
$mixk = @("unrolled", "unrolled_emb", "unrolled_both", "unrolled_blind") | ForEach-Object { "runs_cond/${_}_mixk_s0" }
foreach ($est in "mad", "immerkaer") {
    Write-Output "=== $est gauss-trained $(Get-Date -Format 'HH:mm') ==="
    & $Python -u gate4.py --runs $gauss --train-dir data\DIV2K\DIV2K_train_HR --test-dir data\DIV2K\DIV2K_valid_HR `
        --real-sigma $est --out-root "outputs/gate4_real_$est"
    Write-Output "=== $est mixture-trained $(Get-Date -Format 'HH:mm') ==="
    & $Python -u gate4.py --runs $mixk --blind runs_cond/unrolled_blind_mixk_s0 --train-dir data\DIV2K\DIV2K_train_HR `
        --test-dir data\DIV2K\DIV2K_valid_HR --real-sigma $est --calib-kernels mix --out-root "outputs/gate4_real_${est}_mixk"
}
Write-Output "ALL DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
