# Conditioning-shortcut pilot: does explicit degradation conditioning hurt OOD generalization,
# and does hiding the signal during training (token dropout) fix it?
# Baselines reused from runs/: mix_none (blind) and mix_oracle (order token, no dropout).
# Usage: .\run_cond.ps1 -Train "data\DIV2K\DIV2K_train_HR" -Valid "data\DIV2K\DIV2K_valid_HR" [-Seeds 0]
param(
    [Parameter(Mandatory = $true)][string]$Train,
    [Parameter(Mandatory = $true)][string]$Valid,
    [int]$Iters = 50000,
    [int[]]$Seeds = @(0)
)
$ErrorActionPreference = "Stop"
$configs = @(
    @{ name = "params";       cond = "params"; drop = 0.0 },
    @{ name = "params_do50";  cond = "params"; drop = 0.5 },
    @{ name = "oracle_do50";  cond = "oracle"; drop = 0.5 },
    # control: lossless full-kernel descriptor (separates misspecification from shortcut)
    @{ name = "kernel";       cond = "kernel"; drop = 0.0 },
    @{ name = "kernel_do50";  cond = "kernel"; drop = 0.5 },
    # physics-based conditioning: Wiener estimate with the given kernel as extra input
    @{ name = "wiener";       cond = "wiener"; drop = 0.0 }
)
foreach ($seed in $Seeds) {
    foreach ($c in $configs) {
        $name = "runs_cond/$($c.name)_s$seed"
        if (Test-Path "$name/eval.csv") { Write-Host "skip $name (done)"; continue }
        if (-not (Test-Path "$name/last.pt") -or -not (Select-String -Quiet -Path "$name/train_log.csv" -Pattern "^$Iters,")) {
            python train_2x2.py --train-dir $Train --out $name --train-order mix --cond $c.cond --token-dropout $c.drop --seed $seed --iters $Iters
            if ($LASTEXITCODE -ne 0) { throw "training failed: $name" }
        }
        python eval_2x2.py --run $name --test-dir $Valid
        if ($LASTEXITCODE -ne 0) { throw "evaluation failed: $name" }
    }
}
