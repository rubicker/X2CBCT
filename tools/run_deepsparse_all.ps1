param(
    [switch]$Smoke,
    [switch]$DryRun,
    [ValidateSet("all", "luna16", "real")]
    [string]$Dataset = "all"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".conda\deepsparse\Scripts\python.exe"
$Trainer = Join-Path $ProjectRoot "tools\train_deepsparse.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "DeepSparse environment is missing: $Python"
}

$Configs = @()
if ($Dataset -in @("all", "luna16")) {
    $Configs += 2, 5, 8, 10 | ForEach-Object {
        Join-Path $ProjectRoot "configs\deepsparse\v$($_)_luna16_noaug.yaml"
    }
}
if ($Dataset -in @("all", "real")) {
    $Configs += 2, 5, 8, 10 | ForEach-Object {
        Join-Path $ProjectRoot "configs\deepsparse\v$($_)_real_noaug.yaml"
    }
}

foreach ($Config in $Configs) {
    $Arguments = @($Trainer, "--config", $Config)
    if ($Smoke) {
        $Arguments += "--smoke"
    }
    Write-Host "Running DeepSparse config: $Config"
    if ($DryRun) {
        Write-Host "DRY RUN: $Python $($Arguments -join ' ')"
        continue
    }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "DeepSparse run failed with exit code ${LASTEXITCODE}: $Config"
    }
}
