param(
    [string[]]$ModelVersion = @(
        "v2-hash-a1",
        "v2-hash-a1a",
        "v2-hash-a1b",
        "v2-hash-a2",
        "v2-hash-a2a",
        "v2-hash-a2b"
    ),
    [int]$NumViews = 2,
    [int]$ImageSize = 32,
    [int]$NumPoints = 32,
    [int]$MidChannels = 16,
    [string]$Combine = "mlp",
    [string]$OutputDir = "runs\smoke\v2_hash_cuda"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $ProjectRoot ".conda\x2cbct-v0\python.exe"
$CondaScripts = Join-Path $ProjectRoot ".conda\x2cbct-v0\Scripts"
$SmokeScript = Join-Path $ProjectRoot "tools\smoke_difnet_v0_forward.py"

if (-not (Test-Path $Python)) {
    throw "Python environment not found: $Python"
}

$VsWhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $VsWhere)) {
    throw "vswhere.exe not found: $VsWhere"
}

$VsInstances = & $VsWhere -all -products * -format json | ConvertFrom-Json
$VcVars = $null
foreach ($Instance in $VsInstances) {
    $Candidate = Join-Path $Instance.installationPath "VC\Auxiliary\Build\vcvars64.bat"
    $ClCandidates = Get-ChildItem -Path (Join-Path $Instance.installationPath "VC\Tools\MSVC") -Recurse -Filter cl.exe -ErrorAction SilentlyContinue
    if ((Test-Path $Candidate) -and $ClCandidates) {
        $VcVars = $Candidate
        break
    }
}
if (-not $VcVars) {
    throw "No Visual Studio x64 C++ build environment with cl.exe was found."
}

New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot $OutputDir) | Out-Null

foreach ($Model in $ModelVersion) {
    $Output = Join-Path $ProjectRoot (Join-Path $OutputDir "$($Model)_cuda_forward_backward.json")
    $Command = @(
        "call `"$VcVars`"",
        "set `"CUDA_HOME=!CUDA_PATH!`"",
        "set `"PATH=$CondaScripts;!PATH!`"",
        "`"$Python`" `"$SmokeScript`" --model-version $Model --hashgrid-backend cuda --device cuda --backward --num-views $NumViews --image-size $ImageSize --num-points $NumPoints --mid-channels $MidChannels --combine $Combine --output `"$Output`""
    ) -join " && "
    & cmd.exe /v:on /d /s /c $Command
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA smoke failed for $Model"
    }
}
