param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TrainArgs
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $ProjectRoot ".conda\x2cbct-v0\python.exe"
$CondaScripts = Join-Path $ProjectRoot ".conda\x2cbct-v0\Scripts"
$TrainScript = Join-Path $ProjectRoot "tools\train_difnet_v0b.py"

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
    $MsvcRoot = Join-Path $Instance.installationPath "VC\Tools\MSVC"
    $ClCandidates = @()
    if (Test-Path $MsvcRoot) {
        $ClCandidates = Get-ChildItem -Path $MsvcRoot -Recurse -Filter cl.exe -ErrorAction SilentlyContinue
    }
    if ((Test-Path $Candidate) -and $ClCandidates) {
        $VcVars = $Candidate
        break
    }
}
if (-not $VcVars) {
    throw "No Visual Studio x64 C++ build environment with cl.exe was found."
}

$EnvOutput = & cmd.exe /d /s /c "call `"$VcVars`" >nul && set"
foreach ($Line in $EnvOutput) {
    if ($Line -match "^([^=]+)=(.*)$") {
        [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
    }
}

if (-not $env:CUDA_PATH) {
    throw "CUDA_PATH is not set after loading the build environment."
}
$env:CUDA_HOME = $env:CUDA_PATH
$env:PATH = "$CondaScripts;$env:PATH"

& $Python $TrainScript @TrainArgs
exit $LASTEXITCODE
