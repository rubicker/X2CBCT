$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BasePython = Join-Path $ProjectRoot ".conda\x2cbct-v0\python.exe"
$EnvironmentRoot = Join-Path $ProjectRoot ".conda\lungmask"
$EnvironmentPython = Join-Path $EnvironmentRoot "Scripts\python.exe"
$Requirements = Join-Path $ProjectRoot "configs\lungmask\requirements.txt"

if (-not (Test-Path -LiteralPath $BasePython)) {
    throw "Base X2CBCT Python not found: $BasePython"
}

if (-not (Test-Path -LiteralPath $EnvironmentPython)) {
    & $BasePython -m venv --system-site-packages $EnvironmentRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create LungMask environment"
    }
}

& $EnvironmentPython -m pip install --disable-pip-version-check -r $Requirements
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install LungMask requirements"
}

& $EnvironmentPython -c "import importlib.metadata as md, torch; from lungmask import LMInferer; print('lungmask=' + md.version('lungmask')); print('torch=' + torch.__version__); print('cuda_available=' + str(torch.cuda.is_available())); print('device=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'))"
if ($LASTEXITCODE -ne 0) {
    throw "LungMask environment verification failed"
}
