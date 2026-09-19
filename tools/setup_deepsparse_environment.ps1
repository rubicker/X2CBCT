param(
    [switch]$SkipDownload
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Uv = Get-Command uv -ErrorAction Stop
$BasePython = Join-Path $ProjectRoot ".conda\x2cbct-v0\python.exe"
$Environment = Join-Path $ProjectRoot ".conda\deepsparse"
$Python = Join-Path $Environment "Scripts\python.exe"
$Requirements = Join-Path $ProjectRoot "configs\deepsparse\requirements-runtime.txt"
$UvCache = Join-Path $ProjectRoot ".tmp\uv-cache"
$HfHome = Join-Path $ProjectRoot ".tmp\hf-home"
$Logs = Join-Path $ProjectRoot "external\deepsparse_official\logs"

if (-not (Test-Path -LiteralPath $BasePython)) {
    throw "CUDA base environment is missing: $BasePython"
}

$env:UV_CACHE_DIR = $UvCache
if (-not (Test-Path -LiteralPath $Python)) {
    & $Uv.Source venv $Environment --python $BasePython --system-site-packages
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create DeepSparse environment"
    }
}

& $Uv.Source pip install --python $Python -r $Requirements
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install DeepSparse runtime packages"
}

if (-not $SkipDownload) {
    $env:HF_HOME = $HfHome
    & $Python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='HajihajihaJimmy/DeepSparse', filename='pretrain/ep_700.pth', local_dir=r'$Logs'))"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to download the DeepSparse foundation checkpoint"
    }
}

& $Python (Join-Path $ProjectRoot "tools\check_deepsparse_environment.py")
if ($LASTEXITCODE -ne 0) {
    throw "DeepSparse environment validation failed"
}
