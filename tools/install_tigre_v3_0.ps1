param(
    [string]$ProjectRoot = "D:\code\X2CBCT",
    [string]$Proxy = "http://127.0.0.1:7897",
    [string]$TigreTag = "v3.0",
    [string]$VsDevCmd = "E:\softwave\visual_C\Common7\Tools\VsDevCmd.bat",
    [string]$CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$python = Join-Path $ProjectRoot ".conda\x2cbct-v0\python.exe"
$tigreRoot = Join-Path $ProjectRoot "external\TIGRE"
$logRoot = Join-Path $ProjectRoot "runs\tigre_build"
$cmdPath = Join-Path $logRoot "install_tigre_v3_0.cmd"
$logPath = Join-Path $logRoot "install_v3_0_script.log"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project virtualenv python not found: $python"
}
if (-not (Test-Path -LiteralPath $VsDevCmd)) {
    throw "Visual Studio dev command not found: $VsDevCmd"
}
if (-not (Test-Path -LiteralPath (Join-Path $CudaRoot "bin\nvcc.exe"))) {
    throw "CUDA nvcc not found under: $CudaRoot"
}

New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "external") | Out-Null
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

if (-not (Test-Path -LiteralPath (Join-Path $tigreRoot ".git"))) {
    git -c "http.proxy=$Proxy" -c "https.proxy=$Proxy" clone https://github.com/CERN/TIGRE.git $tigreRoot
}

git -C $tigreRoot checkout $TigreTag

& $python -m pip install "Cython==0.29.37" "matplotlib==3.7.5"

$setupPath = Join-Path $tigreRoot "Python\setup.py"
$setupText = Get-Content -LiteralPath $setupPath -Raw -Encoding UTF8
if ($setupText -notmatch "_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH") {
    $setupText = $setupText.Replace(
        'COMMON_MSVC_FLAGS = ["/MD", "/wd4819", "/EHsc"]',
        'COMMON_MSVC_FLAGS = ["/MD", "/wd4819", "/EHsc", "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"]'
    )
    $setupText = $setupText.Replace(
        "COMMON_NVCC_FLAGS = [`r`n    `"-D__CUDA_NO_HALF_OPERATORS__`",",
        "COMMON_NVCC_FLAGS = [`r`n    `"-allow-unsupported-compiler`",`r`n    `"-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`",`r`n    `"-D__CUDA_NO_HALF_OPERATORS__`","
    )
    $setupText = $setupText.Replace(
        "COMMON_NVCC_FLAGS = [`n    `"-D__CUDA_NO_HALF_OPERATORS__`",",
        "COMMON_NVCC_FLAGS = [`n    `"-allow-unsupported-compiler`",`n    `"-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`",`n    `"-D__CUDA_NO_HALF_OPERATORS__`","
    )
    Set-Content -LiteralPath $setupPath -Value $setupText -Encoding UTF8
}

@"
call "$VsDevCmd" -arch=x64 -host_arch=x64
set "CUDA_PATH=$CudaRoot"
set "CUDA_HOME=$CudaRoot"
set "PATH=$ProjectRoot\.conda\x2cbct-v0\Scripts;$CudaRoot\bin;%PATH%"
cd /d "$tigreRoot\Python"
"$python" -m pip install . --no-deps --no-build-isolation -v
"@ | Set-Content -LiteralPath $cmdPath -Encoding ASCII

cmd.exe /c $cmdPath *> $logPath
if ($LASTEXITCODE -ne 0) {
    throw "TIGRE build failed. See log: $logPath"
}

& $python (Join-Path $ProjectRoot "tools\check_tigre_dependency.py")
