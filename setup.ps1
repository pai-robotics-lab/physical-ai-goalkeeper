param(
    [ValidateSet("cpu", "cuda")]
    [string]$Backend = "cpu",
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$env:QD_OFFLINE_CACHE_FILE_PATH = Join-Path $ProjectRoot ".cache\quadrants"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $Arguments"
    }
}

if (-not $Python) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCommand -and $PythonCommand.Source -notlike "*WindowsApps*") {
        $Python = $PythonCommand.Source
    }
    else {
        $BundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
        if (Test-Path $BundledPython) {
            $Python = $BundledPython
        }
        else {
            throw "Python 3.10-3.13 was not found. Install Python 3.12 and run this script again."
        }
    }
}

Write-Host "Using Python: $Python"
if (-not (Test-Path $VenvPython)) {
    # The Codex-bundled Python intentionally omits ensurepip, so bootstrap pip
    # from the base interpreter after creating an isolated environment.
    Invoke-Checked -FilePath $Python -Arguments @("-m", "venv", "--without-pip", (Join-Path $ProjectRoot ".venv"))
}

try {
    Invoke-Checked -FilePath $VenvPython -Arguments @("-m", "pip", "--version")
}
catch {
    Invoke-Checked -FilePath $Python -Arguments @(
        "-m", "pip", "--python", (Join-Path $ProjectRoot ".venv"),
        "install", "pip", "setuptools", "wheel"
    )
}

Invoke-Checked -FilePath $VenvPython -Arguments @(
    "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"
)

if ($Backend -eq "cuda") {
    Write-Warning "CUDA mode requires a PyTorch-supported NVIDIA GPU. GTX 10-series users should prefer CPU mode."
    Invoke-Checked -FilePath $VenvPython -Arguments @(
        "-m", "pip", "install", "torch==2.14.1", "--index-url", "https://download.pytorch.org/whl/cu126"
    )
}
else {
    Invoke-Checked -FilePath $VenvPython -Arguments @(
        "-m", "pip", "install", "torch==2.14.1", "--index-url", "https://download.pytorch.org/whl/cpu"
    )
}

Invoke-Checked -FilePath $VenvPython -Arguments @(
    "-m", "pip", "install", "-e", "$ProjectRoot[dev,media]"
)
Invoke-Checked -FilePath $VenvPython -Arguments @(
    "-m", "pai_final.check_env", "--backend", $Backend
)

Write-Host ""
Write-Host "Setup complete. Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "Run the smoke test with: python -m pai_final.smoke_test"
