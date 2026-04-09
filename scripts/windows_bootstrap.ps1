param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")),
    [switch]$ForceCpu
)

$ErrorActionPreference = "Stop"

function Test-CommandAvailable {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

Write-Host "Project root: $ProjectRoot"

if (-not (Test-CommandAvailable "uv")) {
    Write-Host "Installing uv..."
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$HOME\\.local\\bin;" + $env:Path
}

$hasGpu = $false
if (-not $ForceCpu -and (Test-CommandAvailable "nvidia-smi")) {
    $hasGpu = $true
}

Set-Location $ProjectRoot
if ($hasGpu) {
    Write-Host "NVIDIA GPU detected. Installing GPU runtime extras."
    uv sync --extra dev --extra windows-gpu
} else {
    Write-Host "GPU not detected or CPU mode forced. Installing CPU runtime extras."
    uv sync --extra dev --extra windows-cpu
}

Write-Host "Running Windows healthcheck..."
uv run python scripts/healthcheck.py windows

Write-Host "Bootstrap completed."
