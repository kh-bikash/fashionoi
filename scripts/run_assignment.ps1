param(
    [int]$MaxImages = 1000,
    [string]$Device = "",
    [switch]$SkipInstall,
    [switch]$WithFaiss
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Venv = Join-Path $Root ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    $SystemPython = Get-Command python -ErrorAction SilentlyContinue
    if (-not $SystemPython) {
        throw "Python 3.10+ was not found on PATH. Install Python, then rerun this script."
    }
    & $SystemPython.Source -m venv $Venv
}

if (-not $SkipInstall) {
    & $Python -m pip install --upgrade pip
    $Requirements = if ($WithFaiss) { "requirements-faiss.txt" } else { "requirements.txt" }
    & $Python -m pip install -r (Join-Path $Root $Requirements)
}

$IndexArgs = @(
    (Join-Path $Root "scripts\index.py"),
    "--image-dir", (Join-Path $Root "val_test2020\test"),
    "--output", (Join-Path $Root "artifacts\fashionpedia-1000"),
    "--max-images", $MaxImages
)
if ($Device) { $IndexArgs += @("--device", $Device) }
if (-not $WithFaiss) { $IndexArgs += "--no-faiss" }
& $Python @IndexArgs

$OutputDir = Join-Path $Root "outputs\evaluation"
New-Item -ItemType Directory -Force $OutputDir | Out-Null
$SearchArgs = @(
    (Join-Path $Root "scripts\search_all.py"),
    "--queries", (Join-Path $Root "evaluation\queries.json"),
    "--index", (Join-Path $Root "artifacts\fashionpedia-1000"),
    "--output", $OutputDir,
    "-k", "10"
)
if ($Device) { $SearchArgs += @("--device", $Device) }
& $Python @SearchArgs

Write-Host ""
Write-Host "Done. Inspect results in: $OutputDir" -ForegroundColor Green
Write-Host "Next: publish the repository, then regenerate the PDF with --repo-url."
