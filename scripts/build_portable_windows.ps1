[CmdletBinding()]
param(
    [string]$BuildPython = "python",
    [string]$OutputDirectory,
    [string]$CacheDirectory
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $repoRoot "portable-build"
}
if (-not $CacheDirectory) {
    $CacheDirectory = Join-Path $repoRoot ".portable-cache"
}

$appVersion = "0.3.0"
$bundleName = "CUS-AI-reader-offline-windows-x64-v$appVersion"
$stageContainer = Join-Path ([System.IO.Path]::GetTempPath()) "CUSAI-portable-build"
$stageRoot = Join-Path $stageContainer $bundleName
$archivePath = Join-Path $OutputDirectory "$bundleName.zip"
$pythonArchive = Join-Path $CacheDirectory "python-3.12.10-embed-amd64.zip"
$pythonUri = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip"
$pythonSha256 = "4ACBED6DD1C744B0376E3B1CF57CE906F9DC9E95E68824584C8099A63025A3C3"

function Get-RemoteFile {
    param([string]$Uri, [string]$Destination)
    Add-Type -AssemblyName System.Net.Http
    $client = [System.Net.Http.HttpClient]::new()
    try {
        $bytes = $client.GetByteArrayAsync($Uri).GetAwaiter().GetResult()
        [System.IO.File]::WriteAllBytes($Destination, $bytes)
    }
    finally {
        $client.Dispose()
    }
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
New-Item -ItemType Directory -Force -Path $CacheDirectory | Out-Null
New-Item -ItemType Directory -Force -Path $stageContainer | Out-Null

if (-not (Test-Path -LiteralPath $pythonArchive)) {
    Write-Host "Downloading the official Python 3.12.10 embedded runtime..."
    Get-RemoteFile -Uri $pythonUri -Destination $pythonArchive
}

$actualPythonHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $pythonArchive).Hash
if ($actualPythonHash -ne $pythonSha256) {
    throw "Python runtime hash verification failed."
}

$stageContainerResolved = [System.IO.Path]::GetFullPath($stageContainer)
$stageRootResolved = [System.IO.Path]::GetFullPath($stageRoot)
if (-not $stageRootResolved.StartsWith($stageContainerResolved, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "The staging folder must remain inside the dedicated temporary build folder."
}
if (Test-Path -LiteralPath $stageRoot) {
    Remove-Item -LiteralPath $stageRoot -Recurse -Force
}
if (Test-Path -LiteralPath $archivePath) {
    Remove-Item -LiteralPath $archivePath -Force
}

$runtimePath = Join-Path $stageRoot "runtime"
$sitePackages = Join-Path $runtimePath "Lib\site-packages"
New-Item -ItemType Directory -Force -Path $runtimePath | Out-Null
Expand-Archive -LiteralPath $pythonArchive -DestinationPath $runtimePath
New-Item -ItemType Directory -Force -Path $sitePackages | Out-Null

$pthFile = Join-Path $runtimePath "python312._pth"
@(
    "python312.zip"
    "."
    ".."
    "Lib\site-packages"
    "import site"
) | Set-Content -LiteralPath $pthFile -Encoding ASCII

Write-Host "Installing the locked application dependencies into the portable runtime..."
& $BuildPython -m pip install `
    --disable-pip-version-check `
    --no-compile `
    --upgrade `
    --target $sitePackages `
    -r (Join-Path $repoRoot "requirements-portable.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed."
}

$filesToCopy = @(
    "app.py"
    "portable_launcher.py"
    "Start CUS AI Reader.cmd"
    "Portable README.txt"
    "README.md"
    "pyproject.toml"
    "requirements.txt"
    "requirements-portable.txt"
)
foreach ($relativePath in $filesToCopy) {
    Copy-Item -LiteralPath (Join-Path $repoRoot $relativePath) -Destination (Join-Path $stageRoot $relativePath)
}
foreach ($folder in @("cus_ai", "models", ".streamlit", "docs")) {
    Copy-Item -LiteralPath (Join-Path $repoRoot $folder) -Destination (Join-Path $stageRoot $folder) -Recurse
}

Write-Host "Running the portable dependency check..."
& (Join-Path $runtimePath "python.exe") (Join-Path $stageRoot "portable_launcher.py") --check
if ($LASTEXITCODE -ne 0) {
    throw "Portable dependency verification failed."
}

$manifestPath = Join-Path $stageRoot "MANIFEST-SHA256.txt"
$manifestLines = Get-ChildItem -LiteralPath $stageRoot -File -Recurse |
    Where-Object { $_.FullName -ne $manifestPath } |
    Sort-Object FullName |
    ForEach-Object {
        $relative = $_.FullName.Substring($stageRoot.Length + 1).Replace("\", "/")
        $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash
        "$hash  $relative"
    }
$manifestLines | Set-Content -LiteralPath $manifestPath -Encoding ASCII

Write-Host "Creating the offline ZIP..."
Compress-Archive -LiteralPath $stageRoot -DestinationPath $archivePath -CompressionLevel Optimal
$archiveHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash
$hashFile = "$archivePath.sha256.txt"
"$archiveHash  $(Split-Path -Leaf $archivePath)" | Set-Content -LiteralPath $hashFile -Encoding ASCII

Write-Host "Portable bundle created:"
Write-Host $archivePath
Write-Host "SHA256: $archiveHash"
