#!/usr/bin/env pwsh
# Script to download VstsTaskSdk directly from the GitHub repository
# and place it in the correct task folder

# Get the repository root directory regardless of where the script is invoked from
$scriptPath = $MyInvocation.MyCommand.Path
$scriptsFolder = Split-Path -Path $scriptPath -Parent
$repoRoot = Split-Path -Path $scriptsFolder -Parent

# load utils
$utilsPath = Join-Path -Path $repoRoot -ChildPath "scripts/utilities.ps1"
. $utilsPath

Write-Host "Repository root: $repoRoot"

# Define the target task directory
$taskModulesPath = Join-Path -Path $repoRoot -ChildPath "tasks/AIAgentEvaluation/ps_modules/VstsTaskSdk"

# Create the ps_modules/VstsTaskSdk directory if it doesn't exist
if (-not (Test-Path -Path $taskModulesPath)) {
    New-Item -Path $taskModulesPath -ItemType Directory -Force | Out-Null
    Write-Host "Created directory: $taskModulesPath"
}

# Create a temporary directory to clone the repo
$tempEnv = $env:TEMP
if ([string]::IsNullOrWhiteSpace($tempEnv)) {
    $tempEnv = $env:TMPDIR
}
if ([string]::IsNullOrWhiteSpace($tempEnv)) {
    $tempEnv = "/tmp"  # fallback default for Linux
}
$tempDir = Join-Path -Path $tempEnv -ChildPath "VstsTaskSdk_$(Get-Random)"

New-Item -Path $tempDir -ItemType Directory -Force | Out-Null
Write-Host "Created temporary directory: $tempDir"

try {
    $currentLocation = Get-Location
    Set-Location -Path $tempDir
    
    # Clone the repository (shallow clone to save time/bandwidth)
    Write-Host "Cloning the azure-pipelines-task-lib repository..."
    git clone --depth 1 https://github.com/microsoft/azure-pipelines-task-lib.git
    
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to clone the repository"
    }
    
    # build powershell folder
    Write-Host "Building the powershell folder..."
    $buildScriptPath = Join-Path -Path $tempDir -ChildPath "azure-pipelines-task-lib/powershell"

    Set-Location -Path $buildScriptPath
    npm ci --force
    npm run build

    # Navigate to the VstsTaskSdk directory in the cloned repo
    $sourceDir = Join-Path -Path $tempDir -ChildPath "azure-pipelines-task-lib/powershell/_build/VstsTaskSdk"
    
    if (-not (Test-Path -Path $sourceDir)) {
        throw "VstsTaskSdk directory not found in cloned repository"
    }
    
    Write-Host "Copying VstsTaskSdk files from '$sourceDir' to '$taskModulesPath'..."

    # check if sourceDir contains "VstsTaskSdk.psm1"
    if (-not (Test-Path -Path "$sourceDir/VstsTaskSdk.psm1")) {
        throw "VstsTaskSdk.psm1 not found in source directory"
    }

    Copy-Directory -SourceDir $sourceDir -DestinationDir $taskModulesPath
    Write-Host "Copied following files:"
    Get-ChildItem -Path $taskModulesPath -File | ForEach-Object { Write-Host $_.FullName }
    Write-Host "Successfully copied VstsTaskSdk to the task module directory"
}
catch {
    Write-Error "An error occurred: $_"
    exit 1
}
finally {
    # Return to the original location
    Set-Location -Path $currentLocation
    
    # Clean up the temporary directory
    if (Test-Path -Path $tempDir) {
        Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "Cleaned up temporary directory"
    }
}

Write-Host "VstsTaskSdk has been successfully downloaded and installed to the correct location."