#!/usr/bin/env pwsh
# Script to download VstsTaskSdk directly from the GitHub repository
# and place it in the correct task folder

# Get the repository root directory regardless of where the script is invoked from
$scriptsFolder = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
. $scriptsFolder\set-variables.ps1

# load utils
. $utilsPath

Write-Host "Repository root: $repoRoot"

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
    $buildResultDir = Join-Path -Path $tempDir -ChildPath "azure-pipelines-task-lib/powershell/_build/VstsTaskSdk"
    
    if (-not (Test-Path -Path $buildResultDir)) {
        throw "VstsTaskSdk directory not found in cloned repository"
    }
    # check if sourceDir contains "VstsTaskSdk.psm1"
    if (-not (Test-Path -Path "$buildResultDir/VstsTaskSdk.psm1")) {
        throw "VstsTaskSdk.psm1 not found in source directory"
    }
    
    # Copy the VstsTaskSdk module to all the versions directory
    foreach ($version in $versions) {
        $vstsTaskSdkDestPath = Join-Path -Path $prodExtensionDir -ChildPath "tasks/AIAgentEvaluation/$version/ps_modules/VstsTaskSdk"
        Copy-Directory -SourceDir $buildResultDir -DestinationDir $vstsTaskSdkDestPath
    }
    Write-Host "Copied VstsTaskSdk module to production directory" -ForegroundColor Green
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