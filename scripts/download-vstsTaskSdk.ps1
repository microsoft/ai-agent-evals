#!/usr/bin/env pwsh
# Script to download VstsTaskSdk directly from the GitHub repository
# and place it in the correct task folder

# Get the repository root directory regardless of where the script is invoked from
$scriptPath = $MyInvocation.MyCommand.Path
$scriptsFolder = Split-Path -Path $scriptPath -Parent
$repoRoot = Split-Path -Path $scriptsFolder -Parent

Write-Host "Repository root: $repoRoot"

# Define the target task directory
$taskModulesPath = Join-Path -Path $repoRoot -ChildPath "tasks/AIAgentEvaluation/ps_modules/VstsTaskSdk"

# Create the ps_modules/VstsTaskSdk directory if it doesn't exist
if (-not (Test-Path -Path $taskModulesPath)) {
    New-Item -Path $taskModulesPath -ItemType Directory -Force | Out-Null
    Write-Host "Created directory: $taskModulesPath"
}

# Create a temporary directory to clone the repo
$tempDir = Join-Path -Path $env:TEMP -ChildPath "VstsTaskSdk_$(Get-Random)"
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
    
    # Navigate to the VstsTaskSdk directory in the cloned repo
    $sourceDir = Join-Path -Path $tempDir -ChildPath "azure-pipelines-task-lib/powershell/VstsTaskSdk"
    
    if (-not (Test-Path -Path $sourceDir)) {
        throw "VstsTaskSdk directory not found in cloned repository"
    }
    
    Write-Host "Copying VstsTaskSdk files from '$sourceDir' to '$taskModulesPath'..."
    Get-ChildItem -Path $sourceDir -Recurse -File | ForEach-Object {
        $relativePath = $_.FullName.Substring($sourceDir.Length).TrimStart('\','/')
        $targetPath = Join-Path $taskModulesPath $relativePath
        $targetDir = Split-Path -Path $targetPath -Parent

        if (-not (Test-Path -Path $targetDir)) {
            New-Item -Path $targetDir -ItemType Directory -Force | Out-Null
        }

        Copy-Item -Path $_.FullName -Destination $targetPath -Force
    }
    
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