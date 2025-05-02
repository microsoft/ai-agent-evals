#!/usr/bin/env pwsh
# Utility functions for AI Agent Evaluations extension scripts

# Function to calculate a version number with an auto-incrementing patch version
# based on seconds since a reference date
function Update-VersionNumber {
    param (
        [Parameter(Mandatory = $true)]
        [string]$CurrentVersion
    )

    Write-Host "Current version: $CurrentVersion"

    # Parse the current version
    $versionParts = $CurrentVersion -split '\.'
    if ($versionParts.Count -lt 2) {
        Write-Host "Version format unexpected. Using defaults 0.0"
        $majorVersion = 0
        $minorVersion = 0
    }
    else {
        $majorVersion = [int]$versionParts[0]
        $minorVersion = [int]$versionParts[1]
    }

    # Calculate seconds since a reference date
    $referenceDate = Get-Date -Year 2025 -Month 4 -Day 20 -Hour 0 -Minute 0 -Second 0
    $currentDate = Get-Date
    $secondsSinceReference = [math]::Floor(($currentDate - $referenceDate).TotalSeconds)

    # Use the seconds as the patch version to ensure it's strictly increasing
    # Format as "major.minor.patch"
    $newVersion = "$majorVersion.$minorVersion.$secondsSinceReference"
    
    Write-Host "New version: $newVersion"
    return $newVersion
}

function Copy-Directory {
    param (
        [Parameter(Mandatory = $true)]
        [string]$SourceDir,

        [Parameter(Mandatory = $true)]
        [string]$DestinationDir
    )

    if (-not (Test-Path -Path $SourceDir)) {
        Write-Host "❌ Source directory does not exist: $SourceDir" -ForegroundColor Red
        return $false
    }

    if (-not (Test-Path -Path $DestinationDir)) {
        New-Item -Path $DestinationDir -ItemType Directory -Force | Out-Null
    }

    Copy-Item -Path "$SourceDir/*" -Destination $DestinationDir -Recurse -Force
    Write-Host "✅ Copied files from '$SourceDir' to '$DestinationDir'" -ForegroundColor Green
    return $true
}

function Copy-FilesFromVssExtension {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourceRootPath,
        
        [Parameter(Mandatory = $true)]
        [string]$DestinationRootPath,
        
        [Parameter(Mandatory = $true)]
        [PSCustomObject]$VssExtensionObject
    )
    
    # Read the files section from vss-extension.json
    $files = $VssExtensionObject.files
    
    if ($files -and $files.Count -gt 0) {
        foreach ($file in $files) {
            $sourcePath = Join-Path -Path $SourceRootPath -ChildPath $file.path
            $destinationPath = Join-Path -Path $DestinationRootPath -ChildPath $file.path
            
            Write-Host "Copying $($file.path) to destination directory..."
            if (Test-Path -Path $sourcePath -PathType Container) {
                # Create the destination directory if it doesn't exist
                Copy-Directory -SourceDir $sourcePath -DestinationDir $destinationPath
            }
            else {
                Copy-Item -Path $sourcePath -Destination $destinationPath -Force
            }
        }
        Write-Host "Copied all files defined in 'files' section" -ForegroundColor Green
        return $true
    }
    else {
        Write-Warning "No files defined in vss-extension.json 'files' section or section not found"
        return $false
    }
}

# Function to check if critical files are present in output directory
function Check-CriticalFiles {
    param (
        [Parameter(Mandatory = $true)]
        [string]$OutputDir,

        [Parameter(Mandatory = $true)]
        [bool]$IsDevExtension
    )

    $AgentFolderName = if ($IsDevExtension) { "AIAgentEvaluationDev" } else { "AIAgentEvaluation" }

    $criticalFiles = @(
        "vss-extension.json",
        "analysis/analysis.py",
        "action.py",
        "logo.png",
        "overview.md",
        "pyproject.toml",
        "tasks/$AgentFolderName/task.json",
        "tasks/$AgentFolderName/run.ps1",
        "tasks/$AgentFolderName/ps_modules/VstsTaskSdk/VstsTaskSdk.psm1"
        "tasks/AIAgentReport/dist/index.html"
    )

    foreach ($file in $criticalFiles) {
        $fullPath = Join-Path -Path $OutputDir -ChildPath $file
        if (-not (Test-Path -Path $fullPath)) {
            Write-Host "❌ Critical file not found: $fullPath" -ForegroundColor Red
            return $false
        }
    }

    Write-Host "✅ All critical files are present in the output directory." -ForegroundColor Green
    return $true
}