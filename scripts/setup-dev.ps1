#!/usr/bin/env pwsh
# Script to create the AIAgentEvaluationDev task from AIAgentEvaluation
# and generate dist/dev/extension files for the dev version.

# Get the repository root directory regardless of where the script is invoked from
$scriptPath = $MyInvocation.MyCommand.Path
$scriptsFolder = Split-Path -Path $scriptPath -Parent
$repoRoot = Split-Path -Path $scriptsFolder -Parent

$devExtensionDir = Join-Path -Path $repoRoot -ChildPath "dist/dev"
New-Item -Path $devExtensionDir -ItemType Directory -Force | Out-Null

# Import the utilities module with shared functions
$utilsPath = Join-Path -Path $scriptsFolder -ChildPath "utilities.ps1"
. $utilsPath

$setupPath = Join-Path -Path $scriptsFolder -ChildPath "setup.ps1"
& $setupPath

# Define the source and destination paths
$sourceFolder = Join-Path -Path $repoRoot -ChildPath "tasks\AIAgentEvaluation"
$destFolder = Join-Path -Path $repoRoot -ChildPath "tasks\AIAgentEvaluationDev"

# Create the destination directory if it doesn't exist
if (-not (Test-Path -Path $destFolder)) {
    New-Item -Path $destFolder -ItemType Directory | Out-Null
    Write-Host "Created directory: $destFolder"
}

# Copy all files from source to destination, except task.json
Get-ChildItem -Path $sourceFolder -Recurse | Where-Object { $_.Name -ne "task.json" } | ForEach-Object {
    $targetPath = $_.FullName -replace [regex]::Escape($sourceFolder), $destFolder
    $targetDir = Split-Path -Path $targetPath -Parent
    
    if (-not (Test-Path -Path $targetDir)) {
        New-Item -Path $targetDir -ItemType Directory | Out-Null
    }
    
    if (-not $_.PSIsContainer) {
        Copy-Item -Path $_.FullName -Destination $targetPath -Force
        Write-Host "Copied: $($_.FullName) to $targetPath"
    }
}

# Read the source task.json
$taskJsonPath = Join-Path $sourceFolder "task.json"
$taskJson = Get-Content -Path $taskJsonPath -Raw | ConvertFrom-Json

# Modify the task.json for the dev version
$taskJson.id = "6c8d5e8b-16f2-4f7b-b991-99e3dfa9f359"  # New GUID for dev version
$taskJson.name = "AIAgentEvaluationDev"
$taskJson.friendlyName = "$($taskJson.friendlyName) (Dev)"
$taskJson.description = "$($taskJson.description) (Dev)"
$taskJson.instanceNameFormat = "$($taskJson.instanceNameFormat) (Dev)"

# Write the modified task.json to the destination
$destTaskJsonPath = Join-Path $destFolder "task.json"
$taskJson | ConvertTo-Json -Depth 10 | Set-Content -Path $destTaskJsonPath -Encoding UTF8

Write-Host "Created modified task.json at: $destTaskJsonPath"

# Now handle the vss-extension.json to vss-extension-dev.json conversion
$vssExtensionPath = Join-Path $repoRoot "vss-extension.json"

# Read the original vss-extension.json
$vssExtension = Get-Content -Path $vssExtensionPath -Raw | ConvertFrom-Json

# Modify the vss-extension.json for the dev version
$vssExtension.id = "microsoft-extension-ai-agent-evaluation-dev"
$vssExtension.publisher = "ms-azure-exp-dev"
$vssExtension.name = "$($vssExtension.name) (Dev)"

# Update the version using the shared function
$vssExtension.version = Update-VersionNumber -CurrentVersion $vssExtension.version

# Update contributions section
$buildResultsContribution = $vssExtension.contributions | Where-Object { $_.id -eq "build-results" }
if ($buildResultsContribution) {
    $buildResultsContribution.id = "build-results-dev"
    $buildResultsContribution.description = "$($buildResultsContribution.description) (Dev)"
    $buildResultsContribution.properties.name = "$($buildResultsContribution.properties.name) (Dev)"
    $buildResultsContribution.properties.supportsTasks = @($taskJson.id)
}

# Update AIAgentEvaluation contribution
$agentEvalContribution = $vssExtension.contributions | Where-Object { $_.id -eq "AIAgentEvaluation" }
if ($agentEvalContribution) {
    $agentEvalContribution.id = "AIAgentEvaluationDev"
    $agentEvalContribution.properties.name = "tasks/AIAgentEvaluationDev"
}

# Update files section
$agentEvalFile = $vssExtension.files | Where-Object {
    $_.packagePath -eq "tasks/AIAgentEvaluation" -and
    $_.path -eq "tasks/AIAgentEvaluation" 
}

if ($agentEvalFile) {
    $agentEvalFile.packagePath = "tasks/AIAgentEvaluationDev"
    $agentEvalFile.path = "tasks/AIAgentEvaluationDev"
}

# Update other file paths
foreach ($file in $vssExtension.files | Where-Object { 
        $_.packagePath -like "tasks/AIAgentEvaluation/*" 
    }) {
    $file.packagePath = $file.packagePath -replace "tasks/AIAgentEvaluation", "tasks/AIAgentEvaluationDev"
}


# Write the modified vss-extension.json to vss-extension-dev.json
$vssExtension | ConvertTo-Json -Depth 10 | Set-Content -Path "$devExtensionDir/vss-extension.json" -Encoding UTF8
Write-Host "Successfully created AIAgentEvaluationDev from AIAgentEvaluation and updated vss-extension.json"

# Copy files defined in vss-extension.json using the utility function
$copySuccess = Copy-FilesFromVssExtension -SourceRootPath $repoRoot -DestinationRootPath $devExtensionDir -VssExtensionObject $vssExtension
if (-not $copySuccess) {
    Write-Error "Failed to copy files defined in vss-extension.json 'files' section or section not found"
    exit 1
}
Write-Host "Copied supporting files for extension" -ForegroundColor Green

$validate = Check-CriticalFiles -OutputDir $devExtensionDir -IsDevExtension $true
if (-not $validate) {
    Write-Error "Critical files check failed for production extension"
    exit 1
}
Write-Host "Critical files check passed for production extension" -ForegroundColor Green
