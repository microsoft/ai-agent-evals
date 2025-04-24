#!/usr/bin/env pwsh
# Script to create the AIAgentEvaluationDev task from AIAgentEvaluation
# and generate vss-extension-dev.json from vss-extension.json

# Get the repository root directory regardless of where the script is invoked from
$scriptPath = $MyInvocation.MyCommand.Path
$scriptsFolder = Split-Path -Path $scriptPath -Parent
$repoRoot = Split-Path -Path $scriptsFolder -Parent

# Import the utilities module with shared functions
$utilsPath = Join-Path -Path $scriptsFolder -ChildPath "utilities.ps1"
. $utilsPath

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
$taskJson.friendlyName = "AI Agent Evaluation (Dev)"
$taskJson.description = "Evaluate AI Agents (Development Version)"
$taskJson.instanceNameFormat = "AI Agent Evaluation (Dev)"

# Write the modified task.json to the destination
$destTaskJsonPath = Join-Path $destFolder "task.json"
$taskJson | ConvertTo-Json -Depth 10 | Set-Content -Path $destTaskJsonPath -Encoding UTF8

Write-Host "Created modified task.json at: $destTaskJsonPath"

# Now handle the vss-extension.json to vss-extension-dev.json conversion
$vssExtensionPath = Join-Path $repoRoot "vss-extension.json"
$vssExtensionDevPath = Join-Path $repoRoot "vss-extension-dev.json"

# Read the original vss-extension.json
$vssExtension = Get-Content -Path $vssExtensionPath -Raw | ConvertFrom-Json

# Modify the vss-extension.json for the dev version
$vssExtension.id = "microsoft-extension-ai-agent-evaluation-dev"
$vssExtension.publisher = "ms-azure-exp-dev"
$vssExtension.name = "Azure AI Evaluations Dev"

# Update the version using the shared function
$vssExtension.version = Update-VersionNumber -CurrentVersion $vssExtension.version

# Update contributions section
$buildResultsContribution = $vssExtension.contributions | Where-Object { $_.id -eq "build-results" }
if ($buildResultsContribution) {
    $buildResultsContribution.id = "build-results-dev"
    $buildResultsContribution.description = "Add AI Evaluation summary tab to the build results view (Dev)"
    $buildResultsContribution.properties.name = "Azure AI Evaluation (Dev)"
    $buildResultsContribution.properties.supportsTasks = @("6c8d5e8b-16f2-4f7b-b991-99e3dfa9f359")
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
$vssExtension | ConvertTo-Json -Depth 10 | Set-Content -Path $vssExtensionDevPath -Encoding UTF8

Write-Host "Created modified vss-extension-dev.json at: $vssExtensionDevPath"
Write-Host "Successfully created AIAgentEvaluationDev from AIAgentEvaluation and updated vss-extension-dev.json"