#!/usr/bin/env pwsh
# Script to create the AIAgentEvaluationDev task from AIAgentEvaluation
# and generate dist/dev/extension files for the dev version.

# Get the repository root directory regardless of where the script is invoked from
$scriptsFolder = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
. $scriptsFolder\set-variables.ps1

New-Item -Path $devExtensionDir -ItemType Directory -Force | Out-Null

# Import the utilities module with shared functions
$utilsPath = Join-Path -Path $scriptsFolder -ChildPath "utilities.ps1"
. $utilsPath

$setupPath = Join-Path -Path $scriptsFolder -ChildPath "setup.ps1"
& $setupPath

Copy-Directory -SourceDir $prodExtensionDir -DestinationDir $devExtensionDir
$vssExtensionPath = Join-Path $devExtensionDir "vss-extension.json"
# Read the original vss-extension.json
$vssExtension = Get-Content -Path $vssExtensionPath -Raw | ConvertFrom-Json

# Modify the vss-extension.json for the dev version
$vssExtension.id = "microsoft-extension-ai-agent-evaluation-dev"
$vssExtension.publisher = "ms-azure-exp-dev"
$vssExtension.name = "$($vssExtension.name) (Dev)"
$vssExtension.public = $false # Dev extension is not public

$buildResultsContribution = $vssExtension.contributions | Where-Object { $_.id -eq "build-results" }
if ($buildResultsContribution) {
    $buildResultsContribution.id = "build-results-dev"
    $buildResultsContribution.description = "$($buildResultsContribution.description) (Dev)"
    $buildResultsContribution.properties.name = "$($buildResultsContribution.properties.name) (Dev)"
    $buildResultsContribution.properties.supportsTasks = @($devTaskId)
}

# Update AIAgentEvaluation contribution
$agentEvalContribution = $vssExtension.contributions | Where-Object { $_.id -eq "AIAgentEvaluation" }
if ($agentEvalContribution) {
    $agentEvalContribution.id = "AIAgentEvaluationDev"
    $agentEvalContribution.properties.name = "tasks/AIAgentEvaluation"
}

$vssExtension | ConvertTo-Json -Depth 10 | Set-Content -Path "$devExtensionDir/vss-extension.json" -Encoding UTF8
$versions = @("V1", "V2")
foreach ($version in $versions) {
    $taskJsonPath = Join-Path $devExtensionDir "tasks/AIAgentEvaluation/$version/task.json"
    $taskJson = Get-Content -Path $taskJsonPath -Raw | ConvertFrom-Json
    $taskJson.id = $devTaskId
    $taskJson.name = "AIAgentEvaluationDev"
    $taskJson.friendlyName = "$($taskJson.friendlyName) (Dev)"
    $taskJson.description = "$($taskJson.description) (Dev)"
    $taskJson.instanceNameFormat = "$($taskJson.instanceNameFormat) (Dev)"
    $taskJson | ConvertTo-Json -Depth 10 | Set-Content -Path $taskJsonPath -Encoding UTF8
    Write-Host "Created modified task.json at: $taskJsonPath"
}

$validate = Check-CriticalFiles -OutputDir $devExtensionDir -IsDevExtension $true
if (-not $validate) {
    Write-Error "Critical files check failed for production extension"
    exit 1
}
Write-Host "Critical files check passed for production extension" -ForegroundColor Green
