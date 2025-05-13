Trace-VstsEnteringInvocation $MyInvocation
try {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

    Write-Host "Checking Python installation..."
    . "$scriptDir\check-python.ps1"

    if (-not $?) {
        Write-Error "Python installation check failed. Cannot proceed."
        exit 1
    }

    Write-Host "Installing Python dependencies..."
    python -m pip install --upgrade pip
    python -m pip install .

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install Python dependencies"
        exit 1
    }
    Write-Host "Dependencies installed successfully"

    Write-Host "Reading task inputs..."
    $endpoint = Get-VstsInput -Name "azure-ai-project-endpoint" -Require
    $deploymentName = Get-VstsInput -Name "deployment-name" -Require
    $apiVersion = Get-VstsInput -Name "api-version"
    $dataPath = Get-VstsInput -Name "data-path" -Require
    $agentIds = Get-VstsInput -Name "agent-ids" -Require
    $baselineAgentId = Get-VstsInput -Name "baseline-agent-id"
    $evaluationResultView = Get-VstsInput -Name "evaluation-result-view"

    # Set as environment variables for Python script
    $env:AZURE_AI_PROJECT_ENDPOINT = $endpoint
    $env:DEPLOYMENT_NAME = $deploymentName
    $env:API_VERSION = $apiVersion
    $env:DATA_PATH = $dataPath
    $env:AGENT_IDS = $agentIds
    $env:BASELINE_AGENT_ID = $baselineAgentId
    $env:EVALUATION_RESULT_VIEW = $evaluationResultView

    # Log inputs (mask sensitive information)
    Write-Host "Endpoint: $endpoint"
    Write-Host "Data path: $dataPath"
    Write-Host "Agent IDs: $agentIds"
    Write-Host "Baseline agent ID: $baselineAgentId"
    Write-Host "deployment name: $deploymentName"
    Write-Host "API version: $apiVersion"
    Write-Host "Evaluation result view: $evaluationResultView"
    Write-Host "Executing action.py"

    $artifactFolder = "ai-agent-eval"
    $artifactFile = "ai-agent-eval-summary.md"

    $outputPath = if ($env:BUILD_ARTIFACTSTAGINGDIRECTORY) { $env:BUILD_ARTIFACTSTAGINGDIRECTORY } else { "." }
    $reportPath = Join-Path -Path $outputPath -ChildPath $artifactFile

    New-Item -Path $reportPath -ItemType "file" -Force | Out-Null
    Write-Host "Report file created at $reportPath"
    $env:ADO_STEP_SUMMARY = $reportPath

    python "$scriptDir\action.py"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Python script failed with exit code $LASTEXITCODE"
        exit 1
    } else {
        Write-Host "Python script executed successfully"
        Write-Host "##vso[artifact.upload artifactname=$artifactFolder]$reportPath"
    }
} catch {
    Write-Error "An error occurred: $($_.Exception.Message)"
    exit 1
} finally {
    Trace-VstsLeavingInvocation $MyInvocation
}
