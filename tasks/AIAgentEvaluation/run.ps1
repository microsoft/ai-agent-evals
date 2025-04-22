Trace-VstsEnteringInvocation $MyInvocation
try {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

    Write-Host "Checking Python installation..."
    . "$scriptDir\check-python.ps1"

    if (-not $?) {
        Write-Error "Python installation check failed. Cannot proceed."
        exit 1
    }
    
     # Check if requirements.txt exists and install dependencies
     $requirementsFile = Join-Path $scriptDir "requirements.txt"
     if (Test-Path $requirementsFile) {
         Write-Host "Installing Python dependencies from requirements.txt"
         python -m pip install --upgrade pip
         python -m pip install -r $requirementsFile
         
         if ($LASTEXITCODE -ne 0) {
             Write-Error "Failed to install Python dependencies"
             exit 1
         }
         Write-Host "Dependencies installed successfully"
     } else {
         Write-Host "No requirements.txt file found."
         exit 1
     }

     Write-Host "Reading task inputs..."
     $connectionString = Get-VstsInput -Name "azure-ai-project-connection-string" -Require
     $dataPath = Get-VstsInput -Name "data-path" -Require
     $agentIds = Get-VstsInput -Name "agent-ids" -Require
     $baselineAgentId = Get-VstsInput -Name "baseline-agent-id"
     
     # Set as environment variables for Python script
     $env:AZURE_AI_PROJECT_CONNECTION_STRING = $connectionString
     $env:DATA_PATH = $dataPath
     $env:AGENT_IDS = $agentIds
     $env:BASELINE_AGENT_ID = $baselineAgentId

       # Log inputs (mask sensitive information)
    Write-Host "Connection string: $connectionString"
    Write-Host "Data path: $dataPath"
    Write-Host "Agent IDs: $agentIds"
    Write-Host "Baseline agent ID: $baselineAgentId"    
    Write-Host "Executing action.py"

    $output = python "$scriptDir\action.py" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Python script failed with exit code $LASTEXITCODE"
        $output | ForEach-Object { Write-Error $_ }
        exit 1
    } else {
        Write-Host "Python script executed successfully"
        $output | ForEach-Object { Write-Host $_ }
    }
} finally {
    Trace-VstsLeavingInvocation $MyInvocation
}
