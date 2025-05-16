$scriptsFolder = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
. $scriptsFolder\set-variables.ps1
. $scriptsFolder\utilities.ps1

New-Item -Path $tempDir -ItemType Directory -Force | Out-Null
Write-Host "Created temporary directory: $tempDir"

# Fetch the tags
$tagsResponse = Invoke-RestMethod -Uri $gitTagsApiUrl -Headers $gitHeaders

$hasV1 = $tagsResponse | Where-Object { $_.name -like "v1" } 

$tagToUseForV1 = 'v1';
if (-not $hasV1) {
    $tagToUseForV1 = 'v1-beta';
} 

Write-Host "Using tag: $tagToUseForV1 for V1"

Push-Location -Path $tempDir
try {
    # Clone the repository (shallow clone to save time/bandwidth)
    Write-Host "Cloning the $gitRepoOwner/$gitRepoName repository..."
    git clone --depth 1 --branch $tagToUseForV1 $gitRepoUrl "$gitRepoName-$tagToUseForV1"

    # Copy analysis and action.py to out/tasks/AIAgentEvaluation/V1
    $v1ClonedDir = Join-Path -Path $tempDir -ChildPath "$gitRepoName-$tagToUseForV1"
    $v1TaskDir = Join-Path -Path $prodExtensionDir -ChildPath "tasks/AIAgentEvaluation/V1"

    if (-not (Test-Path -Path $v1TaskDir)) {
        New-Item -Path $v1TaskDir -ItemType Directory -Force | Out-Null
        New-Item -Path "$v1TaskDir/analysis" -ItemType Directory -Force | Out-Null
        Write-Host "Created directory: $v1TaskDir"
    }
    Write-Host "Copying analysis and action.py to $v1TaskDir..."
    Copy-Directory -SourceDir "$v1ClonedDir/analysis" -DestinationDir "$v1TaskDir/analysis"
    Copy-Item -Path "$v1ClonedDir/action.py" -Destination "$v1TaskDir/action.py" -Force
    Copy-Item -Path "$v1ClonedDir/pyproject.toml" -Destination "$v1TaskDir/pyproject.toml" -Force
    Copy-Item -Path "$v1ClonedDir/tasks/AIAgentEvaluation/task.json" -Destination "$v1TaskDir/task.json" -Force
    Copy-Item -Path "$v1ClonedDir/tasks/AIAgentEvaluation/run.ps1" -Destination "$v1TaskDir/run.ps1" -Force
    Copy-Item -Path "$v1ClonedDir/tasks/AIAgentEvaluation/check-python.ps1" -Destination "$v1TaskDir/check-python.ps1" -Force
    Write-Host "Copied analysis, action.py, task.json, run.ps1, check-python.ps1 and pyproject.toml to $v1TaskDir"
}
finally {
    Pop-Location

    # Clean up the temporary directory
    if (Test-Path -Path $tempDir) {
        Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "Cleaned up temporary directory"
    }
}
