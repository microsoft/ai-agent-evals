# Run this script first to set up the environment for Azure DevOps Extension development.
# It installs the required ps_modules, sets up the task paths, and builds the front-end for AIAgentReport.
# It also updates the version in vss-extension.json and creates a production version of the file.

# Get the repository root directory regardless of where the script is invoked from
$scriptPath = $MyInvocation.MyCommand.Path
$scriptsFolder = Split-Path -Path $scriptPath -Parent
$repoRoot = Split-Path -Path $scriptsFolder -Parent

Push-Location -Path $repoRoot

try {
    Write-Host "Installing VstsTaskSdk module..."
    Install-Module -Name VstsTaskSdk -Force -AllowClobber

    $taskPaths = @(
        "tasks/AIAgentEvaluation/ps_modules/VstsTaskSdk"
    )

    Write-Host "Getting VstsTaskSdk module source path..."
    $moduleSource = (Get-Module VstsTaskSdk -ListAvailable).ModuleBase

    foreach ($modulePath in $taskPaths) {
        $fullModulePath = Join-Path -Path $repoRoot -ChildPath $modulePath
        
        if (-not (Test-Path -Path $fullModulePath)) {
            New-Item -Path $fullModulePath -ItemType Directory -Force | Out-Null
        }
        
        Copy-Item -Path "$moduleSource/*" -Destination $fullModulePath -Recurse -Force
        
        Write-Host "VstsTaskSdk module copied to $modulePath"
    }


    Write-Host "Building AIAgentReport web UI..."
    $reportPath = Join-Path -Path $repoRoot -ChildPath "tasks/AIAgentReport"
    Push-Location -Path $reportPath
    try {
        npm ci
        
        npm run build
        
        Write-Host "AIAgentReport build completed successfully" -ForegroundColor Green
    } catch {
        Write-Error "Error building AIAgentReport: $_"
        exit 1
    } finally {
        # Always return to the previous directory even if there are errors
        Pop-Location
    }

    # Import the utilities module with shared functions
    $utilsPath = Join-Path -Path $repoRoot -ChildPath "scripts/utilities.ps1"
    . $utilsPath

    # Update version in vss-extension.json
    Write-Host "Updating version in vss-extension.json..."
    $vssExtensionPath = Join-Path -Path $repoRoot -ChildPath "vss-extension.json"
    $vssExtensionProdPath = Join-Path -Path $repoRoot -ChildPath "vss-extension-prod.json"

    # Read the vss-extension.json file
    $vssExtension = Get-Content -Path $vssExtensionPath -Raw | ConvertFrom-Json

    # Get the current version
    $currentVersion = $vssExtension.version

    # Update the version using the shared function
    $vssExtension.version = Update-VersionNumber -CurrentVersion $currentVersion

    # Write the updated content back to the file
    $vssExtension | ConvertTo-Json -Depth 10 | Set-Content -Path $vssExtensionProdPath

    Write-Host "Version updated successfully in vss-extension.json" -ForegroundColor Green
} finally {
    # Return to the original directory
    Pop-Location
}