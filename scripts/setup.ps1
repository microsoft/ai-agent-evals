# Run this script first to set up the environment for Azure DevOps Extension development.
# It installs the required ps_modules, sets up the task paths, and builds the front-end for AIAgentReport.
# It also updates the version in vss-extension.json and creates a production version of the file.

# Get the repository root directory regardless of where the script is invoked from
$scriptPath = $MyInvocation.MyCommand.Path
$scriptsFolder = Split-Path -Path $scriptPath -Parent
$repoRoot = Split-Path -Path $scriptsFolder -Parent

$prodExtensionDir = Join-Path -Path $repoRoot -ChildPath "dist/prod"
New-Item -Path $prodExtensionDir -ItemType Directory -Force | Out-Null

Push-Location -Path $repoRoot

try {
    Write-Host "Setting up VstsTaskSdk module..."

    # Use the download-vstsTaskSdk.ps1 
    $downloadScriptPath = Join-Path -Path $scriptsFolder -ChildPath "download-vstsTaskSdk.ps1"
    
    if (Test-Path -Path $downloadScriptPath) {
        Write-Host "Downloading VstsTaskSdk module from GitHub repository..."
        & $downloadScriptPath
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to download VstsTaskSdk module from GitHub. Exit code: $LASTEXITCODE"
            exit 1
        }
    } else {
        Write-Error "download-vstsTaskSdk.ps1 script not found at path: $downloadScriptPath"
        exit 1
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
    $vssExtensionProdPath = Join-Path -Path $prodExtensionDir -ChildPath "vss-extension.json"

    $vssExtension = Get-Content -Path $vssExtensionPath -Raw | ConvertFrom-Json
    $currentVersion = $vssExtension.version
    $vssExtension.version = Update-VersionNumber -CurrentVersion $currentVersion

    $vssExtension | ConvertTo-Json -Depth 10 | Set-Content -Path $vssExtensionProdPath
    Write-Host "Version updated successfully in vss-extension.json for prod" -ForegroundColor Green

    Copy-Item -Path "$repoRoot/tasks/AIAgentReport/dist" -Destination "$prodExtensionDir/tasks/AIAgentReport/dist" -Recurse -Force
    Copy-Item -Path "$repoRoot/tasks/AIAgentEvaluation" -Destination "$prodExtensionDir/tasks/AIAgentEvaluation" -Recurse -Force
    Copy-Item -Path "$repoRoot/logo.png" -Destination "$prodExtensionDir/logo.png" -Force
    Copy-Item -Path "$repoRoot/overview.md" -Destination "$prodExtensionDir/overview.md" -Force
    Copy-Item -Path "$repoRoot/LICENSE" -Destination "$prodExtensionDir/LICENSE" -Force
    Copy-Item -Path "$repoRoot/action.py" -Destination "$prodExtensionDir/action.py" -Force
    Copy-Item -Path "$repoRoot/pyproject.toml" -Destination "$prodExtensionDir/pyproject.toml" -Force
    Copy-Item -Path "$repoRoot/analysis" -Destination "$prodExtensionDir/analysis" -Recurse -Force

    Write-Host "Copied supporting files for extension" -ForegroundColor Green
} finally {
    # Return to the original directory
    Pop-Location
}