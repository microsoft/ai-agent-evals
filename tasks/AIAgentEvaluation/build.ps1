# setup.ps1 - Build and prepare dist folder with production-only dependencies

# Ensure script is running from directory containing package.json
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $projectRoot

$versions = @("V1", "V2");
$latestVersion = "V2"

Copy-Item -Path "$projectRoot\..\..\pyproject.toml" -Destination "$projectRoot\$latestVersion" -Force
Copy-Item -Path "$projectRoot\..\..\action.py" -Destination "$projectRoot\$latestVersion" -Force
Copy-Item -Path "$projectRoot\..\..\analysis\" -Destination "$projectRoot\$latestVersion" -Force -Recurse
Write-Host "Copied additional files to $latestVersion"

# Run the build
Write-Host "Running TypeScript build..."
foreach ($version in $versions) {
    $versionPath = Join-Path $projectRoot $version
    Push-Location $versionPath
    npm ci
    npm run build
    npm prune --production
    Pop-Location
}

Write-Host "✅ Setup complete. 'AIAgentEvaluation' folder is ready."
