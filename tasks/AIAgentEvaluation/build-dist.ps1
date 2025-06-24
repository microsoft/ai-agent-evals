# setup.ps1 - Build and prepare dist folder with production-only dependencies

# Ensure script is running from directory containing package.json
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $projectRoot

# Clean previous dist folder
$distPath = Join-Path $projectRoot "dist"
if (Test-Path $distPath) {
    Remove-Item $distPath -Recurse -Force
}

$versions = @("V1", "V2");
$latestVersion = "V2"



# Create dist directory
New-Item -ItemType Directory -Force -Path $distPath | Out-Null

foreach ($version in $versions) {
    Copy-Item -Path "$projectRoot\$version\" -Destination "$distPath\$version\" -Force -Recurse
}

Copy-Item -Path "$projectRoot\..\..\pyproject.toml" -Destination "$distPath\$latestVersion" -Force
Copy-Item -Path "$projectRoot\..\..\action.py" -Destination "$distPath\$latestVersion" -Force
Copy-Item -Path "$projectRoot\..\..\analysis\" -Destination "$distPath\$latestVersion" -Force -Recurse
Write-Host "Copied additional files to dist"

# Run the build
Write-Host "Running TypeScript build..."
foreach ($version in $versions) {
    $versionPath = Join-Path $distPath $version
    Push-Location $versionPath
    npm ci
    npm run build
    npm prune --production
    Pop-Location
}

Write-Host "✅ Setup complete. 'dist' folder is ready."
