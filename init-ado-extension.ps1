# download ps_modules
Install-Module -Name VstsTaskSdk -AllowClobber

# Define task paths
$taskPaths = @(
    "tasks/AIAgentEvaluation/ps_modules/VstsTaskSdk"
)

# Get the module contents 
$moduleSource = (Get-Module VstsTaskSdk -ListAvailable).ModuleBase

# Copy to each task folder
foreach ($modulePath in $taskPaths) {
    # Create the destination directory structure if needed
    if (-not (Test-Path -Path $modulePath)) {
        New-Item -Path $modulePath -ItemType Directory -Force | Out-Null
    }
    
    # Copy module files directly to the destination folder
    Copy-Item -Path "$moduleSource/*" -Destination $modulePath -Recurse -Force
    
    Write-Host "VstsTaskSdk module copied to $modulePath"
}

