<#
.SYNOPSIS
    Checks if Python 3.10 or higher is installed and available in PATH.
.DESCRIPTION
    Verifies the installed Python version and provides guidance if an appropriate version isn't found.
#>

$minimumPythonMajor = 3
$minimumPythonMinor = 10

try {
    # Check if Python is available in the PATH
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    
    if (-not $pythonCommand) {
        throw "Python command not found in PATH"
    }
    
    $currentVersionOutput = & python --version 2>&1
    if ($currentVersionOutput -match "Python (\d+)\.(\d+)\.(\d+)") {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        $patch = [int]$matches[3]
        
        if ($major -gt $minimumPythonMajor -or ($major -eq $minimumPythonMajor -and $minor -ge $minimumPythonMinor)) {
            Write-Host "✅ Python version $major.$minor.$patch meets requirements (minimum $minimumPythonMajor.$minimumPythonMinor)" -ForegroundColor Green
            return $true
        } else {
            Write-Host "❌ Python version $major.$minor.$patch found, but version $minimumPythonMajor.$minimumPythonMinor or higher is required" -ForegroundColor Yellow
            throw "Insufficient Python version"
        }
    } else {
        throw "Unable to determine Python version"
    }
} catch {
    Write-Host "❌ Python version check failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Installation options:" -ForegroundColor Cyan
    Write-Host "1. Install Python $minimumPythonMajor.$minimumPythonMinor+ from https://www.python.org/downloads/" -ForegroundColor White
    Write-Host "2. In Azure DevOps pipeline, use:" -ForegroundColor White
    Write-Host "   ````yaml" -ForegroundColor Gray
    Write-Host "   - task: UsePythonVersion@0" -ForegroundColor Gray
    Write-Host "     inputs:" -ForegroundColor Gray
    Write-Host "       versionSpec: '3.10'" -ForegroundColor Gray
    Write-Host "   ````" -ForegroundColor Gray
    Write-Host ""
    Write-Host "After installation, ensure Python is in your PATH and try again."
    return $false
}