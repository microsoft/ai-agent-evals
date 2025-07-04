$scriptPath = $MyInvocation.MyCommand.Path
$scriptsFolder = Split-Path -Path $scriptPath -Parent
$repoRoot = Split-Path -Path $scriptsFolder -Parent

$devExtensionDir = Join-Path -Path $repoRoot -ChildPath "out/dev"
$prodExtensionDir = Join-Path -Path $repoRoot -ChildPath "out/prod"

$utilsPath = Join-Path -Path $repoRoot -ChildPath "scripts/utilities.ps1"

$devTaskId = "6c8d5e8b-16f2-4f7b-b991-99e3dfa9f359"
