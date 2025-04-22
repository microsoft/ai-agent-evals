@{
    RootModule = 'VstsTaskSdk.psm1'
    ModuleVersion = '0.21.0' # Do not modify. This value gets replaced at build time with the value from the package.json.
    GUID = 'bbed04e2-4e8e-4089-90a2-58b858fed8d8'
    Author = 'Microsoft Corporation'
    CompanyName = 'Microsoft Corporation'
    Copyright = '(c) Microsoft Corporation. All rights reserved.'
    Description = 'VSTS Task SDK'
    PowerShellVersion = '3.0'
    FunctionsToExport = '*'
    CmdletsToExport = ''
    VariablesToExport = '*'
    AliasesToExport = ''
    PrivateData = @{
        PSData = @{
            ProjectUri = 'https://github.com/Microsoft/azure-pipelines-task-lib'
            CommitHash = '9abb170036d3b4cc1781def2efa436d9b059104c' # Do not modify. This value gets replaced at build time.
        }
    }
    HelpInfoURI = 'https://github.com/Microsoft/azure-pipelines-task-lib'
    DefaultCommandPrefix = 'Vsts'
}
