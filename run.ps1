$ErrorActionPreference = 'Stop'

if ($args.Count -gt 0 -and $args[0] -eq 'build') {
    $arguments = @($args | Select-Object -Skip 1)
    $script = Join-Path $PSScriptRoot 'dataset\build.py'
    $localPython = Join-Path $PSScriptRoot 'dataset\.venv\Scripts\python.exe'
    $workspacePython = Join-Path (Split-Path $PSScriptRoot -Parent) 'icde\.venv\Scripts\python.exe'
} else {
    $script = Join-Path $PSScriptRoot 'run.py'
    $arguments = @($args)
    $localPython = Join-Path $PSScriptRoot 'experiments\.venv\Scripts\python.exe'
    $workspacePython = $null
}

if (Test-Path -LiteralPath $localPython) {
    & $localPython -B -u $script @arguments
} elseif ($workspacePython -and (Test-Path -LiteralPath $workspacePython)) {
    & $workspacePython -B -u $script @arguments
} else {
    & python -B -u $script @arguments
}
exit $LASTEXITCODE
