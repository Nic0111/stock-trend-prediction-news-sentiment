function Test-StockSenseExecutable {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [string[]]$Arguments = @("--version")
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    try {
        $process = Start-Process -FilePath $Path -ArgumentList $Arguments `
            -WindowStyle Hidden -PassThru -Wait
        return $process.ExitCode -eq 0
    } catch {
        return $false
    }
}

function Resolve-StockSensePython {
    param([Parameter(Mandatory = $true)][string]$ProjectRoot)
    $candidates = @(
        (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
        "C:\Users\nicho\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-StockSenseExecutable -Path $candidate) {
            return $candidate
        }
    }
    throw "No working StockSense Python runtime was found."
}

function Resolve-StockSenseNode {
    param([Parameter(Mandatory = $true)][string]$ProjectRoot)
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\nodejs\node.exe"),
        (Join-Path ${env:ProgramFiles} "nodejs\node.exe"),
        "C:\Users\nicho\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-StockSenseExecutable -Path $candidate) {
            return $candidate
        }
    }
    throw "No working StockSense Node.js runtime was found."
}
