$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$stateFile = Join-Path $root ".cache\stocksense-processes.json"

function Stop-ProcessTree([int]$processId) {
    $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$processId" -ErrorAction SilentlyContinue)
    foreach ($child in $children) {
        Stop-ProcessTree ([int]$child.ProcessId)
    }
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
}

if (Test-Path $stateFile) {
    $entries = @(Get-Content -Raw $stateFile | ConvertFrom-Json)
    foreach ($entry in $entries) {
        foreach ($processId in @($entry.Pid)) {
            Stop-ProcessTree ([int]$processId)
        }
    }
    Remove-Item -LiteralPath $stateFile -Force -ErrorAction SilentlyContinue
}

Write-Host "StockSense processes started by the shortcut have been stopped."
