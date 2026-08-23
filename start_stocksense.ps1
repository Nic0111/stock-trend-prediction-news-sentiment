$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $root "scripts\runtime_helpers.ps1")
$python = Resolve-StockSensePython -ProjectRoot $root
$node = Resolve-StockSenseNode -ProjectRoot $root
$vite = Join-Path $root "frontend\node_modules\vite\bin\vite.js"
$stateDirectory = Join-Path $root ".cache"
$stateFile = Join-Path $stateDirectory "stocksense-processes.json"
$logDirectory = Join-Path $stateDirectory "launcher-logs"
$apiStdout = Join-Path $logDirectory "backend.stdout.log"
$apiStderr = Join-Path $logDirectory "backend.stderr.log"
$frontendStdout = Join-Path $logDirectory "frontend.stdout.log"
$frontendStderr = Join-Path $logDirectory "frontend.stderr.log"

if (-not (Test-Path $vite)) {
    throw "Frontend dependencies are missing. Run 'pnpm install' in the frontend folder first."
}

$nodeRoot = Split-Path -Parent $node
$pnpmRoot = Join-Path $env:LOCALAPPDATA "pnpm"
$processPath = [Environment]::GetEnvironmentVariable("Path", "Process")
# Some Windows shells expose both Path and PATH. Normalize the process
# environment before any child-process API is called; otherwise PowerShell's
# Start-Process can fail while converting the duplicate keys.
$env:PATH = $null
$env:Path = "$nodeRoot;$pnpmRoot;$processPath"

function Test-PortListening([int]$port) {
    return $null -ne (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
}

function Stop-ProcessTree([int]$processId) {
    $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$processId" -ErrorAction SilentlyContinue)
    foreach ($child in $children) {
        Stop-ProcessTree ([int]$child.ProcessId)
    }
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
}

function Get-LogTail([string[]]$paths, [int]$lineCount = 18) {
    $sections = @()
    foreach ($path in $paths) {
        if (Test-Path $path) {
            $tail = @(Get-Content -LiteralPath $path -Tail $lineCount -ErrorAction SilentlyContinue)
            if ($tail.Count -gt 0) {
                $sections += "`n--- $(Split-Path -Leaf $path) ---"
                $sections += $tail
            }
        }
    }
    return ($sections -join [Environment]::NewLine)
}

function Start-LoggedProcess {
    param(
        [string]$Executable,
        [string[]]$Arguments,
        [string]$WorkingDirectory,
        [string]$StdoutPath,
        [string]$StderrPath
    )

    # The process environment is normalized above, so Start-Process can launch
    # the services directly with hidden windows. Using cmd.exe here would open
    # one visible console for the API and another for Vite.
    return Start-Process -FilePath $Executable -ArgumentList $Arguments `
        -WorkingDirectory $WorkingDirectory -WindowStyle Hidden `
        -RedirectStandardOutput $StdoutPath -RedirectStandardError $StderrPath -PassThru
}

function Wait-HttpReady {
    param(
        [string]$Name,
        [string]$Uri,
        [System.Diagnostics.Process]$Process,
        [int]$TimeoutSeconds,
        [scriptblock]$ValidateResponse,
        [string[]]$LogPaths
    )

    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    while ($watch.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        if ($Process -and $Process.HasExited) {
            $logs = Get-LogTail $LogPaths
            throw "$Name exited before becoming ready (exit code $($Process.ExitCode)).$logs"
        }

        try {
            $response = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 2
            if (& $ValidateResponse $response) {
                return
            }
        } catch {
            # A refused connection is expected during the short startup window.
        }

        Start-Sleep -Milliseconds 400
    }

    $logs = Get-LogTail $LogPaths
    throw "$Name did not become ready within $TimeoutSeconds seconds. The startup watchdog stopped waiting.$logs"
}

function Get-StockSenseBackendProcesses {
    $rootPattern = [regex]::Escape($root)
    @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -and
        $_.CommandLine -match "backend\.app\.main:app" -and
        $_.CommandLine -match $rootPattern
    })
}

$started = @()
$api = $null
$frontend = $null
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
foreach ($logPath in @($apiStdout, $apiStderr, $frontendStdout, $frontendStderr)) {
    Remove-Item -LiteralPath $logPath -Force -ErrorAction SilentlyContinue
}

$existingApi = @(Get-StockSenseBackendProcesses)
foreach ($process in $existingApi) {
    Stop-ProcessTree ([int]$process.ProcessId)
}
if (Test-PortListening 8000) {
    throw "Port 8000 is already occupied by a process that is not a StockSense backend. Stop that process or choose another API port."
}
$api = Start-LoggedProcess -Executable $python -Arguments @(
    "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000"
) -WorkingDirectory $root -StdoutPath $apiStdout -StderrPath $apiStderr
$started += [pscustomobject]@{ Name = "api"; Pid = $api.Id }

if (-not (Test-PortListening 5173)) {
    # Vite's CLI installs an interactive stdin handler when it detects a TTY.
    # The hidden launcher intentionally has no console/stdin, and Node 24 can
    # surface that mismatch as an unhandled EPIPE.  CI mode disables the
    # shortcut/readline handler while keeping the dev server unchanged.
    $previousCi = $env:CI
    $env:CI = "true"
    try {
        $frontend = Start-LoggedProcess -Executable $node -Arguments @(
            $vite, "--host", "127.0.0.1"
        ) -WorkingDirectory (Join-Path $root "frontend") -StdoutPath $frontendStdout -StderrPath $frontendStderr
    } finally {
        $env:CI = $previousCi
    }
    $started += [pscustomobject]@{ Name = "frontend"; Pid = $frontend.Id }
}

New-Item -ItemType Directory -Path $stateDirectory -Force | Out-Null
ConvertTo-Json -InputObject $started -Depth 3 | Set-Content -Path $stateFile -Encoding UTF8

try {
    Wait-HttpReady `
        -Name "StockSense backend" `
        -Uri "http://127.0.0.1:8000/api/v1/health/ready" `
        -Process $api `
        -TimeoutSeconds 30 `
        -LogPaths @($apiStdout, $apiStderr) `
        -ValidateResponse {
            param($response)
            if ($response.StatusCode -ne 200) {
                return $false
            }
            try {
                return ($response.Content | ConvertFrom-Json).status -eq "ready"
            } catch {
                return $false
            }
        }

    if ($frontend) {
        Wait-HttpReady `
            -Name "StockSense frontend" `
            -Uri "http://127.0.0.1:5173" `
            -Process $frontend `
            -TimeoutSeconds 20 `
            -LogPaths @($frontendStdout, $frontendStderr) `
            -ValidateResponse {
                param($response)
                return $response.StatusCode -ge 200 -and $response.StatusCode -lt 500
            }
    } elseif (-not (Test-PortListening 5173)) {
        throw "The frontend was not started and port 5173 is not listening."
    }

    Start-Process "http://localhost:5173"
    Write-Host "StockSense is available at http://localhost:5173"
} catch {
    foreach ($entry in $started) {
        Stop-ProcessTree ([int]$entry.Pid)
    }
    Remove-Item -LiteralPath $stateFile -Force -ErrorAction SilentlyContinue
    Write-Error @"
$($_.Exception.Message)

The launcher stopped the processes it started so nothing keeps running invisibly.
Diagnostic logs: $logDirectory
"@
    exit 1
} finally {
    if ($api) {
        $api.Dispose()
    }
    if ($frontend) {
        $frontend.Dispose()
    }
}
