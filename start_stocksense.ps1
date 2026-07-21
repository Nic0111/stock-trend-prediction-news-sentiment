$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$node = Join-Path $env:LOCALAPPDATA "Programs\nodejs\node.exe"
$vite = Join-Path $root "frontend\node_modules\vite\bin\vite.js"
$stateDirectory = Join-Path $root ".cache"
$stateFile = Join-Path $stateDirectory "stocksense-processes.json"

if (-not (Test-Path $python)) {
    throw "Python environment is missing: $python"
}
if (-not (Test-Path $node)) {
    throw "Node.js is missing: $node"
}
if (-not (Test-Path $vite)) {
    throw "Frontend dependencies are missing. Run 'pnpm install' in the frontend folder first."
}

$nodeRoot = Split-Path -Parent $node
$pnpmRoot = Join-Path $env:LOCALAPPDATA "pnpm"
$env:Path = "$nodeRoot;$pnpmRoot;$env:Path"

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

function Get-StockSenseBackendProcesses {
    $rootPattern = [regex]::Escape($root)
    @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -and
        $_.CommandLine -match "backend\.app\.main:app" -and
        $_.CommandLine -match $rootPattern
    })
}

$started = @()
$existingApi = @(Get-StockSenseBackendProcesses)
foreach ($process in $existingApi) {
    Stop-ProcessTree ([int]$process.ProcessId)
}
if (Test-PortListening 8000) {
    throw "Port 8000 is already occupied by a process that is not a StockSense backend. Stop that process or choose another API port."
}
$api = Start-Process -FilePath $python -ArgumentList @(
    "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000"
) -WorkingDirectory $root -WindowStyle Hidden -PassThru
$started += [pscustomobject]@{ Name = "api"; Pid = $api.Id }

if (-not (Test-PortListening 5173)) {
    $frontend = Start-Process -FilePath $node -ArgumentList @(
        $vite, "--host", "127.0.0.1"
    ) -WorkingDirectory (Join-Path $root "frontend") -WindowStyle Hidden -PassThru
    $started += [pscustomobject]@{ Name = "frontend"; Pid = $frontend.Id }
}

New-Item -ItemType Directory -Path $stateDirectory -Force | Out-Null
ConvertTo-Json -InputObject $started -Depth 3 | Set-Content -Path $stateFile -Encoding UTF8

$apiReady = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/health/ready" -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200 -and ($response.Content | ConvertFrom-Json).status -eq "ready") {
            $apiReady = $true
            break
        }
    } catch {
        Start-Sleep -Milliseconds 500
    }
}
if (-not $apiReady) {
    throw "The StockSense backend did not become ready within 15 seconds. Check the backend process or run the API command manually."
}

$ready = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:5173" -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
            $ready = $true
            break
        }
    } catch {
        Start-Sleep -Milliseconds 500
    }
}

if (-not $ready) {
    Write-Warning "The frontend did not respond within 15 seconds. Check the hidden process logs or run the two terminal commands manually."
}

Start-Process "http://localhost:5173"
Write-Host "StockSense is available at http://localhost:5173"
