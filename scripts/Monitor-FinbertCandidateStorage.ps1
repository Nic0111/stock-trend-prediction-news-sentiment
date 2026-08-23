[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RunDirectory,

    [Parameter(Mandatory = $true)]
    [string]$ProcessIds,

    [double]$MinimumFreeGiB = 20.0,

    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
$resolvedRunDirectory = (Resolve-Path -LiteralPath $RunDirectory).Path
$markerPath = Join-Path $resolvedRunDirectory "storage-safety-stop.json"
$logPath = Join-Path $resolvedRunDirectory "storage-monitor.log"

function Get-FreeBytes {
    try {
        $drive = [System.IO.DriveInfo]::new("C:\")
        return [int64]$drive.AvailableFreeSpace
    }
    catch {
        return [int64]0
    }
}

function Write-MonitorLog {
    param([string]$Message)
    $line = "{0:u} {1}" -f (Get-Date), $Message
    Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
}

if ($MinimumFreeGiB -le 0 -or $PollSeconds -lt 10) {
    throw "MinimumFreeGiB must be positive and PollSeconds must be at least 10."
}

$processIdList = @(
    $ProcessIds -split '[,;\s]+' |
        Where-Object { $_ -ne '' } |
        ForEach-Object {
            try { [int]$_ }
            catch { throw "ProcessIds must contain only integer process IDs: '$($_)'." }
        }
)
if ($processIdList.Count -eq 0) {
    throw "At least one process ID is required."
}

Write-MonitorLog ("Started for PIDs {0}; stop threshold {1:N2} GiB." -f ($processIdList -join ","), $MinimumFreeGiB)

while ($true) {
    $freeBytes = Get-FreeBytes
    $freeGiB = $freeBytes / 1GB
    $tracked = @(
        $processIdList |
            ForEach-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }
    )

    if ($tracked.Count -eq 0) {
        Write-MonitorLog "Candidate processes are no longer running; monitor finished."
        break
    }

    if ($freeGiB -lt $MinimumFreeGiB) {
        $stopped = @($tracked | Select-Object -ExpandProperty Id)
        $tracked | Stop-Process -Force -ErrorAction SilentlyContinue
        $payload = [ordered]@{
            stopped_at = (Get-Date).ToUniversalTime().ToString("o")
            reason = "C drive free space fell below the configured safety threshold"
            free_gib = [math]::Round($freeGiB, 3)
            threshold_gib = $MinimumFreeGiB
            stopped_process_ids = $stopped
            production_changed = $false
            resume_with_same_run_id = $true
        }
        $payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $markerPath -Encoding UTF8
        Write-MonitorLog ("Stopped candidate PIDs {0}; free space {1:N2} GiB." -f ($stopped -join ","), $freeGiB)
        break
    }

    Write-MonitorLog ("Healthy: free space {0:N2} GiB; tracked PIDs {1}." -f $freeGiB, ($tracked.Id -join ","))
    Start-Sleep -Seconds $PollSeconds
}
