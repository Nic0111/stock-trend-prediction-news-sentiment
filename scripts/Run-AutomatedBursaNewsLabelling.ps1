[CmdletBinding()]
param(
    [string]$Source = "",
    [string]$Output = "",
    [string]$Report = "",
    [string]$Checkpoint = "",
    [string]$Heartbeat = "",
    [string]$ApprovalQueue = "",
    [ValidateRange(2015, 2025)]
    [int]$RecommendedStartYear = 2015,
    [double]$MaxRuntimeMinutes = 45,
    [double]$RequestIntervalSeconds = 1,
    [int]$MaximumNoProgressPasses = 3,
    [int]$RetryDelaySeconds = 20
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "runtime_helpers.ps1")
$Python = Resolve-StockSensePython -ProjectRoot $ProjectRoot
$LabelRoot = Join-Path $ProjectRoot "data\historical-news-fusion\gdelt\external-corpus-2015-2025\external-labelling"

if (-not $Source) {
    $Source = Join-Path $LabelRoot "external-label-input.jsonl"
}
if (-not $Output) {
    $Output = Join-Path $LabelRoot "automated-weak-labels.jsonl"
}
if (-not $Report) {
    $Report = Join-Path $LabelRoot "automated-weak-labels-report.json"
}
if (-not $Checkpoint) {
    $Checkpoint = Join-Path $LabelRoot "automated-weak-labels-checkpoint.jsonl"
}
if (-not $Heartbeat) {
    $Heartbeat = Join-Path $LabelRoot "automated-weak-labels-heartbeat.json"
}
if (-not $ApprovalQueue) {
    $ApprovalQueue = Join-Path $LabelRoot "automated-weak-labels-approval-queue.jsonl"
}

foreach ($Path in @($Source, $Output, $Report, $Checkpoint, $Heartbeat, $ApprovalQueue)) {
    $ResolvedParent = [System.IO.Path]::GetFullPath((Split-Path -Parent $Path))
    if (-not $ResolvedParent.StartsWith(
        [System.IO.Path]::GetFullPath($ProjectRoot),
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Labelling paths must stay inside the StockSense workspace."
    }
}
if (-not (Test-Path -LiteralPath $Source)) {
    throw "Recovered labelling source does not exist: $Source"
}

function Get-CompletedRows {
    if (-not (Test-Path -LiteralPath $Report)) {
        return 0
    }
    try {
        $Value = Get-Content -LiteralPath $Report -Raw | ConvertFrom-Json
        return [int]$Value.completed_rows
    } catch {
        return 0
    }
}

$NoProgressPasses = 0
do {
    $Before = Get-CompletedRows
    & $Python -m pipeline.fusion_cli label-bursa-news-llm `
        --source $Source `
        --output $Output `
        --report $Report `
        --checkpoint $Checkpoint `
        --heartbeat $Heartbeat `
        --approval-queue $ApprovalQueue `
        --recommended-start-year ([string]$RecommendedStartYear) `
        --max-runtime-minutes ([string]$MaxRuntimeMinutes) `
        --request-interval-seconds ([string]$RequestIntervalSeconds) | Out-Host
    $ExitCode = $LASTEXITCODE
    $After = Get-CompletedRows

    if (Test-Path -LiteralPath $Report) {
        $Current = Get-Content -LiteralPath $Report -Raw | ConvertFrom-Json
        if ($null -ne $Current.stop_reason -and $Current.stop_reason -match "daily free-model quota") {
            Write-Warning "OpenRouter's daily free-model quota has been reached. The checkpoint is preserved; resume after the provider quota resets."
            exit 2
        }
        if (
            $null -eq $Current.stop_reason -and
            [int]$Current.completed_rows -ge [int]$Current.input_rows
        ) {
            Write-Host "Automated weak labelling completed: $($Current.completed_rows) rows."
            exit 0
        }
    }

    if ($After -le $Before) {
        $NoProgressPasses++
    } else {
        $NoProgressPasses = 0
    }
    if ($NoProgressPasses -ge $MaximumNoProgressPasses) {
        throw "Labelling made no progress in $NoProgressPasses consecutive resumable passes. Review $Heartbeat."
    }
    if ($ExitCode -notin @(0, 2)) {
        Write-Warning "Labelling exited with code $ExitCode; the checkpoint remains resumable."
    }
    Start-Sleep -Seconds ([Math]::Max(1, $RetryDelaySeconds))
} while ($true)
