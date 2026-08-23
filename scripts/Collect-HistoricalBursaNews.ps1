[CmdletBinding()]
param(
    [string]$ProjectId = "project-e479489e-567d-4f30-bc6",
    [string]$CollectionRoot = "",
    [string]$Years = "2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025",
    [switch]$ExecuteCandidates,
    [switch]$Recover,
    [switch]$RecoveryOnly,
    [switch]$SingleRecoveryPass,
    [double]$MaximumBytesBilledGiB = 20,
    [double]$MonthlyBudgetGiB = 900,
    [double]$MaxRuntimeMinutes = 45,
    [double]$MaxDownloadGiB = 5,
    [double]$RequestIntervalSeconds = 1,
    [int]$BatchSize = 100,
    [ValidateRange(1, 1000000)]
    [int]$RecoveryCandidatesPerTicker = 60,
    [switch]$RecoverAllCandidates
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "runtime_helpers.ps1")
if (-not $CollectionRoot) {
    $CollectionRoot = Join-Path $ProjectRoot "data\historical-news-fusion\gdelt\external-corpus-2015-2025"
}
$CollectionRoot = [System.IO.Path]::GetFullPath($CollectionRoot)
if (-not $CollectionRoot.StartsWith([System.IO.Path]::GetFullPath($ProjectRoot), [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "CollectionRoot must stay inside the StockSense workspace."
}

$Python = Resolve-StockSensePython -ProjectRoot $ProjectRoot

$env:PYTHONPATH = "$(Join-Path $ProjectRoot '.venv\Lib\site-packages');$ProjectRoot"
$ProjectId = $ProjectId.Trim()
if (-not $ProjectId) {
    throw "ProjectId is required."
}
$SelectedYears = @(
    $Years -split "," |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ } |
        ForEach-Object { [int]$_ } |
        Select-Object -Unique
)
if ($SelectedYears.Count -eq 0) {
    throw "At least one year must be supplied through -Years."
}
$InvalidYears = @($SelectedYears | Where-Object { $_ -lt 2015 -or $_ -gt 2025 })
if ($InvalidYears.Count -gt 0) {
    throw "Only years from 2015 through 2025 are supported: $($InvalidYears -join ', ')."
}
if ($RecoveryOnly -and -not $Recover) {
    throw "-RecoveryOnly requires -Recover."
}
if ($RecoveryOnly -and $ExecuteCandidates) {
    throw "-RecoveryOnly cannot be combined with -ExecuteCandidates."
}
$EffectiveRecoveryCandidatesPerTicker = if ($RecoverAllCandidates) {
    1000000
} else {
    $RecoveryCandidatesPerTicker
}

function Invoke-StockSensePython {
    param([string[]]$Arguments)
    # Send the child's JSON/progress output to the console rather than the
    # function pipeline. Otherwise assigning this function's result captures
    # both every printed line and the numeric exit code, making a successful
    # command compare unequal to zero.
    & $Python @Arguments | Out-Host
    $ExitCode = $LASTEXITCODE
    return [int]$ExitCode
}

function Get-RecoveryProgress {
    param([string]$OutputRoot)
    $Raw = Join-Path $OutputRoot "raw"
    $Count = 0
    foreach ($Name in @("publisher-state.jsonl", "ngram-state.jsonl", "archive-state.jsonl")) {
        $Path = Join-Path $Raw $Name
        if (Test-Path -LiteralPath $Path) {
            $Count += (Get-Content -LiteralPath $Path | Measure-Object -Line).Lines
        }
    }
    return $Count
}

Write-Host "StockSense historical-news collection (2015-2025)"
if ($RecoveryOnly) {
    Write-Host "Step 1/3 skipped in recovery-only mode; no BigQuery request was made."
} else {
    Write-Host "Step 1/3: generating queries and checking the complete free-only budget..."
    $PlanPath = Join-Path $CollectionRoot "external-corpus-plan.json"
    $PreviousPlanPath = Join-Path $CollectionRoot "external-corpus-plan.previous.json"
    if (Test-Path -LiteralPath $PlanPath) {
        Move-Item -LiteralPath $PlanPath -Destination $PreviousPlanPath -Force
    }
    $PlanArguments = @(
        "-m", "pipeline.fusion_cli", "gdelt-external-corpus",
        "--project-id", $ProjectId,
        "--output-dir", $CollectionRoot,
        "--reuse-2023-export", (Join-Path $ProjectRoot "data\historical-news-fusion\gdelt\pilot-2023-high-recall-v3b\exports"),
        "--maximum-bytes-billed-gib", [string]$MaximumBytesBilledGiB,
        "--monthly-budget-gib", [string]$MonthlyBudgetGiB
    )
    foreach ($Year in $SelectedYears) {
        $PlanArguments += @("--year", [string]$Year)
    }
    $PlanExit = Invoke-StockSensePython -Arguments $PlanArguments
    if (-not (Test-Path -LiteralPath $PlanPath)) {
        throw "The BigQuery dry-run did not create $PlanPath."
    }
    $Plan = Get-Content -LiteralPath $PlanPath -Raw | ConvertFrom-Json
    $PlanYears = @($Plan.years | ForEach-Object { [int]$_ } | Sort-Object)
    $ExpectedYears = @($SelectedYears | Sort-Object)
    if (($PlanYears -join ",") -ne ($ExpectedYears -join ",")) {
        throw "The BigQuery dry-run plan is incomplete. Expected years $($ExpectedYears -join ', '), but received $($PlanYears -join ', ')."
    }
    if ([string]$Plan.project_id -ne $ProjectId) {
        throw "The BigQuery dry-run plan belongs to a different project."
    }
    if (-not $Plan.ready_for_execution) {
        throw "The full collection exceeds the configured free-only ceiling. No query was executed. Review $PlanPath."
    }
    if ($PlanExit -ne 0) {
        throw "The BigQuery dry-run failed even though a plan file was produced."
    }

    if ($ExecuteCandidates) {
        Write-Host "Step 1/3: executing resumable candidate queries..."
        $ExecuteExit = Invoke-StockSensePython -Arguments ($PlanArguments + @("--execute"))
        if ($ExecuteExit -ne 0) {
            throw "Candidate collection paused or failed. Re-run this same command to resume from query-state.jsonl."
        }
    } else {
        Write-Host "Dry-run complete. Add -ExecuteCandidates to retrieve candidates."
    }
}

if ($Recover) {
    if (-not $ExecuteCandidates) {
        Write-Host "Recovery requested without candidate execution; only already completed exports will be used."
    }
    Write-Host "Step 2/3: resuming publisher, Web-NGram and archive recovery..."
    if ($RecoverAllCandidates) {
        Write-Warning "Full high-recall recovery was explicitly requested. This can include many unverified mentions and take a long time."
    } else {
        Write-Host "Recovery uses a balanced cap of $EffectiveRecoveryCandidatesPerTicker candidates per ticker/year. Full discovery exports remain preserved for later batches."
    }
    foreach ($Year in $SelectedYears) {
        if ($Year -eq 2023) {
            $CandidateDir = Join-Path $ProjectRoot "data\historical-news-fusion\gdelt\pilot-2023-high-recall-v3b\exports"
            $RecoveryDir = Join-Path $ProjectRoot "data\historical-news-fusion\gdelt\pilot-2023-corpus-recovery"
        } else {
            $CandidateDir = Join-Path $CollectionRoot "$Year\exports"
            $RecoveryDir = Join-Path $CollectionRoot "$Year\recovery"
        }
        $CandidateFiles = @(
            Get-ChildItem -LiteralPath $CandidateDir -Filter "$Year-??.jsonl" -File -ErrorAction SilentlyContinue
        )
        if ($CandidateFiles.Count -lt 12) {
            Write-Warning "$Year has only $($CandidateFiles.Count) completed monthly exports; skipping recovery until collection resumes."
            continue
        }

        $NoProgressPasses = 0
        do {
            $Before = Get-RecoveryProgress -OutputRoot $RecoveryDir
            Write-Host "Recovering $Year from checkpoint ($Before completed stage records)..."
            $RecoveryArguments = @(
                "-m", "pipeline.fusion_cli", "gdelt-ngram-pilot",
                "--year", [string]$Year,
                "--candidate-dir", $CandidateDir,
                "--sample-per-ticker", [string]$EffectiveRecoveryCandidatesPerTicker,
                "--paired-per-ticker", "0",
                "--output-dir", $RecoveryDir,
                "--max-runtime-minutes", [string]$MaxRuntimeMinutes,
                "--max-download-gib", [string]$MaxDownloadGiB,
                "--request-interval-seconds", [string]$RequestIntervalSeconds,
                "--skip-labels"
            )
            $RecoveryExit = Invoke-StockSensePython -Arguments $RecoveryArguments
            $After = Get-RecoveryProgress -OutputRoot $RecoveryDir
            if ($After -le $Before) {
                $NoProgressPasses++
            } else {
                $NoProgressPasses = 0
            }

            $ReportPath = Join-Path $RecoveryDir "pilot-$Year-ngram-usability-report.json"
            $Complete = $false
            if (Test-Path -LiteralPath $ReportPath) {
                $Report = Get-Content -LiteralPath $ReportPath -Raw | ConvertFrom-Json
                $Complete = [bool]$Report.completed
            }
            if ($Complete) {
                Write-Host "$Year recovery completed."
                break
            }
            if ($NoProgressPasses -ge 2) {
                Write-Warning "$Year made no checkpoint progress in two bounded passes. Stopping it instead of looping indefinitely."
                break
            }
            if ($SingleRecoveryPass) {
                Write-Host "$Year paused after one bounded pass."
                break
            }
            if ($RecoveryExit -notin @(0, 2)) {
                Write-Warning "$Year recovery exited with code $RecoveryExit."
                break
            }
        } while ($true)
    }
} else {
    Write-Host "Step 2/3 skipped. Add -Recover after candidate exports are ready."
}

Write-Host "Step 3/3: building the current external-labelling package..."
$LabelDir = Join-Path $CollectionRoot "external-labelling"
$Extra2023 = Join-Path $ProjectRoot "data\historical-news-fusion\gdelt\pilot-2023-corpus-recovery\raw\hybrid-recovered.jsonl"
$Cached2023 = Join-Path $ProjectRoot "data\historical-news-fusion\gdelt\pilot-2023-corpus-recovery\external-labelling\recovered-articles.jsonl"
$ExportArguments = @(
    "-m", "pipeline.fusion_cli", "gdelt-external-label-export",
    "--recovery-root", $CollectionRoot,
    "--output-dir", $LabelDir,
    "--batch-size", [string]$BatchSize
)
if (Test-Path -LiteralPath $Extra2023) {
    $ExportArguments += @("--extra-file", $Extra2023)
}
if (Test-Path -LiteralPath $Cached2023) {
    $ExportArguments += @("--extra-file", $Cached2023)
}
$ExportExit = Invoke-StockSensePython -Arguments $ExportArguments
if ($ExportExit -ne 0) {
    throw "The external-label package could not be generated."
}
Write-Host "Package ready at $LabelDir"
