param(
    [string]$RootUrl = "http://127.0.0.1:5173",
    [string]$Ticker = "4677.KL",
    [ValidateSet(1, 5, 10)][int]$Horizon = 5,
    [switch]$IncludeExternalCalls
)

$ErrorActionPreference = "Stop"
$apiRoot = "$($RootUrl.TrimEnd('/'))/api/v1"
$results = [System.Collections.Generic.List[object]]::new()
$analysis = $null

function Add-ConnectionCheck {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )

    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $detail = & $Action
        $timer.Stop()
        $script:results.Add([pscustomobject]@{
            Check = $Name
            Status = "PASS"
            Seconds = [math]::Round($timer.Elapsed.TotalSeconds, 2)
            Detail = [string]$detail
        })
    } catch {
        $timer.Stop()
        $detail = $_.Exception.Message
        if ($_.ErrorDetails.Message) {
            $detail = $_.ErrorDetails.Message
        }
        $script:results.Add([pscustomobject]@{
            Check = $Name
            Status = "FAIL"
            Seconds = [math]::Round($timer.Elapsed.TotalSeconds, 2)
            Detail = [string]$detail
        })
    }
}

Add-ConnectionCheck "Frontend" {
    $response = Invoke-WebRequest -Uri "$($RootUrl.TrimEnd('/'))/" -UseBasicParsing -TimeoutSec 10
    if ($response.StatusCode -ne 200 -or $response.Content -notmatch "<title>StockSense</title>") {
        throw "Port 5173 did not return the StockSense page."
    }
    "HTTP 200"
}

Add-ConnectionCheck "Backend readiness through Vite" {
    $response = Invoke-RestMethod -Uri "$apiRoot/health/ready" -TimeoutSec 30
    if ($response.status -ne "ready") {
        throw ($response | ConvertTo-Json -Compress -Depth 8)
    }
    "models 1/5/10, FinBERT and artifacts ready"
}

Add-ConnectionCheck "System configuration" {
    $response = Invoke-RestMethod -Uri "$apiRoot/system/status" -TimeoutSec 30
    $missing = @()
    if (-not $response.newsapi_configured) { $missing += "NewsAPI" }
    if (-not $response.marketaux_configured) { $missing += "Marketaux" }
    if (-not $response.openrouter_configured) { $missing += "OpenRouter" }
    if ($missing.Count -gt 0) {
        throw "Missing configuration: $($missing -join ', ')"
    }
    "NewsAPI, Marketaux and OpenRouter configured"
}

Add-ConnectionCheck "Company catalogue" {
    $response = Invoke-RestMethod -Uri "$apiRoot/companies" -TimeoutSec 30
    if ($response.Count -ne 10) {
        throw "Expected 10 companies but received $($response.Count)."
    }
    "10 companies"
}

Add-ConnectionCheck "Market data" {
    $response = Invoke-RestMethod -Uri "$apiRoot/market-snapshots/$Ticker" -TimeoutSec 90
    "ticker=$($response.ticker)"
}

foreach ($predictionHorizon in @(1, 5, 10)) {
    Add-ConnectionCheck "Stock prediction ${predictionHorizon}D" {
        $response = Invoke-RestMethod -Uri "$apiRoot/predictions/$Ticker`?horizon=$predictionHorizon" -TimeoutSec 120
        "ticker=$($response.ticker)"
    }
}

Add-ConnectionCheck "Stock Outlook analysis" {
    $body = @{ ticker = $Ticker; horizon = $Horizon } | ConvertTo-Json
    $script:analysis = Invoke-RestMethod -Uri "$apiRoot/analyses" -Method Post `
        -ContentType "application/json" -Body $body -TimeoutSec 180
    "analysis=$($script:analysis.analysis_id), mode=$($script:analysis.combined_outlook.mode)"
}

Add-ConnectionCheck "Ticker comparison" {
    $body = @{ tickers = @("1155.KL", "1295.KL"); horizon = 1 } | ConvertTo-Json
    $response = Invoke-RestMethod -Uri "$apiRoot/comparisons" -Method Post `
        -ContentType "application/json" -Body $body -TimeoutSec 180
    if (@($response.items).Count -ne 2) {
        throw "Comparison did not return two companies."
    }
    "2 companies"
}

if ($IncludeExternalCalls) {
    Add-ConnectionCheck "News retrieval and FinBERT" {
        $body = @{ ticker = $Ticker; horizon = $Horizon } | ConvertTo-Json
        $response = Invoke-RestMethod -Uri "$apiRoot/news" -Method Post `
            -ContentType "application/json" -Body $body -TimeoutSec 180
        $providerStatus = $response.provider_status | ConvertTo-Json -Compress
        "relevant=$($response.relevant_article_count), providers=$providerStatus"
    }

    Add-ConnectionCheck "OpenRouter tutor" {
        $body = @{
            ticker = $Ticker
            horizon = $Horizon
            intent = "general_tutor"
            messages = @(@{
                role = "user"
                content = "In one short paragraph, explain what a stock prediction probability means."
            })
        } | ConvertTo-Json -Depth 6
        $response = Invoke-RestMethod -Uri "$apiRoot/chat" -Method Post `
            -ContentType "application/json" -Body $body -TimeoutSec 120
        if (-not $response.direct_answer) {
            throw "The tutor returned no answer."
        }
        "model=$($response.model)"
    }
}

$results | Format-Table -AutoSize -Wrap
if ($results.Status -contains "FAIL") {
    exit 1
}
