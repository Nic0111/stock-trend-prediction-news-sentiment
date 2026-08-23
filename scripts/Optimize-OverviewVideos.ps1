param(
    [string]$InputDirectory = "frontend\public\media\overview-world\v2\video-1080-60",
    [string]$OutputDirectory = "frontend\public\media\overview-world\v2\video-1080-60-optimized",
    [int]$TimeoutMinutes = 10
)

$ErrorActionPreference = "Stop"
$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
$ffprobe = Get-Command ffprobe -ErrorAction SilentlyContinue
if (-not $ffmpeg -or -not $ffprobe) {
    throw "FFmpeg and FFprobe must be installed and available in PATH. No media files were changed."
}

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$source = [System.IO.Path]::GetFullPath((Join-Path $root $InputDirectory))
$target = [System.IO.Path]::GetFullPath((Join-Path $root $OutputDirectory))
if (-not $source.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase) -or
    -not $target.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Input and output directories must remain inside the StockSense workspace."
}
New-Item -ItemType Directory -Force -Path $target | Out-Null

$manifest = @()
foreach ($input in Get-ChildItem -LiteralPath $source -Filter "*.mp4" | Sort-Object Name) {
    $output = Join-Path $target $input.Name
    if (Test-Path -LiteralPath $output) {
        $existing = Get-Item -LiteralPath $output
        if ($existing.Length -gt 1MB -and $existing.LastWriteTimeUtc -ge $input.LastWriteTimeUtc) {
            Write-Host "Skipping completed output $($input.Name)"
        } else {
            Remove-Item -LiteralPath $output -Force
        }
    }
    if (-not (Test-Path -LiteralPath $output)) {
        $log = Join-Path $target ".$($input.BaseName).ffmpeg.log"
        $arguments = @(
            "-hide_banner", "-y", "-i", $input.FullName,
            "-an", "-c:v", "libx264", "-profile:v", "high", "-level:v", "4.2",
            "-preset", "slow", "-crf", "17", "-r", "60",
            "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", $output
        )
        $process = Start-Process -FilePath $ffmpeg.Source -ArgumentList $arguments `
            -WindowStyle Hidden -PassThru -RedirectStandardError $log
        try {
            Wait-Process -Id $process.Id -Timeout ($TimeoutMinutes * 60) -ErrorAction Stop
        } catch {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            Remove-Item -LiteralPath $output -Force -ErrorAction SilentlyContinue
            throw "Encoding $($input.Name) exceeded the $TimeoutMinutes minute safety limit."
        }
        if ($process.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $output)) {
            throw "Encoding $($input.Name) failed. Review $log."
        }
    }

    $probe = & $ffprobe.Source -v error -select_streams v:0 `
        -show_entries stream=width,height,avg_frame_rate,duration `
        -of json $output | ConvertFrom-Json
    $file = Get-Item -LiteralPath $output
    $stream = $probe.streams[0]
    $manifest += [ordered]@{
        file = $file.Name
        bytes = $file.Length
        width = [int]$stream.width
        height = [int]$stream.height
        frame_rate = [string]$stream.avg_frame_rate
        duration_seconds = [double]$stream.duration
        sha256 = (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$manifestPath = Join-Path $target "manifest.json"
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
Write-Host "Optimized media is staged at $target"
Write-Host "Review it visually before changing overviewJourney.ts to use the staged directory."
