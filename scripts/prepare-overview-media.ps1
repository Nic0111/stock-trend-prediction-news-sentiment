[CmdletBinding()]
param(
  [string]$InputRoot = "$(Join-Path $PSScriptRoot '..\frontend\public\media\overview-world\v2\video')",
  [string]$OutputRoot = "$(Join-Path $PSScriptRoot '..\frontend\public\media\overview-world\v2\video-1080')",
  [switch]$Force
)

$ErrorActionPreference = 'Stop'

function Resolve-Tool([string]$Name) {
  $command = Get-Command $Name -ErrorAction SilentlyContinue
  if ($command) { return $command.Source }
  $local = Join-Path $PSScriptRoot "..\tools\ffmpeg\bin\$Name.exe"
  if (Test-Path -LiteralPath $local) { return (Resolve-Path -LiteralPath $local).Path }
  throw "$Name was not found. Install FFmpeg or place the portable binaries under tools\ffmpeg\bin."
}

$ffmpeg = Resolve-Tool 'ffmpeg'
$ffprobe = Resolve-Tool 'ffprobe'
$inputPath = (Resolve-Path -LiteralPath $InputRoot).Path
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$manifest = @()
Get-ChildItem -LiteralPath $inputPath -Filter '*.mp4' -File | Sort-Object Name | ForEach-Object {
  $output = Join-Path $OutputRoot $_.Name
  if ($Force -or -not (Test-Path -LiteralPath $output)) {
    Write-Host "Encoding $($_.Name) -> 1920x1080"
    & $ffmpeg -hide_banner -loglevel error -y -i $_.FullName -an `
      -vf 'scale=1920:1080:flags=lanczos,unsharp=5:5:0.35:5:5:0.0' `
      -c:v libx264 -preset medium -crf 19 -pix_fmt yuv420p -r 24 `
      -g 8 -keyint_min 8 -sc_threshold 0 -movflags +faststart $output
    if ($LASTEXITCODE -ne 0) { throw "FFmpeg failed for $($_.Name)" }
  }

  $probeJson = & $ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames,codec_name,pix_fmt -show_entries format=duration -of json $output | ConvertFrom-Json
  $stream = $probeJson.streams[0]
  $manifest += [ordered]@{
    file = "/media/overview-world/v2/video-1080/$($_.Name)"
    width = [int]$stream.width
    height = [int]$stream.height
    frameRate = [string]$stream.r_frame_rate
    frameCount = $stream.nb_frames
    durationSeconds = [double]$probeJson.format.duration
    codec = [string]$stream.codec_name
    pixelFormat = [string]$stream.pix_fmt
  }
}

$manifestPath = Join-Path (Split-Path -Parent $OutputRoot) 'video-1080-manifest.json'
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $manifestPath -Encoding utf8
Write-Host "Wrote $manifestPath"
