@echo off
setlocal
cd /d "C:\stock-investment-analytics"
"C:\stock-investment-analytics\.venv\Scripts\python.exe" -u -m pipeline.fusion_cli bursa-news-append recover --work-dir "C:\stock-investment-analytics\data\historical-news-fusion\gdelt\bursa-news-balance-v2" --max-runtime-minutes 45 --max-download-gib 5 --request-interval-seconds 1 --skip-archives >> "C:\stock-investment-analytics\data\historical-news-fusion\gdelt\bursa-news-balance-v2\recovery-worker-scheduled.log" 2>> "C:\stock-investment-analytics\data\historical-news-fusion\gdelt\bursa-news-balance-v2\recovery-worker-scheduled.err.log"
exit /b %errorlevel%
