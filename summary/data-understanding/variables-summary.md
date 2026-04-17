# 3.4.2.1 Variables Summary

## Purpose
This section documents the variables available in the preprocessed stock dataset and describes their basic data characteristics.

## Variable Dictionary
| Variable     | Data Type      | Meaning                                            | Missing Count | Unique Count | Minimum    | Maximum    |
| ------------ | -------------- | -------------------------------------------------- | ------------- | ------------ | ---------- | ---------- |
| Date         | datetime64[us] | Trading date for the daily stock record.           | 0             | 1,223        | 2021-01-04 | 2025-12-31 |
| Ticker       | str            | Stock ticker symbol from Yahoo Finance.            | 0             | 10           | N/A        | N/A        |
| Open         | float64        | Opening price for the trading day.                 | 0             | 1,127        | 0.51       | 15.04      |
| High         | float64        | Highest traded price for the trading day.          | 0             | 1,123        | 0.515      | 15.24      |
| Low          | float64        | Lowest traded price for the trading day.           | 0             | 1,135        | 0.5        | 14.84      |
| Close        | float64        | Closing price for the trading day.                 | 0             | 1,130        | 0.505      | 15.04      |
| Adj Close    | float64        | Adjusted closing price after dividends and splits. | 0             | 5,873        | 0.447784   | 14.336076  |
| Volume       | int64          | Number of shares traded for the day.               | 0             | 11,791       | 0          | 383975900  |
| Dividends    | float64        | Dividend amount recorded for the trading day.      | 0             | 57           | 0          | 0.58       |
| Stock Splits | float64        | Stock split ratio recorded for the trading day.    | 0             | 3            | 0          | 5          |
