# 3.4.1.3 Data Transformation Summary

## Purpose
This step converts the cleaned data into a consistent dataset structure before saving the preprocessed stock dataset.

## Method Applied
- `Date` is converted to datetime and saved as `YYYY-MM-DD`.
- Numeric fields are converted to numeric values.
- Records are sorted in ascending date order.
- Duplicate dates inside each stock file are removed.
- The final column order is kept consistent across all stock datasets.

## Overall Result
- Files processed: 10
- Invalid date rows removed: 0
- Duplicate date rows removed: 0
- Rows after transformation: 12,230

## File-Level Statistics
| File | Invalid Dates Removed | Duplicate Dates Removed | First Date | Last Date | Final Rows |
| --- | --- | --- | --- | --- | --- |
| celcomdigi-berhad-raw.csv | 0 | 0 | 2021-01-04 | 2025-12-31 | 1,223 |
| cimb-group-holdings-berhad-raw.csv | 0 | 0 | 2021-01-04 | 2025-12-31 | 1,223 |
| ihh-healthcare-berhad-raw.csv | 0 | 0 | 2021-01-04 | 2025-12-31 | 1,223 |
| malayan-banking-berhad-raw.csv | 0 | 0 | 2021-01-04 | 2025-12-31 | 1,223 |
| misc-berhad-raw.csv | 0 | 0 | 2021-01-04 | 2025-12-31 | 1,223 |
| petronas-chemicals-group-berhad-raw.csv | 0 | 0 | 2021-01-04 | 2025-12-31 | 1,223 |
| press-metal-aluminium-holdings-berhad-raw.csv | 0 | 0 | 2021-01-04 | 2025-12-31 | 1,223 |
| public-bank-berhad-raw.csv | 0 | 0 | 2021-01-04 | 2025-12-31 | 1,223 |
| tenaga-nasional-berhad-raw.csv | 0 | 0 | 2021-01-04 | 2025-12-31 | 1,223 |
| ytl-corporation-berhad-raw.csv | 0 | 0 | 2021-01-04 | 2025-12-31 | 1,223 |

## Final Column Order
`Date, Ticker, Open, High, Low, Close, Adj Close, Volume, Dividends, Stock Splits`
