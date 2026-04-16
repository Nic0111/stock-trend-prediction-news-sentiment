# 3.4.1.1 Missing Values Summary

## Purpose
This step validates the raw stock dataset columns, fills optional corporate action fields, and removes records that cannot be used because essential market values are missing.

## Method Applied
- Required stock columns are checked before processing.
- `Dividends` and `Stock Splits` are filled with `0.0` because missing values in these fields mean no recorded action for that trading day.
- Rows missing `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, or `Volume` are removed because these fields are required for stock prediction features.

## Overall Result
- Files processed: 10
- Original rows: 12,230
- Rows dropped due to missing essential values: 0
- Missing dividend values filled with zero: 0
- Missing stock split values filled with zero: 0
- Rows after missing-value handling: 12,230

## File-Level Statistics
| File | Original Rows | Rows Dropped | Dividends Filled | Stock Splits Filled | Final Rows |
| --- | --- | --- | --- | --- | --- |
| celcomdigi-berhad-raw.csv | 1,223 | 0 | 0 | 0 | 1,223 |
| cimb-group-holdings-berhad-raw.csv | 1,223 | 0 | 0 | 0 | 1,223 |
| ihh-healthcare-berhad-raw.csv | 1,223 | 0 | 0 | 0 | 1,223 |
| malayan-banking-berhad-raw.csv | 1,223 | 0 | 0 | 0 | 1,223 |
| misc-berhad-raw.csv | 1,223 | 0 | 0 | 0 | 1,223 |
| petronas-chemicals-group-berhad-raw.csv | 1,223 | 0 | 0 | 0 | 1,223 |
| press-metal-aluminium-holdings-berhad-raw.csv | 1,223 | 0 | 0 | 0 | 1,223 |
| public-bank-berhad-raw.csv | 1,223 | 0 | 0 | 0 | 1,223 |
| tenaga-nasional-berhad-raw.csv | 1,223 | 0 | 0 | 0 | 1,223 |
| ytl-corporation-berhad-raw.csv | 1,223 | 0 | 0 | 0 | 1,223 |
