# 3.4.1.4 Preprocessed Dataset Summary

## Purpose
This section documents the dataset that has undergone preprocessing. The output is the cleaned stock dataset used as the source for model input-ready feature construction.

## Dataset Location
`data/preprocessed-stock-dataset/`

## Difference From Raw Dataset
- Raw data is downloaded directly from Yahoo Finance.
- The preprocessed dataset has validated columns, handled missing values, removed logically invalid rows, converted data types, removed duplicate dates, and standardized date formatting.
- The preprocessed dataset is not scaled. Scaling is discussed separately because it depends on the model training scenario.

## Overall Result
- Files processed: 10
- Raw rows: 12,230
- Preprocessed rows: 12,230
- Total rows removed through preprocessing: 0

## File-Level Dataset Output
| Raw File | Ticker | Raw Rows | Preprocessed Rows | Rows Removed | First Date | Last Date | Output File |
| --- | --- | --- | --- | --- | --- | --- | --- |
| celcomdigi-berhad-raw.csv | 6947.KL | 1,223 | 1,223 | 0 | 2021-01-04 | 2025-12-31 | data\preprocessed-stock-dataset\celcomdigi-berhad-preprocessed.csv |
| cimb-group-holdings-berhad-raw.csv | 1023.KL | 1,223 | 1,223 | 0 | 2021-01-04 | 2025-12-31 | data\preprocessed-stock-dataset\cimb-group-holdings-berhad-preprocessed.csv |
| ihh-healthcare-berhad-raw.csv | 5225.KL | 1,223 | 1,223 | 0 | 2021-01-04 | 2025-12-31 | data\preprocessed-stock-dataset\ihh-healthcare-berhad-preprocessed.csv |
| malayan-banking-berhad-raw.csv | 1155.KL | 1,223 | 1,223 | 0 | 2021-01-04 | 2025-12-31 | data\preprocessed-stock-dataset\malayan-banking-berhad-preprocessed.csv |
| misc-berhad-raw.csv | 3816.KL | 1,223 | 1,223 | 0 | 2021-01-04 | 2025-12-31 | data\preprocessed-stock-dataset\misc-berhad-preprocessed.csv |
| petronas-chemicals-group-berhad-raw.csv | 5183.KL | 1,223 | 1,223 | 0 | 2021-01-04 | 2025-12-31 | data\preprocessed-stock-dataset\petronas-chemicals-group-berhad-preprocessed.csv |
| press-metal-aluminium-holdings-berhad-raw.csv | 8869.KL | 1,223 | 1,223 | 0 | 2021-01-04 | 2025-12-31 | data\preprocessed-stock-dataset\press-metal-aluminium-holdings-berhad-preprocessed.csv |
| public-bank-berhad-raw.csv | 1295.KL | 1,223 | 1,223 | 0 | 2021-01-04 | 2025-12-31 | data\preprocessed-stock-dataset\public-bank-berhad-preprocessed.csv |
| tenaga-nasional-berhad-raw.csv | 5347.KL | 1,223 | 1,223 | 0 | 2021-01-04 | 2025-12-31 | data\preprocessed-stock-dataset\tenaga-nasional-berhad-preprocessed.csv |
| ytl-corporation-berhad-raw.csv | 4677.KL | 1,223 | 1,223 | 0 | 2021-01-04 | 2025-12-31 | data\preprocessed-stock-dataset\ytl-corporation-berhad-preprocessed.csv |

## Dataset Columns
`Date, Ticker, Open, High, Low, Close, Adj Close, Volume, Dividends, Stock Splits`
