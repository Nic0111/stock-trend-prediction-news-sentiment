# 3.4.1.5 Data Standardization/Normalization Summary

## Purpose
This section explains standardization and normalization for the stock prediction system. The main preprocessed dataset is not scaled at this stage because scaling requirements depend on the model training scenario.

## Sample-Only Dataset Location
`data/sample-only/standardized-stock-dataset/`

These standardized CSV files are generated only to show what a standardized dataset would look like for documentation. They are not used as the production preprocessing output and are not used by the model input-ready dataset pipeline.

## Understanding Of Scaling
- Standardization converts a feature to a distribution centered around mean `0` with standard deviation `1`.
- Normalization commonly rescales values into a fixed range such as `0` to `1`.
- These methods change feature magnitude, but they do not create new stock market information.

## Sample-Only Method Applied
- Z-score standardization is applied to `Open`, `High`, `Low`, `Close`, `Adj Close`, and `Volume`.
- `Date`, `Ticker`, `Dividends`, and `Stock Splits` are preserved without scaling.
- The full row count and original column order are preserved for each sample-only standardized file.
- If a column has missing or zero standard deviation, the standardized sample column is set to `0.0` and marked in the status field.

## Why Scaling Is Not Applied To The Main Preprocessed Dataset
- The preprocessed dataset is intended to preserve interpretable stock prices, volume, dividends, and split values.
- Scaling before train/test splitting can leak future information because scaling parameters would be calculated using future dates.
- Scaling should be fitted on the training or development period only, then applied to validation/test data.

## When Scaling Is Needed
- Logistic Regression is sensitive to feature scale because it optimizes coefficients across features with different magnitudes.
- The current training pipeline handles this correctly by placing `StandardScaler` inside the Logistic Regression `Pipeline`.
- Random Forest and XGBoost are tree-based models and generally do not require standardization in the same way because they split based on feature thresholds.

## Project Decision
For this project stage, scaling is documented as model-dependent preprocessing knowledge. The main preprocessed dataset remains unscaled, and model-specific scaling is handled during training where required.

## Sample-Only Output Files
| Output File                                                                                               |
| --------------------------------------------------------------------------------------------------------- |
| data\sample-only\standardized-stock-dataset\celcomdigi-berhad-standardized-sample.csv                     |
| data\sample-only\standardized-stock-dataset\cimb-group-holdings-berhad-standardized-sample.csv            |
| data\sample-only\standardized-stock-dataset\ihh-healthcare-berhad-standardized-sample.csv                 |
| data\sample-only\standardized-stock-dataset\malayan-banking-berhad-standardized-sample.csv                |
| data\sample-only\standardized-stock-dataset\misc-berhad-standardized-sample.csv                           |
| data\sample-only\standardized-stock-dataset\petronas-chemicals-group-berhad-standardized-sample.csv       |
| data\sample-only\standardized-stock-dataset\press-metal-aluminium-holdings-berhad-standardized-sample.csv |
| data\sample-only\standardized-stock-dataset\public-bank-berhad-standardized-sample.csv                    |
| data\sample-only\standardized-stock-dataset\tenaga-nasional-berhad-standardized-sample.csv                |
| data\sample-only\standardized-stock-dataset\ytl-corporation-berhad-standardized-sample.csv                |

## Standardization Statistics
| Raw File                                      | Column    | Rows  | Original Mean     | Original Std      | Standardized Mean | Standardized Std | Status       |
| --------------------------------------------- | --------- | ----- | ----------------- | ----------------- | ----------------- | ---------------- | ------------ |
| celcomdigi-berhad-raw.csv                     | Open      | 1,223 | 3.926639          | 0.333053          | 0                 | 1                | standardized |
| celcomdigi-berhad-raw.csv                     | High      | 1,223 | 3.96776           | 0.335381          | -0                | 1                | standardized |
| celcomdigi-berhad-raw.csv                     | Low       | 1,223 | 3.883271          | 0.331366          | 0                 | 1                | standardized |
| celcomdigi-berhad-raw.csv                     | Close     | 1,223 | 3.928324          | 0.337077          | -0                | 1                | standardized |
| celcomdigi-berhad-raw.csv                     | Adj Close | 1,223 | 3.535425          | 0.284831          | 0                 | 1                | standardized |
| celcomdigi-berhad-raw.csv                     | Volume    | 1,223 | 3,244,357.31807   | 3,730,957.1972    | -0                | 1                | standardized |
| cimb-group-holdings-berhad-raw.csv            | Open      | 1,223 | 5.994432          | 1.205478          | -0                | 1                | standardized |
| cimb-group-holdings-berhad-raw.csv            | High      | 1,223 | 6.045822          | 1.212892          | -0                | 1                | standardized |
| cimb-group-holdings-berhad-raw.csv            | Low       | 1,223 | 5.94583           | 1.197513          | 0                 | 1                | standardized |
| cimb-group-holdings-berhad-raw.csv            | Close     | 1,223 | 5.998561          | 1.204624          | 0                 | 1                | standardized |
| cimb-group-holdings-berhad-raw.csv            | Adj Close | 1,223 | 5.113727          | 1.42596           | -0                | 1                | standardized |
| cimb-group-holdings-berhad-raw.csv            | Volume    | 1,223 | 15,635,143.499591 | 10,861,169.525103 | -0                | 1                | standardized |
| ihh-healthcare-berhad-raw.csv                 | Open      | 1,223 | 6.37798           | 0.708775          | 0                 | 1                | standardized |
| ihh-healthcare-berhad-raw.csv                 | High      | 1,223 | 6.433679          | 0.714598          | 0                 | 1                | standardized |
| ihh-healthcare-berhad-raw.csv                 | Low       | 1,223 | 6.319141          | 0.697133          | -0                | 1                | standardized |
| ihh-healthcare-berhad-raw.csv                 | Close     | 1,223 | 6.376533          | 0.702726          | 0                 | 1                | standardized |
| ihh-healthcare-berhad-raw.csv                 | Adj Close | 1,223 | 6.084098          | 0.797603          | 0                 | 1                | standardized |
| ihh-healthcare-berhad-raw.csv                 | Volume    | 1,223 | 5,352,502.044154  | 4,231,211.031271  | 0                 | 1                | standardized |
| malayan-banking-berhad-raw.csv                | Open      | 1,223 | 9.145936          | 0.788511          | -0                | 1                | standardized |
| malayan-banking-berhad-raw.csv                | High      | 1,223 | 9.190294          | 0.792721          | -0                | 1                | standardized |
| malayan-banking-berhad-raw.csv                | Low       | 1,223 | 9.103352          | 0.783452          | 0                 | 1                | standardized |
| malayan-banking-berhad-raw.csv                | Close     | 1,223 | 9.145707          | 0.790752          | 0                 | 1                | standardized |
| malayan-banking-berhad-raw.csv                | Adj Close | 1,223 | 7.705212          | 1.36588           | 0                 | 1                | standardized |
| malayan-banking-berhad-raw.csv                | Volume    | 1,223 | 9,370,074.897792  | 6,068,160.935186  | 0                 | 1                | standardized |
| misc-berhad-raw.csv                           | Open      | 1,223 | 7.324742          | 0.469746          | 0                 | 1                | standardized |
| misc-berhad-raw.csv                           | High      | 1,223 | 7.386312          | 0.467815          | 0                 | 1                | standardized |
| misc-berhad-raw.csv                           | Low       | 1,223 | 7.257555          | 0.464757          | 0                 | 1                | standardized |
| misc-berhad-raw.csv                           | Close     | 1,223 | 7.321529          | 0.464127          | 0                 | 1                | standardized |
| misc-berhad-raw.csv                           | Adj Close | 1,223 | 6.424105          | 0.756738          | 0                 | 1                | standardized |
| misc-berhad-raw.csv                           | Volume    | 1,223 | 2,292,419.296811  | 1,912,548.090489  | 0                 | 1                | standardized |
| petronas-chemicals-group-berhad-raw.csv       | Open      | 1,223 | 6.84704           | 1.918535          | 0                 | 1                | standardized |
| petronas-chemicals-group-berhad-raw.csv       | High      | 1,223 | 6.925159          | 1.934368          | 0                 | 1                | standardized |
| petronas-chemicals-group-berhad-raw.csv       | Low       | 1,223 | 6.763974          | 1.9101            | -0                | 1                | standardized |
| petronas-chemicals-group-berhad-raw.csv       | Close     | 1,223 | 6.840409          | 1.91802           | -0                | 1                | standardized |
| petronas-chemicals-group-berhad-raw.csv       | Adj Close | 1,223 | 6.257621          | 1.565255          | 0                 | 1                | standardized |
| petronas-chemicals-group-berhad-raw.csv       | Volume    | 1,223 | 5,445,750.940311  | 4,428,024.961256  | 0                 | 1                | standardized |
| press-metal-aluminium-holdings-berhad-raw.csv | Open      | 1,223 | 5.160192          | 0.606193          | 0                 | 1                | standardized |
| press-metal-aluminium-holdings-berhad-raw.csv | High      | 1,223 | 5.221611          | 0.611985          | 0                 | 1                | standardized |
| press-metal-aluminium-holdings-berhad-raw.csv | Low       | 1,223 | 5.092719          | 0.597742          | 0                 | 1                | standardized |
| press-metal-aluminium-holdings-berhad-raw.csv | Close     | 1,223 | 5.161124          | 0.609247          | 0                 | 1                | standardized |
| press-metal-aluminium-holdings-berhad-raw.csv | Adj Close | 1,223 | 4.974952          | 0.608877          | -0                | 1                | standardized |
| press-metal-aluminium-holdings-berhad-raw.csv | Volume    | 1,223 | 7,089,375.306623  | 5,369,006.54274   | -0                | 1                | standardized |
| public-bank-berhad-raw.csv                    | Open      | 1,223 | 4.280746          | 0.203518          | 0                 | 1                | standardized |
| public-bank-berhad-raw.csv                    | High      | 1,223 | 4.309859          | 0.20789           | -0                | 1                | standardized |
| public-bank-berhad-raw.csv                    | Low       | 1,223 | 4.251462          | 0.200445          | -0                | 1                | standardized |
| public-bank-berhad-raw.csv                    | Close     | 1,223 | 4.280548          | 0.205868          | -0                | 1                | standardized |
| public-bank-berhad-raw.csv                    | Adj Close | 1,223 | 3.756113          | 0.325379          | -0                | 1                | standardized |
| public-bank-berhad-raw.csv                    | Volume    | 1,223 | 18,968,194.848733 | 14,734,879.283914 | 0                 | 1                | standardized |
| tenaga-nasional-berhad-raw.csv                | Open      | 1,223 | 10.984072         | 2.096186          | 0                 | 1                | standardized |
| tenaga-nasional-berhad-raw.csv                | High      | 1,223 | 11.065372         | 2.119841          | 0                 | 1                | standardized |
| tenaga-nasional-berhad-raw.csv                | Low       | 1,223 | 10.908512         | 2.074096          | -0                | 1                | standardized |
| tenaga-nasional-berhad-raw.csv                | Close     | 1,223 | 10.990818         | 2.096307          | -0                | 1                | standardized |
| tenaga-nasional-berhad-raw.csv                | Adj Close | 1,223 | 9.826688          | 2.436782          | -0                | 1                | standardized |
| tenaga-nasional-berhad-raw.csv                | Volume    | 1,223 | 5,607,093.376942  | 4,570,007.264386  | -0                | 1                | standardized |
| ytl-corporation-berhad-raw.csv                | Open      | 1,223 | 1.465376          | 0.953373          | -0                | 1                | standardized |
| ytl-corporation-berhad-raw.csv                | High      | 1,223 | 1.4911            | 0.971496          | 0                 | 1                | standardized |
| ytl-corporation-berhad-raw.csv                | Low       | 1,223 | 1.43879           | 0.932957          | 0                 | 1                | standardized |
| ytl-corporation-berhad-raw.csv                | Close     | 1,223 | 1.464828          | 0.952149          | 0                 | 1                | standardized |
| ytl-corporation-berhad-raw.csv                | Adj Close | 1,223 | 1.395448          | 0.946536          | 0                 | 1                | standardized |
| ytl-corporation-berhad-raw.csv                | Volume    | 1,223 | 16,328,576.369583 | 20,085,422.0329   | 0                 | 1                | standardized |
