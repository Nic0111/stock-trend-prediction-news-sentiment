# 3.4.1.6 Model Input-Ready Dataset Summary

## Purpose
This section documents the dataset construction step that converts the preprocessed stock dataset into model input-ready rows. This is feature engineering and target construction, not standardization or normalization.

## Dataset Location
`data/model-input-stock-dataset/`

## Method Applied
- Daily and multi-day return features are created from `Adj Close`.
- Lag, moving-average, volatility, price movement, and volume features are created.
- Classification targets are created for 1-day, 5-day, and 10-day future price direction.
- Rows with missing or infinite values from lag, rolling-window, volume-change, or future-target construction are removed.

## Overall Result
- Files processed: 10
- Preprocessed input rows: 12,230
- Rows dropped during model input preparation: 703
- Model input-ready rows: 11,527

## File-Level Statistics
| Preprocessed File                                      | Input Rows | Rows Dropped | Final Rows |
| ------------------------------------------------------ | ---------- | ------------ | ---------- |
| celcomdigi-berhad-preprocessed.csv                     | 1,223      | 75           | 1,148      |
| cimb-group-holdings-berhad-preprocessed.csv            | 1,223      | 69           | 1,154      |
| ihh-healthcare-berhad-preprocessed.csv                 | 1,223      | 69           | 1,154      |
| malayan-banking-berhad-preprocessed.csv                | 1,223      | 71           | 1,152      |
| misc-berhad-preprocessed.csv                           | 1,223      | 69           | 1,154      |
| petronas-chemicals-group-berhad-preprocessed.csv       | 1,223      | 69           | 1,154      |
| press-metal-aluminium-holdings-berhad-preprocessed.csv | 1,223      | 69           | 1,154      |
| public-bank-berhad-preprocessed.csv                    | 1,223      | 72           | 1,151      |
| tenaga-nasional-berhad-preprocessed.csv                | 1,223      | 69           | 1,154      |
| ytl-corporation-berhad-preprocessed.csv                | 1,223      | 71           | 1,152      |

## Feature Columns
- `Open`
- `High`
- `Low`
- `Close`
- `Adj Close`
- `Volume`
- `Return_1D`
- `Return_5D`
- `Lag_1`
- `Lag_2`
- `Lag_3`
- `Lag_5`
- `MA_5`
- `MA_20`
- `MA_60`
- `Volatility_5D`
- `Volatility_20D`
- `High_Low_Spread`
- `Open_Close_Change`
- `Volume_Change_1D`
- `Volume_MA_5`

## Target Columns
- `Target_1D`
- `Target_5D`
- `Target_10D`

## Relationship To Standardization/Normalization
This stage prepares the columns required by the prediction models. Scaling remains model-dependent and is handled during training when needed, such as the `StandardScaler` used inside the Logistic Regression training pipeline.
