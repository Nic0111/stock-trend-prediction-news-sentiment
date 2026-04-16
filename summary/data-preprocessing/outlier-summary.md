# 3.4.1.2 Outlier Summary

## Purpose
This step removes records that are logically impossible for OHLCV stock data and documents statistical outliers for interpretation.

## Method Applied
- Logical invalid rows are removed because they break stock price rules.
- IQR outliers are detected for `Open`, `High`, `Low`, `Close`, `Adj Close`, and `Volume`.
- Statistical outliers are kept in the dataset because sharp market movement and unusual volume may carry predictive information.

## Overall Result
- Files processed: 10
- Logically invalid rows removed: 0
- Rows flagged as statistical outliers but kept: 853
- Rows after outlier handling: 12,230

## File-Level Result
| File | Logical Rows Removed | Statistical Rows Flagged | Final Rows |
| --- | --- | --- | --- |
| celcomdigi-berhad-raw.csv | 0 | 89 | 1,223 |
| cimb-group-holdings-berhad-raw.csv | 0 | 57 | 1,223 |
| ihh-healthcare-berhad-raw.csv | 0 | 119 | 1,223 |
| malayan-banking-berhad-raw.csv | 0 | 49 | 1,223 |
| misc-berhad-raw.csv | 0 | 130 | 1,223 |
| petronas-chemicals-group-berhad-raw.csv | 0 | 65 | 1,223 |
| press-metal-aluminium-holdings-berhad-raw.csv | 0 | 141 | 1,223 |
| public-bank-berhad-raw.csv | 0 | 65 | 1,223 |
| tenaga-nasional-berhad-raw.csv | 0 | 63 | 1,223 |
| ytl-corporation-berhad-raw.csv | 0 | 75 | 1,223 |

## IQR Outlier Details
| File | Column | IQR Lower Bound | IQR Upper Bound | Outlier Count |
| --- | --- | --- | --- | --- |
| celcomdigi-berhad-raw.csv | Open | 2.8175 | 5.0775 | 0 |
| celcomdigi-berhad-raw.csv | High | 2.845 | 5.125 | 0 |
| celcomdigi-berhad-raw.csv | Low | 2.7725 | 5.0325 | 0 |
| celcomdigi-berhad-raw.csv | Close | 2.805 | 5.085 | 0 |
| celcomdigi-berhad-raw.csv | Adj Close | 2.73609 | 4.376227 | 1 |
| celcomdigi-berhad-raw.csv | Volume | -1,327,350 | 6,566,650 | 88 |
| cimb-group-holdings-berhad-raw.csv | Open | 2.42 | 9.62 | 0 |
| cimb-group-holdings-berhad-raw.csv | High | 2.470001 | 9.669999 | 0 |
| cimb-group-holdings-berhad-raw.csv | Low | 2.41 | 9.53 | 0 |
| cimb-group-holdings-berhad-raw.csv | Close | 2.4475 | 9.5875 | 0 |
| cimb-group-holdings-berhad-raw.csv | Adj Close | 0.511234 | 9.919589 | 0 |
| cimb-group-holdings-berhad-raw.csv | Volume | -7,799,475 | 35,709,525 | 57 |
| ihh-healthcare-berhad-raw.csv | Open | 4.68 | 7.96 | 61 |
| ihh-healthcare-berhad-raw.csv | High | 4.71 | 8.07 | 62 |
| ihh-healthcare-berhad-raw.csv | Low | 4.630001 | 7.909999 | 62 |
| ihh-healthcare-berhad-raw.csv | Close | 4.6875 | 7.9475 | 63 |
| ihh-healthcare-berhad-raw.csv | Adj Close | 4.089394 | 8.061899 | 53 |
| ihh-healthcare-berhad-raw.csv | Volume | -2,565,350 | 11,821,850 | 70 |
| malayan-banking-berhad-raw.csv | Open | 6.575 | 11.895 | 0 |
| malayan-banking-berhad-raw.csv | High | 6.590002 | 11.949998 | 0 |
| malayan-banking-berhad-raw.csv | Low | 6.510002 | 11.869998 | 0 |
| malayan-banking-berhad-raw.csv | Close | 6.550002 | 11.909998 | 0 |
| malayan-banking-berhad-raw.csv | Adj Close | 3.107012 | 12.767818 | 0 |
| malayan-banking-berhad-raw.csv | Volume | -3,095,525 | 20,321,875 | 49 |
| misc-berhad-raw.csv | Open | 6.2575 | 8.3175 | 76 |
| misc-berhad-raw.csv | High | 6.32 | 8.4 | 77 |
| misc-berhad-raw.csv | Low | 6.204999 | 8.245 | 79 |
| misc-berhad-raw.csv | Close | 6.25 | 8.33 | 78 |
| misc-berhad-raw.csv | Adj Close | 4.019785 | 8.888901 | 0 |
| misc-berhad-raw.csv | Volume | -1,656,500 | 5,571,900 | 72 |
| petronas-chemicals-group-berhad-raw.csv | Open | 1.274999 | 12.595001 | 0 |
| petronas-chemicals-group-berhad-raw.csv | High | 1.35 | 12.67 | 0 |
| petronas-chemicals-group-berhad-raw.csv | Low | 1.239999 | 12.480001 | 0 |
| petronas-chemicals-group-berhad-raw.csv | Close | 1.2975 | 12.517501 | 0 |
| petronas-chemicals-group-berhad-raw.csv | Adj Close | 2.205833 | 10.580665 | 0 |
| petronas-chemicals-group-berhad-raw.csv | Volume | -3,848,425 | 13,359,775 | 65 |
| press-metal-aluminium-holdings-berhad-raw.csv | Open | 3.9 | 6.3 | 79 |
| press-metal-aluminium-holdings-berhad-raw.csv | High | 3.975001 | 6.334999 | 84 |
| press-metal-aluminium-holdings-berhad-raw.csv | Low | 3.91 | 6.15 | 92 |
| press-metal-aluminium-holdings-berhad-raw.csv | Close | 3.9225 | 6.2625 | 85 |
| press-metal-aluminium-holdings-berhad-raw.csv | Adj Close | 3.718513 | 6.101487 | 85 |
| press-metal-aluminium-holdings-berhad-raw.csv | Volume | -4,043,600 | 16,620,800 | 62 |
| public-bank-berhad-raw.csv | Open | 3.7375 | 4.8375 | 0 |
| public-bank-berhad-raw.csv | High | 3.735 | 4.895 | 1 |
| public-bank-berhad-raw.csv | Low | 3.715 | 4.795 | 0 |
| public-bank-berhad-raw.csv | Close | 3.7125 | 4.8525 | 0 |
| public-bank-berhad-raw.csv | Adj Close | 2.545966 | 4.963692 | 0 |
| public-bank-berhad-raw.csv | Volume | -4,440,550 | 37,614,250 | 65 |
| tenaga-nasional-berhad-raw.csv | Open | 3.215 | 19.415 | 0 |
| tenaga-nasional-berhad-raw.csv | High | 3.1975 | 19.6175 | 0 |
| tenaga-nasional-berhad-raw.csv | Low | 3.199999 | 19.280001 | 0 |
| tenaga-nasional-berhad-raw.csv | Close | 3.195001 | 19.474999 | 0 |
| tenaga-nasional-berhad-raw.csv | Adj Close | 0.425622 | 20.296505 | 0 |
| tenaga-nasional-berhad-raw.csv | Volume | -3,824,700 | 13,518,100 | 63 |
| ytl-corporation-berhad-raw.csv | Open | -1.8625 | 4.7175 | 0 |
| ytl-corporation-berhad-raw.csv | High | -1.935 | 4.865 | 0 |
| ytl-corporation-berhad-raw.csv | Low | -1.8075 | 4.6125 | 0 |
| ytl-corporation-berhad-raw.csv | Close | -1.8475 | 4.6925 | 0 |
| ytl-corporation-berhad-raw.csv | Adj Close | -1.910868 | 4.61923 | 0 |
| ytl-corporation-berhad-raw.csv | Volume | -19,750,400 | 45,416,800 | 75 |
