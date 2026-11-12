# FraudLens raw dataset inspection

Inspected `messy_fraud_dataset_20k.csv` with Python's CSV reader before designing the pipeline. The raw input was not changed.

## Summary

- Rows: 20,000; columns: 12
- Exact duplicate rows: 0
- Target: `is_fraud`; values: 0 = 19,880 (99.4%), 1 = 120 (0.6%)
- Missing: `old_balance_dest` 400, `new_balance_dest` 400, `device_os` 200; zero missing in remaining fields
- Fraud-type/category field: none. Only binary supervised fraud detection is supported.
- `txn_id`: unique per row. `customer_id`: 3,978 distinct values. `device_ip`: 19,916 distinct values.

## Actual columns and observed values

| Column | Observed interpretation | Inspection |
|---|---|---|
| `txn_id` | Identifier | Text, 20,000 unique |
| `timestamp` | Transaction date/time | 19,400 `YYYY-MM-DD HH:MM:SS`; 600 `DD/MM/YYYY HH:MM`; all parse |
| `customer_id` | Customer identifier | Text, 3,978 unique |
| `txn_type` | Transaction category | 10 observed spellings/cases |
| `amount` | Numeric transaction amount | min 50.00, max 166,801.71, mean 5,147.80, median 2,501.46 |
| `old_balance_orig` | Numeric origin balance before transaction | min 883.59, max 1,784,850.25, mean 66,822.89, median 40,585.71 |
| `new_balance_orig` | Numeric origin balance after transaction | min -42,070.00, max 1,781,157.43; 9 negative and 116 zero |
| `old_balance_dest` | Numeric destination balance before transaction | 400 missing; nonmissing min 159.21, max 3,081,577.78 |
| `new_balance_dest` | Numeric destination balance after transaction | 400 missing; nonmissing min 859.21, max 3,087,925.02 |
| `device_ip` | High-cardinality network identifier | Text, 19,916 distinct values |
| `device_os` | Device operating system | 200 missing; Android 10,967, iOS 5,002, Windows 2,849, MacOS 982 |
| `is_fraud` | Binary target | 19,880 zeros, 120 ones |

`txn_type` counts: PAYMENT 6,943; UPI_TRANSFER 4,527; TRANSFER 4,362; CASH_OUT 2,353; NET_BANKING 1,615; PAYM3NT 63; transfer 50; upi_transfer 49; NETBANKING 21; CASH_O0T 17. These variants are consistent with formatting noise and should be canonicalized while unexpected categories remain observable.

## Modeling assessment

- Numeric features: amount, four balance fields, and derived balance deltas.
- Categorical features: transaction type and OS. The timestamp can provide hour, weekday, day, and weekend indicators.
- Identifiers: transaction ID, customer ID, IP address. Exclude from baseline feature matrix to reduce memorization and high-cardinality overfit.
- Leakage risks: target is `is_fraud`; any generator-only typology or post-outcome field must not be used. Balance-after-transaction values may be available at scoring time in this dataset but must be confirmed in a real deployment.
- Cleaning: preserve raw data; parse both observed timestamp formats, normalize observed transaction type variants, coerce numerics, impute missing numeric values inside the fitted train-only preprocessing pipeline, impute missing categorical values, and keep malformed/impossible values visible for review rather than deleting rows by default.
- Feature engineering: balance changes and amount-vs-balance-change residuals, plus calendar features. These are associations/patterns, not causal explanations.
- Split: stratified, customer-grouped train/validation/test (approximately 67/17/17), seed 42. All rows for a customer stay in one partition; validation selects the operating threshold and test is reserved for reporting. Fraud occurs across 117 customers, with 3 customers having repeated fraud rows.
- Imbalance: start with class-weighted Logistic Regression and Random Forest; use scale_pos_weight for XGBoost and class weights for ANN. Any resampling belongs inside training folds only.
- Limitations: synthetic data, only 120 positives, short date range, no actual fraud typology labels, and uncertain production availability of balance-after fields. Do not claim the measured model performance generalizes to real bank traffic.

## Storage boundary

The raw and cleaned training datasets, split/model artifacts, and evaluation outputs are the training side. Prediction history must be implemented later as a separate persistence store owned by the inference application. There must be no shared table/file in which training rows and user-submitted prediction records are mixed.
