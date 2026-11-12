# FraudLens Data Science phase

This phase uses the supplied `messy_fraud_dataset_20k.csv` as its fixed raw input. The raw CSV is never rewritten. `cleaning.py` contains the observed schema-specific cleaning rules and point-in-time features; `train.py` creates deterministic stratified, customer-grouped train/validation/test splits, fits baselines, and saves actual evaluation results and trained artifacts.

## Run

From the project root, install `requirements.txt`, then run `python data_science/train.py`. XGBoost and TensorFlow are optional at runtime and are skipped with a message if unavailable. The full pinned-compatible stack in `requirements.txt` enables all four models.

Outputs are written to `data/processed/`, `trained_models/`, and `evaluation/`. These are training artifacts and evaluation outputs. The prediction-history database belongs to the application/backend phase and must use a separate database/file and schema; training code must never append inference records to the training CSV or use prediction history as training input.

## Dataset caveats

The dataset is synthetic (the adjacent `seed.py` identifies its generator) and has 120 positive examples. No fraud-type label is present in the exported CSV; internal generator typology is not part of the dataset and is not used. Metrics from this synthetic dataset should not be presented as production performance. Identifier fields (`txn_id`, `customer_id`, `device_ip`) are omitted from model features. Customer-grouped stratified partitions keep each customer isolated; validation chooses the threshold and test metrics are reported once.
