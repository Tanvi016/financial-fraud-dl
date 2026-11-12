"""Schema-aware cleaning and feature engineering for the supplied dataset."""
from __future__ import annotations

import pandas as pd

TARGET = "is_fraud"
ID_COLUMNS = ["txn_id", "customer_id", "device_ip"]
NUMERIC_COLUMNS = [
    "amount", "old_balance_orig", "new_balance_orig",
    "old_balance_dest", "new_balance_dest",
]


def clean_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy. Never mutate or overwrite the raw input."""
    data = frame.copy()
    data.columns = [str(c).strip().lower() for c in data.columns]
    required = {"txn_id", "timestamp", "customer_id", "txn_type", "amount", TARGET}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required dataset columns: {sorted(missing)}")

    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].map(lambda value: value.strip() if isinstance(value, str) else value)

    # This dataset has two observed timestamp formats; parse both explicitly.
    stamp = data["timestamp"].astype("string")
    parsed = pd.to_datetime(stamp, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    alternate = parsed.isna()
    parsed.loc[alternate] = pd.to_datetime(
        stamp.loc[alternate], format="%d/%m/%Y %H:%M", errors="coerce"
    )
    if parsed.isna().any():
        raise ValueError(f"Unparseable timestamps: {int(parsed.isna().sum())}")
    data["timestamp"] = parsed

    for col in NUMERIC_COLUMNS:
        if col in data:
            data[col] = pd.to_numeric(data[col].astype("string").str.replace(",", "", regex=False), errors="coerce")
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    if data[TARGET].isna().any() or not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError("is_fraud must contain only binary 0/1 labels")

    # Canonicalize only observed variants. Keep unexpected values for inspection.
    aliases = {
        "PAYM3NT": "PAYMENT", "TRANSFER": "TRANSFER", "CASH_O0T": "CASH_OUT",
        "UPI_TRANSFER": "UPI_TRANSFER", "NETBANKING": "NET_BANKING",
    }
    data["txn_type"] = data["txn_type"].astype("string").str.upper().replace(aliases)

    if "device_os" in data:
        data["device_os"] = data["device_os"].astype("string").str.strip().replace({"": pd.NA})
    # Exact duplicate rows are removed; repeated transactions/customers are retained.
    data = data.drop_duplicates().reset_index(drop=True)
    return data


def make_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create point-in-time transaction features; exclude direct identifiers."""
    data = frame.copy()
    ts = pd.to_datetime(data["timestamp"], errors="coerce")
    features = pd.DataFrame(index=data.index)
    features["hour"] = ts.dt.hour
    features["day_of_week"] = ts.dt.dayofweek
    features["day_of_month"] = ts.dt.day
    features["is_weekend"] = (ts.dt.dayofweek >= 5).astype("int8")
    features["amount"] = data["amount"]
    features["old_balance_orig"] = data["old_balance_orig"]
    features["new_balance_orig"] = data["new_balance_orig"]
    features["old_balance_dest"] = data["old_balance_dest"]
    features["new_balance_dest"] = data["new_balance_dest"]
    features["orig_balance_delta"] = data["old_balance_orig"] - data["new_balance_orig"]
    features["dest_balance_delta"] = data["new_balance_dest"] - data["old_balance_dest"]
    features["orig_amount_delta"] = features["orig_balance_delta"] - data["amount"]
    features["dest_amount_delta"] = features["dest_balance_delta"] - data["amount"]
    features["txn_type"] = data["txn_type"].astype("string").fillna("UNKNOWN")
    features["device_os"] = data["device_os"].astype("string").fillna("UNKNOWN")
    return features
