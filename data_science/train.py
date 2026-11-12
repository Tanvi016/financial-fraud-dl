"""Train comparable fraud classifiers on the immutable source plus labeled simulations."""
from __future__ import annotations

import json
import os
import sys
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve, precision_recall_curve)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from cleaning import clean_transactions, make_model_features, ID_COLUMNS, TARGET

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "messy_fraud_dataset_20k.csv"
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "trained_models"
EVALUATION = ROOT / "evaluation"
SYNTHETIC = ROOT / "data" / "synthetic" / "simulated_fraud_2k.csv"


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_training_progress(model_name, **updates):
    path = EVALUATION / "training_progress.json"
    try:
        progress = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"models": {}}
    except (OSError, json.JSONDecodeError):
        progress = {"models": {}}
    progress.setdefault("models", {})[model_name] = {**progress.get("models", {}).get(model_name, {}), **json_safe(updates)}
    progress["updated_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    temp_path = path.with_name(f"training_progress.{os.getpid()}.tmp")
    temp_path.write_text(json.dumps(progress, indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(12):
        try:
            temp_path.replace(path)
            break
        except PermissionError:
            if attempt == 11:
                # Keep the training run alive if a reader temporarily locks the
                # dashboard status file; the next epoch callback will retry.
                temp_path.unlink(missing_ok=True)
                return
            time.sleep(0.05 * (attempt + 1))


def reconstruction_errors(model, values):
    reconstructed = np.asarray(model(values, training=False))
    return np.mean(np.square(np.asarray(values) - reconstructed), axis=1)


def save_model_weights(model, name):
    """Store dense Keras weights as a single NumPy array (reliable on Windows)."""
    arrays = [np.asarray(weight, dtype="float32").ravel() for weight in model.get_weights()]
    np.save(ARTIFACTS / f"{name}_weights.npy", np.concatenate(arrays))
    (ARTIFACTS / f"{name}_architecture.json").write_text(model.to_json(), encoding="utf-8")


def select_anomaly_threshold(y_valid, anomaly_scores):
    candidates = np.unique(anomaly_scores)
    return float(max(candidates, key=lambda t: f1_score(y_valid, anomaly_scores >= t, zero_division=0)))


def build_preprocessor(numeric, categorical, scale_numeric=False):
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    preprocess = ColumnTransformer([
        ("numeric", Pipeline(num_steps), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    return preprocess


def build_pipeline(model, numeric, categorical, scale_numeric=False):
    return Pipeline([("preprocess", build_preprocessor(numeric, categorical, scale_numeric)), ("model", model)])


def metric_record(y, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return {
        "threshold": float(threshold), "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "roc_curve": [a.tolist() for a in roc_curve(y, prob)[:2]],
        "pr_curve": [a.tolist() for a in precision_recall_curve(y, prob)[:2]],
    }


def generate_simulated_fraud_records(legitimate, count=2000):
    """Create scenario-based positive examples; never present them as real transactions."""
    rng = np.random.default_rng(SEED + 700)
    source = legitimate.reset_index(drop=True)
    anchors = source.iloc[rng.integers(0, len(source), size=count)].reset_index(drop=True)
    amounts = pd.to_numeric(source["amount"], errors="coerce").dropna().to_numpy()
    amount_median = float(np.median(amounts))
    amount_cap = float(np.quantile(amounts, 0.995))
    customer_balances = pd.to_numeric(source["old_balance_orig"], errors="coerce").dropna().to_numpy()
    destination_balances = pd.to_numeric(source["old_balance_dest"], errors="coerce").dropna().to_numpy()
    scenarios = rng.choice(["account_takeover", "balance_drain", "cash_out", "destination_mismatch"],
                           size=count, p=[0.30, 0.30, 0.25, 0.15])
    txn_types = np.select(
        [np.isin(scenarios, ["account_takeover", "destination_mismatch"]), scenarios == "cash_out"],
        ["TRANSFER", "CASH_OUT"], default="UPI_TRANSFER"
    )
    rows = anchors.copy()
    rows["txn_id"] = [f"SIM-TXN-{i:05d}" for i in range(1, count + 1)]
    rows["customer_id"] = [f"SIM-CUST-{i:05d}" for i in range(1, count + 1)]
    if "device_ip" in rows:
        rows["device_ip"] = [f"SIM-IP-{i:05d}" for i in range(1, count + 1)]
    rows["txn_type"] = txn_types
    rows["is_fraud"] = 1
    rows["simulation_scenario"] = scenarios
    rows["_is_simulated"] = True

    # Mix high-value transfers with ordinary-sized takeovers so amount alone is not the label.
    multiplier = rng.lognormal(mean=0.25, sigma=0.85, size=count)
    simulated_amounts = np.clip(amount_median * multiplier, max(50.0, float(np.min(amounts))), amount_cap)
    rows["amount"] = simulated_amounts
    sampled_balance = rng.choice(customer_balances, size=count, replace=True)
    drain = np.isin(scenarios, ["account_takeover", "balance_drain", "cash_out"])
    source_balance = np.maximum(sampled_balance, simulated_amounts * rng.uniform(1.02, 1.25, size=count))
    # Some takeovers leave a small remainder, while others use an ordinary partial transfer.
    partial = rng.random(count) < 0.28
    source_delta = np.where(partial, simulated_amounts * rng.uniform(0.35, 0.85, size=count), simulated_amounts)
    source_delta = np.where(drain, source_delta, simulated_amounts * rng.uniform(0.55, 1.0, size=count))
    rows["old_balance_orig"] = source_balance
    rows["new_balance_orig"] = np.maximum(0.0, source_balance - source_delta)

    destination_before = rng.choice(destination_balances, size=count, replace=True)
    dest_delta = simulated_amounts.copy()
    mismatch = scenarios == "destination_mismatch"
    dest_delta[mismatch] = simulated_amounts[mismatch] * rng.uniform(0.05, 0.45, size=int(mismatch.sum()))
    rows["old_balance_dest"] = destination_before
    rows["new_balance_dest"] = destination_before + dest_delta

    # Simulate some after-hours activity while retaining ordinary timestamps for the rest.
    stamps = pd.to_datetime(rows["timestamp"], errors="coerce")
    night = rng.random(count) < 0.38
    hour_offsets = rng.integers(0, 5, size=count)
    stamps = stamps.dt.floor("D") + pd.to_timedelta(np.where(night, hour_offsets, stamps.dt.hour), unit="h") \
        + pd.to_timedelta(stamps.dt.minute, unit="m")
    rows["timestamp"] = stamps.dt.strftime("%Y-%m-%d %H:%M:%S")
    return rows


def compact_metrics(y, prob, threshold):
    """Small per-source summaries for transparent mixed test-set reporting."""
    pred = np.asarray(prob) >= threshold
    y = np.asarray(y, dtype=int)
    result = {"rows": int(len(y)), "fraud_cases": int(y.sum()),
              "accuracy": float(accuracy_score(y, pred)) if len(y) else None,
              "precision": float(precision_score(y, pred, zero_division=0)) if len(y) else None,
              "recall": float(recall_score(y, pred, zero_division=0)) if len(y) else None,
              "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()}
    if len(np.unique(y)) == 2:
        result["balanced_accuracy"] = float(balanced_accuracy_score(y, pred))
        result["roc_auc"] = float(roc_auc_score(y, prob))
        result["pr_auc"] = float(average_precision_score(y, prob))
    return result


def choose_risk_thresholds(y_valid, val_prob, decision_threshold, high_precision_target=0.80,
                           high_minimum_support=10):
    """Use validation only; require precision and enough observations for a HIGH cutoff."""
    candidates = np.unique(np.r_[val_prob, 1.0])
    eligible = []
    for threshold in candidates:
        if threshold < decision_threshold:
            continue
        predicted = val_prob >= threshold
        if int(predicted.sum()) < high_minimum_support:
            continue
        precision = precision_score(y_valid, predicted, zero_division=0)
        recall = recall_score(y_valid, predicted, zero_division=0)
        if precision >= high_precision_target:
            eligible.append((recall, precision, float(threshold), int(predicted.sum())))
    if eligible:
        recall, precision, high_threshold, count = max(eligible, key=lambda item: (item[0], item[1], item[2]))
        high_validation = {"flagged": count, "precision": float(precision), "recall": float(recall)}
    else:
        # Validation did not support an evidence-based HIGH cutoff.
        high_threshold = None
        high_validation = {"flagged": 0, "precision": None, "recall": 0.0}
    return {
        "method": "Validation-selected operating bands",
        "low": {"below": float(decision_threshold)},
        "medium": {"from": float(decision_threshold), "below": float(high_threshold) if high_threshold is not None else None},
        "high": {"from": float(high_threshold) if high_threshold is not None else None,
                 "validation_precision_target": high_precision_target,
                 "minimum_validation_transactions": high_minimum_support,
                 "enabled": high_threshold is not None},
        "decision_threshold": float(decision_threshold),
        "high_validation": high_validation,
        "selection_note": f"The HIGH cutoff is the validation operating point with greatest recall among thresholds meeting the {high_precision_target:.0%} precision target and at least {high_minimum_support} flagged validation transactions. It is never below the F1-selected decision threshold. HIGH is disabled when validation has no qualifying cutoff.",
    }


def evaluate_risk_bands(y, prob, policy):
    """Report untouched test-set counts and observed label rates per band."""
    medium = policy["medium"]["from"]
    high = policy["high"]["from"]
    masks = {"LOW": prob < medium,
             "MEDIUM": (prob >= medium) & ((prob < high) if high is not None else True),
             "HIGH": (prob >= high) if high is not None else np.zeros(len(prob), dtype=bool)}
    total_fraud = int(np.asarray(y).sum())
    rows = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        frauds = int(np.asarray(y)[mask].sum())
        rows[name] = {
            "transactions": count,
            "fraud_cases": frauds,
            "observed_fraud_rate": float(frauds / count) if count else None,
            "share_of_test_fraud_captured": float(frauds / total_fraud) if total_fraud else None,
        }
    return rows


def main():
    for path in (PROCESSED, ARTIFACTS, EVALUATION):
        path.mkdir(parents=True, exist_ok=True)
    progress_path = EVALUATION / "training_progress.json"
    progress_path.write_text(json.dumps({"models": {}, "updated_at": pd.Timestamp.now(tz="UTC").isoformat()}, indent=2), encoding="utf-8")
    raw = pd.read_csv(RAW, low_memory=False)
    original = clean_transactions(raw)
    original["_is_simulated"] = False
    original_features = make_model_features(original)
    original_labels = original[TARGET].astype(int)
    original_groups = original["customer_id"]
    outer = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    train_idx, hold_idx = next(outer.split(original_features, original_labels, original_groups))
    hold_features, hold_labels = original_features.iloc[hold_idx], original_labels.iloc[hold_idx]
    hold_groups = original_groups.iloc[hold_idx]
    inner = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=SEED + 1)
    valid_rel, test_rel = next(inner.split(hold_features, hold_labels, hold_groups))
    valid_idx, test_idx = hold_idx[valid_rel], hold_idx[test_rel]

    # Generate examples only from legitimate rows in the original training partition.
    original_train = original.iloc[train_idx]
    simulated_raw = generate_simulated_fraud_records(
        original_train.loc[original_train[TARGET].eq(0)], count=2000
    )
    SYNTHETIC.parent.mkdir(parents=True, exist_ok=True)
    simulated_raw.drop(columns=["_is_simulated"], errors="ignore").to_csv(SYNTHETIC, index=False)
    print(f"Generated {len(simulated_raw)} explicitly simulated fraud scenarios at {SYNTHETIC}.", flush=True)
    simulated = clean_transactions(simulated_raw)
    simulated["_is_simulated"] = True
    sim_features = make_model_features(simulated)
    sim_order = np.random.default_rng(SEED + 701).permutation(len(simulated))
    sim_train_idx, sim_valid_idx, sim_test_idx = sim_order[:1334], sim_order[1334:1667], sim_order[1667:]

    original_train_features, original_valid_features, original_test_features = (
        original_features.iloc[idx] for idx in (train_idx, valid_idx, test_idx)
    )
    original_train_labels, original_valid_labels, original_test_labels = (
        original_labels.iloc[idx] for idx in (train_idx, valid_idx, test_idx)
    )
    X_train = pd.concat([original_train_features, sim_features.iloc[sim_train_idx]], ignore_index=True)
    X_valid = pd.concat([original_valid_features, sim_features.iloc[sim_valid_idx]], ignore_index=True)
    X_test = pd.concat([original_test_features, sim_features.iloc[sim_test_idx]], ignore_index=True)
    y_train = pd.concat([original_train_labels.reset_index(drop=True), simulated.iloc[sim_train_idx][TARGET].reset_index(drop=True)], ignore_index=True).astype(int)
    y_valid = pd.concat([original_valid_labels.reset_index(drop=True), simulated.iloc[sim_valid_idx][TARGET].reset_index(drop=True)], ignore_index=True).astype(int)
    y_test = pd.concat([original_test_labels.reset_index(drop=True), simulated.iloc[sim_test_idx][TARGET].reset_index(drop=True)], ignore_index=True).astype(int)
    train_is_simulated = np.r_[np.zeros(len(original_train_features), dtype=bool), np.ones(len(sim_train_idx), dtype=bool)]
    valid_is_simulated = np.r_[np.zeros(len(original_valid_features), dtype=bool), np.ones(len(sim_valid_idx), dtype=bool)]
    test_is_simulated = np.r_[np.zeros(len(original_test_features), dtype=bool), np.ones(len(sim_test_idx), dtype=bool)]
    original_train_positive = int(original_train_labels.sum())
    simulated_weight = original_train_positive / max(1, int(train_is_simulated.sum()))
    train_sample_weight = np.where(train_is_simulated, simulated_weight, 1.0).astype("float32")
    valid_original_mask = ~valid_is_simulated

    # Persist an auditable cleaned copy with the partition and synthetic-source labels.
    partition_frames = []
    for name, orig_idx, sim_idx in (("train", train_idx, sim_train_idx),
                                    ("validation", valid_idx, sim_valid_idx),
                                    ("test", test_idx, sim_test_idx)):
        orig_part = original.iloc[orig_idx].copy()
        sim_part = simulated.iloc[sim_idx].copy()
        orig_part["split"] = name
        sim_part["split"] = name
        partition_frames.extend([orig_part, sim_part])
    data = pd.concat(partition_frames, ignore_index=True)
    data.to_csv(PROCESSED / "cleaned_transactions.csv", index=False)

    # Keep a compact, training-only reference set for local SHAP explanations.
    original_train_features.sample(n=min(8, len(original_train_features)), random_state=SEED).to_csv(
        ARTIFACTS / "shap_background.csv", index=False
    )
    assert set(original_groups.iloc[train_idx]).isdisjoint(original_groups.iloc[valid_idx])
    assert set(original_groups.iloc[train_idx]).isdisjoint(original_groups.iloc[test_idx])
    assert set(original_groups.iloc[valid_idx]).isdisjoint(original_groups.iloc[test_idx])
    assert len(simulated) == 2000 and simulated[TARGET].eq(1).all()
    numeric = list(X_train.select_dtypes(include=["number"]).columns)
    categorical = [c for c in X_train.columns if c not in numeric]
    baseline = {}
    for column in X_train.columns:
        if column in numeric:
            baseline[column] = float(X_train[column].median())
        else:
            mode = X_train[column].mode(dropna=True)
            baseline[column] = str(mode.iloc[0]) if len(mode) else "UNKNOWN"
    with open(ARTIFACTS / "feature_baseline.json", "w", encoding="utf8") as f:
        json.dump(baseline, f, indent=2)
    models = {
        "logistic_regression": build_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED), numeric, categorical, True
        ),
        "random_forest": build_pipeline(
            RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=SEED), numeric, categorical
        ),
    }
    try:
        from xgboost import XGBClassifier
        ratio = max(1.0, float((y_train == 0).sum() / max(1, (y_train == 1).sum())))
        models["xgboost"] = build_pipeline(
            XGBClassifier(n_estimators=450, max_depth=5, learning_rate=0.04, subsample=0.85,
                          colsample_bytree=0.85, reg_lambda=2, scale_pos_weight=ratio,
                          eval_metric="aucpr", n_jobs=-1, random_state=SEED), numeric, categorical
        )
    except ImportError:
        print("xgboost unavailable; skipping XGBoost (install requirements.txt to include it)")

    # Validation threshold selection optimizes F1; test set is used once for final reporting.
    feature_hashes = {name: set(pd.util.hash_pandas_object(frame, index=False).to_numpy())
                      for name, frame in (("train", X_train), ("validation", X_valid), ("test", X_test))}
    results = {"split": {"train": len(y_train), "validation": len(y_valid), "test": len(y_test),
                         "train_positive": int(y_train.sum()), "validation_positive": int(y_valid.sum()),
                         "test_positive": int(y_test.sum()), "grouped_by": "customer_id",
                         "customer_groups_disjoint": True,
                         "original_rows_partitioned_before_simulation": True,
                         "train_original_rows": int(len(original_train_features)),
                         "validation_original_rows": int(len(original_valid_features)),
                         "test_original_rows": int(len(original_test_features)),
                         "validation_original_positive": int(original_valid_labels.sum()),
                         "test_original_positive": int(original_test_labels.sum()),
                         "train_simulated_fraud": int(len(sim_train_idx)),
                         "validation_simulated_fraud": int(len(sim_valid_idx)),
                         "test_simulated_fraud": int(len(sim_test_idx)),
                         "exact_feature_duplicates_across_train_validation": len(feature_hashes["train"] & feature_hashes["validation"]),
                         "exact_feature_duplicates_across_train_test": len(feature_hashes["train"] & feature_hashes["test"]),
                         "exact_feature_duplicates_across_validation_test": len(feature_hashes["validation"] & feature_hashes["test"])}, "models": {}}
    results["training_weights"] = {
        "simulated_positive_row_weight": float(simulated_weight),
        "simulated_positive_total_weight": float(simulated_weight * len(sim_train_idx)),
        "original_training_positive_rows": original_train_positive,
        "weighting_rule": "Each simulated training positive is downweighted so all simulated positives together have the same total sample weight as the original positive training rows.",
    }
    results["data_audit"] = {
        "original_customer_groups_disjoint": True,
        "original_rows_split_before_simulation_generation": True,
        "simulated_rows_anchored_only_to_original_training_legitimate_rows": True,
        "target_column_excluded_from_features": TARGET not in X_train.columns,
        "identifier_columns_excluded_from_features": all(column not in X_train.columns for column in ID_COLUMNS),
        "simulation_scenario_excluded_from_features": "simulation_scenario" not in X_train.columns,
        "test_metrics_include_rule_generated_scenarios": True,
        "warning": "Simulation labels are defined by explicit transaction scenarios. Their held-out results evaluate consistency with those rules, not independent real-world fraud detection.",
    }
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train, model__sample_weight=train_sample_weight)
        val_prob = pipeline.predict_proba(X_valid)[:, 1]
        val_prob_original = val_prob[valid_original_mask]
        y_valid_original = y_valid.to_numpy()[valid_original_mask]
        thresholds = np.linspace(0.01, 0.99, 99)
        best = max(thresholds, key=lambda t: f1_score(y_valid_original, val_prob_original >= t, zero_division=0))
        test_prob = pipeline.predict_proba(X_test)[:, 1]
        results["models"][name] = metric_record(y_test, test_prob, float(best))
        results["models"][name]["test_original_only"] = compact_metrics(
            y_test.to_numpy()[~test_is_simulated], test_prob[~test_is_simulated], float(best)
        )
        results["models"][name]["test_simulated_only"] = compact_metrics(
            y_test.to_numpy()[test_is_simulated], test_prob[test_is_simulated], float(best)
        )
        results["models"][name]["validation_original_only"] = compact_metrics(
            y_valid_original, val_prob_original, float(best)
        )
        results["models"][name]["threshold_validation_scope"] = "original validation rows only"
        risk_policy = choose_risk_thresholds(y_valid_original, val_prob_original, float(best))
        results["models"][name]["risk_policy"] = risk_policy
        results["models"][name]["risk_band_test"] = evaluate_risk_bands(y_test.to_numpy(), test_prob, risk_policy)
        fitted_model = pipeline.named_steps["model"]
        results["models"][name]["configuration"] = fitted_model.get_params(deep=False)
        names = pipeline.named_steps["preprocess"].get_feature_names_out()
        if hasattr(fitted_model, "feature_importances_"):
            importance = np.asarray(fitted_model.feature_importances_)
        elif hasattr(fitted_model, "coef_"):
            importance = np.mean(np.abs(fitted_model.coef_), axis=0)
        else:
            importance = np.zeros(len(names))
        order = np.argsort(importance)[::-1][:25]
        results["models"][name]["feature_importance"] = [
            {"feature": str(names[i]), "importance": float(importance[i])} for i in order
        ]
        joblib.dump(pipeline, ARTIFACTS / f"{name}.joblib")

    # ANN is optional and trained offline. It shares the same split and preprocessing recipe.
    try:
        import tensorflow as tf
        from sklearn.utils.class_weight import compute_class_weight
        tf.keras.utils.set_random_seed(SEED)
        class TrainingProgress(tf.keras.callbacks.Callback):
            def __init__(self, model_name, total_epochs):
                super().__init__()
                self.model_name = model_name
                self.total_epochs = total_epochs

            def on_train_begin(self, logs=None):
                save_training_progress(self.model_name, status="running", epoch=0,
                                       total_epochs=self.total_epochs, metrics={})

            def on_epoch_end(self, epoch, logs=None):
                save_training_progress(self.model_name, status="running", epoch=epoch + 1,
                                       total_epochs=self.total_epochs, metrics=logs or {})

            def on_train_end(self, logs=None):
                progress = json.loads((EVALUATION / "training_progress.json").read_text(encoding="utf-8"))
                current = progress.get("models", {}).get(self.model_name, {})
                save_training_progress(self.model_name, status="completed", epoch=current.get("epoch", 0),
                                       total_epochs=self.total_epochs, metrics=current.get("metrics", {}))
        prep = models["logistic_regression"].named_steps["preprocess"]
        A_train, A_valid, A_test = (prep.transform(x) for x in (X_train, X_valid, X_test))
        A_train, A_valid, A_test = (x.toarray() if hasattr(x, "toarray") else np.asarray(x) for x in (A_train, A_valid, A_test))
        weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train.to_numpy())
        ann_sample_weight = train_sample_weight * np.where(
            y_train.to_numpy() == 1, float(weights[1]), float(weights[0])
        )
        ann = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(A_train.shape[1],)),
            tf.keras.layers.Dense(128), tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.25), tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.20), tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        ann.compile(optimizer="adam", loss="binary_crossentropy",
                    metrics=[tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"), "accuracy"])
        history = ann.fit(A_train, y_train, validation_data=(A_valid, y_valid), epochs=100, batch_size=256,
                          sample_weight=ann_sample_weight,
                          callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                                     TrainingProgress("ann", 100)],
                          verbose=0)
        val_prob = ann.predict(A_valid, verbose=0).ravel()
        val_prob_original = val_prob[valid_original_mask]
        y_valid_original = y_valid.to_numpy()[valid_original_mask]
        best = max(np.linspace(.01, .99, 99), key=lambda t: f1_score(y_valid_original, val_prob_original >= t, zero_division=0))
        test_prob = ann.predict(A_test, verbose=0).ravel()
        results["models"]["ann"] = metric_record(y_test, test_prob, float(best))
        results["models"]["ann"]["test_original_only"] = compact_metrics(
            y_test.to_numpy()[~test_is_simulated], test_prob[~test_is_simulated], float(best)
        )
        results["models"]["ann"]["test_simulated_only"] = compact_metrics(
            y_test.to_numpy()[test_is_simulated], test_prob[test_is_simulated], float(best)
        )
        results["models"]["ann"]["validation_original_only"] = compact_metrics(
            y_valid_original, val_prob_original, float(best)
        )
        results["models"]["ann"]["threshold_validation_scope"] = "original validation rows only"
        risk_policy = choose_risk_thresholds(y_valid_original, val_prob_original, float(best))
        results["models"]["ann"]["risk_policy"] = risk_policy
        results["models"]["ann"]["risk_band_test"] = evaluate_risk_bands(y_test.to_numpy(), test_prob, risk_policy)
        results["ann_training"] = {
            "hidden_dense_layers": 3, "hidden_layer_units": [128, 64, 32],
            "hidden_layer_activations": ["relu", "relu", "relu"],
            "first_layer_detail": "Dense(128, linear) followed by BatchNormalization, ReLU, and Dropout(0.25)",
            "output_units": 1, "output_activation": "sigmoid",
            "optimizer": "Adam", "loss": "binary_crossentropy", "batch_size": 256,
            "max_epochs": 100, "epochs_trained": len(history.history["loss"]),
            "early_stopping": {"monitor": "val_loss", "patience": 10, "restore_best_weights": True},
        }
        print("Saving the fitted ANN weights and architecture.", flush=True)
        save_model_weights(ann, "ann")
        print("Saved ANN; writing its preprocessing and history artifacts.", flush=True)
        prep_path = ARTIFACTS / "ann_preprocessor.joblib"
        joblib.dump(prep, prep_path)
        with open(EVALUATION / "ann_training_history.json", "w", encoding="utf8") as f:
            json.dump(history.history, f, indent=2)
        results["ann_architecture"] = ann.to_json()

        # A reconstruction autoencoder learns only from legitimate training rows.
        print("ANN training complete; preparing legitimate-only autoencoder data.", flush=True)
        normal_train_mask = y_train.to_numpy() == 0
        normal_valid_mask = y_valid.to_numpy() == 0
        ae_preprocessor = build_preprocessor(numeric, categorical, scale_numeric=True)
        print("Fitting autoencoder preprocessing on legitimate training rows.", flush=True)
        ae_preprocessor.fit(X_train.loc[normal_train_mask])
        print("Transforming autoencoder training and validation rows.", flush=True)
        to_dense = lambda x: x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        AE_train = to_dense(ae_preprocessor.transform(X_train.loc[normal_train_mask])).astype("float32")
        AE_valid = to_dense(ae_preprocessor.transform(X_valid.loc[normal_valid_mask])).astype("float32")
        AE_valid_all = to_dense(ae_preprocessor.transform(X_valid)).astype("float32")
        AE_test = to_dense(ae_preprocessor.transform(X_test)).astype("float32")
        print(f"Autoencoder inputs ready: {AE_train.shape[0]} training rows, {AE_train.shape[1]} features.", flush=True)
        autoencoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(AE_train.shape[1],)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu", name="latent"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(AE_train.shape[1], activation="linear"),
        ], name="fraudlens_autoencoder")
        autoencoder.compile(optimizer="adam", loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")])
        print(f"Training autoencoder on {len(AE_train)} legitimate rows for up to 100 epochs.", flush=True)
        ae_history = autoencoder.fit(
            AE_train, AE_train, validation_data=(AE_valid, AE_valid), epochs=100, batch_size=256,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                       TrainingProgress("autoencoder", 100)], verbose=0,
        )
        val_anomaly_scores = reconstruction_errors(autoencoder, AE_valid_all)
        test_anomaly_scores = reconstruction_errors(autoencoder, AE_test)
        val_anomaly_scores_original = val_anomaly_scores[valid_original_mask]
        ae_threshold = select_anomaly_threshold(y_valid.to_numpy()[valid_original_mask], val_anomaly_scores_original)
        results["models"]["autoencoder"] = metric_record(y_test, test_anomaly_scores, ae_threshold)
        results["models"]["autoencoder"]["test_original_only"] = compact_metrics(
            y_test.to_numpy()[~test_is_simulated], test_anomaly_scores[~test_is_simulated], ae_threshold
        )
        results["models"]["autoencoder"]["test_simulated_only"] = compact_metrics(
            y_test.to_numpy()[test_is_simulated], test_anomaly_scores[test_is_simulated], ae_threshold
        )
        results["models"]["autoencoder"]["validation_original_only"] = compact_metrics(
            y_valid.to_numpy()[valid_original_mask], val_anomaly_scores_original, ae_threshold
        )
        results["models"]["autoencoder"]["threshold_validation_scope"] = "original validation rows only"
        ae_risk_policy = choose_risk_thresholds(y_valid.to_numpy()[valid_original_mask], val_anomaly_scores_original, ae_threshold)
        results["models"]["autoencoder"]["risk_policy"] = ae_risk_policy
        results["models"]["autoencoder"]["risk_band_test"] = evaluate_risk_bands(y_test.to_numpy(), test_anomaly_scores, ae_risk_policy)
        results["models"]["autoencoder"]["score_kind"] = "reconstruction_error"
        results["autoencoder_training"] = {
            "trained_on": "legitimate training rows only",
            "normal_train_samples": int(normal_train_mask.sum()),
            "normal_validation_samples": int(normal_valid_mask.sum()),
            "input_features": int(AE_train.shape[1]), "latent_units": 16,
            "optimizer": "Adam", "loss": "mean_squared_error", "batch_size": 256,
            "max_epochs": 100, "epochs_trained": len(ae_history.history["loss"]),
            "early_stopping": {"monitor": "val_loss", "patience": 10, "restore_best_weights": True},
            "anomaly_threshold": ae_threshold,
        }
        print("Saving the fitted autoencoder weights and architecture.", flush=True)
        save_model_weights(autoencoder, "autoencoder")
        joblib.dump(ae_preprocessor, ARTIFACTS / "autoencoder_preprocessor.joblib")
        np.save(ARTIFACTS / "autoencoder_validation_normal_errors.npy",
                reconstruction_errors(autoencoder, AE_valid).astype("float32"))
        with open(EVALUATION / "autoencoder_training_history.json", "w", encoding="utf8") as f:
            json.dump(ae_history.history, f, indent=2)
        results["autoencoder_architecture"] = autoencoder.to_json()
    except ImportError:
        print("tensorflow unavailable; skipping ANN (install requirements.txt to include it)")

    results["dataset"] = {"rows": len(data), "original_rows": len(original),
                          "positive": int(data[TARGET].sum()), "positive_rate": float(data[TARGET].mean()),
                          "simulated_fraud_rows": int(data["_is_simulated"].sum()),
                          "label_note": "2,000 scenario-based simulated fraud records are synthetic, generated from explicit account-takeover, balance-drain, cash-out, and destination-mismatch patterns; they are not real-world observations."}
    results["environment"] = {"python": sys.version.split()[0], "pandas": pd.__version__, "scikit_learn": sklearn.__version__}
    try:
        import xgboost
        results["environment"]["xgboost"] = xgboost.__version__
    except ImportError:
        pass
    try:
        import tensorflow as tf
        results["environment"]["tensorflow"] = tf.__version__
    except ImportError:
        pass
    results["seed"] = SEED
    results["features"] = list(X_train.columns)
    results["warning"] = "Synthetic development dataset; metrics describe this split only and are not evidence of production performance."
    with open(EVALUATION / "metrics.json", "w", encoding="utf8") as f:
        json.dump(json_safe(results), f, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in results.items() if k != "models"}, indent=2))
    print("Saved evaluation metrics and model artifacts. No prediction-history store is created here.")


if __name__ == "__main__":
    main()
