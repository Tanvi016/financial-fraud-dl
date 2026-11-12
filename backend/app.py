"""FraudLens API. Prediction history is persisted to its own SQLite database."""
from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import time
import urllib.error
import urllib.request
import uuid
from functools import lru_cache
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / "backend" / ".env", override=False)
ARTIFACTS = ROOT / "trained_models"
EVALUATION = ROOT / "evaluation"
HISTORY_DB = Path(os.getenv("FRAUDLENS_HISTORY_DB", str(ROOT / "data" / "prediction_history.db")))
MODEL_NAMES = {"logistic_regression", "random_forest", "xgboost", "ann"}
ALLOWED_TYPES = {"PAYMENT", "UPI_TRANSFER", "TRANSFER", "CASH_OUT", "NET_BANKING"}

app = FastAPI(title="FraudLens API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
                   allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])


class Transaction(BaseModel):
    timestamp: str
    txn_type: str
    amount: float = Field(gt=0)
    old_balance_orig: float = Field(ge=0)
    new_balance_orig: float
    old_balance_dest: float | None = Field(default=None, ge=0)
    new_balance_dest: float | None = Field(default=None, ge=0)
    device_os: str | None = None


@contextmanager
def connect():
    HISTORY_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(HISTORY_DB)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def initialize_history_db():
    with connect() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS predictions (
            prediction_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, source TEXT NOT NULL,
            model_name TEXT NOT NULL, txn_type TEXT NOT NULL, amount REAL NOT NULL,
            fraud_probability REAL NOT NULL, predicted_class INTEGER NOT NULL,
            risk_level TEXT NOT NULL, explanation_json TEXT NOT NULL, alert INTEGER NOT NULL,
            input_json TEXT NOT NULL)""")


@app.on_event("startup")
def startup():
    initialize_history_db()
    refresh_stored_risk_bands()
    # Load and execute the default ANN once during service startup. Its first
    # TensorFlow call is costly; warming here keeps that delay out of scoring.
    if model_path("ann").exists():
        try:
            probabilities_for("ann", load_shap_background().iloc[:1])
        except Exception:
            import logging
            logging.getLogger(__name__).exception("ANN warm-up failed; first analyzer request may be slower")


def load_evaluation():
    path = EVALUATION / "metrics.json"
    if not path.exists():
        raise HTTPException(503, "Models are not trained yet. Run data_science/train.py.")
    return json.loads(path.read_text(encoding="utf-8"))


def risk_band(name, probability, evaluation=None):
    evaluation = evaluation or load_evaluation()
    model_metrics = evaluation.get("models", {}).get(name, {})
    policy = model_metrics.get("risk_policy")
    if not policy:
        # Backward compatibility until the training script is rerun.
        return "MEDIUM" if probability >= float(model_metrics.get("threshold", 1.0)) else "LOW"
    high = policy.get("high", {})
    if high.get("enabled") and high.get("from") is not None and probability >= float(high["from"]):
        return "HIGH"
    medium = float(policy.get("medium", {}).get("from", model_metrics.get("threshold", 1.0)))
    return "MEDIUM" if probability >= medium else "LOW"


def refresh_stored_risk_bands():
    """Keep persisted review labels aligned with the active model policy."""
    path = EVALUATION / "metrics.json"
    if not path.exists():
        return
    evaluation = json.loads(path.read_text(encoding="utf-8"))
    with connect() as con:
        rows = con.execute("SELECT prediction_id,model_name,fraud_probability,predicted_class FROM predictions").fetchall()
        for row in rows:
            level = risk_band(row["model_name"], float(row["fraud_probability"]), evaluation)
            con.execute("UPDATE predictions SET risk_level=?,alert=? WHERE prediction_id=?",
                        (level, int(row["predicted_class"] == 1), row["prediction_id"]))


def model_path(name):
    if name in {"ann", "autoencoder"}:
        return ARTIFACTS / f"{name}_weights.npy"
    return ARTIFACTS / f"{name}.joblib"


@lru_cache(maxsize=8)
def load_model(name):
    if name in {"ann", "autoencoder"}:
        import tensorflow as tf
        architecture = (ARTIFACTS / f"{name}_architecture.json").read_text(encoding="utf-8")
        model = tf.keras.models.model_from_json(architecture)
        flattened = np.load(model_path(name))
        weights = []
        offset = 0
        for current in model.get_weights():
            size = current.size
            weights.append(flattened[offset:offset + size].reshape(current.shape))
            offset += size
        if offset != len(flattened):
            raise ValueError(f"Invalid stored weights for {name}: expected {offset}, received {len(flattened)}")
        model.set_weights(weights)
        return model
    return joblib.load(model_path(name))


@lru_cache(maxsize=2)
def load_ann_preprocessor():
    return joblib.load(ARTIFACTS / "ann_preprocessor.joblib")


@lru_cache(maxsize=2)
def load_autoencoder_preprocessor():
    return joblib.load(ARTIFACTS / "autoencoder_preprocessor.joblib")


def autoencoder_output(features, evaluation=None):
    if not model_path("autoencoder").exists() or not (ARTIFACTS / "autoencoder_validation_normal_errors.npy").exists():
        return None
    evaluation = evaluation or load_evaluation()
    preprocessor = load_autoencoder_preprocessor()
    matrix = preprocessor.transform(features)
    matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    model = load_model("autoencoder")
    reconstructed = np.asarray(model(matrix.astype("float32"), training=False))
    error = float(np.mean(np.square(matrix - reconstructed), axis=1)[0])
    reference = np.sort(np.load(ARTIFACTS / "autoencoder_validation_normal_errors.npy"))
    percentile = float(np.searchsorted(reference, error, side="right") / max(1, len(reference)))
    metrics = evaluation.get("models", {}).get("autoencoder", {})
    threshold = float(metrics.get("threshold", 1.0))
    predicted = int(error >= threshold)
    risk = risk_band("autoencoder", error, evaluation)
    return {"score": error, "score_kind": "reconstruction_error", "anomaly_percentile": percentile,
            "threshold": threshold, "predicted_class": predicted,
            "classification": "Anomalous" if predicted else "Typical", "risk_level": risk,
            "review_priority": {"HIGH": "Immediate review", "MEDIUM": "Review", "LOW": "Routine monitoring"}[risk]}


def probability_for(name, features):
    return float(probabilities_for(name, features)[0])


def probabilities_for(name, features):
    if name == "ann":
        preprocessor = load_ann_preprocessor()
        matrix = preprocessor.transform(features)
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        # Calling the model directly avoids Keras predict()'s per-call data
        # adapter and thread-pool setup, which is expensive for single rows.
        output = load_model(name)(matrix, training=False)
        return np.asarray(output).reshape(-1)
    return load_model(name).predict_proba(features)[:, 1]


@lru_cache(maxsize=1)
def load_shap_background():
    path = ARTIFACTS / "shap_background.csv"
    if not path.exists():
        raise HTTPException(503, "SHAP reference data is missing. Run data_science/train.py to create it.")
    return pd.read_csv(path)


@lru_cache(maxsize=2)
def load_shap_category_map(column):
    background = load_shap_background().copy()
    universe = set(background[column].fillna("UNKNOWN").astype(str))
    universe.update(ALLOWED_TYPES if column == "txn_type" else {"Android", "iOS", "Windows", "MacOS", "Unknown", "UNKNOWN"})
    return {index: value for index, value in enumerate(sorted(universe))}


@lru_cache(maxsize=4)
def load_shap_explainer(name):
    import shap
    background = load_shap_background().copy()
    columns = list(background.columns)
    category_maps = {}
    for column in ("txn_type", "device_os"):
        category_maps[column] = load_shap_category_map(column)
        background[column] = background[column].fillna("UNKNOWN").astype(str).map({value: index for index, value in category_maps[column].items()})
    def predict_rows(values):
        frame = pd.DataFrame(values, columns=columns)
        for column in frame.columns:
            if column in {"txn_type", "device_os"}:
                reverse = category_maps[column]
                unknown_code = next((code for code, value in reverse.items() if value == "UNKNOWN"), 0)
                frame[column] = frame[column].round().fillna(unknown_code).astype(int).map(reverse).fillna("UNKNOWN")
            else:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return probabilities_for(name, frame)
    return shap.Explainer(predict_rows, shap.maskers.Independent(background), algorithm="permutation", seed=42)


FEATURE_COPY = {
    "hour": ("Transaction hour", "hour of day"),
    "day_of_week": ("Day of week", "weekday index"),
    "day_of_month": ("Day of month", "calendar day"),
    "is_weekend": ("Weekend", "whether the transaction falls on a weekend"),
    "amount": ("Transaction amount", "amount sent"),
    "old_balance_orig": ("Sender balance before", "sender balance before the transaction"),
    "new_balance_orig": ("Sender balance after", "sender balance after the transaction"),
    "old_balance_dest": ("Recipient balance before", "recipient balance before the transaction"),
    "new_balance_dest": ("Recipient balance after", "recipient balance after the transaction"),
    "orig_balance_delta": ("Sender balance decrease", "sender balance before minus sender balance after"),
    "dest_balance_delta": ("Recipient balance change", "recipient balance after minus recipient balance before"),
    "orig_amount_delta": ("Sender debit mismatch", "sender balance decrease minus transaction amount"),
    "dest_amount_delta": ("Recipient credit mismatch", "recipient balance increase minus transaction amount"),
    "txn_type": ("Transaction type", "transaction category"),
    "device_os": ("Device operating system", "device operating system"),
}


def format_feature_value(value):
    if value is None:
        return "missing"
    try:
        number = float(value)
        return f"{number:,.2f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def explain_probability(name, features, probability, decision_threshold=None, classification=None):
    try:
        explainer = load_shap_explainer(name)
        shap_features = features.copy()
        for column in ("txn_type", "device_os"):
            reverse = load_shap_category_map(column)
            shap_features[column] = shap_features[column].fillna("UNKNOWN").astype(str).map({value: code for code, value in reverse.items()})
        explanation = explainer(shap_features, max_evals=2 * features.shape[1] + 1)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, f"Could not calculate the SHAP explanation: {exc}") from exc
    values = np.asarray(explanation.values).reshape(-1)
    base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
    contributions = []
    for feature, value, shap_value in zip(features.columns, features.iloc[0].tolist(), values):
        numeric = float(shap_value)
        label, meaning = FEATURE_COPY.get(feature, (feature.replace("_", " ").title(), feature.replace("_", " ")))
        contributions.append({"feature": feature, "label": label, "meaning": meaning,
                              "value": None if pd.isna(value) else str(value),
                              "shap_value": numeric, "direction": "increases" if numeric > 1e-8 else "decreases" if numeric < -1e-8 else "neutral"})
    contributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
    increasing = sorted((item for item in contributions if item["shap_value"] > 1e-8), key=lambda item: item["shap_value"], reverse=True)
    decreasing = sorted((item for item in contributions if item["shap_value"] < -1e-8), key=lambda item: item["shap_value"])
    sample_size = len(load_shap_background())
    probability_text = f"{probability * 100:.2f}%"
    baseline_text = f"{base_value * 100:.2f}%"
    sentences = [f"The {name.replace('_', ' ')} model returned {probability_text}. Its reference score was {baseline_text}, averaged over {sample_size} training transactions."]
    for heading, drivers in (("The strongest score-increasing input", increasing[:2]),
                             ("The strongest score-decreasing input", decreasing[:2])):
        if drivers:
            details = "; ".join(
                f"{item['label']} ({item['meaning']}; input {format_feature_value(item['value'])}) "
                f"{'raised' if item['shap_value'] > 0 else 'lowered'} the score by {abs(item['shap_value']) * 100:.2f} percentage points"
                for item in drivers
            )
            sentences.append(f"{heading}: {details}.")
    if decision_threshold is not None and classification:
        sentences.append(f"The decision threshold is {decision_threshold * 100:.2f}%; this score is {'above' if probability >= decision_threshold else 'below'} it, so the model output is {classification}.")
    sentences.append("These contributions explain the model's response to the supplied values; they are not proof of fraud or causal claims.")
    return {"method": "Permutation SHAP", "output": "fraud probability", "base_value": base_value,
            "prediction_value": float(probability), "contributions": contributions,
            "plain_language": " ".join(sentences),
            "additivity_residual": float(probability - base_value - sum(item["shap_value"] for item in contributions))}


def features_for(tx: dict[str, Any]) -> pd.DataFrame:
    try:
        timestamp = pd.to_datetime(tx["timestamp"], errors="raise")
    except Exception as exc:
        raise HTTPException(422, "timestamp must be a valid date/time") from exc
    txn_type = str(tx["txn_type"]).strip().upper()
    txn_type = {"PAYM3NT":"PAYMENT", "CASH_O0T":"CASH_OUT", "NETBANKING":"NET_BANKING"}.get(txn_type, txn_type)
    if txn_type not in ALLOWED_TYPES:
        raise HTTPException(422, f"txn_type must be one of {sorted(ALLOWED_TYPES)}")
    row = {"timestamp": timestamp, "txn_type": txn_type,
           "amount": tx["amount"], "old_balance_orig": tx["old_balance_orig"],
           "new_balance_orig": tx["new_balance_orig"], "old_balance_dest": tx.get("old_balance_dest"),
           "new_balance_dest": tx.get("new_balance_dest"), "device_os": tx.get("device_os") or "UNKNOWN"}
    derived = {
        "hour": timestamp.hour, "day_of_week": timestamp.dayofweek, "day_of_month": timestamp.day,
        "is_weekend": int(timestamp.dayofweek >= 5), "amount": row["amount"],
        "old_balance_orig": row["old_balance_orig"], "new_balance_orig": row["new_balance_orig"],
        "old_balance_dest": row["old_balance_dest"], "new_balance_dest": row["new_balance_dest"],
        "orig_balance_delta": row["old_balance_orig"] - row["new_balance_orig"],
        "dest_balance_delta": (row["new_balance_dest"] - row["old_balance_dest"])
            if row["new_balance_dest"] is not None and row["old_balance_dest"] is not None else np.nan,
        "orig_amount_delta": row["old_balance_orig"] - row["new_balance_orig"] - row["amount"],
        "dest_amount_delta": (row["new_balance_dest"] - row["old_balance_dest"] - row["amount"])
            if row["new_balance_dest"] is not None and row["old_balance_dest"] is not None else np.nan,
        "txn_type": txn_type, "device_os": row["device_os"],
    }
    return pd.DataFrame([derived])


def predict_one(tx: dict[str, Any], model_name: str, include_explanation: bool = True):
    if model_name not in MODEL_NAMES:
        raise HTTPException(422, "Unknown model")
    path = model_path(model_name)
    if not path.exists():
        raise HTTPException(503, f"The {model_name} artifact is not available; train the models first.")
    features = features_for(tx)
    probability = probability_for(model_name, features)
    evaluation = load_evaluation()
    threshold = float(evaluation["models"][model_name]["threshold"])
    predicted = int(probability >= threshold)
    risk = risk_band(model_name, probability, evaluation)
    classification = "Potentially Fraudulent" if predicted else "Legitimate"
    explanation = explain_probability(model_name, features, probability, threshold, classification) if include_explanation else None
    return {"fraud_probability": probability, "predicted_class": predicted,
            "classification": classification, "classification_threshold": threshold,
            "risk_level": risk,
            "review_priority": {"HIGH": "Immediate review", "MEDIUM": "Review", "LOW": "Routine monitoring"}[risk],
            "risk_policy": evaluation["models"][model_name].get("risk_policy"),
            "model_name": model_name, "explanation": explanation}


def save_prediction(tx, result, source):
    record = {"prediction_id": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc).isoformat(),
              "source": source, "model_name": result["model_name"], "txn_type": tx["txn_type"],
              "amount": tx["amount"], "fraud_probability": result["fraud_probability"],
              "predicted_class": result["predicted_class"], "risk_level": result["risk_level"],
              "explanation_json": json.dumps(result["explanation"]),
              "alert": int(result["predicted_class"] == 1), "input_json": json.dumps(tx)}
    with connect() as con:
        con.execute("INSERT INTO predictions VALUES (:prediction_id,:created_at,:source,:model_name,:txn_type,:amount,:fraud_probability,:predicted_class,:risk_level,:explanation_json,:alert,:input_json)", record)
    return record


def generate_and_store_explanation(prediction_id, features, model_name, probability, threshold, classification):
    try:
        explanation = explain_probability(model_name, features, probability, threshold, classification)
    except Exception as exc:
        explanation = {"error": str(getattr(exc, "detail", exc))}
    with connect() as con:
        con.execute("UPDATE predictions SET explanation_json=? WHERE prediction_id=?",
                    (json.dumps(explanation), prediction_id))


@app.get("/api/health")
def health():
    metrics_ready = (EVALUATION / "metrics.json").exists()
    return {"status": "ok", "models_ready": metrics_ready and all(model_path(name).exists() for name in MODEL_NAMES),
            "deep_learning_lab_ready": metrics_ready and model_path("autoencoder").exists(),
            "history_store": str(HISTORY_DB.name), "training_data_is_separate": True,
            "gemini_ready": bool(os.getenv("GEMINI_API_KEY", "").strip())}


@app.post("/api/predict")
def predict(payload: Transaction, background_tasks: BackgroundTasks, model: str = Query(default="ann")):
    tx = payload.model_dump()
    result = predict_one(tx, model, include_explanation=False)
    record = save_prediction(tx, result, "manual")
    background_tasks.add_task(generate_and_store_explanation, record["prediction_id"],
                              features_for(tx), model, result["fraud_probability"],
                              result["classification_threshold"], result["classification"])
    return {**result, "prediction_id": record["prediction_id"], "created_at": record["created_at"],
            "explanation_pending": True}


@app.get("/api/predictions/{prediction_id}/explanation")
def prediction_explanation(prediction_id: str):
    with connect() as con:
        row = con.execute("SELECT explanation_json FROM predictions WHERE prediction_id=?", (prediction_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Prediction not found")
    explanation = json.loads(row["explanation_json"])
    if explanation is None:
        return {"status": "pending"}
    return {"status": "ready", "explanation": explanation}


@app.post("/api/predictions/{prediction_id}/gemini-explanation")
def gemini_explanation(prediction_id: str):
    """Use Gemini to narrate saved model/SHAP evidence; never alter the score."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(503, "Gemini is not configured. Add GEMINI_API_KEY to the project .env and restart the API.")
    with connect() as con:
        row = con.execute(
            "SELECT explanation_json FROM predictions WHERE prediction_id=?", (prediction_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(404, "Prediction not found")
    explanation = json.loads(row["explanation_json"])
    if not explanation or explanation.get("error") or not explanation.get("contributions"):
        raise HTTPException(409, "Wait for the SHAP explanation to finish before asking Gemini.")

    evidence = {
        "model_probability": explanation["prediction_value"],
        "training_reference_probability": explanation["base_value"],
        "contributions": [
            {"feature": item["label"], "input_value": item["value"],
             "direction": item["direction"], "probability_change": item["shap_value"]}
            for item in explanation["contributions"][:8]
        ],
        "threshold_decision": explanation.get("plain_language", "").split("These contributions explain")[0],
    }
    prompt = (
        "Explain this fraud model output to a nontechnical analyst in 3-5 concise sentences. "
        "Use only the supplied model evidence. Describe probability changes as percentage points. "
        "Explain which listed inputs raised or lowered the score and how the threshold relates to the result. "
        "Do not invent facts, infer intent, claim causation, or call the transaction confirmed fraud. "
        "State that this is a model risk estimate. Evidence (JSON): " + json.dumps(evidence, ensure_ascii=False)
    )
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 220}}).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    timeout = max(20, min(180, int(os.getenv("GEMINI_TIMEOUT_SECONDS", "90"))))
    retryable = {429, 500, 502, 503, 504}
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            text = "".join(part.get("text", "") for part in payload["candidates"][0]["content"]["parts"]).strip()
            if not text:
                raise ValueError("Gemini returned an empty explanation")
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            if exc.code in retryable and attempt < 2:
                retry_after = exc.headers.get("Retry-After", "")
                try:
                    delay = max(0, min(10, int(retry_after)))
                except ValueError:
                    delay = 2 ** attempt
                time.sleep(delay)
                continue
            status_code = 503 if exc.code in (429, 503) else 502
            raise HTTPException(status_code, f"Gemini API request failed after {attempt + 1} attempts ({exc.code}): {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, IndexError) as exc:
            if isinstance(exc, (urllib.error.URLError, TimeoutError)) and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise HTTPException(502, f"Could not generate the Gemini explanation after {attempt + 1} attempts: {exc}") from exc

    explanation["gemini_explanation"] = text
    with connect() as con:
        con.execute("UPDATE predictions SET explanation_json=? WHERE prediction_id=?",
                    (json.dumps(explanation), prediction_id))
    return {"explanation": text}


@app.post("/api/compare")
def compare_models(payload: Transaction):
    tx = payload.model_dump()
    features = features_for(tx)
    evaluation = load_evaluation()
    scores = []
    for name in ("logistic_regression", "random_forest", "xgboost", "ann"):
        if not model_path(name).exists() or name not in evaluation["models"]:
            continue
        probability = probability_for(name, features)
        threshold = float(evaluation["models"][name]["threshold"])
        scores.append({"model": name, "fraud_probability": probability,
                       "threshold": threshold, "classification": "Potentially Fraudulent" if probability >= threshold else "Legitimate",
                       "predicted_class": int(probability >= threshold), "risk_level": risk_band(name, probability, evaluation)})
    ann_score = next((item for item in scores if item["model"] == "ann"), None)
    return {"scores": scores,
            "deep_learning": {"ann": {"score": ann_score["fraud_probability"], "score_kind": "fraud_probability",
                                        "threshold": ann_score["threshold"], "classification": ann_score["classification"],
                                        "risk_level": ann_score["risk_level"]}
                               if ann_score else None,
                              "autoencoder": autoencoder_output(features, evaluation)}}


@app.post("/api/predict/batch")
async def batch_predict(file: UploadFile = File(...), model: str = Query(default="ann")):
    if not file.filename or Path(file.filename).suffix.lower() not in {".csv", ".xlsx"}:
        raise HTTPException(415, "Upload a CSV or XLSX file")
    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(413, "File exceeds the 10 MB limit")
    try:
        frame = pd.read_csv(io.BytesIO(raw)) if file.filename.lower().endswith(".csv") else pd.read_excel(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(422, "Could not read the uploaded table") from exc
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    required = {"timestamp", "txn_type", "amount", "old_balance_orig", "new_balance_orig"}
    if not required.issubset(frame.columns):
        raise HTTPException(422, f"Missing required columns: {sorted(required-frame.columns)}")
    if len(frame) > 10000:
        raise HTTPException(422, "A batch may contain at most 10,000 rows")
    if frame.empty:
        raise HTTPException(422, "The uploaded file contains no rows")
    out = []
    for index, row in frame.iterrows():
        tx = row.where(pd.notna(row), None).to_dict()
        try:
            tx = Transaction.model_validate(tx).model_dump()
            result = predict_one(tx, model, include_explanation=False)
            record = save_prediction(tx, result, "batch")
            out.append({"row": int(index)+1, **result, "prediction_id": record["prediction_id"]})
        except HTTPException as exc:
            out.append({"row": int(index)+1, "error": exc.detail})
        except (TypeError, ValueError, ValidationError) as exc:
            out.append({"row": int(index)+1, "error": str(exc)})
    return {"rows": out, "total": len(out), "successful": sum("error" not in r for r in out)}


@app.get("/api/predictions")
def predictions(page: int = 1, page_size: int = 25, search: str = "", risk: str = "", sort: str = "newest"):
    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    order = "ASC" if sort.lower() == "oldest" else "DESC"
    clauses, args = [], []
    if search:
        clauses.append("(prediction_id LIKE ? OR txn_type LIKE ? OR model_name LIKE ?)")
        args += [f"%{search}%"] * 3
    if risk.upper() in {"LOW", "MEDIUM", "HIGH"}:
        clauses.append("risk_level=?"); args.append(risk.upper())
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with connect() as con:
        total = con.execute("SELECT COUNT(*) FROM predictions"+where, args).fetchone()[0]
        rows = con.execute(f"SELECT * FROM predictions{where} ORDER BY created_at {order} LIMIT ? OFFSET ?", args+[page_size,(page-1)*page_size]).fetchall()
    return {"total": total, "page": page, "page_size": page_size,
            "items": [{**dict(r), "explanation": json.loads(r["explanation_json"]), "input": json.loads(r["input_json"])} for r in rows]}


@app.get("/api/alerts")
def alerts():
    with connect() as con:
        rows = con.execute("SELECT * FROM predictions WHERE predicted_class=1 OR alert=1 "
                           "ORDER BY CASE risk_level WHEN 'HIGH' THEN 0 WHEN 'MEDIUM' THEN 1 ELSE 2 END, "
                           "fraud_probability DESC, created_at DESC LIMIT 500").fetchall()
    return [{**dict(r), "review_priority": {"HIGH": "Immediate review", "MEDIUM": "Review", "LOW": "Routine monitoring"}.get(r["risk_level"], "Review")}
            for r in rows]


@app.get("/api/analytics")
def analytics(model: str = ""):
    model_filter = model.lower() if model.lower() in MODEL_NAMES | {"autoencoder"} else ""
    where = " WHERE model_name=?" if model_filter else ""
    params = [model_filter] if model_filter else []
    with connect() as con:
        total, flagged, avg = con.execute("SELECT COUNT(*),COALESCE(SUM(predicted_class),0),COALESCE(AVG(fraud_probability),0) FROM predictions"+where, params).fetchone()
        risks = {r[0]: r[1] for r in con.execute("SELECT risk_level,COUNT(*) FROM predictions"+where+" GROUP BY risk_level", params)}
        trend = [dict(r) for r in con.execute("SELECT substr(created_at,1,10) day,COUNT(*) count,AVG(fraud_probability) avg_probability FROM predictions"+where+" GROUP BY day ORDER BY day", params)]
        types = [dict(r) for r in con.execute("SELECT txn_type,COUNT(*) count,AVG(fraud_probability) avg_probability FROM predictions"+where+" GROUP BY txn_type", params)]
    return {"total_predictions": total, "potentially_fraudulent": flagged,
            "selected_model": model_filter or "all models",
            "average_probability": avg, "risk_distribution": risks, "trend": trend, "by_type": types}


@app.get("/api/models")
def models():
    data = load_evaluation()
    return {"dataset": data.get("dataset"), "models": data["models"], "data_audit": data.get("data_audit"),
            "features": data.get("features", []), "warning": data.get("warning"),
            "ann_architecture": data.get("ann_architecture"),
            "ann_training": data.get("ann_training"),
            "autoencoder_architecture": data.get("autoencoder_architecture"),
            "autoencoder_training": data.get("autoencoder_training"), "split": data.get("split")}


@app.get("/api/models/history")
def model_history():
    histories = {}
    for name in ("ann", "autoencoder"):
        path = EVALUATION / f"{name}_training_history.json"
        histories[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return histories


@app.get("/api/models/progress")
def training_progress():
    path = EVALUATION / "training_progress.json"
    if not path.exists():
        return {"models": {}, "status": "No training run recorded"}
    return json.loads(path.read_text(encoding="utf-8"))
