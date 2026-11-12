# FraudLens

FraudLens is a local academic demonstration for financial transaction fraud detection. It uses the supplied `messy_fraud_dataset_20k.csv` plus a reproducible set of 2,000 clearly labeled scenario-simulated fraud records, then predicts a model score and configurable LOW/MEDIUM/HIGH risk band for a manually entered transaction or CSV/XLSX batch.

> The dataset is synthetic, contains only 120 positive cases, spans a short period, and has no fraud-type labels. Results are educational and must not be presented as evidence of production performance or used as a sole basis for financial decisions.

## Dataset findings

The original raw CSV has 20,000 rows and 12 columns, no fully duplicated rows, and a binary `is_fraud` target (19,880 legitimate / 120 fraud; 0.6% fraud). The training run adds 2,000 reproducible, scenario-based simulated fraud rows (account takeover, balance drain, cash-out, and destination mismatch) saved separately to `data/synthetic/simulated_fraud_2k.csv`; it never edits the supplied raw CSV. These are generated examples, not observed real-world transactions. The combined 22,000-row training/evaluation set has about 9.6% positive labels before partitioning. The added positive examples make evaluation more informative, while balanced accuracy, precision, recall, and PR-AUC are reported alongside ordinary accuracy. The detailed inspection and feature assessment are in [DATASET_INSPECTION.md](DATASET_INSPECTION.md).

No fraud typology is exported. The generator source includes private synthetic-generation metadata, but the model does not use it and the application does not claim to identify account takeover or any other fraud type.

## Architecture and data separation

```text
messy_fraud_dataset_20k.csv (raw, immutable) + data/synthetic/simulated_fraud_2k.csv
          ↓ offline cleaning / training / evaluation
data/processed + trained_models + evaluation
          ↓ inference API
React frontend → FastAPI → separate data/prediction_history.db
```

Training data and prediction history are kept separate. The training CSV and cleaned training output are never used as the inference history store. Manual/batch inference records are stored only in `data/prediction_history.db` (overridable with `FRAUDLENS_HISTORY_DB`). The history database is not used to train models.

## Data science pipeline

`data_science/cleaning.py` parses the observed timestamps, canonicalizes known transaction type variants, preserves rows, and derives timestamp and balance-change features. Identifier columns (`txn_id`, `customer_id`, `device_ip`) are excluded from model inputs. Numeric/categorical preprocessing is fitted within each model pipeline on training data only. Customer-grouped, stratified partitions split original rows before any simulations are generated, using seed 42 so a customer cannot appear in multiple original partitions. Simulated positives are generated only from legitimate original training rows. Decision and risk cutoffs are selected using original-only validation rows; final metrics and curves come from the held-out test partition.

The pipeline compares Logistic Regression, Random Forest, XGBoost, and an offline-trained ANN, and trains a reconstruction autoencoder on legitimate training rows only. The 2,000 simulated positives are partitioned by unique synthetic customer IDs, so validation and test include held-out simulated scenarios; model reports also keep original-only and simulated-only test summaries separate. Class weighting / `scale_pos_weight` is applied within training. Each simulated positive is downweighted so all simulated training positives together have the same total sample weight as the original positive training rows; this prevents the generated scenarios from overwhelming the much smaller original fraud class. It saves actual held-out metrics and validation-selected risk-band evidence to `evaluation/metrics.json`, model artifacts to `trained_models/`, and ANN/autoencoder epoch histories to `evaluation/`. The UI leads with balanced accuracy and rare-class metrics because ordinary accuracy can still hide missed fraud. The training progress file is updated at each neural-network epoch for the Deep Learning Lab.

The held-out split is generated from the fixed seed during training; its exact size and positive count are recorded in `evaluation/metrics.json`. Use the current saved report for metrics rather than copying stale example numbers into a presentation.

The original dataset and scenario-simulated examples are both synthetic. Metrics describe this generated evaluation distribution only and should not be interpreted as expected real-world performance.

## Installation (Windows PowerShell)

Use Python 3.13 or another supported Python 3 version.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r backend\requirements.txt
```

Train models from the project root:

```powershell
python data_science\train.py
```

Run the API in one terminal:

```powershell
uvicorn backend.app:app --reload --port 8000
```

Run the frontend in another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite local URL (usually `http://localhost:5173`). Before model artifacts exist, API health and prediction history remain available, but inference and model evaluation return a clear unavailable/not-trained response.

After installation and the frontend build, start both local services in the background with `powershell -ExecutionPolicy Bypass -File .\start-fraudlens.ps1`; stop the launched services with `powershell -ExecutionPolicy Bypass -File .\stop-fraudlens.ps1`. The launcher uses Vite preview so the served dashboard does not depend on the development-server WebSocket.

## Environment variables

Copy `.env.example` to `.env` and configure the environment values in the shell that starts the API. Risk bands are selected per model from validation predictions: LOW is below the F1-selected review threshold; MEDIUM begins at that threshold; HIGH begins at the highest-recall cutoff meeting at least 80% precision among original-only validation transactions and supported by at least 10 flagged original validation transactions. If no HIGH cutoff meets both criteria, the HIGH band is disabled for that model. These cutoffs are then evaluated on the untouched test split and reported with the count and observed fraud rate in each band. In the current split there are only 19 original fraud cases for validation threshold selection and 20 for the original-only test, so the operating estimates are highly sensitive to individual cases and split choice. They are review-priority bands, not calibrated probabilities or proof of fraud.

## API overview

- `GET /api/health` — service/model readiness and confirmation of separate history storage.
- `POST /api/predict?model=ann` — score one transaction and save its record in prediction history.
- `POST /api/predict/batch?model=ann` — validate and score CSV/XLSX (10 MB / 10,000 row limits).
- `GET /api/predictions` — filterable/paginated stored prediction records.
- `GET /api/alerts` — high-risk stored records.
- `GET /api/analytics` — aggregates computed from prediction history.
- `GET /api/models` and `/api/models/history` — actual model evaluation and ANN training data.
- `GET /api/models/progress` — live epoch progress from the training process.

FastAPI's interactive API documentation is available at `http://localhost:8000/docs` while the backend is running.

## Explainability and limitations

The single-transaction analyzer reports Permutation SHAP contributions in fraud-probability space. It shows the average probability over a small sample of training rows, then signed percentage-point feature contributions that add to the selected model's score. Batch scoring omits SHAP to keep large uploads responsive. Contributions describe model behavior and do not imply causation. Missing model files or SHAP reference artifacts result in a 503 response instead of fabricated predictions. The project does not use real people's financial information. Before real use, the system would need external validation, stronger privacy/security controls, access control, audit policy, drift monitoring, and review by qualified risk teams.

The analyzer also has an optional **Explain with Gemini** action. Set `GEMINI_API_KEY` in the project `.env` (see `.env.example`) and restart the API. Gemini is called only when the user clicks the action; it receives the model score and up to eight leading SHAP factors with their input values. The generated text is saved with that prediction in the separate prediction-history database. Gemini can clarify the evidence but never changes the model probability or decision. The UI discloses the external data transfer before the action.

The **Deep Learning Lab** presents actual ANN and autoencoder training curves, epoch progress, architectures, and held-out evaluation metrics. It also compares both outputs for the latest analyzed transaction: ANN fraud probability and autoencoder reconstruction-error percentile are different signals, not equivalent probabilities. The autoencoder preprocessing and weights are fitted using legitimate training rows only; the anomaly cutoff is selected on validation data.

## Project layout

```text
backend/                 FastAPI inference, uploads, analytics, history API
data_science/            schema-aware cleaning and offline model training
data/processed/          generated cleaned training data
data/prediction_history.db  runtime inference history, separate from training data
trained_models/shap_background.csv  compact training-only reference sample for SHAP
evaluation/              generated actual metrics and ANN history
frontend/                Vite + React dashboard
notebooks/               EDA, preprocessing, and training notebooks
trained_models/          generated fitted pipelines and ANN artifacts
messy_fraud_dataset_20k.csv  fixed, unchanged raw dataset
seed.py                  supplied synthetic data generator
```
#   f i n a n c i a l - f r a u d - d l  
 