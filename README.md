# FraudLens

FraudLens is a local academic demonstration for financial transaction fraud detection. It scores a manually entered transaction or a CSV/XLSX batch and assigns a configurable LOW/MEDIUM/HIGH risk band using trained machine-learning models.

> Results are educational and must not be presented as evidence of production performance or used as a sole basis for financial decisions.

## Architecture

```text
React frontend → FastAPI API → trained model artifacts
                    ↓
          prediction_history.db (separate inference history store)
```

Training data and prediction history are kept separate. Manual and batch inference records are stored only in `data/prediction_history.db` (overridable with `FRAUDLENS_HISTORY_DB`). The history database is never used to train models.

## Data science pipeline

The offline training pipeline (`data_science/train.py`) cleans transaction records, derives features, and trains comparable classifiers:

- Logistic Regression
- Random Forest
- XGBoost
- ANN (neural network)
- Reconstruction autoencoder trained on legitimate rows only

Each model pipeline fits preprocessing on training data only. Class weighting is applied during training. The pipeline reports balanced accuracy, precision, recall, F1, ROC-AUC, and PR-AUC, and saves held-out metrics to `evaluation/metrics.json`, model artifacts to `trained_models/`, and ANN/autoencoder training histories to `evaluation/`.

Risk bands are selected per model from validation predictions: LOW is below the F1-selected review threshold; MEDIUM begins at that threshold; HIGH begins at the highest-recall cutoff meeting at least 80% precision among original-only validation transactions and supported by at least 10 flagged original validation transactions. If no HIGH cutoff meets both criteria, the HIGH band is disabled for that model. These cutoffs are then evaluated on the untouched test split and reported with the count and observed fraud rate in each band. They are review-priority bands, not calibrated probabilities or proof of fraud.

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

Copy `.env.example` to `.env` and configure the environment values in the shell that starts the API.

| Variable | Description |
|---|---|
| `FRAUDLENS_HISTORY_DB` | Path to the SQLite prediction-history database (default: `data/prediction_history.db`) |
| `CORS_ORIGINS` | Allowed CORS origins for the API |
| `GEMINI_API_KEY` | Optional key for the "Explain with Gemini" action |
| `GEMINI_MODEL` | Gemini model name (default: `gemini-2.5-flash`) |
| `GEMINI_TIMEOUT_SECONDS` | Gemini request timeout in seconds (default: `90`) |

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

The single-transaction analyzer reports Permutation SHAP contributions in fraud-probability space. It shows the average probability over a small sample of training rows, then signed percentage-point feature contributions that add to the selected model's score. Batch scoring omits SHAP to keep large uploads responsive. Contributions describe model behavior and do not imply causation. Missing model files or SHAP reference artifacts result in a 503 response instead of fabricated predictions.

The analyzer also has an optional **Explain with Gemini** action. Set `GEMINI_API_KEY` in the project `.env` and restart the API. Gemini is called only when the user clicks the action; it receives the model score and up to eight leading SHAP factors with their input values. The generated text is saved with that prediction in the separate prediction-history database. Gemini can clarify the evidence but never changes the model probability or decision. The UI discloses the external data transfer before the action.

The **Deep Learning Lab** presents actual ANN and autoencoder training curves, epoch progress, architectures, and held-out evaluation metrics. It also compares both outputs for the latest analyzed transaction: ANN fraud probability and autoencoder reconstruction-error percentile are different signals, not equivalent probabilities. The autoencoder preprocessing and weights are fitted using legitimate training rows only; the anomaly cutoff is selected on validation data.

## Project layout

```text
backend/                 FastAPI inference, uploads, analytics, history API
data_science/            schema-aware cleaning and offline model training
data/processed/          generated cleaned training data
data/prediction_history.db  runtime inference history, separate from training data
trained_models/          generated fitted pipelines and ANN artifacts
evaluation/              generated actual metrics and ANN history
frontend/                Vite + React dashboard
notebooks/               EDA, preprocessing, and training notebooks
runtime/                 runtime logs and PID files
requirements.txt         Python dependencies
backend/requirements.txt  FastAPI and API dependencies
start-fraudlens.ps1      PowerShell launcher for API and frontend
stop-fraudlens.ps1       PowerShell script to stop launched services
.env.example             example environment configuration
```

## License

This project is an academic demonstration and is not intended for production use.
