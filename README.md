# Sentinel

Sentinel is a hackathon project for detecting trading behavior patterns from trade history.

It has:
- a React + TypeScript frontend (`webapp/`)
- a FastAPI backend (`app/api/`)
- an XGBoost model loaded at API startup (`app/mltraining/trader_classifier.json`)

There is no separate rules engine service in this codebase.

## Demo

[![Sentinel Demo](https://img.youtube.com/vi/U0l-6MOTziM/0.jpg)](https://www.youtube.com/watch?v=U0l-6MOTziM)

> ▶ [Watch on YouTube](https://www.youtube.com/watch?v=U0l-6MOTziM)

## What The App Does Today

### Frontend (`webapp/`)
- Upload flow from the UI (file chooser accepts `.csv`, `.xls`, `.xlsx`)
- Calls backend endpoints to upload data and fetch analysis/results
- Displays:
  - behavioral profile and bias scores
  - cumulative P/L chart
  - trade heatmap
  - coaching recommendations
  - local saved session history (stored in `localStorage`)

### Backend (`app/api/`)
- Loads an XGBoost model on startup
- Stores uploaded datasets in memory by `session_id`
- Sanitizes and repairs CSV numeric gaps where possible (`app/api/csv_sanitizer.py`)
- Exposes endpoints for upload, metrics, analysis, data retrieval, what-if simulation, and health check

### ML
- Training script: `app/mltraining/train.py`
- Inference path: `app/api/analysis_service.py`
- Model artifacts: `app/mltraining/trader_classifier.json` and `app/mltraining/trader_classifier95.json`

## API Endpoints

- `POST /upload/trade-history`
- `GET /data/{session_id}`
- `GET /data/{session_id}/range`
- `GET /metrics/{session_id}`
- `GET /analyze/{session_id}`
- `POST /what-if/{session_id}/simulate`
- `POST /what-if/{session_id}/download`
- `GET /health`

## Tech Stack

### Backend
- Python 3.14+
- FastAPI
- Polars
- NumPy
- XGBoost

### Frontend
- React 19
- TypeScript
- Vite
- PapaParse
- SheetJS (`xlsx`)
- Custom CSS (not Tailwind)

## Project Structure

```text
NationalBankHachathon/
├── app/
│   ├── api/
│   │   ├── analysis_service.py
│   │   ├── csv_sanitizer.py
│   │   ├── data_service.py
│   │   ├── main.py
│   │   ├── routes.py
│   │   ├── schemas.py
│   │   └── state.py
│   ├── mltraining/
│   │   ├── train.py
│   │   ├── test.py
│   │   ├── trader_classifier.json
│   │   └── trader_classifier95.json
│   ├── Dockerfile
│   └── pyproject.toml
├── datasets/
├── webapp/
│   ├── src/
│   ├── Dockerfile
│   └── package.json
├── docker-compose.yml
└── README.md
```

## Run With Docker

```bash
docker compose up -d --build
docker compose ps
```

Default ports from `docker-compose.yml`:
- backend: `http://localhost:8001` (container port `8000`)
- frontend: `http://localhost:5174` (container port `5173`)

Stop services:

```bash
docker compose down
```

Override ports:

```bash
BACKEND_PORT=8000 FRONTEND_PORT=5173 docker compose up -d --build
```

## Run Locally (Without Docker)

### Backend

```bash
cd app
uv sync
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd webapp
npm install
npm run dev
```

## Important Notes

- The frontend API URL is read from `VITE_API_BASE_URL` in `webapp/src/lib/api.ts`; if you pass only a number, it is treated as a local port (`http://localhost:<port>`).
- The backend upload endpoint expects CSV content (`/upload/trade-history` decodes as UTF-8 CSV).
- The what-if download endpoint currently supports CSV output only; XLSX returns an error by design.

## Sample Data

Sample datasets are in `datasets/` (for example `calm_trader.csv`, `overtrader.csv`, `loss_averse_trader.csv`, `revenge_trader.csv`, and mixed profiles).

## License

MIT
