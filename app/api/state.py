import os

import xgboost as xgb

from .schemas import AnalysisStoreRecord, CsvProcessingSummary

TRADER_TYPES = {
    0: "calm_trader",
    1: "loss_averse_trader",
    2: "overtrader",
    3: "revenge_trader",
}

uploaded_files: dict[str, bytes] = {}
analysis_results: dict[str, AnalysisStoreRecord] = {}
csv_processing_summaries: dict[str, CsvProcessingSummary] = {}
model: xgb.Booster | None = None


def load_model() -> None:
    global model
    try:
        model_path = os.path.join(os.path.dirname(__file__), "../mltraining/trader_classifier.json")
        if os.path.exists(model_path):
            loaded_model = xgb.Booster(model_file=str(model_path))
            model = loaded_model
            print("Model loaded successfully")
        else:
            print(f"Model not found at {model_path}")
    except Exception as exc:
        print(f"Error loading model: {exc}")
