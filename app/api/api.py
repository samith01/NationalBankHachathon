import io
import os
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, time
from typing import AsyncIterator, List, Optional, TypedDict

import numpy as np
import polars as pl
import xgboost as xgb
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    load_model()
    yield


app = FastAPI(title="National Bank Bias Detector API", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trader type mapping
TRADER_TYPES = {
    0: "calm_trader",
    1: "loss_averse_trader",
    2: "overtrader",
    3: "revenge_trader",
}

# In-memory storage for demo purposes
uploaded_files: dict[str, bytes] = {}

ScalarValue = str | float | int | bool | None
MetricValue = str | float | int
TradeRecord = dict[str, ScalarValue]


class MetricsRecord(TypedDict):
    total_trades: int
    win_rate: float
    total_profit_loss: float
    avg_profit_per_trade: float
    max_drawdown: float


class AnalysisSummaryRecord(TypedDict):
    total_trades: int
    win_rate: float
    total_profit_loss: float
    primary_trader_type: str


class AnalysisStoreRecord(TypedDict):
    biases_detected: list[BiasDetectionResult]
    total_trades: int
    win_rate: float
    primary_type: str


class TraderAnalysis(TypedDict):
    type: str
    confidence_score: float
    description: str
    recommendations: list[str]
    all_bias_scores: dict[str, float]


analysis_results: dict[str, AnalysisStoreRecord] = {}


model: xgb.Booster | None = None


def load_model():
    """Load the trained XGBoost model"""
    global model
    try:
        model_path = os.path.join(os.path.dirname(__file__), "../mltraining/trader_classifier.json")
        if os.path.exists(model_path):
            loaded_model = xgb.Booster(model_file=str(model_path))
            model = loaded_model
            print("✓ Model loaded successfully")
        else:
            print(f"⚠ Model not found at {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")


# Pydantic models
class UploadResponse(BaseModel):
    session_id: str
    message: str


class TradeEntry(BaseModel):
    timestamp: str
    asset: str
    side: str
    quantity: Optional[float] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    profit_loss: Optional[float] = None
    balance: Optional[float] = None


class DataResponse(BaseModel):
    session_id: str
    total_records: int
    data: List[TradeEntry]


class RangeDataResponse(DataResponse):
    date_range: dict[str, str]


class BiasDetectionResult(BaseModel):
    type: str
    confidence_score: float
    description: str
    recommendations: List[str]


class AnalysisResponse(BaseModel):
    session_id: str
    biases_detected: List[BiasDetectionResult]
    summary: AnalysisSummaryRecord


# What-if analysis models
class ExcludeCriteria(BaseModel):
    assets: Optional[List[str]] = None
    date_range: Optional[dict[str, str]] = None  # {"start": "2023-01-01", "end": "2023-12-31"}
    min_loss_amount: Optional[float] = None
    max_loss_amount: Optional[float] = None
    trade_ids: Optional[List[int]] = None


class WhatIfRequest(BaseModel):
    exclude_criteria: Optional[ExcludeCriteria] = None
    output_format: str = "timeseries"  # "timeseries", "final_balance", "full_dataset"


class BalancePoint(BaseModel):
    timestamp: str
    original_balance: float
    simulated_balance: float


class WhatIfTimeseriesResponse(BaseModel):
    session_id: str
    simulation_name: str
    original_final_balance: float
    simulated_final_balance: float
    balance_change: float
    balance_timeseries: List[BalancePoint]


class WhatIfFinalBalanceResponse(BaseModel):
    session_id: str
    simulation_name: str
    original_final_balance: float
    simulated_final_balance: float
    balance_improvement: float
    improvement_percentage: float


class SimulatedTradeEntry(TradeEntry):
    included_in_simulation: bool
    simulated_balance: float


class WhatIfFullDatasetResponse(BaseModel):
    session_id: str
    simulation_name: str
    original_trades: int
    included_trades: int
    excluded_trades: int
    dataset: List[SimulatedTradeEntry]


class WhatIfDownloadRequest(BaseModel):
    exclude_criteria: ExcludeCriteria
    report_format: str = "csv"  # "csv", "xlsx"


# Additional models for enhanced features
class BiasSummary(BaseModel):
    bias_type: str
    count: int
    percentage: float


class PerformanceMetrics(BaseModel):
    total_trades: int
    win_rate: float
    total_profit_loss: float
    avg_profit_per_trade: float
    max_drawdown: float


class MetricsResponse(BaseModel):
    session_id: str
    performance_metrics: PerformanceMetrics
    bias_summary: List[BiasSummary]


# Helper functions
def parse_csv_file(file_content: bytes) -> pl.DataFrame:
    """Parse CSV file content into DataFrame"""
    try:
        df = pl.read_csv(io.StringIO(file_content.decode("utf-8")))
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")


def _normalize_value(value: ScalarValue | datetime | date) -> ScalarValue:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return datetime.combine(value, time.min).isoformat()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def timestamp_to_iso(value: ScalarValue | datetime | date) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return datetime.combine(value, time.min).isoformat()
    return str(value)


def optional_float(value: ScalarValue) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = value.strip()
    if not text:
        return None
    return float(text)
    return None


def record_to_trade_entry(record: TradeRecord) -> TradeEntry:
    return TradeEntry(
        timestamp=timestamp_to_iso(record.get("timestamp")),
        asset=str(record.get("asset", "")),
        side=str(record.get("side", "")),
        quantity=optional_float(record.get("quantity")),
        entry_price=optional_float(record.get("entry_price")),
        exit_price=optional_float(record.get("exit_price")),
        profit_loss=optional_float(record.get("profit_loss")),
        balance=optional_float(record.get("balance")),
    )


def dataframe_to_records(df: pl.DataFrame) -> list[TradeRecord]:
    raw_records = df.to_dicts()
    records: list[TradeRecord] = []
    for row in raw_records:
        normalized: TradeRecord = {}
        for key, value in row.items():
            normalized[str(key)] = _normalize_value(value)
        records.append(normalized)
    return records


def get_last_numeric_value(df: pl.DataFrame, column: str, default: float = 0.0) -> float:
    if df.height == 0 or column not in df.columns:
        return default
    value = df.get_column(column).cast(pl.Float64, strict=False).tail(1).item()
    if value is None:
        return default
    return float(value)


def validate_required_columns(df: pl.DataFrame) -> bool:
    """Validate that required columns are present"""
    required_columns = [
        "timestamp",
        "asset",
        "side",
        "quantity",
        "entry_price",
        "exit_price",
        "profit_loss",
        "balance",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_columns}")
    return True


def detect_overtrading(df: pl.DataFrame) -> TraderAnalysis:
    """Detect overtrading bias using ML model"""
    return predict_trader_type_analysis(df)


def predict_trader_type_analysis(df: pl.DataFrame) -> TraderAnalysis:
    """
    Use the trained XGBoost model to predict trader type and return analysis.
    Uses the improved v2 feature set (97.62% accuracy model) matching train_v2.py
    """
    if model is None or len(df) == 0:
        return {
            "type": "Trader Type Prediction",
            "confidence_score": 0.0,
            "description": "Unable to make prediction - model not loaded or insufficient data.",
            "recommendations": [],
            "all_bias_scores": {
                "calm_trader": 0.0,
                "loss_averse_trader": 0.0,
                "overtrader": 0.0,
                "revenge_trader": 0.0,
            },
        }

    try:
        work_df = df.clone()
        work_df = work_df.with_columns(
            pl.when(pl.col("profit_loss").is_null())
            .then(pl.col("exit_price") - pl.col("entry_price"))
            .otherwise(pl.col("profit_loss"))
            .alias("profit_loss")
        )

        critical_cols = ["quantity", "side", "timestamp", "entry_price", "exit_price"]
        work_df = work_df.drop_nulls(subset=critical_cols)
        if "balance" in work_df.columns:
            work_df = work_df.drop("balance")

        work_df = work_df.with_columns(
            pl.col("timestamp")
            .cast(pl.Utf8)
            .str.strptime(pl.Datetime, strict=False)
            .alias("timestamp")
        ).sort("timestamp")

        work_df = work_df.with_columns(
            [
                (pl.col("side") == "BUY").cast(pl.Int64).alias("side_encoded"),
                pl.col("profit_loss").cast(pl.Float64, strict=False).alias("profit_loss_actual"),
                (pl.col("profit_loss").cast(pl.Float64, strict=False) > 0)
                .cast(pl.Int64)
                .alias("is_profit"),
                pl.when(pl.col("profit_loss").cast(pl.Float64, strict=False) < 0)
                .then(pl.col("profit_loss").cast(pl.Float64, strict=False).abs())
                .otherwise(0.0)
                .alias("loss_amount"),
                (
                    pl.col("exit_price").cast(pl.Float64, strict=False)
                    - pl.col("entry_price").cast(pl.Float64, strict=False)
                ).alias("price_range"),
                (
                    (
                        pl.col("exit_price").cast(pl.Float64, strict=False)
                        - pl.col("entry_price").cast(pl.Float64, strict=False)
                    )
                    / (pl.col("entry_price").cast(pl.Float64, strict=False).abs() + 1e-6)
                    * 100
                ).alias("price_range_pct"),
                (
                    pl.col("quantity").cast(pl.Float64, strict=False)
                    * pl.col("entry_price").cast(pl.Float64, strict=False)
                ).alias("trade_value"),
            ]
        )

        for window in [20, 50]:
            work_df = work_df.with_columns(
                [
                    pl.col("is_profit")
                    .cast(pl.Float64)
                    .rolling_mean(window_size=window, min_samples=1)
                    .alias(f"win_rate_{window}"),
                    pl.col("quantity")
                    .cast(pl.Float64, strict=False)
                    .rolling_mean(window_size=window, min_samples=1)
                    .alias(f"avg_qty_{window}"),
                    pl.col("quantity")
                    .cast(pl.Float64, strict=False)
                    .rolling_std(window_size=window, min_samples=1)
                    .fill_null(0.0)
                    .alias(f"qty_volatility_{window}"),
                    pl.col("quantity")
                    .cast(pl.Float64, strict=False)
                    .rolling_max(window_size=window, min_samples=1)
                    .alias(f"qty_max_{window}"),
                    pl.col("profit_loss")
                    .cast(pl.Float64, strict=False)
                    .rolling_mean(window_size=window, min_samples=1)
                    .alias(f"avg_profit_{window}"),
                    pl.col("profit_loss")
                    .cast(pl.Float64, strict=False)
                    .rolling_std(window_size=window, min_samples=1)
                    .fill_null(0.0)
                    .alias(f"profit_std_{window}"),
                    pl.col("profit_loss")
                    .cast(pl.Float64, strict=False)
                    .rolling_min(window_size=window, min_samples=1)
                    .alias(f"max_drawdown_{window}"),
                    pl.col("side_encoded")
                    .cast(pl.Float64)
                    .rolling_mean(window_size=window, min_samples=1)
                    .alias(f"buy_ratio_{window}"),
                ]
            )

        profit_loss_vals = (
            work_df.get_column("profit_loss")
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .to_list()
        )
        quantity_vals = (
            work_df.get_column("quantity").cast(pl.Float64, strict=False).fill_null(0.0).to_list()
        )

        early_close: list[int] = []
        qty_after_loss: list[float] = []
        for idx in range(work_df.height):
            if idx > 0:
                prev_profit = float(profit_loss_vals[idx - 1])
                curr_profit = float(profit_loss_vals[idx])
                early_close.append(1 if (prev_profit > 0 and curr_profit > prev_profit) else 0)
                if prev_profit < 0:
                    qty_ratio = float(quantity_vals[idx]) / (float(quantity_vals[idx - 1]) + 1e-6)
                    qty_after_loss.append(qty_ratio)
                else:
                    qty_after_loss.append(0.0)
            else:
                early_close.append(0)
                qty_after_loss.append(0.0)

        work_df = work_df.with_columns(
            [
                pl.Series("early_close_indicator", early_close),
                pl.Series("qty_after_loss", qty_after_loss),
            ]
        )

        features = [
            col
            for col in work_df.columns
            if col not in ["timestamp", "asset", "side", "profit_loss", "exit_price", "entry_price"]
        ]
        features_matrix = (
            work_df.select(features)
            .with_columns(pl.all().cast(pl.Float64, strict=False))
            .fill_null(0.0)
            .to_numpy()
        )
        features_matrix = np.nan_to_num(features_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        dmatrix = xgb.DMatrix(features_matrix)
        probabilities = model.predict(dmatrix)
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(-1, 1)
        predictions = np.argmax(probabilities, axis=1)

        # Calculate average probability for each trader type across all trades
        avg_probabilities = probabilities.mean(axis=0)

        # Get most common trader type
        unique, counts = np.unique(predictions, return_counts=True)
        most_common_idx = unique[np.argmax(counts)]
        most_common_type = TRADER_TYPES[most_common_idx]

        # Generate recommendations based on trader type
        recommendations_map = {
            "calm_trader": [
                "You show disciplined trading patterns - maintain your consistent approach",
                "Your risk management is solid - continue with current position sizing",
                "Focus on optimizing entry/exit timing for better profit capture",
            ],
            "loss_averse_trader": [
                "You tend to close winning positions too quickly while holding losses",
                "Implement fixed take-profit levels to let winners run",
                "Use stop-loss orders to systematically manage losing trades",
                "Keep detailed trade journal to identify emotional decision patterns",
            ],
            "overtrader": [
                "You're trading too frequently - set daily/weekly trade limits",
                "Implement a cooling-off period between trades to reduce impulsivity",
                "Review transaction costs - they're eating into your profits",
                "Quality over quantity: focus on high-conviction setups only",
            ],
            "revenge_trader": [
                "You show signs of revenge trading after losses - take breaks",
                "Implement a mandatory cooling-off period after consecutive losses",
                "Practice mindfulness and emotional regulation before trading",
                "Use automation to enforce pre-planned risk management rules",
            ],
        }

        trader_type_description = {
            "calm_trader": "You maintain disciplined and consistent trading patterns with good emotional control",
            "loss_averse_trader": "You tend to close winning trades quickly but hold losing positions longer",
            "overtrader": "You trade frequently, potentially due to over-confidence or lack of discipline",
            "revenge_trader": "Your trading shows increased activity and risk after losses, indicating emotional decisions",
        }

        print(
            f"Predicted Trader Type: {most_common_type}, Confidence: {avg_probabilities[most_common_idx]:.2f}"
        )

        # Return results with average probabilities for all bias types
        return {
            "type": f"Trader Type: {most_common_type.replace('_', ' ').title()}",
            "confidence_score": float(avg_probabilities[most_common_idx]),
            "description": trader_type_description.get(
                most_common_type, "Unknown trader type pattern detected"
            ),
            "recommendations": recommendations_map.get(most_common_type, []),
            "all_bias_scores": {
                "calm_trader": float(avg_probabilities[0]),
                "loss_averse_trader": float(avg_probabilities[1]),
                "overtrader": float(avg_probabilities[2]),
                "revenge_trader": float(avg_probabilities[3]),
            },
        }

    except Exception as e:
        return {
            "type": "Trader Type Prediction",
            "confidence_score": 0.0,
            "description": f"Error during prediction: {str(e)}",
            "recommendations": [],
            "all_bias_scores": {
                "calm_trader": 0.0,
                "loss_averse_trader": 0.0,
                "overtrader": 0.0,
                "revenge_trader": 0.0,
            },
        }


def detect_loss_aversion(df: pl.DataFrame) -> TraderAnalysis:
    analysis = predict_trader_type_analysis(df)
    return analysis


def detect_revenge_trading(df: pl.DataFrame) -> TraderAnalysis:
    analysis = predict_trader_type_analysis(df)
    return analysis


def calculate_performance_metrics(df: pl.DataFrame) -> MetricsRecord:
    """Calculate key performance metrics"""
    if df.height == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_profit_loss": 0.0,
            "avg_profit_per_trade": 0.0,
            "max_drawdown": 0.0,
        }

    df_sorted = df.with_columns(
        pl.col("timestamp").cast(pl.Utf8).str.strptime(pl.Datetime, strict=False).alias("timestamp")
    ).sort("timestamp")
    start_balance = get_last_numeric_value(df_sorted.head(1), "balance")
    df_sorted = (
        df_sorted.with_columns(
            pl.col("profit_loss")
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .cum_sum()
            .alias("cumulative_pl")
        )
        .with_columns((pl.col("cumulative_pl") + start_balance).alias("running_balance"))
        .with_columns(pl.col("running_balance").cum_max().alias("peak_balance"))
        .with_columns((pl.col("peak_balance") - pl.col("running_balance")).alias("drawdown"))
    )

    wins = df_sorted.filter(pl.col("profit_loss").cast(pl.Float64, strict=False) > 0).height
    total_pl = float(
        df_sorted.select(
            pl.col("profit_loss").cast(pl.Float64, strict=False).fill_null(0.0).sum()
        ).item()
    )
    avg_pl = float(
        df_sorted.select(
            pl.col("profit_loss").cast(pl.Float64, strict=False).fill_null(0.0).mean()
        ).item()
    )
    max_drawdown = float(df_sorted.select(pl.col("drawdown").max()).item())

    return {
        "total_trades": df.height,
        "win_rate": round(wins / df.height * 100, 2),
        "total_profit_loss": round(total_pl, 2),
        "avg_profit_per_trade": round(avg_pl, 2),
        "max_drawdown": round(max_drawdown, 2),
    }


def get_all_trades(session_id: str) -> list[TradeRecord]:
    """Get all trades for a session"""
    if session_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Session ID not found")

    try:
        content = uploaded_files[session_id]
        df = parse_csv_file(content)
        validate_required_columns(df)

        df = df.with_columns(
            [
                pl.col("timestamp")
                .cast(pl.Utf8)
                .str.strptime(pl.Datetime, strict=False)
                .alias("timestamp"),
                pl.col("side").fill_null("BUY").cast(pl.Utf8),
                pl.col("asset").fill_null("UNKNOWN").cast(pl.Utf8),
            ]
        )

        return dataframe_to_records(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve data: {str(e)}")


def get_trades_in_range(session_id: str, start_date: date, end_date: date) -> list[TradeRecord]:
    """Get trades within a date range"""
    all_trades = get_all_trades(session_id)

    df = pl.DataFrame(all_trades).with_columns(
        pl.col("timestamp").cast(pl.Utf8).str.strptime(pl.Datetime, strict=False).alias("timestamp")
    )
    start_datetime = datetime.combine(start_date, time.min)
    end_datetime = datetime.combine(end_date, time.max)
    filtered_df = df.filter(
        (pl.col("timestamp") >= pl.lit(start_datetime))
        & (pl.col("timestamp") <= pl.lit(end_datetime))
    )
    return dataframe_to_records(filtered_df)


def identify_excluded_trades(df: pl.DataFrame, criteria: ExcludeCriteria) -> List[int]:
    """Identify which trades to exclude based on criteria"""
    indexed = df.with_row_index("row_idx")
    excluded: set[int] = set()

    if criteria.assets:
        excluded.update(indexed.filter(pl.col("asset").is_in(criteria.assets))["row_idx"].to_list())

    if criteria.date_range:
        start_dt = datetime.fromisoformat(criteria.date_range["start"])
        end_dt = datetime.fromisoformat(criteria.date_range["end"])
        excluded.update(
            indexed.with_columns(
                pl.col("timestamp")
                .cast(pl.Utf8)
                .str.strptime(pl.Datetime, strict=False)
                .alias("timestamp")
            )
            .filter(
                (pl.col("timestamp") >= pl.lit(start_dt)) & (pl.col("timestamp") <= pl.lit(end_dt))
            )["row_idx"]
            .to_list()
        )

    if criteria.min_loss_amount is not None:
        excluded.update(
            indexed.filter(
                pl.col("profit_loss").cast(pl.Float64, strict=False) <= criteria.min_loss_amount
            )["row_idx"].to_list()
        )

    if criteria.max_loss_amount is not None:
        excluded.update(
            indexed.filter(
                pl.col("profit_loss").cast(pl.Float64, strict=False) >= criteria.max_loss_amount
            )["row_idx"].to_list()
        )

    if criteria.trade_ids:
        excluded.update(criteria.trade_ids)

    return sorted(excluded)


def calculate_simulated_balances(df: pl.DataFrame, exclude_indices: List[int]) -> pl.DataFrame:
    """Calculate balance progression with excluded trades"""
    start_balance = get_last_numeric_value(df.head(1), "balance")
    simulated_df = (
        df.with_row_index("row_idx")
        .with_columns(
            [
                (~pl.col("row_idx").is_in(exclude_indices)).alias("included_in_simulation"),
                pl.col("timestamp")
                .cast(pl.Utf8)
                .str.strptime(pl.Datetime, strict=False)
                .alias("timestamp"),
            ]
        )
        .sort("timestamp")
        .with_columns(
            pl.when(pl.col("included_in_simulation"))
            .then(pl.col("profit_loss").cast(pl.Float64, strict=False).fill_null(0.0))
            .otherwise(0.0)
            .alias("sim_profit_loss")
        )
        .with_columns(pl.col("sim_profit_loss").cum_sum().alias("simulated_cumulative_pl"))
        .with_columns(
            (pl.col("simulated_cumulative_pl") + start_balance).alias("simulated_balance")
        )
        .drop("row_idx")
    )
    return simulated_df


def generate_simulation_name(criteria: ExcludeCriteria) -> str:
    """Generate descriptive name for simulation"""
    if criteria.assets:
        return f"Exclude {', '.join(criteria.assets)} trades"
    elif criteria.min_loss_amount is not None:
        return f"Exclude losses below ${criteria.min_loss_amount}"
    elif criteria.max_loss_amount is not None:
        return f"Exclude gains above ${criteria.max_loss_amount}"
    elif criteria.date_range:
        return f"Exclude trades from {criteria.date_range['start']} to {criteria.date_range['end']}"
    elif criteria.trade_ids:
        return f"Exclude {len(criteria.trade_ids)} specific trades"
    else:
        return "Custom what-if simulation"


@app.post(
    "/upload/trade-history",
    response_model=UploadResponse,
    summary="Upload Trading History",
    description="Upload a CSV file containing trading history for analysis",
)
async def upload_trade_history(file: UploadFile = File(...)):
    """
    Upload trading history CSV file
    Required columns: timestamp, asset, side, quantity, entry_price, exit_price, profit_loss, balance
    """
    session_id = str(uuid.uuid4())

    # Read file content
    content = await file.read()

    # Store file content (in production, save to disk/database)
    uploaded_files[session_id] = content

    return UploadResponse(
        session_id=session_id,
        message=f"Trade history uploaded successfully. Session ID: {session_id}",
    )


@app.get(
    "/data/{session_id}",
    response_model=DataResponse,
    summary="Get All Trading Data",
    description="Retrieve all trading records from the uploaded file for a given session",
)
async def get_all_trading_data(session_id: str):
    """
    Retrieve all trading records from the uploaded file for a given session
    """
    trades = get_all_trades(session_id)

    trade_entries = [record_to_trade_entry(trade) for trade in trades]

    return DataResponse(
        session_id=session_id,
        total_records=len(trade_entries),
        data=trade_entries,
    )


@app.get(
    "/data/{session_id}/range",
    response_model=RangeDataResponse,
    summary="Get Trading Data by Date Range",
    description="Fetch trading records within a specified date range for a given session",
)
async def get_trading_data_by_range(
    session_id: str,
    start_date: date = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: date = Query(..., description="End date in YYYY-MM-DD format"),
):
    """
    Fetch trading records within a specified date range for a given session
    """
    if start_date > end_date:
        raise HTTPException(
            status_code=400, detail="Start date must be before or equal to end date"
        )

    trades = get_trades_in_range(session_id, start_date, end_date)

    trade_entries = [record_to_trade_entry(trade) for trade in trades]

    return RangeDataResponse(
        session_id=session_id,
        total_records=len(trade_entries),
        date_range={"start": start_date.isoformat(), "end": end_date.isoformat()},
        data=trade_entries,
    )


@app.get(
    "/metrics/{session_id}",
    response_model=MetricsResponse,
    summary="Get Performance Metrics",
    description="Calculate key performance metrics for the trading history",
)
async def get_performance_metrics(session_id: str):
    """
    Calculate key performance metrics for the trading history
    """
    if session_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Session ID not found")

    try:
        # Parse the uploaded file
        content = uploaded_files[session_id]
        df = parse_csv_file(content)
        validate_required_columns(df)

        # Calculate performance metrics
        metrics = calculate_performance_metrics(df)

        # Detect trader type using ML model and generate bias summary
        trader_analysis = predict_trader_type_analysis(df)
        trader_confidence = trader_analysis["confidence_score"]

        # Extract trader type from description
        trader_type_str = trader_analysis["type"].replace("Trader Type: ", "")

        bias_summary = [
            BiasSummary(
                bias_type=trader_type_str,
                count=df.height,
                percentage=trader_confidence * 100,
            )
        ]

        return MetricsResponse(
            session_id=session_id,
            performance_metrics=PerformanceMetrics(**metrics),
            bias_summary=bias_summary,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics calculation failed: {str(e)}")


@app.post(
    "/what-if/{session_id}/simulate",
    summary="Run What-If Simulation",
    description="Calculate alternative balance history by excluding specified trades",
)
async def what_if_simulation(session_id: str, request: WhatIfRequest):
    """
    Calculate alternative balance history by excluding specified trades
    Output formats: timeseries, final_balance, full_dataset
    """
    if session_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Session ID not found")

    try:
        # Parse the uploaded file
        content = uploaded_files[session_id]
        df = parse_csv_file(content)
        validate_required_columns(df)

        # Identify trades to exclude
        exclude_criteria = request.exclude_criteria or ExcludeCriteria()
        exclude_indices = identify_excluded_trades(df, exclude_criteria)

        # Calculate simulated balances
        simulated_df = calculate_simulated_balances(df, exclude_indices)

        # Generate simulation name
        simulation_name = generate_simulation_name(exclude_criteria)

        # Format response based on output_format
        if request.output_format == "timeseries":
            # Prepare timeseries data
            rows = dataframe_to_records(simulated_df)
            timeseries_data = [
                BalancePoint(
                    timestamp=timestamp_to_iso(row["timestamp"]),
                    original_balance=optional_float(row["balance"]) or 0.0,
                    simulated_balance=optional_float(row["simulated_balance"]) or 0.0,
                )
                for row in rows
            ]

            original_final_balance = get_last_numeric_value(df, "balance")
            simulated_final_balance = get_last_numeric_value(simulated_df, "simulated_balance")

            return WhatIfTimeseriesResponse(
                session_id=session_id,
                simulation_name=simulation_name,
                original_final_balance=original_final_balance,
                simulated_final_balance=simulated_final_balance,
                balance_change=simulated_final_balance - original_final_balance,
                balance_timeseries=timeseries_data,
            )

        elif request.output_format == "final_balance":
            original_balance = get_last_numeric_value(df, "balance")
            simulated_balance = get_last_numeric_value(simulated_df, "simulated_balance")
            improvement = simulated_balance - original_balance
            improvement_pct = (improvement / original_balance * 100) if original_balance != 0 else 0

            return WhatIfFinalBalanceResponse(
                session_id=session_id,
                simulation_name=simulation_name,
                original_final_balance=original_balance,
                simulated_final_balance=simulated_balance,
                balance_improvement=improvement,
                improvement_percentage=round(improvement_pct, 2),
            )

        elif request.output_format == "full_dataset":
            # Prepare full dataset with simulation info
            rows = dataframe_to_records(simulated_df)
            full_dataset = [
                SimulatedTradeEntry(
                    timestamp=timestamp_to_iso(row["timestamp"]),
                    asset=str(row["asset"]),
                    side=str(row["side"]),
                    quantity=optional_float(row["quantity"]),
                    entry_price=optional_float(row["entry_price"]),
                    exit_price=optional_float(row["exit_price"]),
                    profit_loss=optional_float(row["profit_loss"]),
                    balance=optional_float(row["balance"]),
                    included_in_simulation=bool(row["included_in_simulation"]),
                    simulated_balance=optional_float(row["simulated_balance"]) or 0.0,
                )
                for row in rows
            ]

            return WhatIfFullDatasetResponse(
                session_id=session_id,
                simulation_name=simulation_name,
                original_trades=df.height,
                included_trades=df.height - len(exclude_indices),
                excluded_trades=len(exclude_indices),
                dataset=full_dataset,
            )

        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid output_format. Must be 'timeseries', 'final_balance', or 'full_dataset'",
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"What-if simulation failed: {str(e)}")


@app.post(
    "/what-if/{session_id}/download",
    summary="Download What-If Report",
    description="Generate downloadable report of what-if simulation",
)
async def download_what_if_report(session_id: str, request: WhatIfDownloadRequest):
    """
    Generate downloadable report of what-if simulation
    Report formats: csv, xlsx
    """
    if session_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Session ID not found")

    try:
        # Parse the uploaded file
        content = uploaded_files[session_id]
        df = parse_csv_file(content)
        validate_required_columns(df)

        # Identify trades to exclude
        exclude_indices = identify_excluded_trades(df, request.exclude_criteria)

        # Calculate simulated balances
        simulated_df = calculate_simulated_balances(df, exclude_indices)

        if request.report_format == "csv":
            # Convert to CSV
            csv_data = simulated_df.write_csv()
            headers = {
                "Content-Disposition": 'attachment; filename="what_if_analysis.csv"',
                "Content-Type": "text/csv",
            }
            return Response(content=csv_data, headers=headers)

        elif request.report_format == "xlsx":
            raise HTTPException(
                status_code=400,
                detail="XLSX export is not available in the Polars-only pipeline. Use CSV.",
            )

        else:
            raise HTTPException(
                status_code=400, detail="Invalid report_format. Must be 'csv' or 'xlsx'"
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@app.get(
    "/analyze/{session_id}",
    response_model=AnalysisResponse,
    summary="Analyze Trading Behavior",
    description="Analyze uploaded trading history for behavioral biases",
)
async def analyze_trading_history(session_id: str):
    """
    Analyze uploaded trading history for behavioral biases using ML model
    """
    if session_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Session ID not found")

    try:
        # Parse the uploaded file
        content = uploaded_files[session_id]
        df = parse_csv_file(content)

        # Validate required columns
        validate_required_columns(df)

        # Detect trader type using ML model
        trader_type_analysis = predict_trader_type_analysis(df)

        # Extract all bias scores
        all_bias_scores = trader_type_analysis["all_bias_scores"]

        # Create individual bias detection results for each type
        bias_descriptions = {
            "calm_trader": "Disciplined and consistent trading patterns with good emotional control",
            "loss_averse_trader": "Tendency to close winning trades quickly but hold losing positions longer",
            "overtrader": "Frequent trading, potentially due to over-confidence or lack of discipline",
            "revenge_trader": "Increased activity and risk after losses, indicating emotional decisions",
        }

        bias_recommendations = {
            "calm_trader": [
                "Maintain your consistent approach",
                "Continue with current position sizing",
                "Optimize entry/exit timing for better profit capture",
            ],
            "loss_averse_trader": [
                "Implement fixed take-profit levels to let winners run",
                "Use stop-loss orders to systematically manage losing trades",
                "Keep detailed trade journal to identify emotional decision patterns",
            ],
            "overtrader": [
                "Set daily/weekly trade limits",
                "Implement a cooling-off period between trades",
                "Focus on high-conviction setups only",
            ],
            "revenge_trader": [
                "Take breaks after consecutive losses",
                "Implement a mandatory cooling-off period",
                "Practice mindfulness and emotional regulation before trading",
            ],
        }

        biases_detected: list[BiasDetectionResult] = []

        # Add results for all bias types with their ML-predicted scores
        for bias_key, score in all_bias_scores.items():
            biases_detected.append(
                BiasDetectionResult(
                    type=bias_key.replace("_", " ").title(),
                    confidence_score=float(score),
                    description=bias_descriptions.get(bias_key, "Unknown bias pattern"),
                    recommendations=bias_recommendations.get(bias_key, []),
                )
            )

        # Store results
        analysis_results[session_id] = {
            "biases_detected": biases_detected,
            "total_trades": df.height,
            "win_rate": (
                df.filter(pl.col("profit_loss").cast(pl.Float64, strict=False) > 0).height
                / df.height
                if df.height > 0
                else 0.0
            ),
            "primary_type": trader_type_analysis["type"],
        }

        total_profit_loss = float(
            df.select(
                pl.col("profit_loss").cast(pl.Float64, strict=False).fill_null(0.0).sum()
            ).item()
        )
        win_count = df.filter(pl.col("profit_loss").cast(pl.Float64, strict=False) > 0).height

        return AnalysisResponse(
            session_id=session_id,
            biases_detected=biases_detected,
            summary={
                "total_trades": df.height,
                "win_rate": round(win_count / df.height * 100, 2) if df.height > 0 else 0,
                "total_profit_loss": round(total_profit_loss, 2) if df.height > 0 else 0,
                "primary_trader_type": trader_type_analysis["type"],
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get(
    "/health",
    summary="Health Check",
    description="Check if the API is running properly",
)
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
