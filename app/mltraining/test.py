import sys
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

TRADER_TYPES = {
    0: "calm_trader",
    1: "loss_averse_trader",
    2: "overtrader",
    3: "revenge_trader",
}

MLTRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = MLTRAINING_DIR.parents[1]
DATASETS_DIR = REPO_ROOT / "datasets"
MODEL_PATH = MLTRAINING_DIR / "trader_classifier.json"


def resolve_csv_path(csv_file: str) -> Path:
    candidate = Path(csv_file)
    if candidate.is_file():
        return candidate.resolve()

    repo_relative = REPO_ROOT / candidate
    if repo_relative.is_file():
        return repo_relative.resolve()

    dataset_name = DATASETS_DIR / candidate.name
    if dataset_name.is_file():
        return dataset_name.resolve()

    raise FileNotFoundError(f"Could not find dataset: {csv_file}")


def prepare_features(df: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    work_df = df.with_columns(
        pl.when(pl.col("profit_loss").is_null())
        .then(pl.col("exit_price") - pl.col("entry_price"))
        .otherwise(pl.col("profit_loss"))
        .alias("profit_loss")
    ).drop_nulls(subset=["quantity", "side", "timestamp", "entry_price", "exit_price"])

    if "balance" in work_df.columns:
        work_df = work_df.drop("balance")

    work_df = work_df.with_columns(
        pl.col("timestamp").cast(pl.Utf8).str.strptime(pl.Datetime, strict=False).alias("timestamp")
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
        work_df.get_column("profit_loss").cast(pl.Float64, strict=False).fill_null(0.0).to_list()
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
                qty_after_loss.append(
                    float(quantity_vals[idx]) / (float(quantity_vals[idx - 1]) + 1e-6)
                )
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

    feature_names = [
        col
        for col in work_df.columns
        if col not in ["timestamp", "asset", "side", "profit_loss", "exit_price", "entry_price"]
    ]
    matrix = (
        work_df.select(feature_names)
        .with_columns(pl.all().cast(pl.Float64, strict=False))
        .fill_null(0.0)
        .to_numpy()
    )
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0), feature_names


def predict_trader_type(csv_file: str) -> None:
    csv_path = resolve_csv_path(csv_file)
    model = xgb.Booster(model_file=MODEL_PATH)
    df = pl.read_csv(csv_path)
    features, feature_names = prepare_features(df)

    probabilities = model.predict(xgb.DMatrix(features, feature_names=feature_names))
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(-1, 1)
    predictions = np.argmax(probabilities, axis=1)

    unique, counts = np.unique(predictions, return_counts=True)
    most_common_idx = int(unique[np.argmax(counts)])
    most_common_type = TRADER_TYPES.get(most_common_idx, "unknown")
    confidence = float(np.max(probabilities, axis=1).mean())

    print(f"\n{'=' * 70}")
    print("TRADER TYPE PREDICTION RESULTS")
    print(f"{'=' * 70}")
    print(f"File: {csv_path}")
    print(f"Total samples analyzed: {features.shape[0]}")
    print(f"\nPrimary trader type: {most_common_type}")
    print(f"Confidence: {confidence * 100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test.py <csv_file>")
        sys.exit(1)

    predict_trader_type(sys.argv[1])
