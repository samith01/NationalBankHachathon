from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

MLTRAINING_DIR = Path(__file__).resolve().parent
PATCHED_DATASETS_DIR = MLTRAINING_DIR.parents[1] / "datasets" / "patched"
MODEL_PATH = MLTRAINING_DIR / "trader_classifier.json"


def prepare_features(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
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

    feature_cols = [
        col
        for col in work_df.columns
        if col
        not in [
            "timestamp",
            "asset",
            "side",
            "trader_type",
            "profit_loss",
            "exit_price",
            "entry_price",
        ]
    ]

    x_matrix = (
        work_df.select(feature_cols)
        .with_columns(pl.all().cast(pl.Float64, strict=False))
        .fill_null(0.0)
        .to_numpy()
    )
    x_matrix = np.nan_to_num(x_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    y_values = work_df.get_column("trader_type").cast(pl.Int64).to_numpy()
    return x_matrix, y_values


def train_test_split(
    x_values: np.ndarray, y_values: np.ndarray, test_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(x_values))
    rng.shuffle(indices)
    test_size = max(1, int(len(indices) * test_ratio))
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return x_values[train_idx], x_values[test_idx], y_values[train_idx], y_values[test_idx]


def main() -> None:
    calm = pl.read_csv(PATCHED_DATASETS_DIR / "calm_trader.csv").with_columns(
        pl.lit(0).alias("trader_type")
    )
    loss_averse = pl.read_csv(PATCHED_DATASETS_DIR / "loss_averse_trader.csv").with_columns(
        pl.lit(1).alias("trader_type")
    )
    overtrader = pl.read_csv(PATCHED_DATASETS_DIR / "overtrader.csv").with_columns(
        pl.lit(2).alias("trader_type")
    )
    revenge = pl.read_csv(PATCHED_DATASETS_DIR / "revenge_trader.csv").with_columns(
        pl.lit(3).alias("trader_type")
    )

    feature_sets: list[np.ndarray] = []
    label_sets: list[np.ndarray] = []
    for dataset in [calm, loss_averse, overtrader, revenge]:
        dataset_x, dataset_y = prepare_features(dataset)
        feature_sets.append(dataset_x)
        label_sets.append(dataset_y)

    x_matrix = np.vstack(feature_sets)
    y_values = np.concatenate(label_sets)
    x_train, x_test, y_train, y_test = train_test_split(x_matrix, y_values, test_ratio=0.2, seed=42)

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

    params: dict[str, str | int | float] = {
        "objective": "multi:softprob",
        "num_class": 4,
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "alpha": 0.5,
        "lambda": 2.0,
        "min_child_weight": 2,
        "gamma": 1.0,
        "seed": 42,
        "eval_metric": "mlogloss",
    }

    booster = xgb.train(
        params, dtrain, num_boost_round=300, evals=[(dtest, "test")], verbose_eval=False
    )
    probs = booster.predict(dtest)
    preds = np.argmax(probs, axis=1)
    accuracy = float((preds == y_test).mean())
    print(f"Accuracy: {accuracy:.4f}")

    booster.save_model(MODEL_PATH)
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
