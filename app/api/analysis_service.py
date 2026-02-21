import numpy as np
import polars as pl
import xgboost as xgb

from . import state
from .data_service import get_last_numeric_value
from .schemas import BiasDetectionResult, MetricsRecord, TraderAnalysis

RECOMMENDATIONS_MAP = {
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

BIAS_DESCRIPTIONS = {
    "calm_trader": "Disciplined and consistent trading patterns with good emotional control",
    "loss_averse_trader": "Tendency to close winning trades quickly but hold losing positions longer",
    "overtrader": "Frequent trading, potentially due to over-confidence or lack of discipline",
    "revenge_trader": "Increased activity and risk after losses, indicating emotional decisions",
}


def detect_overtrading(df: pl.DataFrame) -> TraderAnalysis:
    return predict_trader_type_analysis(df)


def detect_loss_aversion(df: pl.DataFrame) -> TraderAnalysis:
    return predict_trader_type_analysis(df)


def detect_revenge_trading(df: pl.DataFrame) -> TraderAnalysis:
    return predict_trader_type_analysis(df)


def predict_trader_type_analysis(df: pl.DataFrame) -> TraderAnalysis:
    if state.model is None or len(df) == 0:
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
        probabilities = state.model.predict(dmatrix)
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(-1, 1)
        predictions = np.argmax(probabilities, axis=1)

        avg_probabilities = probabilities.mean(axis=0)
        unique, counts = np.unique(predictions, return_counts=True)
        most_common_idx = unique[np.argmax(counts)]
        most_common_type = state.TRADER_TYPES[most_common_idx]

        return {
            "type": f"Trader Type: {most_common_type.replace('_', ' ').title()}",
            "confidence_score": float(avg_probabilities[most_common_idx]),
            "description": BIAS_DESCRIPTIONS.get(
                most_common_type, "Unknown trader type pattern detected"
            ),
            "recommendations": RECOMMENDATIONS_MAP.get(most_common_type, []),
            "all_bias_scores": {
                "calm_trader": float(avg_probabilities[0]),
                "loss_averse_trader": float(avg_probabilities[1]),
                "overtrader": float(avg_probabilities[2]),
                "revenge_trader": float(avg_probabilities[3]),
            },
        }

    except Exception as exc:
        return {
            "type": "Trader Type Prediction",
            "confidence_score": 0.0,
            "description": f"Error during prediction: {str(exc)}",
            "recommendations": [],
            "all_bias_scores": {
                "calm_trader": 0.0,
                "loss_averse_trader": 0.0,
                "overtrader": 0.0,
                "revenge_trader": 0.0,
            },
        }


def calculate_performance_metrics(df: pl.DataFrame) -> MetricsRecord:
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


def build_bias_detection_results(all_bias_scores: dict[str, float]) -> list[BiasDetectionResult]:
    compact_recommendations = {
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

    return [
        BiasDetectionResult(
            type=bias_key.replace("_", " ").title(),
            confidence_score=float(score),
            description=BIAS_DESCRIPTIONS.get(bias_key, "Unknown bias pattern"),
            recommendations=compact_recommendations.get(bias_key, []),
        )
        for bias_key, score in all_bias_scores.items()
    ]
