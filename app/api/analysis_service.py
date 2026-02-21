import numpy as np
import polars as pl
import xgboost as xgb

from . import state
from .data_service import get_last_numeric_value
from .schemas import (
    BiasDetectionResult,
    FrontendPayload,
    HeatmapModeData,
    HeatmapPayload,
    MetricsRecord,
    PnLDistribution,
    TraderAnalysis,
)

MAX_CUMULATIVE_POINTS = 1200
PNL_HISTOGRAM_BINS = 60

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

FALLBACK_FEATURE_COLUMNS = [
    "quantity",
    "side_encoded",
    "profit_loss_actual",
    "is_profit",
    "loss_amount",
    "price_range",
    "price_range_pct",
    "trade_value",
    "win_rate_20",
    "avg_qty_20",
    "qty_volatility_20",
    "qty_max_20",
    "avg_profit_20",
    "profit_std_20",
    "max_drawdown_20",
    "buy_ratio_20",
    "win_rate_50",
    "avg_qty_50",
    "qty_volatility_50",
    "qty_max_50",
    "avg_profit_50",
    "profit_std_50",
    "max_drawdown_50",
    "buy_ratio_50",
    "early_close_indicator",
    "qty_after_loss",
]


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
            pl.col("side").cast(pl.Utf8).str.to_uppercase().alias("side")
        )

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

        model_features = getattr(state.model, "feature_names", None)
        features = (
            [str(col) for col in model_features]
            if model_features
            else list(FALLBACK_FEATURE_COLUMNS)
        )

        missing_features = [feature for feature in features if feature not in work_df.columns]
        if missing_features:
            work_df = work_df.with_columns(
                [pl.lit(0.0).alias(feature) for feature in missing_features]
            )

        features_matrix = (
            work_df.select(features)
            .with_columns(pl.all().cast(pl.Float64, strict=False))
            .fill_null(0.0)
            .to_numpy()
        )
        features_matrix = np.nan_to_num(features_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        dmatrix = xgb.DMatrix(features_matrix, feature_names=features)
        probabilities = state.model.predict(dmatrix)
        if probabilities.ndim == 1:
            if probabilities.size == work_df.height and np.all(np.equal(probabilities % 1, 0)):
                class_predictions = probabilities.astype(int)
                probability_matrix = np.zeros((work_df.height, len(state.TRADER_TYPES)))
                valid = (class_predictions >= 0) & (class_predictions < len(state.TRADER_TYPES))
                probability_matrix[np.arange(work_df.height)[valid], class_predictions[valid]] = 1.0
                probabilities = probability_matrix
            else:
                probabilities = probabilities.reshape(-1, 1)

        if probabilities.shape[1] < len(state.TRADER_TYPES):
            padded = np.zeros((probabilities.shape[0], len(state.TRADER_TYPES)))
            padded[:, : probabilities.shape[1]] = probabilities
            probabilities = padded
        elif probabilities.shape[1] > len(state.TRADER_TYPES):
            probabilities = probabilities[:, : len(state.TRADER_TYPES)]

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


def _evenly_sample(values: list[float], max_points: int) -> list[float]:
    if len(values) <= max_points:
        return values
    step = (len(values) - 1) / (max_points - 1)
    return [float(values[round(i * step)]) for i in range(max_points)]


def build_frontend_payload(df: pl.DataFrame) -> FrontendPayload:
    if df.height == 0:
        return FrontendPayload(
            cumulative_pnl=[],
            hourly_activity=[0] * 24,
            win_count=0,
            loss_count=0,
            average_win=0.0,
            average_loss=0.0,
            trades_per_hour=0.0,
            max_hourly_trades=0,
            pnl_distribution=PnLDistribution(min=0.0, max=0.0, buckets=[0] * PNL_HISTOGRAM_BINS),
            heatmap=HeatmapPayload(
                one_hour=HeatmapModeData(cols=24, sums=[0.0] * (7 * 24), counts=[0] * (7 * 24)),
                two_hour=HeatmapModeData(cols=12, sums=[0.0] * (7 * 12), counts=[0] * (7 * 12)),
                four_hour=HeatmapModeData(cols=6, sums=[0.0] * (7 * 6), counts=[0] * (7 * 6)),
                session=HeatmapModeData(cols=4, sums=[0.0] * (7 * 4), counts=[0] * (7 * 4)),
            ),
        )

    work_df = df.with_columns(
        [
            pl.col("profit_loss").cast(pl.Float64, strict=False).fill_null(0.0).alias("_pnl"),
            pl.col("timestamp")
            .cast(pl.Utf8)
            .str.strptime(pl.Datetime, strict=False)
            .alias("_timestamp"),
        ]
    )

    pnl_values = work_df.get_column("_pnl").to_numpy()
    cumulative = np.cumsum(pnl_values).tolist()
    cumulative_sampled = _evenly_sample([float(v) for v in cumulative], MAX_CUMULATIVE_POINTS)

    win_mask = pnl_values > 0
    loss_mask = pnl_values < 0
    win_count = int(np.count_nonzero(win_mask))
    loss_count = int(np.count_nonzero(loss_mask))
    average_win = float(np.mean(pnl_values[win_mask])) if win_count > 0 else 0.0
    average_loss = float(abs(np.mean(pnl_values[loss_mask]))) if loss_count > 0 else 0.0

    hourly_counts = [0] * 24
    hourly_df = (
        work_df.filter(pl.col("_timestamp").is_not_null())
        .with_columns(pl.col("_timestamp").dt.hour().alias("_hour"))
        .group_by("_hour")
        .len()
    )
    for row in hourly_df.iter_rows(named=True):
        hour = int(row["_hour"])
        count = int(row["len"])
        if 0 <= hour < 24:
            hourly_counts[hour] = count

    max_hourly_trades = max(hourly_counts) if hourly_counts else 0

    valid_ts_df = work_df.filter(pl.col("_timestamp").is_not_null())
    if valid_ts_df.height > 0:
        first_ts = valid_ts_df.select(pl.col("_timestamp").min()).item()
        last_ts = valid_ts_df.select(pl.col("_timestamp").max()).item()
        span_seconds = max((last_ts - first_ts).total_seconds(), 3600.0)
        trades_per_hour = float(df.height / (span_seconds / 3600.0))
    else:
        trades_per_hour = float(df.height)

    non_zero_pnl = pnl_values[pnl_values != 0]
    if non_zero_pnl.size > 0:
        hist, bin_edges = np.histogram(non_zero_pnl, bins=PNL_HISTOGRAM_BINS)
        pnl_min = float(bin_edges[0])
        pnl_max = float(bin_edges[-1])
        pnl_buckets = [int(v) for v in hist.tolist()]
    else:
        pnl_min = 0.0
        pnl_max = 0.0
        pnl_buckets = [0] * PNL_HISTOGRAM_BINS

    ts_non_null = valid_ts_df.with_columns(
        [
            pl.col("_timestamp").dt.strftime("%w").cast(pl.Int32).alias("_day"),
            pl.col("_timestamp").dt.hour().alias("_hour"),
        ]
    ).select(["_day", "_hour", "_pnl"])

    day_values = ts_non_null.get_column("_day").to_numpy()
    hour_values = ts_non_null.get_column("_hour").to_numpy()
    pnl_non_null = ts_non_null.get_column("_pnl").to_numpy()

    one_hour_sums = np.zeros(7 * 24, dtype=np.float64)
    one_hour_counts = np.zeros(7 * 24, dtype=np.int32)
    two_hour_sums = np.zeros(7 * 12, dtype=np.float64)
    two_hour_counts = np.zeros(7 * 12, dtype=np.int32)
    four_hour_sums = np.zeros(7 * 6, dtype=np.float64)
    four_hour_counts = np.zeros(7 * 6, dtype=np.int32)
    session_sums = np.zeros(7 * 4, dtype=np.float64)
    session_counts = np.zeros(7 * 4, dtype=np.int32)

    for idx in range(len(day_values)):
        day = int(day_values[idx])
        hour = int(hour_values[idx])
        pnl = float(pnl_non_null[idx])

        if day < 0 or day > 6 or hour < 0 or hour > 23:
            continue

        one_idx = day * 24 + hour
        one_hour_sums[one_idx] += pnl
        one_hour_counts[one_idx] += 1

        two_col = hour // 2
        two_idx = day * 12 + two_col
        two_hour_sums[two_idx] += pnl
        two_hour_counts[two_idx] += 1

        four_col = hour // 4
        four_idx = day * 6 + four_col
        four_hour_sums[four_idx] += pnl
        four_hour_counts[four_idx] += 1

        if hour < 6:
            session_col = 0
        elif hour < 12:
            session_col = 1
        elif hour < 18:
            session_col = 2
        else:
            session_col = 3
        session_idx = day * 4 + session_col
        session_sums[session_idx] += pnl
        session_counts[session_idx] += 1

    return FrontendPayload(
        cumulative_pnl=[float(v) for v in cumulative_sampled],
        hourly_activity=[int(v) for v in hourly_counts],
        win_count=win_count,
        loss_count=loss_count,
        average_win=round(average_win, 2),
        average_loss=round(average_loss, 2),
        trades_per_hour=round(trades_per_hour, 4),
        max_hourly_trades=int(max_hourly_trades),
        pnl_distribution=PnLDistribution(
            min=pnl_min,
            max=pnl_max,
            buckets=pnl_buckets,
        ),
        heatmap=HeatmapPayload(
            one_hour=HeatmapModeData(
                cols=24,
                sums=[float(v) for v in one_hour_sums.tolist()],
                counts=[int(v) for v in one_hour_counts.tolist()],
            ),
            two_hour=HeatmapModeData(
                cols=12,
                sums=[float(v) for v in two_hour_sums.tolist()],
                counts=[int(v) for v in two_hour_counts.tolist()],
            ),
            four_hour=HeatmapModeData(
                cols=6,
                sums=[float(v) for v in four_hour_sums.tolist()],
                counts=[int(v) for v in four_hour_counts.tolist()],
            ),
            session=HeatmapModeData(
                cols=4,
                sums=[float(v) for v in session_sums.tolist()],
                counts=[int(v) for v in session_counts.tolist()],
            ),
        ),
    )


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
