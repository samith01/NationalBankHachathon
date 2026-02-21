from datetime import datetime

import polars as pl

from .data_service import get_last_numeric_value
from .schemas import ExcludeCriteria


def identify_excluded_trades(df: pl.DataFrame, criteria: ExcludeCriteria) -> list[int]:
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


def calculate_simulated_balances(df: pl.DataFrame, exclude_indices: list[int]) -> pl.DataFrame:
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
    if criteria.assets:
        return f"Exclude {', '.join(criteria.assets)} trades"
    if criteria.min_loss_amount is not None:
        return f"Exclude losses below ${criteria.min_loss_amount}"
    if criteria.max_loss_amount is not None:
        return f"Exclude gains above ${criteria.max_loss_amount}"
    if criteria.date_range:
        return f"Exclude trades from {criteria.date_range['start']} to {criteria.date_range['end']}"
    if criteria.trade_ids:
        return f"Exclude {len(criteria.trade_ids)} specific trades"
    return "Custom what-if simulation"
