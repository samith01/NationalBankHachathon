from datetime import date, datetime, time

import numpy as np
import polars as pl
from fastapi import HTTPException

from .csv_sanitizer import CsvAnalysisSummary, load_csv
from .schemas import ScalarValue, TradeEntry, TradeRecord
from .state import uploaded_files


def parse_csv_file(file_content: bytes) -> pl.DataFrame:
    df, summary = parse_csv_file_with_summary(file_content)
    if summary.error_message:
        raise HTTPException(status_code=400, detail=summary.error_message)
    return df


def parse_csv_file_with_summary(
    file_content: bytes, *, source_name: str = "uploaded.csv"
) -> tuple[pl.DataFrame, CsvAnalysisSummary]:
    try:
        return load_csv(file_content.decode("utf-8"), source_name=source_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(exc)}")


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


def get_all_trades(session_id: str) -> list[TradeRecord]:
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
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve data: {str(exc)}")


def get_trades_in_range(session_id: str, start_date: date, end_date: date) -> list[TradeRecord]:
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
