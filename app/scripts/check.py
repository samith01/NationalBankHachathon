from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import Callable, cast

import polars as pl

NUMERIC_COLUMNS = ("quantity", "entry_price", "exit_price", "profit_loss", "balance")

# Script-level settings (edit these directly)
ABS_TOLERANCE = 1e-2
ZERO_DIV_EPSILON = 1e-12
FALLBACK_QUANTITY = 1.0
MAX_RECOVERY_PASSES = len(NUMERIC_COLUMNS)


@dataclass(slots=True)
class CsvAnalysisSummary:
    source_name: str
    empty_cells: int
    quantity_fills: int
    entry_fills: int
    exit_fills: int
    profit_fixes: int
    balance_fixes: int
    warnings: list[str]
    error_message: str | None = None

    @property
    def status(self) -> str:
        return "error" if self.error_message else "ok"


def load_raw_rows_from_text(raw_csv: str) -> list[dict[str, str]]:
    raw_rows: list[dict[str, str]] = []
    reader = csv.DictReader(io.StringIO(raw_csv))
    for row in reader:
        normalized_row: dict[str, str] = {}
        for key, value in row.items():
            if key is None:
                continue
            normalized_row[key] = value or ""
        raw_rows.append(normalized_row)
    return raw_rows


def source_label(raw_value: str) -> str:
    text = raw_value.strip()
    if not text:
        return "missing"
    return f"invalid '{text}'"


def parse_float(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def infer_trade_single_missing(
    quantity: float | None,
    entry_price: float | None,
    exit_price: float | None,
    profit_loss: float | None,
) -> tuple[str, float] | None:
    if (
        profit_loss is None
        and quantity is not None
        and entry_price is not None
        and exit_price is not None
    ):
        return ("profit_loss", quantity * (exit_price - entry_price))

    if (
        quantity is None
        and profit_loss is not None
        and entry_price is not None
        and exit_price is not None
    ):
        price_delta = exit_price - entry_price
        if abs(price_delta) > ZERO_DIV_EPSILON:
            return ("quantity", profit_loss / price_delta)
        return None

    if (
        entry_price is None
        and profit_loss is not None
        and quantity is not None
        and exit_price is not None
        and abs(quantity) > ZERO_DIV_EPSILON
    ):
        return ("entry_price", exit_price - (profit_loss / quantity))

    if (
        exit_price is None
        and profit_loss is not None
        and quantity is not None
        and entry_price is not None
        and abs(quantity) > ZERO_DIV_EPSILON
    ):
        return ("exit_price", entry_price + (profit_loss / quantity))

    return None


def run_recovery_pass(
    values: dict[str, float | None],
    previous_balance: float | None,
    set_recovered: Callable[[str, float, str], None],
) -> bool:
    changed = False
    if (
        values["profit_loss"] is None
        and previous_balance is not None
        and values["balance"] is not None
    ):
        set_recovered("profit_loss", values["balance"] - previous_balance, "(from balance)")
        changed = True

    inferred = infer_trade_single_missing(
        values["quantity"],
        values["entry_price"],
        values["exit_price"],
        values["profit_loss"],
    )
    if inferred is not None:
        inferred_column, inferred_value = inferred
        if values[inferred_column] is None:
            set_recovered(inferred_column, inferred_value, "")
            changed = True

    if (
        values["balance"] is None
        and previous_balance is not None
        and values["profit_loss"] is not None
    ):
        set_recovered("balance", previous_balance + values["profit_loss"], "")
        changed = True

    return changed


def _build_corrected_df(
    df: pl.DataFrame,
    corrected_quantity: list[float | None],
    corrected_entry_price: list[float | None],
    corrected_exit_price: list[float | None],
    corrected_profit_loss: list[float | None],
    corrected_balance: list[float | None],
) -> pl.DataFrame:
    replacements: dict[str, list[float | None]] = {
        "quantity": corrected_quantity,
        "entry_price": corrected_entry_price,
        "exit_price": corrected_exit_price,
        "profit_loss": corrected_profit_loss,
        "balance": corrected_balance,
    }
    columns: list[pl.Series] = []
    for column_name in df.columns:
        if column_name in replacements:
            columns.append(
                pl.Series(column_name, replacements[column_name], dtype=pl.Float64)
            )
        else:
            columns.append(df.get_column(column_name).clone())
    return pl.DataFrame(columns)


def _analyze_dataframe(
    df: pl.DataFrame,
    raw_rows: list[dict[str, str]],
    *,
    source_name: str,
    abs_tolerance: float,
    fallback_quantity: float,
) -> tuple[pl.DataFrame, CsvAnalysisSummary]:
    missing = set(NUMERIC_COLUMNS).difference(df.columns)
    if missing:
        summary = CsvAnalysisSummary(
            source_name=source_name,
            empty_cells=0,
            quantity_fills=0,
            entry_fills=0,
            exit_fills=0,
            profit_fixes=0,
            balance_fixes=0,
            warnings=[],
            error_message=f"Missing required columns: {', '.join(sorted(missing))}",
        )
        return df, summary

    if len(raw_rows) != df.height:
        raise ValueError("Row count mismatch while loading raw CSV values")

    quantity = cast(
        list[float | None], df["quantity"].cast(pl.Float64, strict=False).to_list()
    )
    entry_price = cast(
        list[float | None], df["entry_price"].cast(pl.Float64, strict=False).to_list()
    )
    exit_price = cast(
        list[float | None], df["exit_price"].cast(pl.Float64, strict=False).to_list()
    )
    profit_loss = cast(
        list[float | None], df["profit_loss"].cast(pl.Float64, strict=False).to_list()
    )
    balance = cast(
        list[float | None], df["balance"].cast(pl.Float64, strict=False).to_list()
    )

    corrected_quantity = list(quantity)
    corrected_entry_price = list(entry_price)
    corrected_exit_price = list(exit_price)
    corrected_profit_loss = list(profit_loss)
    corrected_balance = list(balance)

    warnings: list[str] = []

    quantity_fills = 0
    entry_price_fills = 0
    exit_price_fills = 0
    profit_loss_fixes = 0
    balance_fixes = 0

    empty_cells = sum(not v.strip() for row in raw_rows for v in row.values())
    error_message: str | None = None

    for i in range(df.height):
        raw_row = raw_rows[i]

        raw_values = {
            "quantity": raw_row.get("quantity", ""),
            "entry_price": raw_row.get("entry_price", ""),
            "exit_price": raw_row.get("exit_price", ""),
            "profit_loss": raw_row.get("profit_loss", ""),
            "balance": raw_row.get("balance", ""),
        }
        values: dict[str, float | None] = {
            "quantity": corrected_quantity[i],
            "entry_price": corrected_entry_price[i],
            "exit_price": corrected_exit_price[i],
            "profit_loss": corrected_profit_loss[i],
            "balance": corrected_balance[i],
        }

        def set_recovered(column: str, inferred_value: float, reason: str = "") -> None:
            nonlocal quantity_fills, entry_price_fills, exit_price_fills
            nonlocal profit_loss_fixes, balance_fixes

            values[column] = float(inferred_value)
            if column == "quantity":
                quantity_fills += 1
            elif column == "entry_price":
                entry_price_fills += 1
            elif column == "exit_price":
                exit_price_fills += 1
            elif column == "profit_loss":
                profit_loss_fixes += 1
            elif column == "balance":
                balance_fixes += 1

            suffix = f" {reason}" if reason else ""
            warnings.append(
                f"[WARN] {source_name} row {i}, column {column}: "
                f"{source_label(raw_values[column])} -> {values[column]}{suffix}"
            )

        for column_name in NUMERIC_COLUMNS:
            raw_value = raw_values[column_name]
            parsed_value = values[column_name]
            if raw_value.strip() and parsed_value is None:
                warnings.append(
                    f"[WARN] {source_name} row {i}, column {column_name}: "
                    f"non-numeric value '{raw_value}'"
                )

        previous_balance = corrected_balance[i - 1] if i > 0 else None

        for _ in range(MAX_RECOVERY_PASSES):
            if not run_recovery_pass(values, previous_balance, set_recovered):
                break

        if values["quantity"] is None:
            if values["entry_price"] is not None and values["exit_price"] is not None:
                set_recovered("quantity", fallback_quantity, "(fallback assumption)")
            elif values["profit_loss"] is not None and values["entry_price"] is None and values["exit_price"] is not None:
                set_recovered("quantity", fallback_quantity, "(fallback assumption)")
                quantity_now = values["quantity"]
                if quantity_now is not None:
                    set_recovered(
                        "entry_price",
                        values["exit_price"] - (values["profit_loss"] / quantity_now),
                        f"(derived using fallback quantity={fallback_quantity})",
                    )
            elif values["profit_loss"] is not None and values["exit_price"] is None and values["entry_price"] is not None:
                set_recovered("quantity", fallback_quantity, "(fallback assumption)")
                quantity_now = values["quantity"]
                if quantity_now is not None:
                    set_recovered(
                        "exit_price",
                        values["entry_price"] + (values["profit_loss"] / quantity_now),
                        f"(derived using fallback quantity={fallback_quantity})",
                    )

        for _ in range(MAX_RECOVERY_PASSES):
            if not run_recovery_pass(values, previous_balance, set_recovered):
                break

        if (
            values["quantity"] is not None
            and values["entry_price"] is not None
            and values["exit_price"] is not None
        ):
            expected_profit_loss = values["quantity"] * (
                values["exit_price"] - values["entry_price"]
            )

            if values["profit_loss"] is None:
                set_recovered("profit_loss", expected_profit_loss)
            elif not (abs(values["profit_loss"] - expected_profit_loss) <= abs_tolerance):
                previous_profit = values["profit_loss"]
                values["profit_loss"] = expected_profit_loss
                profit_loss_fixes += 1
                warnings.append(
                    f"[WARN] {source_name} row {i}, column profit_loss: "
                    f"{previous_profit} -> {expected_profit_loss}"
                )

        if i > 0 and previous_balance is not None and values["profit_loss"] is not None:
            expected_balance = previous_balance + values["profit_loss"]
            if values["balance"] is None:
                set_recovered("balance", expected_balance)
            elif not (abs(values["balance"] - expected_balance) <= abs_tolerance):
                previous_balance_value = values["balance"]
                values["balance"] = expected_balance
                raw_prev_balance = parse_float(raw_rows[i - 1].get("balance", ""))
                raw_balance = parse_float(raw_values["balance"])
                raw_profit = parse_float(raw_values["profit_loss"])
                propagated_only = (
                    raw_prev_balance is not None
                    and raw_balance is not None
                    and raw_profit is not None
                    and (abs(raw_balance - (raw_prev_balance + raw_profit)) <= abs_tolerance)
                )
                if not propagated_only:
                    balance_fixes += 1
                    warnings.append(
                        f"[WARN] {source_name} row {i}, column balance: "
                        f"{previous_balance_value} -> {expected_balance}"
                    )

        unresolved_column = next((c for c in NUMERIC_COLUMNS if values[c] is None), None)
        if unresolved_column is not None:
            error_message = (
                f"[ERROR] {source_name} row {i}, column {unresolved_column}: "
                "could not recover value from available fields; halting this file"
            )
            break

        corrected_quantity[i] = values["quantity"]
        corrected_entry_price[i] = values["entry_price"]
        corrected_exit_price[i] = values["exit_price"]
        corrected_profit_loss[i] = values["profit_loss"]
        corrected_balance[i] = values["balance"]

    corrected_df = _build_corrected_df(
        df,
        corrected_quantity,
        corrected_entry_price,
        corrected_exit_price,
        corrected_profit_loss,
        corrected_balance,
    )
    summary = CsvAnalysisSummary(
        source_name=source_name,
        empty_cells=empty_cells,
        quantity_fills=quantity_fills,
        entry_fills=entry_price_fills,
        exit_fills=exit_price_fills,
        profit_fixes=profit_loss_fixes,
        balance_fixes=balance_fixes,
        warnings=warnings,
        error_message=error_message,
    )
    return corrected_df, summary


def load_csv(
    raw_csv: str,
    *,
    source_name: str = "input.csv",
    abs_tolerance: float = ABS_TOLERANCE,
    fallback_quantity: float = FALLBACK_QUANTITY,
) -> tuple[pl.DataFrame, CsvAnalysisSummary]:
    if abs(fallback_quantity) <= ZERO_DIV_EPSILON:
        raise ValueError("fallback_quantity must be non-zero")

    df = pl.read_csv(io.StringIO(raw_csv))
    raw_rows = load_raw_rows_from_text(raw_csv)
    return _analyze_dataframe(
        df,
        raw_rows,
        source_name=source_name,
        abs_tolerance=abs_tolerance,
        fallback_quantity=fallback_quantity,
    )
