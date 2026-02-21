from __future__ import annotations

import argparse
import csv
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import cast

import polars as pl

REQUIRED_COLUMNS = {
    "quantity",
    "entry_price",
    "exit_price",
    "profit_loss",
    "balance",
}
NUMERIC_COLUMNS = ("quantity", "entry_price", "exit_price", "profit_loss", "balance")


def decimal_places(value: str) -> int:
    text = value.strip()
    if not text:
        return 0

    lower = text.lower()
    if "e" in lower:
        try:
            dec = Decimal(text)
        except InvalidOperation:
            return 0
        exponent = dec.as_tuple().exponent
        if isinstance(exponent, int):
            return max(-exponent, 0)
        return 0

    if "." not in text:
        return 0

    return len(text.split(".", 1)[1])


def comparison_precision(value: str) -> int:
    return max(decimal_places(value) - 1, 0)


def load_raw_rows(csv_path: Path) -> list[dict[str, str]]:
    raw_rows: list[dict[str, str]] = []

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized_row = {key: (value or "") for key, value in row.items()}
            raw_rows.append(normalized_row)

    return raw_rows


def round_by_precision(value: float, precision: int) -> float:
    return round(float(value), precision)


def infer_mode_precision(
    raw_values: list[str], *, reduce_by_one: bool = False, default: int = 0
) -> int:
    decimals = [decimal_places(value) for value in raw_values if value.strip()]
    if not decimals:
        return default

    mode_precision = Counter(decimals).most_common(1)[0][0]
    if reduce_by_one:
        return max(mode_precision - 1, 0)
    return mode_precision


def row_precision(raw_value: str, fallback_precision: int) -> int:
    if raw_value.strip():
        return comparison_precision(raw_value)
    return fallback_precision


def source_label(raw_value: str) -> str:
    text = raw_value.strip()
    if not text:
        return "missing"
    return f"invalid '{text}'"


def is_effectively_zero(value: float, epsilon: float = 1e-12) -> bool:
    return -epsilon < value < epsilon


def is_significant_discrepancy(
    actual: float,
    expected: float,
    *,
    precision: int,
    abs_tolerance: float,
) -> bool:
    actual_rounded = round_by_precision(actual, precision)
    expected_rounded = round_by_precision(expected, precision)
    if actual_rounded == expected_rounded:
        return False
    return abs(actual - expected) > abs_tolerance


def fix_file(
    csv_path: Path,
    *,
    balance_abs_tolerance: float,
    profit_abs_tolerance: float,
) -> tuple[int, int, int, int, int, list[str], str | None]:
    df = pl.read_csv(csv_path)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    raw_rows = load_raw_rows(csv_path)
    if len(raw_rows) != df.height:
        raise ValueError("Row count mismatch while loading raw CSV values")

    raw_quantity = [row.get("quantity", "") for row in raw_rows]
    raw_entry_price = [row.get("entry_price", "") for row in raw_rows]
    raw_exit_price = [row.get("exit_price", "") for row in raw_rows]
    raw_profit_loss = [row.get("profit_loss", "") for row in raw_rows]
    raw_balance = [row.get("balance", "") for row in raw_rows]

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

    corrected_quantity: list[float | None] = []
    corrected_entry_price: list[float | None] = []
    corrected_exit_price: list[float | None] = []
    corrected_profit_loss: list[float | None] = []
    corrected_balance: list[float | None] = []
    warnings: list[str] = []

    quantity_fills = 0
    entry_price_fills = 0
    exit_price_fills = 0
    profit_loss_fixes = 0
    balance_fixes = 0

    quantity_fill_precision = infer_mode_precision(raw_quantity, default=1)
    entry_fill_precision = infer_mode_precision(raw_entry_price, default=6)
    exit_fill_precision = infer_mode_precision(raw_exit_price, default=6)
    profit_fill_precision = infer_mode_precision(
        raw_profit_loss, reduce_by_one=True, default=6
    )
    balance_fill_precision = infer_mode_precision(
        raw_balance, reduce_by_one=True, default=6
    )

    for i in range(df.height):
        raw_row = raw_rows[i]
        for column_name, value in raw_row.items():
            if not value.strip():
                warnings.append(
                    f"[WARN] {csv_path.name} row {i}, column {column_name}: empty value"
                )

        entry_value = entry_price[i]
        exit_value = exit_price[i]
        quantity_value = quantity[i]
        profit_value = profit_loss[i]
        balance_value = balance[i]

        raw_quantity_value = raw_row.get("quantity", "")
        raw_entry_value = raw_row.get("entry_price", "")
        raw_exit_value = raw_row.get("exit_price", "")
        raw_profit_value = raw_row.get("profit_loss", "")
        raw_balance_value = raw_row.get("balance", "")

        numeric_columns = (
            ("quantity", raw_quantity_value, quantity_value),
            ("entry_price", raw_entry_value, entry_value),
            ("exit_price", raw_exit_value, exit_value),
            ("profit_loss", raw_profit_value, profit_value),
            ("balance", raw_balance_value, balance_value),
        )
        for column_name, raw_value, parsed_value in numeric_columns:
            if raw_value.strip() and parsed_value is None:
                warnings.append(
                    f"[WARN] {csv_path.name} row {i}, column {column_name}: "
                    f"non-numeric value '{raw_value}'"
                )

        previous_balance = corrected_balance[i - 1] if i > 0 else None

        if profit_value is None and previous_balance is not None and balance_value is not None:
            inferred_profit = round_by_precision(
                balance_value - previous_balance, profit_fill_precision
            )
            profit_value = inferred_profit
            profit_loss_fixes += 1
            warnings.append(
                f"[WARN] {csv_path.name} row {i}, column profit_loss: "
                f"{source_label(raw_profit_value)} -> {inferred_profit} (from balance)"
            )

        # Iteratively recover whichever numeric fields are solvable from available values.
        for _ in range(6):
            changed = False

            if (
                quantity_value is None
                and profit_value is not None
                and entry_value is not None
                and exit_value is not None
            ):
                price_delta = exit_value - entry_value
                if not is_effectively_zero(price_delta):
                    inferred_quantity = round_by_precision(
                        profit_value / price_delta, quantity_fill_precision
                    )
                    quantity_value = inferred_quantity
                    quantity_fills += 1
                    warnings.append(
                        f"[WARN] {csv_path.name} row {i}, column quantity: "
                        f"{source_label(raw_quantity_value)} -> {inferred_quantity}"
                    )
                    changed = True

            if (
                entry_value is None
                and profit_value is not None
                and quantity_value is not None
                and exit_value is not None
                and not is_effectively_zero(quantity_value)
            ):
                inferred_entry = round_by_precision(
                    exit_value - (profit_value / quantity_value), entry_fill_precision
                )
                entry_value = inferred_entry
                entry_price_fills += 1
                warnings.append(
                    f"[WARN] {csv_path.name} row {i}, column entry_price: "
                    f"{source_label(raw_entry_value)} -> {inferred_entry}"
                )
                changed = True

            if (
                exit_value is None
                and profit_value is not None
                and quantity_value is not None
                and entry_value is not None
                and not is_effectively_zero(quantity_value)
            ):
                inferred_exit = round_by_precision(
                    entry_value + (profit_value / quantity_value), exit_fill_precision
                )
                exit_value = inferred_exit
                exit_price_fills += 1
                warnings.append(
                    f"[WARN] {csv_path.name} row {i}, column exit_price: "
                    f"{source_label(raw_exit_value)} -> {inferred_exit}"
                )
                changed = True

            if (
                profit_value is None
                and quantity_value is not None
                and entry_value is not None
                and exit_value is not None
            ):
                inferred_profit = round_by_precision(
                    quantity_value * (exit_value - entry_value), profit_fill_precision
                )
                profit_value = inferred_profit
                profit_loss_fixes += 1
                warnings.append(
                    f"[WARN] {csv_path.name} row {i}, column profit_loss: "
                    f"{source_label(raw_profit_value)} -> {inferred_profit}"
                )
                changed = True

            if not changed:
                break

        if quantity_value is not None and entry_value is not None and exit_value is not None:
            expected_profit_loss = quantity_value * (exit_value - entry_value)
            profit_precision = row_precision(raw_profit_value, profit_fill_precision)
            expected_profit_loss_rounded = round_by_precision(
                expected_profit_loss, profit_precision
            )

            if profit_value is None:
                profit_value = expected_profit_loss_rounded
                profit_loss_fixes += 1
                warnings.append(
                    f"[WARN] {csv_path.name} row {i}, column profit_loss: "
                    f"{source_label(raw_profit_value)} -> {expected_profit_loss_rounded}"
                )
            else:
                if is_significant_discrepancy(
                    actual=profit_value,
                    expected=expected_profit_loss_rounded,
                    precision=profit_precision,
                    abs_tolerance=profit_abs_tolerance,
                ):
                    previous_profit = profit_value
                    profit_value = expected_profit_loss_rounded
                    profit_loss_fixes += 1
                    warnings.append(
                        f"[WARN] {csv_path.name} row {i}, column profit_loss: "
                        f"{previous_profit} -> {expected_profit_loss_rounded}"
                    )

        if i == 0:
            balance_precision = row_precision(raw_balance_value, balance_fill_precision)
            if balance_value is not None:
                balance_value = round_by_precision(balance_value, balance_precision)
        else:
            balance_precision = row_precision(raw_balance_value, balance_fill_precision)
            if previous_balance is not None and profit_value is not None:
                expected_balance = previous_balance + profit_value
                expected_balance_rounded = round_by_precision(
                    expected_balance, balance_precision
                )
                if balance_value is None:
                    balance_value = expected_balance_rounded
                    balance_fixes += 1
                    warnings.append(
                        f"[WARN] {csv_path.name} row {i}, column balance: "
                        f"{source_label(raw_balance_value)} -> {expected_balance_rounded}"
                    )
                else:
                    if is_significant_discrepancy(
                        actual=balance_value,
                        expected=expected_balance_rounded,
                        precision=balance_precision,
                        abs_tolerance=balance_abs_tolerance,
                    ):
                        previous_balance_value = balance_value
                        balance_value = expected_balance_rounded
                        balance_fixes += 1
                        warnings.append(
                            f"[WARN] {csv_path.name} row {i}, column balance: "
                            f"{previous_balance_value} -> {expected_balance_rounded}"
                        )

        unresolved_numeric = (
            ("quantity", raw_quantity_value, quantity_value),
            ("entry_price", raw_entry_value, entry_value),
            ("exit_price", raw_exit_value, exit_value),
            ("profit_loss", raw_profit_value, profit_value),
            ("balance", raw_balance_value, balance_value),
        )
        for column_name, _raw_value, parsed_value in unresolved_numeric:
            if parsed_value is None:
                error_message = (
                    f"[ERROR] {csv_path.name} row {i}, column {column_name}: "
                    "could not recover value from available fields; halting this file"
                )
                return (
                    quantity_fills,
                    entry_price_fills,
                    exit_price_fills,
                    profit_loss_fixes,
                    balance_fixes,
                    warnings,
                    error_message,
                )

        corrected_quantity.append(quantity_value)
        corrected_entry_price.append(entry_value)
        corrected_exit_price.append(exit_value)
        corrected_profit_loss.append(profit_value)
        corrected_balance.append(balance_value)

    if (
        quantity_fills > 0
        or entry_price_fills > 0
        or exit_price_fills > 0
        or profit_loss_fixes > 0
        or balance_fixes > 0
    ):
        updated_columns: dict[str, list[float | None] | pl.Series] = {}
        for column_name in df.columns:
            if column_name == "quantity":
                updated_columns[column_name] = corrected_quantity
            elif column_name == "entry_price":
                updated_columns[column_name] = corrected_entry_price
            elif column_name == "exit_price":
                updated_columns[column_name] = corrected_exit_price
            elif column_name == "profit_loss":
                updated_columns[column_name] = corrected_profit_loss
            elif column_name == "balance":
                updated_columns[column_name] = corrected_balance
            else:
                updated_columns[column_name] = df.get_column(column_name)
        pl.DataFrame(updated_columns).write_csv(csv_path)

    return (
        quantity_fills,
        entry_price_fills,
        exit_price_fills,
        profit_loss_fixes,
        balance_fixes,
        warnings,
        None,
    )


def resolve_dataset_dir(cli_value: str | None) -> Path:
    if cli_value:
        candidate = Path(cli_value).expanduser().resolve()
        if not candidate.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {candidate}")
        return candidate

    cwd_candidate = Path.cwd() / "datasets"
    if cwd_candidate.is_dir():
        return cwd_candidate

    repo_candidate = Path(__file__).resolve().parents[2] / "datasets"
    if repo_candidate.is_dir():
        return repo_candidate

    raise FileNotFoundError(
        "Could not find datasets directory. Pass it with --datasets-dir."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and fix profit_loss and balance columns in dataset CSVs."
    )
    parser.add_argument(
        "--datasets-dir",
        default=None,
        help="Path to the directory containing CSV files (default: ./datasets).",
    )
    parser.add_argument(
        "--balance-abs-epsilon",
        type=float,
        default=1e-2,
        help="Absolute tolerance for balance comparison before flagging a discrepancy.",
    )
    parser.add_argument(
        "--profit-abs-epsilon",
        type=float,
        default=1e-6,
        help="Absolute tolerance for profit_loss comparison before flagging a discrepancy.",
    )
    args = parser.parse_args()

    datasets_dir = resolve_dataset_dir(args.datasets_dir)
    csv_files = sorted(datasets_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {datasets_dir}")
        return 0

    total_entry_fills = 0
    total_exit_fills = 0
    total_profit_fixes = 0
    total_balance_fixes = 0
    total_quantity_fills = 0
    files_with_warnings = 0
    errored_files = 0

    for csv_path in csv_files:
        try:
            (
                quantity_fills,
                entry_fills,
                exit_fills,
                profit_fixes,
                balance_fixes,
                warnings,
                error_message,
            ) = fix_file(
                csv_path,
                balance_abs_tolerance=args.balance_abs_epsilon,
                profit_abs_tolerance=args.profit_abs_epsilon,
            )
        except Exception as exc:  # noqa: BLE001
            errored_files += 1
            print(f"[ERROR] {csv_path.name}: {exc}")
            continue

        if warnings:
            files_with_warnings += 1
            print(
                f"[WARN] {csv_path.name}: "
                f"filled {quantity_fills} quantity rows, "
                f"{entry_fills} entry_price rows, "
                f"{exit_fills} exit_price rows, "
                f"fixed {profit_fixes} profit_loss rows, "
                f"{balance_fixes} balance rows"
            )
            for warning in warnings:
                print(warning)
        else:
            print(f"[OK] {csv_path.name}: no discrepancies found")

        if error_message:
            errored_files += 1
            print(error_message)
            continue

        total_quantity_fills += quantity_fills
        total_entry_fills += entry_fills
        total_exit_fills += exit_fills
        total_profit_fixes += profit_fixes
        total_balance_fixes += balance_fixes

    print(
        f"\nSummary: files with warnings={files_with_warnings}, "
        f"quantity fills={total_quantity_fills}, "
        f"entry_price fills={total_entry_fills}, "
        f"exit_price fills={total_exit_fills}, "
        f"profit_loss fixes={total_profit_fixes}, "
        f"balance fixes={total_balance_fixes}, errors={errored_files}"
    )

    return 1 if errored_files else 0


if __name__ == "__main__":
    raise SystemExit(main())
