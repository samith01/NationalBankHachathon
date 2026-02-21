from __future__ import annotations

from pathlib import Path

try:
    from scripts.check import load_csv
except ModuleNotFoundError:
    from check import load_csv


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_FILES = [
    REPO_ROOT / "datasets" / "calm_trader.csv",
    REPO_ROOT / "datasets" / "loss_averse_trader.csv",
    REPO_ROOT / "datasets" / "overtrader.csv",
    REPO_ROOT / "datasets" / "revenge_trader.csv",
]


def main() -> int:
    for csv_path in DATASET_FILES:
        raw_csv = csv_path.read_text(encoding="utf-8")
        _, summary = load_csv(raw_csv, source_name=csv_path.name)
        print(
            f"{summary.source_name}: "
            f"status={summary.status}, "
            f"empty_cells={summary.empty_cells}, "
            f"quantity_fills={summary.quantity_fills}, "
            f"entry_fills={summary.entry_fills}, "
            f"exit_fills={summary.exit_fills}, "
            f"profit_fixes={summary.profit_fixes}, "
            f"balance_fixes={summary.balance_fixes}, "
            f"warnings={len(summary.warnings)}"
        )
        if summary.error_message is not None:
            print(f"  {summary.error_message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
