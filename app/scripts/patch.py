from __future__ import annotations

from pathlib import Path

try:
    from scripts.check import load_csv
except ModuleNotFoundError:
    from check import load_csv


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = [
    REPO_ROOT / "datasets" / "calm_trader.csv",
    REPO_ROOT / "datasets" / "loss_averse_trader.csv",
    REPO_ROOT / "datasets" / "overtrader.csv",
    REPO_ROOT / "datasets" / "revenge_trader.csv",
]
PATCHED_DIR = REPO_ROOT / "datasets" / "patched"


def main() -> int:
    PATCHED_DIR.mkdir(parents=True, exist_ok=True)

    for source_path in SOURCE_FILES:
        raw_csv = source_path.read_text(encoding="utf-8")
        corrected_df, summary = load_csv(raw_csv, source_name=source_path.name)
        output_path = PATCHED_DIR / source_path.name
        corrected_df.write_csv(output_path)

        print(
            f"{summary.source_name}: "
            f"status={summary.status}, "
            f"empty_cells={summary.empty_cells}, "
            f"quantity_fills={summary.quantity_fills}, "
            f"entry_fills={summary.entry_fills}, "
            f"exit_fills={summary.exit_fills}, "
            f"profit_fixes={summary.profit_fixes}, "
            f"balance_fixes={summary.balance_fixes}, "
            f"warnings={len(summary.warnings)}, "
            f"output={output_path}"
        )
        if summary.error_message is not None:
            print(f"  {summary.error_message}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
