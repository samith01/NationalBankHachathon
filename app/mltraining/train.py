import numpy as np
import polars as pl
import xgboost as xgb


# Load datasets
print("\n📊 Loading datasets...")
calm = pd.read_csv('../../datasets/calm_trader.csv')
calm['trader_type'] = 0

loss_averse = pd.read_csv('../../datasets/loss_averse_trader.csv')
loss_averse['trader_type'] = 1

overtrader = pd.read_csv('../../datasets/overtrader.csv')
overtrader['trader_type'] = 2

revenge = pd.read_csv('../../datasets/revenge_trader.csv')
revenge['trader_type'] = 3

# Combine
df = pd.concat([calm, loss_averse, overtrader, revenge], ignore_index=True)
print(f"✓ Combined dataset: {len(df)} total rows")

# Smart data cleaning
print("\nCleaning data...")
initial_rows = len(df)

# Handle missing profit_loss
if df['profit_loss'].isna().any():
    missing_pl = df['profit_loss'].isna().sum()
    print(f"  - Calculating {missing_pl} missing profit_loss values")
    df.loc[df['profit_loss'].isna(), 'profit_loss'] = (
        df.loc[df['profit_loss'].isna(), 'exit_price'] - 
        df.loc[df['profit_loss'].isna(), 'entry_price']
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


# ============================================================================
# OPTIMIZED MODEL
# ============================================================================
print("\n🚀 Training XGBoost model...")
model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.5,
    reg_lambda=2.0,
    min_child_weight=2,
    gamma=1.0,
    random_state=42,
    verbosity=0,
    eval_metric='mlogloss',
    scale_pos_weight=1
)

model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_test, y_test)],
    verbose=True
)

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

# Save model
model.save_model('trader_classifier2.json')
print("\n✓ Model saved as trader_classifier2.json")
print("=" * 80)
