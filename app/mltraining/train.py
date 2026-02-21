import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
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

# ADD TRADER_ID FIRST (ANTI-LEAK)
print("Generating realistic trader_ids...")
trader_groups = []
for trader_type, group in df.groupby('trader_type'):
    trades_per_trader = 50  # Realistic chunk size
    num_traders = len(group) // trades_per_trader + 1
    for i in range(num_traders):
        trader_id = f"t{trader_type}_{i}"
        chunk_size = min(trades_per_trader, len(group) - i*trades_per_trader)
        trader_groups.extend([trader_id] * chunk_size)

df['trader_id'] = trader_groups[:len(df)]
print(f"✓ Generated {len(set(df['trader_id']))} traders (~50 trades each)")

# Engineer features (AFTER trader_id)
print("\nEngineering features...")
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df['side_encoded'] = (df['side'] == 'BUY').astype(int)
df['is_profit'] = (df['profit_loss'] > 0).astype(int)
df['price_range'] = df['exit_price'] - df['entry_price']
df['price_range_pct'] = (df['price_range'] / df['entry_price'].abs()) * 100
df['profit_loss_pct'] = (df['profit_loss'] / (df['quantity'] * df['entry_price']).abs()) * 100
df['trade_cost'] = df['quantity'] * df['entry_price'].abs()

df['balance_change'] = df['balance'].diff().fillna(0)
df['profit_volatility'] = df['profit_loss'].rolling(10, min_periods=1).std().fillna(0)
df['buy_ratio'] = df['side_encoded'].rolling(20, min_periods=1).mean().fillna(0.5)

print("✓ Features engineered")

# Features
features = ['quantity', 'entry_price', 'exit_price', 'profit_loss', 
            'balance', 'hour', 'side_encoded', 'is_profit', 'balance_change',
            'price_range', 'price_range_pct', 'profit_loss_pct', 'trade_cost',
            'profit_volatility', 'buy_ratio']
X = df[features].fillna(0)
y = df['trader_type']

# GROUP CV FIRST (REAL VALIDATION)
print("\nGroup CV...")
model_cv = xgb.XGBClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42
)

gkf = GroupKFold(n_splits=5)
group_cv_scores = cross_val_score(model_cv, X, y, groups=df['trader_id'], cv=gkf, scoring='accuracy')
print(f"✓ Group CV Accuracy: {group_cv_scores.mean():.4f} ± {group_cv_scores.std():.4f}")

# GROUP SPLIT FOR FINAL MODEL
print("\nGroup train-test split...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=df['trader_id']))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
print(f"✓ {len(set(df['trader_id'].iloc[train_idx]))} train traders, {len(set(df['trader_id'].iloc[test_idx]))} test traders")
print(f"✓ Training: {len(X_train)}, Test: {len(X_test)}")

# FINAL MODEL
print("\nTraining final model...")
model = xgb.XGBClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42,
    verbosity=1, eval_metric='mlogloss'
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=True
)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✓ Final Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['calm_trader', 'loss_averse_trader', 'overtrader', 'revenge_trader']))

# Features
importance_df = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop Features:")
print(importance_df.head(10).to_string(index=False))

# Save
model.save_model('trader_classifier.json')
print("\n✓ Production model saved!")
