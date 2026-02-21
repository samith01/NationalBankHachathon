import pandas as pd
import xgboost as xgb
import sys

# Trader type mapping
TRADER_TYPES = {
    0: 'calm_trader',
    1: 'loss_averse_trader',
    2: 'overtrader',
    3: 'revenge_trader'
}

def predict_trader_type(csv_file):
    """
    Load CSV file and predict trader type using the trained model.
    
    Args:
        csv_file: Path to the CSV file to predict
    """
    # Load the trained model
    model = xgb.XGBClassifier()
    model.load_model('trader_classifier.json')
    
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Apply the same feature engineering as in train.py
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['side_encoded'] = (df['side'] == 'BUY').astype(int)
    df['is_profit'] = (df['profit_loss'] > 0).astype(int)
    df['price_range'] = df['exit_price'] - df['entry_price']
    df['price_range_pct'] = (df['price_range'] / df['entry_price'].abs()) * 100
    df['profit_loss_pct'] = (df['profit_loss'] / (df['quantity'] * df['entry_price']).abs()) * 100
    df['trade_cost'] = df['quantity'] * df['entry_price'].abs()
    
    df['balance_change'] = df['balance'].diff().fillna(0)
    df['profit_volatility'] = df['profit_loss'].std()
    df['buy_ratio'] = df['side_encoded'].mean()
    
    # Select features (same as in train.py)
    features = ['quantity', 'entry_price', 'exit_price', 'profit_loss', 
                'balance', 'hour', 'side_encoded', 'is_profit', 'balance_change',
                'price_range', 'price_range_pct', 'profit_loss_pct', 'trade_cost',
                'profit_volatility', 'buy_ratio']
    X = df[features].fillna(0)
    
    # Make predictions
    predictions = model.predict(X)
    prediction_probs = model.predict_proba(X)
    
    # Show overall class distribution
    print(f"\n{'='*60}")
    print("Overall Predictions Distribution:")
    print(f"{'='*60}")
    unique, counts = pd.Series(predictions).value_counts().sort_index().index, pd.Series(predictions).value_counts().sort_index().values
    for idx, count in zip(unique, counts):
        print(f"{TRADER_TYPES[idx]}: {count} ({count/len(predictions)*100:.1f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test.py <csv_file>")
        print("Example: python test.py ../../datasets/calm_trader.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    predict_trader_type(csv_file)
