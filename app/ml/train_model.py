"""
train_model.py
Trains per-symbol RandomForest and LinearRegression models.
Saves models to app/ml/models/ and metrics to data/model_metrics.json.
"""
import os, json, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH   = "data/historical_stock_data.csv"
MODELS_DIR  = "app/ml/models"
METRICS_OUT = "data/model_metrics.json"

os.makedirs(MODELS_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"{DATA_PATH} not found. Run: python scripts/generate_historical_data.py"
    )

df = pd.read_csv(DATA_PATH)
FEATURES = ["open", "high", "low", "volume"]
TARGET   = "close"

all_metrics = {}

for symbol in df["symbol"].unique():
    sdf = df[df["symbol"] == symbol].copy()
    X, y = sdf[FEATURES], sdf[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_mae   = mean_absolute_error(y_test, rf_preds)
    rf_r2    = r2_score(y_test, rf_preds)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    lr_preds = lr.predict(X_test_s)
    lr_mae   = mean_absolute_error(y_test, lr_preds)
    lr_r2    = r2_score(y_test, lr_preds)

    joblib.dump(rf,     f"{MODELS_DIR}/{symbol}_rf.pkl")
    joblib.dump(lr,     f"{MODELS_DIR}/{symbol}_lr.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/{symbol}_scaler.pkl")

    all_metrics[symbol] = {
        "random_forest":     {"mae": round(rf_mae, 4), "r2": round(rf_r2, 4)},
        "linear_regression": {"mae": round(lr_mae, 4), "r2": round(lr_r2, 4)},
        "best_model":        "random_forest" if rf_mae < lr_mae else "linear_regression",
        "train_samples": len(X_train),
        "test_samples":  len(X_test),
    }
    print(f"{symbol}  RF MAE={rf_mae:.2f} R2={rf_r2:.4f} | LR MAE={lr_mae:.2f} R2={lr_r2:.4f}")

with open(METRICS_OUT, "w") as f:
    json.dump(all_metrics, f, indent=2)

print(f"\nAll models saved to {MODELS_DIR}/")
print(f"Metrics saved to {METRICS_OUT}")
