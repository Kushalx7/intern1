"""
predict.py
Lazy-loading per-symbol prediction. Safe to import even before training.
"""
import os, joblib
import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
_cache = {}

def _load(symbol: str):
    if symbol not in _cache:
        rf_path  = f"{MODELS_DIR}/{symbol}_rf.pkl"
        lr_path  = f"{MODELS_DIR}/{symbol}_lr.pkl"
        sc_path  = f"{MODELS_DIR}/{symbol}_scaler.pkl"
        if not os.path.exists(rf_path):
            raise FileNotFoundError(
                f"No trained model for {symbol}. Run app/ml/train_model.py first."
            )
        _cache[symbol] = {
            "rf":     joblib.load(rf_path),
            "lr":     joblib.load(lr_path),
            "scaler": joblib.load(sc_path),
        }
    return _cache[symbol]

def predict_price(symbol: str, open_price: float, high: float,
                  low: float, volume: int, model: str = "random_forest") -> float:
    models = _load(symbol)
    input_df = pd.DataFrame([{"open": open_price, "high": high,
                               "low": low, "volume": volume}])
    if model == "linear_regression":
        X_s = models["scaler"].transform(input_df)
        return round(float(models["lr"].predict(X_s)[0]), 2)
    return round(float(models["rf"].predict(input_df)[0]), 2)

def predict_all(open_price: float, high: float, low: float, volume: int) -> dict:
    results = {}
    for symbol in ["AAPL", "TSLA", "MSFT", "NVDA", "META", "AMZN"]:
        try:
            results[symbol] = {
                "rf": predict_price(symbol, open_price, high, low, volume, "random_forest"),
                "lr": predict_price(symbol, open_price, high, low, volume, "linear_regression"),
            }
        except Exception as e:
            results[symbol] = {"error": str(e)}
    return results

if __name__ == "__main__":
    p = predict_price("AAPL", 175, 178, 173, 2_000_000)
    print(f"AAPL predicted close: ${p}")
