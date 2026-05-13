"""
main.py  –  FastAPI REST API for the Stock Market Platform
Exposes live prices, analytics, ML predictions, alerts, and ES search.
"""
import os, json
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Stock Market Platform API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Lazy DB helpers ───────────────────────────────────────────
def pg():
    from sqlalchemy import create_engine, text
    eng = create_engine(
        f"postgresql+psycopg2://"
        f"{os.getenv('POSTGRES_USER','postgres')}:{os.getenv('POSTGRES_PASSWORD','postgres')}"
        f"@{os.getenv('POSTGRES_HOST','postgres')}:{os.getenv('POSTGRES_PORT','5432')}"
        f"/{os.getenv('POSTGRES_DB','stocks')}",
        pool_pre_ping=True,
    )
    return eng, text

def mongo_col():
    from pymongo import MongoClient
    c = MongoClient(os.getenv("MONGO_URI", "mongodb://mongodb:27017"))
    return c[os.getenv("MONGO_DB","stocks_db")][os.getenv("MONGO_COLLECTION","live_prices")]

# ── Models ────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    symbol: str
    open_price: float
    high: float
    low: float
    volume: int
    model: str = "random_forest"

# ── Endpoints ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Stock Market Platform API v2", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

@app.get("/stocks/live")
def get_live_prices(symbols: Optional[str] = None, limit: int = 100):
    """Latest prices for all (or specified) stocks from MongoDB."""
    col = mongo_col()
    query = {}
    if symbols:
        query["symbol"] = {"$in": [s.strip().upper() for s in symbols.split(",")]}
    data = list(col.find(query, {"_id": 0}).sort("timestamp", -1).limit(limit))
    return {"count": len(data), "data": data}

@app.get("/stocks/{symbol}/live")
def get_symbol_live(symbol: str, limit: int = 50):
    col = mongo_col()
    data = list(col.find({"symbol": symbol.upper()}, {"_id": 0})
                .sort("timestamp", -1).limit(limit))
    if not data:
        raise HTTPException(404, f"No live data for {symbol}")
    return {"symbol": symbol.upper(), "count": len(data), "data": data}

@app.get("/stocks/{symbol}/history")
def get_symbol_history(symbol: str, days: int = 30):
    """Historical data from PostgreSQL."""
    eng, text = pg()
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()
    with eng.connect() as conn:
        rows = conn.execute(text(
            "SELECT symbol, timestamp, price, volume, open, high, low, close "
            "FROM live_stock_prices WHERE symbol=:s AND timestamp >= :t "
            "ORDER BY timestamp DESC LIMIT 1000"
        ), {"s": symbol.upper(), "t": since}).fetchall()
    return {"symbol": symbol.upper(), "days": days,
            "count": len(rows), "data": [dict(r._mapping) for r in rows]}

@app.get("/analytics/daily")
def get_daily_analytics():
    eng, text = pg()
    with eng.connect() as conn:
        rows = conn.execute(text(
            "SELECT symbol, date, daily_avg_close FROM analytics_daily_avg "
            "ORDER BY date DESC LIMIT 200"
        )).fetchall()
    return {"data": [dict(r._mapping) for r in rows]}

@app.get("/analytics/gainers")
def get_top_gainers():
    eng, text = pg()
    with eng.connect() as conn:
        rows = conn.execute(text(
            "SELECT symbol, AVG(close - open) as avg_gain "
            "FROM live_stock_prices GROUP BY symbol ORDER BY avg_gain DESC"
        )).fetchall()
    return {"data": [dict(r._mapping) for r in rows]}

@app.get("/alerts/spikes")
def get_spike_alerts(limit: int = 50):
    eng, text = pg()
    with eng.connect() as conn:
        rows = conn.execute(text(
            "SELECT * FROM spike_alerts ORDER BY alert_time DESC LIMIT :l"
        ), {"l": limit}).fetchall()
    return {"count": len(rows), "data": [dict(r._mapping) for r in rows]}

@app.post("/ml/predict")
def predict(req: PredictionRequest):
    """Run ML prediction for a given symbol."""
    try:
        from app.ml.predict import predict_price
        pred = predict_price(req.symbol.upper(), req.open_price,
                             req.high, req.low, req.volume, req.model)
        return {"symbol": req.symbol.upper(), "predicted_close": pred,
                "model": req.model, "inputs": req.dict()}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/ml/metrics")
def get_ml_metrics():
    """Return model evaluation metrics."""
    path = "data/model_metrics.json"
    if not os.path.exists(path):
        raise HTTPException(404, "Model metrics not found. Train models first.")
    with open(path) as f:
        return json.load(f)

@app.get("/search")
def search_stocks(q: str, index: str = "stock-prices"):
    """Full-text search via Elasticsearch."""
    try:
        from elasticsearch import Elasticsearch
        es = Elasticsearch(os.getenv("ELASTICSEARCH_HOST", "http://elasticsearch:9200"))
        res = es.search(index=index, body={
            "query": {"multi_match": {"query": q, "fields": ["symbol", "*"]}}
        })
        hits = [h["_source"] for h in res["hits"]["hits"]]
        return {"query": q, "count": len(hits), "results": hits}
    except Exception as e:
        raise HTTPException(500, f"Elasticsearch error: {e}")
