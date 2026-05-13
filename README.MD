# Stock Market Big Data Platform

A complete real-time stock data pipeline: Kafka → Spark → MongoDB/PostgreSQL/Elasticsearch → Streamlit dashboard with ML predictions.

## Quick Start

### 1. Setup environment
```bash
cp .env.example .env
# Edit .env and add your Alpha Vantage API key (free at alphavantage.co)
# Or leave it as-is — the producer has a mock data fallback
```

### 2. Pre-train ML models (runs locally, not in Docker)
```bash
pip install -r requirements.txt
python scripts/generate_historical_data.py   # creates data/historical_stock_data.csv
python app/ml/train_model.py                  # trains models → app/ml/models/
```

### 3. Start all services
```bash
docker-compose up --build
```

### 4. Access the platform
| Service    | URL                        |
|------------|----------------------------|
| Dashboard  | http://localhost:8501       |
| API docs   | http://localhost:8000/docs  |
| Airflow    | http://localhost:8080 (admin/admin) |
| Kafka      | localhost:9092             |

## Architecture
```
Stock API / Mock
      │
 Kafka Producer  ──►  Kafka (topic: stock-prices)
                              │
              ┌───────────────┼───────────────┐
         Spark Streaming   Kafka Consumer       
              │            │       │       │   
         console       MongoDB  PostgreSQL  ES  
                            │       │       │   
                        Dashboard  API   Search 
```

## Services
- **producer** – fetches live stock prices, sends to Kafka every 5s
- **consumer** – reads Kafka, writes to MongoDB + PostgreSQL + Elasticsearch
- **spark**    – streaming job: moving averages + spike detection → PostgreSQL
- **fastapi**  – REST API (port 8000)
- **dashboard** – Streamlit UI with candlesticks, ML, alerts (port 8501)
- **airflow**  – daily batch analytics + ML retraining (port 8080)

## Dashboard tabs
1. **Live Prices** – real-time line/area charts + moving averages
2. **Candlestick** – OHLCV candlestick + MA10/MA20 overlay
3. **ML Predictions** – RandomForest & LinearRegression, MAE/R² metrics, live predictor
4. **Analytics** – trend, avg gain, volatility, correlation heatmap
5. **Spike Alerts** – events where price moved >5% vs open

## API Endpoints
| Method | Endpoint                  | Description                     |
|--------|---------------------------|---------------------------------|
| GET    | /stocks/live              | Latest prices (MongoDB)         |
| GET    | /stocks/{sym}/live        | Per-symbol live data            |
| GET    | /stocks/{sym}/history     | Historical from PostgreSQL      |
| GET    | /analytics/daily          | Daily averages                  |
| GET    | /analytics/gainers        | Top gainers                     |
| GET    | /alerts/spikes            | Spike events                    |
| POST   | /ml/predict               | Run ML prediction               |
| GET    | /ml/metrics               | Model MAE / R² scores           |
| GET    | /search?q=AAPL            | Elasticsearch full-text search  |

## Kubernetes
```bash
kubectl apply -f k8s/
```

## Technologies
Python · Kafka · Spark · Hadoop/HDFS · MongoDB · PostgreSQL · Elasticsearch  
FastAPI · Streamlit · Airflow · Docker · Kubernetes · scikit-learn
