#!/usr/bin/env python3
"""
generate_historical_data.py
Generates synthetic historical stock data for ML training.
Run once before training: python scripts/generate_historical_data.py
"""
import os, random, csv
from datetime import datetime, timedelta

STOCKS = ["AAPL", "TSLA", "MSFT", "NVDA", "META", "AMZN"]
BASE_PRICES = {"AAPL": 165.0, "TSLA": 230.0, "MSFT": 370.0,
               "NVDA": 460.0, "META": 300.0, "AMZN": 140.0}

os.makedirs("data", exist_ok=True)
rows = []
for symbol in STOCKS:
    price = BASE_PRICES[symbol]
    date = datetime(2023, 1, 1)
    for _ in range(500):          # ~500 trading days
        day_return = random.gauss(0.0003, 0.018)
        price = max(10.0, price * (1 + day_return))
        open_p  = round(price * random.uniform(0.99, 1.01), 2)
        high_p  = round(price * random.uniform(1.00, 1.03), 2)
        low_p   = round(price * random.uniform(0.97, 1.00), 2)
        close_p = round(price, 2)
        volume  = random.randint(500_000, 8_000_000)
        rows.append([symbol, date.strftime("%Y-%m-%d"),
                     open_p, high_p, low_p, close_p, volume])
        date += timedelta(days=1)
        while date.weekday() >= 5:   # skip weekends
            date += timedelta(days=1)

with open("data/historical_stock_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["symbol", "date", "open", "high", "low", "close", "volume"])
    writer.writerows(rows)

print(f"Generated {len(rows)} rows → data/historical_stock_data.csv")
