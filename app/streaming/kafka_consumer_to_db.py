"""
kafka_consumer_to_db.py
Fans each Kafka message out to MongoDB, PostgreSQL, Elasticsearch.
Connections are created lazily so a missing DB doesn't crash at startup.
"""
import json, time, logging
from app.utils.kafka_helpers import create_consumer
from app.utils.config import (
    MONGO_URI, MONGO_DB, MONGO_COLLECTION,
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD,
    ELASTICSEARCH_HOST,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Lazy DB clients ───────────────────────────────────────────
_mongo_col   = None
_pg_engine   = None
_es_client   = None

def get_mongo():
    global _mongo_col
    if _mongo_col is None:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        _mongo_col = client[MONGO_DB][MONGO_COLLECTION]
    return _mongo_col

def get_pg():
    global _pg_engine
    if _pg_engine is None:
        from sqlalchemy import create_engine
        _pg_engine = create_engine(
            f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
            f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
            pool_pre_ping=True,
        )
    return _pg_engine

def get_es():
    global _es_client
    if _es_client is None:
        from elasticsearch import Elasticsearch
        _es_client = Elasticsearch(
            ELASTICSEARCH_HOST,
            headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=8",
                     "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8"}
        )
    return _es_client

def store_postgres(record: dict):
    sql = """
        INSERT INTO live_stock_prices
            (symbol, timestamp, price, volume, open, high, low, close)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    with get_pg().raw_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, (
            record.get("symbol"), record.get("timestamp"), record.get("price"),
            record.get("volume"), record.get("open"), record.get("high"),
            record.get("low"),    record.get("close"),
        ))
        conn.commit()

def store_mongo(record: dict):
    doc = {k: v for k, v in record.items() if k != "_id"}
    get_mongo().insert_one(doc)

def store_elasticsearch(record: dict):
    get_es().index(index="stock-prices", document=record)

def run_consumer():
    consumer = create_consumer()
    log.info("Consumer started, waiting for messages…")
    for message in consumer:
        record = message.value
        errors = []
        try:
            store_mongo(record)
        except Exception as e:
            errors.append(f"Mongo: {e}")
        try:
            store_postgres(record)
        except Exception as e:
            errors.append(f"Postgres: {e}")
        try:
            store_elasticsearch(record)
        except Exception as e:
            errors.append(f"ES: {e}")
        if errors:
            log.warning("Storage errors for %s: %s", record.get("symbol"), errors)
        else:
            log.info("Stored %s @ $%s", record.get("symbol"), record.get("price"))

if __name__ == "__main__":
    run_consumer()
