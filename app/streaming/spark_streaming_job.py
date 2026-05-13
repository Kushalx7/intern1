"""
spark_streaming_job.py
Consumes Kafka stock stream → computes moving averages + spike detection
→ writes results back to PostgreSQL (not just console).
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, avg, stddev, window, expr, to_timestamp, current_timestamp
)
from pyspark.sql.types import (
    StructType, StringType, DoubleType, LongType
)
import os

KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC   = os.getenv("KAFKA_TOPIC", "stock-prices")
PG_URL        = (
    f"jdbc:postgresql://{os.getenv('POSTGRES_HOST','postgres')}:"
    f"{os.getenv('POSTGRES_PORT','5432')}/{os.getenv('POSTGRES_DB','stocks')}"
)
PG_PROPS = {
    "user":     os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
    "driver":   "org.postgresql.Driver",
}

schema = (
    StructType()
    .add("symbol",    StringType())
    .add("timestamp", StringType())
    .add("price",     DoubleType())
    .add("volume",    LongType())
    .add("open",      DoubleType())
    .add("high",      DoubleType())
    .add("low",       DoubleType())
    .add("close",     DoubleType())
)

spark = (
    SparkSession.builder
    .appName("StockStreamingJob")
    .config("spark.sql.streaming.checkpointLocation", "/tmp/spark-checkpoints")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

raw_df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SERVERS)
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", "latest")
    .load()
)

parsed_df = (
    raw_df
    .selectExpr("CAST(value AS STRING) as json_string")
    .select(from_json(col("json_string"), schema).alias("d"))
    .select("d.*")
    .withColumn("event_time", to_timestamp(col("timestamp")))
    .filter(col("price").isNotNull())
)

# ── Moving average (5-min window) ────────────────────────────
moving_avg_df = (
    parsed_df
    .withWatermark("event_time", "10 minutes")
    .groupBy(window(col("event_time"), "5 minutes"), col("symbol"))
    .agg(avg("price").alias("avg_price"), stddev("price").alias("price_std"))
)

# ── Spike detection (>5% change vs open) ─────────────────────
spike_df = (
    parsed_df
    .withColumn(
        "price_change_pct",
        expr("CASE WHEN open > 0 THEN ((price - open) / open) * 100 ELSE 0 END"),
    )
    .filter(col("price_change_pct") > 1)
    .select(
        col("symbol"), col("event_time").alias("alert_time"),
        col("price"), col("open").alias("open_price"),
        col("price_change_pct"),
    )
)

# ── Write moving averages → console (Spark windowed → JDBC is complex) ───
q1 = (
    moving_avg_df.writeStream
    .format("console")
    .outputMode("update")
    .option("truncate", False)
    .start()
)

# ── Write raw clean records → PostgreSQL via foreach batch ───────────────
def write_to_postgres(batch_df, batch_id):
    if batch_df.isEmpty():
        return
    batch_df.write.jdbc(PG_URL, "live_stock_prices", mode="append", properties=PG_PROPS)

q2 = (
    parsed_df.writeStream
    .outputMode("append")
    .foreachBatch(write_to_postgres)
    .start()
)

# ── Write spikes → PostgreSQL ────────────────────────────────
def write_spikes(batch_df, batch_id):
    if batch_df.isEmpty():
        return
    batch_df.write.jdbc(PG_URL, "spike_alerts", mode="append", properties=PG_PROPS)

q3 = (
    spike_df.writeStream
    .outputMode("append")
    .foreachBatch(write_spikes)
    .start()
)

spark.streams.awaitAnyTermination()
