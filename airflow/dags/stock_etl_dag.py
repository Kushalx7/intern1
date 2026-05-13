"""
stock_etl_dag.py  –  Daily ETL + ML retraining pipeline
Tasks: ingest check → spark batch → ML retrain → cleanup
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import subprocess, logging

log = logging.getLogger(__name__)

default_args = {
    "owner":            "stock_platform",
    "depends_on_past":  False,
    "start_date":       datetime(2024, 1, 1),
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

dag = DAG(
    "stock_market_pipeline",
    default_args=default_args,
    description="Daily batch analysis + ML retraining",
    schedule_interval="0 2 * * *",   # 2 AM daily
    catchup=False,
    tags=["stock", "etl", "ml"],
)

def run_spark_batch(**ctx):
    result = subprocess.run(
        ["spark-submit", "--master", "local[*]",
         "/opt/airflow/app/batch/historical_analysis.py"],
        capture_output=True, text=True, timeout=600,
    )
    log.info(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Spark batch failed:\n{result.stderr}")

def generate_training_data(**ctx):
    result = subprocess.run(
        ["python", "/opt/airflow/scripts/generate_historical_data.py"],
        capture_output=True, text=True, timeout=60,
    )
    log.info(result.stdout)

def retrain_ml_model(**ctx):
    import sys
    sys.path.insert(0, "/opt/airflow")
    result = subprocess.run(
        ["python", "/opt/airflow/app/ml/train_model.py"],
        capture_output=True, text=True, timeout=300,
    )
    log.info(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"ML training failed:\n{result.stderr}")

def check_data_freshness(**ctx):
    """Verify Kafka consumer is writing data to MongoDB."""
    import os
    from pymongo import MongoClient
    from datetime import timezone
    client = MongoClient(os.getenv("MONGO_URI", "mongodb://mongodb:27017"))
    col    = client["stocks_db"]["live_prices"]
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=10)
    count  = col.count_documents({"timestamp": {"$gt": cutoff.isoformat()}})
    log.info(f"Records in last 10 min: {count}")
    if count == 0:
        log.warning("No recent data — producer may be down.")

t0 = PythonOperator(task_id="check_data_freshness",  python_callable=check_data_freshness, dag=dag)
t1 = PythonOperator(task_id="generate_training_data", python_callable=generate_training_data, dag=dag)
t2 = PythonOperator(task_id="run_spark_batch",        python_callable=run_spark_batch,       dag=dag)
t3 = PythonOperator(task_id="retrain_ml_model",       python_callable=retrain_ml_model,      dag=dag)

t0 >> t1 >> t2 >> t3
