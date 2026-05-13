import json
from kafka import KafkaConsumer
from app.utils.config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC

def create_consumer():
    return KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="stock-consumer-group"
    )
