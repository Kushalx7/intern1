FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=120 -r requirements.txt
COPY . .
ENV PYTHONPATH=/app
CMD ["python", "app/ingestion/stock_producer.py"]
